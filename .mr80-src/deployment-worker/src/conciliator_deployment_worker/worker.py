"""Run with: python -m conciliator_deployment_worker --config worker.json.

Instance supervision uses a detached process group and a persistent journal.
It never overwrites the source repository, installs dependencies or kills a foreign port owner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
from pathlib import Path

import httpx

from conciliator_deployment_worker.redaction import safe_text

DEFAULT_REPOSITORY = "dp/aeolus"
REPOSITORY_URL = "https://code.byted.org/{repository}.git"


class DeploymentWorker:
    def __init__(self, config: dict, client: httpx.AsyncClient):
        self.config = {
            "display_name": config.get("worker_id", ""),
            "vmid": None,
            "extra_python_paths": [],
            **config,
        }
        if not isinstance(self.config["extra_python_paths"], list) or not all(
            isinstance(path, str) and path for path in self.config["extra_python_paths"]
        ):
            raise ValueError("extra_python_paths must be a list of nonempty paths")
        self.config["extra_python_paths"] = [
            os.path.abspath(Path(path).expanduser()) for path in self.config["extra_python_paths"]
        ]
        self.profiles = {config.get("repository", DEFAULT_REPOSITORY): self.config}
        self.client = client
        self.root = Path(config["root_path"]).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "journal.json"
        self.journal = json.loads(self.journal_path.read_text()) if self.journal_path.exists() else {}
        self.cancel_event = asyncio.Event()
        self.running_command: asyncio.subprocess.Process | None = None
        self.lifecycle_lock = asyncio.Lock()

    def save(self):
        temporary = self.journal_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.journal))
        temporary.chmod(0o600)
        temporary.replace(self.journal_path)

    @staticmethod
    def process_identity(pid):
        try:
            # PID plus start time guards against PID reuse; comm may contain spaces.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return None if fields[0] == "Z" else fields[19]
        except (OSError, IndexError):
            return None

    def service_alive(self):
        pid = self.journal.get("pid")
        return bool(pid and self.process_identity(pid) == self.journal.get("process_identity"))

    def port_available(self):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", self.config["service_port"]))
                return True
            except OSError:
                return False

    def readiness_owned(self):
        return True

    async def stop_service(self):
        if self.service_alive():
            pid = self.journal["pid"]
            os.killpg(pid, signal.SIGTERM)
            for _ in range(50):
                if not self.service_alive():
                    break
                await asyncio.sleep(0.1)
            if self.service_alive():
                os.killpg(pid, signal.SIGKILL)
                for _ in range(50):
                    if not self.service_alive():
                        break
                    await asyncio.sleep(0.1)
            if self.service_alive():
                raise RuntimeError("owned service did not stop")
        # The launcher may exit before its children finish closing the listening socket.
        for _ in range(50):
            if self.port_available():
                break
            await asyncio.sleep(0.1)
        if not self.port_available():
            raise RuntimeError("instance port remains occupied; refusing release")
        self.journal.pop("pid", None)
        self.journal.pop("process_identity", None)
        self.save()


    async def command(self, argv, cwd, logfile, deadline_seconds=600):
        if self.cancel_event.is_set():
            raise RuntimeError("deployment cancelled")
        with logfile.open("ab") as output:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
            self.running_command = process
            self.journal["command_pid"] = process.pid
            self.journal["command_identity"] = self.process_identity(process.pid)
            self.save()
            if self.cancel_event.is_set() and process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), deadline_seconds)
            except BaseException:
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
                raise
            finally:
                self.running_command = None
                self.journal.pop("command_pid", None)
                self.journal.pop("command_identity", None)
                self.save()
            if process.returncode:
                raise RuntimeError(f"deployment command failed with exit code {process.returncode}; inspect deploy.log")

    def prepare_start(self, checkout, profile=None):
        profile = self.config if profile is None else profile
        argv = profile.get("start_argv")
        if argv:
            return [str(item).replace("{port}", str(self.config["service_port"])) for item in argv]
        script = (checkout / "run/dev_run.sh").read_text()
        dangerous = "kill -9 $(lsof -i:$PORT0|awk 'NR>=2{print $2}')"
        if dangerous not in script:
            raise RuntimeError("unrecognized startup script; provide reviewed start_argv")
        # Preserve the project's environment/startup, but never execute its port-wide kill.
        script = script.replace(dangerous, ": # process ownership is managed by the deployment worker")
        python_path = profile.get("python_path")
        if not python_path:
            raise RuntimeError("python_path is required; dependency installation is on hold")
        # Keep virtualenv interpreter symlinks: resolving them would select the base environment.
        python_bin = Path(os.path.abspath(Path(python_path).expanduser()))
        if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
            raise RuntimeError("python_path must point to an existing executable")
        identity = profile.get("service_identity", "data.aeolus.api_v3")
        original = f"doas -p {identity} python "
        if original not in script:
            raise RuntimeError("startup script does not match configured service_identity")
        script = script.replace(original, f"exec doas -p {identity} {shlex.quote(str(python_bin))} ")
        target = checkout / "run/worker_dev_run.sh"
        target.write_text(script)
        return ["bash", str(target)]

    async def prepare_submodules(self, checkout, logfile, excluded=()):
        if not (checkout / ".gitmodules").is_file():
            return []
        try:
            declarations = await asyncio.to_thread(
                subprocess.check_output,
                ["git", "config", "--file", ".gitmodules", "--null", "--get-regexp", r"^submodule\..*\.path$"],
                cwd=checkout, text=True,
            )
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:  # No submodule paths declared.
                return []
            raise
        modules = [entry.split("\n", 1) for entry in declarations.split("\0") if entry]
        modules = [(key, path) for key, path in modules if path not in excluded]
        if not modules:
            return []
        # Override both path filters and per-module active=false without changing Git config.
        active = [arg for key, _ in modules for arg in ("-c", f"{key[:-5]}.active=true")]
        await self.git_network_command(
            [
                "git", *active, "submodule", "update", "--init", "--checkout", "--force", "--depth=1", "--",
                *[path for _, path in modules],
            ],
            checkout, logfile,
        )
        for _, path in modules:
            await self.prepare_submodules(checkout / path, logfile)
        return [path for _, path in modules]

    async def prepare_dependencies(self, checkout, profile, logfile):
        # An explicitly prepared SCM parser replaces the legacy Git submodule.
        excluded = (
            ("aeolus_expr_parse",)
            if "aeolus_expr_parse" in profile.get("dependency_paths", {}) else ()
        )
        if (checkout / ".gitmodules").is_file():
            paths = await self.prepare_submodules(checkout, logfile, excluded)
            # Check each tracked checkout directly, even when .gitmodules ignores dirty changes.
            if paths:
                await self.command(
                    [
                        "sh", "-c",
                        f"status=$(git submodule status --recursive -- {shlex.join(paths)}) && "
                        '! printf "%s\\n" "$status" | grep -q "^[+U-]" && '
                        "git submodule foreach --recursive "
                        "'git diff --quiet HEAD -- && test \"$(git rev-parse HEAD)\" = \"$sha1\"'",
                    ],
                    checkout, logfile,
                )
        # Runtime artifacts are provisioned locally, independently of the service checkout.
        for name, source in profile.get("dependency_paths", {}).items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("dependency name must be a single directory name")
            source = Path(source)
            if not source.is_dir():
                raise RuntimeError(f"prepared dependency missing: {name}")
            target = checkout / name
            if (target / ".git").exists():
                if name in excluded:
                    raise RuntimeError("prepared parser requires a fresh checkout without its Git submodule")
                # The deployment commit's submodule takes precedence over a runtime copy.
                continue
            if target.exists():
                # Tracked files and initialized submodules must never be overwritten.
                target.rmdir()
            await asyncio.to_thread(
                shutil.copytree, source, target,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            )

    async def git_network_command(self, argv, cwd, logfile):
        for attempt in range(3):
            try:
                return await self.command(argv, cwd, logfile)
            except RuntimeError as exc:
                if "exit code 128" not in str(exc) or attempt == 2:
                    raise
                await asyncio.sleep(1)

    async def execute(self, deployment):
        if (
            not re.fullmatch(r"[0-9a-f]{32}", deployment["deployment_id"])
            or not re.fullmatch(r"[0-9a-f]{40}", deployment["commit_sha"])
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", deployment["branch"])
        ):
            raise ValueError("invalid deployment identity or branch")
        if deployment["service_port"] != self.config["service_port"]:
            raise ValueError("instance port differs from assigned endpoint")
        self.cancel_event.clear()
        previous = dict(self.journal)
        self.journal = {**previous, "deployment": deployment, "stage": "checkout"}
        self.journal.pop("interrupted", None)
        self.journal.pop("result", None)
        self.journal.pop("reported", None)
        self.save()
        checkout = self.root / "releases" / deployment["deployment_id"]
        logfile = self.root / (deployment["deployment_id"] + ".deploy.log")
        logfile.touch(mode=0o600, exist_ok=True)
        logfile.chmod(0o600)
        status, error = "failed", ""
        try:
            repository = deployment["repository"]
            profile = self.profiles.get(repository)
            if profile is None:
                raise ValueError("worker does not support this repository")
            if deployment.get("profile_revision") != profile.get("profile_revision"):
                raise ValueError("deployment profile changed; submit a new deployment")
            await self.command(["git", "init", str(checkout)], self.root, logfile)
            await self.command(
                ["git", "remote", "add", "origin", REPOSITORY_URL.format(repository=repository)], checkout, logfile,
            )
            # Fetch the fixed commit only, including when using an optional local repository.
            try:
                fetch = self.command if profile.get("source_path") else self.git_network_command
                await fetch(
                    ["git", "fetch", "--depth=1", profile.get("source_path") or "origin", deployment["commit_sha"]],
                    checkout, logfile,
                )
            except RuntimeError:
                if not profile.get("source_path"):
                    raise
                await self.git_network_command(
                    ["git", "fetch", "--depth=1", "origin", deployment["commit_sha"]], checkout, logfile,
                )
            await self.command(["git", "checkout", "--detach", deployment["commit_sha"]], checkout, logfile)
            await self.prepare_dependencies(checkout, profile, logfile)
            self.journal["stage"] = "starting"
            self.save()
            if self.cancel_event.is_set():
                return
            argv = self.prepare_start(checkout, profile)
            async with self.lifecycle_lock:
                if self.cancel_event.is_set():
                    return
                await self.stop_service()
                inherited = (
                    "PATH",
                    "HOME",
                    "USER",
                    "LANG",
                    "LC_ALL",
                    "SSL_CERT_FILE",
                    "REQUESTS_CA_BUNDLE",
                    "TCC_SERVICE_NAME",
                    "STATIC_DIR",
                    "EE_DEPLOY_ENV",
                    "DORADO_DEPLOY_ENV",
                    "QE_DEPLOY_ENV",
                    "ASYNC_DEPLOY_ENV",
                    "WORKER_NUM",
                    "THREAD_NUM",
                )
                env = {key: os.environ[key] for key in inherited if key in os.environ}
                env["PORT0"] = str(self.config["service_port"])
                env["PYTHONPATH"] = os.pathsep.join([str(checkout), *profile["extra_python_paths"]])
                with logfile.open("ab") as output:
                    process = await asyncio.create_subprocess_exec(
                        *argv,
                        cwd=checkout,
                        env=env,
                        stdout=output,
                        stderr=output,
                        start_new_session=True,
                    )
                self.journal.update(
                    pid=process.pid, process_identity=self.process_identity(process.pid), stage="readiness",
                )
                self.save()
            path = profile.get("readiness_path", "/healthz")
            expected_json = profile.get("readiness_json")
            expected_text = profile.get(
                "readiness_text", "ok" if path == "/healthz" and expected_json is None else None,
            )
            expected_status = profile.get("readiness_status", 200)
            for _ in range(profile.get("readiness_attempts", 120)):
                if self.cancel_event.is_set():
                    return
                if process.returncode is not None:
                    raise RuntimeError("service exited before readiness")
                try:
                    response = await self.client.get(f"http://127.0.0.1:{self.config['service_port']}{path}", timeout=2)
                    if self.readiness_owned() and response.status_code == expected_status and (
                        expected_text is None or response.text.strip() == expected_text
                    ) and (expected_json is None or all(
                        response.json().get(key) == value for key, value in expected_json.items()
                    )):
                        status = "succeeded"
                        break
                except (httpx.HTTPError, ValueError, AttributeError):
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("service readiness timed out")
        except Exception as exc:
            error = safe_text(str(exc))[:1000]
            if self.cancel_event.is_set():
                return
        finally:
            if (
                self.journal.get("interrupted")
                and self.journal.get("deployment", {}).get("deployment_id") == deployment["deployment_id"]
            ):
                self.journal["result"] = {
                    "status": "failed",
                    "error": "worker heartbeat or shutdown interrupted deployment; review before retry",
                    "commit_sha": deployment["commit_sha"],
                    "log_path": str(logfile),
                }
                self.save()
        if self.cancel_event.is_set():
            return
        self.journal["result"] = {
            "status": status,
            "error": error,
            "commit_sha": deployment["commit_sha"],
            "log_path": str(logfile),
        }
        self.save()
        await self.report()


async def main(config_path):
    from conciliator_deployment_worker.multi_worker import MachineWorker

    config_path = await asyncio.to_thread(lambda: Path(config_path).expanduser().resolve())
    config = json.loads(await asyncio.to_thread(config_path.read_text))
    if config.get("protocol_version") != 2:
        raise ValueError("only deployment protocol v2 is supported")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", config.get("worker_id", "")):
        raise ValueError("worker_id must be a stable name using letters, digits, underscores or hyphens")
    config.setdefault("root_path", str(config_path.parent / config["worker_id"]))
    async with httpx.AsyncClient(base_url=config["api_base"], timeout=15, trust_env=False) as client:
        worker = MachineWorker(config, client)
        loop = asyncio.get_running_loop()
        for name in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(name, worker.shutdown)
        await worker.run()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    asyncio.run(main(parser.parse_args().config))
