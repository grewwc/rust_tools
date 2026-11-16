"""Concurrent machine worker with per-deployment supervision and durable port ownership."""

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import uuid
from pathlib import Path

import httpx

from conciliator_deployment_worker.profiles import capabilities, runtime_profiles
from conciliator_deployment_worker.worker import DeploymentWorker


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        os.chmod(temporary, 0o600)
        json.dump(value, output)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def listener_owned(port, group):
    """Fail closed if the listener cannot be attributed to the managed process group."""
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in Path(table).read_text().splitlines()[1:]:
                fields = line.split()
                if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                    inodes.add(fields[9])
        except OSError:
            return False
    if not inodes:
        return False
    owned = set()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group:
                continue
            for fd in (process / "fd").iterdir():
                with contextlib.suppress(OSError):
                    target = os.readlink(fd)
                    if target.startswith("socket:["):
                        owned.add(target[8:-1])
        except (OSError, ValueError, IndexError):
            continue
    return inodes <= owned


async def verify_runtime(profile):
    """Check prepared artifacts, never mutate or install into a shared environment."""
    artifacts = profile
    if "runtime_manifest" in profile:
        artifacts = json.loads(await asyncio.to_thread(Path(profile["runtime_manifest"]).read_text))
        if (
            artifacts.get("runtime_revision") != profile["runtime_revision"]
            or artifacts.get("python_version") != profile["python_version"]
        ):
            raise RuntimeError("prepared runtime manifest differs from selected runtime")
    process = await asyncio.create_subprocess_exec(
        profile["python_path"],
        "-c",
        "import sys; print('%d.%d' % sys.version_info[:2])",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode or stdout.decode().strip() != profile["python_version"]:
        raise RuntimeError("actual Python version differs from selected runtime")
    for path in [*profile["extra_python_paths"], *profile.get("dependency_paths", {}).values()]:
        if not await asyncio.to_thread(Path(path).is_dir):
            raise RuntimeError("prepared runtime dependency is missing")
    files = artifacts.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("runtime files must be a checksum mapping")
    for filename, expected in files.items():
        digest = await asyncio.to_thread(lambda p=filename: hashlib.sha256(Path(p).read_bytes()).hexdigest())
        if digest != expected:
            raise RuntimeError("prepared artifact checksum differs from runtime configuration")


class Instance(DeploymentWorker):
    def __init__(self, manager, assignment, profile, port, root):
        self.manager = manager
        self.assignment = assignment
        self.runtime_profile = profile
        super().__init__(
            {
                **profile,
                "repository": assignment["repository"],
                "worker_id": manager.config["worker_id"],
                "root_path": str(root),
                "service_port": port,
            },
            manager.client,
        )

    def save(self):
        atomic_json(self.journal_path, self.journal)

    def readiness_owned(self):
        return self.service_alive() and listener_owned(self.config["service_port"], self.journal["pid"])

    async def prepare_dependencies(self, checkout, profile, logfile):
        preparation = asyncio.create_task(super().prepare_dependencies(checkout, profile, logfile))
        try:
            await asyncio.shield(preparation)
        except asyncio.CancelledError:
            # to_thread copies cannot be cancelled; wait before deleting the instance checkout.
            self.cancel_event.set()
            with contextlib.suppress(Exception):
                await preparation
            raise

    async def stop_service(self):
        command_pid = self.journal.get("command_pid")
        if command_pid and self.process_identity(command_pid) == self.journal.get("command_identity"):
            os.killpg(command_pid, signal.SIGKILL)
            for _ in range(50):
                if self.process_identity(command_pid) != self.journal.get("command_identity"):
                    break
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("owned checkout command did not stop")
        await super().stop_service()

    async def report(self):
        result = self.journal.get("result")
        if not result or self.journal.get("reported"):
            return
        record = self.assignment
        identity = {"session_id": self.manager.session_id, "generation": record["generation"]}
        prefix = f"/api/deployment-workers/{self.manager.config['worker_id']}/deployments/{record['deployment_id']}"
        if result["status"] == "succeeded":
            if not self.readiness_owned():
                raise RuntimeError("listener identity changed before endpoint-ready")
            payload = {
                **identity,
                "commit_sha": record["commit_sha"],
                **{key: record[key] for key in ("runtime_profile_id", "profile_revision", "runtime_revision")},
                "request_id": record["deployment_id"],
                "runtime_attempt_id": self.journal["runtime_attempt_id"],
                "endpoint_host": self.manager.config["advertised_host"],
                "endpoint_port": self.config["service_port"],
                "log_path": result["log_path"],
            }
            self.journal["endpoint_payload"] = {k: v for k, v in payload.items() if k != "session_id"}
            self.save()
            response = await self.manager.api(prefix + "/endpoint-ready", payload)
            if not response["accepted"]:
                self.cancel_event.set()
        else:
            await self.manager.api(prefix + "/result", {**identity, **result})
        self.journal["reported"] = True
        self.save()

    async def execute_assignment(self):
        record = self.assignment
        try:
            await verify_runtime(self.runtime_profile)
            self.journal.setdefault("runtime_attempt_id", uuid.uuid4().hex)
            self.save()
            await super().execute(
                {
                    **record,
                    "service_port": self.config["service_port"],
                        "profile_revision": self.runtime_profile["profile_revision"],
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.journal.get("result"):
                # A lost report response must replay the original result, not become a conflicting failure.
                return
            # Store a safe fixed error, not arbitrary filesystem/manifest contents.
            self.journal["result"] = {
                "status": "failed",
                "error": f"runtime execution failed ({type(exc).__name__})",
                "commit_sha": record["commit_sha"],
                "log_path": str(self.root / f"{record['deployment_id']}.deploy.log"),
            }
            self.journal.pop("reported", None)
            self.save()
            await self.report()


class MachineWorker:
    def __init__(self, config, client):
        self.config = config
        self.client = client
        self.profiles = runtime_profiles(config)
        self.root = Path(config["root_path"]).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (self.root / "journal.json").exists():
            raise ValueError("machine root must use per-deployment journals under instances/")
        self.instances_root = self.root / "instances"
        self.instances_root.mkdir(exist_ok=True, mode=0o700)
        self.ports_path = self.root / "ports.json"
        self.ports = json.loads(self.ports_path.read_text()) if self.ports_path.exists() else {}
        self.rejections_path = self.root / "rejections.json"
        self.rejections = json.loads(self.rejections_path.read_text()) if self.rejections_path.exists() else {}
        self.instances = {}
        self.tasks = {}
        self.maintenance_tasks = {}
        self.session_id = ""
        self.stop_event = asyncio.Event()
        self.claim_id = uuid.uuid4().hex
        self.port_lock = asyncio.Lock()

    async def api(self, path, body):
        response = await self.client.post(path, json=body)
        response.raise_for_status()
        return response.json()

    async def reserve_port(self, deployment_id):
        async with self.port_lock:
            if deployment_id in self.ports:
                return self.ports[deployment_id]
            for _ in range(100):
                with socket.socket() as probe:
                    probe.bind(("0.0.0.0", 0))
                    port = probe.getsockname()[1]
                if port not in self.ports.values():
                    self.ports[deployment_id] = port
                    atomic_json(self.ports_path, self.ports)
                    return port
            raise RuntimeError("cannot allocate a distinct service port")

    async def start_instance(self, record):
        deployment_id = record["deployment_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", deployment_id):
            raise ValueError("invalid deployment identity")
        if record.get("released"):
            return
        if deployment_id in self.instances:
            if self.instances[deployment_id].assignment != record:
                raise RuntimeError("conflicting deployment replay")
            return
        profile = self.profiles.get(record["repository"], {}).get(record["runtime_profile_id"])
        if record["status"] != "running" or profile is None or any(
            profile[key] != record[key] for key in ("profile_revision", "runtime_revision")
        ):
            self.rejections[deployment_id] = record
            atomic_json(self.rejections_path, self.rejections)
            await self.api(
                f"/api/deployment-workers/{self.config['worker_id']}/deployments/{deployment_id}/result",
                {
                    "session_id": self.session_id,
                    "generation": record["generation"],
                    "status": "failed",
                    "commit_sha": record["commit_sha"],
                    "error": "selected runtime profile changed or is unavailable",
                },
            )
            return
        port = await self.reserve_port(deployment_id)
        root = self.instances_root / deployment_id
        root.mkdir(exist_ok=True, mode=0o700)
        # Persist the immutable assignment/profile before scheduling any subprocess.
        atomic_json(root / "assignment.json", {"deployment": record, "profile": profile, "port": port})
        instance = Instance(self, record, profile, port, root)
        self.instances[deployment_id] = instance
        self.tasks[deployment_id] = asyncio.create_task(instance.execute_assignment())

    async def restore(self, expected):
        released = {r["deployment_id"]: r for r in expected if r.get("released")}
        assignments = {r["deployment_id"]: r for r in expected if not r.get("released")}
        for root in self.instances_root.iterdir():
            if not root.is_dir() or not (root / "assignment.json").is_file():
                continue
            stored = json.loads((root / "assignment.json").read_text())
            record = stored["deployment"]
            deployment_id = record["deployment_id"]
            instance = Instance(self, record, stored["profile"], stored["port"], root)
            if instance.journal.get("released"):
                self.ports.pop(deployment_id, None)
                atomic_json(self.ports_path, self.ports)
                continue
            if deployment_id in released and released[deployment_id]["generation"] == record["generation"]:
                if instance.service_alive() or not instance.port_available():
                    raise RuntimeError("released deployment still has local processes or listener")
                instance.journal["released"] = True
                instance.save()
                self.ports.pop(deployment_id, None)
                atomic_json(self.ports_path, self.ports)
                continue
            if root.name != deployment_id or self.ports.get(deployment_id) != stored["port"]:
                raise RuntimeError("port ledger and journal differ; manual reconciliation required")
            if deployment_id not in assignments or assignments[deployment_id]["generation"] != record["generation"]:
                raise RuntimeError("local deployment is absent from server reconciliation")
            self.instances[deployment_id] = instance
            if not instance.journal.get("result"):
                instance.journal["result"] = {
                    "status": "failed",
                    "error": "worker restarted during deployment; cleanup required",
                    "commit_sha": record["commit_sha"],
                    "log_path": "",
                }
                instance.save()
        for deployment_id in list(self.rejections):
            if deployment_id in released:
                self.rejections.pop(deployment_id)
        atomic_json(self.rejections_path, self.rejections)
        if set(self.instances) | set(self.rejections) != set(assignments):
            raise RuntimeError("missing instance journal; refusing new deployments")
        if set(self.ports) != set(self.instances):
            raise RuntimeError("orphan port allocation; refusing new deployments")

    async def observation(self, deployment_id, instance):
        task = self.tasks.get(deployment_id)
        status = "starting" if task and not task.done() and not instance.journal.get("result") else "unknown"
        if instance.service_alive():
            status = "unhealthy"
            if await asyncio.to_thread(instance.readiness_owned):
                try:
                    profile = instance.runtime_profile
                    path = profile.get("readiness_path", "/healthz")
                    response = await self.client.get(
                        f"http://127.0.0.1:{instance.config['service_port']}{path}",
                        timeout=2,
                    )
                    expected_json = profile.get("readiness_json")
                    text = profile.get("readiness_text", None if expected_json else "ok")
                    if (
                        response.status_code == profile.get("readiness_status", 200)
                        and (text is None or response.text.strip() == text)
                        and (
                            expected_json is None or all(response.json().get(k) == v for k, v in expected_json.items())
                        )
                    ):
                        status = "healthy"
                except (httpx.HTTPError, ValueError, AttributeError):
                    pass
        return {
            "deployment_id": deployment_id,
            "generation": instance.assignment["generation"],
            "service_status": status,
            "stage": instance.journal.get("stage", "checkout"),
        }

    async def observations(self):
        # Slow/unhealthy services must not serially delay the entire machine's heartbeat.
        result = list(
            await asyncio.gather(*(self.observation(key, instance) for key, instance in list(self.instances.items())))
        )
        result.extend(
            {"deployment_id": key, "generation": record["generation"], "service_status": "stopped", "stage": "rejected"}
            for key, record in self.rejections.items()
        )
        return result

    async def stop_instance(self, command):
        deployment_id = command["deployment_id"]
        instance = self.instances.get(deployment_id)
        if instance is None:
            rejected = self.rejections.get(deployment_id)
            if rejected and rejected["generation"] == command["generation"]:
                await self.api(
                    f"/api/deployment-workers/{self.config['worker_id']}/stopped",
                    {
                        "session_id": self.session_id,
                        "deployment_id": deployment_id,
                        "generation": command["generation"],
                    },
                )
                self.rejections.pop(deployment_id)
                atomic_json(self.rejections_path, self.rejections)
            return  # Never manufacture a stopped ACK without local evidence.
        if instance.assignment["generation"] != command["generation"]:
            raise RuntimeError("stale release command")
        instance.cancel_event.set()
        task = self.tasks.get(deployment_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        async with instance.lifecycle_lock:
            await instance.stop_service()
            checkout = instance.root / "releases" / deployment_id
            if checkout.is_symlink():
                raise RuntimeError("refusing cleanup of a replaced checkout path")
            if checkout.exists():
                await asyncio.to_thread(shutil.rmtree, checkout)
            reply = await self.api(
                f"/api/deployment-workers/{self.config['worker_id']}/stopped",
                {
                    "session_id": self.session_id,
                    "deployment_id": deployment_id,
                    "generation": command["generation"],
                },
            )
            if not reply.get("port_released", False):
                return
            instance.journal["released"] = True
            instance.save()
            async with self.port_lock:
                self.ports.pop(deployment_id, None)
                atomic_json(self.ports_path, self.ports)
            self.instances.pop(deployment_id, None)
            self.tasks.pop(deployment_id, None)

    async def tick(self, *, reconciled=False):
        observations = await self.observations()
        for observation in observations:
            instance = self.instances.get(observation["deployment_id"])
            if instance is None or (instance.journal.get("result") or {}).get("status") != "succeeded":
                continue
            failures = (
                0 if observation["service_status"] == "healthy" else instance.journal.get("health_failures", 0) + 1
            )
            instance.journal["health_failures"] = failures
            if failures >= 3:
                instance.journal["result"] = {
                    "status": "failed",
                    "error": "service lost health or listener ownership after local readiness",
                    "commit_sha": instance.assignment["commit_sha"],
                    "log_path": instance.journal["result"]["log_path"],
                }
                instance.journal.pop("reported", None)
            instance.save()
        reply = await self.api(
            f"/api/deployment-workers/{self.config['worker_id']}/heartbeat",
            {
                "session_id": self.session_id,
                "reconciled": reconciled,
                "instances": observations,
            },
        )
        commands = {command["deployment_id"]: command for command in reply["commands"]}
        for deployment_id, (_, task) in list(self.maintenance_tasks.items()):
            if task.done():
                self.maintenance_tasks.pop(deployment_id)
        for deployment_id in commands.keys() | self.instances.keys():
            command = commands.get(deployment_id)
            previous = self.maintenance_tasks.get(deployment_id)
            if previous:
                if command and previous[0] is None:
                    previous[1].cancel()  # Cleanup takes priority over a delayed result report.
                else:
                    continue
            instance = self.instances.get(deployment_id)
            startup = self.tasks.get(deployment_id)
            if command or (
                instance
                and (startup is None or startup.done())
                and instance.journal.get("result")
                and not instance.journal.get("reported")
            ):
                task = asyncio.create_task(self.maintain_instance(command, instance))
                self.maintenance_tasks[deployment_id] = (command, task)
        return reply

    async def maintain_instance(self, command, instance):
        # One slow stop or failed report must not hold up other instances, claims or heartbeats.
        try:
            if command:
                await self.stop_instance(command)
            else:
                await instance.report()
        except Exception as exc:
            print("instance maintenance pending:", type(exc).__name__, flush=True)

    def shutdown(self):
        self.stop_event.set()
        for task in [*self.tasks.values(), *(entry[1] for entry in self.maintenance_tasks.values())]:
            if not task.done():
                task.cancel()

    async def run(self):
        with (self.root / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reply = await self.api(
                "/api/deployment-workers/register",
                {
                    "protocol_version": 2,
                    "worker_id": self.config["worker_id"],
                    "display_name": self.config.get("display_name", self.config["worker_id"]),
                    "vmid": self.config.get("vmid"),
                    "advertised_host": self.config["advertised_host"],
                    "runtime_profiles": capabilities(self.profiles),
                },
            )
            self.session_id = reply["session_id"]
            await self.restore(reply["deployments"])
            await self.tick(reconciled=True)
            try:
                while not self.stop_event.is_set():
                    try:
                        heartbeat = await self.tick()
                        if heartbeat["ready"]:
                            response = await self.api(
                                f"/api/deployment-workers/{self.config['worker_id']}/claim",
                                {
                                    "session_id": self.session_id,
                                    "request_id": self.claim_id,
                                },
                            )
                            if response["deployment"]:
                                await self.start_instance(response["deployment"])
                                self.claim_id = uuid.uuid4().hex
                    except Exception as exc:
                        print("machine poll failed:", type(exc).__name__, flush=True)
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self.stop_event.wait(), 2)
            finally:
                self.shutdown()
                await asyncio.gather(
                    *self.tasks.values(),
                    *(entry[1] for entry in self.maintenance_tasks.values()),
                    return_exceptions=True,
                )
