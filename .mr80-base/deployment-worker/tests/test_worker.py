import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from conciliator_deployment_worker.worker import DeploymentWorker as Runner


@pytest.mark.asyncio
@pytest.mark.parametrize("initially_alive", [True, False])
async def test_stop_waits_for_port_after_launcher_exits(tmp_path, monkeypatch, initially_alive):
    async with httpx.AsyncClient() as client:
        runner = Runner({"root_path": str(tmp_path)}, client)
        runner.journal.update(pid=123, process_identity="original")
        runner.save()
        monkeypatch.setattr(runner, "service_alive", Mock(side_effect=[initially_alive, False, False, False]))
        monkeypatch.setattr(runner, "port_available", Mock(side_effect=[False, False, True, True]))
        killpg = Mock()
        monkeypatch.setattr("conciliator_deployment_worker.worker.os.killpg", killpg)
        monkeypatch.setattr("conciliator_deployment_worker.worker.asyncio.sleep", AsyncMock())

        await runner.stop_service()

        assert "pid" not in runner.journal
        assert "process_identity" not in json.loads(runner.journal_path.read_text())
        assert killpg.call_count == int(initially_alive)


@pytest.mark.asyncio
async def test_stop_refuses_persistent_port_occupancy_and_preserves_identity(tmp_path, monkeypatch):
    async with httpx.AsyncClient() as client:
        runner = Runner({"root_path": str(tmp_path)}, client)
        runner.journal.update(pid=123, process_identity="original")
        runner.save()
        monkeypatch.setattr(runner, "service_alive", Mock(return_value=False))
        monkeypatch.setattr(runner, "port_available", Mock(return_value=False))
        killpg = Mock()
        monkeypatch.setattr("conciliator_deployment_worker.worker.os.killpg", killpg)
        sleep = AsyncMock()
        monkeypatch.setattr("conciliator_deployment_worker.worker.asyncio.sleep", sleep)

        with pytest.raises(RuntimeError, match="instance port remains occupied; refusing release"):
            await runner.stop_service()

        assert 0 < sleep.await_count <= 50
        assert json.loads(runner.journal_path.read_text()) == {"pid": 123, "process_identity": "original"}
        killpg.assert_not_called()


def test_aeolus_start_adapter_uses_configured_python_and_removes_port_wide_kill(tmp_path):
    python_bin = tmp_path / "python env/bin/python"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to(sys.executable)
    checkout = tmp_path / "checkout"
    (checkout / "run").mkdir(parents=True)
    (checkout / "run/dev_run.sh").write_text(
        "#!/bin/bash\nkill -9 $(lsof -i:$PORT0|awk 'NR>=2{print $2}')\n"
        "doas -p data.aeolus.api_v3 python -m bytedunicorn aeolus.app:app\n"
    )
    client = httpx.AsyncClient()
    runner = Runner({"root_path": str(tmp_path / "worker"), "python_path": str(python_bin)}, client)
    argv = runner.prepare_start(checkout)
    adapted = (checkout / "run/worker_dev_run.sh").read_text()
    assert argv[0] == "bash"
    assert "kill -9" not in adapted
    assert str(python_bin) in adapted
    assert "kill -9" in (checkout / "run/dev_run.sh").read_text()


def test_worker_rejects_string_instead_of_extra_python_path_list(tmp_path):
    with pytest.raises(ValueError, match="extra_python_paths"):
        Runner({"root_path": str(tmp_path), "extra_python_paths": "/opt/tiger/pyutil"}, httpx.AsyncClient())


def test_vqs_start_adapter_preserves_environment_and_uses_vqs_identity(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "run").mkdir(parents=True)
    original = (
        "#!/bin/bash\nexport TCC_SERVICE_NAME=data.vqs.conf\n"
        "kill -9 $(lsof -i:$PORT0|awk 'NR>=2{print $2}')\n"
        "doas -p data.vqs.api python -m bytedunicorn vqs.app:app\n"
    )
    (checkout / "run/dev_run.sh").write_text(original)
    profile = {"python_path": sys.executable, "service_identity": "data.vqs.api"}
    runner = Runner({"root_path": str(tmp_path / "worker")}, httpx.AsyncClient())
    runner.prepare_start(checkout, profile)
    adapted = (checkout / "run/worker_dev_run.sh").read_text()
    assert "export TCC_SERVICE_NAME=data.vqs.conf" in adapted
    assert f"exec doas -p data.vqs.api {sys.executable} -m bytedunicorn vqs.app:app" in adapted
    assert "kill -9" not in adapted
    assert (checkout / "run/dev_run.sh").read_text() == original
    with pytest.raises(RuntimeError, match="service_identity"):
        runner.prepare_start(checkout, {**profile, "service_identity": "data.aeolus.api_v3"})


@pytest.mark.asyncio
async def test_prepared_dependencies_are_copied_without_overwriting_checkout(tmp_path):
    source = tmp_path / "prepared"
    source.mkdir()
    (source / "__init__.py").write_text("READY = True")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "sdk").mkdir()
    async with httpx.AsyncClient() as client:
        runner = Runner({"root_path": str(tmp_path / "worker")}, client)
        profile = {"dependency_paths": {"sdk": str(source)}}
        await runner.prepare_dependencies(checkout, profile, tmp_path / "deploy.log")
        (checkout / "sdk/__init__.py").write_text("changed")
        assert (source / "__init__.py").read_text() == "READY = True"
        with pytest.raises(OSError):
            await runner.prepare_dependencies(checkout, profile, tmp_path / "deploy.log")
        with pytest.raises(ValueError, match="single directory"):
            await runner.prepare_dependencies(
                checkout, {"dependency_paths": {"../outside": str(source)}}, tmp_path / "deploy.log",
            )
        with pytest.raises(RuntimeError, match="prepared dependency missing"):
            await runner.prepare_dependencies(
                checkout, {"dependency_paths": {"missing": str(tmp_path / "missing")}}, tmp_path / "deploy.log",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["dp/aeolus", "dp/VizQueryService"])
@pytest.mark.parametrize("prepared_parser", [True, False])
@pytest.mark.parametrize("other_submodule", [True, False])
async def test_explicit_parser_dependency_replaces_only_parser_submodule(
    tmp_path, repository, prepared_parser, other_submodule,
):
    def git(path, *args):
        return subprocess.check_output(
            ["git", "-c", "protocol.file.allow=always", "-c", "user.name=test",
             "-c", "user.email=test@example.invalid", *args],
            cwd=path, text=True, stderr=subprocess.DEVNULL,
        ).strip()

    component = tmp_path / "component"
    component.mkdir()
    git(component, "init")
    (component / "value.txt").write_text("pinned submodule")
    git(component, "add", ".")
    git(component, "commit", "-m", "fixture")
    pinned = git(component, "rev-parse", "HEAD")
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "submodule", "add", str(component), "aeolus_expr_parse")
    if other_submodule:
        git(source, "submodule", "add", str(component), "other")
    git(source, "commit", "-am", "fixture")
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", "--no-recurse-submodules", str(source), str(checkout))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "value.txt").write_text("SCM parser")
    async with httpx.AsyncClient() as client:
        runner = Runner({"root_path": str(tmp_path / "worker"), "repository": repository}, client)

        async def local_network(argv, cwd, logfile):
            await runner.command(["git", "-c", "protocol.file.allow=always", *argv[1:]], cwd, logfile)

        runner.git_network_command = local_network
        profile = {"dependency_paths": {"aeolus_expr_parse": str(prepared)}} if prepared_parser else {}
        await runner.prepare_dependencies(checkout, profile, tmp_path / "deploy.log")
        if prepared_parser:
            stale = tmp_path / "stale-checkout"
            git(tmp_path, "clone", "--no-recurse-submodules", str(source), str(stale))
            git(stale, "submodule", "update", "--init", "--", "aeolus_expr_parse")
            with pytest.raises(RuntimeError, match="fresh checkout"):
                await runner.prepare_dependencies(stale, profile, tmp_path / "deploy.log")
            assert (stale / "aeolus_expr_parse/value.txt").read_text() == "pinned submodule"

    parser = checkout / "aeolus_expr_parse"
    assert (parser / "value.txt").read_text() == ("SCM parser" if prepared_parser else "pinned submodule")
    assert (parser / ".git").exists() is not prepared_parser
    assert (checkout / ".git/modules/aeolus_expr_parse").exists() is not prepared_parser
    assert (prepared / "value.txt").read_text() == "SCM parser"
    if other_submodule:
        assert git(checkout / "other", "rev-parse", "HEAD") == pinned


@pytest.mark.asyncio
@pytest.mark.parametrize("active_config", [
    [],
    ["-c", "submodule.active=new_component"],
    [
        "-c", "submodule.active=new_component",
        "-c", "submodule.envmanager.active=false", "-c", "submodule.nested.active=false",
    ],
])
async def test_all_submodules_and_nested_modules_use_pinned_commits_without_runtime_overwrites(
    tmp_path, active_config,
):
    def git(path, *args):
        return subprocess.check_output(
            [
                "git", "-c", "protocol.file.allow=always",
                "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args,
            ],
            cwd=path, text=True, stderr=subprocess.DEVNULL,
        ).strip()

    def repository(name):
        path = tmp_path / name
        path.mkdir()
        git(path, "init")
        (path / "value.txt").write_text("pinned")
        git(path, "add", "value.txt")
        git(path, "commit", "-m", "fixture")
        return path

    nested = repository("nested")
    nested_sha = git(nested, "rev-parse", "HEAD")
    component = repository("component")
    git(component, "submodule", "add", str(nested), "nested")
    git(component, "commit", "-am", "nested fixture")
    component_sha = git(component, "rev-parse", "HEAD")
    envmanager = repository("envmanager")
    envmanager_sha = git(envmanager, "rev-parse", "HEAD")
    source = repository("source")
    git(source, "submodule", "add", str(component), "new_component")
    git(source, "submodule", "add", str(envmanager), "envmanager")
    git(source, "commit", "-am", "component fixture")
    # The upstream repositories advance after the parent pins their versions.
    for path in (nested, component):
        (path / "value.txt").write_text("upstream changed")
        git(path, "commit", "-am", "upstream fixture")
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", "--no-recurse-submodules", str(source), str(checkout))
    async with httpx.AsyncClient() as client:
        runner = Runner({"root_path": str(tmp_path / "worker")}, client)

        async def local_network(argv, cwd, logfile):
            await runner.command(
                ["git", "-c", "protocol.file.allow=always", *active_config, *argv[1:]], cwd, logfile,
            )

        runner.git_network_command = local_network
        logfile = tmp_path / "deploy.log"
        await runner.prepare_dependencies(checkout, {}, logfile)
        assert git(checkout / "new_component", "rev-parse", "HEAD") == component_sha
        assert git(checkout / "new_component/nested", "rev-parse", "HEAD") == nested_sha
        assert git(checkout / "envmanager", "rev-parse", "HEAD") == envmanager_sha
        assert (checkout / "new_component/nested/value.txt").read_text() == "pinned"
        await runner.prepare_dependencies(
            checkout, {"dependency_paths": {"new_component": str(component)}}, logfile,
        )
        assert git(checkout / "new_component", "rev-parse", "HEAD") == component_sha
        assert (checkout / "new_component/value.txt").read_text() == "pinned"
        # Catch a partial/dirty checkout even if the update command reports success.
        (checkout / "new_component/nested/value.txt").write_text("incomplete")
        runner.git_network_command = AsyncMock()
        with pytest.raises(RuntimeError, match="deployment command failed"):
            await runner.prepare_dependencies(checkout, {}, logfile)
        # A successful command must not conceal a missing top-level or nested checkout.
        (checkout / "new_component/nested/value.txt").write_text("pinned")
        for path in (checkout / "envmanager", checkout / "new_component/nested"):
            gitfile = (path / ".git").read_text()
            (path / ".git").unlink()
            with pytest.raises(RuntimeError, match="deployment command failed"):
                await runner.prepare_dependencies(checkout, {}, logfile)
            (path / ".git").write_text(gitfile)


@pytest.mark.asyncio
@pytest.mark.parametrize("repository,revision", [("other/repo", "unknown"), ("dp/aeolus", "stale")])
async def test_invalid_profile_reports_failure_before_running_commands(tmp_path, repository, revision):
    async with httpx.AsyncClient() as client:
        runner = Runner({"root_path": str(tmp_path), "service_port": 12356, "maat_url": "https://test"}, client)
        runner.command = AsyncMock()
        runner.report = AsyncMock()
        await runner.execute({
            "deployment_id": "a" * 32, "repository": repository, "profile_revision": revision,
            "commit_sha": "b" * 40, "branch": "master", "service_port": 12356, "maat_url": "https://test",
        })
        assert runner.journal["result"]["status"] == "failed"
        runner.command.assert_not_awaited()
        runner.report.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("message,attempts", [
    ("deployment command failed with exit code 128; inspect deploy.log", 3),
    ("deployment cancelled", 1),
])
async def test_git_retries_are_bounded_and_do_not_retry_cancellation(tmp_path, message, attempts):
    async with httpx.AsyncClient() as client:
        runner = Runner({"root_path": str(tmp_path)}, client)
        runner.command = AsyncMock(side_effect=RuntimeError(message))
        with pytest.raises(RuntimeError, match=message):
            await runner.git_network_command(["git", "fetch"], tmp_path, tmp_path / "deploy.log")
        assert runner.command.await_count == attempts




@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 99])
async def test_cli_rejects_unsupported_protocol_before_contacting_backend(tmp_path, monkeypatch, version):
    from conciliator_deployment_worker import worker

    config = {"worker_id": "machine", "api_base": "http://test"}
    if version is not None:
        config["protocol_version"] = version
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(config))
    client = Mock(side_effect=AssertionError("invalid config must not contact backend"))
    monkeypatch.setattr(worker.httpx, "AsyncClient", client)
    with pytest.raises(ValueError, match="only deployment protocol v2"):
        await worker.main(str(path))
    client.assert_not_called()


@pytest.mark.asyncio
async def test_cli_starts_only_machine_supervisor_and_keeps_default_root(tmp_path, monkeypatch):
    from conciliator_deployment_worker import multi_worker, worker

    example = Path(__file__).parents[1] / "examples/deployment_worker.v2.example.json"
    config = json.loads(example.read_text())
    config["api_base"] = "http://test"
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(config))
    started = []

    async def capture(manager):
        started.append(manager)

    monkeypatch.setattr(multi_worker.MachineWorker, "run", capture)
    await worker.main(str(path))
    assert len(started) == 1 and started[0].root == tmp_path / config["worker_id"]
    assert set(started[0].profiles) == {"dp/aeolus", "dp/VizQueryService"}
    assert started[0].profiles["dp/aeolus"]["py310"]["python_version"] == "3.10"
    assert started[0].profiles["dp/VizQueryService"]["py37"]["python_version"] == "3.7"
    assert set(started[0].profiles["dp/VizQueryService"]["py37"]["dependency_paths"]) == {
        "aeolus_expr_parse",
    }
    for service in config["services"].values():
        for profile in service["runtime_profiles"].values():
            assert set(profile) == {"python_path", "python_version", "dependency_paths", "extra_paths"}
