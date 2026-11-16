import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conciliator_deployment_worker.multi_worker import MachineWorker


def supervisor(instances=None, commands=None):
    worker = object.__new__(MachineWorker)
    worker.config = {"worker_id": "test-machine"}
    worker.session_id = "test-session"
    worker.instances = instances or {}
    worker.tasks = {}
    worker.maintenance_tasks = {}
    worker.observations = AsyncMock(return_value=[])
    worker.api = AsyncMock(return_value={"ready": True, "commands": commands or []})
    return worker


async def settle(worker):
    await asyncio.wait_for(asyncio.gather(*(entry[1] for entry in worker.maintenance_tasks.values())), 1)


@pytest.mark.asyncio
async def test_slow_stop_does_not_block_heartbeat_or_another_instance_cleanup():
    commands = [{"deployment_id": name, "generation": 1, "command": "release"} for name in ("slow", "fast")]
    worker = supervisor(commands=commands)
    started, finish, fast_stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def stop(command):
        if command["deployment_id"] == "slow":
            started.set()
            await finish.wait()
        else:
            fast_stopped.set()

    worker.stop_instance = AsyncMock(side_effect=stop)
    try:
        assert (await asyncio.wait_for(worker.tick(), 1))["ready"]
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(fast_stopped.wait(), 1)
        worker.api.return_value = {"ready": True, "commands": commands[:1]}
        assert (await asyncio.wait_for(worker.tick(), 1))["ready"]
        assert worker.api.await_count == 2
        assert worker.stop_instance.await_count == 2  # No duplicate cleanup for the slow instance.
    finally:
        finish.set()
        await settle(worker)


@pytest.mark.asyncio
async def test_failed_result_report_is_retried_without_starving_another_instance():
    failed = SimpleNamespace(journal={"result": {"status": "failed"}}, report=AsyncMock(side_effect=RuntimeError()))
    healthy = SimpleNamespace(journal={"result": {"status": "failed"}})

    async def report():
        healthy.journal["reported"] = True

    healthy.report = AsyncMock(side_effect=report)
    worker = supervisor({"failed": failed, "healthy": healthy})
    await worker.tick()
    await settle(worker)
    assert healthy.journal["reported"]
    await worker.tick()
    await settle(worker)
    assert worker.api.await_count == 2
    assert failed.report.await_count == 2 and healthy.report.await_count == 1


@pytest.mark.asyncio
async def test_cleanup_cancels_a_delayed_result_report():
    entered, cancelled, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def report():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    instance = SimpleNamespace(journal={"result": {"status": "failed"}}, report=report)
    worker = supervisor({"one": instance})

    async def stop(_command):
        stopped.set()

    worker.stop_instance = stop
    await worker.tick()
    await asyncio.wait_for(entered.wait(), 1)
    pending_report = worker.maintenance_tasks["one"][1]
    try:
        worker.api.return_value = {
            "ready": True,
            "commands": [{"deployment_id": "one", "generation": 1, "command": "release"}],
        }
        await worker.tick()
        await asyncio.wait_for(cancelled.wait(), 1)
        await asyncio.wait_for(stopped.wait(), 1)
        await settle(worker)
        assert pending_report.cancelled()
    finally:
        pending_report.cancel()
        await asyncio.gather(pending_report, return_exceptions=True)
