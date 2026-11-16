import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conciliator.domain.aeolus_task import CodeContext
from conciliator.integrations.coco.aeolus import (
    CocoAeolusRunner,
    CocoRunFailed,
    CocoSessionContext,
    build_coding_prompt,
    build_evaluation_prompt,
    build_evaluation_retry_prompt,
    build_feedback_prompt,
    parse_coding_result,
    parse_evaluation_result,
)
from conciliator.integrations.coco.client import CocoEvent, CocoTaskResult
from conciliator.utils.retry import TransientError


def _marked(begin: str, end: str, body: dict) -> str:
    return f"prefix\n{begin}\n{json.dumps(body)}\n{end}\nsuffix"


@pytest.mark.asyncio
async def test_resume_coding_only_polls_persisted_run():
    client = SimpleNamespace(
        create_session=AsyncMock(), send_user_message=AsyncMock(),
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message=_marked(
                "AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END",
                {"status": "success", "summary": "完成", "commit_sha": "a" * 40,
                 "tests": [], "unresolved_items": []},
            ), error="", last_event_id="end",
        )),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent")
    result = await runner.resume_coding(session_id="session", run_id="run", event_id="start")
    assert result.result.commit_sha == "a" * 40
    client.fetch_session_result.assert_awaited_once_with(
        "session", workspace_id="workspace", run_id="run", after_event_id="start",
    )
    client.create_session.assert_not_awaited()
    client.send_user_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("final_message, retryable", [("", True), ("agent output", False)])
async def test_terminal_run_is_automatically_retryable_only_without_agent_output(
    final_message, retryable
):
    client = SimpleNamespace(fetch_session_result=AsyncMock(return_value=CocoTaskResult(
        status="failed", final_message=final_message, error="run terminated",
        last_event_id="end",
    )))
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent")

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")

    assert caught.value.retryable is retryable
    assert caught.value.replacement_safe is True


@pytest.mark.asyncio
async def test_malformed_result_gets_one_format_repair_in_the_same_session():
    malformed = _marked(
        "AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END",
        {"status": "success", "summary": "完成", "commit_sha": "short",
         "tests": [], "unresolved_items": []},
    )
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="succeeded", final_message=malformed, last_event_id="end"),
            CocoTaskResult(
                status="succeeded", final_message=_coding_envelope(), last_event_id="end-2",
            ),
        ]),
        send_user_message=AsyncMock(return_value=CocoEvent(
            event_id="repair-event", session_id="session", run_id="repair-run",
            payload={}, raw={},
        )),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent")

    result = await runner.resume_coding(
        session_id="session", run_id="run", event_id="start",
    )

    assert result.run_id == "repair-run"
    assert result.result.commit_sha == "a" * 40
    client.send_user_message.assert_awaited_once()
    prompt = client.send_user_message.await_args.kwargs["text"]
    assert "commit_sha" in prompt
    assert "只修复输出格式" in prompt


@pytest.mark.asyncio
async def test_second_malformed_result_fails_without_another_repair():
    malformed = _marked(
        "AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END",
        {"status": "success", "summary": "完成", "commit_sha": "short",
         "tests": [], "unresolved_items": []},
    )
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message=malformed, last_event_id="end",
        )),
        send_user_message=AsyncMock(return_value=CocoEvent(
            event_id="repair-event", session_id="session", run_id="repair-run",
            payload={}, raw={},
        )),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent")

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")

    assert client.send_user_message.await_count == 1
    assert caught.value.retryable is False
    assert caught.value.replacement_safe is True
    assert "commit_sha" in str(caught.value)


@pytest.mark.asyncio
async def test_event_progress_renews_stall_deadline(monkeypatch):
    """Incremental activity keeps a long-running run alive."""
    clock = {"now": 0.0}
    results = [
        CocoTaskResult(
            status="running", final_message="正在执行测试", last_event_id="event-1"
        ),
        CocoTaskResult(
            status="running", final_message="正在执行测试", last_event_id="event-2"
        ),
        CocoTaskResult(
            status="succeeded", final_message=_coding_envelope(), last_event_id="event-3"
        ),
    ]

    async def fetch(*args, **kwargs):
        clock["now"] += 400.0
        return results.pop(0)

    client = SimpleNamespace(fetch_session_result=fetch, send_interrupt=AsyncMock())
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.time",
        SimpleNamespace(monotonic=lambda: clock["now"]),
    )
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.asyncio.sleep", AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=600,
        interrupt_timeout_seconds=1,
    )

    result = await runner.resume_coding(
        session_id="session", run_id="run", event_id="start",
    )

    assert result.result.commit_sha == "a" * 40
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_stalled_run_requires_confirmed_termination_before_replacement():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", final_message="", error="", last_event_id="start"),
            CocoTaskResult(status="running", final_message="", error="", last_event_id="end",
                           lifecycle_terminal=True),
        ]), send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent", poll_interval=0,
                              stall_timeout_seconds=0, interrupt_timeout_seconds=0)
    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")
    assert caught.value.retryable is True
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_terminal_run_without_result_recovers_without_another_interrupt():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="running", final_message="", error="", last_event_id="end",
            lifecycle_terminal=True,
        )),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent", poll_interval=0,
                              interrupt_timeout_seconds=0)

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")

    assert caught.value.retryable is True
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_resumed_run_with_progress_message_only_recovers_without_interrupt():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message="实现已完成，正在执行独立复审。", error="",
            last_event_id="end", lifecycle_terminal=True,
        )),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent", poll_interval=0,
                              interrupt_timeout_seconds=0)

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")

    assert caught.value.retryable is True
    assert "without an Agent result" in str(caught.value)
    client.send_interrupt.assert_not_awaited()


def _coding_envelope() -> str:
    return _marked(
        "AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END",
        {"status": "success", "summary": "完成", "commit_sha": "a" * 40,
         "tests": [], "unresolved_items": []},
    )


def _started_event() -> CocoEvent:
    return CocoEvent(
        event_id="1", session_id="session", run_id="run",
        payload={"UserMessage": {}}, raw={},
    )


@pytest.mark.asyncio
async def test_requires_action_is_answered_instead_of_waiting_out_the_stall_window():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(
                status="requires_action",
                error="Agent is waiting for tool confirmation",
                requires_action_event_ids=("206",),
                last_event_id="207",
            ),
            CocoTaskResult(
                status="succeeded", final_message=_coding_envelope(),
                lifecycle_terminal=True, last_event_id="208",
            ),
        ]),
        send_interrupt=AsyncMock(),
        send_tool_confirmation=AsyncMock(),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent", poll_interval=0)

    result = await runner._wait("session", _started_event())

    assert result.status == "succeeded"
    client.send_tool_confirmation.assert_awaited_once()
    kwargs = client.send_tool_confirmation.await_args.kwargs
    assert kwargs["tool_use_event_id"] == "206"
    assert kwargs["decision"] == "deny"
    assert kwargs["deny_message"]
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_confirmation_event_is_answered_once():
    pending = CocoTaskResult(
        status="requires_action", error="Agent is waiting for tool confirmation",
        requires_action_event_ids=("206",), last_event_id="207",
    )
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            pending,
            pending,
            CocoTaskResult(
                status="succeeded", final_message=_coding_envelope(),
                lifecycle_terminal=True, last_event_id="208",
            ),
        ]),
        send_tool_confirmation=AsyncMock(),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        max_tool_confirmations=2,
    )

    result = await runner._wait("session", _started_event())

    assert result.status == "succeeded"
    client.send_tool_confirmation.assert_awaited_once()
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_distinct_confirmation_events_consume_the_budget():
    pending = [
        CocoTaskResult(
            status="requires_action", error="Agent is waiting for tool confirmation",
            requires_action_event_ids=(event_id,), last_event_id=event_id,
        )
        for event_id in ("206", "207", "208")
    ]
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=pending),
        send_tool_confirmation=AsyncMock(),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        max_tool_confirmations=2,
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner._wait("session", _started_event())

    assert caught.value.retryable is False
    assert "3 requests" in str(caught.value)
    assert client.send_tool_confirmation.await_count == 2
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmation_policy_fail_reports_the_pause_without_answering():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="requires_action",
            error="Agent is waiting for tool confirmation",
            requires_action_event_ids=("206",),
        )),
        send_tool_confirmation=AsyncMock(),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        tool_confirmation_policy="fail",
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner._wait("session", _started_event())

    assert "waiting for interactive tool confirmation" in str(caught.value)
    assert caught.value.retryable is False
    client.send_tool_confirmation.assert_not_awaited()


@pytest.mark.asyncio
async def test_stalled_run_is_interrupted_and_replaceable_after_terminal_confirmation():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", final_message="", error="", last_event_id="start"),
            CocoTaskResult(status="failed", final_message="", error="interrupted", last_event_id="end"),
        ]),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=1,
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")

    assert caught.value.retryable is True
    client.send_interrupt.assert_awaited_once_with(
        workspace_id="workspace", session_id="session",
    )


@pytest.mark.asyncio
async def test_stalled_run_is_not_replaceable_when_termination_is_unconfirmed():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="running", final_message="", error="", last_event_id="start",
        )),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")

    assert caught.value.retryable is False
    assert caught.value.replacement_safe is False


@pytest.mark.asyncio
async def test_wall_clock_does_not_kill_a_run_that_makes_progress(monkeypatch):
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", last_event_id="event-1"),
            CocoTaskResult(status="running", last_event_id="event-2"),
            CocoTaskResult(status="running", last_event_id="event-3"),
            CocoTaskResult(
                status="succeeded", final_message=_coding_envelope(),
                last_event_id="event-4",
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    # An hour of wall-clock time passes inside the call; the run must survive
    # because every poll brought a new event.
    monotonic_values = iter((0.0, 1000.0, 2000.0, 3000.0))
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.asyncio.sleep", AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=100,
        interrupt_timeout_seconds=1,
    )

    result = await runner.resume_coding(
        session_id="session", run_id="run", event_id="start",
    )

    assert result.result.commit_sha == "a" * 40
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_grace_is_renewed_by_later_events(monkeypatch):
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(
                status="running", last_event_id="event-1", lifecycle_terminal=True,
            ),
            CocoTaskResult(
                status="running", last_event_id="event-2", lifecycle_terminal=True,
            ),
            CocoTaskResult(
                status="running", last_event_id="event-3", lifecycle_terminal=True,
            ),
            CocoTaskResult(
                status="succeeded", final_message=_coding_envelope(),
                last_event_id="event-4", lifecycle_terminal=True,
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    monotonic_values = iter((0.0, 9.0, 18.0, 27.0))
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.asyncio.sleep", AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=100,
        interrupt_timeout_seconds=10,
    )

    result = await runner.resume_coding(
        session_id="session", run_id="run", event_id="start",
    )

    assert result.result.commit_sha == "a" * 40
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_grace_expiry_fails_without_interrupting(monkeypatch):
    terminal = CocoTaskResult(
        status="running", last_event_id="start", lifecycle_terminal=True,
    )
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=terminal),
        send_interrupt=AsyncMock(),
    )
    monotonic_values = iter((0.0, 0.0, 6.0))
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.asyncio.sleep", AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=100,
        interrupt_timeout_seconds=5,
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(
            session_id="session", run_id="run", event_id="start",
        )

    assert "ended without an Agent result" in str(caught.value)
    assert caught.value.retryable is True
    assert caught.value.platform is False
    assert caught.value.replacement_safe is True
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_projection_with_complete_result_is_accepted(monkeypatch):
    """The projection can lead the final event stream; the result still counts."""
    coding = _marked("AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END", {
        "status": "success", "summary": "完成", "commit_sha": "a" * 40,
        "tests": [], "unresolved_items": [],
    })
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="running", final_message=coding, last_event_id="event-2",
            lifecycle_terminal=True,
        )),
        send_interrupt=AsyncMock(),
    )
    clock = [0.0]

    async def advance(_seconds):
        clock[0] += 40.0

    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    monkeypatch.setattr(
        "conciliator.integrations.coco.aeolus.asyncio.sleep",
        AsyncMock(side_effect=advance),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=600,
        interrupt_timeout_seconds=30,
    )

    result = await runner.resume_coding(
        session_id="session", run_id="run", event_id="start",
    )

    assert result.result.commit_sha == "a" * 40
    client.send_interrupt.assert_not_awaited()


@pytest.mark.asyncio
async def test_interrupted_progress_message_is_not_treated_as_final_result():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", last_event_id="start"),
            CocoTaskResult(
                status="succeeded", final_message="仍在修复并运行测试",
                last_event_id="end", lifecycle_terminal=True,
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(
            session_id="session", run_id="run", event_id="start",
        )

    assert caught.value.retryable is True
    assert "ended without an Agent result" in str(caught.value)
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupt_preserves_complete_result_from_failed_projection():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", last_event_id="start"),
            CocoTaskResult(
                status="failed", final_message=_coding_envelope(),
                error="interrupted", last_event_id="end", lifecycle_terminal=True,
                platform_error="request queue size is exceed",
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    result = await runner.resume_coding(
        session_id="session", run_id="run", event_id="start",
    )

    assert result.result.commit_sha == "a" * 40
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupt_preserves_complete_result_when_projection_leads_stream():
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", last_event_id="start"),
            CocoTaskResult(
                status="running", final_message=_coding_envelope(),
                last_event_id="end", lifecycle_terminal=True,
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    result = await runner.resume_coding(
        session_id="session", run_id="run", event_id="start",
    )

    assert result.result.commit_sha == "a" * 40
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupt_reports_platform_failure_when_result_envelope_is_invalid():
    malformed = _marked(
        "AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END",
        {
            "status": "success", "summary": "完成", "commit_sha": "short",
            "tests": [], "unresolved_items": [],
        },
    )
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", last_event_id="start"),
            CocoTaskResult(
                status="failed", final_message=malformed,
                error="interrupted", last_event_id="end", lifecycle_terminal=True,
                platform_error="request queue size is exceed",
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner.resume_coding(
            session_id="session", run_id="run", event_id="start",
        )

    assert caught.value.platform is True
    assert caught.value.retryable is True
    assert caught.value.replacement_safe is True
    assert "平台侧异常" in str(caught.value)
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_succeeded_projection_reports_platform_failure_when_result_envelope_is_invalid():
    malformed = _marked(
        "AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END",
        {
            "status": "success", "summary": "完成", "commit_sha": "short",
            "tests": [], "unresolved_items": [],
        },
    )
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message=malformed, last_event_id="end",
            platform_error="request queue size is exceed",
        )),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
    )

    with pytest.raises(CocoRunFailed) as caught:
        await runner._wait("session", _started_event())

    assert caught.value.platform is True
    assert caught.value.retryable is True
    assert caught.value.replacement_safe is True
    assert "平台侧异常" in str(caught.value)


@pytest.mark.asyncio
async def test_succeeded_projection_keeps_valid_result_despite_platform_error():
    coding = _coding_envelope()
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message=coding, last_event_id="end",
            platform_error="request queue size is exceed",
        )),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
    )

    result = await runner._wait("session", _started_event())

    assert result.status == "succeeded"
    assert result.final_message == coding


@pytest.mark.asyncio
async def test_interrupt_preserves_valid_evaluation_result_from_succeeded_projection():
    evaluation = _marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
        "valid": True, "summary": "通过", "missing_items": [], "risks": [],
        "code_context": asdict(CodeContext()),
    })
    client = SimpleNamespace(
        create_session=AsyncMock(return_value={"Id": "session"}),
        send_user_message=AsyncMock(
            return_value=CocoEvent("event-1", "session", "run-1", {}, {})
        ),
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", last_event_id="start"),
            CocoTaskResult(
                status="succeeded", final_message=evaluation, last_event_id="end",
                lifecycle_terminal=True, platform_error="request queue size is exceed",
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    result = await runner.evaluate("需求正文")

    assert result.valid is True
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluation_and_coding_sessions_bind_the_credential_vaults():
    """评估与编码会话都要拿到凭据库；resolve/retry 不走 runner，故不受影响。"""
    evaluation = _marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
        "valid": True, "summary": "通过", "missing_items": [], "risks": [],
        "code_context": asdict(CodeContext()),
    })
    coding = _marked("AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END", {
        "status": "success", "summary": "完成", "commit_sha": "a" * 40,
        "tests": [], "unresolved_items": [],
    })
    client = SimpleNamespace(
        create_session=AsyncMock(return_value={"Id": "session"}),
        send_user_message=AsyncMock(side_effect=[
            CocoEvent("event-1", "session", "run-1", {}, {}),
            CocoEvent("event-2", "session", "run-2", {}, {}),
        ]),
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="succeeded", final_message=evaluation, last_event_id="end"),
            CocoTaskResult(status="succeeded", final_message=coding, last_event_id="end"),
        ]),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent",
        vault_ids=["vault-1"], poll_interval=0,
    )
    context = CocoSessionContext("task-1", "space", 42, "优化查询性能", "user-1", "tester")

    await runner.evaluate("需求正文", session_context=context)
    await runner.code(
        repo_id="repo-1", branch="feat/1",
        requirement_url="https://doc.example/1", evaluation_summary="通过",
        session_context=context,
    )

    assert [
        call.kwargs["vault_ids"] for call in client.create_session.await_args_list
    ] == [("vault-1",), ("vault-1",)]
    assert [call.kwargs["metadata"]["title"] for call in client.create_session.await_args_list] == [
        "tester | 评估 | 优化查询性能", "tester | 编码 | 优化查询性能",
    ]
    assert [call.kwargs["metadata"]["conciliator.stage"] for call in client.create_session.await_args_list] == [
        "evaluation", "coding",
    ]


def test_session_context_preserves_full_identity_when_truncating_title():
    name = "优化查询性能" * 100
    context = CocoSessionContext("a" * 64, "space", 42, name, "user-key", "tester\n name")
    metadata = context.metadata("evaluation")
    assert metadata["title"].startswith("tester name | 评估 | ")
    assert len(metadata["title"]) == 255
    assert metadata["title"].endswith("…")
    assert metadata["conciliator.work_item_name"] == name
    assert metadata["conciliator.task_id"] == "a" * 64
    assert metadata["conciliator.work_item_id"] == "42"
    assert metadata["conciliator.initiator_user_key"] == "user-key"


def test_legacy_session_context_uses_available_identity_and_item_id():
    context = CocoSessionContext("task", "space", 42, "", "user-key", "")
    assert context.metadata("coding")["title"] == "user-key | 编码 | 工作项 #42"


@pytest.mark.asyncio
async def test_sessions_omit_vault_ids_when_no_vault_is_configured():
    evaluation = _marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
        "valid": True, "summary": "通过", "missing_items": [], "risks": [],
        "code_context": asdict(CodeContext()),
    })
    client = SimpleNamespace(
        create_session=AsyncMock(return_value={"Id": "session"}),
        send_user_message=AsyncMock(
            return_value=CocoEvent("event-1", "session", "run-1", {}, {})
        ),
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message=evaluation, last_event_id="end",
        )),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0
    )

    await runner.evaluate("需求正文")

    client.create_session.assert_awaited_once()
    assert "vault_ids" not in client.create_session.await_args.kwargs


@pytest.mark.asyncio
async def test_evaluation_reuses_session_and_persists_each_run_before_polling():
    message = _marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
        "valid": True, "summary": "通过", "missing_items": [], "risks": [],
        "code_context": asdict(CodeContext()),
    })
    client = SimpleNamespace(
        create_session=AsyncMock(return_value={"Id": "evaluation-session"}),
        send_user_message=AsyncMock(side_effect=[
            CocoEvent("event-1", "evaluation-session", "run-1", {}, {}),
            CocoEvent("event-2", "evaluation-session", "run-2", {}, {}),
        ]),
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message=message, last_event_id="end",
        )),
    )
    saved = AsyncMock()

    async def fetch(*args, **kwargs):
        assert saved.call_args.args == ("evaluation-session", kwargs["run_id"], kwargs["after_event_id"])
        return CocoTaskResult(status="succeeded", final_message=message, last_event_id="end")

    client.fetch_session_result.side_effect = fetch
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="coding-agent")
    await runner.evaluate("旧文档", on_started=saved)
    initial_prompt = client.send_user_message.call_args.kwargs["text"]
    assert "这是新一轮输入评估" not in initial_prompt
    assert "替代此前文档" not in initial_prompt
    assert '"valid":true' in initial_prompt
    await runner.evaluate("最新文档", session_id="evaluation-session", on_started=saved, history=[{
        "round_no": 2, "reviewed_round_no": 1, "feedback": "补充验收标准",
        "result": {"summary": "不要重复发送旧评估输出"},
    }], session_context=CocoSessionContext("task", "space", 42, "更新名称", "other-user", "other"))
    client.create_session.assert_awaited_once()
    assert saved.await_count == 4
    assert "最新文档" in client.send_user_message.call_args.kwargs["text"]
    assert "替代此前文档" in client.send_user_message.call_args.kwargs["text"]
    assert "补充验收标准" in client.send_user_message.call_args.kwargs["text"]
    assert "不要重复发送旧评估输出" not in client.send_user_message.call_args.kwargs["text"]
    assert '"valid":true' not in client.send_user_message.call_args.kwargs["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_resume_evaluation_only_collects_existing_run_and_does_not_repeat_repair(invalid):
    message = _marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
        "valid": True, "summary": "通过", "missing_items": [], "risks": [],
        "code_context": asdict(CodeContext()),
    })
    client = SimpleNamespace(
        create_session=AsyncMock(), send_user_message=AsyncMock(),
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message="invalid" if invalid else message, last_event_id="end",
        )),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="coding-agent")
    if invalid:
        with pytest.raises(ValueError):
            await runner.resume_evaluation(
                session_id="session", run_id="repair-run", event_id="event", repair_attempted=True,
            )
    else:
        assert (await runner.resume_evaluation(session_id="session", run_id="run", event_id="event")).valid
    client.create_session.assert_not_awaited()
    client.send_user_message.assert_not_awaited()


def test_parse_structured_evaluation_result():
    parsed = parse_evaluation_result(
        _marked(
            "AEOLUS_EVALUATION_BEGIN",
            "AEOLUS_EVALUATION_END",
            {
                "valid": False,
                "summary": "缺少验收标准",
                "missing_items": ["验收标准"],
                "risks": ["范围可能扩大"],
                "code_context": asdict(CodeContext()),
            },
        )
    )
    assert parsed.valid is False
    assert parsed.missing_items == ("验收标准",)
    assert parsed.requirement_contract == ""
    assert parsed.code_context == CodeContext()


def test_evaluation_extracts_code_context_and_normalizes_repository_url():
    parsed = parse_evaluation_result(_marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
        "valid": True, "summary": "通过", "missing_items": [], "risks": [],
        "code_context": {"repository": "https://code.byted.org/dp/other.git", "base_branch": "develop",
                         "base_commit": "a" * 40, "target_branch": "release/v1"},
    }))
    assert parsed.code_context == CodeContext("dp/other", "develop", "a" * 40, "release/v1")


@pytest.mark.parametrize("context", [None, [], {}, {"repository": "https://example.com/dp/aeolus"},
                                   {"base_branch": "../master"}, {"base_commit": "HEAD"},
                                   {"target_branch": 42}])
def test_evaluation_rejects_invalid_code_context(context):
    if isinstance(context, dict) and context:
        context = {**asdict(CodeContext()), **context}
    with pytest.raises(ValueError):
        parse_evaluation_result(_marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
            "valid": True, "summary": "通过", "missing_items": [], "risks": [], "code_context": context,
        }))


@pytest.mark.parametrize("contract", ["# 需求共识\n\n## 目标\n更新 README", 42])
def test_evaluation_contract_must_be_text(contract):
    message = _marked("AEOLUS_EVALUATION_BEGIN", "AEOLUS_EVALUATION_END", {
        "valid": True, "summary": "通过", "missing_items": [], "risks": [],
        "requirement_contract": contract,
        "code_context": asdict(CodeContext()),
    })
    if isinstance(contract, str):
        assert parse_evaluation_result(message).requirement_contract == contract
    else:
        with pytest.raises(ValueError, match="requirement_contract must be a string"):
            parse_evaluation_result(message)


def test_parse_coding_result_requires_success_commit():
    parsed = parse_coding_result(
        _marked(
            "AEOLUS_CODING_RESULT_BEGIN",
            "AEOLUS_CODING_RESULT_END",
            {
                "status": "success",
                "summary": "完成实现",
                "commit_sha": "a" * 40,
                "tests": ["pytest tests/unit -q"],
                "unresolved_items": [],
            },
        )
    )
    assert parsed.commit_sha == "a" * 40
    with pytest.raises(ValueError, match="commit_sha"):
        parse_coding_result(
            _marked(
                "AEOLUS_CODING_RESULT_BEGIN",
                "AEOLUS_CODING_RESULT_END",
                {
                    "status": "success",
                    "summary": "bad",
                    "commit_sha": "",
                    "tests": [],
                    "unresolved_items": [],
                },
            )
        )


def test_prompts_lock_scope_and_machine_readable_contract():
    document_access = "DOC_PROXY_INSTRUCTIONS"
    evaluation = build_evaluation_prompt(
        "需求正文", document_access_instructions=document_access
    )
    coding = build_coding_prompt(
        requirement_url="需求正文",
        evaluation_summary="信息充分",
        branch="aeolus/ai-node/task-1",
        document_access_instructions=document_access,
    )
    feedback = build_feedback_prompt("请补测试", document_access)
    assert "目标是否清晰" in evaluation
    assert "AEOLUS_EVALUATION_BEGIN" in evaluation
    assert "aeolus/ai-node/task-1" in coding
    assert "不得修改、推送或强推其他分支" in coding
    assert "当前输入不含主文档正文" in coding
    assert "无需重新下载主文档" not in coding
    assert "commit" in coding and "push" in coding
    assert all("lark_authorization_required" in prompt for prompt in (coding, feedback))
    assert all("lark_authorization_required 属于 blocked" in prompt for prompt in (coding, feedback))
    assert all("status 为 failed、failure_code 为 lark_authorization_required" not in prompt
               for prompt in (coding, feedback))
    assert "请补测试" in feedback
    assert "AEOLUS_CODING_RESULT_BEGIN" in feedback
    assert "target_branch 使用本轮解析出的 base_branch" in evaluation
    assert all(document_access in prompt for prompt in (evaluation, coding, feedback))
    for prompt in (coding, feedback):
        assert "以用户已确认的需求共识和本轮反馈为准" in prompt
        assert "不得创建其他分支" in prompt and "不得修改、推送或强推其他分支" in prompt
        assert "| success |" in prompt and "| blocked |" in prompt and "| failed |" in prompt
        assert "不因首次失败就返回 blocked" in prompt
        assert "不在条件未变化时反复重试" in prompt
        assert "先完成其余可做的工作" in prompt
        assert "再返回 blocked 结束本轮；不得降低验收标准" in prompt
        assert prompt.count("AEOLUS_CODING_RESULT_BEGIN") == 1
        example = parse_coding_result(prompt)
        assert example.status == "blocked" and example.commit_sha == ""


@pytest.mark.parametrize("target", [None, "", "  ", "release/v1", "master"])
def test_mr_target_defaults_to_base_but_preserves_explicit_target(target):
    context = {"base_branch": "develop"}
    if target is not None:
        context["target_branch"] = target
    expected = target.strip() if target and target.strip() else "develop"
    assert CodeContext.from_dict(context).target_branch == expected
    assert CodeContext(base_branch="develop").target_branch == "develop"


def test_first_evaluation_has_no_retry_context():
    prompt = build_evaluation_prompt("首次文档", [{"round_no": 1, "feedback": "", "result": None}])
    assert "首次文档" in prompt and "历史评估" not in prompt
    assert "重新评估" not in prompt and "替代此前文档" not in prompt


def test_new_session_for_later_round_gets_complete_context():
    prompt = build_evaluation_prompt("最新文档", [
        {"round_no": 1, "result": {"requirement_contract": "已有需求共识"}},
        {"round_no": 2, "reviewed_round_no": 1, "feedback": "补充验收标准"},
    ])
    assert "新 Session 的上下文补齐" in prompt
    assert "已有需求共识" in prompt and "补充验收标准" in prompt
    assert '"valid":true' in prompt and "最新文档" in prompt
    assert "不能把示例" in prompt


def test_retry_without_new_feedback_does_not_mislabel_previous_feedback():
    prompt = build_evaluation_retry_prompt("更新后的文档", [
        {"round_no": 2, "reviewed_round_no": 1, "feedback": "之前的反馈", "result": {"summary": "旧输出"}},
        {"round_no": 3, "feedback": "", "result": None},
    ])
    assert "本轮用户反馈：\n无新增反馈" in prompt
    assert "之前的反馈" in prompt and "旧输出" not in prompt
    assert "base_commit 输出空字符串" in prompt
    assert "不写文件" in prompt and "完整需求共识，不只输出差异" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("repository, expected_agent", [
    ("dp/aeolus", "agent-1"),
    ("dp/VizQueryService", "803325104638816"),
    ("dp/another", "another-agent"),
    ("dp/unconfigured", "agent-1"),
])
async def test_runner_mounts_branch_and_returns_structured_coding_result(repository, expected_agent):
    message = _marked(
        "AEOLUS_CODING_RESULT_BEGIN",
        "AEOLUS_CODING_RESULT_END",
        {
            "status": "success",
            "summary": "完成实现",
            "commit_sha": "b" * 40,
            "tests": ["pytest -q"],
            "unresolved_items": [],
        },
    )
    client = SimpleNamespace(
        create_session=AsyncMock(return_value={"Id": "session-1"}),
        send_user_message=AsyncMock(
            return_value=CocoEvent("event-1", "session-1", "run-1", {}, {})
        ),
        fetch_session_result=AsyncMock(
            return_value=CocoTaskResult(
                status="succeeded", final_message=message, last_event_id="event-9"
            )
        ),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace-1", agent_id="agent-1", poll_interval=0,
        coding_agents={"dp/VizQueryService": "803325104638816", "dp/another": "another-agent"},
    )
    started = AsyncMock()
    result = await runner.code(
        repo_id="repo-1",
        repository=repository,
        branch="aeolus/ai-node/task-1",
        requirement_url="需求正文",
        evaluation_summary="信息充分",
        on_started=started,
    )
    client.create_session.assert_awaited_once_with(
        workspace_id="workspace-1",
        agent_id=expected_agent,
        repositories=[("repo-1", "aeolus/ai-node/task-1")],
    )
    assert result.session_id == "session-1"
    assert result.last_event_id == "event-9"
    assert result.result.commit_sha == "b" * 40
    started.assert_awaited_once_with("session-1", "run-1", "event-1")
    await runner.recode(session_id=result.session_id, feedback="补充测试")
    client.create_session.assert_awaited_once()
    assert client.send_user_message.call_args.kwargs["session_id"] == result.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_output, expected_error", [
    ("not json", "missing AEOLUS_EVALUATION_BEGIN/AEOLUS_EVALUATION_END markers"),
    ('AEOLUS_EVALUATION_BEGIN {"requirement_contract":"第一行\n第二行"} AEOLUS_EVALUATION_END',
     "invalid JSON: Invalid control character at (line 1, column 29)"),
    ('AEOLUS_EVALUATION_BEGIN {}\n# 需求共识 AEOLUS_EVALUATION_END',
     "invalid JSON: Extra data (line 2, column 1)"),
])
async def test_evaluation_requests_one_format_repair(invalid_output, expected_error):
    repaired = _marked(
        "AEOLUS_EVALUATION_BEGIN",
        "AEOLUS_EVALUATION_END",
        {
            "valid": True,
            "summary": "信息充分",
            "missing_items": [],
            "risks": [],
            "code_context": asdict(CodeContext()),
        },
    )
    client = SimpleNamespace(
        create_session=AsyncMock(return_value={"Id": "session-1"}),
        send_user_message=AsyncMock(
            side_effect=[
                CocoEvent("event-1", "session-1", "run-1", {}, {}),
                CocoEvent("event-2", "session-1", "run-2", {}, {}),
            ]
        ),
        fetch_session_result=AsyncMock(
            side_effect=[
                CocoTaskResult(
                    status="succeeded",
                    final_message=invalid_output,
                    last_event_id="event-9",
                ),
                CocoTaskResult(
                    status="succeeded",
                    final_message=repaired,
                    last_event_id="event-19",
                ),
            ]
        ),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace-1", agent_id="agent-1", poll_interval=0
    )

    result = await runner.evaluate("需求正文", history=[{
        "round_no": 2, "reviewed_round_no": 1, "feedback": "补充边界条件",
        "result": None,
    }])

    client.create_session.assert_awaited_once_with(
        workspace_id="workspace-1", agent_id="798403950855499", repositories=None
    )

    assert result.valid is True
    assert "补充边界条件" in client.send_user_message.await_args_list[0].kwargs["text"]
    assert client.send_user_message.await_count == 2
    assert "修复格式" in client.send_user_message.await_args_list[1].kwargs["text"]
    repair_prompt = client.send_user_message.await_args_list[1].kwargs["text"]
    assert expected_error in repair_prompt
    assert r'\n、\" 和 \\' in repair_prompt
    assert "不要添加 Markdown 代码围栏" in repair_prompt
    assert "不要在对象结束后另行输出" in repair_prompt


@pytest.mark.parametrize("status,commit", [("blocked", ""), ("blocked", "a" * 40), ("failed", "")])
def test_coding_results_accept_blocked_and_unknown_failure_codes(status, commit):
    result = parse_coding_result(_marked("AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END", {
        "status": status, "summary": "环境未准备", "commit_sha": commit,
        "tests": [], "unresolved_items": ["需要集成环境"], "failure_code": "integration_not_configured",
    }))
    assert result.status == status
    assert result.failure_code == "integration_not_configured"
    assert result.unresolved_items == ("需要集成环境",)


@pytest.mark.asyncio
async def test_platform_failure_is_retryable_and_skips_format_repair():
    platform_message = "We're sorry, the request queue size is exceed."
    client = SimpleNamespace(
        create_session=AsyncMock(),
        send_user_message=AsyncMock(),
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded",
            final_message="",
            error=platform_message,
            last_event_id="4",
            lifecycle_terminal=True,
            platform_error=platform_message,
        )),
    )
    runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent")

    with pytest.raises(CocoRunFailed) as excinfo:
        await runner.evaluate("需求正文", session_id="session-1")

    assert excinfo.value.retryable is True
    assert excinfo.value.platform is True
    assert "平台侧异常" in str(excinfo.value)
    assert platform_message in str(excinfo.value)
    # A saturated platform queue cannot be fixed by asking for a reformat, so
    # the repair message must not be sent.
    assert client.send_user_message.await_count == 1


@pytest.mark.parametrize("commit", ["short", "z" * 40])
def test_blocked_result_rejects_invalid_commit(commit):
    with pytest.raises(ValueError, match="commit_sha"):
        parse_coding_result(_marked("AEOLUS_CODING_RESULT_BEGIN", "AEOLUS_CODING_RESULT_END", {
            "status": "blocked", "summary": "部分交付", "commit_sha": commit,
            "tests": [], "unresolved_items": [], "failure_code": "",
        }))


def _coding_client(first: CocoTaskResult, second: CocoTaskResult) -> SimpleNamespace:
    return SimpleNamespace(
        create_session=AsyncMock(return_value={"Id": "session-1"}),
        send_user_message=AsyncMock(
            return_value=CocoEvent("event-1", "session-1", "run-1", {}, {})
        ),
        fetch_session_result=AsyncMock(side_effect=[first, second]),
        send_interrupt=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_fresh_stall_without_any_output_is_reported_as_platform_failure():
    """A run we started that emitted no Agent output is a platform stall.

    Nothing in the request can explain a runtime that never produced a single
    Agent event, and replacing it immediately lands in the same stall, so the
    caller must wait out the platform backoff instead.
    """
    client = _coding_client(
        CocoTaskResult(status="running", final_message="", last_event_id="event-2"),
        CocoTaskResult(
            status="running", final_message="", last_event_id="event-2",
            lifecycle_terminal=True,
        ),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    with pytest.raises(CocoRunFailed) as excinfo:
        await runner.code(
            repo_id="repo-1", repository="dp/repo", branch="aeolus/task-1",
            requirement_url="需求正文", evaluation_summary="信息充分",
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.platform is True
    assert "平台侧异常" in str(excinfo.value)
    assert "produced no Agent output" in str(excinfo.value)
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_stall_after_agent_output_is_not_blamed_on_the_platform():
    """Output changes stall attribution but does not disable its deadline."""
    progress = "正在按计划实现，稍后给出结构化结果。"
    client = _coding_client(
        CocoTaskResult(status="running", final_message=progress, last_event_id="event-2"),
        CocoTaskResult(
            status="running", final_message=progress, last_event_id="event-2",
            lifecycle_terminal=True,
        ),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    with pytest.raises(CocoRunFailed) as excinfo:
        await runner.code(
            repo_id="repo-1", repository="dp/repo", branch="aeolus/task-1",
            requirement_url="需求正文", evaluation_summary="信息充分",
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.platform is False
    assert "平台侧异常" not in str(excinfo.value)
    client.send_interrupt.assert_awaited_once()


@pytest.mark.asyncio
async def test_stall_after_tool_activity_is_not_blamed_on_the_platform():
    """Thinking or tool calls prove the runtime scheduled the model."""
    client = _coding_client(
        CocoTaskResult(
            status="running", final_message="", last_event_id="event-2",
            agent_activity=True,
        ),
        CocoTaskResult(
            status="running", final_message="", last_event_id="event-2",
            lifecycle_terminal=True, agent_activity=True,
        ),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    with pytest.raises(CocoRunFailed) as excinfo:
        await runner.code(
            repo_id="repo-1", repository="dp/repo", branch="aeolus/task-1",
            requirement_url="需求正文", evaluation_summary="信息充分",
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.platform is False


@pytest.mark.asyncio
async def test_resumed_stall_without_output_is_not_marked_platform():
    """A resumed run starts mid-stream, so absent output proves nothing."""
    client = SimpleNamespace(
        fetch_session_result=AsyncMock(side_effect=[
            CocoTaskResult(status="running", final_message="", last_event_id="event-2"),
            CocoTaskResult(
                status="running", final_message="", last_event_id="event-2",
                lifecycle_terminal=True,
            ),
        ]),
        send_interrupt=AsyncMock(),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0,
        stall_timeout_seconds=0, interrupt_timeout_seconds=0,
    )

    with pytest.raises(CocoRunFailed) as excinfo:
        await runner.resume_coding(session_id="session", run_id="run", event_id="start")

    assert excinfo.value.retryable is True
    assert excinfo.value.platform is False


@pytest.mark.asyncio
async def test_answer_retries_transient_reads_on_the_same_run():
    """A transient read failure resumes on the same cursor, not a new prompt."""
    client = SimpleNamespace(
        send_user_message=AsyncMock(return_value=CocoEvent(
            event_id="event-1", session_id="session", run_id="run-1",
            payload={}, raw={},
        )),
        fetch_session_result=AsyncMock(side_effect=[
            TransientError("read hiccup"),
            CocoTaskResult(
                status="succeeded", final_message="答案文本", last_event_id="end"
            ),
        ]),
    )
    runner = CocoAeolusRunner(
        client, workspace_id="workspace", agent_id="agent", poll_interval=0.01
    )

    answer = await runner.answer(session_id="session", question="为什么幂等？")

    assert answer == "答案文本"
    # The read was retried on the same run cursor: no second message was sent.
    assert client.send_user_message.await_count == 1
