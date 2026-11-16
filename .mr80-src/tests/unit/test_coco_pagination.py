import json
from contextlib import aclosing
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from conciliator.integrations.coco.client import CocoClient, CocoError, CocoEventPage
from tests.unit.test_coco_client import _ok, _resolve_result_message, _secrets, _settings
from tests.unit.test_coco_validation_repair import event


@pytest.mark.asyncio
@pytest.mark.parametrize("round_context", [{}, {"run_id": "r1", "after_event_id": "1"}])
@respx.mock
async def test_short_page_does_not_hide_final_idle(round_context):
    final = _resolve_result_message()
    # Match the observed shape: 199 events ending at 356, then idle at 357.
    events = [
        {"Id": str(i), "RunId": "r1", "Payload": {"AgentToolUse": {}}}
        for i in range(2, 200)
    ] + [{"Id": "356", "RunId": "r1", "Payload": {"AgentMessage": {"Content": final}}}]
    route = respx.post("https://coco.test/v2/?Action=ListManagedAgentSessionEvents").mock(side_effect=[
        _ok({"Events": events, "NextAfterEventId": "356"}),
        _ok({"Events": [{"Id": "357", "RunId": "r1", "Payload": {
            "SessionStatusIdle": {"StopReason": {"EndTurn": {}}},
        }}], "NextAfterEventId": "357"}),
        _ok({"Events": [], "NextAfterEventId": "357"}),
    ])
    respx.post("https://coco.test/v2/?Action=ListManagedAgentSessionRuns").mock(
        return_value=_ok({"Runs": [{"Id": "r1", "Status": "idle"}]}),
    )
    async with aclosing(CocoClient(_settings(), _secrets())) as client:
        result = await client.fetch_task_result("s1", **round_context)
    assert result.status == "succeeded"
    assert result.final_message == final
    assert result.last_event_id == "357"
    assert [json.loads(call.request.content).get("AfterEventId", "") for call in route.calls] == [
        round_context.get("after_event_id", ""), "356", "357",
    ]


@pytest.mark.asyncio
async def test_cursor_is_opaque_and_duplicates_are_folded_only_once():
    async with aclosing(CocoClient(_settings(), _secrets())) as client:
        client.list_session_event_page = AsyncMock(side_effect=[
            CocoEventPage([event("1"), event("5"), event("5")], "cursor-a"),
            CocoEventPage([], "cursor-b"),
            CocoEventPage([event("5"), event("8")], "cursor-c"),
            CocoEventPage([], ""),
        ])
        events = await client.read_round_events(workspace_id="w", session_id="s1", after_event_id="1")
        assert [e.event_id for e in events] == ["5", "8"]
        assert [call.kwargs["after_event_id"] for call in client.list_session_event_page.await_args_list] == [
            "1", "cursor-a", "cursor-b", "cursor-c",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pages", [
    [CocoEventPage([event("2")], "1")],
    [CocoEventPage([event("2")], "2"), CocoEventPage([event("3")], "1")],
    [CocoEventPage([], "2"), CocoEventPage([], "1")],
])
async def test_stalled_or_cycling_cursor_never_returns_partial_result(pages):
    async with aclosing(CocoClient(_settings(), _secrets())) as client:
        client.list_session_event_page = AsyncMock(side_effect=pages)
        with pytest.raises(CocoError, match="cursor stalled or cycled"):
            await client.read_round_events(workspace_id="w", session_id="s1", after_event_id="1")


@pytest.mark.asyncio
async def test_page_budget_never_returns_partial_result():
    async with aclosing(CocoClient(_settings(), _secrets())) as client:
        client.list_session_event_page = AsyncMock(side_effect=[
            CocoEventPage([event(str(i))], str(i)) for i in range(1, 101)
        ])
        with pytest.raises(CocoError, match="page limit reached"):
            await client.read_round_events(workspace_id="w", session_id="s1")
        assert client.list_session_event_page.await_count == 100


@pytest.mark.asyncio
async def test_later_page_failure_never_accepts_earlier_final_message():
    async with aclosing(CocoClient(_settings(), _secrets())) as client:
        client.list_session_event_page = AsyncMock(side_effect=[
            CocoEventPage([event("356", kind="AgentMessage", text=_resolve_result_message())], "356"),
            httpx.ReadTimeout("read failed"),
        ])
        with pytest.raises(httpx.ReadTimeout):
            await client.fetch_task_result("s1", run_id="r1", after_event_id="1")
