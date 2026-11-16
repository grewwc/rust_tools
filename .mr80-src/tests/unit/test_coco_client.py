import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.coco.client import (
    CocoClient,
    CocoError,
    CocoTaskRef,
    CocoTaskResult,
    parse_resolve_result,
    parse_retry_result,
)
from conciliator.utils.secrets import StaticSecretsProvider


def _settings() -> Settings:
    # Settings 仍会从环境变量与 .env 兜底，所以凡影响请求体的字段都在这里显式钉住，
    # 避免开发者本机的配置泄漏进下面的断言。
    return Settings(
        env="dev",
        service_psm="x",
        kms_secret_nextcode_app_secret_alias=("test/credentials/nextcode_app_secret"),
        nextcode_app_id="app-1",
        coco_api_base="https://coco.test/v2/",
        coco_workspace_id="workspace-1",
        coco_default_agent_id="agent-1",
        lark_app_id="x",
        mysql_psm="x",
    )  # type: ignore[arg-type]


def _secrets() -> StaticSecretsProvider:
    return StaticSecretsProvider({"test/credentials/nextcode_app_secret": "app-secret"})


def _ok(result: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "ResponseMetadata": {"RequestId": "request-1"},
            "Result": result,
        },
    )


def _resolve_result() -> dict:
    return {
        "schema_version": "conciliator.resolve.result.v3",
        "status": "succeeded",
        "commit_sha": "c" * 40,
        "conflicts": [
            {
                "conflict_id": "C0001",
                "path": "src/example.py",
                "decision_summary": "保留两侧有效逻辑",
                "source_attribution": {
                    "path": "src/example.py",
                    "start_line": 10,
                    "end_line": 12,
                    "commit_sha": "d" * 40,
                    "author_email": "source@example.com",
                },
                "target_attribution": {
                    "path": "src/example.py",
                    "start_line": 8,
                    "end_line": 11,
                    "commit_sha": "e" * 40,
                    "author_email": "target@example.com",
                },
                "resolved": {
                    "path": "src/example.py",
                    "start_line": 1,
                    "end_line": 1,
                    "text": "resolved",
                },
            }
        ],
        "unresolved_items": [],
    }


def _resolve_result_message(result: dict | None = None) -> str:
    result = result or _resolve_result()
    return f"CONCILIATOR_RESULT_BEGIN\n{json.dumps(result)}\nCONCILIATOR_RESULT_END"


def _retry_result() -> dict:
    result = _resolve_result()
    result["schema_version"] = "conciliator.resolve.retry.result.v2"
    return result


@pytest.mark.asyncio
@respx.mock
async def test_create_session_mounts_repo_by_id_and_revision():
    route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "CreateManagedAgentSession"},
    ).mock(return_value=_ok({"Session": {"Id": "session-1"}}))
    client = CocoClient(_settings(), _secrets())

    session = await client.create_session(
        workspace_id="workspace-1",
        agent_id="agent-1",
        repositories=[("repo-1", "resolve/feat/1")],
    )

    assert session["Id"] == "session-1"
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "WorkspaceId": "workspace-1",
        "AgentId": "agent-1",
        "Resources": [
            {
                "Repository": {
                    "RepoId": "repo-1",
                    "Revision": "resolve/feat/1",
                }
            }
        ],
    }
    assert route.calls[0].request.headers["X-Code-App-Id"] == "app-1"
    assert route.calls[0].request.headers["X-Code-App-Secret"] == "app-secret"
    await client.aclose()


async def _create_and_capture_body(settings: Settings, **kwargs) -> dict:
    route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "CreateManagedAgentSession"},
    ).mock(return_value=_ok({"Session": {"Id": "session-1"}}))
    client = CocoClient(settings, _secrets())
    await client.create_session(workspace_id="workspace-1", agent_id="agent-1", **kwargs)
    body = json.loads(route.calls[0].request.content)
    await client.aclose()
    return body


@pytest.mark.asyncio
@respx.mock
async def test_create_session_honours_an_explicit_vault_list():
    body = await _create_and_capture_body(_settings(), vault_ids=["vault-9"])
    assert body["VaultIds"] == ["vault-9"]


@pytest.mark.asyncio
@respx.mock
async def test_create_session_omits_vaults_unless_caller_asks_for_them():
    body = await _create_and_capture_body(_settings())
    assert "VaultIds" not in body


@pytest.mark.asyncio
@respx.mock
async def test_create_session_sends_title_and_business_metadata_with_app_identity():
    metadata = {
        "title": "tester | 编码 | 优化查询性能",
        "conciliator.task_id": "task-1",
        "conciliator.initiator_user_key": "user-1",
    }
    body = await _create_and_capture_body(
        _settings(), metadata=metadata, repositories=[("repo", "branch")], vault_ids=["vault"],
    )
    assert body["Metadata"] == metadata
    assert body["VaultIds"] == ["vault"]
    assert body["Resources"][0]["Repository"] == {"RepoId": "repo", "Revision": "branch"}
    assert "CreatedById" not in body


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("metadata", [None, {}])
async def test_create_session_omits_empty_metadata(metadata):
    body = await _create_and_capture_body(_settings(), metadata=metadata)
    assert "Metadata" not in body


@pytest.mark.asyncio
async def test_create_session_rejects_a_bare_string_vault():
    """A str would otherwise be iterated character by character into VaultIds."""
    client = CocoClient(_settings(), _secrets())
    with pytest.raises(TypeError):
        await client.create_session(
            workspace_id="workspace-1", agent_id="agent-1", vault_ids="vault-1"
        )
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_send_user_message_returns_real_event_and_run_ids():
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "SendManagedAgentSessionEvents"},
    ).mock(
        return_value=_ok(
            {
                "Events": [
                    {
                        "Id": "1",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {"UserMessage": {}},
                    }
                ]
            }
        )
    )
    client = CocoClient(_settings(), _secrets())

    event = await client.send_user_message(
        workspace_id="workspace-1",
        session_id="session-1",
        text="Reply READY",
    )

    assert event.event_id == "1"
    assert event.session_id == "session-1"
    assert event.run_id == "run-1"
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_submit_resolve_task_mounts_resolver_and_merges_target():
    session_route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "CreateManagedAgentSession"},
    ).mock(return_value=_ok({"Session": {"Id": "session-1"}}))
    message_route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "SendManagedAgentSessionEvents"},
    ).mock(
        return_value=_ok(
            {
                "Events": [
                    {
                        "Id": "event-1",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {"UserMessage": {}},
                    }
                ]
            }
        )
    )
    client = CocoClient(_settings(), _secrets())

    ref = await client.submit_resolve_task(
        "agent-1",
        repo_id="repo-1",
        task_id=123,
        original_mr_number=10,
        mr_version=1,
        resolve_branch="resolve/mr-10/task-123",
        source_branch="feat/source",
        source_commit="a" * 40,
        target_branch="main",
        target_commit="b" * 40,
        attempt=1,
        conflicts=[
            {
                "conflict_id": "C0001",
                "path": "src/example.py",
                "source_text": "source",
                "target_text": "target",
                "context_before": "",
                "context_after": "",
            }
        ],
        instructions="只修改冲突相关文件。",
    )

    assert ref.session_id == "session-1"
    assert ref.run_id == "run-1"
    assert ref.user_event_id == "event-1"
    assert json.loads(session_route.calls[0].request.content)["Resources"] == [
        {
            "Repository": {
                "RepoId": "repo-1",
                "Revision": "resolve/mr-10/task-123",
            }
        }
    ]
    prompt = json.loads(message_route.calls[0].request.content)["Events"][0]["Payload"]["UserMessage"]["Content"][0][
        "Text"
    ]["Text"]
    assert '"expected_resolve_head": "aaaaaaaa' in prompt
    assert '"conflict_id": "C0001"' in prompt
    assert "冻结 target_commit" in prompt
    assert "不修改 source/target 分支、MR、评论或飞书文档" in prompt
    assert '"schema_version": "conciliator.resolve.result.v3"' in prompt
    assert "committer date" in prompt and "author（不是 committer）" in prompt
    assert "author_email 为 null" in prompt
    assert "运行必要测试" not in prompt
    assert '"tests":' not in prompt and '"introduced_fix":' not in prompt
    assert '"merge_parents":' not in prompt
    assert '"locked_conflict_ids":' not in prompt



@pytest.mark.asyncio
async def test_submit_retry_task_locks_non_rejected_blocks(monkeypatch):
    client = CocoClient(_settings(), _secrets())
    monkeypatch.setattr(
        client,
        "create_session",
        AsyncMock(return_value={"Id": "retry-session"}),
    )
    send = AsyncMock(return_value=SimpleNamespace(event_id="event-2", run_id="run-2"))
    monkeypatch.setattr(client, "send_user_message", send)

    ref = await client.submit_retry_task(
        "agent-1",
        repo_id="repo-1",
        task_id=123,
        original_mr_number=10,
        mr_version=1,
        resolve_branch="resolve/mr-10/task-123",
        source_branch="feat/source",
        source_commit="a" * 40,
        target_branch="main",
        target_commit="b" * 40,
        base_commit_sha="f" * 40,
        attempt=2,
        rejected_conflicts=[
            {
                "conflict_id": "C0002",
                "path": "src/example.py",
                "source_text": "source",
                "target_text": "target",
                "previous_resolution": "old",
                "reject_reason": "保留空值处理",
            }
        ],
        locked_conflict_ids=["C0001", "C0003"],
        instructions="只改被拒块。",
    )

    assert ref == CocoTaskRef("retry-session", "run-2", "event-2")
    prompt = send.await_args.kwargs["text"]
    assert '"schema_version": "conciliator.resolve.retry.input.v2"' in prompt
    assert '"reject_reason": "保留空值处理"' in prompt
    assert '"locked_conflict_ids": [' in prompt and '"C0003"' in prompt
    assert '"parent_commits":' not in prompt
    assert "不得再次合入 target" in prompt
    await client.aclose()


def test_parse_retry_result_requires_one_parent_and_retry_schema():
    parsed = parse_retry_result(_resolve_result_message(_retry_result()))
    assert parsed.attempt == 0
    assert parsed.base_commit_sha == ""
    assert parsed.parent_commits == ()
    assert [item.conflict_id for item in parsed.conflicts] == ["C0001"]

    invalid = _retry_result()
    invalid["parent_commits"] = ["f" * 40, "b" * 40]
    with pytest.raises(ValueError, match="keys mismatch"):
        parse_retry_result(_resolve_result_message(invalid))


@pytest.mark.asyncio
@respx.mock
async def test_subscribe_session_events_parses_official_sse_event():
    route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "SubscribeManagedAgentSessionEvents"},
    ).mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                b'data: {"Event":{"Id":"2","SessionId":"session-1",'
                b'"RunId":"run-1","Payload":{"AgentMessage":'
                b'{"Content":[{"Text":{"Text":"OK"}}],"Delta":false}}}}\n\n'
            ),
        )
    )
    client = CocoClient(_settings(), _secrets())

    events = [
        event
        async for event in client.subscribe_session_events(
            workspace_id="workspace-1",
            session_id="session-1",
            after_event_id="1",
        )
    ]

    assert len(events) == 1
    assert events[0].event_id == "2"
    assert events[0].kind == "AgentMessage"
    assert events[0].run_id == "run-1"
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "WorkspaceId": "workspace-1",
        "SessionId": "session-1",
        "AfterEventId": "1",
    }
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_task_result_folds_delta_and_full_message():
    final_message = _resolve_result_message()
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionEvents"},
    ).mock(
        return_value=_ok(
            {
                "Events": [
                    {
                        "Id": "2",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {
                            "AgentThinking": {
                                "Content": [{"Text": {"Text": "analyzing"}}],
                                "Delta": True,
                            }
                        },
                    },
                    {
                        "Id": "3",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {
                            "AgentMessage": {
                                "Content": [{"Text": {"Text": "partial"}}],
                                "Delta": True,
                            }
                        },
                    },
                    {
                        "Id": "4",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {
                            "AgentMessage": {
                                "Content": [{"Text": {"Text": final_message}}],
                                "Delta": False,
                            }
                        },
                    },
                    {
                        "Id": "5",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {"SessionStatusIdle": {"StopReason": {"EndTurn": {}}}},
                    },
                ]
            }
        )
    )
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionRuns"},
    ).mock(return_value=_ok({"Runs": [{"Id": "run-1", "Status": "idle"}]}))
    client = CocoClient(_settings(), _secrets())

    result = await client.fetch_task_result("session-1")

    assert result.status == "succeeded"
    assert result.thinking == ""
    assert result.final_message == final_message
    assert result.resolve_result is not None
    assert result.resolve_result.task_id == 0
    assert result.resolve_result.conflicts[0].conflict_id == "C0001"
    assert result.resolve_result.conflicts[0].resolved.text == "resolved"
    assert result.commit_sha == "c" * 40
    assert result.last_event_id == "5"
    assert result.error is None
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_session_result_preserves_idle_lifecycle_without_agent_message():
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionEvents"},
    ).mock(return_value=_ok({"Events": [{
        "Id": "5",
        "SessionId": "session-1",
        "RunId": "run-1",
        "Payload": {"SessionStatusIdle": {"StopReason": {"EndTurn": {}}}},
    }]}))
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionRuns"},
    ).mock(return_value=_ok({"Runs": [{"Id": "run-1", "Status": "idle"}]}))
    client = CocoClient(_settings(), _secrets())

    result = await client.fetch_session_result(
        "session-1", workspace_id="workspace-1", run_id="run-1",
    )

    assert result.status == "running"
    assert result.final_message == ""
    assert result.lifecycle_terminal is True
    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_task_result_retries_until_final_message_is_visible(
    monkeypatch,
):
    client = CocoClient(_settings(), _secrets())
    incomplete = CocoTaskResult(
        status="succeeded",
        final_message="",
        last_event_id="5",
    )
    complete = CocoTaskResult(
        status="succeeded",
        final_message=_resolve_result_message(),
        last_event_id="6",
    )
    fetch = AsyncMock(side_effect=[incomplete, complete])
    sleep = AsyncMock()
    monkeypatch.setattr(client, "fetch_session_result", fetch)
    monkeypatch.setattr(
        "conciliator.integrations.coco.client.asyncio.sleep",
        sleep,
    )

    result = await client.fetch_task_result("session-1")

    assert result.status == "succeeded"
    assert result.resolve_result is not None
    assert result.resolve_result.task_id == 0
    assert result.last_event_id == "6"
    assert fetch.await_count == 2
    sleep.assert_awaited_once_with(30.0)
    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_task_result_stops_after_bounded_invalid_retries(
    monkeypatch,
):
    client = CocoClient(_settings(), _secrets())
    incomplete = CocoTaskResult(
        status="succeeded",
        final_message="",
        last_event_id="5",
    )
    fetch = AsyncMock(return_value=incomplete)
    sleep = AsyncMock()
    monkeypatch.setattr(client, "fetch_session_result", fetch)
    monkeypatch.setattr(
        "conciliator.integrations.coco.client.asyncio.sleep",
        sleep,
    )

    result = await client.fetch_task_result("session-1")

    assert result.status == "invalid_result"
    assert result.error == ("Coco result must contain exactly one result boundary")
    assert fetch.await_count == 4
    assert sleep.await_count == 3
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_task_result_surfaces_action_required():
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionEvents"},
    ).mock(
        return_value=_ok(
            {
                "Events": [
                    {
                        "Id": "9",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {"SessionStatusIdle": {"StopReason": {"RequiresAction": {"EventIds": ["8"]}}}},
                    }
                ]
            }
        )
    )
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionRuns"},
    ).mock(return_value=_ok({"Runs": [{"Id": "run-1", "Status": "idle"}]}))
    client = CocoClient(_settings(), _secrets())

    result = await client.fetch_task_result("session-1")

    assert result.status == "requires_action"
    assert result.error == "Agent is waiting for tool confirmation"
    assert result.requires_action_event_ids == ("8",)
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_send_tool_confirmation_answers_the_paused_tool_call():
    route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "SendManagedAgentSessionEvents"},
    ).mock(return_value=_ok({"Events": [{"Id": "10", "SessionId": "session-1"}]}))
    client = CocoClient(_settings(), _secrets())

    await client.send_tool_confirmation(
        workspace_id="workspace-1",
        session_id="session-1",
        tool_use_event_id="8",
        decision="deny",
        deny_message="use mktemp -d",
    )

    body = json.loads(route.calls[0].request.content)
    assert body["Events"][0]["Payload"]["UserToolConfirmation"] == {
        "ToolUseEventId": "8",
        "Decision": "deny",
        "DenyMessage": "use mktemp -d",
    }
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_session_result_keeps_basic_smoke_generic():
    events_route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionEvents"},
    ).mock(
        return_value=_ok(
            {
                "Events": [
                    {
                        "Id": "2",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {
                            "AgentMessage": {
                                "Content": [{"Text": {"Text": "COCO_BASIC_SMOKE_OK"}}],
                                "Delta": False,
                            }
                        },
                    },
                    {
                        "Id": "3",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {"SessionStatusIdle": {"StopReason": {"EndTurn": {}}}},
                    },
                ]
            }
        )
    )
    runs_route = respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionRuns"},
    ).mock(return_value=_ok({"Runs": [{"Id": "run-1", "Status": "idle"}]}))
    client = CocoClient(_settings(), _secrets())

    result = await client.fetch_session_result(
        "session-1",
        workspace_id="workspace-provisioned-at-runtime",
    )

    assert result.status == "succeeded"
    assert result.final_message == "COCO_BASIC_SMOKE_OK"
    assert result.resolve_result is None
    assert json.loads(events_route.calls[0].request.content)["WorkspaceId"] == ("workspace-provisioned-at-runtime")
    assert json.loads(runs_route.calls[0].request.content)["WorkspaceId"] == ("workspace-provisioned-at-runtime")
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_session_result_preserves_terminated_reason():
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionEvents"},
    ).mock(
        return_value=_ok(
            {
                "Events": [
                    {
                        "Id": "3",
                        "SessionId": "session-1",
                        "RunId": "run-1",
                        "Payload": {
                            "SessionStatusTerminated": {
                                "Reason": "repository checkout failed"
                            }
                        },
                    }
                ]
            }
        )
    )
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionRuns"},
    ).mock(return_value=_ok({"Runs": [{"Id": "run-1", "Status": "terminated"}]}))
    client = CocoClient(_settings(), _secrets())

    result = await client.fetch_session_result(
        "session-1", workspace_id="workspace-1", run_id="run-1"
    )

    assert result.status == "failed"
    assert result.error == "repository checkout failed"
    await client.aclose()


@pytest.mark.asyncio
async def test_fetch_session_result_requires_workspace_id():
    settings = _settings().model_copy(update={"coco_workspace_id": ""})
    client = CocoClient(settings, _secrets())

    with pytest.raises(CocoError, match="workspace_id or COCO_WORKSPACE_ID"):
        await client.fetch_session_result("session-1")

    await client.aclose()


def test_parse_resolve_result_rejects_text_outside_boundary():
    with pytest.raises(ValueError, match="outside"):
        parse_resolve_result("prefix\n" + _resolve_result_message())


@pytest.mark.parametrize("mutation", ["missing", "extra", "invalid_range"])
def test_parse_resolve_result_rejects_invalid_resolved_contract(mutation):
    result = _resolve_result()
    resolved = result["conflicts"][0]["resolved"]
    if mutation == "missing":
        resolved.pop("text")
    elif mutation == "extra":
        resolved["unexpected"] = True
    else:
        resolved["start_line"] = 2
        resolved["end_line"] = 1

    with pytest.raises(ValueError):
        parse_resolve_result(_resolve_result_message(result))


@pytest.mark.asyncio
@respx.mock
async def test_fetch_session_result_surfaces_platform_error():
    detail = json.dumps({
        "message": "We're sorry, the request queue size is exceed. Please wait and try again later.",
        "trae_error_info": "bad_request",
    })
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionEvents"},
    ).mock(return_value=_ok({"Events": [
        {
            "Id": "3",
            "SessionId": "session-1",
            "RunId": "run-1",
            "Payload": {"SessionError": {
                "Message": "Internal error",
                "Detail": detail,
                "Terminal": False,
                "RetryStatus": "exhausted",
            }},
        },
        {
            "Id": "4",
            "SessionId": "session-1",
            "RunId": "run-1",
            "Payload": {"SessionStatusIdle": {"StopReason": {"EndTurn": {}}}},
        },
    ]}))
    respx.post(
        "https://coco.test/v2/",
        params={"Action": "ListManagedAgentSessionRuns"},
    ).mock(return_value=_ok({"Runs": [{"Id": "run-1", "Status": "terminated"}]}))
    client = CocoClient(_settings(), _secrets())

    result = await client.fetch_session_result(
        "session-1", workspace_id="workspace-1", run_id="run-1",
    )

    expected = "We're sorry, the request queue size is exceed. Please wait and try again later."
    assert result.lifecycle_terminal is True
    assert result.platform_error == expected
    # The platform cause must win over the generic run-status wording.
    assert result.error == expected
    await client.aclose()
