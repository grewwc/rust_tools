import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from conciliator.integrations.meegle.ai_node import AINodeRef, MeegleAINodeClient


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(("data", "code", "expected"), [
    ([{"id": 42, "name": " 优化查询性能 "}], 0, "优化查询性能"),
    ([{"id": 43, "name": "其他任务"}], 0, None),
    ([{"id": 42, "name": ""}], 0, None),
    ([{"id": 42, "name": "a"}, {"id": 42, "name": "b"}], 0, None),
    ({}, 0, None),
    ([], 30005, None),
])
async def test_work_item_name_query_matches_requested_item_and_uses_plugin_identity(data, code, expected):
    respx.post("https://meego.example/open_api/authen/plugin_token").mock(
        return_value=httpx.Response(200, json={"data": {"token": "plugin-token"}}),
    )
    route = respx.post("https://meego.example/open_api/space/work_item/story/query").mock(
        return_value=httpx.Response(200, json={"data": data, "err_code": code}),
    )
    client = MeegleAINodeClient(site_url="https://meego.example", plugin_id="plugin", plugin_secret="secret")
    try:
        if expected is None:
            with pytest.raises(RuntimeError):
                await client.get_work_item_name(AINodeRef("space", 42, "node", "user"), "story")
        else:
            assert await client.get_work_item_name(AINodeRef("space", 42, "node", "user"), "story") == expected
    finally:
        await client.aclose()
    assert json.loads(route.calls[0].request.content) == {"work_item_ids": [42], "fields": ["name"]}
    assert route.calls[0].request.headers["x-user-key"] == "user"
    assert route.calls[0].request.headers["x-plugin-token"] == "plugin-token"


@pytest.mark.asyncio
@pytest.mark.parametrize(("users", "expected"), [
    ([{"user_key": "owner", "email": " owner@example.com "}], "owner@example.com"),
    ([{"user_key": "other", "email": "attacker@example.com"}], None),
    ([{"user_key": "owner", "email": ""}], None),
    ([{"user_key": "owner"}, {"user_key": "owner", "email": "other@example.com"}], None),
])
async def test_identity_email_requires_exact_unambiguous_directory_match(monkeypatch, users, expected):
    client = MeegleAINodeClient(
        site_url="https://meego.example", plugin_id="plugin", plugin_secret="secret",
    )
    query = AsyncMock(return_value=users)
    monkeypatch.setattr(client, "_query_users", query)
    try:
        assert await client.get_user_email("owner") == expected
        query.assert_awaited_once_with("owner", ["owner"])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_node_owners_and_user_emails_use_existing_plugin_identity():
    respx.post("https://meego.example/open_api/authen/plugin_token").mock(
        return_value=httpx.Response(200, json={"data": {"token": "plugin-token", "expire_time": 7200}})
    )
    node_route = respx.post("https://meego.example/open_api/ai_node/query").mock(
        return_value=httpx.Response(200, json={"data": {"node_fields": [
            {"field_key": "owner", "field_type": "multi-user", "value": '["owner-1","owner-1","owner-2"]'}
        ]}})
    )
    user_route = respx.post("https://meego.example/open_api/user/query").mock(
        return_value=httpx.Response(200, json={"err_code": 0, "data": [
            {"user_key": "owner-1", "email": "owner@example.com"}
        ]})
    )
    client = MeegleAINodeClient(site_url="https://meego.example", plugin_id="MII_test", plugin_secret="secret")
    ref = AINodeRef("space", 42, "uuid", "trigger-user")
    try:
        owners = await client.get_node_owner_keys(ref)
        assert owners == ["owner-1", "owner-2"]
        assert (await client.query_users(ref, owners))[0]["email"] == "owner@example.com"
        assert json.loads(user_route.calls[0].request.content) == {"user_keys": owners}
        assert user_route.calls[0].request.headers["x-user-key"] == "trigger-user"
        user_route.mock(return_value=httpx.Response(200, json={"err_code": 10023, "data": []}))
        with pytest.raises(RuntimeError, match="10023"):
            await client.query_users(ref, owners)
        node_route.mock(return_value=httpx.Response(200, json={"data": {"node_fields": [
            {"field_key": "owner", "value": '"not-a-list"'}
        ]}}))
        with pytest.raises(ValueError, match="must contain user keys"):
            await client.get_node_owner_keys(ref)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_edit_and_confirm_use_documented_token_headers_and_bodies():
    token_route = respx.post(
        "https://meego.example/open_api/authen/plugin_token"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {"token": "plugin-token", "expire_time": 7200},
                "error": {"code": 0, "msg": "success"},
            },
        )
    )
    edit_route = respx.post("https://meego.example/open_api/ai_node/edit").mock(
        return_value=httpx.Response(200, json={"data": {}, "error": {"code": 0}})
    )
    confirm_route = respx.post(
        "https://meego.example/open_api/ai_node/confirm"
    ).mock(
        return_value=httpx.Response(200, json={"data": {}, "error": {"code": 0}})
    )
    client = MeegleAINodeClient(
        site_url="https://meego.example",
        plugin_id="MII_test",
        plugin_secret="plugin-secret",
    )
    ref = AINodeRef("space-1", 42, "node-uuid-1", "user-1")
    try:
        await client.edit_status(ref, "done", idem_key="task-1:done")
        await client.confirm(ref, idem_key="task-1:confirm")
    finally:
        await client.aclose()

    assert token_route.call_count == 1
    assert json.loads(token_route.calls[0].request.content) == {
        "plugin_id": "MII_test",
        "plugin_secret": "plugin-secret",
        "type": 0,
    }
    assert json.loads(edit_route.calls[0].request.content) == {
        "project_key": "space-1",
        "work_item_id": 42,
        "node_uuid": "node-uuid-1",
        "status": "done",
    }
    for call, idem_key in (
        (edit_route.calls[0], "task-1:done"),
        (confirm_route.calls[0], "task-1:confirm"),
    ):
        assert call.request.headers["x-plugin-token"] == "plugin-token"
        assert call.request.headers["x-user-key"] == "user-1"
        assert call.request.headers["x-idem-uuid"] == idem_key
    assert json.loads(confirm_route.calls[0].request.content) == {
        "project_key": "space-1",
        "work_item_id": 42,
        "node_uuid": "node-uuid-1",
    }


@pytest.mark.asyncio
@respx.mock
async def test_api_error_is_not_treated_as_success():
    respx.post("https://meego.example/open_api/authen/plugin_token").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {"token": "plugin-token", "expire_time": 7200},
                "error": {"code": 0},
            },
        )
    )
    respx.post("https://meego.example/open_api/ai_node/edit").mock(
        return_value=httpx.Response(
            200, json={"data": None, "error": {"code": 123, "msg": "bad"}}
        )
    )
    client = MeegleAINodeClient(
        site_url="https://meego.example",
        plugin_id="MII_test",
        plugin_secret="plugin-secret",
    )
    try:
        with pytest.raises(RuntimeError, match="123"):
            await client.edit_status(
                AINodeRef("space-1", 42, "node-1", "user-1"),
                "done",
                idem_key="idem",
            )
    finally:
        await client.aclose()
