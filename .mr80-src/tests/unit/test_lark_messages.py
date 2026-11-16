import json

import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.messages import (
    send_all_approved_message,
    send_awaiting_approval_message,
    send_task_received_message,
)


def _settings() -> Settings:
    return Settings(
        env="dev",
        service_psm="x",
        lark_app_id="li",
        mysql_psm="x",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
@respx.mock
async def test_send_all_approved_message_uses_clickable_post_and_uuid():
    route = respx.post(
        "https://open.feishu.cn/open-apis/im/v1/messages"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"message_id": "om_123"},
            },
        )
    )
    client = LarkClient(_settings(), None, tenant_access_token="tok")
    mr_url = "https://code.byted.org/org/repo/merge_requests/15"

    message_id = await send_all_approved_message(
        client,
        open_id="ou_user",
        mr_url=mr_url,
        request_uuid="conciliator-all-approved-7",
    )

    assert message_id == "om_123"
    request = route.calls.last.request
    assert request.url.params["receive_id_type"] == "open_id"
    body = json.loads(request.content)
    assert body["receive_id"] == "ou_user"
    assert body["msg_type"] == "post"
    assert body["uuid"] == "conciliator-all-approved-7"
    content = json.loads(body["content"])
    nodes = content["zh_cn"]["content"][0]
    assert nodes[0]["text"] == (
        "本任务冻结版本中需要人工审批的冲突块已全部 approve，请结合 MR 最新检查结果确认是否合入。MR链接："
    )
    assert nodes[1] == {"tag": "a", "text": mr_url, "href": mr_url}
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", [1, 2])
@respx.mock
async def test_send_awaiting_message_links_resolve_mr_and_identifies_round(attempt):
    route = respx.post("https://open.feishu.cn/open-apis/im/v1/messages").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"message_id": "om_awaiting"}}),
    )
    client = LarkClient(_settings(), None, tenant_access_token="tok")
    mr_url = "https://code.test/mr/99"
    try:
        assert await send_awaiting_approval_message(
            client, open_id="ou_trigger", mr_url=mr_url,
            request_uuid="awaiting-test", task_id=7, attempt=attempt,
        ) == "om_awaiting"
        request = route.calls.last.request
        body = json.loads(request.content)
        assert request.url.params["receive_id_type"] == "open_id"
        assert body["receive_id"] == "ou_trigger"
        assert body["uuid"] == "awaiting-test"
        assert body["msg_type"] == "post"
        nodes = json.loads(body["content"])["zh_cn"]["content"][0]
        assert f"任务 7 第 {attempt} 轮" in nodes[0]["text"]
        assert "等待审批" in nodes[0]["text"]
        assert "已 approve" not in nodes[0]["text"]
        assert nodes[1] == {"tag": "a", "text": "查看冲突并审批", "href": mr_url}
        assert json.loads(body["content"])["zh_cn"]["content"][1] == [{
            "tag": "a", "text": "查看处理进度",
            "href": "https://aeolus-conciliator.byted.org/trace?task_id=7&order=desc",
        }]
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("original_mr_url", ["", "https://code.test/mr/54"])
@respx.mock
async def test_received_message_links_task_trace_before_a_resolver_mr_exists(original_mr_url):
    route = respx.post("https://open.feishu.cn/open-apis/im/v1/messages").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"message_id": "om_received"}}),
    )
    client = LarkClient(_settings(), None, tenant_access_token="tok")
    try:
        assert await send_task_received_message(
            client, open_id="ou_trigger", mr_url=original_mr_url, request_uuid="receipt-49", task_id=49,
        ) == "om_received"
        body = json.loads(route.calls.last.request.content)
        assert body["uuid"] == "receipt-49" and body["receive_id"] == "ou_trigger"
        rows = json.loads(body["content"])["zh_cn"]["content"]
        assert "任务 49 已受理" in rows[0][0]["text"]
        assert "处理已完成" not in rows[0][0]["text"]
        assert rows[0][1] == {
            "tag": "a", "text": "查看处理进度",
            "href": "https://aeolus-conciliator.byted.org/trace?task_id=49&order=desc",
        }
        assert rows[1:] == ([[{"tag": "a", "text": "查看原始 MR", "href": original_mr_url}]] if original_mr_url else [])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_no_manual_approval_message_does_not_claim_human_approval():
    from conciliator.integrations.lark.messages import send_no_manual_approval_message

    route = respx.post("https://open.feishu.cn/open-apis/im/v1/messages").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"message_id": "om_blank"}}),
    )
    client = LarkClient(_settings(), None, tenant_access_token="tok")
    try:
        assert await send_no_manual_approval_message(
            client, open_id="ou_trigger", mr_url="https://code.test/mr/99", request_uuid="blank-test",
        ) == "om_blank"
        body = json.loads(route.calls.last.request.content)
        nodes = json.loads(body["content"])["zh_cn"]["content"][0]
        assert "纯空行冲突已处理，无需人工审批" in nodes[0]["text"]
        assert "已 approve" not in nodes[0]["text"]
        assert nodes[1]["href"] == "https://code.test/mr/99"
    finally:
        await client.aclose()
