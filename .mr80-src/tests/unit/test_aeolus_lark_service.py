import base64
import hashlib
import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from conciliator.api.aeolus import router
from conciliator.domain.aeolus_task import AeolusRound, AeolusState, AeolusTask, DeploymentVerification
from conciliator.services.aeolus_lark_service import AeolusLarkService, InvalidCardCallback
from conciliator.services.deployment_counts import devbox_result_kind


def _walk(value):
    if isinstance(value, dict):
        if "tag" in value:
            yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _card_text(elements):
    return "\n".join(e["content"] for e in _walk(elements) if e["tag"] == "markdown")


def _setup(state=AeolusState.EVALUATION_REVIEW):
    task = AeolusTask(
        "task",
        "space",
        42,
        "node",
        "uuid",
        "user",
        (),
        "private-token",
        state=state,
        evaluation={"valid": True, "summary": "通过"},
        evaluation_rounds=[{"round_no": 1}],
        rounds=[AeolusRound(1, "review")],
    )
    client = AsyncMock()
    client._request.return_value = {"user_list": [{"email": "owner@example.com", "user_id": "ou_user"}]}
    meegle = AsyncMock()
    meegle.get_node_owner_keys.return_value = ["owner"]
    meegle.query_users.return_value = [{"user_key": "owner", "email": "owner@example.com"}]
    cards = AeolusLarkService(client, app_id="app", meegle=meegle, verification_token="verification")
    service = AsyncMock()
    service.get.return_value = task
    return cards, service, task, client


def _payload(cards, task, action, feedback="修改异常处理"):
    return {
        "header": {
            "token": "verification",
            "event_id": "event-1",
            "app_id": "app",
            "event_type": "card.action.trigger",
        },
        "event": {
            "operator": {"open_id": "ou_user"},
            "action": {"value": cards._value(task, action, "owner", "ou_user"), "form_value": {"feedback": feedback}},
        },
    }


@pytest.mark.parametrize("stage,state,label,note", [
    ("maat_cleanup_records_deleted", "succeeded", "环境已释放",
     "服务和实例代码、MAAT 环境及代理记录已清理；旧端口保留预留，可重新部署或继续 Coding。"),
    ("cleanup_pending", "pending", "本机已释放，MAAT 清理中",
     "服务和实例代码已清理；正在自动清理 MAAT 环境和代理记录，旧端口保留预留。"),
])
def test_released_devbox_card_displays_record_cleanup_and_retained_port(stage, state, label, note):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.deployment = {
        "deployment_id": "devbox", "worker_id": "machine", "status": "cancelled",
        "stage": stage, "cleanup_state": state, "released": True, "can_release": False, "generation": 1,
        "commit_sha": "a" * 40, "status_label": label, "status_note": note,
    }
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert label in content and note in content and "旧端口保留预留" in content
    assert "MAAT 待清理" not in content and "释放 Devbox" not in content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,action",
    [
        (AeolusState.EVALUATION_REVIEW, "confirm_evaluation"),
        (AeolusState.EVALUATION_REVIEW, "evaluation_feedback"),
        (AeolusState.VALIDATION_FAILED, "retry_validation"),
        (AeolusState.REVIEW, "confirm"),
        (AeolusState.REVIEW, "submit_feedback"),
        (AeolusState.FAILED, "retry_coding"),
    ],
)
async def test_callback_reuses_task_operations(state, action):
    cards, service, task, _ = _setup(state)
    getattr(service, action).return_value = task
    payload = _payload(cards, task, action)
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    assert result["toast"]["type"] == "success"
    args = ("task",)
    if action in {"evaluation_feedback", "submit_feedback"}:
        args += ("修改异常处理",)
    if action != "retry_validation":
        args += ("event-1",)
    getattr(service, action).assert_awaited_once_with(*args)


@pytest.mark.asyncio
async def test_failed_coding_card_retry_entry_is_not_available_after_platform_completion():
    cards, service, task, _ = _setup(AeolusState.FAILED)
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert "重试编码" in content
    task.platform_status_done = True
    assert "重试编码" not in json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    with pytest.raises(InvalidCardCallback, match="not available"):
        await cards.callback(service, json.dumps(_payload(cards, task, "retry_coding")).encode(), {})
    service.retry_coding.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluation_confirmation_preserves_consensus_in_updated_message():
    cards, service, task, _ = _setup()
    contract = "# 需求共识\n## 本次范围\n- README.md"
    task.evaluation.update(requirement_contract=contract, risks=["测试依赖内部环境"])
    payload = _payload(cards, task, "confirm_evaluation")

    async def confirm(*args):
        task.state = AeolusState.CODING
        return task

    service.confirm_evaluation.side_effect = confirm
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    elements = result["card"]["data"]["body"]["elements"]
    content = _card_text(elements)
    assert "正在编码" in content and "通过" in content
    assert contract in content and "测试依赖内部环境" in content
    assert not any(element["tag"] in {"form", "button"} for element in elements)
    # New coding notifications do not repeat the full evaluation snapshot.
    assert contract not in json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("token", "wrong"), ("app_id", "wrong")])
async def test_callback_rejects_other_apps_or_tokens(field, value):
    cards, service, task, _ = _setup()
    payload = _payload(cards, task, "confirm_evaluation")
    payload["header"][field] = value
    with pytest.raises(InvalidCardCallback):
        await cards.callback(service, json.dumps(payload).encode(), {})
    service.confirm_evaluation.assert_not_awaited()


@pytest.mark.asyncio
async def test_wrong_operator_and_stale_or_tampered_cards_cannot_advance_task():
    cards, service, task, _ = _setup()
    payload = _payload(cards, task, "confirm_evaluation")
    payload["event"]["operator"]["open_id"] = "ou_other"
    with pytest.raises(InvalidCardCallback):
        await cards.callback(service, json.dumps(payload).encode(), {})
    payload["event"]["operator"]["open_id"] = "ou_user"
    payload["event"]["action"]["value"]["signature"] = "tampered"
    assert (await cards.callback(service, json.dumps(payload).encode(), {}))["toast"]["type"] == "info"
    payload = _payload(cards, task, "confirm_evaluation")
    task.evaluation_rounds.append({"round_no": 2})
    assert (await cards.callback(service, json.dumps(payload).encode(), {}))["toast"]["type"] == "info"
    service.confirm_evaluation.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_callback_does_not_advance_new_state():
    cards, service, task, _ = _setup()
    payload = json.dumps(_payload(cards, task, "confirm_evaluation")).encode()

    async def confirm(*args):
        task.state = AeolusState.CODING
        return task

    service.confirm_evaluation.side_effect = confirm
    assert (await cards.callback(service, payload, {}))["toast"]["type"] == "success"
    assert (await cards.callback(service, payload, {}))["toast"]["type"] == "info"
    service.confirm_evaluation.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_feedback_does_not_start_another_round():
    cards, service, task, _ = _setup()
    payload = _payload(cards, task, "evaluation_feedback", feedback="   ")
    assert (await cards.callback(service, json.dumps(payload).encode(), {}))["toast"]["type"] == "error"
    service.evaluation_feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_resolves_node_owner_and_does_not_expose_task_token():
    cards, _, task, client = _setup()
    await cards.notify(task)
    body = client._request.call_args.kwargs["json"]
    assert body["receive_id"] == "ou_user"
    assert body["msg_type"] == "interactive"
    assert "private-token" not in body["content"]
    assert "form_action_type" in body["content"]
    first_uuid = body["uuid"]
    await cards.notify(task)
    assert client._request.call_args.kwargs["json"]["uuid"] == first_uuid
    client.reset_mock()
    cards.meegle.get_node_owner_keys.return_value = []
    await cards.notify(task)
    client._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_removed_owner_cannot_use_previously_sent_card():
    cards, service, task, _ = _setup()
    payload = _payload(cards, task, "confirm_evaluation")
    cards.meegle.get_node_owner_keys.return_value = ["new-owner"]
    with pytest.raises(InvalidCardCallback, match="no longer a node owner"):
        await cards.callback(service, json.dumps(payload).encode(), {})
    service.confirm_evaluation.assert_not_awaited()


@pytest.mark.asyncio
async def test_owner_lookup_failure_does_not_advance_task():
    cards, service, task, _ = _setup()
    cards.meegle.get_node_owner_keys.side_effect = RuntimeError("platform unavailable")
    result = await cards.callback(service, json.dumps(_payload(cards, task, "confirm_evaluation")).encode(), {})
    assert result["toast"]["type"] == "error"
    service.confirm_evaluation.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_owners_get_individually_bound_cards_and_one_failure_is_isolated():
    cards, _, task, client = _setup()
    cards.meegle.get_node_owner_keys.return_value = ["owner", "owner-2", "owner-3"]
    cards.meegle.query_users.return_value = [
        {"user_key": "owner", "email": "owner@example.com"},
        {"user_key": "owner-2", "email": "owner-2@example.com"},
        {"user_key": "owner-3", "email": "owner-3@example.com"},
    ]
    sent = []

    async def request(method, path, *, params, json):
        if path == "/contact/v3/users/batch_get_id":
            email = json["emails"][0]
            return {"user_list": [{"email": email, "user_id": "ou_" + email.split("@")[0]}]}
        if json["receive_id"] == "ou_owner-2":
            raise RuntimeError("send failed")
        sent.append(json)
        return {"message_id": "message"}

    client._request.side_effect = request
    await cards.notify(task)
    assert {body["receive_id"] for body in sent} == {"ou_owner", "ou_owner-3"}
    assert len({body["uuid"] for body in sent}) == 2
    for body in sent:
        value = next(
            e["behaviors"][0]["value"] for e in _walk(json.loads(body["content"]))
            if e["tag"] == "button" and e["behaviors"][0].get("value", {}).get("action") == "confirm_evaluation"
        )
        assert value["recipient"] == body["receive_id"]
        assert value["owner_key"] != task.user_key


@pytest.mark.parametrize("project_key,expected", [
    ("6aabe23024218882059f74a4", "[工作项 7381320257](https://meego.larkoffice.com/4ebse5/story/detail/7381320257)"),
    ("other-space", "工作项 7381320257"),
])
def test_card_work_item_link_only_uses_confirmed_space(project_key, expected):
    cards, _, task, _ = _setup()
    task.project_key = project_key
    task.work_item_id = 7381320257
    content = cards.card(task, "owner", "ou_user")["body"]["elements"][0]["content"]
    assert content.startswith(expected + " · ")


def test_card_uses_markdown_and_explicit_callback_buttons():
    cards, _, task, _ = _setup()
    contract = "# 需求共识\n\n- 验收标准\n\n```python\nassert True\n```"
    task.evaluation["requirement_contract"] = contract
    card = cards.card(task, "owner", "ou_user")
    assert card["schema"] == "2.0"
    elements = card["body"]["elements"]
    assert {"tag": "markdown", "content": contract} in list(_walk(elements))
    confirm = next(element for element in elements if element["tag"] == "button")
    assert confirm["behaviors"] == [
        {"type": "callback", "value": cards._value(task, "confirm_evaluation", "owner", "ou_user")}
    ]
    form = next(element for element in elements if element["tag"] == "form")
    assert form["elements"][0]["max_length"] == 1000
    assert form["elements"][0]["required"] is True
    assert form["elements"][1]["form_action_type"] == "submit"
    assert form["elements"][1]["behaviors"][0]["value"]["action"] == "evaluation_feedback"


@pytest.mark.parametrize("error_source", ["task", "round", "missing"])
def test_evaluation_execution_failure_shows_reason_and_retry(error_source):
    cards, _, task, _ = _setup(AeolusState.VALIDATION_FAILED)
    task.evaluation = None
    error = "ValueError: invalid output; token=private-value"
    task.evaluation_rounds = [
        {"round_no": 1, "result": {"requirement_contract": "已有需求共识"}},
        {"round_no": 2, "status": "error", "feedback": "补充验收标准"},
    ]
    if error_source == "task":
        task.last_error = error
    elif error_source == "round":
        task.evaluation_rounds[-1]["error"] = error
    elements = cards.card(task, "owner", "ou_user")["body"]["elements"]
    content = _card_text(elements)
    assert "评估执行失败" in content and "不代表需求不合理" in content
    assert "请检查需求文档" not in content
    assert "补充验收标准" in content and "非本轮结论" in content and "已有需求共识" in content
    if error_source == "missing":
        assert "未记录具体异常" in content
    else:
        assert "invalid output" in content and "[REDACTED]" in content
        assert "private-value" not in content
    retry = next(e for e in elements if e["tag"] == "button")
    assert retry["text"]["content"] == "重试评估"
    assert retry["behaviors"][0]["value"]["action"] == "retry_validation"
    assert not any(
        e["tag"] == "button" and e["behaviors"][0]["value"]["action"] == "confirm_evaluation"
        for e in elements
    )


def test_platform_failure_card_reports_platform_cause_without_error_kind():
    cards, _, task, _ = _setup(AeolusState.VALIDATION_FAILED)
    task.evaluation = None
    # ``_mark_failed`` persists "<ExceptionType>: <message>", so rounds written
    # before ``error_kind`` existed only carry that prefixed string.
    task.evaluation_rounds = [
        {
            "round_no": 1,
            "status": "error",
            "error": "CocoRunFailed: 平台侧异常：request queue size is exceed",
        },
    ]
    elements = cards.card(task, "owner", "ou_user")["body"]["elements"]
    content = _card_text(elements)
    # The card covers both platform shapes: a saturated queue and a run that
    # never produced any output.
    assert "Coco 平台侧导致" in content
    assert "资源受限，或运行未产生任何输出" in content
    assert "请根据失败原因检查文档访问权限" not in content
    # A recovered task never issued a replacement, so the card must not claim one.
    assert "本次没有执行自动重试" in content
    assert "已按退避策略自动重试一次" not in content


def test_platform_failure_card_reports_the_retry_only_when_it_happened():
    cards, _, task, _ = _setup(AeolusState.VALIDATION_FAILED)
    task.evaluation = None
    task.evaluation_rounds = [
        {
            "round_no": 1,
            "status": "error",
            "error": "CocoRunFailed: 平台侧异常：request queue size is exceed",
            "error_kind": "platform",
            "platform_retry_attempted": True,
        },
    ]
    elements = cards.card(task, "owner", "ou_user")["body"]["elements"]
    content = _card_text(elements)
    assert "已按退避策略自动重试一次" in content
    assert "本次没有执行自动重试" not in content


def test_blocked_evaluation_shows_questions_risks_and_next_step():
    cards, _, task, _ = _setup(AeolusState.VALIDATION_FAILED)
    task.evaluation = {
        "valid": False, "summary": "缺少验收标准", "missing_items": ["怎样验证成功？"],
        "risks": ["存在兼容性风险"], "requirement_contract": "当前需求共识",
    }
    task.evaluation_rounds[-1].update(status="review", feedback="限定修改范围")
    elements = cards.card(task, "owner", "ou_user")["body"]["elements"]
    content = _card_text(elements)
    for expected in (
        "输入评估未通过", "缺少验收标准", "待补充 / 阻断问题", "怎样验证成功？",
        "风险提示", "存在兼容性风险", "当前需求共识", "限定修改范围", "请补充上述阻断问题",
    ):
        assert expected in content
    assert "评估执行失败" not in content
    retry = next(e for e in elements if e["tag"] == "button")
    assert retry["text"]["content"] == "文档已更新，重新评估"


@pytest.mark.parametrize("state", [
    AeolusState.RECODING, AeolusState.REVIEW, AeolusState.COMPLETED, AeolusState.FAILED,
])
def test_coding_card_shows_feedback_and_processing_results(state):
    cards, _, task, _ = _setup(state)
    task.rounds = [AeolusRound(1, "review", summary="旧结果"), AeolusRound(
        2, "review", feedback="补充异常处理", reviewed_round_no=1,
        summary="已补充错误提示", commit_sha="abc123", tests=["pytest: 3 passed"],
        error="编码会话异常" if state == AeolusState.FAILED else "",
    )]
    elements = cards.card(task, "owner", "ou_user")["body"]["elements"]
    content = _card_text(elements)
    assert "第 2 轮编码" in content
    assert "用户反馈（针对第 1 轮）" in content and "补充异常处理" in content
    assert "处理结果" in content and "已补充错误提示" in content
    assert "abc123" in content and "pytest: 3 passed" in content
    assert "非 CI 状态" in content and "旧结果" not in content
    assert ("编码会话异常" in content) == (state == AeolusState.FAILED)
    assert "private-token" not in json.dumps(elements)


def test_coding_card_shows_pending_feedback_without_empty_results():
    cards, _, task, _ = _setup(AeolusState.RECODING)
    task.rounds = [AeolusRound(2, "recoding", feedback="长" * 3000, reviewed_round_no=1)]
    elements = cards.card(task, "owner", "ou_user")["body"]["elements"]
    content = _card_text(elements)
    assert "正在根据反馈修改" in content and "针对第 1 轮" in content
    assert content.count("长") == 2000
    assert "处理结果" not in content and "测试记录" not in content
    assert not any(element["tag"] in {"button", "form"} for element in elements)


@pytest.mark.asyncio
async def test_http_callback_challenge_requires_token_and_feature_enablement():
    cards, service, _, _ = _setup()
    app = FastAPI()
    app.include_router(router)
    app.state.aeolus_service = service
    app.state.aeolus_lark_service = cards
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        challenge = {"type": "url_verification", "challenge": "challenge", "token": "verification"}
        response = await client.post("/hooks/lark/aeolus-card", json=challenge)
        assert response.json() == {"challenge": "challenge"}
        challenge["token"] = "wrong"
        assert (await client.post("/hooks/lark/aeolus-card", json=challenge)).status_code == 401
        app.state.aeolus_lark_service = None
        assert (await client.post("/hooks/lark/aeolus-card", json=challenge)).status_code == 503


@pytest.mark.asyncio
async def test_encrypted_callback_verifies_signature_before_dispatch():
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad

    cards, service, task, _ = _setup()
    cards.encrypt_key = "test-encrypt-key"
    service.confirm_evaluation.return_value = task
    plaintext = json.dumps(_payload(cards, task, "confirm_evaluation")).encode()
    iv = b"0" * 16
    encrypted = AES.new(hashlib.sha256(cards.encrypt_key.encode()).digest(), AES.MODE_CBC, iv).encrypt(
        pad(plaintext, 16)
    )
    raw = json.dumps({"encrypt": base64.b64encode(iv + encrypted).decode()}).encode()
    timestamp = str(int(time.time()))
    headers = {"x-lark-request-timestamp": timestamp, "x-lark-request-nonce": "nonce"}
    headers["x-lark-signature"] = hashlib.sha256((timestamp + "nonce" + cards.encrypt_key).encode() + raw).hexdigest()
    assert (await cards.callback(service, raw, headers))["toast"]["type"] == "success"
    headers["x-lark-signature"] = "wrong"
    with pytest.raises(InvalidCardCallback):
        await cards.callback(service, raw, headers)
    service.confirm_evaluation.assert_awaited_once()


def test_http_notification_callback_requires_token():
    from pydantic import ValidationError

    from conciliator.config import Settings

    options = dict(
        _env_file=None,
        aeolus_lark_callback_mode="http",
        lark_card_verification_token="test",
    )
    assert Settings(**options).aeolus_lark_callback_mode == "http"
    options["lark_card_verification_token"] = ""
    with pytest.raises(ValidationError, match="LARK_CARD_VERIFICATION_TOKEN is required"):
        Settings(**options)
    options["aeolus_lark_callback_mode"] = "websocket"
    assert Settings(**options).aeolus_lark_callback_mode == "websocket"


@pytest.mark.asyncio
async def test_deployment_result_card_has_maat_and_verified_action_is_bound_to_deployment():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.deployment = {
        "worker_id": "box-1", "deployment_id": "deploy-1", "generation": 1, "status": "succeeded", "current": True,
        "resource_owned": True, "commit_sha": "a" * 40, "display_name": "Devbox",
        "maat_url": "https://maat.bytedance.net/project/poc", "log_path": "/tmp/worker/deploy.log",
    }
    task.deployment_workers = []
    payload = _payload(cards, task, "confirm_verified")
    card = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert task.deployment["maat_url"] in card
    assert card.count("完成节点并释放环境") == 1
    assert "验证通过" in card and "跳过验证" in card
    assert "confirm_verified" not in card
    assert "verification_mode" in card
    assert task.action_token not in card
    task.deployment["deployment_id"] = "deploy-2"
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert "已过期" in response["toast"]["content"]
    service.confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_code_confirmation_card_cannot_silently_release_newly_occupied_environment():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    old_card = _payload(cards, task, "confirm")
    task.deployment = {"deployment_id": "deploy-1", "generation": 1, "status": "running", "resource_owned": True}
    result = await cards.callback(service, json.dumps(old_card).encode(), {})
    assert "已过期" in result["toast"]["content"]
    service.confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_card_offers_online_machines_and_profiles_without_slot_queue():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    profile = {"python_version": "3.11", "profile_revision": "b" * 64}
    task.deployment_workers = [
        {"worker_id": "free", "display_name": "Free box", "online": True,
         "runtime_profiles": {"py311": profile}},
        {"worker_id": "busy", "display_name": "Busy box", "online": True,
         "runtime_profiles": {"py311": profile},
         "deployment_counts": {"total": 3, "running": 3, "deploying": 0, "stopping": 0,
                               "unknown": 0, "cleanup_pending": 0}},
        {"worker_id": "offline", "display_name": "Offline box", "online": False},
    ]
    card = cards.card(task, "owner", "ou_user")
    form = next(e for e in card["body"]["elements"] if e.get("name") == "deployment_form")
    machine_select = next(e for e in card["body"]["elements"] if e.get("name") == "worker_id")
    assert [option["value"] for option in machine_select["options"]] == ["free", "busy"]
    assert form["elements"][0]["name"] == "runtime_profile_id"
    assert [option["value"] for option in form["elements"][0]["options"]] == ["free::py311::" + "b" * 64]
    serialized = json.dumps(card, ensure_ascii=False)
    assert "部署 3" in serialized and "排队" not in serialized
    payload = _payload(cards, task, "deploy")
    payload["event"]["action"]["form_value"] = {"worker_id": "busy::py311::" + "b" * 64}
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert response["toast"]["type"] == "success"
    assert response["toast"]["content"] == "Devbox 部署请求已受理"
    service.deployments.run.assert_awaited_once_with(
        "enqueue", "task", "busy", "event-1", 1, "a" * 40, "py311", "b" * 64,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["commit", "deployment"])
async def test_stale_deploy_card_cannot_enqueue_new_code_or_replace_new_deployment(change):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    payload = _payload(cards, task, "deploy")
    payload["event"]["action"]["form_value"] = {"worker_id": "box"}
    if change == "commit":
        task.rounds[-1].commit_sha = "b" * 40
    else:
        task.deployment = {"deployment_id": "new"}
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert "已过期" in response["toast"]["content"]
    service.deployments.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["verified", "skip", None, {}, []])
async def test_single_completion_button_requires_explicit_verification_choice(mode):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.deployment = {"worker_id": "box", "deployment_id": "deploy", "generation": 2,
                       "status": "succeeded", "current": True, "resource_owned": True,
                       "commit_sha": "a" * 40, "maat_url": "https://maat.bytedance.net/poc"}
    service.confirm.return_value = task
    payload = _payload(cards, task, "complete_review")
    payload["event"]["action"]["form_value"] = {"verification_mode": mode}
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    if isinstance(mode, str):
        assert response["toast"]["type"] == "success"
        service.confirm.assert_awaited_once_with("task", "event-1", "deploy" if mode == "verified" else "", True)
    else:
        assert response["toast"]["type"] == "error"
        service.confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_completion_button_rejects_replaced_deployment():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.deployment = {"deployment_id": "deploy", "generation": 2, "current": True}
    payload = _payload(cards, task, "complete_review")
    payload["event"]["action"]["form_value"] = {"verification_mode": "verified"}
    task.deployment["generation"] = 3
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert "已过期" in response["toast"]["content"]
    service.confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_deploy_rejects_offline_worker_without_completing_task():
    from conciliator.services.aeolus_registry import InvalidTaskTransition

    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    service.deployments.run.side_effect = InvalidTaskTransition("worker is offline")
    payload = _payload(cards, task, "deploy")
    payload["event"]["action"]["form_value"] = {"worker_id": "offline"}
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert response["toast"]["type"] == "error"
    service.confirm.assert_not_awaited()


def test_running_deployment_keeps_feedback_and_completion_available():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    task.deployment_workers = [{"worker_id": "box", "display_name": "Devbox", "online": True}]
    task.deployment = {"worker_id": "box", "deployment_id": "deploy", "generation": 1,
                       "status": "running", "resource_owned": True, "commit_sha": "a" * 40}
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert "deployment_form" not in content
    assert "feedback_form" in content
    assert content.count("完成节点并释放环境") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [AeolusState.REVIEW, AeolusState.RECODING, AeolusState.FAILED])
async def test_release_card_action_is_independent_of_coding_confirmation(state):
    cards, service, task, _ = _setup(state)
    task.deployment = {"worker_id": "box", "deployment_id": "deploy", "generation": 1,
                       "status": "succeeded", "current": False, "resource_owned": True, "can_release": True,
                       "status_label": "旧版本部署", "status_note": "不能验证最新代码", "commit_sha": "a" * 40}
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert "释放 Devbox" in content and "旧版本部署" in content
    assert "访问部署环境" not in content
    response = await cards.callback(service, json.dumps(_payload(cards, task, "release_environment")).encode(), {})
    assert response["toast"]["type"] == "success"
    service.deployments.run.assert_awaited_once_with("release", "task", "deploy", 1)
    service.confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_old_card_cannot_release_new_deployment():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.deployment = {"deployment_id": "deploy", "generation": 1, "can_release": True}
    payload = _payload(cards, task, "release_environment")
    task.deployment["deployment_id"] = "new-deploy"
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert "已过期" in response["toast"]["content"]
    service.deployments.run.assert_not_awaited()


def test_releasing_card_has_no_deploy_release_or_verify_buttons():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    task.deployment_workers = [{"worker_id": "box", "display_name": "Devbox", "online": True, "releasing": True}]
    task.deployment = {"worker_id": "box", "deployment_id": "deploy", "generation": 1,
                       "status": "succeeded", "current": False, "resource_owned": True,
                       "resource_releasing": True, "can_release": False, "commit_sha": "a" * 40}
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert "deployment_form" not in content and "completion_form" in content
    assert "Devbox 验证通过" not in content and "跳过验证" in content
    assert '"release_environment"' not in content
    assert "等待 worker 确认" in content and "feedback_form" in content


@pytest.mark.parametrize("state,color", [
    (AeolusState.EVALUATION_REVIEW, "orange"), (AeolusState.VALIDATION_FAILED, "orange"),
    (AeolusState.CODING, "blue"), (AeolusState.RECODING, "blue"), (AeolusState.REVIEW, "orange"),
    (AeolusState.COMPLETING, "blue"), (AeolusState.COMPLETED, "green"), (AeolusState.FAILED, "red"),
])
def test_full_lifecycle_cards_keep_forms_at_root_and_fold_optional_details(state, color):
    cards, _, task, _ = _setup(state)
    task.evaluation.update(requirement_contract="完整需求共识", risks=["兼容性风险"])
    task.rounds[-1].commit_sha = "a" * 40
    task.rounds[-1].tests = ["核心路径检查通过"]
    if state == AeolusState.VALIDATION_FAILED:
        task.evaluation.update(valid=False, missing_items=["补充验收标准"])
    card = cards.card(task, "owner", "ou_user")
    assert card["header"]["template"] == color
    assert f"任务 ID：`{task.task_id}`" in card["body"]["elements"][-1]["content"]
    root_forms = [e for e in card["body"]["elements"] if e["tag"] == "form"]
    assert len(root_forms) == len([e for e in _walk(card) if e["tag"] == "form"])
    for panel in (e for e in _walk(card) if e["tag"] == "collapsible_panel"):
        assert not any(e["tag"] == "form" for e in _walk(panel["elements"]))
        title = panel["header"]["title"]["content"]
        if any(label in title for label in ("共识", "测试记录", "风险提示", "代码提交")):
            assert panel["expanded"] is False
    primary = [e for e in _walk(card) if e["tag"] == "button" and e.get("type") == "primary"]
    assert len(primary) <= 1
    assert task.action_token not in json.dumps(card)
    if state == AeolusState.VALIDATION_FAILED:
        result = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel")
        assert result["expanded"] and "补充验收标准" in _card_text(result)


def test_large_summaries_have_short_visible_preview_and_full_collapsed_content():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].summary = "结果" * 500
    card = cards.card(task, "owner", "ou_user")
    result = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel")
    visible = [e["content"] for e in result["elements"] if e["tag"] == "markdown"]
    assert all(len(content) <= 420 for content in visible)
    detail = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel"
                  and "编码详情" in e["header"]["title"]["content"])
    assert not detail["expanded"] and task.rounds[-1].summary in _card_text(detail)


@pytest.mark.parametrize("content,folded", [
    ("字" * 160, False), ("字" * 161, True),
    ("第一行\n第二行\n第三行", False), ("一\n二\n三\n四", True),
    ("一\n\n二\n\n三", False),
])
def test_optional_details_only_fold_when_long(content, folded):
    cards, _, task, _ = _setup(AeolusState.EVALUATION_REVIEW)
    task.evaluation["requirement_contract"] = content
    card = cards.card(task, "owner", "ou_user")
    panels = [e for e in _walk(card) if e["tag"] == "collapsible_panel"
              and "共识" in e["header"]["title"]["content"]]
    assert bool(panels) is folded
    assert content in _card_text(card)
    if folded:
        assert panels[0]["expanded"] is False
        header = panels[0]["header"]
        assert "<font color='blue'>**" in header["title"]["content"]
        assert "展开查看" in header["title"]["content"]
        assert header["background_color"] == "grey"


def test_review_keeps_short_commit_visible_and_moves_traceability_into_details():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.evaluation["requirement_contract"] = "简短需求共识"
    task.rounds[-1].commit_sha = "a" * 40
    card = cards.card(task, "owner", "ou_user")
    assert "简短需求共识" in _card_text(card)
    assert "a" * 40 in _card_text(card)
    result = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel")
    visible = "\n".join(e["content"] for e in result["elements"] if e["tag"] == "markdown")
    assert "a" * 12 in visible and "a" * 40 not in visible
    details = next(e for e in card["body"]["elements"] if e["tag"] == "collapsible_panel"
                   and "编码详情" in e["header"]["title"]["content"])
    assert not details["expanded"]
    assert "简短需求共识" in _card_text(details)
    assert "a" * 40 in _card_text(details)


@pytest.mark.parametrize("has_environment", [False, True])
def test_review_primary_action_and_feedback_precede_optional_deployment_forms(has_environment):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    task.deployment_workers = [{
        "worker_id": "machine", "display_name": "Devbox", "online": True,
        "runtime_profiles": {"default": {"python_version": "3.11", "runtime_revision": "r1",
                                        "profile_revision": "p1"}},
    }]
    if has_environment:
        task.deployment = {
            "worker_id": "machine", "deployment_id": "deployment", "generation": 1, "status": "succeeded",
            "current": True, "resource_owned": True, "commit_sha": "a" * 40,
            "display_name": "Devbox", "maat_url": "https://maat.bytedance.net/project/test",
        }
    root = cards.card(task, "owner", "ou_user")["body"]["elements"]
    confirm = next(i for i, e in enumerate(root) if e.get("name") == "completion_form")
    feedback = next(i for i, e in enumerate(root) if e.get("name") == "feedback_form")
    deploy = next(i for i, e in enumerate(root) if e.get("name") == "deployment_form")
    assert confirm < feedback < deploy
    input_ = next(e for e in _walk(root[feedback]) if e["tag"] == "input")
    assert input_["max_length"] == 1000
    assert "1000" not in _card_text(root) and "4000" not in _card_text(root)
    assert "长反馈请在 Meego 填写" in input_["placeholder"]["content"]
    trace = next(i for i, e in enumerate(root) if e["tag"] == "collapsible_panel"
                 and "编码详情" in e["header"]["title"]["content"])
    assert deploy < trace


@pytest.mark.parametrize("devbox,ppe", [(False, False), (True, False), (False, True), (True, True)])
def test_completion_choices_match_available_environments_without_default(devbox, ppe):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.deployment = {"deployment_id": "box", "current": devbox, "status": "succeeded",
                       "generation": 1, "commit_sha": "a" * 40, "maat_url": "https://maat.bytedance.net/test"}
    _ppe(task)
    task.ppe_deployment["can_verify"] = ppe
    card = cards.card(task, "owner", "ou_user")
    select = next(e for e in _walk(card) if e.get("name") == "verification_mode")
    assert [o["value"] for o in select["options"]] == (
        (["all_verified"] if ppe or devbox else []) + ["skip"]
    )
    assert select["required"] is True
    assert "initial_option" not in select


@pytest.mark.parametrize("released", [False, True])
def test_environment_access_stays_at_root_and_release_stays_in_collapsed_details(released):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment.update(
        released=released, release_requested=released, can_release=not released,
        can_view_environment=not released, can_access_environment=not released,
    )
    task.deployment = {
        "worker_id": "box", "deployment_id": "deploy", "generation": 1,
        "status": "succeeded", "current": not released, "resource_owned": not released,
        "can_release": not released, "commit_sha": "a" * 40, "released": released,
        "maat_url": "https://maat.bytedance.net/test",
    }
    root = cards.card(task, "owner", "ou_user")["body"]["elements"]
    links = [e for e in root if e["tag"] == "button"
             and any(b["type"] == "open_url" for b in e.get("behaviors", []))]
    assert [e["text"]["content"] for e in links] == ([] if released else ["打开 MAAT", "查看 PPE 环境"])
    assert not any(e["tag"] == "button" and any(
        b.get("value", {}).get("action") in {"release_environment", "release_ppe"}
        for b in e.get("behaviors", [])) for e in root)
    for title, action in (("Devbox 详情", "release_environment"), ("PPE 详情", "release_ppe")):
        details = next(e for e in root if e["tag"] == "collapsible_panel"
                       and title in e["header"]["title"]["content"])
        assert not details["expanded"]
        release = [e for e in _walk(details) if e["tag"] == "button" and any(
            b.get("value", {}).get("action") == action for b in e.get("behaviors", []))]
        assert len(release) == (0 if released else 1)
    completion = next(e for e in root if e.get("name") == "completion_form")
    columns = completion["elements"][0]
    assert columns["tag"] == "column_set" and columns["flex_mode"] == "stretch"
    assert [e["name"] for e in _walk(columns) if e["tag"] in {"select_static", "button"}] == [
        "verification_mode", "submit_completion",
    ]


@pytest.mark.parametrize("devbox_state", ["new", "ready", "stale", "running", "released"])
@pytest.mark.parametrize("ppe_state", ["new", "ready", "updating", "failed"])
def test_each_environment_keeps_its_state_and_controls_together(devbox_state, ppe_state):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    task.deployment_workers = [{
        "worker_id": "box", "display_name": "Devbox", "online": True,
        "runtime_profiles": {"default": {"python_version": "3.11", "runtime_revision": "r1"}},
    }]
    if devbox_state != "new":
        task.deployment = {
            "worker_id": "box", "deployment_id": "deploy", "generation": 1,
            "status": "running" if devbox_state == "running" else "succeeded",
            "current": devbox_state == "ready", "resource_owned": devbox_state != "released",
            "can_release": devbox_state != "released", "commit_sha": "a" * 40,
            "released": devbox_state == "released", "superseded": devbox_state == "stale",
            "maat_url": "https://maat.bytedance.net/test",
        }
    _ppe(task)
    if ppe_state == "new":
        task.ppe_deployment = None
    elif ppe_state in {"updating", "failed"}:
        task.ppe_deployment.update(
            status="running" if ppe_state == "updating" else "failed",
            can_verify=False, can_retry=ppe_state == "failed",
            current_deployment={"deployment_id": "old", "commit_sha": "b" * 40, "image_version": "old"},
        )
    root = cards.card(task, "owner", "ou_user")["body"]["elements"]
    confirm = next(i for i, e in enumerate(root) if e.get("name") == "completion_form")
    feedback = next(i for i, e in enumerate(root) if e.get("name") == "feedback_form")
    devbox_headings = [i for i, e in enumerate(root) if "**Devbox · " in e.get("content", "")]
    ppe_headings = [i for i, e in enumerate(root) if "**PPE · " in e.get("content", "")]
    assert len(devbox_headings) == len(ppe_headings) == 1
    devbox, ppe = devbox_headings[0], ppe_headings[0]
    assert confirm < feedback < devbox < ppe
    assert not any(e["tag"] in {"button", "select_static", "form"} for e in root[confirm + 1:feedback])
    for i, element in enumerate(root):
        for child in _walk(element):
            for behavior in child.get("behaviors", []):
                action = behavior.get("value", {}).get("action")
                if action in {"deploy", "choose_deployment_worker", "release_environment"}:
                    assert devbox < i < ppe
                elif action in {"deploy_ppe", "retry_ppe", "release_ppe"}:
                    assert i > ppe
    if devbox_state != "new":
        details = next(i for i, e in enumerate(root) if e["tag"] == "collapsible_panel"
                       and "Devbox 详情" in e["header"]["title"]["content"])
        forms = [i for i, e in enumerate(root) if e.get("name") == "deployment_form"]
        assert all(i < details for i in forms)
    assert not any(e["tag"] == "form" for panel in root if panel["tag"] == "collapsible_panel"
                   for e in _walk(panel))


@pytest.mark.asyncio
@pytest.mark.parametrize("devbox,ppe", [(True, True), (True, False), (False, True)])
async def test_lark_verification_confirms_all_available_environments(devbox, ppe):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment["can_verify"] = ppe
    task.deployment = {"deployment_id": "box", "generation": 1, "current": devbox,
                       "status": "succeeded", "commit_sha": "a" * 40, "maat_url": "https://maat.bytedance.net/test"}
    service.confirm.return_value = task
    payload = _payload(cards, task, "complete_review")
    payload["event"]["action"]["form_value"] = {"verification_mode": "all_verified"}
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert response["toast"]["type"] == "success"
    targets = (tuple([DeploymentVerification("devbox", "box")]) if devbox else ()) + (
        tuple([DeploymentVerification("ppe", "ppe-record")]) if ppe else ()
    )
    service.confirm.assert_awaited_once_with(
        "task", "event-1", release_authorized=True, verification_targets=targets,
    )


@pytest.mark.asyncio
async def test_lark_verification_rejects_changed_environment_availability():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    payload = _payload(cards, task, "complete_review")
    payload["event"]["action"]["form_value"] = {"verification_mode": "all_verified"}
    task.ppe_deployment["can_verify"] = False
    response = await cards.callback(service, json.dumps(payload).encode(), {})
    assert "过期" in response["toast"]["content"]
    service.confirm.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["skip", "verified", "ppe_verified", "", None])
async def test_completion_without_environment_requires_explicit_skip(mode):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    service.confirm.return_value = task
    payload = _payload(cards, task, "complete_review")
    payload["event"]["action"]["form_value"] = {"verification_mode": mode}
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    if mode == "skip":
        assert result["toast"]["type"] == "success"
        service.confirm.assert_awaited_once_with("task", "event-1", "", True)
    else:
        assert result["toast"]["type"] == "error"
        service.confirm.assert_not_awaited()


@pytest.mark.parametrize("profiles", [{}, {"only": {"python_version": "3.11", "runtime_revision": "r1"}}])
def test_runtime_select_handles_empty_profiles_and_stale_recommendation(profiles):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    task.deployment_workers = [{
        "worker_id": "box", "display_name": "box", "online": True,
        "runtime_profiles": profiles, "recommended_runtime_profile_id": "removed",
    }]
    card = cards.card(task, "owner", "ou_user")
    forms = [e for e in _walk(card) if e.get("name") == "deployment_form"]
    if profiles:
        assert "initial_option" not in forms[0]["elements"][0]
    else:
        assert not forms
        assert "暂无可用运行配置" in _card_text(card)


def test_execution_failure_without_coding_rounds_still_shows_redacted_error():
    cards, _, task, _ = _setup(AeolusState.FAILED)
    task.rounds = []
    task.last_error = "failed; token=private-value"
    card = cards.card(task, "owner", "ou_user")
    assert "failed" in _card_text(card)
    assert "private-value" not in json.dumps(card)
    assert card["header"]["template"] == "red"


@pytest.mark.asyncio
async def test_blocked_card_shows_partial_results_and_only_accepts_feedback():
    cards, service, task, _ = _setup(AeolusState.CODING_BLOCKED)
    task.rounds = [AeolusRound(
        1, "blocked", summary="已完成部分修改", tests=["56 passed"],
        unresolved_items=["需要提供集成环境"], failure_code="env_missing",
        code_url="https://code.byted.org/dp/aeolus/commit/" + "a" * 40,
    )]
    card = cards.card(task, "owner", "ou_user")
    text = _card_text(card)
    assert "已完成部分修改" in text
    assert "1. 需要提供集成环境" in text
    assert "env_missing" in text
    assert card["header"]["template"] == "orange"
    buttons = [element for element in _walk(card) if element["tag"] == "button"]
    labels = [element["text"]["content"] for element in buttons]
    assert "Review 已有代码" in labels
    assert "提交意见并继续 Coding" in labels
    assert not any("完成节点" in label or "部署当前代码" in label or "重试编码" in label for label in labels)
    service.submit_feedback.return_value = task
    result = await cards.callback(service, json.dumps(_payload(cards, task, "submit_feedback")).encode(), {})
    assert result["toast"]["type"] == "success"
    service.submit_feedback.assert_awaited_once()
    with pytest.raises(InvalidCardCallback, match="not available"):
        await cards.callback(service, json.dumps(_payload(cards, task, "confirm")).encode(), {})


def test_v2_machine_choices_show_counts_and_bind_named_runtime():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = 'a' * 40
    task.deployment_workers = [{
        'worker_id': 'machine', 'display_name': 'Devbox', 'online': True, 'protocol_version': 2,
        'deployment_counts': {
            'total': 3, 'running': 2, 'deploying': 1, 'stopping': 0, 'unknown': 0, 'cleanup_pending': 1,
        },
        'runtime_profiles': {'py311': {'python_version': '3.11', 'runtime_revision': 'r1'}},
    }]
    card = cards.card(task, 'owner', 'recipient')
    rendered = json.dumps(card, ensure_ascii=False)
    assert '部署 3' in rendered and '清理待处理 1' in rendered
    assert 'machine::py311' in rendered and 'Python 3.11' in rendered
    assert '占用，可排队' not in rendered


@pytest.mark.asyncio
async def test_worker_switch_only_refreshes_profiles_and_submit_binds_the_selected_machine():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    task.deployment_workers = [
        {"worker_id": "one", "display_name": "One", "online": True,
         "runtime_profiles": {"py310": {"python_version": "3.10", "profile_revision": "a" * 64}}},
        {"worker_id": "two", "display_name": "Two", "online": True, "recommended_runtime_profile_id": "py311",
         "runtime_profiles": {"py311": {"python_version": "3.11", "profile_revision": "b" * 64}}},
    ]
    payload = _payload(cards, task, "choose_deployment_worker")
    payload["event"]["action"]["option"] = "two"
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    service.deployments.run.assert_not_awaited()
    card = result["card"]["data"]
    machine = next(e for e in card["body"]["elements"] if e.get("name") == "worker_id")
    assert machine["initial_option"] == "two"
    form = next(e for e in card["body"]["elements"] if e.get("name") == "deployment_form")
    runtime = form["elements"][0]
    assert runtime["name"] == "runtime_profile_id"
    assert [o["value"] for o in runtime["options"]] == ["two::py311::" + "b" * 64]
    assert runtime["initial_option"] == "two::py311::" + "b" * 64
    payload = _payload(cards, task, "deploy")
    payload["event"]["action"]["form_value"] = {"runtime_profile_id": runtime["initial_option"]}
    await cards.callback(service, json.dumps(payload).encode(), {})
    service.deployments.run.assert_awaited_once_with(
        "enqueue", "task", "two", "event-1", 1, "a" * 40, "py311", "b" * 64,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_id", ["offline", "missing", None])
async def test_invalid_worker_switch_cannot_enqueue_or_update_card(worker_id):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    task.deployment_workers = [{"worker_id": "offline", "display_name": "Offline", "online": False}]
    payload = _payload(cards, task, "choose_deployment_worker")
    payload["event"]["action"]["option"] = worker_id
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    assert result["toast"]["type"] == "error" and "card" not in result
    service.deployments.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_notification_uuid_distinguishes_maat_creation_completion_and_failure():
    cards, _, task, client = _setup(AeolusState.REVIEW)
    task.deployment = {
        "deployment_id": "deployment", "worker_id": "machine", "status": "running", "stage": "checkout",
        "commit_sha": "a" * 40, "generation": 1,
    }
    async def sent_uuid():
        await cards._send(task, "owner", "recipient")
        return client._request.call_args.kwargs["json"]["uuid"]
    starting = await sent_uuid()
    task.deployment.update(stage="maat_entry_pending", maat_environment_status="deploying_file")
    pending = await sent_uuid()
    task.deployment.update(maat_environment_status="deploy_file_ok")
    complete = await sent_uuid()
    assert await sent_uuid() == complete
    task.deployment.update(status="failed", stage="maat_deployment_failed")
    failed = await sent_uuid()
    assert len({starting, pending, complete, failed}) == 4


def _ppe(task):
    task.rounds[-1].commit_sha = 'a' * 40
    task.ppe_capability = {
        'enabled': True, 'profile_revision': 'b' * 64,
        'runtime_profiles': {'py3': {'python_version': '3.10', 'profile_revision': 'b' * 64}},
        'recommended_runtime_profile_id': 'py3',
    }
    task.ppe_deployment = {
        'deployment_id': 'ppe-record', 'environment_id': 'c' * 32, 'generation': 1,
        'environment_name': 'ppe_ai_ae_' + 'c' * 16, 'status': 'deployed', 'stage': 'entry_pending',
        'status_label': 'PPE 部署完成，待业务验证', 'resource_state': 'running', 'commit_sha': 'a' * 40,
        'round_no': 1, 'can_verify': True, 'can_release': True, 'can_deploy': False, 'clusters': [],
        'can_view_environment': True, 'can_access_environment': True,
        'environment_url': 'https://bits.bytedance.net/env/test', 'access_hint': 'X-TT-ENV: ppe_test',
    }


@pytest.mark.parametrize("verified", [False, True])
def test_ppe_card_preserves_unready_status_and_recovers_verification_label(verified):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment.update(
        can_verify=False, verified=verified, status_label="PPE 暂未就绪，正在重新检查",
    )
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert "**PPE · PPE 暂未就绪，正在重新检查**" in content
    assert "**PPE · 待验证**" not in content and "**PPE · 已验证**" not in content
    assert '"value": "all_verified"' not in content

    task.ppe_deployment.update(can_verify=True, status_label="PPE 部署完成，待业务验证")
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert "暂未就绪" not in content
    assert f"**PPE · {'已验证' if verified else '待验证'}**" in content
    assert '"value": "all_verified"' in content


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["dp/aeolus", "dp/VizQueryService"])
async def test_ppe_card_works_without_devbox_and_binds_operation_identity(repository):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.evaluation = {"code_context": {"repository": repository}}
    task.ppe_deployment.update(
        bits_run_url="https://bits.bytedance.net/devops/space/pipeline/detail/1?runId=42",
        scm_url="https://cloud.bytedance.net/scm/detail/572600/versions/",
        image_url="https://cloud.bytedance.net/icm/detail/217258/versions?version=1.0.2.651",
    )
    assert task.deployment_workers is None
    content = json.dumps(cards.card(task, 'owner', 'ou_user'), ensure_ascii=False)
    assert '一键部署 PPE' not in content and '验证通过' in content
    assert "BITS 构建运行" in content and "runId=42" in content
    assert "SCM 制品" in content and "ICM 镜像构建" in content
    task.rounds.append(AeolusRound(2, "review", commit_sha='d' * 40))
    task.ppe_deployment['can_deploy'] = True
    assert '一键部署 PPE' in json.dumps(cards.card(task, 'owner', 'ou_user'), ensure_ascii=False)
    payload = _payload(cards, task, 'deploy_ppe')
    payload['event']['action']['form_value'] = {'ppe_runtime_profile_id': 'py3::' + 'b' * 64}
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    assert result['toast']['type'] == 'success'
    assert result['toast']['content'] == 'PPE 部署请求已受理'
    service.ppe_deployments.create.assert_awaited_once_with('task', 'event-1', 2, 'd'*40, 'b'*64, 'py3')
    stale = _payload(cards, task, 'release_ppe')
    task.ppe_deployment['generation'] = 2
    result = await cards.callback(service, json.dumps(stale).encode(), {})
    assert '过期' in result['toast']['content']
    service.ppe_deployments.release.assert_not_awaited()


@pytest.mark.parametrize("recommended", ["py27", None])
def test_ppe_message_offers_runtime_selection_and_sha_evidence(recommended):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment = None
    task.ppe_capability.update(
        runtime_profiles={
            "py27": {"python_version": "2.7", "profile_revision": "p27", "compatible": True},
            "py3": {"python_version": "3.10", "profile_revision": "p3", "compatible": False},
        },
        recommended_runtime_profile_id=recommended,
        runtime_detection={
            "commit_sha": "a" * 40,
            "evidence": [{"path": "requirements.txt", "detail": "antlr4-python2-runtime 要求 Python 2"}],
            "warnings": ["请手动选择"] if recommended is None else [],
            "branch_hints": ["py3"],
        },
    )
    card = cards.card(task, "owner", "ou_user")
    root = card["body"]["elements"]
    form = next(e for e in root if e.get("name") == "ppe_deployment_form")
    selector, button = form["elements"]
    assert selector["name"] == "ppe_runtime_profile_id" and selector["required"]
    assert [o["value"] for o in selector["options"]] == ["py27::p27"]
    assert ("initial_option" in selector) == bool(recommended)
    if recommended:
        assert selector["initial_option"] == "py27::p27"
        assert "推荐" in selector["options"][0]["text"]["content"]
    assert button["form_action_type"] == "submit"
    assert button["behaviors"][0]["value"]["ppe_runtime_profiles"] == {"py27": "p27"}
    assert "requirements.txt" in _card_text(root)
    assert "aaaaaaaaaaaa" in _card_text(root)
    assert "仅供参考" in _card_text(root)
    assert "private-token" not in json.dumps(card)


@pytest.mark.asyncio
async def test_ppe_message_manual_choice_uses_selected_revision_without_recommendation():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment = None
    task.ppe_capability.update(
        profile_revision="", recommended_runtime_profile_id=None,
        runtime_profiles={
            "py27": {"python_version": "2.7", "profile_revision": "p27"},
            "py3": {"python_version": "3.10", "profile_revision": "p3"},
        },
    )
    payload = _payload(cards, task, "deploy_ppe")
    payload["event"]["action"]["form_value"] = {"ppe_runtime_profile_id": "py27::p27"}
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    assert result["toast"]["type"] == "success"
    service.ppe_deployments.create.assert_awaited_once_with("task", "event-1", 1, "a" * 40, "p27", "py27")


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [None, [], "py3", "py3::old", "unknown::p3", "py3::p3::extra", "py27::p27"])
async def test_ppe_message_rejects_missing_invalid_or_incompatible_selection(selection):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_capability["runtime_profiles"] = {
        "py3": {"python_version": "3.10", "profile_revision": "p3", "compatible": True},
        "py27": {"python_version": "2.7", "profile_revision": "p27", "compatible": False},
    }
    payload = _payload(cards, task, "deploy_ppe")
    payload["event"]["action"]["form_value"] = {"ppe_runtime_profile_id": selection}
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    assert result["toast"]["type"] == "error"
    service.ppe_deployments.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["sha", "revision", "compatibility"])
async def test_ppe_message_selection_expires_when_sha_or_candidates_change(change):
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    payload = _payload(cards, task, "deploy_ppe")
    payload["event"]["action"]["form_value"] = {"ppe_runtime_profile_id": "py3::" + "b" * 64}
    if change == "sha":
        task.rounds[-1].commit_sha = "d" * 40
    else:
        task.ppe_capability["runtime_profiles"]["py3"].update(
            {"profile_revision": "new"} if change == "revision" else {"compatible": False},
        )
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    assert result["toast"]["type"] == "info"
    service.ppe_deployments.create.assert_not_awaited()


def test_ppe_message_does_not_offer_deploy_when_all_candidates_are_incompatible():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment = None
    task.ppe_capability["runtime_profiles"]["py3"]["compatible"] = False
    card = cards.card(task, "owner", "ou_user")
    assert not any(e.get("name") == "ppe_deployment_form" for e in _walk(card))
    assert "暂无与当前代码兼容" in _card_text(card)


@pytest.mark.parametrize("recommended", ["py27", None])
def test_devbox_message_respects_recommendation_and_compatibility(recommended):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.rounds[-1].commit_sha = "a" * 40
    task.deployment_workers = [{
        "worker_id": "box", "display_name": "Devbox", "online": True,
        "recommended_runtime_profile_id": recommended,
        "runtime_profiles": {
            "py27": {"python_version": "2.7", "profile_revision": "p27", "compatible": True},
            "py3": {"python_version": "3.10", "profile_revision": "p3", "compatible": False},
        },
        "runtime_detection": {"commit_sha": "a" * 40, "warnings": ["请手动选择"], "evidence": []},
    }]
    card = cards.card(task, "owner", "ou_user")
    selector = next(e for e in _walk(card) if e.get("name") == "runtime_profile_id")
    assert [o["value"] for o in selector["options"]] == ["box::py27::p27"]
    assert ("initial_option" in selector) == bool(recommended)
    assert "请手动选择" in _card_text(card)


@pytest.mark.asyncio
async def test_ppe_card_completes_with_explicit_verification_target():
    cards, service, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    service.confirm.return_value = task
    payload = _payload(cards, task, 'complete_review')
    payload['event']['action']['form_value'] = {'verification_mode': 'ppe_verified'}
    result = await cards.callback(service, json.dumps(payload).encode(), {})
    assert result['toast']['type'] == 'success'
    service.confirm.assert_awaited_once_with('task', 'event-1', 'ppe-record', True, 'ppe')


def test_ppe_card_keeps_previous_version_and_access_notice_during_update():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment.update(
        status="running", stage="image_wait", status_label="等待 ICM 镜像构建",
        image_version="1.0.2.674", can_verify=False,
        current_deployment={
            "deployment_id": "old", "commit_sha": "b" * 40, "image_version": "1.0.2.672",
        },
    )
    content = json.dumps(cards.card(task, 'owner', 'ou_user'), ensure_ascii=False)
    assert "保留上一版 PPE 环境" in content
    assert content.index("上一版就绪镜像") < content.index("目标提交")
    assert "1.0.2.672" in content and "1.0.2.674" in content
    assert "环境访问方式（更新时保持不变）" in content
    task.ppe_deployment.update(
        released=True, release_requested=True, can_view_environment=False, can_access_environment=False,
    )
    content = json.dumps(cards.card(task, 'owner', 'ou_user'), ensure_ascii=False)
    assert "保留上一版 PPE 环境" not in content
    assert "环境访问方式（更新时保持不变）" not in content


@pytest.mark.asyncio
async def test_ppe_notifications_change_uuid_with_stage_and_propagate_delivery_failure():
    cards, _, task, client = _setup(AeolusState.REVIEW)
    _ppe(task)
    await cards.notify(task, strict=True, ppe_only=True)
    before = client._request.call_args.kwargs['json']['uuid']
    task.ppe_deployment.update(status='failed', stage='failed', error='构建失败')
    await cards.notify(task, strict=True, ppe_only=True)
    after = client._request.call_args.kwargs['json']['uuid']
    assert before != after
    client._request.side_effect = RuntimeError('offline')
    with pytest.raises(RuntimeError):
        await cards.notify(task, strict=True, ppe_only=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,stage", [
    ("pending", "preflight"), ("running", "bits_wait"), ("running", "image_wait"),
    ("running", "ticket_wait"), ("running", "observe"), ("releasing", "release"),
    ("released", "released"), ("superseded", "superseded"),
])
async def test_ppe_result_notify_does_not_resolve_owners_or_send_progress(status, stage):
    cards, _, task, client = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment.update(status=status, stage=stage)
    await cards.notify(task, strict=True, ppe_only=True)
    cards.meegle.get_node_owner_keys.assert_not_awaited()
    client._request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,stage,resource_state,title", [
    ("deployed", "entry_pending", "running", "PPE 部署成功"),
    ("failed", "failed", "planned", "PPE 部署失败"),
    ("unknown", "unknown", "deploying", "PPE 部署结果待确认"),
    ("failed", "failed", "cleanup_failed", "PPE 释放失败"),
])
async def test_ppe_result_notification_is_compact_and_deduplicates_transient_details(
    status, stage, resource_state, title,
):
    cards, _, task, client = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment.update(
        status=status, stage=stage, resource_state=resource_state, round_no=1,
        image_version="1.0.2.674", error="失败原因" * 100,
        bits_run_url="https://bits.bytedance.net/build",
        scm_version="1.0.0.2146", clusters=[{"name": "online", "ready": True}],
        can_view_environment=resource_state != "planned", can_access_environment=status == "deployed",
    )
    await cards.notify(task, strict=True, ppe_only=True)
    sent = client._request.call_args.kwargs["json"]
    card = json.loads(sent["content"])
    assert title in card["header"]["title"]["content"]
    assert len(card["body"]["elements"]) == (5 if status == "deployed" or resource_state == "cleanup_failed" else
                                            3 if resource_state == "planned" else 4)
    assert f"任务 ID：`{task.task_id}`" in card["body"]["elements"][-1]["content"]
    assert "1.0.2.674" in sent["content"] and "aaaaaaaa" in sent["content"]
    assert "private-token" not in sent["content"]
    assert "collapsible_panel" not in sent["content"] and '"tag": "form"' not in sent["content"]
    assert "SCM" not in sent["content"] and "BITS" not in sent["content"]
    assert "online" not in sent["content"] and "需求共识" not in sent["content"]
    if status == "deployed":
        assert "X-TT-ENV" in sent["content"] and "Meego 卡片确认结果" in sent["content"]
    else:
        assert len(card["body"]["elements"][1]["content"]) <= 201
    task.ppe_deployment.update(status_label="状态描述变化", error="暂时读失败")
    task.state = AeolusState.RECODING
    await cards.notify(task, strict=True, ppe_only=True)
    assert client._request.call_args.kwargs["json"]["uuid"] == sent["uuid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,resource_state,title", [
    ("released", "released", "PPE 释放完成"),
    ("failed", "cleanup_failed", "PPE 释放失败"),
    ("unknown", "releasing", "PPE 释放结果待确认"),
])
async def test_ppe_release_results_are_distinct_from_deployment_results(status, resource_state, title):
    cards, _, task, client = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment.update(
        status=status, stage=status, resource_state=resource_state, release_requested=True,
        superseded=True, released=status == "released", release_ticket_id="222",
        can_access_environment=False, can_view_environment=False, can_release=False,
        error="原错误不应污染完成通知" if status == "released" else "释放平台调用失败",
    )
    await cards.notify(task, strict=True, ppe_only=True)
    sent = client._request.call_args.kwargs["json"]
    card = json.loads(sent["content"])
    assert title in sent["content"]
    assert not any(e["tag"] == "button" for e in _walk(card))
    if status == "released":
        assert card["header"]["template"] == "green"
        assert "已确认回收" in sent["content"] and "原错误" not in sent["content"]
    else:
        assert "资源尚未确认回收" in sent["content"]
    await cards.notify(task, strict=True, ppe_only=True)
    assert client._request.call_args.kwargs["json"]["uuid"] == sent["uuid"]
    if status != "released":
        task.ppe_deployment["release_ticket_id"] = "333"
        await cards.notify(task, strict=True, ppe_only=True)
        assert client._request.call_args.kwargs["json"]["uuid"] != sent["uuid"]


@pytest.mark.asyncio
async def test_ppe_release_request_suppresses_a_late_deployment_success():
    cards, _, task, client = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment["release_requested"] = True
    await cards.notify(task, ppe_only=True)
    cards.meegle.get_node_owner_keys.assert_not_awaited()
    client._request.assert_not_awaited()


@pytest.mark.parametrize("view,access", [(False, False), (True, False), (True, True)])
def test_ppe_management_entry_and_business_hint_follow_server_eligibility(view, access):
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    _ppe(task)
    task.ppe_deployment.update(
        status="running", stage="ticket_wait", can_verify=False,
        can_view_environment=view, can_access_environment=access,
    )
    content = json.dumps(cards.card(task, "owner", "ou_user"), ensure_ascii=False)
    assert ("查看 PPE 环境" in content) == view
    assert ("请求头 X-TT-ENV" in content) == access
    assert ("环境访问方式（更新时保持不变）" in content) == access
    assert ("可查看部署进度" in content) == (view and not access)
    task.ppe_deployment.update(status="failed", stage="failed", error="构建或部署失败")
    result = json.dumps(cards.ppe_result_card(task, "owner", "ou_user"), ensure_ascii=False)
    assert ("查看 PPE 环境" in result) == view
    assert "X-TT-ENV" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {}, {"stage": "maat_proxy"}, {"stage": "maat_entry_pending", "maat_environment_status": "deploying_file"},
    {"status": "cancelled", "stage": "cleanup_pending", "released": True, "cleanup_state": "pending"},
    {"status": "cancelled", "stage": "cleanup_pending", "released": True, "cleanup_state": "running"},
    {"status": "cancelled", "stage": "maat_cleanup_records_deleted", "released": True, "cleanup_state": "pending"},
    {"status": "cancel_requested", "resource_releasing": True},
    {"status": "cancelled", "released": True}, {"status": "succeeded", "current": False},
    {"stage": "maat_entry_pending", "maat_environment_status": "deploy_file_ok", "superseded": True},
    {"stage": "maat_entry_pending", "maat_environment_status": "deploy_file_ok", "resource_releasing": True},
])
async def test_devbox_result_notifications_skip_progress_and_unusable_environments(changes):
    cards, _, task, client = _setup(AeolusState.REVIEW)
    task.deployment = {
        "deployment_id": "devbox", "worker_id": "machine", "status": "running", "stage": "checkout",
        "commit_sha": "a" * 40, "maat_url": "https://test.maat.bytedance.net", **changes,
    }
    await cards.notify(task, devbox_only=True)
    cards.meegle.get_node_owner_keys.assert_not_awaited()
    client._request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("changes,title", [
    ({"status": "succeeded", "stage": "ready", "current": True}, "Devbox 部署成功"),
    ({"stage": "maat_entry_pending", "maat_environment_status": "deploy_file_ok"}, "Devbox 部署成功"),
    ({"status": "failed", "stage": "maat_deployment_failed"}, "Devbox 部署失败"),
    ({"status": "cancel_requested", "deployment_failed": True}, "Devbox 部署失败"),
    ({"status": "cancel_requested", "stage": "maat_write_unknown"}, "Devbox 部署结果待确认"),
    ({"status": "cancelled", "stage": "cleanup_pending", "released": True,
      "cleanup_state": "manual_required"}, "Devbox 释放待人工处理"),
    ({"status": "cancelled", "stage": "maat_cleanup_records_deleted", "released": True,
      "cleanup_state": "succeeded"}, "Devbox 释放完成"),
    ({"status": "failed", "deployment_failed": True, "stage": "maat_cleanup_records_deleted", "released": True,
      "cleanup_state": "succeeded"}, "Devbox 释放完成"),
    ({"status": "cancelled", "stage": "released", "released": True,
      "cleanup_state": "succeeded"}, "Devbox 释放完成"),
    ({"status": "cancelled", "stage": "maat_cleanup_unknown", "released": True,
      "cleanup_state": "manual_required"}, "Devbox 释放结果待确认"),
])
async def test_devbox_result_notifications_are_compact_and_keep_only_usable_links(changes, title):
    cards, _, task, client = _setup(AeolusState.REVIEW)
    task.deployment = {
        "deployment_id": "devbox", "worker_id": "machine", "display_name": "测试机器", "status": "running",
        "stage": "checkout", "commit_sha": "a" * 40, "round_no": 1,
        "maat_url": "https://test.maat.bytedance.net", "runtime_revision": "internal-revision",
        "log_path": "/private/worker/log", "error": "失败原因" * 100, **changes,
    }
    await cards.notify(task, devbox_only=True, strict=True)
    sent = client._request.call_args.kwargs["json"]
    card = json.loads(sent["content"])
    assert title in card["header"]["title"]["content"]
    assert "测试机器" in sent["content"] and "aaaaaaaa" in sent["content"]
    assert f"任务 ID：`{task.task_id}`" in card["body"]["elements"][-1]["content"]
    assert "private-token" not in sent["content"] and "/private/worker/log" not in sent["content"]
    assert "internal-revision" not in sent["content"] and "需求共识" not in sent["content"]
    assert not any(e["tag"] in {"form", "collapsible_panel"} for e in _walk(card))
    if devbox_result_kind(task.deployment) == "success":
        assert "访问部署环境（MAAT）" in sent["content"]
    else:
        assert "访问部署环境（MAAT）" not in sent["content"]
        assert len(card["body"]["elements"][1]["content"]) <= 201
    if devbox_result_kind(task.deployment) == "released":
        assert card["header"]["template"] == "green"
        if task.deployment["stage"] == "maat_cleanup_records_deleted":
            assert "代理记录已清理" in sent["content"] and "旧端口继续保留预留" in sent["content"]
        else:
            assert "端口预留已释放" in sent["content"]
        assert "失败原因" not in sent["content"]
        assert not any(e["tag"] == "button" for e in _walk(card))
    task.deployment.update(error="暂时查询失败", status_label="新描述")
    task.state = AeolusState.RECODING
    await cards.notify(task, devbox_only=True)
    assert client._request.call_args.kwargs["json"]["uuid"] == sent["uuid"]


@pytest.mark.asyncio
async def test_devbox_failure_and_confirmed_release_have_distinct_deduplicated_messages():
    cards, _, task, client = _setup(AeolusState.REVIEW)
    task.deployment = {
        "deployment_id": "devbox", "worker_id": "machine", "commit_sha": "a" * 40,
        "status": "cancel_requested", "stage": "starting", "deployment_failed": True, "error": "服务启动失败",
    }
    await cards.notify(task, devbox_only=True)
    identity = client._request.call_args.kwargs["json"]["uuid"]
    task.deployment.update(status="failed", stage="released", released=True, cleanup_state="succeeded")
    await cards.notify(task, devbox_only=True)
    sent = client._request.call_args.kwargs["json"]
    assert sent["uuid"] != identity
    assert "Devbox 释放完成" in sent["content"] and "端口预留已释放" in sent["content"]
    await cards.notify(task, devbox_only=True)
    assert client._request.call_args.kwargs["json"]["uuid"] == sent["uuid"]


def test_devbox_success_links_work_item_inline_like_ppe():
    cards, _, task, _ = _setup(AeolusState.REVIEW)
    task.project_key = "6aabe23024218882059f74a4"
    task.deployment = {
        "deployment_id": "devbox", "worker_id": "machine", "commit_sha": "a" * 40,
        "status": "succeeded", "stage": "ready", "current": True,
        "maat_url": "https://test.maat.bytedance.net",
    }
    card = cards.devbox_result_card(task, "owner", "ou_user")
    assert "[工作项 42](https://meego.larkoffice.com/4ebse5/story/detail/42)" in _card_text(card)
    buttons = [element for element in _walk(card) if element["tag"] == "button"]
    assert len(buttons) == 1
    assert buttons[0]["text"]["content"] == "访问部署环境（MAAT）"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["devbox", "ppe"])
@pytest.mark.parametrize("scenario", [
    "release", "unavailable", "generation_changed", "deployment_changed", "wrong_recipient", "owner_removed",
])
async def test_success_notification_release_button_uses_existing_bound_callback(target, scenario):
    cards, service, task, client = _setup(AeolusState.REVIEW)
    if target == "ppe":
        _ppe(task)
        deployment = task.ppe_deployment
        action, label = "release_ppe", "释放 PPE"
    else:
        task.rounds[-1].commit_sha = "a" * 40
        task.deployment = {
            "deployment_id": "devbox", "worker_id": "machine", "generation": 1, "round_no": 1,
            "commit_sha": "a" * 40, "status": "succeeded", "stage": "ready", "current": True,
            "resource_owned": True, "can_release": True, "status_label": "部署成功",
            "maat_url": "https://test.maat.bytedance.net",
        }
        deployment = task.deployment
        action, label = "release_environment", "释放 Devbox"
    if scenario == "unavailable":
        deployment["can_release"] = False
    await cards.notify(task, ppe_only=target == "ppe", devbox_only=target == "devbox", strict=True)
    sent = client._request.call_args.kwargs["json"]
    card = json.loads(sent["content"])
    callbacks = [
        element for element in _walk(card) if element["tag"] == "button"
        and any(behavior["type"] == "callback" for behavior in element.get("behaviors", []))
    ]
    assert task.action_token not in sent["content"]
    assert any(behavior["type"] == "open_url" for element in _walk(card)
               for behavior in element.get("behaviors", []))
    if scenario == "unavailable":
        assert not callbacks
        return
    assert len(callbacks) == 1 and callbacks[0]["text"]["content"] == label
    value = callbacks[0]["behaviors"][0]["value"]
    assert value == cards._value(task, action, "owner", "ou_user")
    payload = _payload(cards, task, action)
    payload["event"]["action"]["value"] = value
    if scenario == "generation_changed":
        deployment["generation"] += 1
    elif scenario == "deployment_changed":
        deployment["deployment_id"] = "new-deployment"
    elif scenario == "wrong_recipient":
        payload["event"]["operator"]["open_id"] = "another-user"
    elif scenario == "owner_removed":
        cards.meegle.get_node_owner_keys.return_value = []
    if scenario in {"wrong_recipient", "owner_removed"}:
        with pytest.raises(InvalidCardCallback):
            await cards.callback(service, json.dumps(payload).encode(), {})
    else:
        response = await cards.callback(service, json.dumps(payload).encode(), {})
        assert response["toast"]["type"] == ("success" if scenario == "release" else "info")
        if scenario == "release":
            assert response["toast"]["content"] == f"{'PPE' if target == 'ppe' else 'Devbox'} 释放请求已受理"
    if scenario == "release" and target == "ppe":
        service.ppe_deployments.release.assert_awaited_once_with("task", "ppe-record", "c" * 32, 1)
        service.deployments.run.assert_not_awaited()
    elif scenario == "release":
        service.deployments.run.assert_awaited_once_with("release", "task", "devbox", 1)
        service.ppe_deployments.release.assert_not_awaited()
    else:
        service.deployments.run.assert_not_awaited()
        service.ppe_deployments.release.assert_not_awaited()
    service.confirm.assert_not_awaited()
