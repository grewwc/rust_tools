import hashlib
import json

import pytest

from conciliator.domain.aeolus_task import (
    InvalidAeolusWebhook,
    extract_requirement_url,
    parse_start_event,
    verify_webhook_signature,
)


def _payload(*, event_type: int = 8001, status: str = "running") -> dict:
    payload = {
        "request_time": 1767872405616,
        "source_plugin_id": "MII_test",
        "data": {
            "idempotent_key": "idem-1",
            "project_key": "space-1",
            "event_type": event_type,
            "user_info": {"user_key": "user-1"},
            "work_item_info": [{"work_item_id": 42}],
            "change_ai_node_info": {
                "after": {"entity_id": "node-uuid-1", "status": status}
            },
            "ai_node_info": {
                "state_key": "state-1",
                "ai_info": {
                    "props": [
                        {
                            "prop_key": "design_doc",
                            "prop_value": {
                                "field": [
                                    {
                                        "field_type": "link_cloud_doc",
                                        "field_value": json.dumps(
                                            ["https://example.feishu.cn/docx/doccnABC"]
                                        ),
                                    }
                                ]
                            },
                        }
                    ]
                },
            },
        },
    }
    payload["signature"] = hashlib.sha256(
        f"MII_test{payload['request_time']}webhook-secret".encode()
    ).hexdigest()
    return payload


def test_verifies_documented_raw_sha256_signature():
    payload = _payload()
    assert verify_webhook_signature(payload, "webhook-secret")
    payload["signature"] = "0" * 64
    assert not verify_webhook_signature(payload, "webhook-secret")


def test_parses_running_8001_event_and_cloud_doc_url():
    event = parse_start_event(
        _payload(), expected_plugin_id="MII_test", webhook_token="webhook-secret"
    )
    assert event is not None
    assert event.task_id == "idem-1"
    assert event.work_item_id == 42
    assert event.node_uuid == "node-uuid-1"
    assert event.state_key == "state-1"
    assert event.user_key == "user-1"
    assert extract_requirement_url(event) == "https://example.feishu.cn/docx/doccnABC"


@pytest.mark.parametrize("nested", [False, True])
def test_start_event_retains_meegle_item_type(nested):
    payload = _payload()
    target = payload["data"]["work_item_info"][0] if nested else payload["data"]
    target["work_item_type_key"] = "story"
    event = parse_start_event(payload, expected_plugin_id="MII_test", webhook_token="webhook-secret")
    assert event.work_item_type_key == "story"
    assert event.user_key == "user-1"


@pytest.mark.parametrize("event_type,status", [(8002, "running"), (8001, "done")])
def test_non_start_events_are_ignored(event_type, status):
    assert (
        parse_start_event(
            _payload(event_type=event_type, status=status),
            expected_plugin_id="MII_test",
            webhook_token="webhook-secret",
        )
        is None
    )


def test_rejects_wrong_plugin_or_signature():
    with pytest.raises(InvalidAeolusWebhook, match="plugin"):
        parse_start_event(
            _payload(), expected_plugin_id="MII_other", webhook_token="webhook-secret"
        )
    with pytest.raises(InvalidAeolusWebhook, match="signature"):
        parse_start_event(
            _payload(), expected_plugin_id="MII_test", webhook_token="wrong"
        )


@pytest.mark.parametrize(
    "value",
    [
        ["https://example.feishu.cn/wiki/wikcnABC"],
        "https://example.feishu.cn/docx/doccnABC",
        '["https://example.feishu.cn/docx/doccnABC"]',
    ],
)
def test_requirement_url_accepts_documented_and_compatible_value_shapes(value):
    payload = _payload()
    payload["data"]["ai_node_info"]["ai_info"]["props"][0]["prop_value"][
        "field"
    ][0]["field_value"] = value
    payload["signature"] = hashlib.sha256(
        f"MII_test{payload['request_time']}webhook-secret".encode()
    ).hexdigest()
    event = parse_start_event(
        payload, expected_plugin_id="MII_test", webhook_token="webhook-secret"
    )
    assert event is not None
    assert extract_requirement_url(event).startswith("https://")


def test_missing_design_doc_is_a_validation_error():
    payload = _payload()
    payload["data"]["ai_node_info"]["ai_info"]["props"] = []
    event = parse_start_event(
        payload, expected_plugin_id="MII_test", webhook_token="webhook-secret"
    )
    assert event is not None
    with pytest.raises(ValueError, match="design_doc"):
        extract_requirement_url(event)
