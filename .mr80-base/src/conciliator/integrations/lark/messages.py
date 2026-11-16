import json

from conciliator.integrations.lark.client import LarkClient, LarkError
from conciliator.utils.task_links import task_trace_url


async def send_task_received_message(
    client: LarkClient, *, open_id: str, mr_url: str, request_uuid: str, task_id: int,
) -> str:
    links = [("查看处理进度", task_trace_url(task_id))]
    if mr_url:
        links.append(("查看原始 MR", mr_url))
    return await _send_notification(
        client, open_id=open_id, request_uuid=request_uuid,
        text=f"你发起的解冲突任务 {task_id} 已受理。当前处理状态请查看任务追踪页：",
        links=links,
    )


async def send_all_approved_message(
    client: LarkClient,
    *,
    open_id: str,
    mr_url: str,
    request_uuid: str,
) -> str:
    """Send the final resolver-MR notification as the Conciliator bot."""
    return await _send_notification(
        client,
        open_id=open_id,
        links=[(mr_url, mr_url)],
        request_uuid=request_uuid,
        text="本任务冻结版本中需要人工审批的冲突块已全部 approve，请结合 MR 最新检查结果确认是否合入。MR链接：",
    )


async def send_no_manual_approval_message(
    client: LarkClient, *, open_id: str, mr_url: str, request_uuid: str,
) -> str:
    return await _send_notification(
        client, open_id=open_id, links=[(mr_url, mr_url)], request_uuid=request_uuid,
        text="纯空行冲突已处理，无需人工审批。请结合 MR 最新检查结果确认是否合入。MR链接：",
    )


async def send_awaiting_approval_message(
    client: LarkClient,
    *,
    open_id: str,
    mr_url: str,
    request_uuid: str,
    task_id: int,
    attempt: int,
) -> str:
    return await _send_notification(
        client,
        open_id=open_id,
        links=[("查看冲突并审批", mr_url), ("查看处理进度", task_trace_url(task_id))],
        request_uuid=request_uuid,
        text=(
            f"你发起的解冲突任务 {task_id} 第 {attempt} 轮处理已完成，进入等待审批状态。"
            "请在解决冲突的 MR 中查看冲突块及处理结果，并确认审批进度。MR链接："
        ),
    )


async def _send_notification(
    client: LarkClient,
    *,
    open_id: str,
    links: list[tuple[str, str]],
    request_uuid: str,
    text: str,
) -> str:
    if not open_id.strip():
        raise ValueError("open_id must not be empty")
    if not links or any(not url.startswith("https://") for _, url in links):
        raise ValueError("notification links must be HTTPS URLs")
    if not request_uuid.strip() or len(request_uuid) > 50:
        raise ValueError("request_uuid must contain 1 to 50 characters")

    anchors = [{"tag": "a", "text": label, "href": url} for label, url in links]
    content = {
        "zh_cn": {
            "content": [
                [
                    {
                        "tag": "text",
                        "text": text,
                    },
                    anchors[0],
                ],
                *[[anchor] for anchor in anchors[1:]],
            ]
        }
    }
    data = await client._request(
        "POST",
        "/im/v1/messages",
        params={"receive_id_type": "open_id"},
        json={
            "receive_id": open_id,
            "msg_type": "post",
            "content": json.dumps(content, ensure_ascii=False),
            "uuid": request_uuid,
        },
    )
    message_id = str(data.get("message_id") or "")
    if not message_id:
        raise LarkError("Lark message response has no message_id")
    return message_id
