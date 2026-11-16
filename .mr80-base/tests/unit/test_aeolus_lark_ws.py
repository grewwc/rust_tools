import asyncio
import base64
import json
from unittest.mock import AsyncMock

import pytest
from lark_oapi.ws.const import HEADER_MESSAGE_ID, HEADER_SEQ, HEADER_SUM, HEADER_TYPE
from lark_oapi.ws.pb.pbbp2_pb2 import Frame

from conciliator.workers.aeolus_lark_ws import AeolusLarkWS, _CardClient


def _frame(payload, *, kind="card", count=1, seq=0):
    frame = Frame(SeqID=1, LogID=1, service=1, method=1)
    frame.payload = payload
    for key, value in (
        (HEADER_TYPE, kind), (HEADER_SUM, str(count)),
        (HEADER_SEQ, str(seq)), (HEADER_MESSAGE_ID, "message"),
    ):
        header = frame.headers.add()
        header.key, header.value = key, value
    return frame


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["card", "event"])
async def test_callback_frames_dispatch_and_ack(kind):
    handler = AsyncMock(return_value={"toast": {"type": "success", "content": "已接收"}})
    client = _CardClient("app", "secret", handler, asyncio.get_running_loop())
    client._write_message = AsyncMock()
    await client._handle_data_frame(_frame(b'{"event":{}}', kind=kind))
    handler.assert_awaited_once_with({"event": {}})
    ack = Frame.FromString(client._write_message.call_args.args[0])
    result = json.loads(ack.payload)
    assert result["code"] == 200
    assert json.loads(base64.b64decode(result["data"]))["toast"]["content"] == "已接收"
    assert next(h.value for h in ack.headers if h.key == HEADER_TYPE) == kind
    assert "secret" not in client._fmt_log("connected to {}", "secret-url")


@pytest.mark.asyncio
async def test_fragmented_card_waits_for_complete_payload():
    handler = AsyncMock(return_value={})
    client = _CardClient("app", "secret", handler, asyncio.get_running_loop())
    client._write_message = AsyncMock()
    await client._handle_data_frame(_frame(b'{"event":', count=2, seq=0))
    handler.assert_not_awaited()
    client._write_message.assert_not_awaited()
    await client._handle_data_frame(_frame(b'{}}', count=2, seq=1))
    handler.assert_awaited_once_with({"event": {}})


@pytest.mark.asyncio
async def test_callback_failure_sends_error_ack():
    client = _CardClient("app", "secret", AsyncMock(side_effect=ValueError("private")), asyncio.get_running_loop())
    client._write_message = AsyncMock()
    await client._handle_data_frame(_frame(b"{}"))
    result = json.loads(Frame.FromString(client._write_message.call_args.args[0]).payload)
    assert result["code"] == 500
    assert "private" not in json.dumps(result)


@pytest.mark.asyncio
async def test_worker_stops_its_sdk_thread(monkeypatch):
    from conciliator.workers.aeolus_lark_ws import sdk

    def start(client):
        sdk.loop.run_until_complete(asyncio.Event().wait())

    monkeypatch.setattr(_CardClient, "start", start)
    worker = AeolusLarkWS(app_id="app", app_secret="secret", cards=AsyncMock(), service=AsyncMock())
    worker.start()
    await worker.stop()
    assert not worker.thread.is_alive()
    assert worker.loop.is_closed()
