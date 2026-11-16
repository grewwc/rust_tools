"""SDK transport adapter for card callbacks; task operations stay on the app loop."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time

from lark_oapi.core.json import JSON
from lark_oapi.ws import client as sdk
from lark_oapi.ws.const import (
    HEADER_BIZ_RT,
    HEADER_MESSAGE_ID,
    HEADER_SEQ,
    HEADER_SUM,
    HEADER_TYPE,
)
from lark_oapi.ws.enum import MessageType
from lark_oapi.ws.model import Response

from conciliator.services.aeolus_lark_service import InvalidCardCallback
from conciliator.utils.logging import get_logger

log = get_logger(__name__)


class _CardClient(sdk.Client):
    # The installed SDK returns without an ACK for CARD frames.
    # source: https://github.com/larksuite/oapi-sdk-python/issues/126
    # Wire format/fragment assembly/ACK: lark_oapi.ws.client.Client._handle_data_frame
    def __init__(self, app_id, app_secret, handler, app_loop):
        super().__init__(app_id, app_secret)
        self.handler = handler
        self.app_loop = app_loop

    def _fmt_log(self, fmt, *args):
        # SDK log arguments can contain the full connection URL or callback body.
        return "aeolus_lark_ws: " + fmt

    async def _connect(self):
        await super()._connect()
        log.info("aeolus_lark_ws_connected")

    async def _handle_data_frame(self, frame):
        headers = frame.headers
        kind = sdk._get_by_key(headers, HEADER_TYPE)
        if kind not in {MessageType.CARD.value, MessageType.EVENT.value}:
            return
        payload = frame.payload
        count = int(sdk._get_by_key(headers, HEADER_SUM))
        if count > 1:
            payload = self._combine(
                sdk._get_by_key(headers, HEADER_MESSAGE_ID), count,
                int(sdk._get_by_key(headers, HEADER_SEQ)), payload,
            )
            if payload is None:
                return
        started = time.monotonic()
        response = Response(code=200)
        future = None
        try:
            event = json.loads(payload)
            log.info("aeolus_lark_ws_callback_received")
            future = asyncio.run_coroutine_threadsafe(self.handler(event), self.app_loop)
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=2.5)
            response.data = base64.b64encode(json.dumps(result, ensure_ascii=False).encode())
        except Exception as exc:
            if future is not None:
                future.cancel()
            log.warning("aeolus_lark_ws_callback_failed", error_type=type(exc).__name__)
            response = Response(code=500)
        timing = headers.add()
        timing.key = HEADER_BIZ_RT
        timing.value = str(int((time.monotonic() - started) * 1000))
        frame.payload = JSON.marshal(response).encode()
        await self._write_message(frame.SerializeToString())


class AeolusLarkWS:
    def __init__(self, *, app_id, app_secret, cards, service):
        self.cards = cards
        self.service = service
        self.client = _CardClient(app_id, app_secret, self._handle, asyncio.get_running_loop())
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="aeolus-lark-ws", daemon=True)

    async def _handle(self, payload):
        try:
            return await self.cards.handle_event(self.service, payload)
        except InvalidCardCallback as exc:
            log.warning("aeolus_lark_callback_rejected", reason=str(exc))
            return {"toast": {"type": "error", "content": "卡片操作验证失败，请使用最新卡片"}}

    def start(self):
        self.thread.start()

    def _run(self):
        # SDK start() owns a module-level loop. Keep it off the ASGI loop;
        # this app has exactly one SDK WebSocket client per process.
        asyncio.set_event_loop(self.loop)
        sdk.loop = self.loop
        try:
            self.client.start()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("aeolus_lark_ws_stopped", error_type=type(exc).__name__)
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()

    async def stop(self):
        if not self.thread.is_alive():
            return

        async def disconnect():
            self.client._auto_reconnect = False
            await self.client._disconnect()
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()

        future = asyncio.run_coroutine_threadsafe(disconnect(), self.loop)
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=3)
        except (TimeoutError, asyncio.CancelledError):
            future.cancel()
        await asyncio.to_thread(self.thread.join, 3)
        if self.thread.is_alive():
            log.warning("aeolus_lark_ws_shutdown_timeout")
