import asyncio
import json
import logging
import sys
import threading
import time
import types
from unittest.mock import AsyncMock

from conciliator.config import Settings
from conciliator.utils import logging as logging_utils
from conciliator.utils.logging import get_logger, setup_logging
from conciliator.utils.trace_context import context, trace_scope


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    """Poll ``predicate`` until it holds or the deadline passes."""

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return predicate()
        time.sleep(0.005)
    return True


def _detach_test_handlers() -> None:
    """Detach Argos wrappers and SDK fakes left behind by an earlier test.

    ``monkeypatch`` restores the module global, so the wrapper has to be found on
    the loggers themselves; a leftover wrapper keeps its ``_argos_forwarded``
    mark on every later record and starves the handler under test.
    """

    for logger in (logging.getLogger(), logging.getLogger("uvicorn")):
        for handler in list(logger.handlers):
            if isinstance(handler, logging_utils._ArgosQueueHandler) or (
                type(handler).__module__ == __name__
            ):
                logger.removeHandler(handler)


class _FakeStreamLogHandler(logging.Handler):
    instances: list["_FakeStreamLogHandler"] = []

    def __init__(
        self, level=logging.NOTSET, tags=None, background=False,
        client_queue_size=20, **kwargs,
    ):
        super().__init__(level)
        self.tags = tags or {}
        self.background = background
        self.client_queue_size = client_queue_size
        self.messages: list[str] = []
        self.closed = False
        self.instances.append(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))

    def close(self) -> None:
        self.closed = True
        super().close()


def _make_settings(**overrides) -> Settings:
    base = dict(
        service_psm="x",
        nextcode_app_id="x", coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="x", mysql_psm="x",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _install_fake_argos(monkeypatch) -> list[bool]:
    started: list[bool] = []
    backgrounds = types.ModuleType("backgrounds")
    backgrounds.start = lambda: started.append(True)  # type: ignore[attr-defined]
    bytedlogger = types.ModuleType("bytedlogger")
    bytedlogger.StreamLogHandler = _FakeStreamLogHandler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "backgrounds", backgrounds)
    monkeypatch.setitem(sys.modules, "bytedlogger", bytedlogger)
    monkeypatch.setattr(logging_utils, "_ARGOS_BACKGROUND_STARTED", False)
    # The handler is reused across setup_logging() calls, so each test starts
    # from a process without one.
    monkeypatch.setattr(logging_utils, "_ARGOS_HANDLER", None)
    _detach_test_handlers()
    _FakeStreamLogHandler.instances.clear()
    return started


def test_json_log_output(capsys):
    # Disabled so the SDK, when installed, cannot open the ttlogagent socket.
    setup_logging(_make_settings(log_level="INFO", argos_log_enabled=False))
    log = get_logger("test")
    log.info("hello", task_id="t1", stage="started")
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["event"] == "hello"
    assert parsed["task_id"] == "t1"
    assert parsed["stage"] == "started"
    assert parsed["level"] == "info"


def test_argos_handler_forwards_json(monkeypatch, capsys):
    started = _install_fake_argos(monkeypatch)
    setup_logging(_make_settings(argos_log_enabled=True))

    get_logger("conciliator.test").info("hello", task_id="t1", stage="started")

    assert started == [True]
    handler = _FakeStreamLogHandler.instances[-1]
    assert handler.tags == {"_psm": "x", "_taskName": "x"}
    assert handler.background is True
    # The SDK handler is reached from the drain thread; loggers hold the wrapper.
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None
    assert _wait_for(lambda: len(handler.messages) == 1)
    uvicorn_logger = logging.getLogger("uvicorn")
    assert wrapper in uvicorn_logger.handlers
    # uvicorn.error propagates into uvicorn; a second attachment would duplicate.
    assert wrapper not in logging.getLogger("uvicorn.error").handlers
    # Access logs contain query strings (OAuth code/state), so they stay local.
    access_logger = logging.getLogger("uvicorn.access")
    assert wrapper not in access_logger.handlers
    payload = json.loads(handler.messages[-1])
    assert payload["event"] == "hello"
    assert payload["task_id"] == "t1"
    assert payload["level"] == "info"
    assert json.loads(capsys.readouterr().out)["event"] == "hello"

    # Production shape: uvicorn keeps propagate=False, uvicorn.error does not.
    monkeypatch.setattr(uvicorn_logger, "propagate", False)
    before = len(handler.messages)
    logging.getLogger("uvicorn.error").warning("uvicorn started")
    assert _wait_for(lambda: len(handler.messages) == before + 1)
    assert json.loads(handler.messages[-1])["event"] == "uvicorn started"

    before = len(handler.messages)
    access_logger.info("GET /oauth/lark/callback?code=secret")
    wrapper.flush()
    assert len(handler.messages) == before


def test_uvicorn_error_delivered_once_without_uvicorn_config(monkeypatch):
    """A record that also reaches root must not be forwarded twice."""

    _install_fake_argos(monkeypatch)
    setup_logging(_make_settings())

    handler = _FakeStreamLogHandler.instances[-1]
    # Force the shape uvicorn has before its own LOGGING_CONFIG is applied.
    monkeypatch.setattr(logging.getLogger("uvicorn"), "propagate", True)
    seen_at_root: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen_at_root.append(record.getMessage())

    capture = _Capture()
    root = logging.getLogger()
    root.addHandler(capture)
    try:
        before = len(handler.messages)
        logging.getLogger("uvicorn.error").warning("server exploded")
    finally:
        root.removeHandler(capture)

    assert seen_at_root == ["server exploded"]
    assert _wait_for(lambda: len(handler.messages) == before + 1)


def test_third_party_loggers_not_forwarded(monkeypatch):
    _install_fake_argos(monkeypatch)
    setup_logging(_make_settings())

    handler = _FakeStreamLogHandler.instances[-1]
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None
    before = len(handler.messages)
    logging.getLogger("httpx").info(
        "HTTP Request: GET https://open.feishu.cn/open-apis/docx/v1/documents/a?token=SECRET"
    )
    # Deterministic: anything the filter let through would be flushed here.
    wrapper.flush()

    assert len(handler.messages) == before


def test_argos_disabled_skips_sdk(monkeypatch):
    started = _install_fake_argos(monkeypatch)

    setup_logging(_make_settings(argos_log_enabled=False))

    assert started == []
    assert _FakeStreamLogHandler.instances == []


def test_missing_argos_sdk_falls_back_to_stdout(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "bytedlogger", None)
    monkeypatch.setattr(logging_utils, "_ARGOS_BACKGROUND_STARTED", False)

    setup_logging(_make_settings())
    get_logger("test").info("hello")

    assert json.loads(capsys.readouterr().out)["event"] == "hello"


def test_repeated_setup_reuses_argos_handler(monkeypatch):
    # StreamLogHandler(background=True) registers a periodic callback with
    # ``backgrounds`` in its constructor, and the SDK never unregisters it.
    # A second setup must therefore reconfigure the existing handler instead of
    # creating another one, otherwise every call leaks a scheduler job.
    _install_fake_argos(monkeypatch)
    setup_logging(_make_settings())
    first = _FakeStreamLogHandler.instances[-1]

    setup_logging(_make_settings(service_psm="y", log_level="DEBUG"))

    assert _FakeStreamLogHandler.instances == [first]
    assert first.closed is False
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None
    assert logging.getLogger().handlers.count(wrapper) == 1
    assert logging.getLogger("uvicorn").handlers.count(wrapper) == 1
    assert first.level == logging.DEBUG
    assert first.tags["_psm"] == "y"


def test_disabling_argos_detaches_handler_without_closing(monkeypatch):
    _install_fake_argos(monkeypatch)
    setup_logging(_make_settings())
    handler = _FakeStreamLogHandler.instances[-1]
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    setup_logging(_make_settings(argos_log_enabled=False))

    assert wrapper not in logging.getLogger().handlers
    assert wrapper not in logging.getLogger("uvicorn").handlers
    # Closing cannot unregister the backgrounds job, so the handler stays
    # reusable for a later setup_logging() call.
    assert handler.closed is False
    assert logging_utils._ARGOS_HANDLER is wrapper


def test_external_root_handlers_preserved(monkeypatch):
    _install_fake_argos(monkeypatch)
    marker = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(marker)
    try:
        setup_logging(_make_settings())
        assert marker in root.handlers
    finally:
        root.removeHandler(marker)


def test_trace_id_and_task_id_added_to_log_fields(monkeypatch, capsys):
    _install_fake_argos(monkeypatch)
    setup_logging(_make_settings())

    with trace_scope("t1"):
        trace_id = context.get()["trace_id"]
        get_logger("conciliator.test").info("scoped")

    parsed = json.loads(capsys.readouterr().out)
    assert parsed["trace_id"] == trace_id
    assert parsed["task_id"] == "t1"


class _QueueWorker:
    def __init__(self) -> None:
        self.queue: list[tuple[str, dict]] = []


class _QueueAwareStreamLogHandler(logging.Handler):
    """Reproduces the queueing contract of bytedlogger 0.15.2.

    ``BackgroundClient.emit`` calls ``commit()`` on the calling thread as soon
    as ``len(queue) >= queue_size``, and ``commit()`` performs the socket
    write. That synchronous flush is what must never reach a business thread.
    """

    instances: list["_QueueAwareStreamLogHandler"] = []

    def __init__(
        self, level=logging.NOTSET, tags=None, end="\n", timeout=0.2,
        socket_send_buf=None, background=False, client_queue_size=20, version=1,
    ):
        super().__init__(level)
        self.tags = tags or {}
        self.background = background
        self._log_agent_client = types.SimpleNamespace(
            _worker=_QueueWorker(), _queue_size=client_queue_size
        )
        self.commit_calls = 0
        self.instances.append(self)

    def emit(self, record: logging.LogRecord) -> None:
        worker = self._log_agent_client._worker
        if len(worker.queue) >= self._log_agent_client._queue_size:
            self.commit()
        worker.queue.append((self.format(record), dict(self.tags)))

    def commit(self) -> None:
        self.commit_calls += 1
        time.sleep(0.05)  # a stalled ttlogagent socket write


def _install_sdk(monkeypatch, handler_type) -> None:
    """Install ``handler_type`` as ``bytedlogger.StreamLogHandler``."""

    backgrounds = types.ModuleType("backgrounds")
    backgrounds.start = lambda: None  # type: ignore[attr-defined]
    bytedlogger = types.ModuleType("bytedlogger")
    bytedlogger.StreamLogHandler = handler_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "backgrounds", backgrounds)
    monkeypatch.setitem(sys.modules, "bytedlogger", bytedlogger)
    monkeypatch.setattr(logging_utils, "_ARGOS_BACKGROUND_STARTED", False)
    monkeypatch.setattr(logging_utils, "_ARGOS_HANDLER", None)
    _detach_test_handlers()
    handler_type.instances.clear()


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        "conciliator.test", logging.INFO, __file__, 1, message, (), None
    )


def _build_argos_wrapper(handler_type):
    """Build and configure the wrapper the way ``setup_logging`` does."""

    wrapper = logging_utils._build_argos_handler()
    assert isinstance(wrapper, logging_utils._ArgosQueueHandler)
    wrapper.configure(
        level=logging.INFO,
        formatter=logging.Formatter("%(message)s"),
        tags={"_psm": "x", "_taskName": "x"},
    )
    assert isinstance(wrapper._inner, handler_type)
    return wrapper


def test_argos_drop_guard_skips_inline_commit(monkeypatch, capsys):
    """The SDK's full-queue flush must never run on the logging thread."""

    _install_sdk(monkeypatch, _QueueAwareStreamLogHandler)
    handler = _build_argos_wrapper(_QueueAwareStreamLogHandler)._inner
    client = handler._log_agent_client
    client._worker.queue.extend([("", {})] * logging_utils._ARGOS_QUEUE_SIZE)
    # The SDK threshold must match the guard, otherwise its own full-queue
    # branch could still fire before the guard sees the backlog.
    assert client._queue_size == logging_utils._ARGOS_QUEUE_SIZE

    started = time.monotonic()
    for index in range(50):
        handler.handle(_record(f"burst-{index}"))
    elapsed = time.monotonic() - started

    assert handler.commit_calls == 0
    assert handler.dropped_records == 50
    assert len(client._worker.queue) == logging_utils._ARGOS_QUEUE_SIZE
    assert elapsed < 0.2
    assert "argos log queue full" in capsys.readouterr().err


_SLOW_SDK_SECONDS = 0.05


class _SlowStreamLogHandler(logging.Handler):
    """Model the SDK, whose ``emit`` also submits a metrics counter inline."""

    instances: list["_SlowStreamLogHandler"] = []

    def __init__(
        self, level=logging.NOTSET, tags=None, background=False,
        client_queue_size=20, **kwargs,
    ):
        super().__init__(level)
        self.tags = tags or {}
        self.background = background
        self.client_queue_size = client_queue_size
        self.messages: list[str] = []
        self.emit_threads: list[str] = []
        self.instances.append(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.emit_threads.append(threading.current_thread().name)
        time.sleep(_SLOW_SDK_SECONDS)
        self.messages.append(self.format(record))


class _BlockingStreamLogHandler(logging.Handler):
    """Park ``emit`` until the test releases it, modelling a stalled agent."""

    instances: list["_BlockingStreamLogHandler"] = []

    def __init__(
        self, level=logging.NOTSET, tags=None, background=False,
        client_queue_size=20, **kwargs,
    ):
        super().__init__(level)
        self.tags = tags or {}
        self.background = background
        self.client_queue_size = client_queue_size
        self.messages: list[str] = []
        self.emit_threads: list[str] = []
        self.gate = threading.Event()
        self.instances.append(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.emit_threads.append(threading.current_thread().name)
        self.messages.append(self.format(record))
        self.gate.wait(5.0)


def test_argos_records_reach_the_sdk_from_the_drain_thread(monkeypatch):
    """A slow SDK must not hold the thread that logged."""

    _install_sdk(monkeypatch, _SlowStreamLogHandler)
    setup_logging(_make_settings())
    inner = _SlowStreamLogHandler.instances[-1]

    started = time.monotonic()
    for index in range(5):
        logging.getLogger("conciliator.test").info("burst-%d", index)
    elapsed = time.monotonic() - started

    # Five inline SDK calls cost 5 * _SLOW_SDK_SECONDS; logging returns at once.
    assert elapsed < _SLOW_SDK_SECONDS
    assert _wait_for(lambda: len(inner.messages) == 5)
    assert inner.emit_threads == ["argos-log-drain"] * 5


def test_argos_wrapper_queue_full_drops_instead_of_blocking(monkeypatch, capsys):
    """The queue in front of the SDK is bounded and never waited on."""

    _install_sdk(monkeypatch, _BlockingStreamLogHandler)
    setup_logging(_make_settings())
    inner = _BlockingStreamLogHandler.instances[-1]
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    # Park the drain thread inside the SDK, then fill the queue behind it.
    wrapper.handle(_record("first"))
    assert _wait_for(lambda: inner.emit_threads == ["argos-log-drain"])

    started = time.monotonic()
    for index in range(logging_utils._ARGOS_QUEUE_SIZE + 50):
        wrapper.handle(_record(f"burst-{index}"))
    elapsed = time.monotonic() - started

    # A synchronous emit would have waited on the gate for every record.
    assert elapsed < 0.5
    assert wrapper.dropped_records == 50

    # The diagnostic belongs to the drain thread, so nothing reaches stderr while
    # the caller is dropping records.
    assert "argos log queue full" not in capsys.readouterr().err
    inner.gate.set()
    assert _wait_for(lambda: "argos log queue full" in capsys.readouterr().err)


def test_drain_thread_start_failure_does_not_reach_the_caller(monkeypatch):
    """A failed async handoff must not raise into the logging caller.

    ``Thread.start()`` raises ``RuntimeError`` when the process is out of threads
    or already shutting down; the resolve worker and the webhook handlers log
    while holding claimed work, so that failure has to stay inside the handler.
    """

    _install_sdk(monkeypatch, _SlowStreamLogHandler)
    setup_logging(_make_settings())
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    def _failing_start(self, *args, **kwargs):
        raise RuntimeError("cannot start new thread")

    with monkeypatch.context() as failing_start:
        failing_start.setattr(threading.Thread, "start", _failing_start)
        wrapper.handle(_record("no thread"))  # must not raise

    # The record is kept for a later start or for flush() at shutdown, so it is
    # waiting, not dropped.
    assert wrapper.dropped_records == 0
    assert wrapper._queue.qsize() == 1


def test_unwritable_stderr_does_not_reach_the_caller(monkeypatch, capsys):
    """Drop diagnostics never write to the caller's stderr, so they cannot raise."""

    _install_sdk(monkeypatch, _BlockingStreamLogHandler)
    setup_logging(_make_settings())
    inner = _BlockingStreamLogHandler.instances[-1]
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    wrapper.handle(_record("first"))
    assert _wait_for(lambda: inner.emit_threads == ["argos-log-drain"])

    class _UnwritableStream:
        def write(self, text: str) -> int:
            raise OSError("stderr unavailable")

        def flush(self) -> None:
            raise OSError("stderr unavailable")

        def isatty(self) -> bool:
            return False

    with monkeypatch.context() as broken_stderr:
        broken_stderr.setattr(sys, "stderr", _UnwritableStream())
        for index in range(logging_utils._ARGOS_QUEUE_SIZE + 10):
            wrapper.handle(_record(f"burst-{index}"))  # must not raise

    assert wrapper.dropped_records == 10

    # The drain thread still reports the drops once stderr works again.
    inner.gate.set()
    assert _wait_for(lambda: "argos log queue full" in capsys.readouterr().err)


def test_render_failure_is_counted_without_printing_on_the_caller(monkeypatch, capsys):
    """A record the formatter rejects is dropped, not raised or printed here."""

    _install_sdk(monkeypatch, _SlowStreamLogHandler)
    setup_logging(_make_settings())
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    def _explode(record: logging.LogRecord) -> str:
        raise ValueError("cannot render this record")

    with monkeypatch.context() as broken_format:
        broken_format.setattr(wrapper, "format", _explode)
        wrapper.handle(_record("unrenderable"))  # must not raise

    assert wrapper.dropped_records == 1
    # ``handleError`` used to write the traceback to the calling thread's stderr.
    assert "cannot render this record" not in capsys.readouterr().err
    wrapper._pending_warning = None  # keep the shutdown flush quiet


def test_unexpected_handoff_failure_is_counted(monkeypatch, capsys):
    """Whatever is left unhandled in the hand-off is absorbed and counted."""

    _install_sdk(monkeypatch, _SlowStreamLogHandler)
    setup_logging(_make_settings())
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    class _UnavailableQueue:
        def put_nowait(self, item: logging.LogRecord) -> None:
            raise RuntimeError("queue unavailable")

    with monkeypatch.context() as broken_queue:
        broken_queue.setattr(wrapper, "_queue", _UnavailableQueue())
        wrapper.handle(_record("no queue"))  # must not raise

    assert wrapper.dropped_records == 1
    assert wrapper._pending_warning == logging_utils._HANDOFF_FAILED_WARNING
    assert "Traceback" not in capsys.readouterr().err
    wrapper._pending_warning = None  # keep the shutdown flush quiet


def test_flush_reports_drops_the_drain_thread_never_reported(monkeypatch, capsys):
    """Drops are still reported when no drain thread ever started."""

    _install_sdk(monkeypatch, _FakeStreamLogHandler)
    setup_logging(_make_settings())
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    # Fill the queue without starting the drain thread, then overflow it: the
    # diagnostic is pending and no thread is around to write it.
    for index in range(logging_utils._ARGOS_QUEUE_SIZE):
        wrapper._queue.put_nowait(_record(f"prefill-{index}"))
    wrapper.handle(_record("overflow"))

    assert wrapper.dropped_records == 1
    assert "argos log queue full" not in capsys.readouterr().err

    wrapper.flush()

    assert "argos log queue full" in capsys.readouterr().err


def test_close_delivers_records_the_drain_thread_has_not_taken(monkeypatch):
    _install_sdk(monkeypatch, _SlowStreamLogHandler)
    setup_logging(_make_settings())
    inner = _SlowStreamLogHandler.instances[-1]
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None

    # The slow SDK still has records to work through when close() runs, so they
    # have to survive shutdown instead of being dropped with the thread.
    for index in range(5):
        wrapper.handle(_record(f"burst-{index}"))
    wrapper.close()

    assert len(inner.messages) == 5


def test_handler_starts_a_new_drain_thread_after_close(monkeypatch):
    """``logging.shutdown()`` may close the handler before the last log line."""

    _install_sdk(monkeypatch, _SlowStreamLogHandler)
    setup_logging(_make_settings())
    inner = _SlowStreamLogHandler.instances[-1]
    wrapper = logging_utils._ARGOS_HANDLER
    assert wrapper is not None
    wrapper.close()

    wrapper.handle(_record("after close"))

    assert _wait_for(lambda: len(inner.messages) == 1)
    assert inner.emit_threads == ["argos-log-drain"]


def test_resolve_stage_json_reaches_argos_with_task_identity(monkeypatch):
    from conciliator.services.resolve_service import ResolveService

    _install_fake_argos(monkeypatch)
    setup_logging(_make_settings())
    handler = _FakeStreamLogHandler.instances[-1]
    task = types.SimpleNamespace(
        id=7, status="BRANCH_CREATED", version=0,
        conflict_files_snapshot=[{
            "file_path": "a.py", "first_path": "a.py", "second_path": "a.py", "conflicts_id": "c",
            "content": "<<<<<<< source\nprivate_source()\n=======\nprivate_target()\n>>>>>>> target\n",
        }],
    )
    repo = types.SimpleNamespace(
        save_conflict_blocks=AsyncMock(), update_status=AsyncMock(return_value=True), add_event=AsyncMock(),
    )
    service = ResolveService(types.SimpleNamespace(repo=repo))
    trace_ids = []
    # Separate executions (as for worker and poller) join by task_id, not trace_id.
    for _ in range(2):
        task.status = "BRANCH_CREATED"
        with trace_scope(task.id):
            trace_ids.append(context.get()["trace_id"])
            asyncio.run(service._step2_identify_conflicts(task))
    wrapper = logging_utils._ARGOS_HANDLER
    wrapper.flush()
    assert _wait_for(lambda: len(handler.messages) >= 8)
    records = [json.loads(message) for message in handler.messages]
    assert all(record["task_id"] == 7 for record in records)
    assert {record["trace_id"] for record in records} == set(trace_ids)
    assert len(set(trace_ids)) == 2
    for event in ("resolve_stage_started", "resolve_stage_finished", "resolve_status_changed", "resolve_milestone"):
        assert len([record for record in records if record["event"] == event]) == 2
    assert "private_source" not in str(records) and "private_target" not in str(records)
