"""Structured JSON logging with optional Argos StreamLog forwarding.

The production image installs the internal ``bytedlogger`` SDK (see
``byted-requirements.txt``); local and CI environments do not have it. When the
SDK is unavailable, logging falls back to stdout only, so development and tests
need no internal dependency.
"""

import atexit
import contextlib
import logging
import queue
import sys
import threading
from typing import Any

import structlog
from structlog.types import Processor

from conciliator.config import Settings
from conciliator.services.trace_collector import trace_log

# Loggers outside structlog (uvicorn, SQLAlchemy, ...) also pass through this
# chain before JSON rendering. ``trace_log`` is excluded on purpose: it writes
# task history and should only receive application log events.
_FOREIGN_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]

_APP_PROCESSORS: list[Processor] = [*_FOREIGN_PROCESSORS, trace_log]

# Only these logger trees reach Argos. Third-party loggers stay on stdout:
# ``httpx`` logs the full request URL at INFO, including query strings such as
# the Lark document token used by ``integrations/lark/docs.py``.
_ARGOS_LOG_NAMES = ("conciliator", "uvicorn")
# Access logs always carry the query string (/oauth/lark/callback puts the
# authorization code and state there), so they are never forwarded even if a
# custom uvicorn config lets them propagate.
_ARGOS_EXCLUDED_NAMES = ("uvicorn.access",)

_ARGOS_BACKGROUND_STARTED = False
_ARGOS_HANDLER: "_ArgosQueueHandler | None" = None
_CONSOLE_HANDLER: logging.Handler | None = None

# Upper bound for records waiting in the Argos handler. ``StreamLogHandler``
# defaults to 20, and ``BackgroundClient.emit`` flushes the queue on the calling
# thread once it is full -- the socket write then happens in whichever thread
# logged, e.g. a webhook request or the resolve worker. The queue is drained
# every second, so 1024 is far above steady-state throughput while keeping the
# worst-case backlog bounded; beyond it records are dropped, never flushed
# inline. ``_ArgosQueueHandler`` uses the same bound for the queue feeding the
# SDK, so at most two batches of records are ever held in memory.
_ARGOS_QUEUE_SIZE = 1024

# The drain thread waits for records with a timeout so it can notice ``close()``
# without a sentinel that a full queue could reject.
_ARGOS_DRAIN_POLL_SECONDS = 0.1
_ARGOS_DRAIN_JOIN_SECONDS = 2.0

# Drop diagnostics. They are written from the drain thread only: the logging
# caller can be a webhook request or the resolve worker, and ``sys.stderr``
# blocks when the container's stream backs up and raises when it is closed.
_QUEUE_FULL_WARNING = "argos log queue full; dropping new records\n"
_RENDER_FAILED_WARNING = "argos log record could not be rendered; dropping\n"
_HANDOFF_FAILED_WARNING = "argos log handoff failed; dropping records\n"


class _ArgosRecordFilter(logging.Filter):
    """Allow this service and uvicorn, and deliver each record only once.

    ``uvicorn.error`` propagates into ``uvicorn`` (where the Argos handler is
    attached) and, when uvicorn has not configured itself, on to the root
    handler. The marker on the record keeps that path from producing a second
    copy.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if name in _ARGOS_EXCLUDED_NAMES:
            return False
        if not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _ARGOS_LOG_NAMES
        ):
            return False
        if getattr(record, "_argos_forwarded", False):
            return False
        record.__dict__["_argos_forwarded"] = True
        return True


class _ArgosDropGuard:
    """Drop records instead of flushing synchronously when the queue is full.

    Mixed into the SDK handler at runtime because ``bytedlogger`` is optional
    and must not be imported at module load time. ``BackgroundClient.emit``
    calls ``commit()`` as soon as the queue reaches ``queue_size``, and
    ``commit()`` performs the socket write in the calling thread. Checking the
    backlog before delegating keeps every write on the ``backgrounds`` worker;
    when the backlog is full the record is dropped and counted.

    The caller is ``_ArgosQueueHandler``'s drain thread, so the inline flush
    this prevents never lands on a request or worker thread either, and the
    drain thread itself keeps draining instead of waiting on the socket.
    """

    dropped_records = 0

    def emit(self, record: logging.LogRecord) -> None:
        client = getattr(self, "_log_agent_client", None)
        worker = getattr(client, "_worker", None)
        if worker is None:
            # Unknown SDK shape: fall back to the stock behaviour rather than
            # silently dropping every record.
            super().emit(record)  # type: ignore[misc]
            return
        # ``logging.Handler.handle`` already holds this RLock; re-entering keeps
        # the check atomic when ``emit`` is called directly.
        with self.lock:  # type: ignore[attr-defined]
            if len(worker.queue) >= _ARGOS_QUEUE_SIZE:
                self.dropped_records += 1
                if self.dropped_records == 1:
                    sys.stderr.write(_QUEUE_FULL_WARNING)
                    sys.stderr.flush()
                return
            super().emit(record)  # type: ignore[misc]


class _RenderedMessageFormatter(logging.Formatter):
    """Return the message that was rendered before the record was queued.

    Records reach the SDK with the finished JSON document in ``msg``. Running
    the structlog formatter again would wrap that JSON in a second document.
    """

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class _ArgosQueueHandler(logging.Handler):
    """Hand records to the Argos SDK from a dedicated thread.

    ``StreamLogHandler.emit`` appends to the SDK queue and then emits a
    ``bytedance.metrics`` counter, both on the calling thread; the socket writes
    behind those calls can block. ``emit`` therefore renders the record, puts it
    on a bounded queue, and returns, so a stalled log agent or metrics transport
    cannot delay a webhook, the resolve worker or any response path.

    The queue is bounded and never waited on: when it is full the record is
    dropped and counted in ``dropped_records``.

    Delivery is best effort. Logging never decides whether a webhook or a resolve
    run succeeds, so rendering, queueing and thread startup all absorb their own
    failures instead of raising into the caller.
    """

    def __init__(self, inner: logging.Handler) -> None:
        super().__init__()
        self._inner = inner
        self._queue: queue.Queue[logging.LogRecord] = queue.Queue(
            maxsize=_ARGOS_QUEUE_SIZE
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped_records = 0
        self._warned = False
        self._pending_warning: str | None = None
        # Guards ``_pending_warning`` / ``_warned``: a logging caller records the
        # drop while the drain thread (or the shutdown ``flush()``) reports it.
        self._warning_lock = threading.Lock()
        # Registered after the SDK handler exists, so it runs before the SDK's
        # own ``atexit`` commit and the records still in this queue reach it.
        atexit.register(self.flush)

    def configure(
        self,
        *,
        level: int,
        formatter: logging.Formatter,
        tags: dict[str, str],
    ) -> None:
        """Apply the settings a repeated ``setup_logging()`` call may change."""

        self.setLevel(level)
        self.setFormatter(formatter)
        self.filters = [_ArgosRecordFilter()]
        self._inner.setLevel(level)
        self._inner.setFormatter(_RenderedMessageFormatter())
        if hasattr(self._inner, "tags"):
            self._inner.tags = tags

    def emit(self, record: logging.LogRecord) -> None:
        """Hand the record to the drain thread without ever failing the caller.

        Every failure below is absorbed: the record is counted in
        ``dropped_records`` and the diagnostic is deferred to ``_drain``, because
        writing to ``sys.stderr`` from here can block on backpressure or raise.
        """

        try:
            self._hand_off(record)
        except Exception:
            self._drop(_HANDOFF_FAILED_WARNING)

    def _hand_off(self, record: logging.LogRecord) -> None:
        try:
            rendered = self.format(record)
        except Exception:
            # ``handleError`` prints the traceback to ``sys.stderr`` on this
            # thread, which is exactly what the drain thread exists to avoid.
            self._drop(_RENDER_FAILED_WARNING)
            return
        queued = logging.makeLogRecord(dict(record.__dict__))
        queued.msg = rendered
        queued.args = ()
        queued.exc_info = None
        queued.exc_text = None
        queued.__dict__.pop("_from_structlog", None)
        queued.__dict__.pop("_record", None)
        try:
            self._queue.put_nowait(queued)
        except queue.Full:
            self._drop(_QUEUE_FULL_WARNING)
            return
        with contextlib.suppress(Exception):
            # A failed start keeps the record queued: the next emit retries it and
            # ``flush()`` delivers it at shutdown.
            self._ensure_drain_thread()

    def _drop(self, warning: str) -> None:
        """Count a record that never reached the SDK, and note why once."""

        self.dropped_records += 1
        with self._warning_lock:
            if self._pending_warning is None:
                self._pending_warning = warning

    def flush(self) -> None:
        """Deliver queued records synchronously (shutdown, ``close()``)."""

        # A pending drop diagnostic also gets its chance here, so a process whose
        # drain thread never started still reports the drops it accumulated.
        self._report_pending_warning()
        while True:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._inner.emit(record)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_ARGOS_DRAIN_JOIN_SECONDS)
        self.flush()
        # The SDK handler is deliberately left open: its ``close()`` cannot
        # unregister the ``backgrounds`` job created by ``background=True``, and
        # the instance is reused by the next ``setup_logging()`` call.
        super().close()

    def _ensure_drain_thread(self) -> None:
        """Start the drain thread if it is missing or was stopped by close()."""

        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(
            target=self._drain, name="argos-log-drain", daemon=True
        )
        self._thread = thread
        thread.start()

    def _drain(self) -> None:
        while True:
            self._report_pending_warning()
            try:
                record = self._queue.get(timeout=_ARGOS_DRAIN_POLL_SECONDS)
            except queue.Empty:
                if self._stop.is_set():
                    self._report_pending_warning()
                    return
                continue
            try:
                self._inner.emit(record)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def _report_pending_warning(self) -> None:
        """Write the drop diagnostic from the drain thread, best effort.

        It runs here rather than in ``emit`` so that a blocked or closed
        ``sys.stderr`` cannot affect a webhook request or the resolve worker.
        """

        with self._warning_lock:
            warning = self._pending_warning
            if warning is None or self._warned:
                return
            self._warned = True
            self._pending_warning = None
        # Outside the lock: the write can block on container stderr, and no
        # logging caller should inherit that wait.
        with contextlib.suppress(Exception):
            sys.stderr.write(warning)


def setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            *_APP_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=_FOREIGN_PROCESSORS,
    )
    root = logging.getLogger()
    _install_console_handler(root, level, formatter)

    _install_argos_handler(settings, level, formatter)


def _install_console_handler(
    root: logging.Logger, level: int, formatter: logging.Formatter
) -> None:
    """Replace only the handler this module installed.

    ``setup_logging`` runs again in tests and under ``--reload``; clearing
    ``root.handlers`` would drop handlers installed by the host (pytest
    ``caplog``, platform SDKs) as well.
    """

    global _CONSOLE_HANDLER
    if _CONSOLE_HANDLER is not None:
        root.removeHandler(_CONSOLE_HANDLER)
        _CONSOLE_HANDLER.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    _CONSOLE_HANDLER = console
    root.setLevel(level)
    root.addHandler(console)


def _install_argos_handler(
    settings: Settings, level: int, formatter: logging.Formatter
) -> None:
    """Attach the Argos handler, reusing the instance created earlier.

    ``StreamLogHandler(background=True)`` registers a periodic callback with
    ``backgrounds`` from its constructor, while ``close()`` only flushes the
    queue and ``backgrounds`` exposes no unregister API. Rebuilding the handler
    on every ``setup_logging()`` would therefore leave one scheduler job and one
    worker queue behind per call, so an existing handler is reconfigured in
    place instead of being closed and recreated.
    """

    global _ARGOS_HANDLER
    if not settings.argos_log_enabled:
        _detach_argos_handler()
        return

    handler = _ARGOS_HANDLER
    if handler is None:
        handler = _build_argos_handler()
        if handler is None:
            _detach_argos_handler()
            return
        _ARGOS_HANDLER = handler
    handler.configure(
        level=level,
        formatter=formatter,
        # TCE injects TCE_PSM, but pin the routing tags to this service so a
        # missing or incorrect environment value cannot misroute logs.
        tags={"_psm": settings.service_psm, "_taskName": settings.service_psm},
    )

    root = logging.getLogger()
    if handler not in root.handlers:
        root.addHandler(handler)
    # Uvicorn logs server errors through ``uvicorn.error``, which propagates to
    # ``uvicorn`` but never to root (``uvicorn`` has ``propagate=False``), so
    # the handler has to sit on ``uvicorn`` itself. If uvicorn's own config is
    # absent the record also reaches root; ``_ArgosRecordFilter`` drops that
    # second copy.
    uvicorn_logger = logging.getLogger("uvicorn")
    if handler not in uvicorn_logger.handlers:
        uvicorn_logger.addHandler(handler)


def _build_argos_handler() -> _ArgosQueueHandler | None:
    """Create an Argos handler that feeds the SDK from its own thread.

    ``bytedlogger`` only exists in the production image. Any import or
    initialization failure degrades to stdout logging instead of failing the
    service at startup.
    """

    try:
        import backgrounds
        from bytedlogger import StreamLogHandler
    except Exception:
        return None

    global _ARGOS_BACKGROUND_STARTED
    try:
        if not _ARGOS_BACKGROUND_STARTED:
            # StreamLogHandler(background=True) only flushes its queue after the
            # scheduler starts. Uvicorn/FastAPI, unlike Euler, does not start it.
            backgrounds.start()
            _ARGOS_BACKGROUND_STARTED = True
        handler_type: Any = type(
            "_GuardedStreamLogHandler", (_ArgosDropGuard, StreamLogHandler), {}
        )
        inner = handler_type(
            background=True,
            client_queue_size=_ARGOS_QUEUE_SIZE,
        )
        return _ArgosQueueHandler(inner)
    except Exception:
        return None


def _detach_argos_handler() -> None:
    """Detach the handler without closing it.

    Closing cannot unregister the ``backgrounds`` job created by
    ``StreamLogHandler(background=True)``, so the instance is kept around and
    reused once logging is enabled again.
    """

    handler = _ARGOS_HANDLER
    if handler is None:
        return
    for logger in (logging.getLogger(), logging.getLogger("uvicorn")):
        logger.removeHandler(handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
