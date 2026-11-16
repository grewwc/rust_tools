from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from conciliator.config import Settings
from conciliator.integrations.trace import error_event, response_event, start_call, traced_operation, traced_request
from conciliator.services.trace_collector import emit
from conciliator.utils.redaction import coco_identity
from conciliator.utils.retry import TransientError, retry_async
from conciliator.utils.secrets import SecretsProvider


@dataclass(frozen=True)
class CocoError(Exception):
    status: int
    action: str
    message: str
    code: str = ""
    request_id: str = ""

    def __post_init__(self) -> None:
        detail = f"Coco {self.action} failed ({self.status})"
        if self.code:
            detail += f" [{self.code}]"
        detail += f": {self.message[:300]}"
        if self.request_id:
            detail += f" (request_id={self.request_id})"
        Exception.__init__(self, detail)


@dataclass(frozen=True)
class CocoEvent:
    event_id: str
    session_id: str
    run_id: str
    payload: dict[str, Any]
    raw: dict[str, Any]

    @property
    def kind(self) -> str:
        return next(iter(self.payload), "")


@dataclass(frozen=True)
class CocoEventPage:
    events: list[CocoEvent]
    next_after_event_id: str


@dataclass(frozen=True)
class CocoTaskRef:
    # 兼容现有数据库字段名；OpenAPI Next 中的真实资源是 Session，不是 Task。
    task_id: str
    run_id: str = ""
    user_event_id: str = ""

    @property
    def session_id(self) -> str:
        return self.task_id


@dataclass(frozen=True)
class FileDiff:
    file_path: str
    before: str
    after: str


@dataclass(frozen=True)
class ConflictAttribution:
    path: str
    start_line: int
    end_line: int
    commit_sha: str
    author_email: str | None


@dataclass(frozen=True)
class ResolvedConflict:
    path: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class ConflictResolution:
    conflict_id: str
    path: str
    decision_summary: str
    source_attribution: ConflictAttribution
    target_attribution: ConflictAttribution
    resolved: ResolvedConflict


@dataclass(frozen=True)
class CocoResolveResult:
    schema_version: str
    task_id: int
    status: str
    attempt: int
    resolve_branch: str
    commit_sha: str
    merge_parents: tuple[str, ...]
    conflicts: tuple[ConflictResolution, ...]
    unresolved_items: tuple[str, ...]


@dataclass(frozen=True)
class CocoRetryResult:
    schema_version: str
    task_id: int
    status: str
    attempt: int
    resolve_branch: str
    base_commit_sha: str
    commit_sha: str
    parent_commits: tuple[str, ...]
    conflicts: tuple[ConflictResolution, ...]
    unresolved_items: tuple[str, ...]


class CocoResultError(ValueError):
    pass


@dataclass(frozen=True)
class CocoTaskResult:
    status: str
    diffs: list[FileDiff] = field(default_factory=list)
    resolve_result: CocoResolveResult | None = None
    retry_result: CocoRetryResult | None = None
    thinking: str = ""
    final_message: str = ""
    commit_sha: str | None = None
    error: str | None = None
    last_event_id: str = ""
    lifecycle_terminal: bool = False
    # Platform-side failure reported by the runtime (for example request queue
    # saturation). It is kept apart from Agent output problems so callers can
    # retry it as infrastructure trouble instead of diagnosing the model.
    platform_error: str | None = None
    # Tool-use events the runtime is blocked on until somebody answers the
    # confirmation.  A paused run repeats them on every poll.
    requires_action_event_ids: tuple[str, ...] = ()
    # Any Agent-produced event (message, thinking, tool use).  A run that only
    # emitted thinking or tool calls has still been scheduled, so a later stall
    # is not a platform failure.
    agent_activity: bool = False


class CocoClient:
    """Coco OpenAPI Next client using the documented Managed Agent Actions."""

    def __init__(self, settings: Settings, secrets: SecretsProvider):
        self._settings = settings
        self._base = settings.coco_api_base.rstrip("/") + "/"
        self._app_id = settings.nextcode_app_id
        self._app_secret = secrets.get(settings.kms_secret_nextcode_app_secret_alias)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @traced_operation
    async def call(
        self,
        action: str,
        body: dict[str, Any],
        *,
        read_only: bool = False,
    ) -> dict[str, Any]:
        if read_only:
            return await self._call_read(action, body)
        return await self._call_once(action, body)

    @retry_async("coco_request", exceptions=(TransientError,))
    async def _call_read(
        self,
        action: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._call_once(action, body)

    async def _call_once(
        self,
        action: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await traced_request(
                self._http,
                "coco",
                action,
                "POST",
                self._base,
                params={"Action": action},
                json=body,
                headers=self._auth_headers(),
            )
        except httpx.RequestError as exc:
            raise TransientError(f"Coco network error: {exc}") from exc

        if response.status_code >= 500:
            raise TransientError(f"Coco server error {response.status_code}: {response.text[:300]}")
        return self._parse_envelope(action, response)

    def _parse_envelope(
        self,
        action: str,
        response: httpx.Response,
    ) -> dict[str, Any]:
        try:
            envelope = response.json()
        except ValueError as exc:
            raise CocoError(
                status=response.status_code,
                action=action,
                message=response.text,
            ) from exc

        metadata = envelope.get("ResponseMetadata") or {}
        request_id = str(metadata.get("RequestId") or "")
        error = metadata.get("Error") or {}
        if response.status_code >= 400 or error:
            raise CocoError(
                status=response.status_code,
                action=action,
                code=str(error.get("Code") or ""),
                message=str(error.get("Message") or response.text),
                request_id=request_id,
            )

        result = envelope.get("Result")
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise CocoError(
                status=response.status_code,
                action=action,
                message="response Result is not an object",
                request_id=request_id,
            )
        return result

    def _auth_headers(self) -> dict[str, str]:
        # 官方契约区分两类身份：普通用户使用 X-Code-User-JWT，平台服务账号
        # 使用 X-Code-App-Id / X-Code-App-Secret。只有代目标用户调用时才需要
        # SignUserJWT；Conciliator 的长期 Workspace/Agent 由 App 身份持有。
        return {
            "Content-Type": "application/json",
            "X-Code-App-Id": self._app_id,
            "X-Code-App-Secret": self._app_secret,
        }

    async def create_workspace(self, *, name: str, slug: str) -> dict[str, Any]:
        result = await self.call(
            "CreateManagedAgentWorkspace",
            {"Name": name, "Slug": slug},
        )
        return _require_object(result, "Workspace")

    async def list_workspace_devices(
        self,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        result = await self.call(
            "ListManagedAgentWorkspaceDevices",
            {"WorkspaceId": workspace_id},
            read_only=True,
        )
        return _object_list(result, "Devices")

    async def create_agent(
        self,
        *,
        workspace_id: str,
        name: str,
        device_id: str,
        runtime: str = "traex",
        instruction: str = "",
        permission_mode: str,
    ) -> dict[str, Any]:
        profile: dict[str, Any] = {"PermissionMode": permission_mode}
        if instruction:
            profile["Instruction"] = instruction
        result = await self.call(
            "CreateManagedAgent",
            {
                "WorkspaceId": workspace_id,
                "Name": name,
                "DeviceId": device_id,
                "Runtime": runtime,
                "Profile": profile,
            },
        )
        return _require_object(result, "Agent")

    async def create_session(
        self,
        *,
        workspace_id: str,
        agent_id: str,
        repositories: list[tuple[str, str]] | None = None,
        vault_ids: Sequence[str] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a session, optionally bound to ``vault_ids`` credential vaults.

        The platform injects those vaults' credentials into the sandbox, so the
        session's CLIs start authenticated.
        """
        if isinstance(vault_ids, str):
            # A bare string would be iterated character by character below.
            raise TypeError("vault_ids must be a sequence of ids, not str")
        body: dict[str, Any] = {
            "WorkspaceId": workspace_id,
            "AgentId": agent_id,
        }
        if vault_ids:
            body["VaultIds"] = list(vault_ids)
        if metadata:
            # Metadata.title is the platform's session name; custom keys are caller-owned.
            body["Metadata"] = dict(metadata)
        if repositories:
            body["Resources"] = [
                {
                    "Repository": {
                        "RepoId": repo_id,
                        "Revision": revision,
                    }
                }
                for repo_id, revision in repositories
            ]
        result = await self.call("CreateManagedAgentSession", body)
        return _require_object(result, "Session")

    async def send_events(
        self,
        *,
        workspace_id: str,
        session_id: str,
        events: list[dict[str, Any]],
    ) -> list[CocoEvent]:
        result = await self.call(
            "SendManagedAgentSessionEvents",
            {
                "WorkspaceId": workspace_id,
                "SessionId": session_id,
                "Events": events,
            },
        )
        return [_parse_event(item) for item in _object_list(result, "Events")]

    async def send_user_message(
        self,
        *,
        workspace_id: str,
        session_id: str,
        text: str,
    ) -> CocoEvent:
        events = await self.send_events(
            workspace_id=workspace_id,
            session_id=session_id,
            events=[{"Payload": {"UserMessage": {"Content": [{"Text": {"Text": text}}]}}}],
        )
        if not events:
            raise CocoError(
                status=200,
                action="SendManagedAgentSessionEvents",
                message="response does not contain Result.Events[0]",
            )
        return events[0]

    async def send_interrupt(
        self,
        *,
        workspace_id: str,
        session_id: str,
    ) -> list[CocoEvent]:
        return await self.send_events(
            workspace_id=workspace_id,
            session_id=session_id,
            events=[{"Payload": {"UserInterrupt": {}}}],
        )

    async def send_tool_confirmation(
        self,
        *,
        workspace_id: str,
        session_id: str,
        tool_use_event_id: str,
        decision: str,
        deny_message: str = "",
        updated_input: dict[str, Any] | None = None,
    ) -> list[CocoEvent]:
        """Answer a pending tool confirmation.

        The runtime pauses the whole run until this answer arrives, and an
        unattended service has nobody to ask.
        """
        confirmation: dict[str, Any] = {
            "ToolUseEventId": tool_use_event_id,
            "Decision": decision,
        }
        if deny_message:
            confirmation["DenyMessage"] = deny_message
        if updated_input is not None:
            confirmation["UpdatedInputJson"] = json.dumps(
                updated_input, ensure_ascii=False
            )
        return await self.send_events(
            workspace_id=workspace_id,
            session_id=session_id,
            events=[{"Payload": {"UserToolConfirmation": confirmation}}],
        )

    async def list_session_events(
        self,
        *,
        workspace_id: str,
        session_id: str,
        limit: int = 200,
        after_event_id: str = "",
    ) -> list[CocoEvent]:
        page = await self.list_session_event_page(
            workspace_id=workspace_id, session_id=session_id,
            limit=limit, after_event_id=after_event_id,
        )
        return page.events

    async def list_session_event_page(
        self,
        *,
        workspace_id: str,
        session_id: str,
        limit: int = 200,
        after_event_id: str = "",
    ) -> CocoEventPage:
        body: dict[str, Any] = {"WorkspaceId": workspace_id, "SessionId": session_id, "Limit": limit}
        if after_event_id:
            body["AfterEventId"] = after_event_id
        result = await self.call("ListManagedAgentSessionEvents", body, read_only=True)
        return CocoEventPage(
            events=[_parse_event(item) for item in _object_list(result, "Events")],
            next_after_event_id=str(result.get("NextAfterEventId") or ""),
        )

    async def read_round_events(
        self,
        *,
        workspace_id: str,
        session_id: str,
        after_event_id: str = "",
    ) -> list[CocoEvent]:
        """Drain using server cursors, including short pages with a continuation."""
        cursor = after_event_id
        events: list[CocoEvent] = []
        seen = {after_event_id} if after_event_id else set()
        cursors = {cursor}
        for _ in range(100):
            page = await self.list_session_event_page(
                workspace_id=workspace_id,
                session_id=session_id,
                after_event_id=cursor,
            )
            for event in page.events:
                if event.event_id not in seen:
                    events.append(event)
                    seen.add(event.event_id)
            next_cursor = page.next_after_event_id
            if not next_cursor or (not page.events and next_cursor == cursor):
                return events
            if next_cursor in cursors:
                raise CocoError(
                    status=0, action="ListManagedAgentSessionEvents",
                    message="event cursor stalled or cycled",
                )
            cursors.add(next_cursor)
            cursor = next_cursor
        raise CocoError(status=0, action="ListManagedAgentSessionEvents", message="event page limit reached")

    async def list_session_runs(
        self,
        *,
        workspace_id: str,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        result = await self.call(
            "ListManagedAgentSessionRuns",
            {
                "WorkspaceId": workspace_id,
                "SessionId": session_id,
                "Limit": limit,
            },
            read_only=True,
        )
        return _object_list(result, "Runs")

    async def list_session_resources(
        self,
        *,
        workspace_id: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        result = await self.call(
            "ListManagedAgentSessionResources",
            {"WorkspaceId": workspace_id, "SessionId": session_id},
            read_only=True,
        )
        return _object_list(result, "SessionResources")

    async def subscribe_session_events(
        self,
        *,
        workspace_id: str,
        session_id: str,
        after_event_id: str = "",
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[CocoEvent]:
        body = {"WorkspaceId": workspace_id, "SessionId": session_id}
        if after_event_id:
            body["AfterEventId"] = after_event_id
        action = "SubscribeManagedAgentSessionEvents"
        timeout = httpx.Timeout(
            timeout_seconds or float(self._settings.coco_task_timeout_seconds),
            connect=10.0,
        )
        fields, started = start_call("coco", action, body)
        try:
            async with self._http.stream(
                "POST",
                self._base,
                params={"Action": action},
                json=body,
                headers={**self._auth_headers(), "Accept": "text/event-stream"},
                timeout=timeout,
            ) as response:
                response_event(fields, started, response, body={"stream": "connected"})
                if response.status_code >= 400:
                    await response.aread()
                    self._parse_envelope(action, response)
                async for item in _iter_sse_json(response.aiter_lines()):
                    emit(
                        "EXTERNAL_STREAM_EVENT",
                        **{**fields, "details": {**fields.get("details", {}), **coco_identity(item)}},
                    )
                    raw_events = _sse_events(item)
                    for raw in raw_events:
                        yield _parse_event(raw)
            emit("EXTERNAL_STREAM_END", summary="stream closed", **fields)
        except httpx.RequestError as exc:
            error_event(fields, started, exc)
            raise TransientError(f"Coco SSE network error: {exc}") from exc
        except Exception as exc:
            error_event(fields, started, exc)
            raise

    async def archive_session(
        self,
        *,
        workspace_id: str,
        session_id: str,
    ) -> None:
        await self.call(
            "ArchiveManagedAgentSession",
            {"WorkspaceId": workspace_id, "Id": session_id},
        )

    async def delete_agent(self, *, workspace_id: str, agent_id: str) -> None:
        await self.call(
            "DeleteManagedAgent",
            {"WorkspaceId": workspace_id, "Id": agent_id},
        )

    async def delete_workspace(self, workspace_id: str) -> None:
        await self.call("DeleteManagedAgentWorkspace", {"Id": workspace_id})

    async def submit_resolve_task(
        self,
        agent_id: str,
        *,
        repo_id: str,
        task_id: int,
        original_mr_number: int,
        mr_version: int,
        resolve_branch: str,
        source_branch: str,
        source_commit: str,
        target_branch: str,
        target_commit: str,
        attempt: int,
        conflicts: list[dict[str, Any]],
        instructions: str,
    ) -> CocoTaskRef:
        """Compatibility entry point backed by Session + UserMessage.

        The caller must create ``resolve_branch`` from the source revision first.
        The repository is mounted by RepoId and Revision; Coco never receives a
        clone URL or a TCE-local checkout.
        """
        workspace_id = self._settings.coco_workspace_id
        if not workspace_id:
            raise CocoError(
                status=0,
                action="CreateManagedAgentSession",
                message="COCO_WORKSPACE_ID is required",
            )
        session = await self.create_session(
            workspace_id=workspace_id,
            agent_id=agent_id,
            repositories=[(repo_id, resolve_branch)],
        )
        session_id = str(session.get("Id") or "")
        if not session_id:
            raise CocoError(
                status=200,
                action="CreateManagedAgentSession",
                message="response does not contain Result.Session.Id",
            )

        manifest = {
            "schema_version": "conciliator.resolve.input.v2",
            "task_id": task_id,
            "repo_id": repo_id,
            "original_mr_number": original_mr_number,
            "mr_version": mr_version,
            "source_branch": source_branch,
            "source_commit": source_commit,
            "target_branch": target_branch,
            "target_commit": target_commit,
            "attempt": attempt,
            "resolve_branch": resolve_branch,
            "expected_resolve_head": source_commit,
            "conflicts": conflicts,
        }
        from conciliator.integrations.coco.prompts import execution_prompt

        prompt = execution_prompt(manifest, instructions=instructions, retry=False)
        event = await self.send_user_message(
            workspace_id=workspace_id,
            session_id=session_id,
            text=prompt,
        )
        return CocoTaskRef(
            task_id=session_id,
            run_id=event.run_id,
            user_event_id=event.event_id,
        )

    async def submit_retry_task(
        self,
        agent_id: str,
        *,
        repo_id: str,
        task_id: int,
        original_mr_number: int,
        mr_version: int,
        resolve_branch: str,
        source_branch: str,
        source_commit: str,
        target_branch: str,
        target_commit: str,
        base_commit_sha: str,
        attempt: int,
        rejected_conflicts: list[dict[str, Any]],
        locked_conflict_ids: list[str],
        instructions: str,
    ) -> CocoTaskRef:
        """Dispatch a retry that may only change the rejected conflict set."""
        workspace_id = self._settings.coco_workspace_id
        if not workspace_id:
            raise CocoError(
                status=0,
                action="CreateManagedAgentSession",
                message="COCO_WORKSPACE_ID is required",
            )
        session = await self.create_session(
            workspace_id=workspace_id,
            agent_id=agent_id,
            repositories=[(repo_id, resolve_branch)],
        )
        session_id = str(session.get("Id") or "")
        if not session_id:
            raise CocoError(
                status=200,
                action="CreateManagedAgentSession",
                message="response does not contain Result.Session.Id",
            )
        manifest = {
            "schema_version": "conciliator.resolve.retry.input.v2",
            "task_id": task_id,
            "repo_id": repo_id,
            "original_mr_number": original_mr_number,
            "mr_version": mr_version,
            "source_branch": source_branch,
            "source_commit": source_commit,
            "target_branch": target_branch,
            "target_commit": target_commit,
            "attempt": attempt,
            "resolve_branch": resolve_branch,
            "expected_resolve_head": base_commit_sha,
            "rejected_conflicts": rejected_conflicts,
            "locked_conflict_ids": locked_conflict_ids,
        }
        from conciliator.integrations.coco.prompts import execution_prompt

        prompt = execution_prompt(manifest, instructions=instructions, retry=True)
        event = await self.send_user_message(
            workspace_id=workspace_id,
            session_id=session_id,
            text=prompt,
        )
        return CocoTaskRef(
            task_id=session_id,
            run_id=event.run_id,
            user_event_id=event.event_id,
        )

    async def fetch_session_result(
        self,
        session_id: str,
        *,
        workspace_id: str | None = None,
        run_id: str = "",
        after_event_id: str = "",
    ) -> CocoTaskResult:
        """Fold generic Session events without imposing the resolver schema."""
        workspace_id = workspace_id or self._settings.coco_workspace_id
        if not workspace_id:
            raise CocoError(
                status=0,
                action="ListManagedAgentSessionEvents",
                message="workspace_id or COCO_WORKSPACE_ID is required",
            )
        if run_id or after_event_id:
            events = await self.read_round_events(
                workspace_id=workspace_id,
                session_id=session_id,
                after_event_id=after_event_id,
            )
            # An event without RunId is usable only after a known message boundary.
            events = [
                event
                for event in events
                if not run_id or event.run_id == run_id or (after_event_id and not event.run_id)
            ]
        else:
            events = await self.read_round_events(workspace_id=workspace_id, session_id=session_id)
        runs = await self.list_session_runs(workspace_id=workspace_id, session_id=session_id)
        if run_id:
            runs = [run for run in runs if str(run.get("Id") or run.get("RunId") or "") == run_id]
        status, error = _derive_status(events, runs)
        if run_id and runs:
            run_status = str(runs[-1].get("Status") or "").lower()
            if run_status in {"cancelled", "canceled", "failed", "terminated", "error"}:
                status = "failed"
                error = error or _run_failure_message(runs[-1], run_status)
        lifecycle_terminal = status in {"succeeded", "failed", "cancelled"}
        platform_error = _session_error_message(events)
        if platform_error and lifecycle_terminal:
            # A runtime error is the actionable cause; a generic "run ended with
            # terminated" hides whether the platform or the Agent failed.
            error = platform_error
        final_message = _fold_stream_text(events, "AgentMessage")
        if (
            (run_id or after_event_id)
            and status == "succeeded"
            and not platform_error
            and (not final_message or not any(e.kind == "SessionStatusIdle" for e in events))
        ):
            status = "running"  # Run projection can lead the final event stream.
        last_event_id = events[-1].event_id if events else ""
        return CocoTaskResult(
            status=status,
            final_message=final_message,
            error=error,
            last_event_id=last_event_id,
            lifecycle_terminal=lifecycle_terminal,
            platform_error=platform_error,
            requires_action_event_ids=(
                _requires_action_event_ids(events) if status == "requires_action" else ()
            ),
            agent_activity=any(event.kind.startswith("Agent") for event in events),
        )

    async def fetch_task_result(self, task_id: str, **round_context: Any) -> CocoTaskResult:
        """Read and validate the resolver result for a Managed Agent Session."""
        for attempt in range(_RESULT_READ_MAX_ATTEMPTS):
            session_result = await self.fetch_session_result(task_id, **round_context)
            if session_result.status != "succeeded":
                return session_result
            try:
                resolve_result = parse_resolve_result(session_result.final_message)
            except CocoResultError as exc:
                if not round_context and attempt + 1 < _RESULT_READ_MAX_ATTEMPTS:
                    await asyncio.sleep(_RESULT_READ_RETRY_DELAY_SECONDS)
                    continue
                return CocoTaskResult(
                    status="invalid_result",
                    final_message=session_result.final_message,
                    error=str(exc),
                    last_event_id=session_result.last_event_id,
                )
            return CocoTaskResult(
                status=session_result.status,
                resolve_result=resolve_result,
                final_message=session_result.final_message,
                commit_sha=resolve_result.commit_sha,
                error=session_result.error,
                last_event_id=session_result.last_event_id,
            )
        raise AssertionError("unreachable")

    async def fetch_retry_task_result(self, task_id: str, **round_context: Any) -> CocoTaskResult:
        """Read and validate one retry result after the Session becomes idle."""
        for attempt in range(_RESULT_READ_MAX_ATTEMPTS):
            session_result = await self.fetch_session_result(task_id, **round_context)
            if session_result.status != "succeeded":
                return session_result
            try:
                retry_result = parse_retry_result(session_result.final_message)
            except CocoResultError as exc:
                if not round_context and attempt + 1 < _RESULT_READ_MAX_ATTEMPTS:
                    await asyncio.sleep(_RESULT_READ_RETRY_DELAY_SECONDS)
                    continue
                return CocoTaskResult(
                    status="invalid_result",
                    final_message=session_result.final_message,
                    error=str(exc),
                    last_event_id=session_result.last_event_id,
                )
            return CocoTaskResult(
                status=session_result.status,
                retry_result=retry_result,
                final_message=session_result.final_message,
                commit_sha=retry_result.commit_sha,
                error=session_result.error,
                last_event_id=session_result.last_event_id,
            )
        raise AssertionError("unreachable")


def _require_object(result: dict[str, Any], key: str) -> dict[str, Any]:
    value = result.get(key)
    if not isinstance(value, dict):
        raise CocoError(
            status=200,
            action="parse response",
            message=f"Result.{key} is not an object",
        )
    return value


def _object_list(result: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = result.get(key) or []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise CocoError(
            status=200,
            action="parse response",
            message=f"Result.{key} is not an object list",
        )
    return value


def _parse_event(raw: dict[str, Any]) -> CocoEvent:
    payload = raw.get("Payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    return CocoEvent(
        event_id=str(raw.get("Id") or ""),
        session_id=str(raw.get("SessionId") or ""),
        run_id=str(raw.get("RunId") or ""),
        payload=payload,
        raw=raw,
    )


async def _iter_sse_json(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    data_lines: list[str] = []
    async for line in lines:
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data and data != "[DONE]":
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        yield parsed
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        parsed = json.loads("\n".join(data_lines))
        if isinstance(parsed, dict):
            yield parsed


def _sse_events(item: dict[str, Any]) -> list[dict[str, Any]]:
    result = item.get("Result")
    if isinstance(result, dict) and isinstance(result.get("Events"), list):
        return [event for event in result["Events"] if isinstance(event, dict)]
    event = item.get("Event")
    if isinstance(event, dict):
        return [event]
    if "Payload" in item:
        return [item]
    return []


def _event_text(event: CocoEvent) -> str:
    value = event.payload.get(event.kind)
    return _extract_text(value)


def _session_error_message(events: list[CocoEvent]) -> str | None:
    """Return the latest runtime-side error of a session, if any.

    ``SessionError.Detail`` carries the actionable platform message (request
    queue saturation, environment failures) while ``Message`` is usually a
    generic phrase such as "Internal error".
    """
    for event in reversed(events):
        if event.kind != "SessionError":
            continue
        payload = event.payload.get(event.kind)
        detail: Any = None
        if isinstance(payload, dict):
            detail = payload.get("Detail")
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except ValueError:
                detail = detail.strip()
        if isinstance(detail, dict):
            for key in ("message", "Message", "Error", "error", "code"):
                value = detail.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:500]
        elif isinstance(detail, str) and detail:
            return detail[:500]
        text = _extract_text(payload)
        return text[:500] if text else "Coco session reported a runtime error"
    return None


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_extract_text(item) for item in value)
    if not isinstance(value, dict):
        return ""
    for key in (
        "Text",
        "Content",
        "Message",
        "Error",
        "Reason",
        "FailureReason",
        "StatusReason",
    ):
        if key in value:
            return _extract_text(value[key])
    return ""


def _run_failure_message(run: dict[str, Any], status: str) -> str:
    for key in ("Error", "Message", "Reason", "FailureReason", "StatusReason"):
        message = _extract_text(run.get(key))
        if message:
            return message[:500]
    return f"Current Coco Run ended with {status}"


def _fold_stream_text(events: list[CocoEvent], kind: str) -> str:
    parts: list[str] = []
    for event in events:
        if event.kind != kind:
            continue
        value = event.payload.get(kind)
        text = _extract_text(value)
        if not text:
            continue
        if isinstance(value, dict) and value.get("Delta") is False:
            parts = [text]
        else:
            parts.append(text)
    return "".join(parts)


def _derive_status(
    events: list[CocoEvent],
    runs: list[dict[str, Any]],
) -> tuple[str, str | None]:
    for event in reversed(events):
        if event.kind in {"UserMessage", "SessionStatusRunning", "SessionStatusBusy"}:
            return "running", None
        if event.kind == "SessionStatusIdle":
            stop_reason = event.payload[event.kind].get("StopReason") or {}
            requires_action = stop_reason.get("RequiresAction") or {}
            if requires_action.get("EventIds"):
                return "requires_action", "Agent is waiting for tool confirmation"
            return "succeeded", None
        if event.kind == "SessionStatusTerminated":
            return "failed", _event_text(event) or "Session terminated"

    if runs:
        run_status = str(runs[-1].get("Status") or "").lower()
        if run_status in {"idle", "completed", "succeeded", "success"}:
            return "succeeded", None
        if run_status in {"cancelled", "canceled", "closed"}:
            return "cancelled", f"Run status is {run_status}"
        if run_status in {"terminated", "failed", "error"}:
            return "failed", f"Run status is {run_status}"
    return "running", None


def _requires_action_event_ids(events: list[CocoEvent]) -> tuple[str, ...]:
    """Return the tool-use events the latest pause is waiting on."""
    for event in reversed(events):
        if event.kind != "SessionStatusIdle":
            continue
        stop_reason = event.payload[event.kind].get("StopReason") or {}
        requires_action = stop_reason.get("RequiresAction") or {}
        return tuple(str(item) for item in requires_action.get("EventIds") or [] if item)
    return ()


_RESULT_BEGIN = "CONCILIATOR_RESULT_BEGIN"
_RESULT_END = "CONCILIATOR_RESULT_END"
_RESULT_READ_MAX_ATTEMPTS = 4
_RESULT_READ_RETRY_DELAY_SECONDS = 30.0
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_CONFLICT_KEYS = {
    "conflict_id",
    "path",
    "decision_summary",
    "source_attribution",
    "target_attribution",
    "resolved",
}
_ATTRIBUTION_KEYS = {
    "path",
    "start_line",
    "end_line",
    "commit_sha",
    "author_email",
}
_RESOLVED_KEYS = {"path", "start_line", "end_line", "text"}


def parse_resolve_result(message: str) -> CocoResolveResult:
    raw = _parse_bounded_result(message)
    if not isinstance(raw, dict) or raw.get("schema_version") != "conciliator.resolve.result.v3":
        raise CocoResultError("unsupported Coco result schema; expected conciliator.resolve.result.v3")
    return _parse_compact_result(raw, retry=False)


def parse_retry_result(message: str) -> CocoRetryResult:
    raw = _parse_bounded_result(message)
    if not isinstance(raw, dict) or raw.get("schema_version") != "conciliator.resolve.retry.result.v2":
        raise CocoResultError("unsupported Coco retry schema; expected conciliator.resolve.retry.result.v2")
    return _parse_compact_result(raw, retry=True)


_COMPACT_KEYS = {"schema_version", "status", "commit_sha", "conflicts", "unresolved_items"}


def _parse_compact_result(raw: dict, *, retry: bool):
    data = _strict_object(raw, _COMPACT_KEYS, "result")
    if _string(data, "status", "result") != "succeeded":
        raise CocoResultError("Coco result status is not succeeded")
    for idx, item in enumerate(_list(data, "conflicts", "result")):
        _strict_object(item, _CONFLICT_KEYS, f"result.conflicts[{idx}]")
    fields = dict(
        schema_version=data["schema_version"],
        task_id=0,
        status="succeeded",
        attempt=0,
        resolve_branch="",
        commit_sha=_sha(data.get("commit_sha"), "result.commit_sha"),
        conflicts=tuple(_parse_conflict(item, idx) for idx, item in enumerate(data["conflicts"])),
        unresolved_items=tuple(
            _plain_string(item, "result.unresolved_items") for item in _list(data, "unresolved_items", "result")
        ),
    )
    if retry:
        return CocoRetryResult(**fields, base_commit_sha="", parent_commits=())
    return CocoResolveResult(**fields, merge_parents=())


def result_to_dict(result: CocoResolveResult | CocoRetryResult) -> dict:
    """Persist server-owned context, without obsolete report fields."""
    from dataclasses import asdict

    data = asdict(result)
    return data


def _parse_bounded_result(message: str) -> Any:
    if len(message.encode("utf-8")) > 256_000:
        raise CocoResultError("Coco result exceeds 256 KB")
    if message.count(_RESULT_BEGIN) != 1 or message.count(_RESULT_END) != 1:
        raise CocoResultError("Coco result must contain exactly one result boundary")
    prefix, remainder = message.split(_RESULT_BEGIN, 1)
    payload, suffix = remainder.split(_RESULT_END, 1)
    if prefix.strip() or suffix.strip():
        raise CocoResultError("Coco result contains text outside the result boundary")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CocoResultError("Coco result JSON is invalid") from exc


def _parse_conflict(value: Any, idx: int) -> ConflictResolution:
    where = f"result.conflicts[{idx}]"
    data = _strict_object(value, _CONFLICT_KEYS, where)
    return ConflictResolution(
        conflict_id=_string(data, "conflict_id", where),
        path=_string(data, "path", where),
        decision_summary=_string(data, "decision_summary", where),
        source_attribution=_parse_attribution(data.get("source_attribution"), f"{where}.source_attribution"),
        target_attribution=_parse_attribution(data.get("target_attribution"), f"{where}.target_attribution"),
        resolved=_parse_resolved(data.get("resolved"), f"{where}.resolved"),
    )


def _parse_attribution(value: Any, where: str) -> ConflictAttribution:
    data = _strict_object(value, _ATTRIBUTION_KEYS, where)
    empty = data.get("author_email") is None
    start_line = _integer(data, "start_line", where, minimum=0 if empty else 1)
    end_line = _integer(data, "end_line", where, minimum=start_line)
    if empty and (start_line != 0 or end_line != 0 or data.get("commit_sha") != ""):
        raise CocoResultError(f"{where}: empty side requires 0 lines and empty commit_sha")
    return ConflictAttribution(
        path=_string(data, "path", where),
        start_line=start_line,
        end_line=end_line,
        commit_sha="" if empty else _sha(data.get("commit_sha"), f"{where}.commit_sha"),
        author_email=None if empty else _string(data, "author_email", where),
    )


def _parse_resolved(value: Any, where: str) -> ResolvedConflict:
    data = _strict_object(value, _RESOLVED_KEYS, where)
    start_line = _integer(data, "start_line", where, minimum=0)
    end_line = _integer(data, "end_line", where, minimum=start_line)
    if start_line == 0 and (end_line != 0 or data.get("text") != ""):
        raise CocoResultError(f"{where}: zero anchor requires empty text and 0/0 lines")
    # The service only accepts 0/0 for a server-classified blank block in an empty file.
    return ResolvedConflict(
        path=_string(data, "path", where),
        start_line=start_line,
        end_line=end_line,
        text=_plain_string(data.get("text"), f"{where}.text"),
    )


def _strict_object(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CocoResultError(f"{where} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise CocoResultError(f"{where} keys mismatch: missing={missing}, extra={extra}")
    return value


def _list(data: dict[str, Any], key: str, where: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise CocoResultError(f"{where}.{key} must be a list")
    return value


def _string(data: dict[str, Any], key: str, where: str) -> str:
    return _plain_string(data.get(key), f"{where}.{key}", allow_empty=False)


def _plain_string(value: Any, where: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CocoResultError(f"{where} must be a string")
    return value


def _integer(
    data: dict[str, Any],
    key: str,
    where: str,
    *,
    minimum: int | None = None,
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CocoResultError(f"{where}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise CocoResultError(f"{where}.{key} must be >= {minimum}")
    return value


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise CocoResultError(f"{where} must be a 40-character commit SHA")
    return value.lower()
