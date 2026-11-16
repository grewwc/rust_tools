#!/usr/bin/env python3
"""Run a redacted Coco OpenAPI Next conversation smoke test.

The ``--provision`` path creates a temporary Workspace and Agent, runs
one no-repository task, verifies AgentMessage + SessionStatusIdle, and removes
the cloud resources. Secret values are fetched through Conciliator's KMS
provider and are never printed or written to disk.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from conciliator.config import (
    COCO_DEFAULT_AGENT_ID_DEFAULT,
    COCO_WORKSPACE_ID_DEFAULT,
    NEXTCODE_APP_SECRET_ALIAS_DEFAULT,
    Settings,
)
from conciliator.integrations.coco.client import CocoClient, CocoEvent
from conciliator.utils.secrets import KmsSecretsProvider

DEFAULT_APP_ID = "792763146702616"
EXPECTED_REPLY = "COCO_BASIC_SMOKE_OK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Probe Coco App-only headers, Session events, Agent reply, and idle state.")
    )
    parser.add_argument(
        "--provision",
        action="store_true",
        help="create a temporary Workspace and Agent for this run",
    )
    parser.add_argument(
        "--workspace-id",
        default=os.getenv("COCO_WORKSPACE_ID", COCO_WORKSPACE_ID_DEFAULT),
        help="reuse an existing Workspace when --provision is omitted",
    )
    parser.add_argument(
        "--agent-id",
        default=os.getenv("COCO_DEFAULT_AGENT_ID", COCO_DEFAULT_AGENT_ID_DEFAULT),
        help="reuse an existing Agent when --provision is omitted",
    )
    parser.add_argument(
        "--app-id",
        default=os.getenv("NEXTCODE_APP_ID", DEFAULT_APP_ID),
    )
    parser.add_argument(
        "--app-secret-alias",
        default=os.getenv(
            "KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS",
            NEXTCODE_APP_SECRET_ALIAS_DEFAULT,
        ),
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv(
            "COCO_API_BASE",
            "https://codebase-api.byted.org/v2/",
        ),
    )
    parser.add_argument(
        "--kms-zone",
        help="set KMS_ZONE before SDK initialization, for example cn on a Mac",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--keep-resources",
        action="store_true",
        help="do not clean up resources created by --provision",
    )
    args = parser.parse_args()
    if not args.provision and (not args.workspace_id or not args.agent_id):
        parser.error("pass --provision, or provide both --workspace-id and --agent-id")
    return args


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        env="prod",
        nextcode_app_id=args.app_id,
        kms_secret_nextcode_app_secret_alias=args.app_secret_alias,
        coco_api_base=args.api_base,
        coco_task_timeout_seconds=max(1, int(args.timeout)),
    )


async def _collect_turn(
    client: CocoClient,
    *,
    workspace_id: str,
    session_id: str,
    after_event_id: str,
    timeout_seconds: float,
) -> list[CocoEvent]:
    events: list[CocoEvent] = []

    async def consume() -> None:
        async for event in client.subscribe_session_events(
            workspace_id=workspace_id,
            session_id=session_id,
            after_event_id=after_event_id,
            timeout_seconds=timeout_seconds,
        ):
            events.append(event)
            if event.kind in {"SessionStatusIdle", "SessionStatusTerminated"}:
                return

    await asyncio.wait_for(consume(), timeout=timeout_seconds)
    return events


async def run(args: argparse.Namespace) -> int:
    if args.kms_zone:
        os.environ["KMS_ZONE"] = args.kms_zone

    provider = KmsSecretsProvider()
    client: CocoClient | None = None
    workspace_id = args.workspace_id
    agent_id = args.agent_id
    session_id = ""
    created_workspace = False
    created_agent = False
    exit_code = 1

    try:
        client = CocoClient(_settings(args), provider)
        if args.provision:
            suffix = f"{int(time.time())}-{os.getpid()}"
            workspace = await client.create_workspace(
                name=f"conciliator-coco-smoke-{suffix}",
                slug=f"ccs-{suffix}",
            )
            workspace_id = str(workspace.get("Id") or "")
            if not workspace_id:
                raise RuntimeError("CreateManagedAgentWorkspace returned no Id")
            created_workspace = True

            devices = await client.list_workspace_devices(workspace_id)
            system_devices = [item for item in devices if str(item.get("Kind") or "").lower() == "system"]
            selected = (system_devices or devices)[0] if devices else None
            if selected is None or not selected.get("Id"):
                raise RuntimeError("Workspace has no usable system device")
            runtimes = selected.get("Runtimes") or []
            first_runtime = runtimes[0] if runtimes else "traex"
            runtime = (
                str(first_runtime.get("Name") or "traex") if isinstance(first_runtime, dict) else str(first_runtime)
            )
            agent = await client.create_agent(
                workspace_id=workspace_id,
                name="conciliator-basic-smoke",
                device_id=str(selected["Id"]),
                runtime=runtime,
                instruction="Reply concisely and follow exact output instructions.",
                permission_mode="default",
            )
            agent_id = str(agent.get("Id") or "")
            if not agent_id:
                raise RuntimeError("CreateManagedAgent returned no Id")
            created_agent = True

        session = await client.create_session(
            workspace_id=workspace_id,
            agent_id=agent_id,
        )
        session_id = str(session.get("Id") or "")
        if not session_id:
            raise RuntimeError("CreateManagedAgentSession returned no Id")

        user_event = await client.send_user_message(
            workspace_id=workspace_id,
            session_id=session_id,
            text=f"Reply with exactly {EXPECTED_REPLY} and nothing else.",
        )
        streamed = await _collect_turn(
            client,
            workspace_id=workspace_id,
            session_id=session_id,
            after_event_id=user_event.event_id,
            timeout_seconds=args.timeout,
        )
        persisted = await client.list_session_events(
            workspace_id=workspace_id,
            session_id=session_id,
        )
        runs = await client.list_session_runs(
            workspace_id=workspace_id,
            session_id=session_id,
        )
        result = await client.fetch_session_result(
            session_id,
            workspace_id=workspace_id,
        )

        streamed_kinds = {event.kind for event in streamed}
        persisted_kinds = {event.kind for event in persisted}
        has_message = "AgentMessage" in streamed_kinds | persisted_kinds
        has_idle = "SessionStatusIdle" in streamed_kinds | persisted_kinds
        reply_ok = EXPECTED_REPLY in result.final_message
        run_ok = bool(runs and str(runs[-1].get("Id") or ""))

        print("coco_connectivity=ok")
        print(f"workspace_id={workspace_id}")
        print(f"agent_id={agent_id}")
        print(f"session_id={session_id}")
        print(f"run_id={user_event.run_id}")
        print(f"agent_message_seen={str(has_message).lower()}")
        print(f"session_idle_seen={str(has_idle).lower()}")
        print(f"expected_reply_seen={str(reply_ok).lower()}")
        print(f"run_list_nonempty={str(run_ok).lower()}")
        print(f"derived_status={result.status}")

        if not all((has_message, has_idle, reply_ok, run_ok, result.status == "succeeded")):
            print("coco_basic_task=failed", file=sys.stderr)
        else:
            print("coco_basic_task=ok")
            exit_code = 0
    finally:
        cleanup_errors: list[str] = []
        if client is not None and session_id and not args.keep_resources:
            try:
                await client.archive_session(
                    workspace_id=workspace_id,
                    session_id=session_id,
                )
            except Exception as exc:
                cleanup_errors.append(f"archive_session:{type(exc).__name__}")
        if client is not None and created_agent and not args.keep_resources:
            try:
                await client.delete_agent(
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                )
            except Exception as exc:
                cleanup_errors.append(f"delete_agent:{type(exc).__name__}")
        if client is not None and created_workspace and not args.keep_resources:
            try:
                await client.delete_workspace(workspace_id)
            except Exception as exc:
                cleanup_errors.append(f"delete_workspace:{type(exc).__name__}")
        if client is not None:
            await client.aclose()
        provider.close()
        if cleanup_errors:
            print(
                f"coco_cleanup=failed actions={','.join(cleanup_errors)}",
                file=sys.stderr,
            )
            exit_code = 1
    return exit_code


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except Exception as exc:
        message = " ".join(str(exc).splitlines())
        print(
            f"coco_basic_task=failed error_type={type(exc).__name__} error={message}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
