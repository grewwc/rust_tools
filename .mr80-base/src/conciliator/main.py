from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from conciliator.api import router as api_router
from conciliator.api.comment_handlers import (
    ApprovalCommentHandler,
    ResolveCommandHandler,
)
from conciliator.api.comment_router import CommentDispatcher, CommentRouter
from conciliator.api.meegle_logging import MeegleLoggingMiddleware
from conciliator.config import COMMENT_SUPER_ADMIN_EMAILS, Settings, get_settings
from conciliator.domain.aeolus_task import AeolusTask
from conciliator.integrations.coco.aeolus import CocoAeolusRunner
from conciliator.integrations.coco.client import CocoClient
from conciliator.integrations.lark.client import LarkClient, LarkError
from conciliator.integrations.lark.contacts import LarkUserNotFoundError, get_user_open_id_by_email
from conciliator.integrations.lark.docs import read_document_text
from conciliator.integrations.meegle.ai_node import MeegleAINodeClient
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.discussions import get_thread, reply_discussion
from conciliator.integrations.nextcode.mrs import add_mr_comment
from conciliator.integrations.ppe.environment import load_environment_profile
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.lark_user_token_repo import LarkUserTokenRepository
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.aeolus_comment_service import (
    AeolusCommentHandler,
    AeolusCommentService,
)
from conciliator.services.aeolus_document_access import (
    AeolusDocumentAccessService,
    public_origin_from_redirect_uri,
)
from conciliator.services.aeolus_lark_service import AeolusLarkService
from conciliator.services.aeolus_registry import AeolusTaskRegistry
from conciliator.services.aeolus_service import AeolusService, NextCodeAeolusHost
from conciliator.services.approval_service import ApprovalService
from conciliator.services.comment_events import CommentEvent, ThreadComment
from conciliator.services.deployment_maat_runtime import build_maat_runtime
from conciliator.services.deployment_service import DeploymentService
from conciliator.services.lark_notification_service import (
    LarkNotificationService,
)
from conciliator.services.lark_user_token_service import LarkUserTokenService, OAuthIdentityMismatch
from conciliator.services.notification_service import NotificationService
from conciliator.services.ppe_deployment_service import PpeDeploymentService
from conciliator.services.ppe_runner import build_ppe_runtime
from conciliator.services.resolve_service import ResolveDeps, ResolveService
from conciliator.services.runtime_selection import RuntimeDetectionService
from conciliator.services.trace_collector import TraceCollector
from conciliator.utils.logging import get_logger, setup_logging
from conciliator.utils.retry import TransientError
from conciliator.utils.secrets import SecretsProvider, build_secrets_provider
from conciliator.workers.aeolus_lark_ws import AeolusLarkWS
from conciliator.workers.lark_notification_runner import LarkNotificationRunner
from conciliator.workers.task_runner import TaskRunner

log = get_logger(__name__)


async def _resolve_lark_identity(
    meegle: MeegleAINodeClient, lark: LarkClient, user_key: str,
) -> str:
    try:
        email = await meegle.get_user_email(user_key)
    except (RuntimeError, httpx.HTTPError) as exc:
        raise TransientError("暂时无法核验 Meegle 用户身份，请稍后重试。") from exc
    if not email:
        raise OAuthIdentityMismatch("找不到 Meegle 用户的邮箱，无法核验授权身份。")
    try:
        return await get_user_open_id_by_email(lark, email)
    except (ValueError, LarkUserNotFoundError) as exc:
        raise OAuthIdentityMismatch("找不到对应的飞书账号，无法核验授权身份。") from exc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    setup_logging(settings)

    engine = create_engine_from_settings(settings)
    session_factory = get_session_factory(engine)
    task_repo = TaskRepository(session_factory)
    collector = TraceCollector(session_factory)
    app.state.trace_session_factory = session_factory
    await collector.start()

    app.state.settings = settings
    app.state.engine = engine
    app.state.task_repo = task_repo

    secrets: SecretsProvider | None = None
    nc_client: NextCodeClient | None = None
    coco_client: CocoClient | None = None
    lark_client: LarkClient | None = None
    meegle_client: MeegleAINodeClient | None = None
    aeolus_service: AeolusService | None = None
    worker: TaskRunner | None = None
    lark_notification_worker: LarkNotificationRunner | None = None
    aeolus_lark_ws: AeolusLarkWS | None = None
    maat_runtime = None
    ppe_runtime = None
    comment_router: CommentRouter | None = None
    aeolus_comment_service: AeolusCommentService | None = None
    aeolus_comment_handler: AeolusCommentHandler | None = None
    # Handlers are registered by the domain blocks below and consulted in
    # registration order, so a domain can be enabled on its own.
    comment_dispatcher = CommentDispatcher()
    app.state.comment_dispatcher = comment_dispatcher
    try:
        secrets = build_secrets_provider(settings)
        nc_client = NextCodeClient(settings, secrets)
        app.state.nextcode_client = nc_client
        coco_client = CocoClient(settings, secrets)
        app.state.coco_client = coco_client
        lark_client = LarkClient(settings, secrets)
        app.state.lark_client = lark_client

        notification = NotificationService(nc_client)
        resolve_service = ResolveService(ResolveDeps(
            nc=nc_client, coco=coco_client, lark=lark_client,
            repo=task_repo, notification=notification, settings=settings,
        ))
        approval_service = ApprovalService(
            nc=nc_client, repo=task_repo, notification=notification,
            session_factory=session_factory, super_admin_emails=COMMENT_SUPER_ADMIN_EMAILS,
        )
        comment_router = CommentRouter(
            resolve_service=resolve_service, approval_service=approval_service, task_repo=task_repo,
        )
        app.state.comment_router = comment_router
        app.state.resolve_service = resolve_service
        app.state.approval_service = approval_service
        worker = TaskRunner(repo=task_repo, resolve_service=resolve_service, settings=settings)
        app.state.worker = worker
        lark_notification_service = LarkNotificationService(
            lark=lark_client, repo=task_repo, settings=settings,
        )
        lark_notification_worker = LarkNotificationRunner(
            service=lark_notification_service, settings=settings,
        )
        app.state.lark_notification_service = lark_notification_service
        app.state.lark_notification_worker = lark_notification_worker

        meegle_client = MeegleAINodeClient(
            site_url=settings.meegle_site_url, plugin_id=settings.meegle_plugin_id,
            plugin_secret=secrets.get(settings.kms_secret_meegle_plugin_secret_alias),
        )
        registry = AeolusTaskRegistry(session_factory, deployments_enabled=True, ppe_enabled=True)
        runtime_detector = RuntimeDetectionService(session_factory, nc_client)
        ppe_deployments = PpeDeploymentService(
            session_factory, load_environment_profile(),
            profiles=(
                load_environment_profile(repository="dp/VizQueryService"),
                load_environment_profile(python_version="2.7"),
            ),
            runtime_detector=runtime_detector,
        )
        ppe_runtime = build_ppe_runtime(settings, ppe_deployments, secrets)
        app.state.ppe_deployment_service = ppe_deployments
        deployments = DeploymentService(session_factory, runtime_detector=runtime_detector)
        app.state.deployment_service = deployments
        maat_runtime = await build_maat_runtime(settings, deployments, secrets)
        app.state.maat_runtime = maat_runtime
        aeolus_lark = AeolusLarkService(
            lark_client, app_id=settings.lark_app_id, meegle=meegle_client,
            verification_token=settings.lark_card_verification_token.get_secret_value(),
            encrypt_key=settings.lark_card_encrypt_key.get_secret_value(),
        )
        app.state.aeolus_lark_service = aeolus_lark

        lark_user_tokens = LarkUserTokenService(
            settings=settings, secrets=secrets, repo=LarkUserTokenRepository(session_factory),
            identity_resolver=partial(_resolve_lark_identity, meegle_client, lark_client),
        )
        app.state.lark_user_token_service = lark_user_tokens
        if not lark_user_tokens.enabled:
            log.warning("lark_oauth_disabled_tenant_document_access",
                        missing_setting="LARK_OAUTH_REDIRECT_URI")

        meegle_webhook_token = secrets.get(settings.kms_secret_meegle_webhook_token_alias)
        document_access = None
        if lark_user_tokens.enabled:
            document_access = AeolusDocumentAccessService(
                registry=registry, user_tokens=lark_user_tokens, lark_client=lark_client,
                public_base_url=public_origin_from_redirect_uri(settings.lark_oauth_redirect_uri),
                signing_secret=meegle_webhook_token,
            )
            app.state.aeolus_document_access_service = document_access

        async def document_reader(url: str, user_key: str) -> str:
            assert lark_client is not None
            if document_access is not None:
                # The proxy and inline reader share the same document access policy.
                return await document_access.read_for_user(url, user_key)

            # Without OAuth, use an existing operator token or tenant document access.
            try:
                user_token = await lark_user_tokens.user_access_token(user_key)
            except TransientError:
                user_token = None
            if user_token:
                try:
                    return await read_document_text(lark_client, url, access_token=user_token)
                except LarkError as exc:
                    if exc.code != 131006:
                        raise
                    log.warning(
                        "aeolus_document_user_token_forbidden", user_key=user_key, lark_code=exc.code,
                    )
            return await read_document_text(lark_client, url)

        aeolus_service = AeolusService(
            registry=registry, document_reader=document_reader,
            coco=CocoAeolusRunner(
                coco_client, workspace_id=settings.coco_workspace_id,
                agent_id=settings.aeolus_coding_agent_id, coding_agents=settings.aeolus_coding_agents,
                vault_ids=[
                    item.strip() for item in settings.aeolus_credential_vault_ids.split(",") if item.strip()
                ],
                stall_timeout_seconds=settings.coco_stall_timeout_seconds,
                interrupt_timeout_seconds=settings.coco_interrupt_timeout_seconds,
                tool_confirmation_policy=settings.coco_tool_confirmation_policy,
                max_tool_confirmations=settings.coco_max_tool_confirmations,
            ),
            code_host=NextCodeAeolusHost(nc_client), meegle=meegle_client,
            plugin_id=settings.meegle_plugin_id, webhook_token=meegle_webhook_token,
            notify=aeolus_lark.notify, deployments=deployments, ppe_deployments=ppe_deployments,
            document_access=document_access,
            platform_retry_delay_seconds=settings.coco_platform_retry_delay_seconds,
        )
        app.state.aeolus_registry = registry
        app.state.meegle_ai_node_client = meegle_client
        app.state.aeolus_service = aeolus_service

        async def reply_comment(event: CommentEvent, body: str) -> None:
            assert nc_client is not None
            if event.discussion_id:
                await reply_discussion(
                    nc_client,
                    event.repo_id,
                    event.mr_iid,
                    event.discussion_id,
                    body,
                )
            else:
                await add_mr_comment(nc_client, event.repo_id, event.mr_iid, body)

        async def read_thread(repo_id: str, thread_id: str) -> list[ThreadComment]:
            assert nc_client is not None
            thread = await get_thread(nc_client, repo_id, thread_id)
            return [
                ThreadComment(
                    comment_id=str(comment.get("Id") or ""),
                    body=str(comment.get("Content") or ""),
                    author_email=str(
                        (comment.get("CreatedBy") or {}).get("Email") or ""
                    ),
                )
                for comment in thread.comments
            ]

        async def answer_question(
            task: AeolusTask, question: str, position: str,
            on_started=None,
        ) -> str:
            return await aeolus_service.answer_question(
                task.task_id, question, position=position, on_started=on_started
            )

        aeolus_comment_service = AeolusCommentService(
            service=aeolus_service,
            retry_timeout_seconds=settings.coco_task_timeout_seconds,
        )
        aeolus_comment_handler = AeolusCommentHandler(
            registry=registry,
            comments=aeolus_comment_service,
            meegle=meegle_client,
            allowlist=[
                item.strip()
                for item in settings.aeolus_comment_trigger_emails.split(",")
                if item.strip()
            ],
            reply=reply_comment,
            thread_reader=read_thread,
            answerer=answer_question,
            resumer=getattr(aeolus_service, "resume_answer", None),
        )

        async def notify_ppe(task_id, deployment_id):
            task = await aeolus_service.get(task_id)
            snapshot = await ppe_deployments.snapshot(task_id, deployment_id)
            task.ppe_deployment = snapshot["ppe_deployment"]
            await aeolus_lark.notify(task, strict=True, ppe_only=True)

        async def notify_deployment(task_id):
            await aeolus_lark.notify(await aeolus_service.get(task_id), devbox_only=True)

        ppe_runtime.notifier = notify_ppe
        maat_runtime.notifier = notify_deployment
        # Start workers only after all required clients and credentials are initialized.
        await worker.start()
        await lark_notification_worker.start()
        await ppe_runtime.start()
        await maat_runtime.start()
        await aeolus_service.start()
        if aeolus_comment_handler is not None:
            await aeolus_comment_handler.start()
        if settings.aeolus_lark_callback_mode == "websocket":
            aeolus_lark_ws = AeolusLarkWS(
                app_id=settings.lark_app_id,
                app_secret=secrets.get(settings.kms_secret_lark_app_secret_alias),
                cards=aeolus_lark, service=aeolus_service,
            )
            aeolus_lark_ws.start()

        # Precedence: an explicit /resolve command, then Aeolus review
        # feedback, then the approval thread fallback. Registering the
        # approval fallback last keeps it from swallowing inline comments.
        if comment_router is not None:
            comment_dispatcher.register(
                ResolveCommandHandler(comment_router, pipeline_enabled=True)
            )
        if aeolus_comment_handler is not None:
            comment_dispatcher.register(aeolus_comment_handler)
        if comment_router is not None:
            comment_dispatcher.register(
                ApprovalCommentHandler(comment_router, pipeline_enabled=True)
            )

        yield
    finally:
        if ppe_runtime is not None:
            await ppe_runtime.aclose()
        if maat_runtime is not None:
            await maat_runtime.aclose()
        if aeolus_lark_ws is not None:
            await aeolus_lark_ws.stop()
        if aeolus_comment_handler is not None:
            # Cancel the recovery scan before the services it drives close.
            await aeolus_comment_handler.aclose()
        if aeolus_comment_service is not None:
            await aeolus_comment_service.aclose()
        if aeolus_service is not None:
            await aeolus_service.aclose()
        if lark_notification_worker is not None:
            await lark_notification_worker.stop()
        if worker is not None:
            await worker.stop()
        if coco_client is not None:
            await coco_client.aclose()
        if lark_client is not None:
            await lark_client.aclose()
        if meegle_client is not None:
            await meegle_client.aclose()
        if nc_client is not None:
            await nc_client.aclose()
        if secrets is not None:
            secrets.close()
        await collector.stop()
        engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="Conciliator", lifespan=lifespan)
    app.add_middleware(MeegleLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://meego.larkoffice.com",
            "https://meegoplg.larkofficepkg.com",
        ],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Aeolus-Task-Token"],
    )

    @app.middleware("http")
    async def trace_headers(request, call_next):
        response = await call_next(request)
        if request.url.path == "/trace" or request.url.path.startswith(("/trace/", "/api/trace/")):
            response.headers["Cache-Control"] = "private, no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        return response

    app.include_router(api_router)
    return app


# 模块级 app 实例(供 gunicorn/uvicorn 直接 `conciliator.main:app` 启动)
app = create_app()
