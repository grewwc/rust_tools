"""Exercise the reader injected by the real lifespan without external resources."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest

from conciliator.domain.aeolus_task import AeolusState
from conciliator.integrations.lark.client import LarkError
from conciliator.services.aeolus_document_access import (
    DocumentNotAllowed,
    LarkAuthorizationRequired,
)
from conciliator.services.aeolus_registry import TaskNotFound

DOCX_URL = "https://example.feishu.cn/docx/doc123"
WIKI_URL = "https://example.feishu.cn/wiki/wiki123"
USER_KEY = "operator@example.test"
USER_TOKEN = "test-user-token"
MARKDOWN = "# Requirement\n\n**Preserve formatting**\n\n- first item"


@pytest.fixture
def reader_runtime(monkeypatch):
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("LARK_APP_ID", "cli_test_app")
    from conciliator import main

    settings = SimpleNamespace(
        lark_app_id="test-app",
        aeolus_lark_callback_mode="websocket",
        lark_card_verification_token=SimpleNamespace(get_secret_value=lambda: ""),
        lark_card_encrypt_key=SimpleNamespace(get_secret_value=lambda: ""),
        kms_secret_lark_app_secret_alias="test-lark-secret",
        aeolus_credential_vault_ids="",
        aeolus_comment_trigger_emails="",
        meegle_site_url="https://project.example.test",
        meegle_plugin_id="test-plugin",
        kms_secret_meegle_plugin_secret_alias="test-plugin-secret",
        kms_secret_meegle_webhook_token_alias="test-webhook-secret",
        lark_oauth_redirect_uri="https://app.example.test/oauth/callback",
        coco_workspace_id="test-workspace",
        aeolus_coding_agent_id="test-agent",
        aeolus_coding_agents={},
        coco_task_timeout_seconds=30,
        coco_stall_timeout_seconds=20,
        coco_interrupt_timeout_seconds=5,
        coco_platform_retry_delay_seconds=0,
        coco_tool_confirmation_policy="deny",
        coco_max_tool_confirmations=3,
    )
    tokens = SimpleNamespace(
        enabled=True,
        user_access_token=AsyncMock(return_value=USER_TOKEN),
        authorize_url=Mock(return_value="https://auth.example.test/authorize"),
    )
    registry = SimpleNamespace(get=AsyncMock())
    lark = SimpleNamespace(_request=AsyncMock(), aclose=AsyncMock())
    service = SimpleNamespace(start=AsyncMock(), aclose=AsyncMock())
    service_factory = Mock(return_value=service)
    access_factory = Mock(wraps=main.AeolusDocumentAccessService)
    raw_reader = AsyncMock(wraps=main.read_document_text)
    replacements = {
        "get_settings": Mock(return_value=settings),
        "setup_logging": Mock(),
        "create_engine_from_settings": Mock(return_value=Mock()),
        "get_session_factory": Mock(return_value=Mock()),
        "TaskRepository": Mock(),
        "TraceCollector": Mock(return_value=SimpleNamespace(
            start=AsyncMock(), stop=AsyncMock(),
        )),
        "build_secrets_provider": Mock(return_value=Mock(
            get=Mock(return_value="test-signing-secret"),
        )),
        "LarkClient": Mock(return_value=lark),
        "AeolusTaskRegistry": Mock(return_value=registry),
        "LarkUserTokenRepository": Mock(),
        "LarkUserTokenService": Mock(return_value=tokens),
        "CocoAeolusRunner": Mock(),
        "NextCodeAeolusHost": Mock(),
        "AeolusService": service_factory,
        "AeolusDocumentAccessService": access_factory,
        "read_document_text": raw_reader,
        "DeploymentService": Mock(),
        "PpeDeploymentService": Mock(),
        "build_maat_runtime": AsyncMock(return_value=SimpleNamespace(start=AsyncMock(), aclose=AsyncMock())),
        "build_ppe_runtime": Mock(return_value=SimpleNamespace(start=AsyncMock(), aclose=AsyncMock())),
        "AeolusLarkService": Mock(return_value=SimpleNamespace(notify=AsyncMock())),
        "AeolusLarkWS": Mock(return_value=SimpleNamespace(start=Mock(), stop=AsyncMock())),
    }
    for name in ("TaskRunner", "LarkNotificationRunner"):
        replacements[name] = Mock(return_value=SimpleNamespace(start=AsyncMock(), stop=AsyncMock()))
    for name in ("NextCodeClient", "CocoClient", "MeegleAINodeClient"):
        replacements[name] = Mock(return_value=SimpleNamespace(aclose=AsyncMock()))
    for name, replacement in replacements.items():
        monkeypatch.setattr(main, name, replacement)

    @asynccontextmanager
    async def open_runtime(*, oauth_enabled=True):
        tokens.enabled = oauth_enabled
        app = main.create_app()
        # Keep startup wiring real; only resource boundaries and workers are fake.
        async with app.router.lifespan_context(app):
            service_factory.assert_called_once()
            service.start.assert_awaited_once()
            args = service_factory.call_args.kwargs
            yield SimpleNamespace(
                app=app,
                reader=args["document_reader"],
                access=args["document_access"],
                access_factory=access_factory,
                tokens=tokens,
                lark=lark,
                registry=registry,
                raw_reader=raw_reader,
            )
        service.aclose.assert_awaited_once()
        lark.aclose.assert_awaited_once()

    return open_runtime


@pytest.mark.asyncio
async def test_lifespan_starts_and_closes_maat_runtime_before_database(reader_runtime, monkeypatch):
    from conciliator import main

    settings = main.get_settings()
    deployments = object()
    closed = []
    maat = SimpleNamespace(start=AsyncMock(), aclose=AsyncMock(side_effect=lambda: closed.append("maat")))
    builder = AsyncMock(return_value=maat)
    monkeypatch.setattr(main, "DeploymentService", Mock(return_value=deployments))
    monkeypatch.setattr(main, "build_maat_runtime", builder)
    main.create_engine_from_settings.return_value.dispose.side_effect = lambda: closed.append("database")
    async with reader_runtime() as runtime:
        builder.assert_awaited_once_with(settings, deployments, main.build_secrets_provider.return_value)
        maat.start.assert_awaited_once()
        assert runtime.app.state.maat_runtime is maat
    maat.aclose.assert_awaited_once()
    assert closed == ["maat", "database"]


def document_responses(url):
    responses = []
    if url == WIKI_URL:
        responses.append({"node": {"obj_type": "docx", "obj_token": "doc123"}})
    return [*responses, {"document": {"content": MARKDOWN}}]


def assert_markdown_requests(lark, url):
    requests = lark._request.await_args_list
    assert len(requests) == (2 if url == WIKI_URL else 1)
    assert all(item.kwargs["access_token"] is None for item in requests)
    if url == WIKI_URL:
        assert requests[0] == call(
            "GET", "/wiki/v2/spaces/get_node",
            params={"token": "wiki123"}, access_token=None,
        )
    assert requests[-1].args == ("POST", "/docs_ai/v1/documents/doc123/fetch")
    assert requests[-1].kwargs["json"]["format"] == "markdown"


@pytest.mark.parametrize("url", [DOCX_URL, WIKI_URL], ids=["docx", "wiki"])
@pytest.mark.parametrize("user_token", [None, USER_TOKEN])
async def test_oauth_reader_and_proxy_use_app_access_without_user_authorization(
    reader_runtime, monkeypatch, url, user_token,
):
    async with reader_runtime() as runtime:
        runtime.tokens.user_access_token.return_value = user_token
        runtime.access_factory.assert_called_once()
        assert runtime.access is runtime.app.state.aeolus_document_access_service
        read_for_user = AsyncMock(wraps=runtime.access.read_for_user)
        monkeypatch.setattr(runtime.access, "read_for_user", read_for_user)
        runtime.lark._request.side_effect = document_responses(url)

        assert await runtime.reader(url, USER_KEY) == MARKDOWN
        read_for_user.assert_awaited_once_with(url, USER_KEY)
        runtime.tokens.user_access_token.assert_not_awaited()
        assert_markdown_requests(runtime.lark, url)

        task = SimpleNamespace(
            task_id="task123", user_key=USER_KEY, state=AeolusState.EVALUATING,
            requirement_url=url, requirement_text="",
        )
        runtime.registry.get.return_value = task
        assert runtime.access.instructions(task) == ""
        with pytest.raises(DocumentNotAllowed):
            await runtime.access.read_markdown(
                task_id=task.task_id, capability=runtime.access.issue(task), url=url,
            )
        read_for_user.assert_awaited_once_with(url, USER_KEY)
        runtime.tokens.user_access_token.assert_not_awaited()

        task.requirement_url = "https://example.feishu.cn/docx/another-primary"
        task.requirement_text = f"[引用资料]({url})"
        runtime.lark._request.reset_mock(side_effect=True)
        runtime.lark._request.side_effect = document_responses(url)
        assert await runtime.access.read_markdown(
            task_id=task.task_id, capability=runtime.access.issue(task), url=url,
        ) == MARKDOWN
        assert read_for_user.await_args_list == [call(url, USER_KEY)] * 2
        assert_markdown_requests(runtime.lark, url)
        runtime.raw_reader.assert_not_awaited()


async def test_oauth_reader_requires_authorization_when_app_denied_and_user_missing(
    reader_runtime,
):
    async with reader_runtime() as runtime:
        runtime.tokens.user_access_token.return_value = None
        runtime.lark._request.side_effect = LarkError("app access forbidden", code=131006)
        with pytest.raises(LarkAuthorizationRequired) as exc:
            await runtime.reader(DOCX_URL, USER_KEY)

        assert exc.value.authorization_url == "https://auth.example.test/authorize"
        runtime.tokens.user_access_token.assert_awaited_once_with(USER_KEY)
        runtime.tokens.authorize_url.assert_called_once_with(USER_KEY)
        runtime.lark._request.assert_awaited_once()
        assert runtime.lark._request.call_args.kwargs["access_token"] is None
        runtime.raw_reader.assert_not_awaited()


async def test_oauth_reader_requires_authorization_for_document_acl_denial(
    reader_runtime,
):
    async with reader_runtime() as runtime:
        runtime.tokens.user_access_token.return_value = None
        runtime.lark._request.side_effect = LarkError(
            "app access forbidden", code=3380004,
        )

        with pytest.raises(LarkAuthorizationRequired) as exc:
            await runtime.reader(DOCX_URL, USER_KEY)

        assert exc.value.authorization_url == "https://auth.example.test/authorize"
        runtime.tokens.user_access_token.assert_awaited_once_with(USER_KEY)
        runtime.tokens.authorize_url.assert_called_once_with(USER_KEY)
        runtime.lark._request.assert_awaited_once()
        assert runtime.lark._request.call_args.kwargs["access_token"] is None
        runtime.raw_reader.assert_not_awaited()


@pytest.mark.parametrize(
    ("url", "failure_index"),
    [(DOCX_URL, 0), (WIKI_URL, 0), (WIKI_URL, 1)],
    ids=["docx-export", "wiki-resolution", "wiki-export"],
)
async def test_oauth_reader_restarts_complete_read_as_user_on_app_denial(
    reader_runtime, url, failure_index,
):
    async with reader_runtime() as runtime:
        error = LarkError("app access forbidden", code=131006)
        responses = document_responses(url)
        runtime.lark._request.side_effect = [*responses[:failure_index], error, *responses]

        assert await runtime.reader(url, USER_KEY) == MARKDOWN

        requests = runtime.lark._request.await_args_list
        assert [item.kwargs["access_token"] for item in requests] == (
            [None] * (failure_index + 1) + [USER_TOKEN] * len(responses)
        )
        assert requests[-1].args == ("POST", "/docs_ai/v1/documents/doc123/fetch")
        runtime.tokens.user_access_token.assert_awaited_once_with(USER_KEY)
        runtime.raw_reader.assert_not_awaited()


@pytest.mark.parametrize("token_mode", ["missing", "user", "forbidden"])
async def test_disabled_oauth_preserves_raw_content_reader_without_proxy(
    reader_runtime, token_mode,
):
    async with reader_runtime(oauth_enabled=False) as runtime:
        runtime.access_factory.assert_not_called()
        assert runtime.access is None
        assert not hasattr(runtime.app.state, "aeolus_document_access_service")
        runtime.tokens.user_access_token.return_value = (
            None if token_mode == "missing" else USER_TOKEN
        )
        responses = [{"content": "  legacy raw content  "}]
        if token_mode == "forbidden":
            responses.insert(0, LarkError("user access forbidden", code=131006))
        runtime.lark._request.side_effect = responses

        assert await runtime.reader(DOCX_URL, USER_KEY) == "legacy raw content"

        expected_tokens = {
            "missing": [None], "user": [USER_TOKEN],
            "forbidden": [USER_TOKEN, None],
        }[token_mode]
        assert runtime.lark._request.await_args_list == [
            call("GET", "/docx/v1/documents/doc123/raw_content", access_token=token)
            for token in expected_tokens
        ]
        assert runtime.raw_reader.await_count == len(expected_tokens)
        runtime.tokens.user_access_token.assert_awaited_once_with(USER_KEY)


@pytest.mark.asyncio
async def test_lifespan_wires_both_deployment_services_and_closes_before_database(reader_runtime, monkeypatch):
    from conciliator import main
    deployments = object()
    closed = []
    runtime = SimpleNamespace(start=AsyncMock(), aclose=AsyncMock(side_effect=lambda: closed.append('ppe')))
    builder = Mock(return_value=runtime)
    monkeypatch.setattr(main, 'build_ppe_runtime', builder)
    monkeypatch.setattr(main, 'PpeDeploymentService', Mock(return_value=deployments))
    main.create_engine_from_settings.return_value.dispose.side_effect = lambda: closed.append('database')
    async with reader_runtime() as current:
        assert main.AeolusService.call_args.kwargs['ppe_deployments'] is deployments
        assert main.AeolusService.call_args.kwargs['deployments'] is main.DeploymentService.return_value
        runtime.start.assert_awaited_once()
        assert current.app.state.ppe_deployment_service is deployments
    assert closed == ['ppe', 'database']


async def test_lifespan_initializes_all_features(reader_runtime):
    from conciliator import main

    async with reader_runtime() as current:
        for name in ("worker", "lark_notification_worker", "aeolus_service", "maat_runtime"):
            getattr(current.app.state, name).start.assert_awaited_once()
        main.AeolusTaskRegistry.assert_called_once_with(
            main.get_session_factory.return_value, deployments_enabled=True, ppe_enabled=True,
        )
        main.AeolusLarkWS.return_value.start.assert_called_once()
        assert main.AeolusService.call_args.kwargs["notify"] is current.app.state.aeolus_lark_service.notify
    main.AeolusLarkWS.return_value.stop.assert_awaited_once()


async def test_lifespan_wires_coco_runtime_controls(reader_runtime):
    from conciliator import main

    async with reader_runtime():
        settings = main.get_settings()
        main.CocoAeolusRunner.assert_called_once_with(
            main.CocoClient.return_value,
            workspace_id=settings.coco_workspace_id,
            agent_id=settings.aeolus_coding_agent_id,
            coding_agents=settings.aeolus_coding_agents,
            vault_ids=[],
            stall_timeout_seconds=settings.coco_stall_timeout_seconds,
            interrupt_timeout_seconds=settings.coco_interrupt_timeout_seconds,
            tool_confirmation_policy=settings.coco_tool_confirmation_policy,
            max_tool_confirmations=settings.coco_max_tool_confirmations,
        )
        assert (
            main.AeolusService.call_args.kwargs["platform_retry_delay_seconds"]
            == settings.coco_platform_retry_delay_seconds
        )


async def test_started_app_serves_aeolus_queries_without_legacy_state_flag(reader_runtime):
    async with reader_runtime() as current:
        current.app.state.aeolus_service.find_by_node = AsyncMock(side_effect=TaskNotFound("missing"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=current.app), base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/aeolus/tasks/by-node",
                params={"space_id": "test", "work_item_id": 1, "node_id": "test"},
            )
        assert response.status_code == 404
        assert response.json() == {"detail": "Aeolus task not found"}
        current.app.state.aeolus_service.find_by_node.assert_awaited_once_with("test", 1, "test")


async def test_missing_deployment_credentials_abort_startup_and_close_resources(reader_runtime, monkeypatch):
    from conciliator import main
    from conciliator.integrations.maat.client import MaatError

    monkeypatch.setattr(main, "build_maat_runtime", AsyncMock(side_effect=MaatError("credentials missing")))
    app = main.create_app()
    with pytest.raises(MaatError, match="credentials missing"):
        async with app.router.lifespan_context(app):
            pytest.fail("service must not start without deployment credentials")
    main.TaskRunner.return_value.start.assert_not_awaited()
    main.LarkNotificationRunner.return_value.start.assert_not_awaited()
    main.build_ppe_runtime.return_value.aclose.assert_awaited_once()
    for name in ("NextCodeClient", "CocoClient", "LarkClient", "MeegleAINodeClient"):
        getattr(main, name).return_value.aclose.assert_awaited_once()
    main.build_secrets_provider.return_value.close.assert_called_once()
    main.create_engine_from_settings.return_value.dispose.assert_called_once()
