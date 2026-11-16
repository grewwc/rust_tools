from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from conciliator.domain.aeolus_task import AeolusState
from conciliator.integrations.lark.client import LarkError
from conciliator.services.aeolus_document_access import (
    DOCUMENT_CAPABILITY_HEADER,
    AeolusDocumentAccessService,
    DocumentAccessInactive,
    DocumentNotAllowed,
    InvalidDocumentCapability,
    LarkAuthorizationRequired,
    public_origin_from_redirect_uri,
)
from conciliator.utils.retry import TransientError


class FakeRegistry:
    def __init__(self, task):
        self.task = task

    async def get(self, task_id):
        assert task_id == self.task.task_id
        return self.task


class FakeUserTokens:
    def __init__(self, token="user-access-token"):
        self.token = token
        self.lookups = []

    async def user_access_token(self, user_key):
        self.lookups.append(user_key)
        return self.token

    def authorize_url(self, user_key):
        return f"https://auth.example/{user_key}"


class FakeLarkClient:
    def __init__(self):
        self.calls = []

    async def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == "/wiki/v2/spaces/get_node":
            return {"node": {"obj_type": "docx", "obj_token": "WikiDocToken"}}
        return {"document": {"content": "# Exported Markdown\n\nBody"}}


def _task(**overrides):
    values = {
        "task_id": "task-1",
        "user_key": "user-1",
        "action_token": "server-only-action-token",
        "state": AeolusState.EVALUATING,
        "requirement_url": "https://bytedance.larkoffice.com/wiki/WikiToken1",
        "requirement_text": (
            "See https://bytedance.sg.larkoffice.com/docx/DocToken2 for details."
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _service(
    task=None,
    *,
    user_token="user-access-token",
    signing_secret="server-capability-secret",
    clock=lambda: 1000.0,
):
    task = task or _task()
    tokens = FakeUserTokens(user_token)
    lark = FakeLarkClient()
    service = AeolusDocumentAccessService(
        registry=FakeRegistry(task),
        user_tokens=tokens,
        lark_client=lark,
        public_base_url="https://conciliator.example/",
        signing_secret=signing_secret,
        ttl_seconds=60,
        clock=clock,
    )
    return service, tokens, lark


def test_instructions_contain_only_scoped_capability():
    service, _, _ = _service()

    instructions = service.instructions(_task())

    assert "https://conciliator.example/api/aeolus/tasks/task-1/lark-document" in instructions
    assert DOCUMENT_CAPABILITY_HEADER in instructions
    assert "WikiToken1" not in instructions
    assert "DocToken2" in instructions
    assert "server-only-action-token" not in instructions
    assert "user-access-token" not in instructions
    assert "lark_authorization_required" in instructions
    assert "停止重试" in instructions
    assert "authorization_url" not in instructions
    assert "curl -fsS" not in instructions
    assert "$(jq" not in instructions
    assert "| jq" not in instructions


@pytest.mark.asyncio
async def test_cite_reference_is_exposed_and_allowed():
    url = "https://bytedance.sg.larkoffice.com/docx/CitedDocToken"
    task = _task(
        requirement_url="https://bytedance.sg.larkoffice.com/docx/MainDoc",
        requirement_text=(
            '<cite doc-id="CitedDocToken" file-type="docx" type="doc">'
            "引用设计文档</cite>"
        ),
    )
    service, tokens, _ = _service(task)

    instructions = service.instructions(task)
    markdown = await service.read_markdown(
        task_id=task.task_id,
        capability=service.issue(task),
        url=url,
    )

    assert url in instructions
    assert "MainDoc" not in instructions
    assert markdown == "# Exported Markdown\n\nBody"
    assert tokens.lookups == []


@pytest.mark.parametrize("text", [
    "",
    "# 需求\n实现既定需求，无外部引用。",
    "https://example.com/docx/OtherDoc",
    "https://bytedance.larkoffice.com/sheets/OtherSheet",
    "https://bytedance.larkoffice.com/wiki/",
    "https://bytedance.larkoffice.com.evil.example/docx/OtherDoc",
])
def test_no_document_references_do_not_issue_capability(text):
    task = _task(requirement_text=text)
    service, _, _ = _service(task)
    service.issue = Mock(side_effect=AssertionError("must not issue a capability"))

    assert service.instructions(task) == ""
    service.issue.assert_not_called()


@pytest.mark.parametrize("kind", ["docx", "wiki"])
def test_self_links_do_not_enable_document_tools(kind):
    task = _task(
        requirement_url=f"https://bytedance.larkoffice.com/{kind}/MainDoc?from=card#intro",
        requirement_text=(
            f"[需求](https://bytedance.larkoffice.com/{kind}/MainDoc?from=copy#details)\n"
            f"http://BYTEDANCE.SG.LARKOFFICE.COM/{kind}/MainDoc/\n"
            f"https://team.feishu.cn/{kind}/MainDoc\n"
            f'<cite file-type="{kind}" doc-id="MainDoc">主文档</cite>'
        ),
    )
    service, _, _ = _service(task)
    service.issue = Mock(side_effect=AssertionError("must not issue a capability"))

    assert service.instructions(task) == ""
    service.issue.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("via_proxy", [False, True])
@pytest.mark.parametrize("user_token", [None, "user-access-token"])
async def test_readers_export_markdown_with_app_token(via_proxy, user_token):
    task = _task()
    service, tokens, lark = _service(task, user_token=user_token)
    url = "https://bytedance.larkoffice.com/docx/DocToken2?from=task"
    if via_proxy:
        markdown = await service.read_markdown(
            task_id=task.task_id, capability=service.issue(task), url=url,
        )
    else:
        markdown = await service.read_for_user(url, task.user_key)

    assert markdown == "# Exported Markdown\n\nBody"
    assert tokens.lookups == []
    assert lark.calls == [
        (
            "POST",
            "/docs_ai/v1/documents/DocToken2/fetch",
            {
                "json": {
                    "format": "markdown",
                    "export_option": {
                        "export_block_id": False,
                        "export_cite_extra_data": False,
                        "export_style_attrs": False,
                    },
                    "extra_param": (
                        '{"enable_user_cite_reference_map":true,'
                        '"return_html5_block_data":true}'
                    ),
                },
                "access_token": None,
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("via_proxy", [False, True])
async def test_readers_resolve_wiki_node_before_markdown_export(via_proxy):
    task = _task(requirement_text="See https://bytedance.larkoffice.com/wiki/WikiToken2")
    service, tokens, lark = _service(task)
    token = "WikiToken2" if via_proxy else "WikiToken1"
    url = f"https://bytedance.larkoffice.com/wiki/{token}"

    if via_proxy:
        markdown = await service.read_markdown(
            task_id=task.task_id, capability=service.issue(task), url=url,
        )
    else:
        markdown = await service.read_for_user(url, task.user_key)

    assert markdown == "# Exported Markdown\n\nBody"
    assert tokens.lookups == []
    assert lark.calls[0] == (
        "GET",
        "/wiki/v2/spaces/get_node",
        {
            "params": {"token": token},
            "access_token": None,
        },
    )
    assert lark.calls[1][1] == "/docs_ai/v1/documents/WikiDocToken/fetch"
    assert lark.calls[1][2]["access_token"] is None


@pytest.mark.asyncio
async def test_backend_reader_requires_authorization_only_after_app_acl_denial():
    service, tokens, lark = _service(user_token=None)
    lark._request = AsyncMock(side_effect=LarkError("app cannot read", code=131006))
    with pytest.raises(LarkAuthorizationRequired):
        await service.read_for_user(_task().requirement_url, "user-2")
    assert tokens.lookups == ["user-2"]
    lark._request.assert_awaited_once()
    assert lark._request.call_args.kwargs["access_token"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("via_proxy", [False, True])
@pytest.mark.parametrize("code", [131006, 99991672])
async def test_readers_preserve_permission_errors_after_both_identities_fail(via_proxy, code):
    task = _task()
    service, _, lark = _service(task)
    error = LarkError("permission denied", http_status=403, code=code)
    lark._request = AsyncMock(side_effect=error)
    with pytest.raises(LarkError) as exc_info:
        if via_proxy:
            await service.read_markdown(
                task_id=task.task_id,
                capability=service.issue(task),
                url="https://bytedance.larkoffice.com/docx/DocToken2",
            )
        else:
            await service.read_for_user(task.requirement_url, task.user_key)
    assert exc_info.value is error
    assert [c.kwargs["access_token"] for c in lark._request.await_args_list] == [
        None, "user-access-token",
    ]


@pytest.mark.asyncio
async def test_capability_rejects_tampering_expiry_and_unlisted_url():
    now = [1000.0]
    task = _task()
    service, _, _ = _service(task, clock=lambda: now[0])
    capability = service.issue(task)

    with pytest.raises(InvalidDocumentCapability):
        await service.read_markdown(
            task_id=task.task_id,
            capability=capability + "x",
            url=task.requirement_url,
        )

    for other_task in (_task(task_id="task-2"), _task(user_key="user-2")):
        other_service, _, _ = _service(other_task, clock=lambda: now[0])
        with pytest.raises(InvalidDocumentCapability):
            await other_service.read_markdown(
                task_id=other_task.task_id,
                capability=capability,
                url=other_task.requirement_url,
            )

    attacker_service, _, _ = _service(
        task, signing_secret=task.action_token, clock=lambda: now[0]
    )
    with pytest.raises(InvalidDocumentCapability):
        await service.read_markdown(
            task_id=task.task_id,
            capability=attacker_service.issue(task),
            url=task.requirement_url,
        )

    with pytest.raises(DocumentNotAllowed):
        await service.read_markdown(
            task_id=task.task_id,
            capability=capability,
            url="https://bytedance.larkoffice.com/docx/OtherToken",
        )

    now[0] += 61
    with pytest.raises(InvalidDocumentCapability):
        await service.read_markdown(
            task_id=task.task_id,
            capability=capability,
            url=task.requirement_url,
        )


@pytest.mark.asyncio
async def test_capability_requires_active_task_and_one_readable_identity():
    url = "https://bytedance.larkoffice.com/docx/DocToken2"
    inactive = _task(state=AeolusState.REVIEW)
    service, _, _ = _service(inactive)
    with pytest.raises(DocumentAccessInactive):
        await service.read_markdown(
            task_id=inactive.task_id,
            capability=service.issue(inactive),
            url=url,
        )

    task = _task()
    service, _, lark = _service(task, user_token=None)
    lark._request = AsyncMock(side_effect=LarkError("app cannot read", code=131006))
    with pytest.raises(LarkAuthorizationRequired) as exc_info:
        await service.read_markdown(
            task_id=task.task_id,
            capability=service.issue(task),
            url=url,
        )
    assert exc_info.value.authorization_url == "https://auth.example/user-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["docx", "wiki"])
@pytest.mark.parametrize("origin, suffix", [
    ("https://bytedance.larkoffice.com", ""),
    ("http://TEAM.feishu.cn", "?from=copy#details"),
])
async def test_proxy_rejects_primary_document_even_with_self_links(kind, origin, suffix):
    primary_url = f"https://bytedance.larkoffice.com/{kind}/MainDoc"
    task = _task(requirement_url=primary_url)
    task.requirement_text += f"\n[需求原文]({primary_url}#section)"
    service, tokens, lark = _service(task)

    instructions = service.instructions(task)
    assert "DocToken2" in instructions
    assert "MainDoc" not in instructions
    with pytest.raises(DocumentNotAllowed):
        await service.read_markdown(
            task_id=task.task_id,
            capability=service.issue(task),
            url=f"{origin}/{kind}/MainDoc{suffix}",
        )
    assert tokens.lookups == []
    assert lark.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["remove_reference", "make_reference_primary"])
async def test_existing_capability_rechecks_current_references(change):
    task = _task()
    service, tokens, lark = _service(task)
    url = "https://bytedance.larkoffice.com/docx/DocToken2"
    capability = service.issue(task)
    if change == "remove_reference":
        task.requirement_text = "更新后的需求不再引用外部文档。"
    else:
        task.requirement_url = url

    assert service.instructions(task) == ""
    with pytest.raises(DocumentNotAllowed):
        await service.read_markdown(
            task_id=task.task_id, capability=capability, url=url,
        )
    assert tokens.lookups == []
    assert lark.calls == []


def test_public_origin_is_derived_from_oauth_callback():
    assert public_origin_from_redirect_uri(
        "https://conciliator.example/api/aeolus/lark/callback"
    ) == "https://conciliator.example"
    with pytest.raises(ValueError):
        public_origin_from_redirect_uri("file:///tmp/callback")


@pytest.mark.asyncio
@pytest.mark.parametrize("via_proxy", [False, True])
@pytest.mark.parametrize("kind", ["docx", "wiki"])
@pytest.mark.parametrize("app_error", [
    LarkError("app ACL denied", code=131006),
    LarkError("app scope missing", code=99991672),
    TransientError("app temporarily unavailable"),
])
async def test_user_can_read_when_application_cannot(via_proxy, kind, app_error):
    url = f"https://bytedance.larkoffice.com/{kind}/DocToken2"
    task = _task(requirement_text=f"See {url}")
    service, tokens, lark = _service(task)
    request = lark._request

    async def respond(method, path, **kwargs):
        if kwargs["access_token"] is None:
            raise app_error
        return await request(method, path, **kwargs)

    lark._request = AsyncMock(side_effect=respond)
    if via_proxy:
        result = await service.read_markdown(
            task_id=task.task_id, capability=service.issue(task), url=url,
        )
    else:
        result = await service.read_for_user(url, task.user_key)

    assert result == "# Exported Markdown\n\nBody"
    assert tokens.lookups == [task.user_key]
    expected_tokens = [None, "user-access-token"]
    if kind == "wiki":
        expected_tokens.append("user-access-token")
    assert [c.kwargs["access_token"] for c in lark._request.await_args_list] == expected_tokens


@pytest.mark.asyncio
async def test_application_read_does_not_depend_on_user_token_store():
    service, tokens, _ = _service()
    tokens.user_access_token = AsyncMock(side_effect=TransientError("store unavailable"))
    result = await service.read_for_user(_task().requirement_url, "user-1")
    assert result == "# Exported Markdown\n\nBody"
    tokens.user_access_token.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("app_error", [
    LarkError("app scope missing", code=99991672),
    LarkError("app misconfigured", code=999),
    TransientError("app temporarily unavailable"),
])
async def test_non_acl_app_errors_without_user_token_do_not_request_oauth(app_error):
    service, _, lark = _service(user_token=None)
    lark._request = AsyncMock(side_effect=app_error)
    with pytest.raises(type(app_error)) as exc:
        await service.read_for_user(_task().requirement_url, "user-1")
    assert exc.value is app_error
    lark._request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("outage_identity", ["app", "user"])
async def test_outage_is_not_misreported_as_authorization_failure(outage_identity):
    service, _, lark = _service()
    outage = TransientError("temporarily unavailable")
    denied = LarkError("no document permission", code=131006)
    lark._request = AsyncMock(side_effect=(
        [outage, denied] if outage_identity == "app" else [denied, outage]
    ))
    with pytest.raises(TransientError) as exc:
        await service.read_for_user("https://bytedance.larkoffice.com/docx/DocToken2", "user-1")
    assert exc.value is outage
    assert lark._request.await_count == 2
