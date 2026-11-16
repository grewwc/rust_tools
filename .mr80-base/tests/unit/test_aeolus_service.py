import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from structlog.contextvars import merge_contextvars
from structlog.testing import capture_logs

import conciliator.services.aeolus_registry as registry_module
from conciliator.api.aeolus import _snapshot
from conciliator.domain.aeolus_task import AeolusStartEvent, AeolusState, AeolusTask, CodeContext
from conciliator.integrations.coco.aeolus import (
    CocoCodingRun,
    CocoRunFailed,
    CodingResult,
    EvaluationResult,
)
from conciliator.integrations.lark.client import LarkError
from conciliator.repositories.models import Base, MeegleAINodeTask
from conciliator.services.aeolus_comment_service import (
    AeolusCommentHandler,
    AeolusCommentService,
    CommentEvent,
)
from conciliator.services.aeolus_document_access import LarkAuthorizationRequired
from conciliator.services.aeolus_registry import (
    AeolusTaskRegistry,
    InvalidTaskTransition,
    TaskLeaseLost,
    task_lease,
)
from conciliator.services.aeolus_service import (
    AeolusService,
    NextCodeAeolusHost,
    PreparedBranch,
    ReviewRef,
    _review_contract_excerpt,
    _review_title,
)
from conciliator.utils.retry import TransientError


def _event(task_id: str = "idem-1") -> AeolusStartEvent:
    return AeolusStartEvent(
        task_id=task_id,
        project_key="space-1",
        work_item_id=42,
        state_key="state-1",
        node_uuid="node-uuid-1",
        user_key="user-1",
        work_item_type_key="story",
        props=(
            {
                "prop_key": "design_doc",
                "prop_value": {
                    "field": [
                        {
                            "field_type": "link_cloud_doc",
                            "field_value": [
                                "https://example.feishu.cn/docx/doccnABC"
                            ],
                        }
                    ]
                },
            },
        ),
    )


class FakeRunner:
    def __init__(self, evaluation: EvaluationResult):
        self.evaluation = evaluation
        self.feedbacks: list[str] = []
        self.code_inputs: list[str] = []
        self.evaluation_inputs: list[tuple[str, str]] = []
        self.session_contexts = []

    async def evaluate(
        self, text: str, *, history=None, session_id="", on_started=None,
        document_access_instructions="",
        session_context=None,
    ) -> EvaluationResult:
        self.evaluation_inputs.append((text, document_access_instructions))
        if session_context is not None:
            self.session_contexts.append(session_context)
        if on_started:
            await on_started(session_id or "evaluation-session", "evaluation-run", "evaluation-event")
        return self.evaluation

    async def code(self, **kwargs) -> CocoCodingRun:
        self.code_inputs.append(kwargs["requirement_url"])
        self.session_contexts.append(kwargs["session_context"])
        on_started = kwargs.get("on_started")
        if on_started:
            await on_started("session-1", "run-1", "event-1")
        return CocoCodingRun(
            session_id="session-1",
            run_id="run-1",
            last_event_id="event-9",
            result=CodingResult(
                "success", "首轮完成", "a" * 40, ("pytest -q",), ()
            ),
        )

    async def recode(
        self, *, session_id: str, feedback: str, on_started=None
    ) -> CocoCodingRun:
        assert session_id == "session-1"
        self.feedbacks.append(feedback)
        if on_started:
            await on_started("session-1", "run-2", "event-11")
        return CocoCodingRun(
            session_id="session-1",
            run_id="run-2",
            last_event_id="event-19",
            result=CodingResult(
                "success", "反馈已处理", "b" * 40, ("pytest targeted -q",), ()
            ),
        )


class FailingRunner(FakeRunner):
    async def code(self, **kwargs) -> CocoCodingRun:
        await kwargs["on_started"]("session-failed", "run-failed", "event-start")
        raise CocoRunFailed("Coco run terminated", retryable=True)


class RecoveringFeedbackRunner(FakeRunner):
    def __init__(self, evaluation: EvaluationResult):
        super().__init__(evaluation)
        self.code_calls = 0
        self.recovery_prompt = ""

    async def code(self, **kwargs) -> CocoCodingRun:
        self.code_calls += 1
        if self.code_calls == 1:
            return await super().code(**kwargs)
        self.recovery_prompt = kwargs["requirement_url"]
        await kwargs["on_started"]("session-2", "run-3", "event-21")
        return CocoCodingRun(
            session_id="session-2",
            run_id="run-3",
            last_event_id="event-29",
            result=CodingResult(
                "success", "恢复后完成", "c" * 40, ("pytest recovery -q",), ()
            ),
        )

    async def recode(self, *, session_id: str, feedback: str, on_started=None):
        self.feedbacks.append(feedback)
        await on_started("session-1", "run-2", "event-11")
        raise CocoRunFailed("session terminated", retryable=True)


class FakeCodeHost:
    def __init__(self):
        self.verified: list[tuple[str, str, str]] = []
        self.reviews = 0
        self.review_titles: list[str] = []
        self.review_contracts: list[str] = []
        self.prepared_contexts: list[CodeContext] = []
        self.review_targets: list[tuple[str, str]] = []

    async def resolve_context(self, context: CodeContext) -> CodeContext:
        return replace(context, base_commit=context.base_commit or "a" * 40)

    async def prepare_branch(self, task_id: str, context: CodeContext) -> PreparedBranch:
        self.prepared_contexts.append(context)
        return PreparedBranch("repo-1", f"aeolus/ai-node/{task_id}")

    async def verify_commit(self, repo_id: str, branch: str, commit_sha: str) -> None:
        self.verified.append((repo_id, branch, commit_sha))

    async def create_review(
        self, *, repo_id: str, branch: str, task_id: str, title: str, target_branch: str,
        requirement_contract: str = ""
    ) -> ReviewRef:
        self.reviews += 1
        self.review_titles.append(title)
        self.review_contracts.append(requirement_contract)
        self.review_targets.append((repo_id, target_branch))
        return ReviewRef("mr-id", 7, "https://code.example/mr/7")


class FakeMeegle:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def get_work_item_name(self, ref, work_item_type_key):
        return "优化查询性能"

    async def query_users(self, ref, user_keys):
        return [{"user_key": key, "email": "tester@example.com"} for key in user_keys]

    async def edit_status(self, ref, status: str, *, idem_key: str) -> None:
        self.calls.append(("edit", status))

    async def confirm(self, ref, *, idem_key: str) -> None:
        self.calls.append(("confirm", idem_key))


class FailConfirmOnceMeegle(FakeMeegle):
    def __init__(self):
        super().__init__()
        self.fail = True

    async def confirm(self, ref, *, idem_key: str) -> None:
        self.calls.append(("confirm", idem_key))
        if self.fail:
            self.fail = False
            raise RuntimeError("temporary confirm failure")


def _service(
    evaluation: EvaluationResult, *, notify=None, engine=None, document_access=None
):
    if engine is None:
        # Registry calls run in worker threads. Serialize checkout of the one
        # in-memory connection so a scan cannot roll back a worker's lease.
        # Tests needing independent concurrent transactions pass a file engine.
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
        )
    Base.metadata.create_all(engine)
    registry = AeolusTaskRegistry(
        sessionmaker(bind=engine, expire_on_commit=False)
    )
    runner = FakeRunner(evaluation)
    host = FakeCodeHost()
    meegle = FakeMeegle()

    async def read_document(url: str, user_key: str) -> str:
        assert url.endswith("doccnABC")
        return "需求正文"

    service = AeolusService(
        registry=registry,
        document_reader=read_document,
        coco=runner,
        code_host=host,
        meegle=meegle,
        minimum_processing_seconds=0,
        notify=notify,
        document_access=document_access,
    )
    return service, registry, runner, host, meegle


@pytest.mark.asyncio
async def test_new_sessions_query_names_and_persist_only_work_item_type():
    service, registry, runner, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    meegle.get_work_item_name = AsyncMock(side_effect=["优化查询性能", "更新后的任务名称"])
    await service.accept_event(_event())
    await service.wait_for_idle()
    task = await AeolusTaskRegistry(registry._sf).get("idem-1")
    assert task.work_item_type_key == "story"
    assert runner.session_contexts[0].metadata("evaluation")["title"] == "tester | 评估 | 优化查询性能"
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    assert runner.session_contexts[1].metadata("coding")["title"] == "tester | 编码 | 更新后的任务名称"
    assert meegle.get_work_item_name.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("failed_query", "title"), [
    ("get_work_item_name", "tester | 评估 | 工作项 #42"),
    ("query_users", "user-1 | 评估 | 优化查询性能"),
])
async def test_name_lookup_failure_falls_back_without_blocking_evaluation(failed_query, title):
    service, registry, runner, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    setattr(meegle, failed_query, AsyncMock(side_effect=RuntimeError("lookup failed")))
    await service.accept_event(_event())
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.EVALUATION_REVIEW
    assert runner.session_contexts[0].metadata("evaluation")["title"] == title


@pytest.mark.asyncio
@pytest.mark.parametrize(("users", "prefix"), [
    ([{"user_key": "user-1", "email": "tester@example.com", "name_en": "English"}], "tester"),
    ([{"user_key": "user-1", "email": " ", "name_en": " English ", "name_cn": "中文"}], "English"),
    ([{"user_key": "user-1", "name_cn": "中文"}], "中文"),
    ([{"user_key": "user-1"}], "user-1"),
    ([{"user_key": "other", "email": "other@example.com"}], "user-1"),
    ([{"user_key": "user-1"}, {"user_key": "user-1", "email": "other@example.com"}], "user-1"),
])
async def test_session_user_prefix_requires_exact_directory_match(users, prefix):
    service, registry, _, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    task, _ = await registry.create_from_event(_event())
    meegle.query_users = AsyncMock(return_value=users)
    context = await service._coco_session_context(task)
    assert context.metadata("coding")["title"] == f"{prefix} | 编码 | 优化查询性能"
    assert context.initiator_user_key == task.user_key
    meegle.query_users.assert_awaited_once_with(
        ANY, [task.user_key],
    )


@pytest.mark.asyncio
async def test_session_names_are_queried_concurrently():
    service, registry, _, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    task, _ = await registry.create_from_event(_event())
    item_started, user_started = asyncio.Event(), asyncio.Event()

    async def item_name(*args):
        item_started.set()
        await user_started.wait()
        return "任务名称"

    async def users(*args):
        user_started.set()
        await item_started.wait()
        return [{"user_key": task.user_key, "email": "tester@example.com"}]

    meegle.get_work_item_name = AsyncMock(side_effect=item_name)
    meegle.query_users = AsyncMock(side_effect=users)
    context = await asyncio.wait_for(service._coco_session_context(task), timeout=2)
    assert context.metadata("coding")["title"] == "tester | 编码 | 任务名称"


@pytest.mark.asyncio
@pytest.mark.parametrize(("slow_query", "title"), [
    ("get_work_item_name", "tester | 编码 | 工作项 #42"),
    ("query_users", "user-1 | 编码 | 优化查询性能"),
])
async def test_session_name_timeout_preserves_other_query_result(monkeypatch, slow_query, title):
    service, registry, _, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    task, _ = await registry.create_from_event(_event())
    timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, "timeout", lambda seconds: timeout(0.01))

    async def pending(*args):
        await asyncio.Event().wait()

    setattr(meegle, slow_query, AsyncMock(side_effect=pending))
    context = await service._coco_session_context(task)
    assert context.metadata("coding")["title"] == title


@pytest.mark.asyncio
async def test_legacy_task_without_type_skips_item_query():
    service, registry, _, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    task, _ = await registry.create_from_event(replace(_event(), work_item_type_key=""))
    meegle.get_work_item_name = AsyncMock()
    context = await service._coco_session_context(task)
    assert context.metadata("evaluation")["title"] == "tester | 评估 | 工作项 #42"
    meegle.get_work_item_name.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario, expected_state, needs_oauth", [
    ("app_only", AeolusState.EVALUATION_REVIEW, False),
    ("user_only", AeolusState.EVALUATION_REVIEW, False),
    ("mixed", AeolusState.EVALUATION_REVIEW, False),
    ("no_main_access", AeolusState.VALIDATION_FAILED, True),
    ("no_reference_access", AeolusState.VALIDATION_FAILED, True),
    ("both_acl_denied", AeolusState.VALIDATION_FAILED, False),
])
async def test_evaluation_uses_either_identity_before_requesting_oauth(
    scenario, expected_state, needs_oauth,
):
    from conciliator.services.aeolus_document_access import AeolusDocumentAccessService

    document = '需求正文 <cite doc-id="Reference" file-type="docx" type="doc">引用</cite>'
    user_token = "user-token" if scenario in {"user_only", "mixed", "both_acl_denied"} else None
    tokens = SimpleNamespace(
        user_access_token=AsyncMock(return_value=user_token),
        authorize_url=lambda user: f"https://service.example/oauth/lark/start?user_key={user}",
    )

    async def request(_method, path, **kwargs):
        main_document = "doccnABC" in path
        if kwargs["access_token"] is None:
            allowed = scenario == "app_only" or (
                main_document and scenario in {"mixed", "no_reference_access"}
            )
        else:
            allowed = scenario in {"user_only", "mixed"}
        if not allowed:
            raise LarkError("no document permission", code=131006)
        return {"document": {"content": document if main_document else "引用正文"}}

    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    lark = SimpleNamespace(_request=AsyncMock(side_effect=request))
    access = AeolusDocumentAccessService(
        registry=registry, user_tokens=tokens, lark_client=lark,
        signing_secret="test-secret", public_base_url="https://service.example",
    )
    service._document_access = access
    service._read_document = access.read_for_user
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await service.get("idem-1")
    assert task.state == expected_state
    assert bool(task.evaluation.get("document_authorization_required")) is needs_oauth
    if expected_state == AeolusState.EVALUATION_REVIEW:
        assert runner.evaluation_inputs[0][0] == document
        assert any("Reference" in c.args[1] for c in lark._request.await_args_list)
    else:
        assert runner.evaluation_inputs == []
    if scenario == "app_only":
        tokens.user_access_token.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_enabled", [False, True])
@pytest.mark.parametrize("document", [
    "需求正文",
    "# 需求\n[引用](https://example.feishu.cn/docx/Reference)\n\n"
    + "| 场景 | 行为 |\n|---|---|\n| 正常 | 保持兼容 |\n" * 1024
    + "\n末尾验收标准：异常路径也必须覆盖。",
])
async def test_evaluation_always_receives_full_backend_snapshot(proxy_enabled, document):
    class DocumentAccess:
        def instructions(self, task):
            assert task.requirement_url == "https://example.feishu.cn/docx/doccnABC"
            assert task.requirement_text == document
            return "DOC_PROXY_INSTRUCTIONS"

    document_access = DocumentAccess() if proxy_enabled else None
    service, _, runner, _, _ = _service(
        EvaluationResult(True, "通过", (), ()),
        document_access=document_access,
    )
    service._read_document = AsyncMock(return_value=document)

    await service.accept_event(_event())
    await service.wait_for_idle()

    expected_instructions = "DOC_PROXY_INSTRUCTIONS" if proxy_enabled else ""
    assert runner.evaluation_inputs == [(document, expected_instructions)]
    expected_reads = [("https://example.feishu.cn/docx/doccnABC", "user-1")]
    if proxy_enabled and "Reference" in document:
        expected_reads.append(("https://example.feishu.cn/docx/Reference", "user-1"))
    assert [call.args for call in service._read_document.await_args_list] == expected_reads
    task = await service.get("idem-1")
    assert task.requirement_text == document
    assert task.state == AeolusState.EVALUATION_REVIEW


@pytest.mark.asyncio
async def test_reference_precheck_reports_all_permission_failures_before_evaluation():
    service, _, runner, _, _ = _service(
        EvaluationResult(True, "通过", (), ()),
        document_access=SimpleNamespace(instructions=lambda task: ""),
    )
    references = [f"https://example.feishu.cn/docx/Reference{i}" for i in range(10)]
    document = "\n".join(f"[引用 {i}]({url})" for i, url in enumerate(references))
    checked: list[str] = []

    async def read_document(url: str, user_key: str) -> str:
        assert user_key == "user-1"
        if url.endswith("doccnABC"):
            return document
        checked.append(url)
        if int(url.rsplit("Reference", 1)[1]) % 2:
            raise LarkError("forbidden", http_status=403, code=131006)
        return "可读引用"

    service._read_document = read_document
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await service.get("idem-1")
    assert set(checked) == set(references)
    assert runner.evaluation_inputs == []
    assert task.state == AeolusState.VALIDATION_FAILED
    assert task.evaluation["document_authorization_required"] is False
    precheck = task.evaluation["document_precheck"]
    assert precheck["total_documents"] == 10
    assert [item["url"] for item in precheck["failed_documents"]] == references[1::2]
    assert {item["reason"] for item in precheck["failed_documents"]} == {"permission_denied"}


@pytest.mark.parametrize(
    ("code", "message", "expected_read_error_code"),
    [
        (131006, "应用和当前用户均无法读取该文档", 131006),
        (3380004, "应用无该文档权限", 3380004),
    ],
)
@pytest.mark.asyncio
async def test_reference_precheck_classifies_document_acl_codes(
    code, message, expected_read_error_code,
):
    service, _, runner, _, _ = _service(
        EvaluationResult(True, "通过", (), ()),
        document_access=SimpleNamespace(instructions=lambda task: ""),
    )
    reference = "https://example.feishu.cn/docx/Reference"
    document = f"[引用]({reference})"

    async def read_document(url: str, user_key: str) -> str:
        assert user_key == "user-1"
        if url.endswith("doccnABC"):
            return document
        raise LarkError("forbidden", http_status=403, code=code)

    service._read_document = read_document
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await service.get("idem-1")
    assert runner.evaluation_inputs == []
    assert task.state == AeolusState.VALIDATION_FAILED
    precheck = task.evaluation["document_precheck"]
    assert precheck["failed_documents"][0]["reason"] == "permission_denied"
    assert message in precheck["failed_documents"][0]["message"]
    assert task.evaluation["document_read_error_code"] == expected_read_error_code
    assert task.evaluation["document_authorization_required"] is False


@pytest.mark.asyncio
async def test_reference_precheck_reports_expired_oauth_without_acl_failures():
    service, _, runner, _, _ = _service(
        EvaluationResult(True, "通过", (), ()),
        document_access=SimpleNamespace(instructions=lambda task: ""),
    )
    reference = "https://example.feishu.cn/docx/Reference"

    async def read_document(url: str, _user_key: str) -> str:
        if url.endswith("doccnABC"):
            return f"[引用]({reference})"
        raise LarkAuthorizationRequired("https://auth.example/start")

    service._read_document = read_document
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await service.get("idem-1")
    assert runner.evaluation_inputs == []
    assert task.state == AeolusState.VALIDATION_FAILED
    assert task.evaluation["document_authorization_required"] is True
    assert task.evaluation["document_precheck"] == {
        "status": "authorization_required",
        "total_documents": 1,
        "failed_documents": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_enabled", [False, True])
@pytest.mark.parametrize("error", [
    ValueError("wiki node points to a sheet"),
    ValueError("document has no readable content"),
])
async def test_invalid_reference_is_structured_only_with_operator_access(proxy_enabled, error):
    access = SimpleNamespace(instructions=lambda task: "") if proxy_enabled else None
    service, _, runner, _, _ = _service(
        EvaluationResult(True, "通过", (), ()), document_access=access,
    )
    reference = "https://example.feishu.cn/wiki/Reference"
    document = f"需求正文\n[引用]({reference})"
    service._read_document = AsyncMock(side_effect=[document, error])

    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await service.get("idem-1")
    assert task.requirement_text == document
    if not proxy_enabled:
        assert service._read_document.await_count == 1
        assert task.state == AeolusState.EVALUATION_REVIEW
        assert runner.evaluation_inputs == [(document, "")]
        return
    assert service._read_document.await_count == 2
    assert task.state == AeolusState.VALIDATION_FAILED
    assert runner.evaluation_inputs == []
    assert task.evaluation["document_authorization_required"] is False
    precheck = task.evaluation["document_precheck"]
    assert precheck["status"] == "failed"
    assert precheck["total_documents"] == 1
    assert precheck["failed_documents"][0]["url"] == reference
    assert precheck["failed_documents"][0]["reason"] == "invalid_document"
    assert "为空" in precheck["failed_documents"][0]["message"]
    assert task.evaluation_rounds[-1]["result"] == task.evaluation


@pytest.mark.asyncio
async def test_disabled_oauth_does_not_precheck_unshared_references():
    service, _, runner, host, _ = _service(EvaluationResult(True, "通过", (), ()))
    document = "需求正文\n[引用](https://example.feishu.cn/docx/Unshared)"
    service._read_document = AsyncMock(side_effect=[
        document, LarkError("permission denied", http_status=400, code=131006),
    ])

    await service.accept_event(_event())
    await service.wait_for_idle()

    assert service._read_document.await_count == 1
    assert (await service.get("idem-1")).state == AeolusState.EVALUATION_REVIEW
    assert runner.evaluation_inputs == [(document, "")]
    assert runner.code_inputs == []
    assert host.prepared_contexts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable, expected_calls, expected_state", [
    (True, 2, AeolusState.EVALUATION_REVIEW),
    (False, 1, AeolusState.VALIDATION_FAILED),
])
async def test_evaluation_replaces_only_terminal_runs_without_agent_results(
    retryable, expected_calls, expected_state,
):
    evaluation = EvaluationResult(True, "通过", (), ())
    service, _, runner, _, _ = _service(evaluation)
    calls = 0

    async def evaluate(text, **kwargs):
        nonlocal calls
        calls += 1
        await kwargs["on_started"](
            f"evaluation-session-{calls}", f"evaluation-run-{calls}",
            f"evaluation-event-{calls}",
        )
        if calls == 1:
            raise CocoRunFailed("run terminated", retryable=retryable)
        return evaluation

    runner.evaluate = AsyncMock(side_effect=evaluate)

    await service.accept_event(_event())
    await service.wait_for_idle()

    assert calls == expected_calls
    assert (await service.get("idem-1")).state == expected_state


@pytest.mark.asyncio
async def test_restarting_a_node_reuses_its_task_without_replaying_evaluation():
    service, _, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    first, created = await service.accept_event(_event())
    await service.wait_for_idle()

    restarted, restarted_created = await service.accept_event(_event(task_id="idem-2"))
    await service.wait_for_idle()

    assert created is True
    assert restarted_created is False
    assert restarted.task_id == first.task_id
    assert len(runner.evaluation_inputs) == 1
    assert (await service.get("idem-1")).state == AeolusState.EVALUATION_REVIEW


@pytest.mark.asyncio
async def test_restart_event_resumes_a_task_stuck_in_received():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await registry.create_from_event(_event())

    task, created = await service.accept_event(_event(task_id="idem-2"))
    await service.wait_for_idle()

    assert created is False
    assert task.task_id == "idem-1"
    assert len(runner.evaluation_inputs) == 1
    assert (await service.get("idem-1")).state == AeolusState.EVALUATION_REVIEW


@pytest.mark.asyncio
async def test_restart_after_a_round_committed_without_a_run_starts_coding():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()

    # Simulate a process that died right after committing CODING: the round
    # exists, but no attempt (and so no resume cursor) was ever recorded.
    task, started = await registry.review_evaluation("idem-1", "req-1")
    assert started is True
    assert task.state == AeolusState.CODING
    assert not task.rounds[-1].attempts
    assert "idem-1" in registry._recoverable_task_ids()

    restarted, created = await service.accept_event(_event(task_id="idem-2"))
    await service.wait_for_idle()

    assert created is False
    assert restarted.task_id == "idem-1"
    assert len(runner.code_inputs) == 1
    assert (await service.get("idem-1")).state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_recovery_scan_starts_a_committed_round_without_a_run():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await registry.review_evaluation("idem-1", "req-1")

    await service.recover_pending()
    await service.wait_for_idle()

    assert len(runner.code_inputs) == 1
    assert (await service.get("idem-1")).state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_recovery_scan_keeps_the_feedback_of_a_pending_round():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "req-1")
    await service.wait_for_idle()
    assert (await service.get("idem-1")).state == AeolusState.REVIEW

    # Simulate a process that died after committing the feedback round but
    # before its first run; the round itself carries the requested changes.
    task, started = await registry.start_feedback("idem-1", "把日志级别改成 debug", "req-2")
    assert started is True
    assert task.state == AeolusState.RECODING
    assert not task.rounds[-1].attempts

    await service.recover_pending()
    await service.wait_for_idle()

    assert runner.feedbacks == ["把日志级别改成 debug"]
    assert (await service.get("idem-1")).state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_acked_feedback_is_resolved_when_the_running_round_fails():
    """A comment acknowledged during RECODING must not outlive a failed task."""
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "req-1")
    await service.wait_for_idle()

    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_recode(*, session_id: str, feedback: str, on_started=None):
        assert session_id == "session-1"
        await on_started("session-1", "run-2", "event-11")
        started.set()
        await release.wait()
        raise CocoRunFailed("Coco run terminated", retryable=False)

    runner.recode = failing_recode
    await service.submit_feedback("idem-1", "第一条反馈", "req-2")
    await started.wait()

    stored = await service.save_pending_feedback(
        "idem-1",
        [("comment-2", "第二条反馈")],
        available_at=datetime.now(UTC),
    )
    assert stored is True
    assert await registry.pending_feedback("idem-1") == [("comment-2", "第二条反馈")]

    release.set()
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert "1 条已确认的评论反馈" in task.rounds[-1].error
    assert await registry.pending_feedback("idem-1") == []

    # The recovery scan must not resurrect or keep retrying the resolved rows.
    await service.recover_pending()
    await service.wait_for_idle()
    assert await registry.pending_feedback("idem-1") == []


@pytest.mark.asyncio
async def test_feedback_acked_while_the_failure_is_recorded_is_not_dropped():
    """A comment saved during failure handling must still reach a terminal state."""
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "req-1")
    await service.wait_for_idle()

    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_recode(*, session_id: str, feedback: str, on_started=None):
        await on_started("session-1", "run-2", "event-11")
        started.set()
        await release.wait()
        raise CocoRunFailed("Coco run terminated", retryable=False)

    runner.recode = failing_recode
    await service.submit_feedback("idem-1", "第一条反馈", "req-2")
    await started.wait()

    # The comment is acknowledged after the failure began but before the task
    # turns FAILED: the window a snapshot taken before the transition misses.
    record_failure = service._fail_running_attempt

    async def save_then_fail(task_id: str, exc: Exception) -> None:
        assert await service.save_pending_feedback(
            task_id, [("comment-2", "第二条反馈")], available_at=datetime.now(UTC)
        )
        await record_failure(task_id, exc)

    service._fail_running_attempt = save_then_fail

    release.set()
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert "1 条已确认的评论反馈" in task.last_error
    assert "1 条已确认的评论反馈" in task.rounds[-1].error
    assert await registry.pending_feedback("idem-1") == []

    await service.recover_pending()
    await service.wait_for_idle()
    assert await registry.pending_feedback("idem-1") == []


@pytest.mark.asyncio
async def test_redelivered_later_comment_does_not_start_another_round():
    service, _, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "req-1")
    await service.wait_for_idle()

    first = await service.submit_feedback(
        "idem-1",
        "改成批量化\n\n还有这里",
        ["mr-comment:c-1", "mr-comment:c-2"],
    )
    await service.wait_for_idle()
    assert (await service.get("idem-1")).state == AeolusState.REVIEW

    redelivered = await service.submit_feedback(
        "idem-1", "还有这里", ["mr-comment:c-2"]
    )

    assert redelivered.state == AeolusState.REVIEW
    assert len(redelivered.rounds) == len(first.rounds)
    assert runner.feedbacks == ["改成批量化\n\n还有这里"]


@pytest.mark.asyncio
async def test_stale_initial_coding_does_not_start_a_second_run():
    service, _, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "req-1")
    await service.wait_for_idle()
    assert (await service.get("idem-1")).state == AeolusState.REVIEW
    assert len(runner.code_inputs) == 1

    # A duplicate spawner can still be queued when the first run already moved
    # the task to Review; it must not open a second Coco session.
    await service._run_initial_coding("idem-1", EvaluationResult(True, "通过", (), ()))

    assert len(runner.code_inputs) == 1


@pytest.mark.asyncio
async def test_refreshed_props_round_trip_as_domain_tuple():
    service, registry, runner, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    new_url = "https://example.feishu.cn/docx/Updated"
    fresh = [{"prop_key": "design_doc", "prop_value": {"field": [{
        "field_type": "link_cloud_doc", "field_value": [new_url],
    }]}}]
    meegle.query = AsyncMock(return_value={"ai_info": {"props": fresh}})
    service._read_document = AsyncMock(return_value="更新的需求")

    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await AeolusTaskRegistry(registry._sf).get("idem-1")
    assert task.props == tuple(fresh)
    assert task.requirement_url == new_url
    assert task.state == AeolusState.EVALUATION_REVIEW
    service._read_document.assert_awaited_once_with(new_url, "user-1")
    assert runner.evaluation_inputs == [("更新的需求", "")]


@pytest.mark.asyncio
async def test_task_timing_tracks_all_transition_paths_and_survives_restart(monkeypatch):
    base = datetime(2026, 10, 12, tzinfo=UTC)
    seconds = 0

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return base + timedelta(seconds=seconds)

    monkeypatch.setattr(registry_module, "datetime", Clock)
    _, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task, _ = await registry.create_from_event(_event())
    assert task.timing["automatic_ms"] == 0
    seconds = 12
    await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.EVALUATION_REVIEW,
                              evaluation={"valid": True})
    seconds = 17
    await registry.update("idem-1", last_error="")
    await registry.transition("idem-1", {AeolusState.EVALUATION_REVIEW}, AeolusState.EVALUATION_REVIEW)
    seconds = 22
    await registry.review_evaluation("idem-1", "confirm-evaluation")
    seconds = 25
    _, repeated = await registry.review_evaluation("idem-1", "confirm-evaluation")
    assert repeated is False
    seconds = 32
    await registry.transition("idem-1", {AeolusState.CODING}, AeolusState.FAILED)
    seconds = 42
    await registry.start_coding_retry("idem-1", "retry-coding")
    seconds = 52
    await registry.transition("idem-1", {AeolusState.CODING}, AeolusState.REVIEW)
    seconds = 62
    await registry.start_feedback("idem-1", "修改代码", "coding-feedback")
    seconds = 72
    await registry.transition("idem-1", {AeolusState.RECODING}, AeolusState.REVIEW)
    seconds = 82
    await registry.transition("idem-1", {AeolusState.REVIEW}, AeolusState.COMPLETING)
    seconds = 83
    await registry.transition("idem-1", {AeolusState.COMPLETING}, AeolusState.COMPLETED)
    registry = AeolusTaskRegistry(registry._sf)
    task = await registry.get("idem-1")
    assert {key: value for key, value in task.timing.items() if key not in {"rounds", "current_round"}} == {
        "automatic_ms": 43_000, "human_wait_ms": 40_000, "retry_wait_ms": 10_000,
        "current_phase": "completed", "phase_started_at": (base + timedelta(seconds=83)).isoformat(),
        "completed_at": (base + timedelta(seconds=83)).isoformat(),
    }
    assert task.timing["current_round"] == "coding:2"
    for key, automatic, waiting, retry, started, finished in [
        ("evaluation:1", 12, 10, 0, 0, 22),
        ("coding:1", 20, 20, 10, 22, 62),
        ("coding:2", 11, 10, 0, 62, 83),
    ]:
        assert task.timing["rounds"][key] == {
            "automatic_ms": automatic * 1000, "human_wait_ms": waiting * 1000,
            "retry_wait_ms": retry * 1000, "current_phase": "completed",
            "started_at": (base + timedelta(seconds=started)).isoformat(),
            "phase_started_at": (base + timedelta(seconds=finished)).isoformat(),
            "completed_at": (base + timedelta(seconds=finished)).isoformat(),
        }
    seconds = 100
    await registry.update("idem-1", last_error="")
    assert (await registry.get("idem-1")).timing == task.timing
    assert _snapshot(task)["timing"] == task.timing


@pytest.mark.asyncio
async def test_evaluation_retry_wait_and_legacy_tasks_are_not_backfilled(monkeypatch):
    base = datetime(2026, 10, 12, tzinfo=UTC)
    seconds = 0

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return base + timedelta(seconds=seconds)

    monkeypatch.setattr(registry_module, "datetime", Clock)
    _, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await registry.create_from_event(_event())
    seconds = 10
    await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.VALIDATION_FAILED,
                              evaluation_rounds=[{"round_no": 1, "status": "error"}])
    seconds = 30
    task, _ = await registry.review_evaluation("idem-1", "retry-evaluation", feedback="重新处理")
    assert task.timing["automatic_ms"] == 10_000
    assert task.timing["human_wait_ms"] == task.timing["retry_wait_ms"] == 20_000
    assert task.timing["current_phase"] == "automatic"
    assert task.timing["rounds"]["evaluation:1"]["automatic_ms"] == 10_000
    assert task.timing["rounds"]["evaluation:1"]["human_wait_ms"] == 20_000
    assert task.timing["rounds"]["evaluation:1"]["completed_at"] == (base + timedelta(seconds=30)).isoformat()
    assert task.timing["rounds"]["evaluation:2"]["automatic_ms"] == 0
    with registry._sf() as session:
        session.query(MeegleAINodeTask).one().timing = None
        session.commit()
    await registry.transition("idem-1", {AeolusState.READING_DOCUMENT}, AeolusState.EVALUATING)
    task = await registry.get("idem-1")
    assert task.timing is None
    assert _snapshot(task)["timing"] is None


@pytest.mark.asyncio
async def test_legacy_overall_timing_starts_tracking_only_new_rounds():
    _, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await registry.create_from_event(_event())
    with registry._sf() as session:
        record = session.query(MeegleAINodeTask).one()
        record.timing = {key: value for key, value in record.timing.items()
                         if key not in {"rounds", "current_round"}}
        session.commit()
    task = await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.VALIDATION_FAILED,
                                     evaluation_rounds=[{"round_no": 1, "status": "error"}])
    assert "rounds" not in task.timing
    task, _ = await registry.review_evaluation("idem-1", "retry", feedback="重新评估")
    assert list(task.timing["rounds"]) == ["evaluation:2"]


@pytest.mark.asyncio
async def test_task_lease_exclusion_expiry_and_stale_writer():
    _, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await registry.create_from_event(_event())
    assert await registry.acquire_lease("idem-1", "first")
    assert not await registry.acquire_lease("idem-1", "second")
    assert await registry.renew_lease("idem-1", "first")
    with registry._sf() as session:
        record = session.query(MeegleAINodeTask).one()
        record.lease_expires_at = datetime.now() - timedelta(days=1)
        session.commit()
    assert not await registry.renew_lease("idem-1", "first")
    assert await registry.acquire_lease("idem-1", "second")
    with task_lease("first"), pytest.raises(TaskLeaseLost):
        await registry.update("idem-1", last_error="stale")
    await registry.release_lease("idem-1", "first")
    assert not await registry.acquire_lease("idem-1", "third")
    await registry.release_lease("idem-1", "second")
    assert await registry.acquire_lease("idem-1", "third")


def test_recovery_scan_cannot_roll_back_an_uncommitted_task_lease():
    _, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    registry._create_from_event(_event())
    engine = registry._sf.kw["bind"]
    lease_written = threading.Event()
    scan_finished = threading.Event()

    def pause_before_commit(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE meegle_ai_node_tasks SET lease_owner="):
            lease_written.set()
            # StaticPool lets the scan reuse this connection and roll back the
            # lease. A checked-out QueuePool connection keeps the scan waiting
            # until this bounded pause ends and the lease transaction commits.
            scan_finished.wait(timeout=1)

    event.listen(engine, "after_cursor_execute", pause_before_commit)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            acquisition = executor.submit(
                registry._change_lease, "idem-1", "worker", 60, False
            )
            try:
                assert lease_written.wait(timeout=5)
                registry._pending_feedback_task_ids(datetime.now(UTC))
            finally:
                scan_finished.set()
            assert acquisition.result(timeout=5)
        with task_lease("worker"):
            assert registry._get("idem-1").task_id == "idem-1"
    finally:
        event.remove(engine, "after_cursor_execute", pause_before_commit)
        engine.dispose()


@pytest.mark.asyncio
async def test_failed_renewal_cancels_polling_without_failing_the_run(monkeypatch):
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def blocked(**kwargs):
        await kwargs["on_started"]("session-1", "run-1", "event-start")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runner.code = blocked

    async def failed_renewal(*args):
        await started.wait()
        return False

    registry.renew_lease = AsyncMock(side_effect=failed_renewal)
    monkeypatch.setattr("conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 0.01)
    await service.confirm_evaluation("idem-1", "confirm")
    await asyncio.wait_for(service.wait_for_idle(), timeout=2)
    assert cancelled.is_set()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.CODING
    assert task.rounds[-1].attempts[-1].status == "running"
    assert not task.last_error
    assert await registry.recoverable_task_ids() == ["idem-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("contract", ["# 需求共识\n\n## 验收标准\n- pytest 通过", ""])
async def test_review_creation_uses_title_and_reuses_existing_branch_mr(
    monkeypatch, existing, contract,
):
    prefix = "conciliator.services.aeolus_service."
    monkeypatch.setattr(prefix + "get_repository", AsyncMock(return_value=SimpleNamespace(id="repo")))
    mr = SimpleNamespace(id="mr", iid=7, web_url="https://code.example/mr/7")
    listing = AsyncMock(return_value=[mr] if existing else [])
    creation = AsyncMock(return_value=mr)
    monkeypatch.setattr(prefix + "list_repo_mrs", listing)
    monkeypatch.setattr(prefix + "create_mr", creation)
    client = AsyncMock()
    review = await NextCodeAeolusHost(client).create_review(
        repo_id="repo", branch="coding", task_id="idem-1", title="[Aeolus AI] 需求标题",
        target_branch="release/demo",
        requirement_contract=contract,
    )
    assert review.iid == 7
    client.call.assert_not_awaited()
    listing.assert_awaited_once_with(client, "repo", source_branch="coding", target_branch="release/demo")
    if existing:
        creation.assert_not_awaited()
    else:
        assert creation.call_args.kwargs["title"] == "[Aeolus AI] 需求标题"
        assert creation.call_args.args[1] == "repo"
        assert creation.call_args.kwargs["target"] == "release/demo"
        assert "idem-1" in creation.call_args.kwargs["description"]
        description = creation.call_args.kwargs["description"]
        if contract:
            assert description.endswith("## 已确认的需求要点\n\n## 验收标准\n- pytest 通过")
        else:
            assert "已确认的需求要点" not in description


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_commit", ["", "b" * 40, "b" * 7])
async def test_code_context_resolves_selected_repository_branches_and_commit(monkeypatch, explicit_commit):
    prefix = "conciliator.services.aeolus_service."
    repository = AsyncMock(return_value=SimpleNamespace(id="other-repo"))
    branches = AsyncMock(side_effect=[SimpleNamespace(commit_sha="a" * 40), SimpleNamespace(commit_sha="c" * 40)])
    commit = AsyncMock(return_value=SimpleNamespace(id="b" * 40))
    monkeypatch.setattr(prefix + "get_repository", repository)
    monkeypatch.setattr(prefix + "get_branch", branches)
    monkeypatch.setattr(prefix + "get_commit", commit)
    context = CodeContext("dp/other", "develop", explicit_commit, "release/v1")
    client = object()
    resolved = await NextCodeAeolusHost(client).resolve_context(context)
    repository.assert_awaited_once_with(client, path="dp/other")
    assert [call.args[2] for call in branches.await_args_list] == ["develop", "release/v1"]
    assert resolved.base_commit == ("b" * 40 if explicit_commit else "a" * 40)
    assert resolved.repository == "dp/other" and resolved.target_branch == "release/v1"
    if explicit_commit:
        commit.assert_awaited_once_with(client, "other-repo", explicit_commit)
    else:
        commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["base", "target"])
async def test_code_context_missing_branch_does_not_fall_back(monkeypatch, missing):
    prefix = "conciliator.services.aeolus_service."
    monkeypatch.setattr(prefix + "get_repository", AsyncMock(return_value=SimpleNamespace(id="repo")))
    monkeypatch.setattr(prefix + "get_branch", AsyncMock(side_effect=(
        [None] if missing == "base" else [SimpleNamespace(commit_sha="a" * 40), None]
    )))
    with pytest.raises(ValueError, match="不可读取"):
        await NextCodeAeolusHost(object()).resolve_context(CodeContext("dp/other", "develop", "", "release"))


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_branch_creation_uses_frozen_commit_or_reuses_existing_branch(monkeypatch, existing):
    prefix = "conciliator.services.aeolus_service."
    monkeypatch.setattr(prefix + "get_repository", AsyncMock(return_value=SimpleNamespace(id="other-repo")))
    branch = SimpleNamespace(name="aeolus/ai-node/demo", commit_sha="b" * 40)
    lookup = AsyncMock(return_value=branch if existing else None)
    creation = AsyncMock(return_value=branch)
    monkeypatch.setattr(prefix + "get_branch", lookup)
    monkeypatch.setattr(prefix + "create_branch", creation)
    context = CodeContext("dp/other", "develop", "a" * 40, "release")
    client = object()
    result = await NextCodeAeolusHost(client).prepare_branch("demo", context)
    assert result == PreparedBranch("other-repo", branch.name)
    lookup.assert_awaited_once_with(client, "other-repo", branch.name)
    if existing:
        creation.assert_not_awaited()
    else:
        creation.assert_awaited_once_with(client, "other-repo", "a" * 40, branch.name)


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["dp/other", "dp/VizQueryService"])
async def test_custom_context_is_persisted_confirmed_and_reused_for_feedback_and_retry(repository):
    context = CodeContext(repository, "develop", "c" * 40, "release/v1")
    service, registry, runner, host, _ = _service(EvaluationResult(
        True, "通过", (), (), "# 需求共识：修改测试", context,
    ))
    prepare = host.prepare_branch

    async def selected_repository(task_id, code_context):
        return replace(await prepare(task_id, code_context), repo_id="selected-repo")

    host.prepare_branch = selected_repository
    runner.code = AsyncMock(wraps=runner.code)
    await service.accept_event(_event())
    await service.wait_for_idle()
    pending = await registry.get("idem-1")
    assert pending.evaluation["code_context"] == asdict(context)
    assert pending.evaluation_rounds[-1]["result"]["code_context"] == asdict(context)
    assert repository in pending.evaluation["requirement_contract"]
    # After evaluation, even a moving base branch must not cause another resolution.
    host.resolve_context = AsyncMock(side_effect=AssertionError("must reuse frozen context"))
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    await service.submit_feedback("idem-1", "补充测试", "feedback")
    await service.wait_for_idle()
    await registry.transition("idem-1", {AeolusState.REVIEW}, AeolusState.FAILED)
    await service.retry_coding("idem-1", "retry")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert task.evaluation["code_context"] == asdict(context)
    assert task.rounds[0].base_commit_sha == "c" * 40
    assert task.rounds[1].base_commit_sha == "a" * 40
    assert host.prepared_contexts == [context]
    assert task.repo_id == "selected-repo"
    assert host.review_targets == [("selected-repo", "release/v1")]
    assert runner.code.call_args.kwargs["repo_id"] == "selected-repo"
    assert all(call.kwargs["repository"] == repository for call in runner.code.call_args_list)
    assert "release/v1" in runner.code_inputs[0]
    host.resolve_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_commit_mismatch_does_not_use_branch_head(monkeypatch):
    prefix = "conciliator.services.aeolus_service."
    monkeypatch.setattr(prefix + "get_repository", AsyncMock(return_value=SimpleNamespace(id="repo")))
    monkeypatch.setattr(prefix + "get_branch", AsyncMock(return_value=SimpleNamespace(commit_sha="a" * 40)))
    monkeypatch.setattr(prefix + "get_commit", AsyncMock(return_value=SimpleNamespace(id="c" * 40)))
    with pytest.raises(ValueError, match="不一致"):
        await NextCodeAeolusHost(object()).resolve_context(CodeContext(base_commit="b" * 40))


@pytest.mark.asyncio
async def test_unreadable_code_context_blocks_input_review_and_coding():
    service, registry, _, host, _ = _service(EvaluationResult(True, "通过", (), ()))
    host.resolve_context = AsyncMock(side_effect=ValueError("仓库或分支不可读取"))
    await service.accept_event(_event())
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.VALIDATION_FAILED
    assert "仓库或分支不可读取" in task.last_error
    assert not task.rounds and not host.prepared_contexts and host.reviews == 0


@pytest.mark.asyncio
async def test_old_task_without_code_context_keeps_default_repository_and_target():
    service, registry, _, host, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    legacy = dict(task.evaluation)
    legacy.pop("code_context")
    legacy["requirement_contract"] = ""
    await registry.update("idem-1", evaluation=legacy)
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert task.evaluation["code_context"] == asdict(CodeContext(base_commit="a" * 40))
    assert host.prepared_contexts == [CodeContext(base_commit="a" * 40)]
    assert host.review_targets == [("repo-1", "master")]


def test_review_contract_excerpt_keeps_scope_and_nested_acceptance_only():
    contract = """# 需求共识：补充测试说明
## 目标
让研发可以本地运行测试。
## 需求
### REQ-1：补充 README
冗长需求描述不应放入 MR。
**来源**：> 原文引用不应放入 MR。
**验收标准**：
- AC-1：README 包含本地测试命令。
## 本次范围
- README.md
## 非目标（明确不做）
- 不修改业务代码。
## 代码核验依据
- 完整核验记录不应放入 MR。
"""
    excerpt = _review_contract_excerpt(contract)
    assert "研发可以本地运行测试" in excerpt
    assert "REQ-1" in excerpt and "AC-1" in excerpt
    assert "README.md" in excerpt and "不修改业务代码" in excerpt
    assert "不应放入 MR" not in excerpt
    long_excerpt = _review_contract_excerpt("## 目标\n" + "长" * 4000)
    assert len(long_excerpt) < 3100 and "摘录已截断" in long_excerpt


@pytest.mark.parametrize("contract,document,expected", [
    ("# 需求共识：完善 README 本地测试说明\n## 目标", "原文标题", "完善 README 本地测试说明"),
    ("# 需求共识", "\n# 原文标题\n正文", "原文标题"),
    ("", "  原文   标题\n正文", "原文 标题"),
    ("", "\n  ", "实现需求变更"),
    ("# 需求共识：" + "长" * 100, "", "长" * 79 + "…"),
])
def test_review_title_is_readable_and_bounded(contract, document, expected):
    task = AeolusTask(**vars(_event()), action_token="test-token",
                      requirement_text=document, evaluation={"requirement_contract": contract})
    assert _review_title(task) == f"[Aeolus AI] {expected}"


@pytest.mark.asyncio
@pytest.mark.parametrize("feedback_round", [False, True])
@pytest.mark.parametrize("result_saved", [False, True])
async def test_restart_recovers_existing_run_once_without_resending(feedback_round, result_saved, tmp_path):
    # Separate connections are required to exercise competing transactions.
    engine = create_engine(f"sqlite:///{tmp_path / 'tasks.db'}")
    service, registry, runner, host, meegle = _service(
        EvaluationResult(True, "通过", (), ()), engine=engine,
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    started = asyncio.Event()

    async def blocked(**kwargs):
        await kwargs["on_started"]("session-1", "run-2" if feedback_round else "run-1", "event-start")
        started.set()
        await asyncio.Event().wait()

    if feedback_round:
        await service.confirm_evaluation("idem-1", "confirm")
        await service.wait_for_idle()
        runner.recode = blocked
        await service.submit_feedback("idem-1", "请补测试", "feedback")
    else:
        runner.code = blocked
        await service.confirm_evaluation("idem-1", "confirm")
    await asyncio.wait_for(started.wait(), timeout=2)
    await service.aclose()
    if result_saved:
        # Crash after accepting output but before transitioning the task to Review.
        await registry.update("idem-1", coco_last_event_id="event-end")
        await registry.update_last_attempt("idem-1", status="succeeded")
        await registry.update_last_round("idem-1", status="review", commit_sha="b" * 40)
    runner.code = AsyncMock(side_effect=AssertionError("must not create a session"))
    runner.recode = AsyncMock(side_effect=AssertionError("must not resend feedback"))
    run_id = "run-2" if feedback_round else "run-1"
    runner.resume_coding = AsyncMock(return_value=CocoCodingRun(
        "session-1", run_id, "event-end", CodingResult("success", "恢复完成", "b" * 40, (), ()),
    ))
    notified = []

    async def notify(task):
        notified.append(task.state)

    def instance():
        return AeolusService(registry=AeolusTaskRegistry(registry._sf),
            document_reader=service._read_document, coco=runner, code_host=host,
            meegle=meegle, minimum_processing_seconds=0, notify=notify)

    first, second = instance(), instance()
    await asyncio.gather(first.recover_pending(), second.recover_pending())
    await asyncio.gather(first.wait_for_idle(), second.wait_for_idle())
    task = await registry.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert len(task.rounds) == (2 if feedback_round else 1)
    assert len(task.rounds[-1].attempts) == 1
    assert task.rounds[-1].attempts[0].status == "succeeded"
    assert host.reviews == 1
    assert notified == [AeolusState.REVIEW]
    runner.resume_coding.assert_awaited_once_with(
        session_id="session-1", run_id=run_id, event_id="event-start",
        on_started=ANY,
    )
    runner.code.assert_not_awaited()
    runner.recode.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("feedback_round,repair_attempted", [(False, False), (True, False), (True, True)])
async def test_evaluation_restart_collects_run_once_with_competing_instances(
    feedback_round, repair_attempted, tmp_path,
):
    service, registry, runner, host, meegle = _service(
        EvaluationResult(True, "通过", (), ()),
        engine=create_engine(f"sqlite:///{tmp_path / 'evaluation.db'}"),
    )
    started = asyncio.Event()

    async def blocked(text, *, history, session_id, on_started, session_context=None):
        await on_started(session_id or "evaluation-session", "pending-run", "pending-event")
        if repair_attempted:
            await on_started("evaluation-session", "", "")
            await on_started("evaluation-session", "repair-run", "repair-event")
        started.set()
        await asyncio.Event().wait()

    if feedback_round:
        await service.accept_event(_event())
        await service.wait_for_idle()
        runner.evaluate = blocked
        await service.evaluation_feedback("idem-1", "补充验收标准", "feedback")
    else:
        runner.evaluate = blocked
        await service.accept_event(_event())
    await asyncio.wait_for(started.wait(), timeout=2)
    await service.aclose()
    runner.evaluate = AsyncMock(side_effect=AssertionError("must not resend"))
    runner.resume_evaluation = AsyncMock(return_value=EvaluationResult(True, "恢复完成", (), ()))
    reader = AsyncMock(side_effect=AssertionError("must not reread updated document during recovery"))
    notified = []

    async def notify(task):
        notified.append(task.state)

    def instance():
        return AeolusService(
            registry=AeolusTaskRegistry(registry._sf), document_reader=reader,
            coco=runner, code_host=host, meegle=meegle, minimum_processing_seconds=0, notify=notify,
        )

    first, second = instance(), instance()
    await asyncio.gather(first.recover_pending(), second.recover_pending())
    await asyncio.gather(first.wait_for_idle(), second.wait_for_idle())
    task = await registry.get("idem-1")
    assert task.state == AeolusState.EVALUATION_REVIEW
    assert len(task.evaluation_rounds) == (2 if feedback_round else 1)
    run_id = "repair-run" if repair_attempted else "pending-run"
    event_id = "repair-event" if repair_attempted else "pending-event"
    assert task.evaluation_rounds[-1]["run_id"] == run_id
    assert task.evaluation_rounds[-1]["result"]["summary"] == "恢复完成"
    assert notified == [AeolusState.EVALUATION_REVIEW]
    runner.resume_evaluation.assert_awaited_once()
    args = runner.resume_evaluation.call_args.kwargs
    assert (args["session_id"], args["run_id"], args["event_id"]) == (
        "evaluation-session", run_id, event_id,
    )
    assert args["repair_attempted"] == repair_attempted
    runner.evaluate.assert_not_awaited()
    reader.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_read_recovery_respects_existing_lease():
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await registry.create_from_event(_event())
    await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.READING_DOCUMENT)
    assert await registry.acquire_lease("idem-1", "another-instance")
    await service.recover_pending()
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.READING_DOCUMENT
    await registry.release_lease("idem-1", "another-instance")
    await service.recover_pending()
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.EVALUATION_REVIEW
    assert task.evaluation_rounds[-1]["session_id"] == "evaluation-session"


@pytest.mark.asyncio
@pytest.mark.parametrize("session_saved", [False, True])
async def test_evaluation_without_run_cursor_fails_instead_of_resending(session_saved):
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await registry.create_from_event(_event())
    await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.EVALUATING)
    if session_saved:
        await registry.update("idem-1", evaluation_rounds=[{
            "round_no": 1, "status": "evaluating", "session_id": "session",
            "run_id": "", "event_id": "", "repair_attempted": True,
        }])
    runner.evaluate = AsyncMock()
    await service.recover_pending()
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.VALIDATION_FAILED
    assert "缺少恢复游标" in task.last_error
    runner.evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,feedback_round", [
    ("success", False), ("success", True), ("failed", True),
    ("invalid-result", True),
    ("timeout", True), ("transport", True), ("saved", False), ("missing", False),
])
async def test_coding_retry_reuses_round_and_survives_competing_recovery(mode, feedback_round, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'retry.db'}")
    service, registry, runner, host, meegle = _service(EvaluationResult(True, "通过", (), ()), engine=engine)
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    if feedback_round:
        await registry.start_feedback("idem-1", "请补充异常处理测试", "feedback")
        await registry.append_attempt("idem-1", session_id="session-1", run_id="run-2", event_id="event-11")
    else:
        await registry.update("idem-1", mr_id="", mr_iid=0, mr_url="")
    await registry.update_last_attempt("idem-1", status="failed", error="original failure")
    await registry.update_last_round("idem-1", status="error", error="original failure",
                                    commit_sha="a" * 40 if mode == "saved" else "")
    if mode == "missing":
        await registry.append_attempt("idem-1", session_id="", run_id="", event_id="")
        await registry.update_last_attempt("idem-1", status="failed", error="missing cursor")
        await registry.update("idem-1", repo_id="", work_branch="", coco_run_id="")
    current = await registry.get("idem-1")
    await registry.transition(
        "idem-1", {current.state}, AeolusState.FAILED, last_error="original failure",
        evaluation={**current.evaluation, "coding_authorization_required": True},
    )
    runner.resume_coding = AsyncMock(return_value=CocoCodingRun(
        "session-1", "run-2" if feedback_round else "run-1", "event-end",
        CodingResult("success", "恢复完成", "b" * 40, (), ()),
    ))
    if mode == "failed":
        runner.resume_coding.side_effect = CocoRunFailed(
            "remote run failed", retryable=True, replacement_safe=True
        )
    elif mode == "invalid-result":
        runner.resume_coding.side_effect = CocoRunFailed(
            "a reported commit_sha must be a 40-character commit SHA",
            replacement_safe=True,
        )
    elif mode == "timeout":
        runner.resume_coding.side_effect = TimeoutError("still running")
    elif mode == "transport":
        runner.resume_coding.side_effect = ValueError("invalid API response")
    runner.code = AsyncMock(wraps=runner.code)
    started, claimed = await registry.start_coding_retry("idem-1", "retry")
    assert claimed and started.rounds[-1].status == "retrying"
    assert "coding_authorization_required" not in started.evaluation
    assert (await registry.get("idem-1")).evaluation == started.evaluation
    assert (await registry.start_coding_retry("idem-1", "retry"))[1] is False
    with pytest.raises(InvalidTaskTransition):
        await registry.start_coding_retry("idem-1", "another-click")

    def instance():
        return AeolusService(registry=AeolusTaskRegistry(registry._sf),
            document_reader=service._read_document, coco=runner, code_host=host,
            meegle=meegle, minimum_processing_seconds=0)

    first, second = instance(), instance()
    await asyncio.gather(first.recover_pending(), second.recover_pending())
    await asyncio.gather(first.wait_for_idle(), second.wait_for_idle())
    task = await registry.get("idem-1")
    assert len(task.rounds) == (2 if feedback_round else 1)
    assert task.rounds[-1].attempts[0].error == "original failure"
    if mode in {"timeout", "transport"}:
        assert task.state == AeolusState.FAILED
        runner.code.assert_not_awaited()
    else:
        assert task.state == AeolusState.REVIEW
        assert task.rounds[-1].attempts[-1].status == "succeeded"
        assert task.rounds[-1].error == ""
        if mode in {"failed", "invalid-result", "missing"}:
            runner.code.assert_awaited_once()
            prompt = runner.code.call_args.kwargs["requirement_url"]
            assert "需求文档链接：" in prompt and "doccnABC" in prompt
            assert "需求正文" not in prompt
            if feedback_round:
                assert "请补充异常处理测试" in prompt and "a" * 40 in prompt
        else:
            runner.code.assert_not_awaited()
    if mode in {"saved", "missing"}:
        runner.resume_coding.assert_not_awaited()
    else:
        runner.resume_coding.assert_awaited_once()
    assert host.reviews == (1 if feedback_round or mode in {"timeout", "transport"} else 2)


@pytest.mark.asyncio
async def test_retry_coding_after_branch_preparation_failure():
    service, registry, runner, host, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    with pytest.raises(InvalidTaskTransition):
        await service.retry_coding("idem-1", "premature")
    prepare = host.prepare_branch
    host.prepare_branch = AsyncMock(side_effect=RuntimeError("branch unavailable"))
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert not task.rounds[-1].attempts[-1].run_id
    assert "branch unavailable" in task.rounds[-1].attempts[-1].error
    host.prepare_branch = prepare
    await service.retry_coding("idem-1", "retry")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert len(task.rounds) == 1 and len(task.rounds[0].attempts) == 2
    assert len(runner.code_inputs) == 1


@pytest.mark.asyncio
async def test_notification_failure_does_not_fail_task():
    notified = []

    async def notify(task):
        notified.append(task.state)
        raise RuntimeError("message delivery failed")

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()), notify=notify)
    await service.accept_event(_event())
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.EVALUATION_REVIEW
    assert notified == [AeolusState.EVALUATION_REVIEW]


@pytest.mark.asyncio
async def test_validation_failure_stops_before_branch_creation():
    service, registry, _, host, _ = _service(
        EvaluationResult(False, "缺信息", ("验收标准",), ())
    )
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.EVALUATION_REVIEW
    assert task.evaluation["missing_items"] == ["验收标准"]
    assert host.reviews == 0


@pytest.mark.asyncio
async def test_missing_wiki_scope_is_reported_as_actionable_validation_failure():
    service, registry, _, host, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )

    async def fail_to_read_document(url: str, user_key: str) -> str:
        raise LarkError(
            "access denied", http_status=400, code=99991672
        )

    service._read_document = fail_to_read_document
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.VALIDATION_FAILED
    assert task.evaluation["missing_items"] == [
        "飞书应用缺少读取文档所需的 API 权限（包括知识库 wiki:node:read 等），"
        "请联系应用管理员开通后重试"
    ]
    assert task.evaluation["document_read_error_code"] == 99991672
    assert host.reviews == 0


@pytest.mark.asyncio
async def test_unshared_document_requires_document_access_not_user_authorization():
    service, registry, _, host, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )

    async def fail_to_read_document(url: str, user_key: str) -> str:
        raise LarkError("node permission denied", http_status=400, code=131006)

    service._read_document = fail_to_read_document
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.VALIDATION_FAILED
    assert task.evaluation["missing_items"] == [
        "应用和当前用户均无法读取该文档，"
        "请为应用添加阅读权限或为当前用户申请文档访问权限后重试"
    ]
    assert task.evaluation["document_read_error_code"] == 131006
    assert task.evaluation["document_authorization_required"] is False
    await registry.update("idem-1", evaluation={
        **task.evaluation,
        "missing_items": ["当前用户尚未完成飞书授权或授权已失效"],
    })
    assert await service.resume_authorization_blocked("user-1") == 0
    assert (await registry.get("idem-1")).state == AeolusState.VALIDATION_FAILED
    assert host.reviews == 0


@pytest.mark.asyncio
async def test_successful_coding_creates_one_review_and_records_round():
    contract = "# 需求共识\n只更新 README，验证 diff"
    service, registry, runner, host, _ = _service(
        EvaluationResult(True, "信息充分", (), ("注意兼容性",), contract)
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    pending = await registry.get("idem-1")
    assert pending.state == AeolusState.EVALUATION_REVIEW
    assert pending.rounds == []
    persisted = await AeolusTaskRegistry(registry._sf).get("idem-1")
    frozen_contract = persisted.evaluation["requirement_contract"]
    assert frozen_contract.startswith(contract)
    assert "Base commit：" + "a" * 40 in frozen_contract
    assert persisted.evaluation_rounds[0]["result"]["requirement_contract"] == frozen_contract
    assert host.reviews == 0
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")

    task = await registry.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert task.mr_url == "https://code.example/mr/7"
    assert task.rounds[0].commit_sha == "a" * 40
    assert host.reviews == 1
    assert contract in runner.code_inputs[0]
    assert host.review_titles == ["[Aeolus AI] 需求正文"]
    assert host.review_contracts == [frozen_contract]
    assert task.rounds[0].base_commit_sha == "a" * 40
    assert host.review_targets == [("repo-1", "master")]


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_enabled", [False, True])
async def test_evaluation_feedback_rereads_document_and_preserves_history(
    proxy_enabled,
):
    class DocumentAccess:
        def instructions(self, task):
            return "DOC_PROXY_INSTRUCTIONS"

    service, registry, runner, host, _ = _service(
        EvaluationResult(False, "缺验收标准", ("验收标准",), ()),
        document_access=DocumentAccess() if proxy_enabled else None,
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    with pytest.raises(InvalidTaskTransition):
        await service.confirm_evaluation("idem-1", "blocked-confirm")

    prompts = []

    async def read_latest(url, user_key):
        return "更新后的需求正文"

    async def evaluate_latest(
        text, *, history, session_id, on_started, document_access_instructions=""
    ):
        assert session_id == "evaluation-session"
        await on_started(session_id, f"evaluation-run-{len(prompts) + 2}", "evaluation-event")
        prompts.append((text, history, document_access_instructions))
        return EvaluationResult(True, "信息充分", (), ("兼容性",))

    service._read_document = read_latest
    runner.evaluate = evaluate_latest
    await service.evaluation_feedback("idem-1", "验收标准已补充", "feedback-1")
    await service.wait_for_idle()
    await service.evaluation_feedback("idem-1", "验收标准已补充", "feedback-1")
    task = await AeolusTaskRegistry(registry._sf).get("idem-1")
    assert len(task.evaluation_rounds) == 2
    assert task.evaluation_rounds[0]["status"] == "revision_requested"
    assert task.evaluation_rounds[0]["result"]["summary"] == "缺验收标准"
    assert task.evaluation_rounds[1]["reviewed_round_no"] == 1
    assert len(prompts) == 1
    assert prompts[0][0] == "更新后的需求正文"
    assert task.requirement_text == "更新后的需求正文"
    assert prompts[0][1][1]["feedback"] == "验收标准已补充"
    assert prompts[0][2] == (
        "DOC_PROXY_INSTRUCTIONS" if proxy_enabled else ""
    )
    assert task.state == AeolusState.EVALUATION_REVIEW
    assert host.reviews == 0

    await service.evaluation_feedback("idem-1", "兼容老接口", "feedback-2")
    await service.wait_for_idle()
    assert [prompt[0] for prompt in prompts] == ["更新后的需求正文"] * 2
    assert [item["feedback"] for item in prompts[-1][1]] == [
        "", "验收标准已补充", "兼容老接口"
    ]
    await service.confirm_evaluation("idem-1", "blocked-confirm")
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).evaluation_rounds[-1]["status"] == "confirmed"
    assert "https://example.feishu.cn/docx/doccnABC" in runner.code_inputs[0]
    assert "验收标准已补充" in runner.code_inputs[0]
    assert "兼容老接口" in runner.code_inputs[0]
    assert "更新后的需求正文" not in runner.code_inputs[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_snapshot", [False, True])
async def test_missing_user_authorization_blocks_evaluation_until_retry(legacy_snapshot):
    service, registry, runner, host, _ = _service(
        EvaluationResult(True, "通过", (), ()),
        document_access=SimpleNamespace(instructions=lambda task: "DOC_PROXY_INSTRUCTIONS"),
    )
    service._read_document = AsyncMock(side_effect=[
        LarkAuthorizationRequired("https://auth.example/authorize?state=private"),
        "# 授权后读取的正文",
    ])
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.VALIDATION_FAILED
    assert "当前用户" in task.evaluation["missing_items"][0]
    assert "飞书授权" in task.evaluation["missing_items"][0]
    # The card shows its authorize block only for 131006 while the operator has
    # no usable token, so a missing token must report that code rather than none.
    assert task.evaluation["document_read_error_code"] == 131006
    assert task.evaluation["document_authorization_required"] is True
    assert "state=private" not in str(task.evaluation)
    assert runner.evaluation_inputs == []
    assert host.prepared_contexts == []

    if legacy_snapshot:
        evaluation = dict(task.evaluation)
        evaluation.pop("document_authorization_required")
        await registry.update("idem-1", evaluation=evaluation)
    assert await service.resume_authorization_blocked("another-user") == 0
    assert (await registry.get("idem-1")).state == AeolusState.VALIDATION_FAILED
    assert await service.resume_authorization_blocked("user-1") == 1
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.EVALUATION_REVIEW
    assert service._read_document.await_count == 2
    assert runner.evaluation_inputs == [("# 授权后读取的正文", "DOC_PROXY_INSTRUCTIONS")]


@pytest.mark.asyncio
async def test_evaluation_error_can_be_retried_and_keeps_error_history():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))

    async def fail_evaluation(text, *, history, session_id, on_started, session_context=None):
        raise RuntimeError("evaluation session failed")

    runner.evaluate = fail_evaluation
    await service.accept_event(_event())
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.VALIDATION_FAILED
    assert task.evaluation_rounds[0]["status"] == "error"
    assert "evaluation session failed" in task.evaluation_rounds[0]["error"]
    service._coco = FakeRunner(EvaluationResult(True, "通过", (), ()))
    await service.retry_validation("idem-1")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert len(task.evaluation_rounds) == 2
    assert task.state == AeolusState.EVALUATION_REVIEW


@pytest.mark.asyncio
async def test_coding_authorization_failure_is_persisted_for_user_interfaces():
    service, _, runner, _, _ = _service(EvaluationResult(True, "信息充分", (), ()))
    successful_code = runner.code

    async def fail_for_authorization(**kwargs) -> CocoCodingRun:
        await kwargs["on_started"]("session-auth", "run-auth", "event-auth")
        return CocoCodingRun(
            "session-auth",
            "run-auth",
            "event-auth-end",
            CodingResult(
                "failed",
                "需要重新授权",
                "",
                (),
                ("请重新授权",),
                "lark_authorization_required",
            ),
        )

    runner.code = fail_for_authorization
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    task = await service.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert task.evaluation["coding_authorization_required"] is True

    assert await service.resume_authorization_blocked("user-1") == 0
    await service.wait_for_idle()
    assert (await service.get("idem-1")).state == AeolusState.FAILED

    runner.resume_coding = AsyncMock(side_effect=CocoRunFailed(
        "authorization failed", retryable=True
    ))
    runner.code = successful_code
    await service.retry_coding("idem-1", "retry-after-authorization")
    await service.wait_for_idle()

    task = await service.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert "coding_authorization_required" not in task.evaluation
    assert len(task.rounds) == 1


@pytest.mark.asyncio
async def test_failed_coding_keeps_coco_run_identifiers():
    service, registry, _, host, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )
    service._coco = FailingRunner(EvaluationResult(True, "信息充分", (), ()))

    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert task.coco_session_id == "session-failed"
    assert task.coco_run_id == "run-failed"
    assert task.coco_last_event_id == "event-start"
    assert [attempt.status for attempt in task.rounds[0].attempts] == [
        "failed",
        "failed",
    ]
    assert host.reviews == 0


@pytest.mark.asyncio
async def test_question_holds_the_session_lease_until_answer_finishes():
    service, registry, runner, _, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    answer_started = asyncio.Event()
    finish_answer = asyncio.Event()
    recode_started = asyncio.Event()
    recode_waiting = asyncio.Event()
    original_recode = runner.recode
    original_acquire = registry.acquire_lease

    async def answer(**kwargs):
        assert kwargs["session_id"] == "session-1"
        answer_started.set()
        await finish_answer.wait()
        return "answer"

    async def recode(**kwargs):
        recode_started.set()
        return await original_recode(**kwargs)

    async def acquire_lease(*args, **kwargs):
        acquired = await original_acquire(*args, **kwargs)
        if not acquired:
            recode_waiting.set()
        return acquired

    runner.answer = answer
    runner.recode = recode
    registry.acquire_lease = AsyncMock(side_effect=acquire_lease)

    question = asyncio.create_task(
        service.answer_question("idem-1", "为什么这样实现？")
    )
    await asyncio.wait_for(answer_started.wait(), timeout=2)
    await service.submit_feedback("idem-1", "请补测试", "request-1")
    await asyncio.wait_for(recode_waiting.wait(), timeout=2)

    assert not recode_started.is_set()
    waiting_task = await registry.get("idem-1")
    assert waiting_task.state == AeolusState.RECODING
    assert waiting_task.rounds[-1].status == "queued"

    finish_answer.set()
    assert await asyncio.wait_for(question, timeout=2) == "answer"
    await asyncio.wait_for(service.wait_for_idle(), timeout=2)

    assert recode_started.is_set()
    assert (await registry.get("idem-1")).state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_queued_feedback_recovers_once_after_restart_while_waiting_for_lease(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'aeolus.db'}",
        connect_args={"check_same_thread": False},
    )
    evaluation = EvaluationResult(True, "信息充分", (), ())
    service, registry, runner, _, _ = _service(evaluation, engine=engine)
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    answer_started = asyncio.Event()
    recode_waiting = asyncio.Event()
    original_acquire = registry.acquire_lease

    async def answer(**_kwargs):
        answer_started.set()
        await asyncio.Event().wait()

    async def acquire_lease(*args, **kwargs):
        acquired = await original_acquire(*args, **kwargs)
        if not acquired:
            recode_waiting.set()
        return acquired

    runner.answer = answer
    question = asyncio.create_task(service.answer_question("idem-1", "为什么？"))
    await asyncio.wait_for(answer_started.wait(), timeout=2)
    registry.acquire_lease = AsyncMock(side_effect=acquire_lease)
    await service.submit_feedback("idem-1", "请补测试", "request-1")
    await asyncio.wait_for(recode_waiting.wait(), timeout=2)

    queued = await registry.get("idem-1")
    assert queued.state == AeolusState.RECODING
    assert queued.rounds[-1].status == "queued"
    assert not queued.rounds[-1].attempts

    await service.aclose()
    question.cancel()
    await asyncio.gather(question, return_exceptions=True)

    second, second_registry, second_runner, _, _ = _service(evaluation, engine=engine)
    third, _, third_runner, _, _ = _service(evaluation, engine=engine)
    await asyncio.gather(second.recover_pending(), third.recover_pending())
    await asyncio.wait_for(
        asyncio.gather(second.wait_for_idle(), third.wait_for_idle()), timeout=2
    )

    recovered = await second_registry.get("idem-1")
    assert recovered.state == AeolusState.REVIEW
    assert recovered.rounds[-1].status == "review"
    assert len(recovered.rounds[-1].attempts) == 1
    assert second_runner.feedbacks + third_runner.feedbacks == ["请补测试"]
    await asyncio.gather(second.aclose(), third.aclose())


@pytest.mark.asyncio
async def test_recovery_does_not_resend_ambiguous_feedback_without_an_attempt(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'aeolus.db'}",
        connect_args={"check_same_thread": False},
    )
    evaluation = EvaluationResult(True, "信息充分", (), ())
    service, registry, _, _, _ = _service(evaluation, engine=engine)
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    await registry.start_feedback("idem-1", "请补测试", "request-1")
    claimed = await registry.claim_queued_feedback("idem-1")
    assert claimed is not None and claimed.rounds[-1].status == "coding"

    restarted, restarted_registry, restarted_runner, _, _ = _service(
        evaluation, engine=engine
    )
    await restarted.recover_pending()
    await asyncio.wait_for(restarted.wait_for_idle(), timeout=2)

    failed = await restarted_registry.get("idem-1")
    assert failed.state == AeolusState.FAILED
    assert failed.rounds[-1].status == "error"
    assert restarted_runner.feedbacks == []
    await asyncio.gather(service.aclose(), restarted.aclose())


@pytest.mark.asyncio
async def test_question_rechecks_review_state_after_acquiring_the_lease():
    service, registry, runner, _, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    original_acquire = registry.acquire_lease

    async def acquire_then_start_feedback(*args, **kwargs):
        acquired = await original_acquire(*args, **kwargs)
        assert acquired
        await registry.start_feedback("idem-1", "请补测试", "request-1")
        return acquired

    runner.answer = AsyncMock(return_value="must not run")
    registry.acquire_lease = AsyncMock(side_effect=acquire_then_start_feedback)

    with pytest.raises(InvalidTaskTransition, match="questions need a free session"):
        await service.answer_question("idem-1", "为什么这样实现？")

    runner.answer.assert_not_awaited()
    assert await original_acquire("idem-1", "after-answer")
    await registry.release_lease("idem-1", "after-answer")


@pytest.mark.asyncio
async def test_question_lease_loss_cancels_answer_without_failing_task(monkeypatch):
    service, registry, runner, _, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    answer_started = asyncio.Event()
    answer_cancelled = asyncio.Event()

    async def answer(**_kwargs):
        answer_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            answer_cancelled.set()
            raise

    async def fail_renewal(*_args, **_kwargs):
        await answer_started.wait()
        return False

    runner.answer = answer
    registry.renew_lease = AsyncMock(side_effect=fail_renewal)
    monkeypatch.setattr(
        "conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 0.01
    )

    with pytest.raises(TaskLeaseLost, match="lease expired"):
        await asyncio.wait_for(
            service.answer_question("idem-1", "为什么这样实现？"), timeout=2
        )

    assert answer_cancelled.is_set()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert not task.last_error
    assert await registry.acquire_lease("idem-1", "after-answer")
    await registry.release_lease("idem-1", "after-answer")


@pytest.mark.asyncio
async def test_feedback_reuses_session_branch_and_review():
    service, registry, runner, host, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    await service.submit_feedback("idem-1", "请补测试", "request-1")
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.REVIEW
    assert [round_.commit_sha for round_ in task.rounds] == ["a" * 40, "b" * 40]
    assert task.rounds[0].status == "revision_requested"
    assert task.rounds[1].trigger_type == "review_feedback"
    assert task.rounds[1].reviewed_round_no == 1
    assert task.rounds[1].base_commit_sha == "a" * 40
    assert runner.feedbacks == ["请补测试"]
    assert host.reviews == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["dp/aeolus", "dp/VizQueryService"])
async def test_feedback_session_failure_retries_within_same_round(repository):
    evaluation = EvaluationResult(True, "信息充分", (), (), code_context=CodeContext(repository=repository))
    service, registry, _, host, _ = _service(
        evaluation
    )
    runner = RecoveringFeedbackRunner(evaluation)
    runner.code = AsyncMock(wraps=runner.code)
    service._coco = runner

    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    await service.submit_feedback("idem-1", "请补测试", "request-1")
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert len(task.rounds) == 2
    assert [attempt.status for attempt in task.rounds[1].attempts] == [
        "failed",
        "succeeded",
    ]
    assert task.rounds[1].commit_sha == "c" * 40
    assert "需求文档链接：" in runner.recovery_prompt
    assert "doccnABC" in runner.recovery_prompt
    assert "需求正文" not in runner.recovery_prompt
    assert "当前分支最新提交：" + "a" * 40 in runner.recovery_prompt
    assert "第 2 轮：请补测试" in runner.recovery_prompt
    assert host.reviews == 1
    assert runner.code.await_count == 2
    assert all(call.kwargs["repository"] == repository for call in runner.code.call_args_list)


@pytest.mark.asyncio
async def test_confirmation_updates_done_then_confirms_and_is_idempotent():
    service, registry, _, _, meegle = _service(
        EvaluationResult(True, "信息充分", (), ())
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    await service.confirm("idem-1", "request-confirm")
    await service.wait_for_idle()
    await service.confirm("idem-1", "request-confirm")

    assert (await registry.get("idem-1")).state == AeolusState.COMPLETED
    assert meegle.calls[0] == ("edit", "done")
    assert meegle.calls[1][0] == "confirm"
    assert len(meegle.calls) == 2


@pytest.mark.asyncio
async def test_workflow_logs_link_evaluation_coding_and_completion_without_payloads():
    service, registry, _, _, meegle = _service(EvaluationResult(True, "private-evaluation", (), ()))
    with capture_logs(processors=[merge_contextvars]) as logs:
        await service.accept_event(_event())
        await service.wait_for_idle()
        await service.confirm_evaluation("idem-1", "evaluation-confirm")
        await service.wait_for_idle()
        await service.confirm("idem-1", "request-confirm")
        await service.wait_for_idle()
        await service.confirm("idem-1", "request-confirm")
    assert (await registry.get("idem-1")).state == AeolusState.COMPLETED
    states = [row for row in logs if row["event"] == "meegle_task_state_changed"]
    assert [row["state"] for row in states] == [
        "reading_document", "evaluating", "evaluation_review", "coding", "review", "completing", "completed",
    ]
    assert all(row["task_id"] == "idem-1" and row["workflow"] == "meegle" for row in states)
    assert states[4]["mr_id"] == "mr-id"
    assert any(row.get("run_id") == "evaluation-run" for row in logs)
    assert any(row.get("run_id") == "run-1" and row.get("round_no") == 1 for row in logs)
    assert len(meegle.calls) == 2
    text = json.dumps(logs, ensure_ascii=False)
    assert "private-evaluation" not in text and "需求正文" not in text
    assert "doccnABC" not in text and "action_token" not in text


@pytest.mark.asyncio
async def test_confirmation_retry_does_not_repeat_done_update():
    service, registry, _, _, _ = _service(
        EvaluationResult(True, "信息充分", (), ())
    )
    failing = FailConfirmOnceMeegle()
    service._meegle = failing
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    await service.confirm("idem-1", "confirm-1")
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.FAILED

    await service.confirm("idem-1", "confirm-2")
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.COMPLETED
    assert [call[0] for call in failing.calls] == ["edit", "confirm", "confirm"]


@pytest.mark.asyncio
async def test_recovery_completes_pending_platform_confirmation_without_recoding():
    service, registry, runner, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    await registry.begin_completion("idem-1", "complete", False)
    await service.recover_pending()
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.COMPLETED
    assert len(runner.code_inputs) == 1
    assert meegle.calls == [("edit", "done"), ("confirm", "idem-1:confirm")]


@pytest.mark.asyncio
async def test_done_update_failure_is_retried_as_completion_not_coding():
    from conciliator.domain.aeolus_task import can_retry_coding, can_retry_completion

    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    platform = AsyncMock()
    platform.edit_status.side_effect = [RuntimeError("transient platform failure"), None]
    service._meegle = platform
    await service.confirm("idem-1", "finish-1")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert can_retry_completion(task) and not can_retry_coding(task)
    await service.confirm("idem-1", "finish-2")
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.COMPLETED
    assert len(runner.code_inputs) == 1
    assert platform.confirm.await_count == 1


@pytest.mark.asyncio
async def test_retry_validation_refreshes_document_link_from_meegle():
    """A replaced requirement document must be picked up on re-evaluation.

    The webhook props freeze the link at trigger time; Meegle's node query
    returns the current props, so the retry path must refresh instead of
    re-reading the stale link stored in the task.
    """
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))

    class FreshMeegle(FakeMeegle):
        async def query(self, ref):
            return {
                "ai_info": {
                    "props": [
                        {
                            "prop_key": "design_doc",
                            "prop_type": ["field"],
                            "prop_value": {
                                "field": [
                                    {
                                        "field_key": "wiki",
                                        "field_name": "需求文档",
                                        "field_type": "link",
                                        "value": "https://example.feishu.cn/wiki/doccnFRESH",
                                    }
                                ]
                            },
                        }
                    ]
                },
            }

    service._meegle = FreshMeegle()
    seen: list[str] = []

    async def read_document(url: str, user_key: str) -> str:
        seen.append(url)
        return "新文档正文"

    service._read_document = read_document
    await service.accept_event(_event())
    await service.wait_for_idle()
    assert seen and seen[0].endswith("doccnFRESH")
    task = await registry.get("idem-1")
    assert task.requirement_url.endswith("doccnFRESH")
    assert task.props[0]["prop_value"]["field"][0]["value"].endswith("doccnFRESH")


@pytest.mark.asyncio
async def test_retry_validation_falls_back_to_stored_props_when_meegle_query_fails():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))

    class BrokenMeegle(FakeMeegle):
        async def query(self, ref):
            raise RuntimeError("meegle query down")

    service._meegle = BrokenMeegle()
    await service.accept_event(_event())
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.EVALUATION_REVIEW
    assert task.requirement_url.endswith("doccnABC")


@pytest.mark.asyncio
async def test_document_read_failure_without_lark_code_reports_no_code():
    service, registry, _, host, _ = _service(EvaluationResult(True, "通过", (), ()))

    async def fail_to_read_document(url: str, user_key: str) -> str:
        raise ValueError("document url is malformed")

    service._read_document = fail_to_read_document
    await service.accept_event(_event())
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert task.state == AeolusState.VALIDATION_FAILED
    # Only authorization problems carry a code the card can act on.
    assert task.evaluation["document_read_error_code"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("commit", ["", "a" * 40, "c" * 40])
@pytest.mark.parametrize("feedback_status", ["blocked", "success"])
async def test_blocked_coding_preserves_results_and_accepts_feedback(commit, feedback_status):
    service, registry, runner, host, meegle = _service(EvaluationResult(True, "通过", (), ()))
    original_code = runner.code

    async def blocked_code(**kwargs):
        run = await original_code(**kwargs)
        return CocoCodingRun(run.session_id, run.run_id, run.last_event_id, CodingResult(
            "blocked", "部分实现，集成环境不可用", commit, ("pytest unit: 56 passed",),
            ("请提供集成环境", "完整验收待执行"), "integration_unavailable",
        ))

    runner.code = AsyncMock(side_effect=blocked_code)
    host.verify_commit = AsyncMock()
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.CODING_BLOCKED
    assert task.last_error == ""
    assert task.rounds[-1].status == "blocked"
    assert task.rounds[-1].summary == "部分实现，集成环境不可用"
    assert task.rounds[-1].tests == ["pytest unit: 56 passed"]
    assert task.rounds[-1].unresolved_items == ["请提供集成环境", "完整验收待执行"]
    assert task.rounds[-1].failure_code == "integration_unavailable"
    assert task.rounds[-1].attempts[-1].status == "blocked"
    assert task.rounds[-1].finished_at is not None
    assert task.coco_session_id == "session-1"
    assert task.coco_last_event_id == "event-9"
    assert task.timing["current_phase"] == "human_wait"
    assert task.task_id not in await registry.recoverable_task_ids()
    assert host.reviews == int(commit == "c" * 40)
    if commit:
        host.verify_commit.assert_awaited_once_with("repo-1", task.work_branch, commit)
    else:
        host.verify_commit.assert_not_awaited()
    with pytest.raises(InvalidTaskTransition):
        await service.confirm("idem-1", "must-not-complete")
    meegle.calls.clear()
    if feedback_status == "blocked":
        original_recode = runner.recode

        async def blocked_feedback(**kwargs):
            run = await original_recode(**kwargs)
            return CocoCodingRun(run.session_id, run.run_id, run.last_event_id, CodingResult(
                "blocked", "环境仍未就绪", "", (), ("需提供连接方式",), "environment_unavailable",
            ))

        runner.recode = blocked_feedback
    await service.submit_feedback("idem-1", "已提供环境，请继续", "feedback")
    await service.wait_for_idle()
    await service.submit_feedback("idem-1", "已提供环境，请继续", "feedback")
    await service.wait_for_idle()
    updated = await registry.get("idem-1")
    assert len(updated.rounds) == 2
    assert updated.rounds[0].status == "blocked"
    assert updated.rounds[0].unresolved_items == task.rounds[0].unresolved_items
    assert updated.rounds[1].base_commit_sha == (commit or "a" * 40)
    assert updated.rounds[1].reviewed_round_no == 1
    assert runner.feedbacks == ["已提供环境，请继续"]
    assert updated.state == (AeolusState.CODING_BLOCKED if feedback_status == "blocked" else AeolusState.REVIEW)
    if feedback_status == "success":
        assert updated.mr_url and updated.rounds[-1].code_url == updated.mr_url


@pytest.mark.asyncio
async def test_blocked_commit_remains_reviewable_if_mr_creation_fails():
    service, registry, runner, host, _ = _service(EvaluationResult(True, "通过", (), ()))
    original_code = runner.code

    async def blocked(**kwargs):
        run = await original_code(**kwargs)
        return CocoCodingRun(run.session_id, run.run_id, run.last_event_id, CodingResult(
            "blocked", "部分交付", "c" * 40, (), ("需人工提供环境",), "env_missing",
        ))

    runner.code = blocked
    host.create_review = AsyncMock(side_effect=RuntimeError("MR unavailable"))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.CODING_BLOCKED
    assert task.mr_url == ""
    assert task.rounds[-1].code_url == "https://code.byted.org/dp/aeolus/commit/" + "c" * 40


@pytest.mark.asyncio
async def test_failed_result_with_unknown_code_keeps_real_summary():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    original_code = runner.code

    async def failed(**kwargs):
        run = await original_code(**kwargs)
        return CocoCodingRun(run.session_id, run.run_id, run.last_event_id, CodingResult(
            "failed", "沙箱初始化失败", "", (), ("运行环境需要修复",), "sandbox_not_ready",
        ))

    runner.code = failed
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert "沙箱初始化失败" in task.last_error
    assert task.rounds[-1].failure_code == "sandbox_not_ready"
    assert task.rounds[-1].unresolved_items == ["运行环境需要修复"]


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_recovery_collects_blocked_run_without_starting_another_session(retry):
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    ready = asyncio.Event()

    async def pending(**kwargs):
        await kwargs["on_started"]("session-1", "run-1", "event-start")
        ready.set()
        await asyncio.Event().wait()

    runner.code = pending
    await service.confirm_evaluation("idem-1", "confirm")
    await asyncio.wait_for(ready.wait(), timeout=2)
    await service.aclose()
    if retry:
        await registry.transition("idem-1", {AeolusState.CODING}, AeolusState.FAILED)
        await registry.start_coding_retry("idem-1", "retry")
    runner.code = AsyncMock(side_effect=AssertionError("must not start another session"))
    runner.resume_coding = AsyncMock(return_value=CocoCodingRun(
        "session-1", "run-1", "event-end", CodingResult(
            "blocked", "需要集成环境", "", (), ("请提供环境地址",), "environment_unavailable",
        ),
    ))
    await service.recover_pending()
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.CODING_BLOCKED
    assert task.rounds[-1].status == "blocked"
    assert task.rounds[-1].unresolved_items == ["请提供环境地址"]
    runner.code.assert_not_awaited()
    runner.resume_coding.assert_awaited_once()
    await service.recover_pending()
    await service.wait_for_idle()
    runner.resume_coding.assert_awaited_once()


@pytest.mark.asyncio
async def test_blocked_result_cannot_expose_an_unverified_commit():
    service, registry, runner, host, _ = _service(EvaluationResult(True, "通过", (), ()))
    original_code = runner.code

    async def blocked(**kwargs):
        run = await original_code(**kwargs)
        return replace(run, result=CodingResult("blocked", "部分实现", "c" * 40, (), ("缺环境",)))

    runner.code = blocked
    host.verify_commit = AsyncMock(side_effect=ValueError("commit is not on the coding branch"))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    assert task.state == AeolusState.FAILED
    assert task.rounds[-1].commit_sha == ""
    assert task.rounds[-1].code_url == ""
    assert host.reviews == 0


@pytest.mark.asyncio
async def test_evaluation_platform_failure_backs_off_then_replaces_session():
    evaluation = EvaluationResult(True, "通过", (), ())
    service, registry, runner, _, _ = _service(evaluation)
    sessions: list[str] = []

    async def evaluate(text, **kwargs):
        sessions.append(kwargs.get("session_id", ""))
        if len(sessions) == 1:
            await kwargs["on_started"]("session-1", "run-1", "event-1")
            raise CocoRunFailed(
                "平台侧异常：request queue size is exceed",
                retryable=True,
                platform=True,
            )
        await kwargs["on_started"]("session-2", "run-2", "event-2")
        return evaluation

    runner.evaluate = AsyncMock(side_effect=evaluate)
    delay = AsyncMock()
    service._platform_retry_delay = delay

    await service.accept_event(_event())
    await service.wait_for_idle()

    # A saturated platform queue needs time to drain before the replacement.
    delay.assert_awaited_once()
    assert sessions == ["", ""]
    task = await registry.get("idem-1")
    current = task.evaluation_rounds[-1]
    # The replacement is platform recovery: it must neither consume the format
    # repair budget nor leave the card claiming a retry that never happened.
    assert not current.get("repair_attempted")
    assert current["platform_retry_attempted"] is True


@pytest.mark.asyncio
async def test_restart_during_the_platform_backoff_starts_the_replacement_run():
    """A restart inside the platform backoff must not lose the replacement run.

    The flow persists the attempt as failed before it waits out the backoff, so
    the cursor left in the database points at a run the platform already
    terminated.  Resuming it only replays that failure, which the worker turns
    into a failed task.
    """
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()

    await registry.review_evaluation("idem-1", "confirm")
    await registry.append_attempt(
        "idem-1", session_id="session-1", run_id="run-1", event_id="event-1"
    )
    await registry.update_last_attempt(
        "idem-1", status="failed", error="CocoRunFailed: terminated"
    )
    runner.resume_coding = AsyncMock()

    await service.recover_pending()
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    runner.resume_coding.assert_not_awaited()
    assert len(runner.code_inputs) == 1
    assert task.state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_recovery_replaces_a_run_that_ended_before_its_attempt_was_marked():
    """A cursor outlives its run when the process stops before the failure lands."""
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()

    await registry.review_evaluation("idem-1", "confirm")
    await registry.append_attempt(
        "idem-1", session_id="session-1", run_id="run-1", event_id="event-1"
    )
    runner.resume_coding = AsyncMock(
        side_effect=CocoRunFailed(
            "Coco run ended without a result", retryable=True, replacement_safe=True
        )
    )

    await service.recover_pending()
    await service.wait_for_idle()

    task = await registry.get("idem-1")
    assert runner.resume_coding.await_count == 1
    assert len(runner.code_inputs) == 1
    assert task.state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_restarting_a_completed_node_reconfirms_it():
    """A restarted node is running again and must be told the work is done.

    Reusing the completed task is right, but a silent reuse leaves the node
    running with nothing left to execute.
    """
    service, registry, runner, _, meegle = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    await service.confirm("idem-1", "finish")
    await service.wait_for_idle()
    assert (await registry.get("idem-1")).state == AeolusState.COMPLETED

    meegle.calls.clear()
    runner.evaluation_inputs.clear()
    await service.accept_event(_event(task_id="idem-2"))
    await service.wait_for_idle()

    assert (await registry.get("idem-1")).state == AeolusState.COMPLETED
    assert meegle.calls == [("edit", "done"), ("confirm", "idem-2:confirm")]
    assert runner.evaluation_inputs == []


@pytest.mark.asyncio
async def test_restarting_a_validation_failed_node_reevaluates():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    current = await registry.get("idem-1")
    await registry.transition(
        "idem-1", {current.state}, AeolusState.VALIDATION_FAILED, last_error="boom"
    )

    runner.evaluation_inputs.clear()
    await service.accept_event(_event(task_id="idem-2"))
    await service.wait_for_idle()

    assert (await registry.get("idem-1")).state == AeolusState.EVALUATION_REVIEW
    assert len(runner.evaluation_inputs) == 1


@pytest.mark.asyncio
async def test_transient_coding_write_error_is_retried_once():
    """A write call has no transport retry of its own; the coding step retries it."""
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    original = runner.code
    calls = 0

    async def code(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientError("connection reset by peer")
        return await original(**kwargs)

    runner.code = code
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()

    assert (await registry.get("idem-1")).state == AeolusState.REVIEW
    assert calls == 2
    assert len(runner.code_inputs) == 1


@pytest.mark.asyncio
async def test_recovery_replaces_a_terminal_evaluation_run():
    """The evaluation cursor can outlive its run, exactly like the coding one."""
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    current = await registry.get("idem-1")
    await registry.transition("idem-1", {current.state}, AeolusState.EVALUATING)
    await registry.update("idem-1", evaluation_rounds=[{
        "round_no": 1, "status": "evaluating", "session_id": "session-1",
        "run_id": "run-1", "event_id": "event-1",
    }])
    runner.resume_evaluation = AsyncMock(side_effect=CocoRunFailed(
        "Coco run ended without a result", retryable=True, replacement_safe=True,
    ))
    runner.evaluation_inputs.clear()

    await service.recover_pending()
    await service.wait_for_idle()

    assert runner.resume_evaluation.await_count == 1
    assert len(runner.evaluation_inputs) == 1
    assert (await registry.get("idem-1")).state == AeolusState.EVALUATION_REVIEW


@pytest.mark.asyncio
async def test_transient_feedback_write_error_is_retried_once():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    original = runner.recode
    calls = 0

    async def recode(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientError("connection reset by peer")
        return await original(**kwargs)

    runner.recode = recode
    await service.submit_feedback("idem-1", "请补充异常处理测试", "feedback-1")
    await service.wait_for_idle()

    # The failed write is not repeated on the same session; the retry starts a
    # replacement run with the recovery prompt.
    assert calls == 1
    assert len(runner.code_inputs) == 2
    assert (await registry.get("idem-1")).state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_restarting_a_failed_node_retries_instead_of_parking():
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "confirm")
    await service.wait_for_idle()
    current = await registry.get("idem-1")
    await registry.transition("idem-1", {current.state}, AeolusState.FAILED, last_error="boom")

    await service.accept_event(_event(task_id="idem-2"))
    await service.wait_for_idle()

    assert (await registry.get("idem-1")).state == AeolusState.REVIEW


@pytest.mark.asyncio
async def test_acknowledged_feedback_is_recovered_after_a_restart(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'aeolus.db'}",
        connect_args={"check_same_thread": False},
    )
    evaluation = EvaluationResult(True, "信息充分", (), ())
    service, _, _, _, _ = _service(evaluation, engine=engine)
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    assert await service.save_pending_feedback(
        "idem-1",
        [("c-1", "请补测试")],
        available_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    restarted, restarted_registry, restarted_runner, _, _ = _service(
        evaluation, engine=engine
    )
    await restarted.recover_pending()
    await asyncio.wait_for(restarted.wait_for_idle(), timeout=2)

    assert restarted_runner.feedbacks == ["请补测试"]
    assert (await restarted_registry.get("idem-1")).state == AeolusState.REVIEW
    assert await restarted_registry.pending_feedback("idem-1") == []


@pytest.mark.asyncio
async def test_completion_is_refused_while_a_comment_waits_for_its_window(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'race.db'}",
        connect_args={"check_same_thread": False},
    )
    service, registry, runner, _, _ = _service(
        EvaluationResult(True, "信息充分", (), ()), engine=engine
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()

    assert await service.save_pending_feedback(
        "idem-1", [("c-1", "请补测试")], available_at=datetime.now(UTC)
    )

    # The comment was acknowledged to its author, so completion must not
    # overtake it while it is still inside the aggregation window.
    with pytest.raises(InvalidTaskTransition):
        await service.confirm("idem-1", "complete-1")
    assert (await registry.get("idem-1")).state == AeolusState.REVIEW
    assert await registry.pending_feedback("idem-1") != []

    # Once the deferred feedback starts its recoding round, completion has to wait
    # for that round; a late comment must not be acknowledged after it starts.
    await service.recover_pending()
    await asyncio.wait_for(service.wait_for_idle(), timeout=2)
    assert runner.feedbacks == ["请补测试"]

    await service.confirm("idem-1", "complete-2")
    with pytest.raises(InvalidTaskTransition):
        await service.save_pending_feedback(
            "idem-1", [("c-2", "迟到的评论")], available_at=datetime.now(UTC)
        )


@pytest.mark.asyncio
async def test_missing_pending_table_fails_startup_and_writes(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'aeolus.db'}",
        connect_args={"check_same_thread": False},
    )
    service, registry, _, _, _ = _service(
        EvaluationResult(True, "信息充分", (), ()), engine=engine
    )
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "evaluation-confirm")
    await service.wait_for_idle()
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE meegle_ai_node_pending_feedback")

    # The table is mandatory: startup fails instead of silently accepting
    # feedback into process memory that completion cannot observe.
    with pytest.raises(RuntimeError, match="pending_feedback is missing"):
        await service.start()

    with pytest.raises(OperationalError, match="no such table"):
        await registry.save_pending_feedback(
            "idem-1", [("c-1", "请补测试")], available_at=datetime.now(UTC)
        )


def _comment_event(comment_id: str, body: str) -> CommentEvent:
    return CommentEvent(
        comment_id=comment_id,
        repo_id="repo-1",
        mr_iid=7,
        body=body,
        author_email="dev@bytedance.com",
        discussion_id=f"thread-{comment_id}",
        position={"Path": "aeolus/lod.py", "StartLine": 12},
    )


class _HandlerRegistry:
    """Share the real registry while pinning the MR lookup to the fixture task."""

    def __init__(self, registry: AeolusTaskRegistry, task: AeolusTask) -> None:
        self._registry = registry
        self._task = task

    async def find_active_for_mr(self, repo_id: str, mr_iid: int) -> AeolusTask:
        return self._task

    def __getattr__(self, name: str) -> object:
        return getattr(self._registry, name)


class _TaskSpawner:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[None]] = []

    def spawn(self, operation) -> None:
        self.tasks.append(asyncio.ensure_future(operation))


def _question_handler(registry, task, answerer, reply):
    spawner = _TaskSpawner()
    comments = SimpleNamespace(enqueue=AsyncMock(), spawn=spawner.spawn)
    comments.spawner = spawner
    return AeolusCommentHandler(
        registry=_HandlerRegistry(registry, task),
        comments=comments,
        meegle=SimpleNamespace(
            get_user_email=AsyncMock(return_value="dev@bytedance.com")
        ),
        reply=reply,
        answerer=answerer,
    )


async def _review_task(service, registry):
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "req-1")
    await service.wait_for_idle()
    return await registry.get("idem-1")


@pytest.mark.asyncio
async def test_question_redelivery_retries_after_worker_shutdown_cancels_answer():
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    started = asyncio.Event()

    async def answerer(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    handler = _question_handler(registry, task, answerer, AsyncMock())
    event = _comment_event("q-1", "/ask 这段为什么幂等？")
    await handler.handle(event)
    await started.wait()
    answer_task = next(
        task
        for task in handler._comments.spawner.tasks
        if "_answer_later" in task.get_coro().__qualname__
    )
    answer_task.cancel()
    with suppress(asyncio.CancelledError):
        await answer_task

    # The cancelled answer released its claim, so the redelivery can try again.
    assert await registry.claim_comment(
        "idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL
    )


@pytest.mark.asyncio
async def test_question_redelivery_republishes_the_saved_answer():
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    answerer = AsyncMock(return_value="这段逻辑是幂等的。")
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)
        if len(replies) == 1:
            raise RuntimeError("reply api unavailable")

    handler = _question_handler(registry, task, answerer, reply)
    event = _comment_event("q-1", "/ask 这段为什么幂等？")
    await handler.handle(event)
    await asyncio.wait_for(asyncio.gather(*handler._comments.spawner.tasks), timeout=5)
    saved = await registry.question_record(task.task_id, event.comment_id)
    assert saved["status"] == "answered"
    assert saved["answer_text"] == "这段逻辑是幂等的。"
    assert len(replies) == 1

    await handler.handle(event)
    await asyncio.wait_for(asyncio.gather(*handler._comments.spawner.tasks), timeout=5)

    # The saved answer is republished without spending a second agent run.
    assert answerer.await_count == 1
    assert len(replies) == 2
    assert replies[0] == replies[1]
    assert (await registry.question_record(task.task_id, event.comment_id))["status"] == "published"


@pytest.mark.asyncio
async def test_answered_question_stays_deduplicated_after_the_claim_expires():
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    answerer = AsyncMock(return_value="这段逻辑是幂等的。")
    handler = _question_handler(registry, task, answerer, AsyncMock())
    event = _comment_event("q-1", "/ask 这段为什么幂等？")

    await handler.handle(event)
    await asyncio.sleep(0.1)
    assert answerer.await_count == 1

    # Another worker picking up the redelivery long after the answer lease
    # would have expired must still lose the claim: the reply is already public.
    assert not await registry.claim_comment(
        "idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL
    )
    other = _question_handler(registry, task, answerer, AsyncMock())
    await other.handle(event)
    await asyncio.sleep(0.1)

    assert answerer.await_count == 1


@pytest.mark.asyncio
async def test_due_snapshot_missed_comment_is_still_merged_into_the_round():
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await _review_task(service, registry)
    now = datetime.now(UTC)
    await registry.save_pending_feedback(
        "idem-1", [("c-1", "/code 第一条")], available_at=now
    )
    await registry.save_pending_feedback(
        "idem-1", [("c-2", "/code 第二条")], available_at=now
    )

    # The flusher's snapshot only saw the first comment.
    await service.submit_feedback("idem-1", "/code 第一条", ["mr-comment:c-1"])
    await service.discard_pending_feedback("idem-1", ["c-1"])

    task = await registry.get("idem-1")
    rounds = [r for r in task.rounds if r.trigger_type == "review_feedback"]
    assert len(rounds) == 1
    assert "第一条" in rounds[0].feedback
    assert "第二条" in rounds[0].feedback
    assert await registry.pending_feedback("idem-1") == []


@pytest.mark.asyncio
async def test_two_workers_answer_a_duplicate_question_once():
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await service.accept_event(_event())
    await service.wait_for_idle()
    await service.confirm_evaluation("idem-1", "req-1")
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    answerer = AsyncMock(return_value="这段逻辑是幂等的。")
    reply = AsyncMock()
    spawner: list = []

    def make_handler() -> AeolusCommentHandler:
        return AeolusCommentHandler(
            registry=_LookupOnly(registry, task),
            comments=SimpleNamespace(
                enqueue=AsyncMock(),
                spawn=lambda operation: spawner.append(operation),
            ),
            meegle=SimpleNamespace(
                get_user_email=AsyncMock(
                    return_value="wangwenchao.129@bytedance.com"
                )
            ),
            reply=reply,
            answerer=answerer,
        )

    event = _comment_event("q-1", "/ask 这段为什么幂等？")
    for handler in (make_handler(), make_handler()):
        await handler.handle(event)
    for operation in spawner:
        await operation
    # The second worker owns no in-process state; only the shared claim can
    # stop it from running the same answer twice.
    assert answerer.await_count == 1
    assert reply.await_count == 1


class _LookupOnly:
    """One worker's view: its own lookup result, the shared registry for claims."""

    def __init__(self, registry: AeolusTaskRegistry, task: AeolusTask) -> None:
        self._registry = registry
        self._task = task

    async def find_active_for_mr(self, repo_id: str, mr_iid: int):
        return self._task

    def __getattr__(self, name: str):
        return getattr(self._registry, name)


@pytest.mark.asyncio
async def test_claimed_comment_text_is_not_resubmitted_with_the_next_batch():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    service, registry, _, _, _ = _service(
        EvaluationResult(True, "通过", (), ()), engine=engine
    )
    await _review_task(service, registry)
    # The first round consumed c-1 but its pending row survived the cleanup.
    await service.submit_feedback("idem-1", "旧要求：改成批量化", ["mr-comment:c-1"])
    # The round runs as a background task; wait for it instead of racing it.
    await service.wait_for_idle()
    await registry.save_pending_feedback(
        "idem-1",
        [("c-1", "旧要求：改成批量化"), ("c-2", "新要求：加上缓存")],
        available_at=datetime.now(UTC),
    )

    entries = await registry.pending_feedback("idem-1")
    await service.submit_feedback(
        "idem-1",
        "\n\n".join(text for _, text in entries),
        [f"mr-comment:{comment_id}" for comment_id, _ in entries],
        entries=entries,
    )

    rounds = [
        round_
        for round_ in (await registry.get("idem-1")).rounds
        if round_.trigger_type == "review_feedback"
    ]
    assert len(rounds) == 2
    assert "新要求" in rounds[1].feedback
    assert "旧要求" not in rounds[1].feedback


@pytest.mark.asyncio
async def test_flusher_sends_only_the_text_of_unclaimed_comments():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    service, registry, _, _, _ = _service(
        EvaluationResult(True, "通过", (), ()), engine=engine
    )
    await _review_task(service, registry)
    await service.submit_feedback("idem-1", "旧要求：改成批量化", ["mr-comment:c-1"])
    await service.wait_for_idle()
    task = await registry.get("idem-1")
    if task.state is not AeolusState.REVIEW:
        await registry.transition("idem-1", {task.state}, AeolusState.REVIEW)
    await registry.save_pending_feedback(
        "idem-1",
        [("c-1", "旧要求：改成批量化"), ("c-2", "新要求：加上缓存")],
        available_at=datetime.now(UTC),
    )

    comments = AeolusCommentService(
        service=service, window_seconds=0, defer_seconds=0.01, max_deferrals=1
    )
    await comments.enqueue(
        task=await registry.get("idem-1"), event=_comment_event("c-2", "/code 新要求：加上缓存")
    )
    await comments._pending["idem-1"].flusher

    rounds = [
        round_
        for round_ in (await registry.get("idem-1")).rounds
        if round_.trigger_type == "review_feedback"
    ]
    assert len(rounds) == 2
    assert "新要求" in rounds[1].feedback
    assert "旧要求" not in rounds[1].feedback


@pytest.mark.asyncio
async def test_completion_record_is_written_only_after_the_reply_is_visible():
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    order: list[str] = []

    async def reply(event, body):
        order.append("reply")

    handler = _question_handler(
        registry, task, AsyncMock(return_value="这段逻辑是幂等的。"), reply
    )
    real_complete = registry.complete_comment

    async def complete(task_id, request_id, token=None):
        order.append("complete")
        return await real_complete(task_id, request_id, token)

    handler._registry.complete_comment = complete
    await handler.handle(_comment_event("q-1", "/ask 这段为什么幂等？"))
    await asyncio.sleep(0.1)

    # A permanent record before the reply would turn a crash into a silently
    # lost answer: nothing would ever retry a comment the record already covers.
    assert order == ["reply", "complete"]
    # A completed claim refuses the retry: the comment is answered for good.
    assert (
        await registry.claim_comment(
            "idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL
        )
        is None
    )


@pytest.mark.asyncio
async def test_answer_lost_before_publishing_is_not_recorded_as_completed():
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    dying = _question_handler(
        registry, task, AsyncMock(return_value="这段逻辑是幂等的。"), reply
    )
    dying._registry.complete_comment = AsyncMock(side_effect=RuntimeError("db down"))
    dying._send_reply = AsyncMock(return_value=False)
    event = _comment_event("q-1", "/ask 这段为什么幂等？")

    await dying.handle(event)
    await asyncio.sleep(0.1)

    assert replies == []
    # Nothing was published, so the claim must stay a retryable lease instead of
    # a permanent record; take it back so the redelivery below can claim it.
    lease = await registry.claim_comment(
        "idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL
    )
    assert lease is not None
    await registry.release_comment("idem-1", "mr-comment-answer:q-1", lease)

    second = _question_handler(
        registry, task, AsyncMock(return_value="这段逻辑是幂等的。"), reply
    )
    await second.handle(event)
    await asyncio.sleep(0.1)

    assert len(replies) == 1


@pytest.mark.asyncio
async def test_completion_write_failure_does_not_publish_a_duplicate_answer():
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL
    from conciliator.services.comment_events import ThreadComment

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(
        registry, task, AsyncMock(return_value="这段逻辑是幂等的。"), reply
    )
    handler._registry.complete_comment = AsyncMock(side_effect=RuntimeError("db down"))
    event = _comment_event("q-1", "/ask 这段为什么幂等？")

    await handler.handle(event)
    await asyncio.sleep(0.1)

    # The reply is out, the record is not; the lease expires and a redelivery
    # has to recognise its own answer in the thread instead of posting again.
    assert len(replies) == 1

    second = _question_handler(
        registry, task, AsyncMock(return_value="这段逻辑是幂等的。"), reply
    )
    second._thread_reader = AsyncMock(
        return_value=[ThreadComment("reply-1", replies[0], "bot@bytedance.com")]
    )
    await second.handle(event)
    await asyncio.sleep(0.1)

    assert len(replies) == 1
    # A completed claim refuses the retry: the comment is answered for good.
    assert (
        await registry.claim_comment(
            "idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL
        )
        is None
    )


@pytest.mark.asyncio
async def test_worker_that_lost_its_answer_lease_cannot_publish():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    service, registry, _, _, _ = _service(
        EvaluationResult(True, "通过", (), ()), engine=engine
    )
    task = await _review_task(service, registry)
    replies: list[str] = []
    started = asyncio.Event()
    finish = asyncio.Event()

    async def reply(event, body):
        replies.append(body)

    async def slow_answer(*_args, **_kwargs):
        started.set()
        await finish.wait()
        return "旧 worker 的答案"

    old = _question_handler(registry, task, slow_answer, reply)
    event = _comment_event("q-1", "/ask 这段为什么幂等？")
    await old.handle(event)
    await started.wait()

    # The old worker stalled past its lease; the comment is re-claimable.
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE meegle_ai_node_comment_claim SET expires_at = '2000-01-01 00:00:00'"
        )
    fresh = _question_handler(
        registry, task, AsyncMock(return_value="新 worker 的答案"), reply
    )
    await fresh.handle(event)
    await asyncio.sleep(0.1)
    assert len(replies) == 1

    finish.set()
    for spawned in old._comments.spawner.tasks:
        await spawned

    # The stalled worker came back to a lease it no longer owns.
    assert len(replies) == 1
    assert "新 worker" in replies[0]


@pytest.mark.asyncio
async def test_worker_that_lost_its_answer_lease_stays_silent_on_failure():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    service, registry, _, _, _ = _service(
        EvaluationResult(True, "通过", (), ()), engine=engine
    )
    task = await _review_task(service, registry)
    replies: list[str] = []
    started = asyncio.Event()
    finish = asyncio.Event()

    async def reply(event, body):
        replies.append(body)

    async def failing_answer(*_args, **_kwargs):
        started.set()
        await finish.wait()
        raise CocoRunFailed("agent failed", retryable=False)

    old = _question_handler(registry, task, failing_answer, reply)
    event = _comment_event("q-1", "/ask 这段为什么幂等？")
    await old.handle(event)
    await started.wait()

    # The old worker stalled past its lease; the comment is re-claimable.
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE meegle_ai_node_comment_claim SET expires_at = '2000-01-01 00:00:00'"
        )
    fresh = _question_handler(
        registry, task, AsyncMock(return_value="新 worker 的答案"), reply
    )
    await fresh.handle(event)
    await asyncio.sleep(0.1)

    finish.set()
    for spawned in old._comments.spawner.tasks:
        await spawned

    # The stale worker's failure must not mark the record failed or tell the
    # author: the new claim owner is answering right now.
    assert len(replies) == 1
    assert "新 worker" in replies[0]
    record = await registry.question_record(task.task_id, event.comment_id)
    assert record is not None
    assert record["status"] == "published"


@pytest.mark.asyncio
async def test_stale_due_task_snapshot_does_not_flush_new_future_comment():
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await _review_task(service, registry)
    await registry.save_pending_feedback(
        "idem-1",
        [("c-1", "新要求：加上缓存")],
        available_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    # The recovery scan selected this task while an older row was due; by the
    # time the flush runs only the future row is left.
    await service._flush_pending_feedback("idem-1")

    task = await registry.get("idem-1")
    assert not [r for r in task.rounds if r.trigger_type == "review_feedback"]
    assert await registry.pending_feedback("idem-1") == [("c-1", "新要求：加上缓存")]


@pytest.mark.asyncio
async def test_question_is_answered_while_coding_is_blocked():
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await _review_task(service, registry)
    await registry.transition("idem-1", {AeolusState.REVIEW}, AeolusState.CODING_BLOCKED)
    task = await registry.get("idem-1")
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(
        registry, task, AsyncMock(return_value="这段逻辑是幂等的。"), reply
    )
    await handler.handle(_comment_event("q-1", "/ask 这段为什么幂等？"))
    await asyncio.wait_for(asyncio.gather(*handler._comments.spawner.tasks), timeout=2)

    assert len(replies) == 1
    assert "幂等" in replies[0]


@pytest.mark.asyncio
async def test_question_is_queued_while_a_round_is_recoding():
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    await _review_task(service, registry)
    await registry.transition("idem-1", {AeolusState.REVIEW}, AeolusState.RECODING)
    task = await registry.get("idem-1")
    answerer = AsyncMock(return_value="编码期间不该回答")
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(registry, task, answerer, reply)
    await handler.handle(_comment_event("q-1", "/ask 这段为什么幂等？"))
    await asyncio.sleep(0.1)

    answerer.assert_not_awaited()
    assert replies == []
    # The question is on disk and owns no claim, so the scan answers it later.
    record = await registry.question_record("idem-1", "q-1")
    assert record is not None and record["status"] == "pending"
    token = await registry.claim_comment(
        "idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL
    )
    assert token is not None


@pytest.mark.asyncio
async def test_recovery_scan_answers_a_pending_question_record():
    """A question persisted by a crashed worker is answered on restart."""
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    await registry.record_question(
        "idem-1",
        "q-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="thread-q-1",
        position={"Path": "a.py", "StartLine": 3},
        parent=None,
        question_text="/ask 这段为什么幂等？",
        author_email="dev@bytedance.com",
    )
    answerer = AsyncMock(return_value="这段逻辑是幂等的。")
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(registry, task, answerer, reply)
    await handler.resume_pending_questions()

    assert answerer.await_count == 1
    assert len(replies) == 1
    record = await registry.question_record("idem-1", "q-1")
    assert record["status"] == "published"


@pytest.mark.asyncio
async def test_recovery_scan_republishes_a_saved_answer_without_reanswering():
    """An answered-but-unpublished record is published without an agent run."""
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    await registry.record_question(
        "idem-1",
        "q-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="thread-q-1",
        position={"Path": "a.py", "StartLine": 3},
        parent=None,
        question_text="/ask 这段为什么幂等？",
        author_email="dev@bytedance.com",
    )
    token = await registry.claim_comment("idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL)
    assert await registry.mark_question_answered(
        "idem-1", "q-1", "已保存的答案", token=token, attempt=0
    )
    await registry.release_comment("idem-1", "mr-comment-answer:q-1", token)
    answerer = AsyncMock(return_value="不应被调用的答案")
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(registry, task, answerer, reply)
    await handler.resume_pending_questions()

    assert answerer.await_count == 0
    assert len(replies) == 1
    assert "已保存的答案" in replies[0]
    record = await registry.question_record("idem-1", "q-1")
    assert record["status"] == "published"


@pytest.mark.asyncio
async def test_recovery_defers_a_question_while_a_round_is_recoding():
    """A busy session defers the question instead of failing it for good."""
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    await registry.record_question(
        "idem-1",
        "q-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="thread-q-1",
        position={"Path": "a.py", "StartLine": 3},
        parent=None,
        question_text="/ask 这段为什么幂等？",
        author_email="dev@bytedance.com",
    )
    await registry.transition("idem-1", {AeolusState.REVIEW}, AeolusState.RECODING)
    answerer = AsyncMock(return_value="不应被调用的答案")
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(registry, task, answerer, reply)
    await handler.resume_pending_questions()

    record = await registry.question_record("idem-1", "q-1")
    assert record["status"] == "pending"
    assert answerer.await_count == 0
    assert replies == []


@pytest.mark.asyncio
async def test_publish_scan_waits_for_the_worker_holding_the_claim():
    """A scan must not publish an answer another worker may still be sending."""
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    await registry.record_question(
        "idem-1",
        "q-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="thread-q-1",
        position={"Path": "a.py", "StartLine": 3},
        parent=None,
        question_text="/ask 这段为什么幂等？",
        author_email="dev@bytedance.com",
    )
    token = await registry.claim_comment("idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL)
    assert token is not None
    assert await registry.mark_question_answered(
        "idem-1", "q-1", "已保存的答案", token=token, attempt=0
    )
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(registry, task, AsyncMock(), reply)
    await handler.resume_pending_questions()

    record = await registry.question_record("idem-1", "q-1")
    assert record["status"] == "answered"
    assert replies == []


@pytest.mark.asyncio
async def test_recovery_does_not_resend_a_question_with_no_cursor():
    """A send that may have happened is reported, never repeated."""
    from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL

    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    await registry.record_question(
        "idem-1",
        "q-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="thread-q-1",
        position={"Path": "a.py", "StartLine": 3},
        parent=None,
        question_text="/ask 这段为什么幂等？",
        author_email="dev@bytedance.com",
    )
    token = await registry.claim_comment("idem-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL)
    assert await registry.mark_question_sending(
        "idem-1", "q-1", session_id="coding-session", token=token, attempt=0
    )
    await registry.release_comment("idem-1", "mr-comment-answer:q-1", token)
    answerer = AsyncMock(return_value="不应被调用的答案")
    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(registry, task, answerer, reply)
    await handler.resume_pending_questions()

    assert answerer.await_count == 0
    assert len(replies) == 1
    assert "重新提问" in replies[0]
    record = await registry.question_record("idem-1", "q-1")
    assert record["status"] == "unknown"


@pytest.mark.asyncio
async def test_question_is_marked_as_sending_before_the_prompt_goes_out():
    """The record says a send may have happened before the prompt leaves."""
    service, registry, _, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    await registry.record_question(
        "idem-1",
        "q-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="thread-q-1",
        position={"Path": "a.py", "StartLine": 3},
        parent=None,
        question_text="/ask 这段为什么幂等？",
        author_email="dev@bytedance.com",
    )
    seen: dict[str, str | None] = {}

    async def answerer(task_obj, text, position, on_started=None):
        await on_started(task_obj.coco_session_id, "", "")
        record = await registry.question_record("idem-1", "q-1")
        seen["status"] = record["status"]
        seen["session_id"] = record["session_id"]
        await on_started(task_obj.coco_session_id, "run-1", "evt-1")
        return "这段逻辑是幂等的。"

    replies: list[str] = []

    async def reply(event, body):
        replies.append(body)

    handler = _question_handler(registry, task, answerer, reply)
    await handler.resume_pending_questions()

    assert seen["status"] == "sending"
    assert seen["session_id"] == task.coco_session_id
    record = await registry.question_record("idem-1", "q-1")
    assert record["status"] == "published"


@pytest.mark.asyncio
async def test_resume_answer_holds_the_task_lease():
    """Resuming a run serializes on the task lease like a normal answer."""
    service, registry, runner, _, _ = _service(EvaluationResult(True, "通过", (), ()))
    task = await _review_task(service, registry)
    leases: list[bool] = []

    async def resume_answer(session_id, run_id, event_id):
        leases.append(await registry.acquire_lease("idem-1", "other-worker"))
        return "续读的答案"

    runner.resume_answer = resume_answer

    answer = await service.resume_answer(
        "idem-1", task.coco_session_id, "run-1", "evt-1"
    )

    assert answer == "续读的答案"
    assert leases == [False]
