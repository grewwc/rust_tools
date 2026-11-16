import pytest

from conciliator.services.templates import (
    build_discussion_body,
    build_lark_doc_markdown,
    build_lark_doc_xml,
    build_mr_description,
    build_retry_discussion_body,
)


def test_build_lark_doc_xml_contains_blocks():
    xml = build_lark_doc_xml(
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        blocks=[
            {
                "idx": 1,
                "conflict_id": "C0001",
                "file_path": "a.py",
                "start_line": 10,
                "end_line": 20,
                "source_text": "ours",
                "target_text": "theirs",
                "ai_resolution": "merged",
                "decision_summary": "keep both",
                "introduced_fix": "guard null",
                "source_author_email": "source@x.com",
                "target_author_email": "target@x.com",
            }
        ],

    )
    assert "<h1>" in xml
    assert "a.py" in xml
    assert "C0001" in xml
    assert "source@x.com" in xml
    assert "target@x.com" in xml
    assert "merged" in xml
    assert "验证结果" not in xml and "额外修复" not in xml


def test_build_lark_doc_markdown_contains_attribution_and_final_files():
    markdown = build_lark_doc_markdown(
        original_mr_number=10,
        source_branch="feat/source",
        source_commit="a" * 40,
        target_branch="main",
        target_commit="b" * 40,
        resolve_branch="fix/feat-source-to-main-conflict-20260828-t123",
        resolve_commit="c" * 40,
        blocks=[
            {
                "idx": 1,
                "conflict_id": "C0001",
                "file_path": "src/example.py",
                "start_line": 4,
                "end_line": 10,
                "source_text": "source()",
                "target_text": "target()",
                "decision_summary": "保留两侧逻辑",
                "introduced_fix": "补充组合后的空值处理",
                "source_author_email": "source@example.com",
                "source_author_commit_sha": "d" * 40,
                "source_path": "src/example.py",
                "source_start_line": 4,
                "source_end_line": 6,
                "target_author_email": "target@example.com",
                "target_author_commit_sha": "e" * 40,
                "target_path": "src/example.py",
                "target_start_line": 4,
                "target_end_line": 7,
            }
        ],

        resolved_files={"src/example.py": "def resolved():\n    return True\n"},
    )

    assert "marker" not in markdown
    assert "Source 作者和文件" in markdown
    assert "d" * 40 not in markdown and "e" * 40 not in markdown
    assert "source@example.com" in markdown
    assert "target@example.com" in markdown
    assert "保留两侧逻辑" in markdown
    assert "最终文件内容" not in markdown and "def resolved" not in markdown
    assert "验证结果" not in markdown and "额外修复" not in markdown


def test_build_mr_description_marks_approved():
    desc = build_mr_description(
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        lark_doc_url="https://lark/x",
        blocks=[
            {
                "idx": 1,
                "conflict_id": "C0001",
                "file_path": "a.py",
                "start_line": 10,
                "end_line": 20,
                "source_author_email": "source@x.com",
                "target_author_email": "target@x.com",
                "approved": True,
            }
        ],
    )
    assert "[x]" in desc
    assert "[ ]" not in desc


def test_build_mr_description_distinguishes_rejected_and_pending():
    desc = build_mr_description(
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        lark_doc_url="https://lark/x",
        retry_count=1,
        task_status="PARTIAL_REJECTED",
        blocks=[
            {
                "idx": 1,
                "conflict_id": "C0001",
                "file_path": "a.py",
                "start_line": 10,
                "end_line": 20,
                "approval_status": "REJECTED",
                "reject_reason": "保留 source\n并合入 target | guard",
            },
            {
                "idx": 2,
                "conflict_id": "C0002",
                "file_path": "b.py",
                "start_line": 2,
                "end_line": 3,
                "approval_status": "PENDING",
            },
        ],
    )
    assert "0 approved / 1 rejected / 1 pending" in desc
    assert "`PARTIAL_REJECTED`" in desc
    assert "重解次数**：1" in desc
    assert "✗ REJECTED：保留 source 并合入 target \\| guard" in desc


def test_build_retry_discussion_body_contains_reason_and_attempt_marker():
    body = build_retry_discussion_body(
        lark_doc_url="https://lark.test/file/retry-report",
        task_id=7,
        attempt=2,
        conflict_id="C0004",
        idx=4,
        file_path="src/policy.py",
        start_line=30,
        end_line=35,
        ai_resolution="if safe:\n    return target",
        decision_summary="按反馈保留安全检查",

        reject_reason="keep target validation",
    )
    assert "conciliator:task=7:block=C0004:attempt=2" in body
    assert "第 2 轮重解" in body
    assert "keep target validation" in body
    assert "重新回复 `/approval`" in body


def test_build_discussion_body_contains_approve_instruction():
    body = build_discussion_body(
        lark_doc_url="https://lark.test/file/report",
        task_id=123,
        conflict_id="C0001",
        idx=1,
        file_path="a.py",
        start_line=10,
        end_line=20,
        source_author_email="source@x.com",
        target_author_email="target@x.com",
        ours="x",
        theirs="y",
        ai_resolution="z",
        decision_summary="keep both",

        thinking="raw private reasoning",
    )
    assert "冲突块 #1" in body
    assert "approve" in body
    assert "keep both" in body
    assert "Source 侧冲突块内容" in body
    assert "Target 侧冲突块内容" in body
    assert "conciliator:task=123:block=C0001" in body
    assert "raw private reasoning" not in body


def test_mr_description_assigns_global_write_commands_to_resolve_initiator():
    description = build_mr_description(
        original_mr_number=42,
        source_branch="source",
        target_branch="target",
        resolve_branch="resolve/source-to-target",
        resolve_commit="a" * 40,
        lark_doc_url="https://lark.test/file/token",
        blocks=[],

    )

    assert description.count("限 `/resolve` 发起人") == 3
    assert "MR author" not in description


def test_build_discussion_body_renders_multiline_code_without_template_indent():
    body = build_discussion_body(
        lark_doc_url="https://lark.test/file/report",
        task_id=4,
        conflict_id="C0003",
        idx=3,
        file_path="validation/resolve_approval_e2e/policy.py",
        start_line=34,
        end_line=37,
        source_author_email="source@example.com",
        target_author_email="target@example.com",
        ours='if change.owner_approved:\n    return "auto"',
        theirs='if change.security_review_required:\n    return "manual"',
        ai_resolution=(
            'if change.security_review_required:\n    return "manual"\nif change.owner_approved:\n    return "auto"'
        ),
        decision_summary="保持安全审查优先级",
    )

    expected = "\n".join(
        [
            "<!-- conciliator:task=4:block=C0003 -->",
            "**冲突块 #3** · `validation/resolve_approval_e2e/policy.py:34-37`  ",
            "**Source 作者和文件**：source@example.com · `validation/resolve_approval_e2e/policy.py`  ",
            "**Target 作者和文件**：target@example.com · `validation/resolve_approval_e2e/policy.py`",
            "",
            "**Source 侧冲突块内容**：",
            "",
            "```",
            "if change.owner_approved:",
            '    return "auto"',
            "```",
            "",
            "**Target 侧冲突块内容**：",
            "",
            "```",
            "if change.security_review_required:",
            '    return "manual"',
            "```",
            "",
            "**🤖 解决后的代码**：",
            "",
            "```",
            "if change.security_review_required:",
            '    return "manual"',
            "if change.owner_approved:",
            '    return "auto"',
            "```",
            "",
            "**解决决策**：保持安全审查优先级",
            "",
            "---",
            "👤 回复 `/approval`、`approve` 或 `reject <reason>` 即可确认",
            "",
        ]
    )
    assert body == expected


@pytest.mark.parametrize("field", ["ours", "theirs", "ai_resolution", "decision_summary"])
def test_long_discussion_preserves_identity_authors_locations_and_approval(field):
    kwargs = dict(
        task_id=27, conflict_id="C0001", idx=1,
        file_path="resolved.py", start_line=143, end_line=143,
        source_author_email="source@example.com", source_path="source.py",
        source_start_line=134, source_end_line=134,
        target_author_email="target@example.com", target_path="target.py",
        target_start_line=143, target_end_line=143,
        ours="source()", theirs="target()", ai_resolution="resolved()",
        decision_summary="keep both", lark_doc_url="https://lark.test/file/report",
    )
    kwargs[field] = "长内容\n" * 2000
    body = build_discussion_body(**kwargs)
    assert len(body) <= 6000
    assert body == (
        "<!-- conciliator:task=27:block=C0001 -->\n"
        "**冲突块 #1** · `resolved.py:143-143`  \n"
        "**Source 作者和文件**：source@example.com · `source.py:134-134`  \n"
        "**Target 作者和文件**：target@example.com · `target.py:143-143`\n\n"
        "冲突块内容过长，详情见：[飞书文档链接](https://lark.test/file/report)\n\n"
        "👤 回复 `/approval`、`approve` 或 `reject <reason>` 即可确认\n"
    )


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("length", [5999, 6000, 6001])
def test_comment_limit_includes_markdown_and_marker(retry, length):
    kwargs = dict(
        task_id=27, conflict_id="C0001", idx=1, file_path="a.py",
        start_line=1, end_line=1, ai_resolution="", decision_summary="",
        lark_doc_url="https://lark.test/file/report",
    )
    if retry:
        builder = build_retry_discussion_body
        kwargs.update(attempt=2, reject_reason="reason")
    else:
        builder = build_discussion_body
        kwargs.update(ours="", theirs="")
    overhead = len(builder(**kwargs))
    kwargs["ai_resolution"] = "中" * (length - overhead)
    body = builder(**kwargs)
    if length <= 6000:
        assert len(body) == length
        assert kwargs["ai_resolution"] in body
        assert "冲突块内容过长" not in body
    else:
        assert len(body) <= 6000
        assert "[飞书文档链接](https://lark.test/file/report)" in body
        assert "conciliator:task=27:block=C0001" in body
        if retry:
            assert ":attempt=2" in body
            assert "第 2 轮重解" in body
            assert "重新回复 `/approval`" in body


@pytest.mark.parametrize("field", ["decision_summary", "reject_reason"])
def test_long_retry_metadata_also_uses_report(field):
    kwargs = dict(
        task_id=27, conflict_id="C0001", idx=1, attempt=2, file_path="a.py",
        start_line=1, end_line=1, ai_resolution="fixed()", decision_summary="",
        reject_reason="", lark_doc_url="https://lark.test/file/retry-report",
    )
    kwargs[field] = "x" * 6000
    body = build_retry_discussion_body(**kwargs)
    assert len(body) <= 6000
    assert "[飞书文档链接](https://lark.test/file/retry-report)" in body
    assert ":attempt=2" in body
    assert "fixed()" not in body
