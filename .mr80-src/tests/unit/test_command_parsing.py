import pytest

from conciliator.domain.commands import (
    ApproveAll,
    ApproveBlock,
    Cancel,
    Ignored,
    RejectBlock,
    Retry,
    Status,
    parse_comment,
)


@pytest.mark.parametrize(
    "body",
    [
        "approve",
        "APPROVE",
        "/approval",
        "lgtm",
        "LGTM",
        "✅",
        "+1",
        "👍",
    ],
)
def test_single_block_approve_in_discussion(body):
    cmd = parse_comment(body, in_discussion=True)
    assert isinstance(cmd, ApproveBlock)


@pytest.mark.parametrize(
    "body",
    [
        "[LGTM]",
        "[JIAYI]",
        "[CheckMark]",
        "[THUMBSUP]",
        "[lgtm] looks good",
    ],
)
def test_single_block_approve_bracket_aliases(body):
    cmd = parse_comment(body, in_discussion=True)
    assert isinstance(cmd, ApproveBlock)


@pytest.mark.parametrize(
    "body",
    [
        "reject typo here",
        "/reject typo here",
        "REJECT no",
        "❌ keep theirs",
        "-1 bad",
        "👎 nope",
    ],
)
def test_single_block_reject_in_discussion(body):
    cmd = parse_comment(body, in_discussion=True)
    assert isinstance(cmd, RejectBlock)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("[MinusOne]keep source behavior", "keep source behavior"),
        ("[CrossMark]保留目标分支校验", "保留目标分支校验"),
        ("[ThumbsDown]do not remove fallback", "do not remove fallback"),
        ("[MinusOne] keep source behavior", "keep source behavior"),
        ("[CrossMark] 保留目标分支校验", "保留目标分支校验"),
        ("[thumbsdown] do not remove fallback", "do not remove fallback"),
    ],
)
def test_single_block_reject_bracket_aliases(body, reason):
    cmd = parse_comment(body, in_discussion=True)
    assert isinstance(cmd, RejectBlock)
    assert cmd.reason == reason


@pytest.mark.parametrize("body", ["[MinusOne]", "[CrossMark]   ", "[ThumbsDown]"])
def test_reject_bracket_aliases_require_non_empty_reason(body):
    cmd = parse_comment(body, in_discussion=True)
    assert isinstance(cmd, Ignored)


def test_reject_requires_non_empty_reason():
    cmd = parse_comment("reject", in_discussion=True)
    assert isinstance(cmd, Ignored)


def test_global_approve_all_in_discussion():
    cmd = parse_comment("/resolve approve all", in_discussion=True)
    assert isinstance(cmd, ApproveAll)


def test_global_retry_in_main_line():
    cmd = parse_comment("/resolve retry", in_discussion=False)
    assert isinstance(cmd, Retry)


def test_global_status_anywhere():
    assert isinstance(parse_comment("/resolve status", False), Status)
    assert isinstance(parse_comment("/resolve status", True), Status)


def test_global_cancel_case_insensitive():
    assert isinstance(parse_comment("/Resolve CANCEL", False), Cancel)


def test_unknown_command_ignored():
    assert isinstance(parse_comment("looks good to me!", True), Ignored)
    assert isinstance(parse_comment("random comment", False), Ignored)


def test_empty_body_ignored():
    assert isinstance(parse_comment("", True), Ignored)
    assert isinstance(parse_comment("   ", False), Ignored)


def test_approve_token_outside_discussion_ignored():
    # "approve" 不带 /resolve 在 MR 主线不算单块指令
    assert isinstance(parse_comment("approve", False), Ignored)


@pytest.mark.parametrize("body", ["/resolve cancel 123", "/RESOLVE CANCEL 123"])
def test_cancel_can_select_task(body):
    assert parse_comment(body, in_discussion=False) == Cancel(task_id=123)


@pytest.mark.parametrize("body", ["/resolve cancel 0", "/resolve cancel -1", "/resolve cancel abc"])
def test_cancel_rejects_invalid_task_id(body):
    assert isinstance(parse_comment(body, in_discussion=False), Ignored)
