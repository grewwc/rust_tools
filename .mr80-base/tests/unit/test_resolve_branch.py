from datetime import UTC, datetime

from conciliator.domain.resolve_branch import (
    MAX_RESOLVE_BRANCH_BYTES,
    build_resolve_branch,
)


def test_build_resolve_branch_uses_slugs_and_shanghai_task_date():
    branch = build_resolve_branch(
        source_branch="Feat/Infra Only_Mode",
        target_branch="release/main@{candidate}",
        task_id=123,
        created_at=datetime(2026, 8, 27, 16, 30, tzinfo=UTC),
    )

    assert branch == "fix/feat-infra-only-mode-to-release-main-candidate-conflict-20260828-t123"


def test_build_resolve_branch_preserves_suffix_when_truncated():
    branch = build_resolve_branch(
        source_branch="feature/" + "source" * 80,
        target_branch="release/" + "target" * 80,
        task_id=987654,
        created_at=datetime(2026, 8, 28, 1, 2, tzinfo=UTC),
    )

    assert len(branch.encode("utf-8")) <= MAX_RESOLVE_BRANCH_BYTES
    assert branch.endswith("-conflict-20260828-t987654")
    assert "-to-" in branch


def test_build_resolve_branch_is_stable_for_same_task():
    kwargs = {
        "source_branch": "feat/source",
        "target_branch": "main",
        "task_id": 7,
        "created_at": datetime(2026, 8, 28, tzinfo=UTC),
    }

    assert build_resolve_branch(**kwargs) == build_resolve_branch(**kwargs)


def test_build_resolve_branch_rejects_non_positive_task_id():
    try:
        build_resolve_branch(
            source_branch="feat/source",
            target_branch="main",
            task_id=0,
            created_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    except ValueError as exc:
        assert str(exc) == "task_id must be positive"
    else:
        raise AssertionError("expected ValueError")
