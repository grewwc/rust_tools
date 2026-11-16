from conciliator.api.webhooks import _normalize_trigger_payload
from conciliator.integrations.nextcode.mrs import MR


def test_normalize_extracts_official_camel_case_fields():
    payload = {
        "Repository": {
            "Id": "repo-1",
            "CloneURL": "https://code.example.com/r.git",
        },
        "MergeRequest": {
            "Number": 42,
            "Title": "feat: add X",
            "SourceBranchName": "feature-x",
            "TargetBranchName": "main",
        },
    }
    normalized = _normalize_trigger_payload(payload)
    assert normalized["source_branch"] == "feature-x"
    assert normalized["target_branch"] == "main"
    assert normalized["repo_url"] == "https://code.example.com/r.git"
    assert normalized["mr_title"] == "feat: add X"
    assert normalized["mr_iid"] == 42
    assert normalized["repo_id"] == "repo-1"
    assert normalized["_raw"] is payload


def test_normalize_comment_event_with_fetched_mr():
    mr = MR(
        id="mr-id",
        iid=7,
        title="fix",
        description="",
        web_url="https://code.example.com/mr/7",
        source_repo_id="r2",
        target_repo_id="r2",
        source_branch="fix-bug",
        target_branch="release",
    )
    normalized = _normalize_trigger_payload(
        {"Repository": {"Id": "r2", "CloneURL": "https://clone/r2.git"}}, mr
    )
    assert normalized["source_branch"] == "fix-bug"
    assert normalized["target_branch"] == "release"
    assert normalized["mr_title"] == "fix"
    assert normalized["mr_iid"] == 7


def test_normalize_defaults_empty_on_missing():
    normalized = _normalize_trigger_payload({})
    assert normalized["source_branch"] == ""
    assert normalized["target_branch"] == ""
    assert normalized["mr_iid"] == 0
