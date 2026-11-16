import base64
from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class ConflictFile:
    file_path: str
    content: str
    first_path: str
    second_path: str
    conflicts_id: str


class ConflictSnapshotMismatchError(RuntimeError):
    """The live conflict snapshot belongs to a different pair of commits."""


def _decode_binary(content: str) -> str:
    """Decode Thrift `binary` as represented by the JSON protocol."""
    return base64.b64decode(content, validate=True).decode("utf-8")


async def list_conflict_files(
    client: NextCodeClient,
    repo_id: str,
    mr_iid: int,
    *,
    expected_conflicts_id: str | None = None,
) -> list[ConflictFile]:
    result = await client.call(
        "ListMergeRequestConflicts",
        {"RepoId": repo_id, "Number": mr_iid, "WithContent": True},
        read_only=True,
    )
    conflicts_id = result["ConflictsId"]
    # Codebase identifies a live snapshot as source_sha..target_sha. Check
    # before loading its contents, including when the conflict list is empty.
    if expected_conflicts_id is not None and conflicts_id != expected_conflicts_id:
        raise ConflictSnapshotMismatchError("conflict snapshot does not match the captured branch HEADs")
    files: list[ConflictFile] = []
    for item in result["Conflicts"]:
        header = item["ConflictHeader"]
        first_path = header["First"]["Path"]
        second_path = header["Second"]["Path"]
        content = item.get("Content")
        if content is None:
            detail = await get_conflict(
                client,
                repo_id,
                mr_iid,
                conflicts_id=conflicts_id,
                first_path=first_path,
                second_path=second_path,
            )
            content = detail.get("Content")
        files.append(
            ConflictFile(
                file_path=second_path or first_path,
                content=_decode_binary(content) if content is not None else "",
                first_path=first_path,
                second_path=second_path,
                conflicts_id=conflicts_id,
            )
        )
    return files


async def get_conflict(
    client: NextCodeClient,
    repo_id: str,
    mr_iid: int,
    *,
    conflicts_id: str,
    first_path: str,
    second_path: str,
) -> dict:
    result = await client.call(
        "GetMergeRequestConflict",
        {
            "RepoId": repo_id,
            "Number": mr_iid,
            "ConflictsId": conflicts_id,
            "FirstPath": first_path,
            "SecondPath": second_path,
            "WithContent": True,
        },
        read_only=True,
    )
    return result.get("Conflict") or {}
