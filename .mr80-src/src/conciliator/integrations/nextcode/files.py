import base64
from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient, NextCodeError


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    content: str
    encoding: str


async def get_file(
    client: NextCodeClient,
    repo_id: str,
    *,
    revision: str,
    path: str,
) -> RepositoryFile | None:
    try:
        result = await client.call(
            "GetFile",
            {"RepoId": repo_id, "Revision": revision, "Path": path},
            read_only=True,
        )
    except NextCodeError as exc:
        if exc.code.startswith("NotFound"):
            return None
        raise
    data = result.get("File")
    if not isinstance(data, dict):
        return None
    normal_file = data.get("NormalFile")
    if isinstance(normal_file, dict):
        meta = normal_file.get("Meta") or {}
        path = str(meta.get("Path") or path)
        encoding = str(normal_file.get("Encoding") or "")
        content = str(normal_file.get("Content") or "")
    else:
        path = str(data.get("Path") or path)
        encoding = str(data.get("Encoding") or "")
        content = str(data.get("Content") or "")
    if encoding.lower() == "base64":
        content = base64.b64decode(content, validate=True).decode("utf-8")
    return RepositoryFile(path=path, content=content, encoding=encoding)
