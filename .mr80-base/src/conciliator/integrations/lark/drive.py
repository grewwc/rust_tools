from dataclasses import dataclass

from conciliator.integrations.lark.client import LarkClient, LarkError

MAX_MARKDOWN_UPLOAD_BYTES = 20 * 1024 * 1024
LARK_FILE_URL_BASE = "https://bytedance.larkoffice.com/file"


@dataclass(frozen=True)
class MarkdownUpload:
    file_token: str
    file_name: str
    size_bytes: int
    web_url: str


async def upload_markdown(
    client: LarkClient,
    *,
    file_name: str,
    content: bytes,
    folder_token: str,
) -> MarkdownUpload:
    """Upload one Drive-native Markdown file with the app tenant identity."""
    if not file_name.endswith(".md"):
        raise ValueError("file_name must end with .md")
    if not content:
        raise ValueError("Markdown content must not be empty")
    if len(content) > MAX_MARKDOWN_UPLOAD_BYTES:
        raise ValueError("Markdown content exceeds the 20 MiB upload limit")
    if not folder_token.strip():
        raise ValueError("folder_token must not be empty")

    response = await client.request(
        "POST",
        "/drive/v1/files/upload_all",
        data={
            "file_name": file_name,
            "parent_type": "explorer",
            "parent_node": folder_token,
            "size": str(len(content)),
        },
        files={"file": (file_name, content, "text/markdown")},
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise LarkError(
            "Lark Markdown upload returned invalid JSON "
            f"(HTTP {response.status_code})"
        ) from exc
    if response.status_code >= 400 or payload.get("code") != 0:
        raise LarkError(
            "Lark Markdown upload failed "
            f"(HTTP {response.status_code}, code={payload.get('code')}, "
            f"msg={payload.get('msg', '')})"
        )
    file_token = str((payload.get("data") or {}).get("file_token") or "")
    if not file_token:
        raise LarkError("Lark Markdown upload response has no file_token")
    return MarkdownUpload(
        file_token=file_token,
        file_name=file_name,
        size_bytes=len(content),
        web_url=f"{LARK_FILE_URL_BASE}/{file_token}",
    )
