from dataclasses import dataclass
from urllib.parse import urlparse

from conciliator.integrations.lark.client import LarkClient


@dataclass(frozen=True)
class DocRef:
    doc_token: str
    url: str


@dataclass(frozen=True)
class DocumentRef:
    kind: str
    token: str


def extract_document_ref(url: str) -> DocumentRef:
    parsed = urlparse(url.strip())
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme not in {"http", "https"} or len(parts) < 2:
        raise ValueError("requirement document must be a Feishu docx or wiki link")
    kind, token = parts[0], parts[1]
    if kind not in {"docx", "wiki"} or not token:
        raise ValueError("requirement document must be a Feishu docx or wiki link")
    return DocumentRef(kind=kind, token=token)


async def _resolve_document_id(
    client: LarkClient, ref: DocumentRef, *, access_token: str | None
) -> str:
    document_id = ref.token
    if ref.kind == "wiki":
        # Contract verified against lark-oapi's generated GetNodeSpaceRequest.
        node_data = await client._request(
            "GET",
            "/wiki/v2/spaces/get_node",
            params={"token": ref.token},
            access_token=access_token,
        )
        node = node_data.get("node") or {}
        if node.get("obj_type") != "docx" or not node.get("obj_token"):
            raise ValueError("the Feishu wiki node is not a docx document")
        document_id = str(node["obj_token"])
    return document_id


async def read_document_text(
    client: LarkClient, url: str, *, access_token: str | None = None
) -> str:
    ref = extract_document_ref(url)
    document_id = await _resolve_document_id(
        client, ref, access_token=access_token
    )

    # Contract verified against lark-oapi's generated RawContentDocumentRequest.
    data = await client._request(
        "GET",
        f"/docx/v1/documents/{document_id}/raw_content",
        access_token=access_token,
    )
    content = str(data.get("content") or "").strip()
    if not content:
        raise ValueError("the requirement document is empty")
    return content


async def read_document_markdown(
    client: LarkClient, url: str, *, access_token: str | None = None
) -> str:
    """Export Markdown using the supplied user token, or the application's token."""
    ref = extract_document_ref(url)
    document_id = await _resolve_document_id(
        client, ref, access_token=access_token
    )
    data = await client._request(
        "POST",
        f"/docs_ai/v1/documents/{document_id}/fetch",
        json={
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
        access_token=access_token,
    )
    document = data.get("document") or {}
    content = str(document.get("content") or "").strip()
    if not content:
        raise ValueError("the Feishu document is empty")
    return content


async def create_doc(
    client: LarkClient,
    *,
    title: str,
    content_xml: str,
    folder_token: str,
) -> DocRef:
    """创建飞书 docx 文档(使用 XML 内容)。"""
    data = await client._request(
        "POST",
        "/docx/v1/documents",
        json={
            "folder_token": folder_token,
            "title": title,
            "content": content_xml,
        },
    )
    doc_token = data["document"]["document_id"]
    url = f"https://bytedance.larkoffice.com/docx/{doc_token}"
    return DocRef(doc_token=doc_token, url=url)


async def update_doc(
    client: LarkClient,
    *,
    doc_token: str,
    command: str,  # "append" | "str_replace" | ...
    content_xml: str,
) -> None:
    await client._request(
        "PATCH",
        f"/docx/v1/documents/{doc_token}",
        json={"command": command, "content": content_xml},
    )
