"""Upload the reusable Markdown fixture with Conciliator's Lark client.

For a local contract check, inject ``LARK_TENANT_ACCESS_TOKEN`` into this
process. Without that variable, the script uses Conciliator's SecretsProvider
to read the App Secret and exchange it for a tenant token. Neither credential
is printed.
"""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from conciliator.config import (
    LARK_APP_ID_DEFAULT,
    LARK_APP_SECRET_ALIAS_DEFAULT,
    LARK_MARKDOWN_FOLDER_TOKEN_DEFAULT,
    Settings,
)
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.drive import upload_markdown
from conciliator.utils.secrets import SecretsProvider, build_secrets_provider

DEFAULT_FIXTURE = Path("tests/fixtures/lark-upload-validation.md")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Conciliator's tenant-token Markdown upload method."
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--app-id")
    parser.add_argument("--folder-token")
    parser.add_argument("--kms-secret-alias")
    return parser.parse_args()


async def _run() -> int:
    args = _args()
    settings = Settings(
        env="prod",
        service_psm="lark-upload-verifier",
        mysql_psm="unused",
        lark_app_id=args.app_id or LARK_APP_ID_DEFAULT,
        lark_markdown_folder_token=(
            args.folder_token or LARK_MARKDOWN_FOLDER_TOKEN_DEFAULT
        ),
        kms_secret_lark_app_secret_alias=(
            args.kms_secret_alias or LARK_APP_SECRET_ALIAS_DEFAULT
        ),
    )
    source_path = args.file.resolve()
    content = source_path.read_bytes()
    remote_name = f"conciliator-lark-upload-{args.run_id}.md"

    secrets: SecretsProvider | None = None
    client: LarkClient | None = None
    try:
        injected_token = os.getenv("LARK_TENANT_ACCESS_TOKEN")
        if injected_token:
            token_source = "process_env"
            client = LarkClient(
                settings,
                None,
                tenant_access_token=injected_token,
            )
        else:
            token_source = "kms"
            secrets = build_secrets_provider(settings)
            client = LarkClient(settings, secrets)
        uploaded = await upload_markdown(
            client,
            file_name=remote_name,
            content=content,
            folder_token=settings.lark_markdown_folder_token,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "app_id": settings.lark_app_id,
                    "folder_token": settings.lark_markdown_folder_token,
                    "token_source": token_source,
                    "file_name": uploaded.file_name,
                    "file_token": uploaded.file_token,
                    "size_bytes": uploaded.size_bytes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "app_id": settings.lark_app_id,
                    "folder_token": settings.lark_markdown_folder_token,
                    "file_name": remote_name,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    finally:
        if client is not None:
            await client.aclose()
        if secrets is not None:
            secrets.close()


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
