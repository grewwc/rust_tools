#!/usr/bin/env python3
"""Fetch the two NextCode secrets through Conciliator's KMS provider.

By default, plaintext is written only after both reads and validations succeed,
and must stay in the git-ignored ``.local`` tree. ``--redacted-only`` performs
the same validation without writing either secret value.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from conciliator.utils.secrets import KmsSecretsProvider

DEFAULT_APP_SECRET_ALIAS = "conciliator/credentials/nextcode_app_secret"
DEFAULT_WEBHOOK_SECRET_ALIAS = "conciliator/credentials/nextcode_webhook_secret"
DEFAULT_OUTPUT = Path(".local/kms/nextcode-secrets.json")
DEFAULT_STATUS_OUTPUT = Path(".local/kms/verification-status.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the NextCode App Secret and Webhook Secret through the "
            "production Conciliator KMS provider. Secret values are never "
            "printed, but the output file contains plaintext."
        )
    )
    parser.add_argument(
        "--app-secret-alias",
        default=os.getenv(
            "KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS",
            DEFAULT_APP_SECRET_ALIAS,
        ),
    )
    parser.add_argument(
        "--webhook-secret-alias",
        default=os.getenv(
            "KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS",
            DEFAULT_WEBHOOK_SECRET_ALIAS,
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"plaintext JSON output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--kms-zone",
        help=(
            "set KMS_ZONE before the SDK is initialized; use cn on a Mac "
            "without Consul region discovery, and omit it in TCE"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file after a successful fetch",
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=DEFAULT_STATUS_OUTPUT,
        help=f"redacted status JSON (default: {DEFAULT_STATUS_OUTPUT})",
    )
    parser.add_argument(
        "--redacted-only",
        action="store_true",
        help="validate both secrets without writing their plaintext values",
    )
    return parser.parse_args()


def write_secret_file(
    output: Path,
    *,
    app_alias: str,
    app_secret: str,
    webhook_alias: str,
    webhook_secret: str,
    overwrite: bool,
) -> None:
    output = output.resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700)

    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")

    payload = {
        "nextcode_app_secret": {
            "alias": app_alias,
            "value": app_secret,
        },
        "nextcode_webhook_secret": {
            "alias": webhook_alias,
            "value": webhook_secret,
        },
    }

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        os.chmod(output, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_status_file(output: Path, payload: dict[str, object]) -> None:
    output = output.resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        os.chmod(output, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    if args.kms_zone:
        os.environ["KMS_ZONE"] = args.kms_zone

    provider: KmsSecretsProvider | None = None
    try:
        provider = KmsSecretsProvider()
        app_secret = provider.get(args.app_secret_alias)
        webhook_secret = provider.get(args.webhook_secret_alias)
        secrets_distinct = app_secret != webhook_secret
        if not args.redacted_only:
            write_secret_file(
                args.output,
                app_alias=args.app_secret_alias,
                app_secret=app_secret,
                webhook_alias=args.webhook_secret_alias,
                webhook_secret=webhook_secret,
                overwrite=args.overwrite,
            )
    except Exception as exc:
        message = " ".join(str(exc).splitlines())
        write_status_file(
            args.status_output,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": message,
                "secret_file_written": False,
                "app_secret_alias": args.app_secret_alias,
                "webhook_secret_alias": args.webhook_secret_alias,
            },
        )
        print(
            f"kms_fetch=failed error_type={type(exc).__name__} error={message}",
            file=sys.stderr,
        )
        print("secret_file_written=false", file=sys.stderr)
        return 1
    finally:
        if provider is not None:
            provider.close()

    output = args.output.resolve()
    write_status_file(
        args.status_output,
        {
            "status": "ok",
            "secret_file_written": not args.redacted_only,
            "secret_file": None if args.redacted_only else str(output),
            "app_secret_alias": args.app_secret_alias,
            "app_secret_nonempty": bool(app_secret),
            "webhook_secret_alias": args.webhook_secret_alias,
            "webhook_secret_nonempty": bool(webhook_secret),
            "secrets_distinct": secrets_distinct,
        },
    )
    print("kms_fetch=ok")
    if args.redacted_only:
        print("secret_file_written=false")
    else:
        print(f"secret_file={output}")
        print("secret_file_mode=0600")
    print(f"app_secret_nonempty={bool(app_secret)}")
    print(f"webhook_secret_nonempty={bool(webhook_secret)}")
    print(f"secrets_distinct={str(secrets_distinct).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
