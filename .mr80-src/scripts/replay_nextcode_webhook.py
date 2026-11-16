"""Replay a captured NextCode webhook using the official header/signature contract.

Example:
  uv run python scripts/replay_nextcode_webhook.py --event merge_request_review \
    --payload tests/fixtures/nextcode/merge_request_review_commented.json
"""

import argparse
import hashlib
import hmac
import json
import sys
import uuid

import httpx

from conciliator.config import Settings
from conciliator.utils.secrets import build_secrets_provider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/hooks/codebase")
    parser.add_argument(
        "--delivery-id",
        help="reuse the same ID to verify persistent webhook deduplication",
    )
    args = parser.parse_args()

    settings = Settings(
        service_psm="nextcode.webhook.replay",
        mysql_psm="unused",
    )
    if not settings.kms_secret_nextcode_webhook_secret_alias:
        print(
            "KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS must be configured",
            file=sys.stderr,
        )
        return 2
    secrets = build_secrets_provider(settings)
    try:
        secret = secrets.get(settings.kms_secret_nextcode_webhook_secret_alias)
    finally:
        secrets.close()

    with open(args.payload, "rb") as payload_file:
        body = payload_file.read()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    delivery = args.delivery_id or str(uuid.uuid4())
    response = httpx.post(
        args.url,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Vecode-Event": args.event,
            "X-Vecode-App-Id": settings.nextcode_app_id,
            "X-Vecode-Event-Id": delivery,
            "X-Vecode-Delivery": delivery,
            "X-Vecode-Signature": signature,
        },
        timeout=30,
    )
    try:
        response_body = response.json()
    except json.JSONDecodeError:
        response_body = response.text
    print(json.dumps({"status": response.status_code, "body": response_body}, ensure_ascii=False))
    return 0 if response.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
