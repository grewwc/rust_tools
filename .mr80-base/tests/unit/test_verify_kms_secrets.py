import json
import sys

import pytest

from scripts import verify_kms_secrets


@pytest.mark.parametrize(
    ("app_secret", "webhook_secret", "expected_distinct"),
    [
        ("shared-secret", "shared-secret", False),
        ("app-secret", "webhook-secret", True),
    ],
)
def test_redacted_smoke_accepts_readable_nonempty_secrets(
    monkeypatch,
    tmp_path,
    capsys,
    app_secret,
    webhook_secret,
    expected_distinct,
):
    class FakeKmsSecretsProvider:
        def get(self, alias: str) -> str:
            if alias == verify_kms_secrets.DEFAULT_APP_SECRET_ALIAS:
                return app_secret
            if alias == verify_kms_secrets.DEFAULT_WEBHOOK_SECRET_ALIAS:
                return webhook_secret
            raise KeyError(alias)

        def close(self) -> None:
            return None

    status_output = tmp_path / "status.json"
    monkeypatch.setattr(
        verify_kms_secrets,
        "KmsSecretsProvider",
        FakeKmsSecretsProvider,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_kms_secrets.py",
            "--redacted-only",
            "--status-output",
            str(status_output),
        ],
    )

    assert verify_kms_secrets.main() == 0

    status = json.loads(status_output.read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    assert status["secret_file_written"] is False
    assert status["app_secret_nonempty"] is True
    assert status["webhook_secret_nonempty"] is True
    assert status["secrets_distinct"] is expected_distinct
    assert (
        f"secrets_distinct={str(expected_distinct).lower()}"
        in capsys.readouterr().out
    )
