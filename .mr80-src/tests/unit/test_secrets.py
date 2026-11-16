import sys
import traceback
from types import ModuleType

import pytest

from conciliator.utils.secrets import KmsSecretsProvider, StaticSecretsProvider

SCALAR_CREDENTIALS = {
    "NEXTCODE_APP_SECRET": "kms_secret_nextcode_app_secret_alias",
    "NEXTCODE_WEBHOOK_SECRET": "kms_secret_nextcode_webhook_secret_alias",
    "LARK_APP_SECRET": "kms_secret_lark_app_secret_alias",
    "MEEGLE_PLUGIN_SECRET": "kms_secret_meegle_plugin_secret_alias",
    "MEEGLE_WEBHOOK_TOKEN": "kms_secret_meegle_webhook_token_alias",
}


def test_static_provider_returns_value():
    provider = StaticSecretsProvider({"k1": "v1"})
    assert provider.get("k1") == "v1"


def test_static_provider_missing_raises():
    provider = StaticSecretsProvider({})
    with pytest.raises(KeyError):
        provider.get("missing")


@pytest.mark.parametrize("value", ["", "   "])
def test_static_provider_rejects_empty_required_secret(value):
    with pytest.raises(RuntimeError, match="Secret is empty"):
        StaticSecretsProvider({"test/credentials/secret": value}).get("test/credentials/secret")


@pytest.mark.parametrize(("env_name", "alias_field"), SCALAR_CREDENTIALS.items())
def test_dev_credentials_use_static_provider(monkeypatch, env_name, alias_field):
    from conciliator.config import Settings
    from conciliator.utils.secrets import build_secrets_provider

    monkeypatch.setenv(env_name, "dev-test-secret")
    settings = Settings(_env_file=None, env="dev", lark_app_id="cli_test_app")
    assert build_secrets_provider(settings).get(getattr(settings, alias_field)) == "dev-test-secret"


@pytest.mark.parametrize("override", [False, True])
def test_dev_identity_and_credentials_load_from_dotenv(monkeypatch, tmp_path, override):
    from conciliator.config import Settings
    from conciliator.utils.secrets import build_secrets_provider

    configured = {
        "LARK_APP_ID": "cli_test_app",
        "MEEGLE_PLUGIN_ID": "MII_6AAF4159C3C0CCDB",
        **{name: f"file-test-{name.lower()}" for name in SCALAR_CREDENTIALS},
    }
    for env_name in configured:
        monkeypatch.delenv(env_name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ENV=dev\n" + "".join(f"{name}={value}\n" for name, value in configured.items()),
        encoding="utf-8",
    )
    if override:
        configured = {name: f"env-{value}" for name, value in configured.items()}
        for env_name, value in configured.items():
            monkeypatch.setenv(env_name, value)

    settings = Settings(_env_file=env_file, env="dev")
    provider = build_secrets_provider(settings)
    assert settings.lark_app_id == configured["LARK_APP_ID"]
    assert settings.meegle_plugin_id == configured["MEEGLE_PLUGIN_ID"]
    for name, alias_field in SCALAR_CREDENTIALS.items():
        assert provider.get(getattr(settings, alias_field)) == configured[name]
        assert name.lower() not in settings.model_dump()
        for output in (repr(settings), str(settings.model_dump()), settings.model_dump_json()):
            assert configured[name] not in output


@pytest.mark.parametrize("env", ["staging", "prod"])
def test_deployed_credentials_ignore_local_values(monkeypatch, env):
    from conciliator.config import Settings
    from conciliator.utils.secrets import build_secrets_provider

    class FakeKmsClient:
        def get_secret(self, alias):
            return f"kms-test-value:{alias}"

        def close(self):
            pass

    kmsv2_package = ModuleType("kmsv2")
    kmsv2_package.KmsClient = FakeKmsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kmsv2", kmsv2_package)
    for name in SCALAR_CREDENTIALS:
        monkeypatch.setenv(name, "must-not-be-used")
    settings = Settings(_env_file=None, env=env)
    provider = build_secrets_provider(settings)
    try:
        for field in SCALAR_CREDENTIALS.values():
            alias = getattr(settings, field)
            assert provider.get(alias) == f"kms-test-value:{alias}"
    finally:
        provider.close()


@pytest.mark.parametrize("env", ["staging", "prod"])
def test_deployed_bytecloud_pair_uses_kms_and_ignores_local_values(monkeypatch, env):
    from conciliator.config import Settings
    from conciliator.utils.secrets import build_secrets_provider, get_bytecloud_credentials

    calls = []
    class FakeKmsClient:
        def get_secret(self, alias):
            calls.append(alias)
            assert alias == "test/credentials/bytecloud_credentials"
            return '{"ak":"kms-test-ak","sk":"kms-test-sk"}'

        def close(self):
            pass

    kmsv2_package = ModuleType("kmsv2")
    kmsv2_package.KmsClient = FakeKmsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kmsv2", kmsv2_package)
    monkeypatch.setenv("BYTECLOUD_AK", "must-not-be-used")
    monkeypatch.setenv("BYTECLOUD_SK", "must-not-be-used")
    settings = Settings(
        _env_file=None, env=env,
        kms_secret_bytecloud_credentials_alias="test/credentials/bytecloud_credentials",
    )
    provider = build_secrets_provider(settings)
    try:
        for _ in range(2):
            credentials = get_bytecloud_credentials(settings, provider)
            assert credentials.ak.get_secret_value() == "kms-test-ak"
            assert credentials.sk.get_secret_value() == "kms-test-sk"
        assert calls == ["test/credentials/bytecloud_credentials"]
    finally:
        provider.close()


@pytest.mark.parametrize("override", [False, True])
def test_dev_bytecloud_pair_loads_from_dotenv(monkeypatch, tmp_path, override):
    from conciliator.config import Settings
    from conciliator.utils.secrets import build_secrets_provider, get_bytecloud_credentials

    configured = {"BYTECLOUD_AK": "file-test-ak", "BYTECLOUD_SK": "file-test-sk"}
    for env_name in configured:
        monkeypatch.delenv(env_name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "".join(f"{name}={value}\n" for name, value in configured.items()), encoding="utf-8",
    )
    if override:
        configured = {name: f"env-{value}" for name, value in configured.items()}
        for env_name, value in configured.items():
            monkeypatch.setenv(env_name, value)
    settings = Settings(_env_file=env_file, env="dev", lark_app_id="cli_test_app")
    credentials = get_bytecloud_credentials(settings, build_secrets_provider(settings))
    assert credentials.ak.get_secret_value() == configured["BYTECLOUD_AK"]
    assert credentials.sk.get_secret_value() == configured["BYTECLOUD_SK"]
    for output in (
        repr(settings), settings.model_dump_json(), str(settings.model_dump()),
        repr(credentials), credentials.model_dump_json(),
    ):
        assert configured["BYTECLOUD_AK"] not in output
        assert configured["BYTECLOUD_SK"] not in output


def test_removed_dev_prefixes_are_not_loaded(monkeypatch):
    from conciliator.config import Settings

    for name in (*SCALAR_CREDENTIALS, "BYTECLOUD_AK", "BYTECLOUD_SK"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(f"_DEV_{name}", "obsolete-test-secret")
    settings = Settings(_env_file=None, env="dev", lark_app_id="cli_test_app")
    for name in (*SCALAR_CREDENTIALS, "BYTECLOUD_AK", "BYTECLOUD_SK"):
        assert getattr(settings, name.lower()).get_secret_value() == ""


@pytest.mark.parametrize("value", [
    "old-test-secret", "{broken-test-secret", "null", "[]",
    '{"ak":"test-sensitive-ak"}', '{"sk":"test-sensitive-sk"}',
    '{"ak":"","sk":"test-sensitive-sk"}', '{"ak":"test-sensitive-ak","sk":" "}',
    '{"ak":" test-sensitive-ak","sk":"test-sensitive-sk"}',
    '{"ak":123,"sk":"test-sensitive-sk"}', '{"ak":"test-sensitive-ak","sk":null}',
    '{"ak":"test-sensitive-ak","sk":"test-sensitive-sk","extra":"test-sensitive-extra"}',
])
def test_bytecloud_pair_rejects_invalid_json_without_leaking_values(value):
    from conciliator.config import Settings
    from conciliator.utils.secrets import get_bytecloud_credentials

    settings = Settings(_env_file=None)
    provider = StaticSecretsProvider({settings.kms_secret_bytecloud_credentials_alias: value})
    with pytest.raises(RuntimeError, match="nonempty string ak and sk") as caught:
        get_bytecloud_credentials(settings, provider)
    output = "".join(traceback.format_exception(caught.value))
    assert value not in output
    assert "test-sensitive" not in output
    assert "broken-test-secret" not in output


def test_kms_provider_uses_v2_client_and_caches(monkeypatch):
    calls: list[tuple[str, str] | tuple[str]] = []

    class FakeKmsClient:
        def __init__(self):
            calls.append(("init",))

        def get_secret(self, alias: str) -> str:
            calls.append(("get", alias))
            return f"value:{alias}"

        def close(self) -> None:
            calls.append(("close",))

    kmsv2_package = ModuleType("kmsv2")
    kmsv2_package.KmsClient = FakeKmsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kmsv2", kmsv2_package)

    provider = KmsSecretsProvider()

    alias = "test/credentials/secret"
    assert provider.get(alias) == f"value:{alias}"
    assert provider.get(alias) == f"value:{alias}"
    provider.close()
    assert calls == [("init",), ("get", alias), ("close",)]


@pytest.mark.parametrize(
    "alias",
    ["secret", "key/secret", "keyring/key/", "/key/secret", "a/b/c/d"],
)
def test_kms_provider_rejects_non_full_alias(monkeypatch, alias):
    class FakeKmsClient:
        def get_secret(self, secret_alias: str) -> str:
            raise AssertionError(f"must not fetch invalid alias: {secret_alias}")

        def close(self) -> None:
            return None

    kmsv2_package = ModuleType("kmsv2")
    kmsv2_package.KmsClient = FakeKmsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kmsv2", kmsv2_package)

    provider = KmsSecretsProvider()
    with pytest.raises(ValueError, match="keyring_alias/key_alias/secret_alias"):
        provider.get(alias)
