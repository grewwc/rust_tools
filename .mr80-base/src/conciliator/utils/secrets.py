import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

from conciliator.config import Settings


class ByteCloudCredentials(BaseModel):
    """一个启动快照中的 ByteCloud AK/SK，不在错误或序列化输出中暴露值。"""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    ak: SecretStr
    sk: SecretStr

    @field_validator("ak", "sk")
    @classmethod
    def validate_credential(cls, value: SecretStr) -> SecretStr:
        text = value.get_secret_value()
        if not text.strip() or text != text.strip():
            raise ValueError("credential must be nonempty without surrounding whitespace")
        return value


class SecretsProvider(Protocol):
    """敏感配置来源的抽象。"""

    def get(self, secret_alias: str) -> str: ...

    def close(self) -> None: ...


def get_bytecloud_credentials(
    settings: Settings, secrets: SecretsProvider,
) -> ByteCloudCredentials:
    value = secrets.get(settings.kms_secret_bytecloud_credentials_alias)
    try:
        return ByteCloudCredentials.model_validate_json(value)
    except ValidationError:
        raise RuntimeError(
            "ByteCloud credentials must be JSON with nonempty string ak and sk; "
            "configure KMS or BYTECLOUD_AK/BYTECLOUD_SK for ENV=dev"
        ) from None


class KmsSecretsProvider:
    """生产用:走 bytedkmsv2。

    启动时一次性拉取并缓存。配置变更需重启或主动 refresh。
    """

    def __init__(self) -> None:
        from kmsv2 import KmsClient  # 延迟 import,本地测试不需要装 bytedkmsv2

        # TCE 运行时身份由平台 ZTI 提供，KmsClient 不接收 PSM 参数。
        self._client = KmsClient()
        self._cache: dict[str, str] = {}

    def get(self, secret_alias: str) -> str:
        _validate_secret_alias(secret_alias)
        if secret_alias not in self._cache:
            value = self._client.get_secret(secret_alias)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    f"KMS secret is empty or not text: {secret_alias}"
                )
            self._cache[secret_alias] = value
        return self._cache[secret_alias]

    def close(self) -> None:
        self._client.close()


class StaticSecretsProvider:
    """测试/本地开发用:从 dict 直接取。"""

    def __init__(self, values: dict[str, str]):
        self._values = values

    def get(self, secret_alias: str) -> str:
        value = self._values[secret_alias]
        if not value.strip():
            raise RuntimeError(f"Secret is empty: {secret_alias}")
        return value

    def close(self) -> None:
        return None


def _validate_secret_alias(secret_alias: str) -> None:
    parts = secret_alias.split("/")
    if (
        len(parts) != 3
        or any(not part for part in parts)
        or any(part != part.strip() for part in parts)
    ):
        raise ValueError(
            "KMS secret alias must be keyring_alias/key_alias/secret_alias"
        )


def build_secrets_provider(settings: Settings) -> SecretsProvider:
    """ENV=dev 使用环境配置，staging/prod 使用 KMS。"""
    if settings.env == "dev":
        values = {
            settings.kms_secret_nextcode_app_secret_alias: settings.nextcode_app_secret.get_secret_value(),
            settings.kms_secret_nextcode_webhook_secret_alias: settings.nextcode_webhook_secret.get_secret_value(),
            settings.kms_secret_lark_app_secret_alias: settings.lark_app_secret.get_secret_value(),
            settings.kms_secret_meegle_plugin_secret_alias: settings.meegle_plugin_secret.get_secret_value(),
            settings.kms_secret_meegle_webhook_token_alias: settings.meegle_webhook_token.get_secret_value(),
            settings.kms_secret_bytecloud_credentials_alias: json.dumps({
                "ak": settings.bytecloud_ak.get_secret_value(),
                "sk": settings.bytecloud_sk.get_secret_value(),
            }),
        }
        return StaticSecretsProvider(values)
    return KmsSecretsProvider()
