from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Final, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 评论指令超级管理员；修改名单后随服务版本部署。
COMMENT_SUPER_ADMIN_EMAILS: Final[tuple[str, ...]] = (
    "cuijiawei.2026@bytedance.com",
    "huaguojin@bytedance.com",
)

SERVICE_PSM_DEFAULT: Final[str] = "data.aeolus.conciliator"
NEXTCODE_APP_SECRET_ALIAS_DEFAULT: Final[str] = (
    "conciliator/credentials/nextcode_app_secret"
)
NEXTCODE_WEBHOOK_SECRET_ALIAS_DEFAULT: Final[str] = (
    "conciliator/credentials/nextcode_webhook_secret"
)
COCO_WORKSPACE_ID_DEFAULT: Final[str] = "793550816067727"
COCO_DEFAULT_AGENT_ID_DEFAULT: Final[str] = "793550816045691"
LARK_APP_ID_DEFAULT: Final[str] = "cli_aa1d2aa842385bee"
LARK_OAUTH_REDIRECT_URI_DEFAULT: Final[str] = (
    "https://aeolus-conciliator.byted.org/oauth/lark/callback"
)
LARK_APP_SECRET_ALIAS_DEFAULT: Final[str] = (
    "conciliator/credentials/lark_app_secret"
)
LARK_MARKDOWN_FOLDER_TOKEN_DEFAULT: Final[str] = (
    "K0zefVjqUlsicAdu4TjccNhjnjh"
)
MEEGLE_SITE_URL_DEFAULT: Final[str] = "https://meego.larkoffice.com"
MEEGLE_PLUGIN_ID_DEFAULT: Final[str] = "MII_6AED80FF53004C14"
MEEGLE_PLUGIN_SECRET_ALIAS_DEFAULT: Final[str] = (
    "conciliator/credentials/meegle_plugin_secret"
)
MEEGLE_WEBHOOK_TOKEN_ALIAS_DEFAULT: Final[str] = (
    "conciliator/credentials/meegle_webhook_token"
)
MYSQL_PSM_DEFAULT: Final[str] = "toutiao.mysql.aeolus_govern_write"
MYSQL_DB_NAME_DEFAULT: Final[str] = "aeolus_govern"


class Settings(BaseSettings):
    """稳定业务值随代码维护；环境变量仅需覆盖环境差异和运行参数。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── 服务 ──
    env: Literal["dev", "staging", "prod"] = "prod"
    log_level: str = "INFO"
    # 生产镜像安装 bytedlogger 后自动接入 Argos StreamLog；本地无 SDK 时仅输出 stdout。
    argos_log_enabled: bool = True
    trace_viewers: dict[str, SecretStr] = Field(default_factory=dict)
    service_psm: str = SERVICE_PSM_DEFAULT
    aeolus_coding_agent_id: str = "800938312911307"
    aeolus_coding_agents: dict[str, str] = Field(
        default_factory=lambda: {"dp/VizQueryService": "803325104638816"}
    )
    # 凭据库 ID（逗号分隔），只下发到 Aeolus 的评估与编码会话。平台把库内凭据
    # 注入沙箱，CLI 启动即可用，不必把 token 写进 prompt。空值表示不绑定。
    aeolus_credential_vault_ids: str = "803299439698129"
    # 允许通过 MR 评论触发重新编码的邮箱白名单（逗号分隔）。任务发起人
    # 始终允许；白名单用于额外放行，默认空。
    aeolus_comment_trigger_emails: str = ""
    # ByteCloud AK/SK 成对轮转；部署环境整对从 KMS 获取。
    bytecloud_ak: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    bytecloud_sk: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    # 固定部署配置随代码发布。
    maat_action_user: ClassVar[str] = "huaguojin"
    maat_baseline_environment: ClassVar[str] = "xiaorui-aeolus-hua"
    maat_baseline_file: ClassVar[Path] = Path(__file__).parent / "integrations/maat/baseline.json"
    maat_baseline_revision: ClassVar[str] = "e12872174562186a924fdbda8bb77156e757aca5cc2430aeb4511ffb71624ed2"
    maat_auth_proxy: ClassVar[str] = "http://sys-proxy-rd-relay.byted.org:8118"
    aeolus_lark_callback_mode: Literal["http", "websocket"] = "websocket"
    lark_card_verification_token: SecretStr = SecretStr("")
    lark_card_encrypt_key: SecretStr = SecretStr("")

    # ── KMS ──
    # 完整格式均为 keyring_alias/key_alias/secret_alias；这里只保存定位信息。
    kms_secret_nextcode_app_secret_alias: str = (
        NEXTCODE_APP_SECRET_ALIAS_DEFAULT
    )
    kms_secret_nextcode_webhook_secret_alias: str = (
        NEXTCODE_WEBHOOK_SECRET_ALIAS_DEFAULT
    )
    kms_secret_lark_app_secret_alias: str = LARK_APP_SECRET_ALIAS_DEFAULT
    kms_secret_meegle_plugin_secret_alias: str = (
        MEEGLE_PLUGIN_SECRET_ALIAS_DEFAULT
    )
    kms_secret_meegle_webhook_token_alias: str = (
        MEEGLE_WEBHOOK_TOKEN_ALIAS_DEFAULT
    )
    kms_secret_bytecloud_credentials_alias: str = (
        "conciliator/credentials/bytecloud_credentials"
    )

    # ── NextCode ──
    nextcode_app_id: str = "792763146702616"
    # 凭据仅 ENV=dev 使用环境配置；staging/prod 从 KMS 读取。
    nextcode_app_secret: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    nextcode_webhook_secret: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    nextcode_api_base: str = "https://codebase-api.byted.org/v2/"
    nextcode_zti_required: bool = True

    # ── Coco ──
    # Coco 与 NextCode 共用 Codebase v2 Endpoint。平台服务账号使用 App headers；
    # 普通用户使用 X-Code-User-JWT，只有代目标用户调用时才需要 SignUserJWT。
    coco_api_base: str = "https://codebase-api.byted.org/v2/"
    coco_workspace_id: str = COCO_WORKSPACE_ID_DEFAULT
    coco_default_agent_id: str = COCO_DEFAULT_AGENT_ID_DEFAULT
    # Read timeout for the Coco event stream.  It is not a run lifetime: a run
    # lives until it reports a result or stops producing events.
    coco_task_timeout_seconds: int = 900
    # How long a run may go without a new event on its own event stream before
    # it is treated as stalled.  A tool call or a subagent wait emits nothing on
    # the parent run, and token queueing emits nothing at all, so this window
    # has to absorb those silences; it is renewed by every new event.
    coco_stall_timeout_seconds: int = 1800
    coco_interrupt_timeout_seconds: int = 30
    # Backoff before replacing a run that the Coco platform itself aborted.
    coco_platform_retry_delay_seconds: int = 120
    # How to answer a tool call the runtime is holding for user confirmation.
    # "deny" keeps the run alive with a corrective message; "fail" reports the
    # pause instead.  Nobody is available to confirm in an automated run.
    coco_tool_confirmation_policy: str = "deny"
    coco_max_tool_confirmations: int = 3

    # ── 飞书 ──
    lark_app_id: str = LARK_APP_ID_DEFAULT
    lark_app_secret: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    # OAuth callback for reading documents as the operator; empty disables it.
    lark_oauth_redirect_uri: str = LARK_OAUTH_REDIRECT_URI_DEFAULT
    lark_markdown_folder_token: str = LARK_MARKDOWN_FOLDER_TOKEN_DEFAULT

    # ── Meegle / Aeolus AI 节点 ──
    meegle_site_url: str = MEEGLE_SITE_URL_DEFAULT
    meegle_plugin_id: str = MEEGLE_PLUGIN_ID_DEFAULT
    # 本地测试凭据随 Settings 读取 .env；仅 ENV=dev 使用，不参与配置输出。
    meegle_plugin_secret: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    meegle_webhook_token: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)

    # ── MySQL(bytedmysql + db_psm)──
    mysql_psm: str = MYSQL_PSM_DEFAULT
    mysql_db_name: str = MYSQL_DB_NAME_DEFAULT
    mysql_pool_size: int = 10
    mysql_pool_max_overflow: int = 20

    # ── 运行时 ──
    worker_concurrency: int = 4
    worker_poll_interval_seconds: int = 5
    lark_notification_worker_concurrency: int = 1
    lark_notification_poll_interval_seconds: int = 5
    lark_notification_max_attempts: int = 8
    lark_notification_retry_base_seconds: int = 5
    lark_notification_retry_max_seconds: int = 300
    lark_notification_sending_stale_seconds: int = 300

    @field_validator("aeolus_coding_agents")
    @classmethod
    def validate_coding_agents(cls, agents: dict[str, str]) -> dict[str, str]:
        if any(
            not repository.strip() or repository != repository.strip()
            or not agent.strip() or agent != agent.strip()
            for repository, agent in agents.items()
        ):
            raise ValueError(
                "Coding repository paths and agent IDs must be nonempty and have no surrounding whitespace"
            )
        return agents

    @model_validator(mode="after")
    def validate_integration_settings(self) -> "Settings":
        """Validate the complete service; credentials are checked during startup."""
        if (
            self.env == "dev"
            and self.aeolus_lark_callback_mode == "websocket"
            and self.lark_app_id == LARK_APP_ID_DEFAULT
        ):
            raise ValueError(
                "ENV=dev WebSocket callbacks require a separate test LARK_APP_ID; "
                "a local instance could consume production callbacks"
            )
        if (
            self.aeolus_lark_callback_mode == "http"
            and not self.lark_card_verification_token.get_secret_value().strip()
        ):
            raise ValueError("LARK_CARD_VERIFICATION_TOKEN is required")
        required_nextcode = {
            "KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS": (
                self.kms_secret_nextcode_app_secret_alias
            ),
            "KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS": (
                self.kms_secret_nextcode_webhook_secret_alias
            ),
            "NEXTCODE_APP_ID": self.nextcode_app_id,
        }
        missing = [
            name for name, value in required_nextcode.items() if not value.strip()
        ]
        if missing:
            raise ValueError(
                "missing NextCode settings: " + ", ".join(missing)
            )

        if (
            self.kms_secret_nextcode_app_secret_alias
            == self.kms_secret_nextcode_webhook_secret_alias
        ):
            raise ValueError(
                "NextCode App Secret and Webhook Secret must use different "
                "KMS aliases"
            )

        required_shared_ai = {
            "KMS_SECRET_LARK_APP_SECRET_ALIAS": (
                self.kms_secret_lark_app_secret_alias
            ),
            "COCO_WORKSPACE_ID": self.coco_workspace_id,
            "COCO_DEFAULT_AGENT_ID": self.coco_default_agent_id,
            "LARK_APP_ID": self.lark_app_id,
        }
        required_pipeline = {
            **required_shared_ai,
            "LARK_MARKDOWN_FOLDER_TOKEN": self.lark_markdown_folder_token,
        }
        missing = [
            name for name, value in required_pipeline.items() if not value.strip()
        ]
        if missing:
            raise ValueError("missing pipeline settings: " + ", ".join(missing))

        required_aeolus = {
            **required_shared_ai,
            "AEOLUS_CODING_AGENT_ID": self.aeolus_coding_agent_id,
            "MEEGLE_SITE_URL": self.meegle_site_url,
            "MEEGLE_PLUGIN_ID": self.meegle_plugin_id,
            "KMS_SECRET_MEEGLE_PLUGIN_SECRET_ALIAS": (
                self.kms_secret_meegle_plugin_secret_alias
            ),
            "KMS_SECRET_MEEGLE_WEBHOOK_TOKEN_ALIAS": (
                self.kms_secret_meegle_webhook_token_alias
            ),
        }
        missing = [
            name for name, value in required_aeolus.items() if not value.strip()
        ]
        if missing:
            raise ValueError("missing Aeolus AI node settings: " + ", ".join(missing))

        secret_aliases = {
            name: value
            for name, value in {
                **required_nextcode, **required_pipeline, **required_aeolus,
                "KMS_SECRET_BYTECLOUD_CREDENTIALS_ALIAS": (
                    self.kms_secret_bytecloud_credentials_alias
                ),
            }.items()
            if name.startswith("KMS_SECRET_")
        }
        invalid_aliases = [
            name
            for name, value in secret_aliases.items()
            if len(value.split("/")) != 3
            or any(not part or part != part.strip() for part in value.split("/"))
        ]
        if invalid_aliases:
            raise ValueError(
                "KMS secret aliases must use keyring/key/secret format: "
                + ", ".join(invalid_aliases)
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
