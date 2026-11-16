import pytest
from pydantic import ValidationError

from conciliator.config import LARK_APP_ID_DEFAULT, Settings, get_settings

REMOVED_SWITCHES = (
    "INTEGRATIONS_ENABLED", "NEXTCODE_ENABLED", "PIPELINE_ENABLED",
    "AEOLUS_AI_NODE_ENABLED", "AEOLUS_DEPLOYMENT_ENABLED",
    "AEOLUS_PPE_DEPLOYMENT_ENABLED", "AEOLUS_LARK_NOTIFICATIONS_ENABLED",
)


def test_stable_defaults_ship_with_service(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in Settings.model_fields:
        monkeypatch.delenv(key.upper(), raising=False)
    settings = Settings()
    assert settings.service_psm == "data.aeolus.conciliator"
    assert settings.bytecloud_ak.get_secret_value() == ""
    assert settings.mysql_psm == "toutiao.mysql.aeolus_govern_write"
    assert settings.mysql_db_name == "aeolus_govern"
    assert settings.nextcode_app_id == "792763146702616"
    assert settings.coco_workspace_id == "793550816067727"
    assert settings.coco_default_agent_id == "793550816045691"
    assert settings.coco_task_timeout_seconds == 900
    assert settings.coco_stall_timeout_seconds == 1800
    assert settings.coco_interrupt_timeout_seconds == 30
    assert settings.coco_platform_retry_delay_seconds == 120
    assert settings.coco_tool_confirmation_policy == "deny"
    assert settings.coco_max_tool_confirmations == 3
    assert settings.aeolus_coding_agent_id == "800938312911307"
    assert settings.aeolus_credential_vault_ids == "803299439698129"
    assert settings.lark_app_id == "cli_aa1d2aa842385bee"
    assert settings.lark_oauth_redirect_uri == "https://aeolus-conciliator.byted.org/oauth/lark/callback"
    assert settings.meegle_plugin_id == "MII_6AED80FF53004C14"
    assert settings.kms_secret_meegle_webhook_token_alias == "conciliator/credentials/meegle_webhook_token"
    assert settings.kms_secret_bytecloud_credentials_alias == "conciliator/credentials/bytecloud_credentials"


def test_obsolete_environment_switches_cannot_bypass_validation(monkeypatch):
    for key in REMOVED_SWITCHES:
        monkeypatch.setenv(key, "false")
        assert key.lower() not in Settings.model_fields
    with pytest.raises(ValidationError, match="missing Aeolus AI node settings"):
        Settings(_env_file=None, meegle_plugin_id="")
    assert not hasattr(Settings(_env_file=None), "maat_deployment_enabled")


def test_environment_can_override_defaults_and_runtime_parameters(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "ENV": "staging", "SERVICE_PSM": "test.conciliator",
        "MYSQL_PSM": "test.mysql", "MYSQL_DB_NAME": "test_db",
        "NEXTCODE_APP_ID": "test-app", "WORKER_CONCURRENCY": "2",
        "LARK_OAUTH_REDIRECT_URI": "https://test.example/oauth/lark/callback",
        "KMS_SECRET_BYTECLOUD_CREDENTIALS_ALIAS": "test/credentials/bytecloud_credentials",
        "KMS_SECRET_MEEGLE_WEBHOOK_TOKEN_ALIAS": "test/credentials/meegle_webhook_token",
        "MEEGLE_PLUGIN_ID": "MII_6AAF4159C3C0CCDB",
        "COCO_STALL_TIMEOUT_SECONDS": "600",
        "COCO_INTERRUPT_TIMEOUT_SECONDS": "10",
        "COCO_PLATFORM_RETRY_DELAY_SECONDS": "60",
        "COCO_TOOL_CONFIRMATION_POLICY": "fail",
        "COCO_MAX_TOOL_CONFIRMATIONS": "5",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.env == "staging"
        assert settings.service_psm == "test.conciliator"
        assert settings.mysql_psm == "test.mysql"
        assert settings.mysql_db_name == "test_db"
        assert settings.nextcode_app_id == "test-app"
        assert settings.lark_oauth_redirect_uri == "https://test.example/oauth/lark/callback"
        assert settings.worker_concurrency == 2
        assert settings.kms_secret_bytecloud_credentials_alias == "test/credentials/bytecloud_credentials"
        assert settings.kms_secret_meegle_webhook_token_alias == "test/credentials/meegle_webhook_token"
        assert settings.meegle_plugin_id == "MII_6AAF4159C3C0CCDB"
        assert settings.coco_stall_timeout_seconds == 600
        assert settings.coco_interrupt_timeout_seconds == 10
        assert settings.coco_platform_retry_delay_seconds == 60
        assert settings.coco_tool_confirmation_policy == "fail"
        assert settings.coco_max_tool_confirmations == 5
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(("field", "message"), [
    ("nextcode_app_id", "missing NextCode settings"),
    ("kms_secret_nextcode_app_secret_alias", "missing NextCode settings"),
    ("kms_secret_lark_app_secret_alias", "missing pipeline settings"),
    ("lark_markdown_folder_token", "missing pipeline settings"),
    ("meegle_plugin_id", "missing Aeolus AI node settings"),
    ("kms_secret_meegle_webhook_token_alias", "missing Aeolus AI node settings"),
    ("kms_secret_bytecloud_credentials_alias", "KMS secret aliases"),
])
def test_required_settings_cannot_be_cleared(field, message):
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None, **{field: ""})


@pytest.mark.parametrize("field", [
    "kms_secret_nextcode_app_secret_alias", "kms_secret_nextcode_webhook_secret_alias",
    "kms_secret_lark_app_secret_alias", "kms_secret_meegle_plugin_secret_alias",
    "kms_secret_meegle_webhook_token_alias", "kms_secret_bytecloud_credentials_alias",
])
def test_all_secret_aliases_require_full_format(field):
    with pytest.raises(ValidationError, match="KMS secret aliases"):
        Settings(_env_file=None, **{field: "invalid"})


def test_nextcode_app_and_webhook_secrets_must_be_distinct():
    with pytest.raises(ValidationError, match="must use different KMS aliases"):
        Settings(
            _env_file=None, kms_secret_nextcode_app_secret_alias="test/credentials/shared",
            kms_secret_nextcode_webhook_secret_alias="test/credentials/shared",
        )


def test_maat_template_and_operator_are_code_managed(monkeypatch):
    monkeypatch.setenv("MAAT_BASELINE_ENVIRONMENT", "other-environment")
    monkeypatch.setenv("MAAT_BASELINE_FILE", "/missing/other.json")
    monkeypatch.setenv("MAAT_ACTION_USER", "other-operator")
    settings = Settings(_env_file=None)
    assert settings.maat_baseline_environment == "xiaorui-aeolus-hua"
    assert settings.maat_baseline_file.is_file()
    assert settings.maat_action_user == "huaguojin"


def test_coding_agent_repository_defaults(monkeypatch):
    monkeypatch.delenv("AEOLUS_CODING_AGENTS", raising=False)
    settings = Settings(_env_file=None, aeolus_coding_agent_id="existing-aeolus-agent")
    assert settings.aeolus_coding_agents == {"dp/VizQueryService": "803325104638816"}
    assert settings.aeolus_coding_agent_id == "existing-aeolus-agent"


def test_coding_agents_can_be_extended_from_env(monkeypatch):
    monkeypatch.setenv("AEOLUS_CODING_AGENTS", (
        '{"dp/VizQueryService":"803325104638816","dp/another":"another-agent"}'
    ))
    assert Settings(_env_file=None).aeolus_coding_agents == {
        "dp/VizQueryService": "803325104638816", "dp/another": "another-agent",
    }


@pytest.mark.parametrize("agents", [
    {"dp/VizQueryService": ""}, {"dp/VizQueryService": " agent "},
    {"": "agent"}, {" dp/repo": "agent"},
])
def test_coding_agent_mapping_rejects_blank_or_padded_entries(agents):
    with pytest.raises(ValidationError, match="nonempty and have no surrounding whitespace"):
        Settings(_env_file=None, aeolus_coding_agents=agents)


def test_local_websocket_rejects_production_app():
    with pytest.raises(ValidationError, match="separate test LARK_APP_ID"):
        Settings(
            _env_file=None, env="dev", aeolus_lark_callback_mode="websocket",
            lark_app_id=LARK_APP_ID_DEFAULT,
        )


def test_local_websocket_accepts_test_app():
    settings = Settings(
        _env_file=None, env="dev", aeolus_lark_callback_mode="websocket",
        lark_app_id="cli_test_app",
    )
    assert settings.lark_app_id == "cli_test_app"


def test_local_http_does_not_consume_production_websocket_callbacks():
    settings = Settings(
        _env_file=None, env="dev", aeolus_lark_callback_mode="http",
        lark_app_id=LARK_APP_ID_DEFAULT, lark_card_verification_token="test-verification",
    )
    assert settings.aeolus_lark_callback_mode == "http"
