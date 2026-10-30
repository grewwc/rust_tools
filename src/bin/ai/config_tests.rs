use std::env;

use super::load_config;
use crate::ai::{models, test_support::ENV_LOCK};
use crate::commonw::configw;

#[test]
fn load_config_accepts_default_model_specific_api_key_config_key() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());

    // Discover a Volcengine entry by its endpoint/adapter properties instead of
    // pinning a concrete key: the ark endpoint with the volcano api-key slot.
    let model_key = crate::ai::model_names::all()
        .iter()
        .find(|m| {
            m.endpoint
                .as_deref()
                .unwrap_or_default()
                .contains("volces.com")
                && m.api_key_config_key.as_deref() == Some("volcano.api_key")
        })
        .map(|m| m.key.clone())
        .expect("model registry must contain a volcano-endpoint model using volcano.api_key");

    let cfg_path = std::env::temp_dir().join(format!(
        "rt_volcano_model_{}.configw",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(
        &cfg_path,
        format!("ai.model.default = {model_key}\nvolcano.api_key = test-volcano-key\n"),
    )
    .unwrap();
    let old_cfg = env::var_os("CONFIGW_PATH");
    unsafe { env::set_var("CONFIGW_PATH", &cfg_path) };
    configw::refresh();

    let loaded = load_config();
    let resolved_key = match loaded.as_ref() {
        Ok(app) => Some(models::api_key_for_model(&model_key, &app.api_key)),
        Err(_) => None,
    };

    match old_cfg {
        Some(value) => unsafe { env::set_var("CONFIGW_PATH", value) },
        None => unsafe { env::remove_var("CONFIGW_PATH") },
    }
    configw::refresh();
    let _ = std::fs::remove_file(&cfg_path);

    loaded.expect("default model specific api key should pass startup validation");
    assert_eq!(resolved_key.as_deref(), Some("test-volcano-key"));
}

#[test]
fn history_max_chars_is_fixed_and_window_independent_unless_configured() {
    let model_with_window = |predicate: fn(usize) -> bool| {
        crate::ai::model_names::all()
            .into_iter()
            .find(|entry| predicate(models::context_window_tokens(&entry.key)))
            .map(|entry| entry.key.clone())
    };
    let small = model_with_window(|window| window <= 128_000)
        .expect("registry must contain a <=128K-token model");
    let large = model_with_window(|window| window >= 1_000_000)
        .expect("registry must contain a >=1M-token model");

    let unset = configw::ConfigW::parse("");
    // The budget no longer scales with the model window: a 1M-token model gets
    // exactly the same fixed default as a 128K one, so a large declared window
    // cannot inflate resident history and per-request tokens stay stable.
    assert_eq!(
        super::history_max_chars_for_model(&unset, &small),
        super::DEFAULT_HISTORY_MAX_CHARS
    );
    assert_eq!(
        super::history_max_chars_for_model(&unset, &large),
        super::DEFAULT_HISTORY_MAX_CHARS
    );

    let explicit = configw::ConfigW::parse("ai.history.max_chars = 90000\n");
    assert_eq!(super::history_max_chars_for_model(&explicit, &small), 90_000);
    assert_eq!(super::history_max_chars_for_model(&explicit, &large), 90_000);
    let disabled = configw::ConfigW::parse("ai.history.max_chars = 0\n");
    assert_eq!(super::history_max_chars_for_model(&disabled, &large), 0);
}

#[test]
fn pure_startup_default_follows_the_config_key() {
    let unset = configw::ConfigW::parse("");
    let enabled = configw::ConfigW::parse("ai.prompt.pure_mode = true\n");
    let disabled = configw::ConfigW::parse("ai.prompt.pure_mode = false\n");

    // Follow the persisted key, defaulting to off. Explicit agent selection
    // (`--agent`, `--pure`) and `--no-pure` are applied by the caller.
    assert!(!super::pure_startup_default(&unset));
    assert!(super::pure_startup_default(&enabled));
    assert!(!super::pure_startup_default(&disabled));
}

#[test]
fn pure_startup_default_accepts_common_boolean_spellings_and_ignores_invalid_values() {
    for raw in ["1", "true", "TRUE", "yes", "on", "enabled"] {
        let cfg = configw::ConfigW::parse(&format!("ai.prompt.pure_mode = {raw}\n"));
        assert!(super::pure_startup_default(&cfg), "expected on for {raw}");
    }
    for raw in ["0", "false", "FALSE", "no", "off", "disabled"] {
        let cfg = configw::ConfigW::parse(&format!("ai.prompt.pure_mode = {raw}\n"));
        assert!(!super::pure_startup_default(&cfg), "expected off for {raw}");
    }
    // A typo must not strip the normal system prompt unexpectedly.
    let invalid = configw::ConfigW::parse("ai.prompt.pure_mode = maybe\n");
    assert!(!super::pure_startup_default(&invalid));
}

#[test]
fn pure_startup_default_enabled_reads_the_configured_file() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());

    let cfg_path = std::env::temp_dir().join(format!(
        "rt_pure_mode_{}.configw",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&cfg_path, "ai.prompt.pure_mode = true\n").unwrap();
    let old_cfg = env::var_os("CONFIGW_PATH");
    unsafe { env::set_var("CONFIGW_PATH", &cfg_path) };
    configw::refresh();

    let from_config = super::pure_startup_default_enabled();

    match old_cfg {
        Some(value) => unsafe { env::set_var("CONFIGW_PATH", value) },
        None => unsafe { env::remove_var("CONFIGW_PATH") },
    }
    configw::refresh();
    let _ = std::fs::remove_file(&cfg_path);

    assert!(from_config, "persisted ai.prompt.pure_mode = true must enable");
}
