use std::env;

use super::load_config;
use crate::ai::{models, test_support::ENV_LOCK};
use crate::commonw::configw;

#[test]
fn load_config_accepts_default_model_specific_api_key_config_key() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());

    let cfg_path = std::env::temp_dir().join(format!(
        "rt_volcano_model_{}.configw",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(
        &cfg_path,
        "ai.model.default = deepseek-v4-flash-volcano\nvolcano.api_key = test-volcano-key\n",
    )
    .unwrap();
    let old_cfg = env::var_os("CONFIGW_PATH");
    unsafe { env::set_var("CONFIGW_PATH", &cfg_path) };
    configw::refresh();

    let loaded = load_config();
    let resolved_key = match loaded.as_ref() {
        Ok(app) => Some(models::api_key_for_model(
            "deepseek-v4-flash-volcano",
            &app.api_key,
        )),
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
fn history_max_chars_derives_from_model_window_unless_configured() {
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
    let small_budget = super::history_max_chars_for_model(&unset, &small);
    let large_budget = super::history_max_chars_for_model(&unset, &large);
    let small_soft_chars = crate::ai::request::soft_target_chars_for_model(&small);
    // Small windows keep the former fixed default: a lower budget would
    // compact the cached prefix more often than before.
    assert!(small_soft_chars * 2 / 3 < super::MIN_DERIVED_HISTORY_MAX_CHARS);
    assert_eq!(small_budget, super::MIN_DERIVED_HISTORY_MAX_CHARS);
    assert!(
        large_budget > 4 * small_budget,
        "1M-token windows must keep far more resident history: {large_budget} vs {small_budget}"
    );

    let explicit = configw::ConfigW::parse("ai.history.max_chars = 90000\n");
    assert_eq!(super::history_max_chars_for_model(&explicit, &small), 90_000);
    assert_eq!(super::history_max_chars_for_model(&explicit, &large), 90_000);
    let disabled = configw::ConfigW::parse("ai.history.max_chars = 0\n");
    assert_eq!(super::history_max_chars_for_model(&disabled, &large), 0);
}
