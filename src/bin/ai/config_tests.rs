use std::env;

use super::{AgentTeamOptimizations, load_config};
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
    // A larger advertised context window must not increase resident history.
    assert_eq!(
        super::history_max_chars_for_model(&unset, &small),
        super::DEFAULT_HISTORY_MAX_CHARS
    );
    assert_eq!(
        super::history_max_chars_for_model(&unset, &large),
        super::DEFAULT_HISTORY_MAX_CHARS
    );

    let explicit = configw::ConfigW::parse("ai.history.max_chars = 90000\n");
    assert_eq!(
        super::history_max_chars_for_model(&explicit, &small),
        90_000
    );
    assert_eq!(
        super::history_max_chars_for_model(&explicit, &large),
        90_000
    );
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

    assert!(
        from_config,
        "persisted ai.prompt.pure_mode = true must enable"
    );
}

const AGENT_TEAM_OPTIMIZATION_KEYS: [&str; 5] = [
    "ai.agent_team.dedupe_base_prompt",
    "ai.agent_team.lazy_group_schemas",
    "ai.agent_team.persist_status_cursor",
    "ai.agent_team.merge_ready_results",
    "ai.agent_team.auto_mount_lifecycle_tools",
];

fn agent_team_optimization_states(switches: &AgentTeamOptimizations) -> [bool; 5] {
    [
        switches.dedupe_base_prompt(),
        switches.lazy_group_schemas(),
        switches.persist_status_cursor(),
        switches.merge_ready_results(),
        switches.auto_mount_lifecycle_tools(),
    ]
}

#[test]
fn agent_team_optimizations_default_to_off() {
    let cfg = configw::ConfigW::parse("");
    let switches = AgentTeamOptimizations::from_config(&cfg);

    assert_eq!(switches, AgentTeamOptimizations::default());
    assert!(!switches.optimizations_enabled());
    assert_eq!(agent_team_optimization_states(&switches), [false; 5]);
}

#[test]
fn agent_team_optimizations_require_master_switch() {
    let individual_switches = AGENT_TEAM_OPTIMIZATION_KEYS
        .map(|key| format!("{key} = true\n"))
        .concat();
    for master in [None, Some("false"), Some("off"), Some("invalid")] {
        let master_switch = master
            .map(|raw| format!("ai.agent_team.optimizations_enabled = {raw}\n"))
            .unwrap_or_default();
        let cfg = configw::ConfigW::parse(&format!("{individual_switches}{master_switch}"));
        let switches = AgentTeamOptimizations::from_config(&cfg);
        assert!(!switches.optimizations_enabled(), "master: {master:?}");
        assert_eq!(
            agent_team_optimization_states(&switches),
            [false; 5],
            "master: {master:?}"
        );
    }
}

#[test]
fn agent_team_optimizations_can_be_enabled_independently() {
    let master_only = configw::ConfigW::parse("ai.agent_team.optimizations_enabled = true\n");
    let switches = AgentTeamOptimizations::from_config(&master_only);
    assert!(switches.optimizations_enabled());
    assert_eq!(agent_team_optimization_states(&switches), [false; 5]);

    for (index, key) in AGENT_TEAM_OPTIMIZATION_KEYS.iter().enumerate() {
        let cfg = configw::ConfigW::parse(&format!(
            "ai.agent_team.optimizations_enabled = true\n{key} = true\n"
        ));
        let switches = AgentTeamOptimizations::from_config(&cfg);
        let mut expected = [false; 5];
        expected[index] = true;
        assert_eq!(agent_team_optimization_states(&switches), expected, "{key}");
    }
}

#[test]
fn agent_team_optimizations_follow_existing_switch_parsing() {
    for (raw, expected) in [
        ("1", true),
        ("TRUE", true),
        ("yes", true),
        ("on", true),
        ("enable", true),
        ("enabled", true),
        ("0", false),
        ("FALSE", false),
        ("no", false),
        ("off", false),
        ("disable", false),
        ("disabled", false),
        ("maybe", false),
    ] {
        let individual_switches = AGENT_TEAM_OPTIMIZATION_KEYS
            .map(|key| format!("{key} = {raw}\n"))
            .concat();
        let cfg = configw::ConfigW::parse(&format!(
            "ai.agent_team.optimizations_enabled = true\n{individual_switches}"
        ));
        let switches = AgentTeamOptimizations::from_config(&cfg);
        assert_eq!(
            agent_team_optimization_states(&switches),
            [expected; 5],
            "individual switches: {raw}"
        );

        let individual_switches = AGENT_TEAM_OPTIMIZATION_KEYS
            .map(|key| format!("{key} = true\n"))
            .concat();
        let cfg = configw::ConfigW::parse(&format!(
            "ai.agent_team.optimizations_enabled = {raw}\n{individual_switches}"
        ));
        let switches = AgentTeamOptimizations::from_config(&cfg);
        assert_eq!(switches.optimizations_enabled(), expected, "master: {raw}");
        assert_eq!(
            agent_team_optimization_states(&switches),
            [expected; 5],
            "master: {raw}"
        );
    }
}

#[test]
fn agent_team_optimizations_read_the_configured_file() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let cfg_path = std::env::temp_dir().join(format!(
        "rt_agent_team_{}.configw",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(
        &cfg_path,
        concat!(
            "api_key = test-agent-team-key\n",
            "ai.history.keep_last = 64\n",
            "ai.agent_team.optimizations_enabled = true\n",
            "ai.agent_team.dedupe_base_prompt = true\n",
            "ai.agent_team.lazy_group_schemas = false\n",
            "ai.agent_team.persist_status_cursor = true\n",
            "ai.agent_team.merge_ready_results = false\n",
            "ai.agent_team.auto_mount_lifecycle_tools = true\n",
        ),
    )
    .unwrap();
    let old_cfg = env::var_os("CONFIGW_PATH");
    unsafe { env::set_var("CONFIGW_PATH", &cfg_path) };
    configw::refresh();

    let from_config = super::agent_team_optimizations();
    let loaded = load_config();
    let from_app = loaded
        .as_ref()
        .ok()
        .map(|app| app.agent_team_optimizations());

    match old_cfg {
        Some(value) => unsafe { env::set_var("CONFIGW_PATH", value) },
        None => unsafe { env::remove_var("CONFIGW_PATH") },
    }
    configw::refresh();
    let _ = std::fs::remove_file(&cfg_path);

    let loaded = loaded.expect("agent-team switches must preserve existing config loading");
    assert_eq!(loaded.history_keep_last, 64);
    assert_eq!(from_app, Some(from_config));
    assert!(from_config.optimizations_enabled());
    assert_eq!(
        agent_team_optimization_states(&from_config),
        [true, false, true, false, true]
    );
}
