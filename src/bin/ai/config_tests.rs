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
            "ai.model.default = glm-5.2-volcano\nvolcano.api_key = test-volcano-key\n",
        )
        .unwrap();
        let old_cfg = env::var_os("CONFIGW_PATH");
        unsafe { env::set_var("CONFIGW_PATH", &cfg_path) };
        configw::refresh();

        let loaded = load_config();
        let resolved_key = match loaded.as_ref() {
            Ok(app) => Some(models::api_key_for_model("glm-5.2-volcano", &app.api_key)),
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
