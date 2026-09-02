//! Model registry and VL-model selection tests.

use super::super::*;
use super::{any_vl_model_handle, vl_model_handle_at};

#[test]
fn default_model_names_exist() {
    assert!(!model_names::all().is_empty());
}

#[test]
fn determine_vl_model_supports_selector_and_fuzzy_name() {
    // Empty input goes through default_vl_model (quality_tier first); a numeric index goes through "the Nth entry after filtering by is_vl".
    // The two paths are not equivalent; assert each against its own invariant to avoid hardcoding specific model names.
    let empty = models::determine_vl_model("");
    let zero = models::determine_vl_model("0");
    let first_vl = any_vl_model_handle();
    assert_eq!(
        zero, first_vl,
        "selector \"0\" should pick first VL in model registry"
    );
    // empty only requires a VL model (best-by-tier may differ from first_vl).
    assert!(
        model_names::find_by_identifier(&empty)
            .map(|m| m.is_vl)
            .unwrap_or(false)
    );

    if let Some(vl1) = vl_model_handle_at(1) {
        assert_eq!(models::determine_vl_model("1"), vl1);
    } else {
        // Fall back to default_vl_model when out of range
        assert_eq!(models::determine_vl_model("1"), empty);
    }

    // Feeding a known VL model name directly should return the same name (exact match).
    let canonical = models::determine_vl_model(&first_vl);
    assert_eq!(canonical, first_vl);
}

#[test]
fn tools_default_flag_is_respected_per_model_entry() {
    // This used to hardcode the tools_enabled behavior of qwen3.5-flash / qwen3-max; both models
    // have been removed from the model registry. Scan the real entries instead and verify that
    // models::tools_enabled stays aligned with ModelDef.tools_default_enabled, keeping the "config is truth" invariant.
    for def in model_names::all() {
        assert_eq!(
            models::tools_enabled(&def.name),
            def.tools_default_enabled,
            "model {} tools_enabled should match its tools_default_enabled flag",
            def.name
        );
    }
}
