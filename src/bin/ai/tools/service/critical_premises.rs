use serde::Serialize;
use serde_json::Value;

use super::state::ValidatedUpdate;

const PROVISIONAL_KIND: &str = "critical_premises.provisional.v1";
const MAX_PROVISIONAL_BYTES: usize = 32_768;

#[derive(Serialize)]
struct Provisional<'a> {
    kind: &'static str,
    semantic_origin: &'static str,
    update: &'a ValidatedUpdate,
}

/// Syntax-only executor: no state, evidence lookup, reviewer or history mutation.
/// The driver must replace this provisional result with `PremiseState::apply`'s
/// receipt before append; a binding failure must instead become a tool error.
pub(in crate::ai) fn execute(args: &Value) -> Result<String, String> {
    let update = ValidatedUpdate::from_value(args)?;
    let result = serde_json::to_string(&Provisional {
        kind: PROVISIONAL_KIND,
        semantic_origin: "model_derived",
        update: &update,
    })
    .map_err(|e| e.to_string())?;
    if result.len() > MAX_PROVISIONAL_BYTES {
        return Err("Provisional result exceeds the bounded size.".into());
    }
    Ok(result)
}

/// Extract only a syntax-validated update from this executor's own provisional
/// envelope. A runtime receipt marker or a reviewer response is not accepted.
pub(in crate::ai) fn validated_from_provisional(content: &str) -> Result<ValidatedUpdate, String> {
    if content.len() > MAX_PROVISIONAL_BYTES {
        return Err("Provisional result exceeds the bounded size.".into());
    }
    let value: Value = serde_json::from_str(content).map_err(|e| e.to_string())?;
    let object = value.as_object().ok_or("Expected provisional object.")?;
    if object.len() != 3
        || value["kind"] != PROVISIONAL_KIND
        || value["semantic_origin"] != "model_derived"
    {
        return Err("Expected syntax-only critical_premises provisional result.".into());
    }
    ValidatedUpdate::from_value(&value["update"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_premises_provisional_never_claims_runtime_support() {
        let args = serde_json::json!({
            "id": "key", "proposition": "The target was verified",
            "impact": "main_answer", "status": "supported",
            "evidence": [{ "tool_call_id": "unknown", "object": "/known/file", "coverage": "full" }]
        });
        let result = execute(&args).unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["kind"], PROVISIONAL_KIND);
        assert_eq!(value["semantic_origin"], "model_derived");
        assert!(value.get("evidence").is_none());
        assert!(validated_from_provisional(&result).is_ok());
        let mut forged = value;
        forged["kind"] = "critical_premises.runtime_bound.v1".into();
        assert!(validated_from_provisional(&forged.to_string()).is_err());
    }
}
