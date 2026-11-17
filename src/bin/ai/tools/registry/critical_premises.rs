use crate::ai::tools::common::{ToolRegistration, ToolSpec};

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "critical_premises",
        description: "",
        execute: super::service::execute,
    }
});

#[cfg(test)]
mod tests {
    use serde_json::Value;

    #[test]
    fn critical_premises_metadata_is_lazy_and_enableable_by_name() {
        let metadata: Value = serde_json::from_str(include_str!(
            "../../tool_descriptions/critical_premises.json"
        ))
        .unwrap();
        assert_eq!(metadata["name"], "critical_premises");
        assert_eq!(metadata["groups"], serde_json::json!(["builtin"]));
        assert!(metadata.get("hidden").is_none());
        assert!(
            metadata["first_use_guidance"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        );
        assert!(
            inventory::iter::<super::ToolRegistration>
                .into_iter()
                .any(|r| r.spec.name == "critical_premises")
        );
    }
}
