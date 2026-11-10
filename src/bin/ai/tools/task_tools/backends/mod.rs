//! The sole registration entry point for subagent backend implementations.

use std::sync::{Arc, LazyLock};

use rust_tools::commonw::FastMap;
use serde_json::Value;

use super::backend::{Backend, BackendId, Descriptor};

mod kernel;
mod traecli;

pub(crate) struct BackendRegistry {
    backends: FastMap<BackendId, Arc<dyn Backend>>,
    task_names: FastMap<&'static str, BackendId>,
    use_names: FastMap<&'static str, BackendId>,
    ordered_ids: Vec<BackendId>,
}

impl BackendRegistry {
    pub(crate) fn new(
        backends: impl IntoIterator<Item = Arc<dyn Backend>>,
    ) -> Result<Self, String> {
        let mut registry = Self {
            backends: FastMap::default(),
            task_names: FastMap::default(),
            use_names: FastMap::default(),
            ordered_ids: Vec::new(),
        };
        for backend in backends {
            let descriptor = backend.descriptor();
            let id = descriptor.id;
            if id.as_str().is_empty() || !descriptor.task_names.contains(&id.as_str()) {
                return Err(format!(
                    "backend '{}' must declare its canonical task name",
                    id.as_str()
                ));
            }
            if registry.backends.insert(id, backend).is_some() {
                return Err(format!("duplicate backend id '{}'", id.as_str()));
            }
            for (names, index) in [
                (descriptor.task_names, &mut registry.task_names),
                (descriptor.use_names, &mut registry.use_names),
            ] {
                for &name in names {
                    if name.is_empty() || index.insert(name, id).is_some() {
                        return Err(format!("duplicate or empty backend name '{name}'"));
                    }
                }
            }
            registry.ordered_ids.push(id);
        }
        for (name, id) in &registry.use_names {
            if registry
                .task_names
                .get(name)
                .is_some_and(|task_id| task_id != id)
            {
                return Err(format!("conflicting task and /use backend name '{name}'"));
            }
        }
        registry
            .ordered_ids
            .sort_by_key(|id| (registry.backends[id].descriptor().display_order, *id));
        Ok(registry)
    }

    pub(crate) fn get(&self, id: BackendId) -> Option<Arc<dyn Backend>> {
        self.backends.get(&id).cloned()
    }

    pub(crate) fn descriptors(&self) -> impl Iterator<Item = &'static Descriptor> + '_ {
        self.ordered_ids
            .iter()
            .map(|id| self.backends[id].descriptor())
    }

    pub(crate) fn canonical_id(&self, name: &str) -> Option<BackendId> {
        self.ordered_ids
            .iter()
            .copied()
            .find(|id| id.as_str() == name)
    }

    pub(crate) fn task_id(&self, name: &str) -> Option<BackendId> {
        self.task_names.get(name).copied()
    }

    pub(crate) fn use_id(&self, name: &str) -> Option<BackendId> {
        self.use_names.get(name).copied()
    }

    pub(crate) fn worker_use_id(&self, name: &str) -> Option<BackendId> {
        self.use_id(name)
            .filter(|id| self.backends[id].descriptor().supports_worker_model)
    }

    pub(crate) fn use_descriptors(&self) -> Vec<&'static Descriptor> {
        let mut descriptors: Vec<_> = self
            .descriptors()
            .filter(|descriptor| !descriptor.use_names.is_empty())
            .collect();
        descriptors.sort_by_key(|descriptor| (descriptor.use_order, descriptor.id));
        descriptors
    }

    pub(crate) fn use_names(&self) -> Vec<&'static str> {
        self.use_descriptors()
            .into_iter()
            .flat_map(|descriptor| descriptor.use_names.iter().copied())
            .collect()
    }

    pub(crate) fn use_usage(&self) -> String {
        let choices = self
            .use_descriptors()
            .into_iter()
            .map(|descriptor| {
                let name = descriptor.use_names[0];
                if descriptor.supports_model_pick {
                    format!("{name} [model]")
                } else {
                    name.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("|");
        format!("/use [{choices}]")
    }

    /// Kernel state still displays the first worker backend's configured model.
    pub(crate) fn worker_backend_for(&self, selected: BackendId) -> Option<BackendId> {
        self.get(selected)
            .filter(|backend| backend.descriptor().supports_worker_model)
            .map(|_| selected)
            .or_else(|| {
                self.use_descriptors()
                    .into_iter()
                    .find(|descriptor| descriptor.supports_worker_model)
                    .map(|descriptor| descriptor.id)
            })
    }

    /// Honor a resumable session default; otherwise retain the first resume backend.
    pub(crate) fn resume_backend_for(&self, selected: BackendId) -> BackendId {
        self.get(selected)
            .filter(|backend| backend.descriptor().supports_resume)
            .map(|_| selected)
            .or_else(|| {
                self.descriptors()
                    .find(|descriptor| descriptor.supports_resume)
                    .map(|descriptor| descriptor.id)
            })
            .unwrap_or(selected)
    }

    pub(crate) fn resume_required_error(&self) -> String {
        let choices = self
            .descriptors()
            .filter(|descriptor| descriptor.supports_resume)
            .map(|descriptor| format!("'{}'", descriptor.id.as_str()))
            .collect::<Vec<_>>()
            .join(" or ");
        format!("'backend_continue_from' needs backend {choices}")
    }

    /// Expand built-in schema placeholders before user metadata overrides apply.
    pub(crate) fn expand_tool_schema(&self, value: &mut Value) {
        let descriptors: Vec<_> = self.descriptors().collect();
        let names = descriptors
            .iter()
            .map(|descriptor| Value::from(descriptor.id.as_str()))
            .collect::<Vec<_>>();
        let choices = descriptors
            .iter()
            .map(|descriptor| format!("'{}'", descriptor.id.as_str()))
            .collect::<Vec<_>>()
            .join(" or ");
        let worker_choices = descriptors
            .iter()
            .filter(|descriptor| descriptor.supports_worker_model)
            .map(|descriptor| format!("'{}'", descriptor.id.as_str()))
            .collect::<Vec<_>>()
            .join(" or ");
        let worker_description = descriptors
            .iter()
            .filter_map(|descriptor| descriptor.text.worker_model_schema)
            .collect::<Vec<_>>()
            .join(" ");
        let continuation_description = descriptors
            .iter()
            .filter_map(|descriptor| descriptor.text.continuation_schema)
            .collect::<Vec<_>>()
            .join(" ");
        let replacements = [
            ("{subagent.backend_choices}", choices),
            ("{subagent.worker_backend_choices}", worker_choices),
            ("{subagent.backend_model_description}", worker_description),
            (
                "{subagent.backend_continue_description}",
                continuation_description,
            ),
        ];
        fn expand(value: &mut Value, names: &[Value], replacements: &[(&str, String)]) {
            match value {
                Value::Array(values)
                    if values.len() == 1 && values[0] == "{subagent.backend_ids}" =>
                {
                    *values = names.to_vec();
                }
                Value::String(text) => {
                    for (placeholder, replacement) in replacements {
                        *text = text.replace(placeholder, replacement);
                    }
                }
                Value::Array(values) => {
                    for value in values {
                        expand(value, names, replacements);
                    }
                }
                Value::Object(values) => {
                    for value in values.values_mut() {
                        expand(value, names, replacements);
                    }
                }
                _ => {}
            }
        }
        expand(value, &names, &replacements);
    }

    pub(crate) fn parse_optional(
        &self,
        value: Option<&Value>,
    ) -> Result<Option<BackendId>, String> {
        let Some(value) = value.filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        let name = value
            .as_str()
            .ok_or_else(|| "'backend' must be a string".to_string())?;
        if name.is_empty() {
            return Ok(None);
        }
        self.task_id(name).map(Some).ok_or_else(|| {
            let expected = self
                .ordered_ids
                .iter()
                .map(|id| format!("'{}'", id.as_str()))
                .collect::<Vec<_>>()
                .join(" or ");
            format!("unknown subagent backend '{name}' (expected {expected})")
        })
    }
}

static BACKENDS: LazyLock<BackendRegistry> = LazyLock::new(|| {
    let backends: [Arc<dyn Backend>; 2] = [
        Arc::new(kernel::KernelBackend),
        Arc::new(traecli::TraeCliBackend),
    ];
    BackendRegistry::new(backends).expect("invalid subagent backend registry")
});

/// Force static validation at startup, without querying models or launching a CLI.
pub(crate) fn backend_registry() -> &'static BackendRegistry {
    &BACKENDS
}

#[cfg(test)]
mod tests {
    use super::super::backend::{
        Admission, BackendFuture, Continuation, ContinueContext, Events, ExecutionIdentity,
        ModelContext, Outcome, RunContext,
    };
    use super::*;
    use std::sync::atomic::AtomicBool;

    struct MetadataOnly(&'static Descriptor);

    impl Backend for MetadataOnly {
        fn descriptor(&self) -> &'static Descriptor {
            self.0
        }
        fn preflight(&self, _: &Admission<'_>) -> Result<(), String> {
            panic!("unexpected admission")
        }
        fn attribution(&self, _: &ModelContext<'_>) -> ExecutionIdentity {
            panic!("unexpected attribution")
        }
        fn continuation(&self, _: &ContinueContext<'_>) -> Result<Continuation, String> {
            panic!("unexpected continuation")
        }
        fn model_names(&self) -> Vec<String> {
            panic!("registration must not discover models")
        }
        fn run<'a>(
            &'a self,
            _: RunContext,
            _: Events,
            _: Arc<AtomicBool>,
        ) -> BackendFuture<'a, Outcome> {
            panic!("registration must not run a backend")
        }
    }

    #[test]
    fn registry_validates_ids_and_aliases_without_discovery() {
        static ALIAS_COLLISION: Descriptor = Descriptor {
            id: BackendId::new("other"),
            task_names: &["other", "trae-cli"],
            use_names: &[],
            ..kernel::DESCRIPTOR
        };
        static USE_COLLISION: Descriptor = Descriptor {
            id: BackendId::new("other"),
            task_names: &["other"],
            use_names: &["trae-cli"],
            ..kernel::DESCRIPTOR
        };
        let make = |descriptors: Vec<&'static Descriptor>| {
            BackendRegistry::new(
                descriptors
                    .into_iter()
                    .map(|descriptor| Arc::new(MetadataOnly(descriptor)) as Arc<dyn Backend>),
            )
        };
        assert!(make(vec![&kernel::DESCRIPTOR, &traecli::DESCRIPTOR]).is_ok());
        assert!(make(vec![&kernel::DESCRIPTOR, &kernel::DESCRIPTOR]).is_err());
        assert!(make(vec![&traecli::DESCRIPTOR, &ALIAS_COLLISION]).is_err());
        assert!(make(vec![&traecli::DESCRIPTOR, &USE_COLLISION]).is_err());
        let registry = backend_registry();
        for name in ["traecli", "trae-cli", "coco"] {
            assert_eq!(registry.task_id(name), Some(BackendId::TRAECLI));
        }
        assert_eq!(registry.use_id("traecli"), Some(BackendId::TRAECLI));
        assert_eq!(registry.use_id("coco"), None);
        assert_eq!(BackendId::from_stored(Some("coco")), BackendId::KERNEL);
        for value in [None, Some(Value::Null), Some(Value::from(""))] {
            assert_eq!(BackendId::parse(value.as_ref()).unwrap(), BackendId::KERNEL);
        }
        for (value, expected) in [
            (
                Value::from(" "),
                "unknown subagent backend ' ' (expected 'kernel' or 'traecli')",
            ),
            (
                Value::from("other"),
                "unknown subagent backend 'other' (expected 'kernel' or 'traecli')",
            ),
            (Value::from(1), "'backend' must be a string"),
        ] {
            assert_eq!(BackendId::parse(Some(&value)).unwrap_err(), expected);
        }
        assert_eq!(
            registry.use_usage(),
            "/use [traecli [model]|kernel [model]]"
        );
        assert_eq!(registry.use_names(), ["traecli", "kernel"]);
        assert_eq!(
            registry.resume_backend_for(BackendId::KERNEL),
            BackendId::TRAECLI
        );
        assert_eq!(
            registry.resume_required_error(),
            "'backend_continue_from' needs backend 'traecli'"
        );
        assert_eq!(BackendId::KERNEL.stored_value().as_deref(), Some("kernel"));
        assert_eq!(
            BackendId::TRAECLI.stored_value().as_deref(),
            Some("traecli")
        );
        for alias in ["traecli", "trae-cli", "coco"] {
            let id = BackendId::parse(Some(&Value::from(alias))).unwrap();
            assert_eq!(id.stored_value().as_deref(), Some("traecli"));
        }
        for alias in ["trae-cli", "coco"] {
            assert_eq!(BackendId::from_stored(Some(alias)), BackendId::KERNEL);
        }

        // A metadata-only plugin must reach every consumer without model discovery.
        static CODEX: Descriptor = Descriptor {
            id: BackendId::new("codex"),
            task_names: &["codex", "codex-cli"],
            use_names: &["codex", "cx"],
            supports_worker_model: true,
            supports_resume: true,
            display_order: 2,
            use_order: 2,
            text: super::super::backend::BackendText {
                display_name: "Codex",
                worker_model_label: Some("Codex worker model"),
                worker_model_schema: Some("Codex model choice."),
                continuation_schema: Some("Codex resume target."),
                ..kernel::DESCRIPTOR.text
            },
            ..kernel::DESCRIPTOR
        };
        let plugins = make(vec![&kernel::DESCRIPTOR, &traecli::DESCRIPTOR, &CODEX]).unwrap();
        assert_eq!(plugins.use_id("cx"), Some(CODEX.id));
        assert_eq!(plugins.worker_use_id("cx"), Some(CODEX.id));
        assert_eq!(plugins.task_id("codex-cli"), Some(CODEX.id));
        assert_eq!(plugins.canonical_id("codex-cli"), None);
        assert_eq!(
            plugins.use_usage(),
            "/use [traecli [model]|kernel [model]|codex [model]]"
        );
        assert_eq!(plugins.use_names(), ["traecli", "kernel", "codex", "cx"]);
        assert_eq!(plugins.resume_backend_for(CODEX.id), CODEX.id);
        assert_eq!(plugins.worker_backend_for(CODEX.id), Some(CODEX.id));
        assert_eq!(
            plugins.worker_backend_for(BackendId::KERNEL),
            Some(BackendId::TRAECLI)
        );
        let schema_template = serde_json::json!({
            "backend": { "enum": ["{subagent.backend_ids}"], "description": "{subagent.backend_choices}" },
            "model": "{subagent.worker_backend_choices}",
            "backend_model": "{subagent.backend_model_description}",
            "backend_continue_from": "{subagent.backend_continue_description}",
        });
        let mut original = schema_template.clone();
        registry.expand_tool_schema(&mut original);
        assert_eq!(
            original["backend"]["enum"],
            serde_json::json!(["kernel", "traecli"])
        );
        assert_eq!(original["backend"]["description"], "'kernel' or 'traecli'");
        assert_eq!(
            original["backend_model"],
            traecli::DESCRIPTOR.text.worker_model_schema.unwrap()
        );
        assert_eq!(
            original["backend_continue_from"],
            traecli::DESCRIPTOR.text.continuation_schema.unwrap()
        );
        let mut extended = schema_template;
        plugins.expand_tool_schema(&mut extended);
        assert_eq!(
            extended["backend"]["enum"],
            serde_json::json!(["kernel", "traecli", "codex"])
        );
        assert_eq!(extended["model"], "'traecli' or 'codex'");
        assert!(
            extended["backend_model"]
                .as_str()
                .unwrap()
                .ends_with("Codex model choice.")
        );
        assert!(
            extended["backend_continue_from"]
                .as_str()
                .unwrap()
                .ends_with("Codex resume target.")
        );
    }

    #[tokio::test]
    async fn kernel_registry_routes_callback_with_failure_evidence() {
        let backend = backend_registry().get(BackendId::KERNEL).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let expected_cancel = cancel.clone();
        let run = RunContext::kernel(move |_, cancel| {
            assert!(Arc::ptr_eq(&cancel, &expected_cancel));
            Box::pin(async {
                Outcome {
                    execution_error: Some("turn failed".to_string()),
                    parent_payload: "retained evidence".to_string(),
                    validation_body: "final body".to_string(),
                    progress: "working".to_string(),
                    ..Outcome::default()
                }
            })
        });
        let outcome = backend.run(run, Events::default(), cancel).await;
        assert_eq!(outcome.execution_error.as_deref(), Some("turn failed"));
        assert_eq!(outcome.parent_payload, "retained evidence");
        assert_eq!(outcome.validation_body, "final body");
    }
}
