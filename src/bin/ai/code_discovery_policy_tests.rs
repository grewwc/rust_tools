    use super::{
        CodeDiscoveryConfidence, CodeDiscoveryKind, CodeDiscoveryRecord, PolicyOverride,
        apply_override, classify_finding, default_policy, parse_record_line, recall_rank,
        render_record,
    };

    #[test]
    fn classify_finding_uses_default_root_cause_rule() {
        let record = classify_finding(
            "read_file_lines",
            "root cause: config cache is empty due to missing APP_ENV",
            "- read_file_lines(file=src/main.rs, lines=1..20) => root cause: config cache is empty due to missing APP_ENV",
        )
        .expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::RootCause);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::High);
    }

    #[test]
    fn render_and_parse_record_round_trip() {
        let record = CodeDiscoveryRecord {
            finding: "code_search(...) => fn main()".to_string(),
            kind: CodeDiscoveryKind::EntryPoint,
            confidence: CodeDiscoveryConfidence::High,
        };
        let rendered = render_record(&record);
        assert_eq!(parse_record_line(&rendered), Some(record));
    }

    #[test]
    fn override_updates_kind_weights() {
        let mut policy = default_policy();
        let override_policy: PolicyOverride = serde_json::from_str(
            r#"{
              "recall": {
                "kind_weight": {
                  "root_cause": 999
                }
              }
            }"#,
        )
        .unwrap();
        apply_override(&mut policy, override_policy);

        let high_root = CodeDiscoveryRecord {
            finding: "a".to_string(),
            kind: CodeDiscoveryKind::RootCause,
            confidence: CodeDiscoveryConfidence::High,
        };
        let high_symbol = CodeDiscoveryRecord {
            finding: "b".to_string(),
            kind: CodeDiscoveryKind::Symbol,
            confidence: CodeDiscoveryConfidence::High,
        };
        assert!(recall_rank(&high_root) > recall_rank(&high_symbol));
    }
