    use super::*;
    use chrono::Utc;

    #[test]
    fn test_memory_importance_score() {
        let mut importance = MemoryImportance::new();
        assert_eq!(importance.frequency, 0);
        assert_eq!(importance.recency, 1.0);
        assert_eq!(importance.generality, 0.5);
        assert!(!importance.user_validated);

        // Initial score
        let initial_score = importance.score();
        assert!(initial_score > 0.0 && initial_score < 1.0);

        // Add a reference
        for _ in 0..10 {
            importance.increment_frequency();
        }
        assert_eq!(importance.frequency, 10);

        // User confirmation
        importance.mark_user_validated();
        assert!(importance.user_validated);

        // Score should increase
        let new_score = importance.score();
        assert!(new_score > initial_score);
    }

    #[test]
    fn test_memory_importance_recency_decay() {
        let mut importance = MemoryImportance::new();

        // Memory from 30 days ago
        let old_timestamp = (Utc::now() - chrono::Duration::days(30)).to_rfc3339();
        importance.update_recency(&old_timestamp);

        // Recency should have decayed to about 0.5 (half-life)
        assert!(importance.recency > 0.4 && importance.recency < 0.6);

        // Memory from 90 days ago
        let very_old_timestamp = (Utc::now() - chrono::Duration::days(90)).to_rfc3339();
        importance.update_recency(&very_old_timestamp);

        // Recency should be very low
        assert!(importance.recency < 0.2);
    }

    #[test]
    fn test_memory_importance_generality() {
        let mut importance = MemoryImportance::new();

        // General category
        importance.evaluate_generality("common_sense", &vec![]);
        assert!(importance.generality >= 0.7);

        // Specific category
        importance.evaluate_generality("user_specific", &vec![]);
        assert!(importance.generality <= 0.5);

        // With general tags
        importance.evaluate_generality(
            "user_specific",
            &vec!["general".to_string(), "core".to_string()],
        );
        assert!(importance.generality >= 0.5);
    }

    #[test]
    fn test_should_prune() {
        let mut importance = MemoryImportance::new();

        // High-value memories must not be pruned
        importance.frequency = 10;
        importance.user_validated = true;
        assert!(!importance.should_prune(0.3));

        // Low-value memories should be pruned
        let mut low_importance = MemoryImportance::new();
        low_importance.frequency = 0;
        low_importance.recency = 0.1;
        low_importance.generality = 0.2;
        assert!(low_importance.should_prune(0.3));
    }
