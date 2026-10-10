use std::fs;
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn distilled_test_store(label: &str) -> MemoryStore {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        MemoryStore::for_tests_with_path(std::env::temp_dir().join(format!(
            "rt_distilled_{label}_{stamp}.jsonl"
        )))
    }

    fn distilled_test_entry(id: &str, note: &str, source: &str) -> AgentMemoryEntry {
        use crate::ai::knowledge::distilled::{
            DISTILLED_SCHEMA, DistilledEvidence, DistilledMetadata, digest, entry_content_digest,
        };
        let mut entry = AgentMemoryEntry {
            id: Some(id.to_string()),
            category: "project_memory".to_string(),
            note: note.to_string(),
            ..AgentMemoryEntry::default()
        };
        entry.distilled = Some(DistilledMetadata {
            schema: DISTILLED_SCHEMA,
            scope: "/project".to_string(),
            revision: 1,
            topic_key: "build".to_string(),
            verified: true,
            content_digest: entry_content_digest(&entry),
            evidence: vec![DistilledEvidence {
                source_digest: digest(source),
                message_id: "message-1".to_string(),
                role: "user".to_string(),
                quote: note.to_string(),
                text_digest: digest(note),
            }],
            source_digests: vec![digest(source)],
            previous_revisions: Vec::new(),
        });
        entry
    }

    #[test]
    fn distilled_batch_late_failure_preserves_all_canonical_rows() {
        let store = distilled_test_store("batch_failure");
        store.upsert_distilled(distilled_test_entry("build", "Use cargo check", "source-1"), None).unwrap();
        let before = fs::read(store.path()).unwrap();
        let mut first = distilled_test_entry("tests", "Run focused tests", "source-2");
        first.distilled.as_mut().unwrap().topic_key = "testing".to_string();
        let second = distilled_test_entry("build", "Use cargo test", "source-3");
        let mut invalid = second.clone();
        invalid.distilled = None;
        let failures = [
            (second.clone(), Some(("build".to_string(), 2))),
            (second, None),
            (invalid, None),
        ];
        for (entry, expected) in failures {
            for dry_run in [true, false] {
                assert!(store.upsert_distilled_batch(
                    vec![(first.clone(), None), (entry.clone(), expected.clone())], dry_run,
                ).is_err());
                assert_eq!(fs::read(store.path()).unwrap(), before);
            }
        }
        let committed = store.upsert_distilled_batch(vec![(first, None)], false).unwrap();
        assert!(committed[0].inserted);
        assert_eq!(store.active_distilled_entries("/project").unwrap().len(), 2);
        let _ = fs::remove_file(store.path());
    }

    #[test]
    fn distilled_batch_preview_matches_commit_with_new_source_and_unchanged_note() {
        let store = distilled_test_store("batch_preview");
        store.upsert_distilled(distilled_test_entry("build", "Use cargo check", "source-1"), None).unwrap();
        let mut duplicate = distilled_test_entry("lint", "Run clippy", "lint-source");
        duplicate.distilled.as_mut().unwrap().topic_key = "lint".to_string();
        store.upsert_distilled(duplicate.clone(), None).unwrap();
        let before = fs::read(store.path()).unwrap();
        let new_evidence = distilled_test_entry("build", "Use cargo check", "source-2");
        let mut inserted = distilled_test_entry("tests", "Run focused tests", "test-source");
        inserted.distilled.as_mut().unwrap().topic_key = "testing".to_string();
        let pending = vec![
            (new_evidence.clone(), Some(("build".to_string(), 1))),
            (inserted, None),
            (duplicate, Some(("lint".to_string(), 1))),
        ];
        let preview = store.upsert_distilled_batch(pending.clone(), true).unwrap();
        assert!(preview[0].updated && !preview[0].duplicate);
        assert!(preview[1].inserted);
        assert!(preview[2].duplicate);
        assert_eq!(fs::read(store.path()).unwrap(), before);
        assert_eq!(preview, store.upsert_distilled_batch(pending, false).unwrap());
        let active = store.active_distilled_entries("/project").unwrap();
        assert_eq!(active.len(), 3);
        let metadata = active.iter().find(|entry| entry.id.as_deref() == Some("build"))
            .unwrap().distilled.as_ref().unwrap();
        assert_eq!(metadata.revision, 2);
        assert_eq!(metadata.source_digests.len(), 2);
        assert_eq!(metadata.previous_revisions.len(), 1);
        assert_eq!(metadata.evidence, new_evidence.distilled.as_ref().unwrap().evidence);
        let stable = fs::read(store.path()).unwrap();
        let replay = vec![(new_evidence, Some(("build".to_string(), 2)))];
        let preview = store.upsert_distilled_batch(replay.clone(), true).unwrap();
        assert!(preview[0].duplicate);
        assert_eq!(preview, store.upsert_distilled_batch(replay, false).unwrap());
        assert_eq!(fs::read(store.path()).unwrap(), stable);
        let _ = fs::remove_file(store.path());
    }

    #[test]
    fn distilled_batch_preview_does_not_create_store_or_parent_directory() {
        let path = distilled_test_store("preview_missing").path().join("memory.jsonl");
        let store = MemoryStore::for_tests_with_path(path.clone());
        assert!(!path.parent().unwrap().exists());
        let reports = store.upsert_distilled_batch(
            vec![(distilled_test_entry("build", "Use cargo check", "source-1"), None)], true,
        ).unwrap();
        assert!(reports[0].inserted);
        assert!(!path.parent().unwrap().exists());
    }

    #[test]
    fn distilled_batch_repeated_identity_is_rejected_before_commit() {
        let store = distilled_test_store("batch_repeated");
        let original = distilled_test_entry("build", "Use cargo check", "source-1");
        store.upsert_distilled(original.clone(), None).unwrap();
        let before = fs::read(store.path()).unwrap();
        for dry_run in [true, false] {
            assert!(store.upsert_distilled_batch(
                vec![(original.clone(), None), (original.clone(), None)], dry_run,
            ).unwrap_err().contains("repeated canonical ID"));
            assert_eq!(fs::read(store.path()).unwrap(), before);
        }
        let _ = fs::remove_file(store.path());
    }

    #[test]
    fn distilled_batch_concurrent_stale_writer_cannot_commit_its_other_entry() {
        let store = distilled_test_store("batch_concurrent");
        store.upsert_distilled(distilled_test_entry("build", "Use cargo check", "source-1"), None).unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = (0..2).map(|index| {
            let store = MemoryStore::for_tests_with_path(store.path().to_path_buf());
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let id = format!("extra-{index}");
                let mut extra = distilled_test_entry(&id, "Run focused tests", &id);
                extra.distilled.as_mut().unwrap().topic_key = id.clone();
                let update = distilled_test_entry("build", &format!("Build choice {index}"), &id);
                barrier.wait();
                (id, store.upsert_distilled_batch(vec![
                    (extra, None), (update, Some(("build".to_string(), 1))),
                ], false))
            })
        }).collect();
        let results: Vec<_> = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|(_, result)| result.is_ok()).count(), 1);
        let (losing_id, result) = results.iter().find(|(_, result)| result.is_err()).unwrap();
        assert!(result.as_ref().unwrap_err().contains("Stale"));
        let active = store.active_distilled_entries("/project").unwrap();
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|entry| entry.id.as_ref() != Some(losing_id)));
        let _ = fs::remove_file(store.path());
    }

    #[test]
    fn distilled_revision_insert_update_duplicate_and_stale_preserves_rows() {
        let store = distilled_test_store("revisions");
        let ordinary = AgentMemoryEntry {
            id: Some("ordinary".to_string()),
            note: "Keep this non-distilled note".to_string(),
            ..AgentMemoryEntry::default()
        };
        store.append(&ordinary).unwrap();
        let original = distilled_test_entry("canonical", "Use cargo check", "source-1");
        let inserted = store.upsert_distilled(original.clone(), None).unwrap();
        assert_eq!(inserted, DistilledUpsertReport {
            inserted: true, updated: false, duplicate: false, entry_id: "canonical".to_string(),
        });
        let bytes_before_duplicate = fs::read(store.path()).unwrap();
        let mut retry = original.clone();
        retry.id = Some("retry-id".to_string());
        assert!(store.upsert_distilled(retry, None).unwrap().duplicate);
        assert_eq!(fs::read(store.path()).unwrap(), bytes_before_duplicate);

        let updated_entry = distilled_test_entry("ignored-new-id", "Use a focused test", "source-2");
        let updated = store.upsert_distilled(updated_entry.clone(), Some(("canonical".to_string(), 1))).unwrap();
        assert_eq!(updated, DistilledUpsertReport {
            inserted: false, updated: true, duplicate: false, entry_id: "canonical".to_string(),
        });
        let active = store.active_distilled_entries("/project").unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id.as_deref(), Some("canonical"));
        assert_eq!(active[0].note, updated_entry.note);
        let metadata = active[0].distilled.as_ref().unwrap();
        assert_eq!(metadata.revision, 2);
        assert_eq!(metadata.previous_revisions.len(), 1);
        assert_eq!(metadata.previous_revisions[0].revision, 1);
        assert_eq!(metadata.previous_revisions[0].note, original.note);
        assert_eq!(metadata.previous_revisions[0].evidence, original.distilled.as_ref().unwrap().evidence);
        assert_eq!(metadata.source_digests.len(), 2);
        let stable_bytes = fs::read(store.path()).unwrap();
        assert!(store.upsert_distilled(original.clone(), Some(("canonical".to_string(), 2)))
            .unwrap_err().contains("Previously consumed source"));
        assert_eq!(fs::read(store.path()).unwrap(), stable_bytes);
        assert!(store.upsert_distilled(updated_entry.clone(), Some(("canonical".to_string(), 1))).unwrap_err().contains("Stale"));
        assert!(store.upsert_distilled(updated_entry, Some(("canonical".to_string(), 2))).unwrap().duplicate);
        assert_eq!(fs::read(store.path()).unwrap(), stable_bytes);
        let all = store.current_entries_while_locked().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(serde_json::to_value(all.iter().find(|entry| entry.id == ordinary.id).unwrap()).unwrap(), serde_json::to_value(ordinary).unwrap());

        let next = distilled_test_entry("ignored", "Use targeted test names", "source-3");
        store.upsert_distilled(next, Some(("canonical".to_string(), 2))).unwrap();
        let active = store.active_distilled_entries("/project").unwrap();
        let metadata = active[0].distilled.as_ref().unwrap();
        assert_eq!(metadata.revision, 3);
        assert_eq!(metadata.previous_revisions.len(), 2);
        assert_eq!(metadata.previous_revisions[0].note, original.note);
        let _ = fs::remove_file(store.path());
    }

    #[test]
    fn distilled_revision_rejects_collisions_and_invalid_content_without_mutation() {
        let store = distilled_test_store("collisions");
        let original = distilled_test_entry("canonical", "Original note", "source-1");
        store.upsert_distilled(original.clone(), None).unwrap();
        let stable_bytes = fs::read(store.path()).unwrap();
        let mut invalid = original.clone();
        invalid.note = "Manual edit without updating the digest".to_string();
        assert!(store.upsert_distilled(invalid, Some(("canonical".to_string(), 1))).is_err());
        let replacement = distilled_test_entry("other", "Another note", "source-2");
        assert!(store.upsert_distilled(replacement.clone(), None).is_err());
        assert!(store.upsert_distilled(replacement, Some(("missing".to_string(), 1))).is_err());
        let mut other_scope = original.clone();
        other_scope.distilled.as_mut().unwrap().scope = "/other-project".to_string();
        assert!(store.upsert_distilled(other_scope.clone(), Some(("canonical".to_string(), 1))).is_err());
        assert!(store.upsert_distilled(other_scope, None).is_err());
        let mut other_owner = original.clone();
        other_owner.owner_pid = Some(u64::MAX);
        assert!(store.upsert_distilled(other_owner, Some(("canonical".to_string(), 1))).is_err());
        assert_eq!(fs::read(store.path()).unwrap(), stable_bytes);
        let _ = fs::remove_file(store.path());
    }

    #[test]
    fn distilled_revision_reads_only_verified_current_scope_and_keeps_archive() {
        let store = distilled_test_store("canonical");
        let original = distilled_test_entry("canonical", "Current note", "source-1");
        store.upsert_distilled(original.clone(), None).unwrap();
        let archive = store.path().with_extension("jsonl.1");
        let archived = serde_json::to_string(&original).unwrap();
        fs::write(&archive, format!("{archived}\n")).unwrap();
        let archive_bytes = fs::read(&archive).unwrap();
        let changed = distilled_test_entry("ignored", "New current note", "source-2");
        store.upsert_distilled(changed, Some(("canonical".to_string(), 1))).unwrap();
        assert_eq!(fs::read(&archive).unwrap(), archive_bytes);
        let mut invalid = distilled_test_entry("manual", "Verified then edited", "source-3");
        invalid.note = "Manual edit".to_string();
        store.apply_batch_update(&[], &[invalid]).unwrap();
        let mut other_scope = distilled_test_entry("other-project", "Other project note", "source-4");
        other_scope.distilled.as_mut().unwrap().scope = "/other-project".to_string();
        store.upsert_distilled(other_scope, None).unwrap();
        assert_eq!(store.active_distilled_entries("/project").unwrap().len(), 1);
        assert_eq!(store.active_distilled_entries("/other-project").unwrap().len(), 1);
        store.apply_batch_update_while_locked(&["canonical"], &[], false).unwrap();
        assert!(store.active_distilled_entries("/project").unwrap().is_empty());
        assert!(store.upsert_distilled(original, Some(("canonical".to_string(), 1))).is_err());
        assert_eq!(fs::read(&archive).unwrap(), archive_bytes);
        let _ = fs::remove_file(store.path());
        let _ = fs::remove_file(archive);
    }

    #[test]
    fn distilled_revision_concurrent_compare_and_swap_has_one_winner() {
        let store = distilled_test_store("concurrent");
        store.upsert_distilled(distilled_test_entry("canonical", "Original", "source-1"), None).unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = (0..2).map(|index| {
            let barrier = barrier.clone();
            let path = store.path().to_path_buf();
            std::thread::spawn(move || {
                let store = MemoryStore::for_tests_with_path(path);
                let entry = distilled_test_entry("ignored", &format!("Revision {index}"), &format!("source-{index}"));
                barrier.wait();
                store.upsert_distilled(entry, Some(("canonical".to_string(), 1)))
            })
        }).collect();
        let results: Vec<_> = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        // The loser can fail with a lock/IO error under load, not only with the
        // Stale message; only the one-winner invariant and the revision bump
        // are under test, so any single error satisfies the loser assertion.
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        assert_eq!(store.active_distilled_entries("/project").unwrap()[0].distilled.as_ref().unwrap().revision, 2);
        let _ = fs::remove_file(store.path());
    }

    #[test]
    fn test_search_recall_ngram() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e1 = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            category: "log".to_string(),
            note: "parsing login error occurred".to_string(),
            distilled: None,
            tags: vec!["auth".to_string()],
            source: Some("svc".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let e2 = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-02T00:00:00Z".to_string(),
            category: "info".to_string(),
            note: "user profile updated".to_string(),
            distilled: None,
            tags: vec!["user".to_string()],
            source: Some("svc".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e1).unwrap();
        store.append(&e2).unwrap();
        let out = store.search("parse login", 5).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("parsing login")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_synonym_login() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_syn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-03T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "user login failed due to authentication error".to_string(),
            distilled: None,
            tags: vec!["login".to_string()],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e).unwrap();
        let out = store.search("signin failure", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("login failed")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_search_recall_chinese_login_variants() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_cn_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let e = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-04T00:00:00Z".to_string(),
            category: "auth".to_string(),
            note: "登录失败，密码错误".to_string(),
            distilled: None,
            tags: vec!["登录".to_string()],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        store.append(&e).unwrap();
        let out = store.search("登陆失败", 3).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().any(|(x, _)| x.note.contains("登录失败")));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_dedup_scans_tail_window_across_bad_lines() {
        let path = std::env::temp_dir().join(format!(
            "rt_mem_tail_window_{}.jsonl",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let mk = |note: &str| AgentMemoryEntry {
            distilled: None,
            id: None,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            category: "self_note".to_string(),
            note: note.to_string(),
            tags: Vec::new(),
            source: Some("session:test".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        // Write the JSONL directly with trailing corrupt/blank lines to cover
        // the backward scan's tolerance for bad tail lines.
        let mut buf = String::new();
        for i in 0..5 {
            buf.push_str(&serde_json::to_string(&mk(&format!("note-{i}"))).unwrap());
            buf.push('\n');
        }
        buf.push_str("not-json\n\n");
        std::fs::write(&path, buf).unwrap();

        // Duplicate of the newest good entry: must hit the tail-window dedup.
        store.append(&mk("note-4")).unwrap();
        // Duplicate of an older-but-still-in-window entry: deduped too.
        store.append(&mk("note-0")).unwrap();
        // A genuinely new note must be appended.
        store.append(&mk("brand-new")).unwrap();

        let recent = store.recent(20).unwrap();
        assert_eq!(recent.iter().filter(|e| e.note == "note-4").count(), 1);
        assert_eq!(recent.iter().filter(|e| e.note == "note-0").count(), 1);
        assert!(recent.iter().any(|e| e.note == "brand-new"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn learning_entries_deduplicate_recent_exact_writes() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_dedup_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());

        let entry = AgentMemoryEntry {
            id: None,
            timestamp: "2025-01-05T00:00:00Z".to_string(),
            category: "self_note".to_string(),
            note: "Do: verify before write".to_string(),
            distilled: None,
            tags: vec!["agent".to_string()],
            source: Some("session:test".to_string()),
            priority: Some(120),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };

        store.append(&entry).unwrap();
        store.append(&entry).unwrap();

        let recent = store.recent(10).unwrap();
        assert_eq!(
            recent
                .iter()
                .filter(|e| e.category == "self_note" && e.note == "Do: verify before write")
                .count(),
            1
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn idempotent_knowledge_write_returns_existing_entry_without_appending() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_mem_knowledge_dedup_{ts}.jsonl"));
        let store = MemoryStore::for_tests_with_path(path.clone());
        let first = AgentMemoryEntry {
            id: Some("mem_existing".to_string()),
            timestamp: "2025-01-05T00:00:00Z".to_string(),
            category: "user_memory".to_string(),
            note: "Keep project decisions in the architecture log.".to_string(),
            distilled: None,
            tags: vec!["architecture".to_string(), "decision".to_string()],
            source: Some("project:demo".to_string()),
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let retry = AgentMemoryEntry {
            id: Some("mem_retry".to_string()),
            timestamp: "2025-01-06T00:00:00Z".to_string(),
            category: " USER_MEMORY ".to_string(),
            note: "  keep project decisions in the architecture log.  ".to_string(),
            distilled: None,
            tags: vec!["decision".to_string(), "ARCHITECTURE".to_string()],
            source: Some(" PROJECT:DEMO ".to_string()),
            priority: Some(200),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };

        assert_eq!(
            store.append_idempotent_knowledge(&first).unwrap(),
            KnowledgeAppendOutcome::Appended
        );
        assert_eq!(
            store.append_idempotent_knowledge(&retry).unwrap(),
            KnowledgeAppendOutcome::Duplicate {
                existing_id: Some("mem_existing".to_string())
            }
        );
        assert_eq!(store.recent(10).unwrap().len(), 1);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db"));
    }

    #[test]
    fn delete_subagent_memory_removes_only_its_knowledge_dedup_cache_entry() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("rt_mem_cache_lifecycle_{ts}"));
        std::fs::create_dir_all(&dir).unwrap();
        let removed_path = dir.join("agent_memory.subagent-removed.jsonl");
        let retained_path = dir.join("agent_memory.subagent-retained.jsonl");
        let entry = AgentMemoryEntry {
            id: Some("mem_cache_lifecycle".to_string()),
            timestamp: "2025-01-05T00:00:00Z".to_string(),
            category: "user_memory".to_string(),
            note: "Cache lifecycle test entry.".to_string(),
            distilled: None,
            tags: vec!["test".to_string()],
            source: Some("test".to_string()),
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };

        for path in [&removed_path, &retained_path] {
            let store = MemoryStore::for_tests_with_path(path.clone());
            assert_eq!(
                store.append_idempotent_knowledge(&entry).unwrap(),
                KnowledgeAppendOutcome::Appended
            );
        }
        {
            let cache = KNOWLEDGE_DEDUP_CACHE.lock().unwrap();
            assert!(cache.contains_key(&removed_path));
            assert!(cache.contains_key(&retained_path));
        }

        crate::ai::history::delete_subagent_memory(&removed_path).unwrap();

        {
            let cache = KNOWLEDGE_DEDUP_CACHE.lock().unwrap();
            assert!(!cache.contains_key(&removed_path));
            assert!(cache.contains_key(&retained_path));
        }

        crate::ai::history::delete_subagent_memory(&retained_path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_batch_update_rewrites_delete_and_append_in_one_pass() {
        let path = std::env::temp_dir().join(format!(
            "rt_mem_batch_update_{}.jsonl",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let entry_with_id = |id: &str, note: &str, ts: &str| AgentMemoryEntry {
            distilled: None,
            id: Some(id.to_string()),
            timestamp: ts.to_string(),
            category: "user_memory".to_string(),
            note: note.to_string(),
            tags: Vec::new(),
            source: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let write_lines = |entries: &[AgentMemoryEntry]| {
            let mut buf = String::new();
            for entry in entries {
                buf.push_str(&serde_json::to_string(entry).unwrap());
                buf.push('\n');
            }
            std::fs::write(&path, buf).unwrap();
        };
        let read_entries = || -> Vec<AgentMemoryEntry> {
            std::fs::read_to_string(&path)
                .unwrap_or_default()
                .lines()
                .filter_map(|line| serde_json::from_str::<AgentMemoryEntry>(line.trim()).ok())
                .collect()
        };
        write_lines(&[
            entry_with_id("mem_1", "keep me", "2025-01-01T00:00:00Z"),
            entry_with_id("mem_2", "drop me", "2025-01-01T00:00:01Z"),
            entry_with_id("mem_3", "merge me", "2025-01-01T00:00:02Z"),
        ]);

        let store = MemoryStore::for_tests_with_path(path.clone());
        let merged = entry_with_id("mem_merged", "merged note", "2025-01-02T00:00:00Z");
        let report = store
            .apply_batch_update(&["mem_2", "mem_3"], &[merged.clone()])
            .unwrap();

        assert_eq!(
            report,
            MemoryBatchUpdateReport {
                deleted: 2,
                appended: 1
            }
        );

        let kept = read_entries();
        assert_eq!(kept.len(), 2);
        assert!(
            kept.iter()
                .any(|entry| entry.id.as_deref() == Some("mem_1"))
        );
        assert!(
            kept.iter()
                .any(|entry| entry.id.as_deref() == Some("mem_merged"))
        );
        assert!(
            !kept
                .iter()
                .any(|entry| entry.id.as_deref() == Some("mem_2"))
        );
        assert!(
            !kept
                .iter()
                .any(|entry| entry.id.as_deref() == Some("mem_3"))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_batch_update_deletes_across_rotation_archives() {
        let dir = std::env::temp_dir().join(format!(
            "rt_mem_archive_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let current = dir.join("agent_memory.jsonl");
        let archive = dir.join("agent_memory.jsonl.20260101000000");
        let entry_with_id = |id: &str, note: &str, ts: &str| AgentMemoryEntry {
            distilled: None,
            id: Some(id.to_string()),
            timestamp: ts.to_string(),
            category: "user_memory".to_string(),
            note: note.to_string(),
            tags: Vec::new(),
            source: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let write_lines = |path: &std::path::Path, entries: &[AgentMemoryEntry]| {
            let mut buf = String::new();
            for entry in entries {
                buf.push_str(&serde_json::to_string(entry).unwrap());
                buf.push('\n');
            }
            std::fs::write(path, buf).unwrap();
        };
        let read_ids = |path: &std::path::Path| -> Vec<String> {
            std::fs::read_to_string(path)
                .unwrap_or_default()
                .lines()
                .filter_map(|line| serde_json::from_str::<AgentMemoryEntry>(line.trim()).ok())
                .filter_map(|entry| entry.id)
                .collect()
        };
        // Current file: keep cur_a, delete cur_b; archives: delete arch_c, keep arch_d.
        write_lines(
            &current,
            &[
                entry_with_id("cur_a", "keep me", "2025-01-01T00:00:00Z"),
                entry_with_id("cur_b", "drop me", "2025-01-01T00:00:01Z"),
            ],
        );
        write_lines(
            &archive,
            &[
                entry_with_id("arch_c", "drop in archive", "2025-01-01T00:00:02Z"),
                entry_with_id("arch_d", "keep in archive", "2025-01-01T00:00:03Z"),
            ],
        );

        let store = MemoryStore::for_tests_with_path(current.clone());
        let merged = entry_with_id("merged_1", "merged note", "2025-01-02T00:00:00Z");
        let report = store
            .apply_batch_update(&["cur_b", "arch_c"], &[merged.clone()])
            .unwrap();

        assert_eq!(
            report,
            MemoryBatchUpdateReport {
                deleted: 2,
                appended: 1
            }
        );
        assert_eq!(
            read_ids(&current),
            vec!["cur_a".to_string(), "merged_1".to_string()]
        );
        assert_eq!(read_ids(&archive), vec!["arch_d".to_string()]);

        // all_with_archives must see entries from both the main file and the rotated archives.
        let all_ids: Vec<String> = store
            .all_with_archives()
            .unwrap()
            .into_iter()
            .filter_map(|entry| entry.id)
            .collect();
        assert_eq!(all_ids.len(), 3);
        assert!(all_ids.contains(&"cur_a".to_string()));
        assert!(all_ids.contains(&"arch_d".to_string()));
        assert!(all_ids.contains(&"merged_1".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }
