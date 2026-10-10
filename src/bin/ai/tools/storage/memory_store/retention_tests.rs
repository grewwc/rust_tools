    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
use std::path::{Path, PathBuf};

    fn unique_path(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("rt_mem_retention_{tag}_{nanos}.jsonl"))
    }

    fn entry(category: &str, note: &str, ts: &str, priority: u8) -> AgentMemoryEntry {
        AgentMemoryEntry {
            distilled: None,
            id: None,
            timestamp: ts.to_string(),
            category: category.to_string(),
            note: note.to_string(),
            tags: vec![],
            source: None,
            priority: Some(priority),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        }
    }

    fn entry_with_id(
        id: &str,
        category: &str,
        note: &str,
        ts: &str,
        priority: u8,
    ) -> AgentMemoryEntry {
        let mut entry = entry(category, note, ts, priority);
        entry.id = Some(id.to_string());
        entry
    }

    fn write_lines(path: &Path, entries: &[AgentMemoryEntry]) {
        let mut buf = String::new();
        for e in entries {
            buf.push_str(&serde_json::to_string(e).unwrap());
            buf.push('\n');
        }
        std::fs::write(path, buf).unwrap();
    }

    fn read_entries(path: &Path) -> Vec<AgentMemoryEntry> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<AgentMemoryEntry>(l.trim()).ok())
            .collect()
    }

    /// P0-1 regression: the original double-delete bug wrongly removed the entry right after a priority=255 one once the quota was full;
    /// build a "low-priority + permanent" mix here and assert that after enforce all priority=255
    /// entries survive and the quota is pressed back under max_entries.
    #[test]
    fn prune_low_value_removes_matching_ids_in_one_pass() {
        let path = unique_path("prune_batch");
        let mut all = Vec::new();
        // Old low-priority entries: must be pruned.
        for i in 0..3 {
            all.push(entry_with_id(
                &format!("old-{i}"),
                "tool_stat",
                &format!("old-{i}"),
                "2025-01-01T00:00:00Z",
                50,
            ));
        }
        // Old but permanent: must survive regardless of age.
        all.push(entry_with_id(
            "perm",
            "safety_rules",
            "keep",
            "2025-01-01T00:00:00Z",
            255,
        ));
        // Low priority but too recent: must survive.
        all.push(entry_with_id(
            "fresh",
            "tool_stat",
            "fresh",
            "2099-01-01T00:00:00Z",
            50,
        ));
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        // tool_stat with no general tags scores 0.1 < 0.2; safety_rules and
        // priority >= 200 are exempt, so exactly the three old-* entries go.
        let removed = store.prune_low_value_memories(0.2, 365).unwrap();
        assert_eq!(removed, 3);

        let kept = read_entries(&path);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|e| e.id.as_deref() == Some("perm")));
        assert!(kept.iter().any(|e| e.id.as_deref() == Some("fresh")));
        assert!(
            kept.iter()
                .all(|e| !e.id.as_deref().unwrap_or("").starts_with("old-"))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn enforce_max_entries_keeps_all_permanent_entries() {
        let path = unique_path("enforce_perm");
        let mut all = Vec::new();
        // 100 ordinary low-priority entries, ordered old to new
        for i in 0..100 {
            all.push(entry(
                "tool_stat",
                &format!("note-{i}"),
                &format!("2025-01-01T00:00:{:02}Z", i % 60),
                50,
            ));
        }
        // 5 permanent entries (safety_rules)
        for i in 0..5 {
            all.push(entry(
                "safety_rules",
                &format!("perm-{i}"),
                &format!("2025-02-01T00:00:{:02}Z", i),
                255,
            ));
        }
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        store.enforce_max_entries(50, 10).unwrap();

        let kept = read_entries(&path);
        assert!(
            kept.len() <= 50,
            "expected <=50 entries, got {}",
            kept.len()
        );
        let perm_kept = kept.iter().filter(|e| e.priority == Some(255)).count();
        assert_eq!(perm_kept, 5, "all permanent entries must survive");

        let _ = std::fs::remove_file(&path);
    }

    /// P0-2 regression: after rotate the new main file may contain only priority=255 entries,
    /// while the archive file contains all original entries.
    #[test]
    fn rotate_preserves_permanent_entries_in_main_file() {
        let path = unique_path("rotate_perm");
        let mut all = Vec::new();
        // Inflate the file to a few KB with a large enough note
        let big = "x".repeat(2048);
        for i in 0..20 {
            all.push(entry(
                "tool_cache",
                &format!("{}-{}", big, i),
                &format!("2025-01-01T00:00:{:02}Z", i),
                80,
            ));
        }
        all.push(entry(
            "safety_rules",
            "do not run rm -rf /",
            "2025-02-02T00:00:00Z",
            255,
        ));
        all.push(entry(
            "self_note",
            "always read before edit",
            "2025-02-02T00:00:01Z",
            255,
        ));
        write_lines(&path, &all);

        let store = MemoryStore::for_tests_with_path(path.clone());
        // Threshold deliberately smaller than the current size to force a rotate
        let rotated = store.rotate_if_exceeds(1024).unwrap();
        assert!(rotated, "expected rotate to happen");

        // Main file keeps only permanent entries
        let head = read_entries(&path);
        assert_eq!(head.len(), 2);
        assert!(head.iter().all(|e| e.priority == Some(255)));

        // Archive file exists and contains all original entries
        let parent = path.parent().unwrap();
        let archives: Vec<_> = std::fs::read_dir(parent)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let head_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.starts_with(head_name) && p != &path
            })
            .collect();
        assert_eq!(archives.len(), 1, "expected exactly one archive");
        let archived = read_entries(&archives[0]);
        assert_eq!(archived.len(), all.len());

        let _ = std::fs::remove_file(&path);
        for a in archives {
            let _ = std::fs::remove_file(a);
        }
    }

    /// P0 regression: -ns memo retrieval (include_archives=true) must scan all archives,
    /// not truncated by keep_last_archives; it must also handle `.jsonl.bak` files left by legacy migration.
    /// Otherwise historical memos moved out by rotation or migration become permanently unretrievable.
    #[test]
    fn entries_by_category_include_archives_scans_all_archives() {
        let path = unique_path("memo_arch_scan");
        let parent = path.parent().unwrap();
        std::fs::create_dir_all(parent).unwrap();
        let base = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();
        let legacy_base = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();

        // Main file: 1 memo
        write_lines(
            &path,
            &[entry("memo", "main memo", "2026-07-16T00:00:00Z", 100)],
        );

        // Create 5 archive files (beyond the default keep_last_archives=3 window).
        // Put "二次分析" in the oldest archive, simulating a user record moved out by rotation.
        let archive_notes = [
            "oldest: 二次分析问题排查",
            "archive2 memo",
            "archive3 memo",
            "archive4 memo",
            "archive5 memo",
        ];
        let mut archive_paths = Vec::new();
        for (i, note) in archive_notes.iter().enumerate() {
            let ap = parent.join(format!("{base}.2026070{}170000", i + 1));
            write_lines(&ap, &[entry("memo", note, "2026-07-01T00:00:00Z", 100)]);
            // Set increasing mtimes so ordering is deterministic (oldest -> newest)
            let times = std::fs::FileTimes::new();
            let _ = times.set_modified(
                UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000 + i as u64),
            );
            std::fs::File::open(&ap).unwrap().set_times(times).unwrap();
            archive_paths.push(ap);
        }

        // Older versions left pre-migration data in this naming format; it is not an ordinary rotation archive.
        let legacy_path = parent.join(format!(
            "{legacy_base}.legacy-migrate-20260701180745.jsonl.bak"
        ));
        write_lines(
            &legacy_path,
            &[entry(
                "memo",
                "legacy: 二次分析问题排查 mysql",
                "2026-06-30T00:00:00Z",
                100,
            )],
        );

        let store = MemoryStore::for_tests_with_path(path.clone());
        let memos = store.entries_by_category("memo", 100_000, true).unwrap();

        // Main file 1 + normal archives 5 + legacy migration backup 1 = 7 entries.
        assert_eq!(
            memos.len(),
            7,
            "include_archives=true 必须扫描全部归档及旧迁移备份，不应被 keep_last 截断"
        );
        assert!(
            memos.iter().any(|m| m.note.contains("二次分析")),
            "最旧归档和旧迁移备份中的 memo 都必须可检索"
        );

        // Cleanup
        let _ = std::fs::remove_file(&path);
        for p in archive_paths {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_file(legacy_path);
    }

    #[test]
    fn entries_by_category_current_file_ignores_global_archive_search() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        let cfg_path = unique_path("current_only_config");
        std::fs::write(
            &cfg_path,
            "ai.memory.search_archives.enable = true\nai.memory.search_archives.keep_last = 10\n",
        )
        .unwrap();
        let old_cfg = std::env::var_os("CONFIGW_PATH");
        unsafe { std::env::set_var("CONFIGW_PATH", &cfg_path) };
        crate::commonw::configw::refresh();

        let path = unique_path("current_only");
        let parent = path.parent().unwrap();
        std::fs::create_dir_all(parent).unwrap();
        let base = path.file_name().and_then(|n| n.to_str()).unwrap();
        let archive_path = parent.join(format!("{base}.20260729120000"));

        write_lines(
            &path,
            &[entry("memo", "main memo", "2026-07-29T00:00:00Z", 100)],
        );
        write_lines(
            &archive_path,
            &[entry("memo", "archive memo", "2026-07-28T00:00:00Z", 100)],
        );

        let store = MemoryStore::for_tests_with_path(path.clone());
        let configured_scan = store.entries_by_category("memo", 10, false).unwrap();
        let current_only = store.entries_by_category_current_file("memo", 10).unwrap();

        match old_cfg {
            Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();

        let _ = std::fs::remove_file(cfg_path);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(archive_path);

        assert_eq!(configured_scan.len(), 2);
        assert_eq!(current_only.len(), 1);
        assert_eq!(current_only[0].note, "main memo");
    }
