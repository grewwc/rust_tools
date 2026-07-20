    use super::{render_text_attachment_block, text_file_contents};
    use std::fs;
    use std::path::PathBuf;

    fn make_temp_path(name: &str, ext: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ai-attachment-{}-{}.{}",
            name,
            uuid::Uuid::new_v4(),
            ext
        ));
        path
    }

    #[test]
    fn attachment_block_keeps_file_boundaries_for_multiple_files() {
        let first = make_temp_path("first", "txt");
        let second = make_temp_path("second", "txt");
        fs::write(&first, "alpha").unwrap();
        fs::write(&second, "beta").unwrap();

        let rendered = text_file_contents(&[
            first.to_string_lossy().to_string(),
            second.to_string_lossy().to_string(),
        ])
        .unwrap();

        assert!(rendered.contains(&format!("[Attached text file: {}]", first.display())));
        assert!(rendered.contains(&format!("[Attached text file: {}]", second.display())));
        assert!(rendered.contains("[/Attached text file]"));

        let _ = fs::remove_file(first);
        let _ = fs::remove_file(second);
    }

    #[test]
    fn attachment_block_truncates_large_files_and_points_to_read_file() {
        let path = make_temp_path("large", "rs");
        let content = (1..=400)
            .map(|idx| format!("fn item_{idx}() {{}}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, &content).unwrap();

        let rendered = render_text_attachment_block(path.to_string_lossy().as_ref()).unwrap();

        assert!(
            rendered.contains("Attachment preview only"),
            "rendered: {rendered}"
        );
        assert!(rendered.contains("read_file("), "rendered: {rendered}");
        assert!(rendered.contains("Symbol outline"), "rendered: {rendered}");

        let _ = fs::remove_file(path);
    }
