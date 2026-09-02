//! File suffix / MIME-type helper tests.

use super::super::*;

#[test]
fn image_file_detection_by_suffix() {
    assert!(files::is_image_path("/tmp/hello.png"));
    assert!(files::is_image_path("/tmp/hello.JPEG"));
    assert!(!files::is_image_path("/tmp/hello.pdf"));
}

#[test]
fn image_mime_type_matches_suffix() {
    assert_eq!(files::image_mime_type("a.png"), "image/png");
    assert_eq!(files::image_mime_type("a.jpg"), "image/jpeg");
    assert_eq!(files::image_mime_type("a.unknown"), "image/jpeg");
}
