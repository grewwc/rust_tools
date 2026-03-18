use rust_tools::strw::{check, format, r#move, split};

#[test]
fn test_check_blank() {
    assert!(check::is_blank(""));
    assert!(check::is_blank(" \t\n"));
    assert!(!check::is_blank("x"));
    assert!(check::all_blank(&["", " "]));
    assert!(!check::all_blank(&["", "x"]));
    assert!(check::any_blank(&["x", " "]));
}

#[test]
fn test_split_no_empty() {
    assert_eq!(split::split_no_empty("", ","), Vec::<&str>::new());
    assert_eq!(split::split_no_empty("a,,b", ","), vec!["a", "b"]);
}

#[test]
fn test_split_by_cutset() {
    assert_eq!(
        split::split_by_cutset("a,b;c:d", ",;:"),
        vec!["a", "b", "c", "d"]
    );
}

#[test]
fn test_split_by_str_keep_quotes() {
    let s = r#"apple,"banana split",cherry"#;
    let parts = split::split_by_str_keep_quotes(s, ",", "\"", true);
    assert_eq!(parts, vec!["apple", r#""banana split""#, "cherry"]);

    let parts = split::split_by_str_keep_quotes(s, ",", "\"", false);
    assert_eq!(parts, vec!["apple", "banana split", "cherry"]);
}

#[test]
fn test_move_and_reverse() {
    assert_eq!(r#move::move_to_end_all("a--b--c", "--"), "abc----");
    assert_eq!(r#move::reverse("abc"), "cba");
}

#[test]
fn test_wrap_and_format_int64() {
    assert_eq!(format::wrap("hello world", 5, 0, " "), "hello\nworld\n");
    assert_eq!(format::format_int64(1024), "1.0K");
}
