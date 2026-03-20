use std::sync::Arc;

use rust_tools::terminalw;

#[test]
fn test_parser_basic_contains_and_positional() {
    let mut p = terminalw::Parser::new();
    p.add_bool("v", false, "");
    p.add_string("f", "", "");
    p.add_bool("a", false, "");
    p.parse_args("-f file -a something night -v", &[]);

    assert!(p.contains_flag("v"));
    assert!(p.contains_flag("-v"));
    assert!(p.contains_flag("a"));
    assert!(p.contains_flag("f"));

    assert_eq!(p.positional.to_vec(), vec!["something", "night"]);
}

#[test]
fn test_parser_combined_bool_flags_split() {
    let mut p = terminalw::Parser::new();
    p.add_bool("l", false, "");
    p.add_bool("r", false, "");
    p.add_bool("t", false, "");
    p.parse_args("-lrt", &[]);
    assert!(p.contains_flag_strict("l"));
    assert!(p.contains_flag_strict("r"));
    assert!(p.contains_flag_strict("t"));
}

#[test]
fn test_parser_alias_and_long_flag_normalization() {
    let mut p = terminalw::Parser::new();
    p.add_string("search", "", "");
    p.alias("search", "q");
    p.parse_args("--q hello", &[]);
    assert_eq!(p.flag_value_with_default("search", ""), "hello");
    assert_eq!(p.flag_value_with_default("q", ""), "hello");
}

#[test]
fn test_parser_num_args_and_exclude_positional() {
    let mut p = terminalw::Parser::new();
    p.add_bool("v", false, "");
    p.parse_args("-10 -v foo", &[]);
    assert_eq!(p.num_args(), 10);
    let pos = p.positional_args(true);
    assert_eq!(pos, vec!["foo"]);
}

#[test]
fn test_on_execute_runs_actions() {
    let mut p = terminalw::Parser::new();
    p.add_bool("v", false, "");
    p.parse_args("-v", &[]);
    let hit = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hit2 = Arc::clone(&hit);
    p.on(move |pp| pp.contains_flag_strict("v"))
        .do_action(move || {
            hit2.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    p.execute();
    assert!(hit.load(std::sync::atomic::Ordering::Relaxed));
}
