use rust_tools::strw::{calc, indices, search, split};

#[test]
fn test_indices_find_all_and_substring() {
    assert_eq!(indices::find_all("abababa", "aba"), vec![0, 2, 4]);
    assert_eq!(indices::substring_quiet("abcdef", 1, 4), "bcd");
}

#[test]
fn test_search_helpers() {
    assert!(search::contains("hello world", "world"));
    assert!(search::any_equals("a", &["b", "a"]));
    assert!(search::all_contains("a b c", &["a", "b", "c"]));
    assert_eq!(search::find_first_substr("x=1;y=2", true, &["="]), 2);
}

#[test]
fn test_split_extras() {
    let s = r#"a,"b,c",d"#;
    assert_eq!(
        split::replace_all_in_quote_unchange(s, ',', ';'),
        r#"a;"b,c";d"#
    );
    assert_eq!(
        split::replace_all_out_quote_unchange(s, ',', ';'),
        r#"a,"b;c",d"#
    );
}

#[test]
fn test_calc_basic() {
    assert_eq!(calc::plus("0.3", "0.513"), "0.813");
    assert_eq!(calc::minus("1", "0.3"), "0.7");
    assert_eq!(calc::mul("0.3", "0.02"), "0.006");
    assert_eq!(calc::div("1", "8", 4), "0.125");
    assert_eq!(calc::modulo("10", "3"), "1");
}
