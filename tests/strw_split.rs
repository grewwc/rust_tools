use rust_tools::strw::split::{split_keep_symbol, split_space_keep_symbol};


#[test]
fn test_split_keep_quotes_simple() {
    let input = r#"hello "world with spaces" test"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", "\"world with spaces\"", "test"]);
}

#[test]
fn test_split_keep_symbol_iterator() {
    let input = r#"hello "world with spaces" test"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", "\"world with spaces\"", "test"]);
}

#[test]
fn test_multiple_quotes() {
    let input = r#"command "arg one" "arg two" final"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["command", "\"arg one\"", "\"arg two\"", "final"]);
}

#[test]
fn test_no_quotes() {
    let input = "hello world test";
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", "world", "test"]);
}

#[test]
fn test_empty_quotes() {
    let input = r#"hello "" world"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", "\"\"", "world"]);
}

#[test]
fn test_empty_split_chars() {
    let input = "hello world";
    let result = split_keep_symbol(input, "", "").collect::<Vec<&str>>();
    assert_eq!(result, vec!["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]);
}

#[test]
fn test_split_chinese() {
    let s = "feature/20251103_27134199_fix_console_count_invoke_1 2025-11-03 fix:控制台计量统计问题";
    let s = "feature/20251030_27098361_fix_security_issues_1 2025-10-30 fix:校验是否是合法的ossendpoin";
    let s = "feature/20251030_27098361_fix_security_issues_1 2025-10-30 fix:校验是否是合法的ossendpoint";
    let result = split_space_keep_symbol(s, "\"");
    println!("result, {}", result.collect::<String>());
}

#[test]
fn test_nested_quotes() {
    let input = r#"outer "inner 'nested' quotes" end"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#""'"#).collect();
    assert_eq!(result, vec!["outer", "\"inner 'nested' quotes\"", "end"]);
}

#[test]
fn test_single_quotes() {
    let input = "hello 'world with spaces' test";
    let result: Vec<&str> = split_keep_symbol(input, " ", "'").collect();
    assert_eq!(result, vec!["hello", "'world with spaces'", "test"]);
}

#[test]
fn test_multiple_whitespace_types() {
    let input = "hello\t\"world\ntest\"\r\nend";
    let result: Vec<&str> = split_space_keep_symbol(input, r#"""#).collect();
    assert_eq!(result, vec!["hello", "\"world\ntest\"", "end"]);
}

#[test]
fn test_consecutive_whitespace() {
    let input = "hello    \"world\"    test";
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", r#""world""#, "test"]);
}

#[test]
fn test_leading_trailing_whitespace() {
    let input = "   hello \"world\" test   ";
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", r#""world""#, "test"]);
}

#[test]
fn test_only_quoted_string() {
    let input = r#""hello world""#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec![r#""hello world""#]);
}

#[test]
fn test_unclosed_quote() {
    let input = r#"hello "world test"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", "\"world test"]);
}

#[test]
fn test_empty_string() {
    let input = "";
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, Vec::<&str>::new());
}

#[test]
fn test_only_whitespace() {
    let input = "    ";
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, Vec::<&str>::new());
}

#[test]
fn test_mixed_symbols() {
    let input = r#"cmd [arg1] "arg 2" 'arg 3'"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#""'[]"#).collect();
    assert_eq!(result, vec!["cmd", "[arg1]", "\"arg 2\"", "'arg 3'"]);
}

#[test]
fn test_special_characters_in_quotes() {
    let input = r#"echo "hello@#$%^&*()world""#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["echo", "\"hello@#$%^&*()world\""]);
}

#[test]
fn test_unicode_in_quotes() {
    let input = r#"hello "世界 🌍 مرحبا" test"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["hello", "\"世界 🌍 مرحبا\"", "test"]);
}

#[test]
fn test_comma_as_separator() {
    let input = r#"apple,"banana split",cherry"#;
    let result: Vec<&str> = split_keep_symbol(input, ",", r#"""#).collect();
    assert_eq!(result, vec!["apple", r#""banana split""#, "cherry"]);
}

#[test]
fn test_multiple_separators() {
    let input = "a,b;c:d";
    let result: Vec<&str> = split_keep_symbol(input, ",;:", "").collect();
    assert_eq!(result, vec!["a", "b", "c", "d"]);
}

#[test]
fn test_quote_at_start_and_end() {
    let input = r#""start" middle "end""#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec![r#""start""#, "middle", r#""end""#]);
}

#[test]
fn test_adjacent_quoted_strings() {
    let input = r#""hello""world""#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec![r#""hello""world""#]);
}

#[test]
fn test_backslash_in_quotes() {
    let input = r#"test "path\\to\\file" end"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec!["test", r#""path\\to\\file""#, "end"]);
}

#[test]
fn test_alternating_quotes() {
    let input = r#""one" two "three" four"#;
    let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
    assert_eq!(result, vec![r#""one""#, "two", r#""three""#, "four"]);
}

