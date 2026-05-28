use std::path::Path;

use clap::Parser;
use rust_tools::{clipboardw::string_content, jsonw};
use serde_json::Value;

#[derive(Parser)]
#[command(about = "JSON diff/format utilities (go_tools jsondiff compatible subset)")]
struct Cli {
    #[arg(short = 'f', value_name = "FILE", num_args = 0..=1, default_missing_value = "")]
    format: Option<String>,

    #[arg(short = 'o', default_value = "", value_name = "FILE")]
    output: String,

    #[arg(long, default_value_t = false, help = "sort arrays before diff")]
    sort: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "multi-thread (accepted, currently ignored)"
    )]
    mt: bool,

    #[arg(short = 'p', default_value_t = false, help = "print result to stdout")]
    print: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "escape clipboard/stdin as JSON string"
    )]
    quote: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "remove \\n and \\r when quoting/printing"
    )]
    oneline: bool,

    #[arg(long, default_value_t = false, help = "print JSON length in clipboard")]
    len: bool,

    #[arg(value_name = "OLD NEW", num_args = 0..=2)]
    files: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    if cli.quote {
        let mut content = string_content::get_clipboard_content();
        if cli.oneline {
            content = content.replace(['\n', '\r'], "");
        }
        if content.is_empty() {
            content = read_stdin_all();
        }
        let quoted = serde_json::to_string(&content).unwrap_or_default();
        println!("{quoted}");
        return;
    }

    if cli.oneline && cli.format.is_none() && !cli.len && cli.files.is_empty() {
        let content = string_content::get_clipboard_content().replace(['\n', '\r'], "");
        println!("{content}");
        return;
    }

    if cli.len {
        let content = string_content::get_clipboard_content();
        match jsonw::Json::from_str(&content, jsonw::ParseOptions::default()) {
            Ok(j) => {
                let msg = if j.is_array() { "Array" } else { "Object" };
                println!("{msg}: {}", j.len());
            }
            Err(_) => {
                println!("clipboard content is not a valid json");
                std::process::exit(1);
            }
        }
        return;
    }

    if let Some(fname) = cli.format.as_deref() {
        let options = jsonw::ParseOptions::default();
        let j = if fname.is_empty() {
            parse_clipboard_json(options)
        } else {
            expand_nested_json_strings_in_json(
                jsonw::Json::from_file(fname, options).unwrap(),
                options,
            )
        };

        let mut formatted = j.to_pretty_string();
        if cli.oneline {
            formatted = formatted.split_whitespace().collect::<String>();
        }
        println!("{}", formatted.chars().take(1024).collect::<String>());

        let output_fname = if fname.is_empty() {
            "_f.json".to_string()
        } else {
            format!("{}_f.json", base_no_ext(fname))
        };
        println!("write file to {output_fname}");
        j.to_file(output_fname, true).unwrap();
        return;
    }

    if cli.files.len() == 2 {
        let options = jsonw::ParseOptions::default();
        let old = jsonw::Json::from_file(&cli.files[0], options).unwrap();
        let new = jsonw::Json::from_file(&cli.files[1], options).unwrap();

        let diff = jsonw::diff_json(old.value(), new.value(), cli.sort);
        let diff_value = serde_json::to_value(diff).unwrap_or(Value::Null);
        let diff_json = jsonw::Json::new(diff_value);

        if cli.print {
            println!("{}", diff_json.to_pretty_string());
        }

        let fname = if cli.output.is_empty() {
            format!(
                "{}_{}_diff.json",
                base_no_ext(&cli.files[0]),
                base_no_ext(&cli.files[1])
            )
        } else {
            cli.output.clone()
        };
        diff_json.to_file(&fname, true).unwrap();
        println!("write to {fname}");
        return;
    }

    eprintln!("usage: j old.json new.json  |  j -f [file]  |  j --quote [--oneline]  |  j --len");
}

fn base_no_ext(path: &str) -> String {
    let p = Path::new(path);
    let file = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string();
    let mut name = file;
    if let Some(idx) = name.rfind('.') {
        name.truncate(idx);
    }
    name.replace(' ', "")
}

fn read_stdin_all() -> String {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .unwrap_or_default();
    buf
}

/// 从剪贴板解析 JSON，支持处理双重转义的 JSON 字符串
///
/// 该函数会尝试解析剪贴板内容。如果解析结果是一个字符串（说明剪贴板内容是
/// 一个 JSON 字符串，如 `"{\"key\":\"value\"}"`），则会尝试将该字符串再次
/// 解析为 JSON；同时也会递归扫描对象/数组中的字符串字段，把其中嵌套的
/// JSON 字符串继续展开。
fn parse_clipboard_json(options: jsonw::ParseOptions) -> jsonw::Json {
    let s = string_content::get_clipboard_content();
    let j = jsonw::Json::from_str(&s, options).unwrap_or_else(|e| {
        eprintln!("Failed to parse clipboard content as JSON: {e}");
        std::process::exit(1);
    });
    expand_nested_json_strings_in_json(j, options)
}

fn expand_nested_json_strings_in_json(
    j: jsonw::Json,
    options: jsonw::ParseOptions,
) -> jsonw::Json {
    jsonw::Json::new(expand_nested_json_strings(j.raw_value().clone(), options))
}

fn expand_nested_json_strings(value: Value, options: jsonw::ParseOptions) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, expand_nested_json_strings(v, options)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| expand_nested_json_strings(item, options))
                .collect(),
        ),
        Value::String(s) => try_parse_nested_json_string(&s, options)
            .map(|parsed| expand_nested_json_strings(parsed, options))
            .or_else(|| decode_json_escaped_text(&s).map(Value::String))
            .unwrap_or(Value::String(s)),
        other => other,
    }
}

fn try_parse_nested_json_string(s: &str, options: jsonw::ParseOptions) -> Option<Value> {
    let trimmed = s.trim();
    if !looks_like_json_string_payload(trimmed) {
        return None;
    }
    jsonw::Json::from_str(trimmed, options)
        .ok()
        .map(|j| j.raw_value().clone())
}

fn looks_like_json_string_payload(s: &str) -> bool {
    matches!(
        s.as_bytes().first().copied(),
        Some(b'{') | Some(b'[') | Some(b'"')
    )
}

fn decode_json_escaped_text(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut changed = false;

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }

        let Some(escaped) = chars.next() else {
            out.push('\\');
            break;
        };

        match escaped {
            '"' => {
                out.push('"');
                changed = true;
            }
            '\\' => {
                out.push('\\');
                changed = true;
            }
            '/' => {
                out.push('/');
                changed = true;
            }
            'b' => {
                out.push('\u{0008}');
                changed = true;
            }
            'f' => {
                out.push('\u{000c}');
                changed = true;
            }
            'n' => {
                out.push('\n');
                changed = true;
            }
            'r' => {
                out.push('\r');
                changed = true;
            }
            't' => {
                out.push('\t');
                changed = true;
            }
            'u' => {
                let Some(decoded) = decode_json_unicode_escape(&mut chars) else {
                    out.push('\\');
                    out.push('u');
                    continue;
                };
                out.push_str(&decoded);
                changed = true;
            }
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }

    changed.then_some(out)
}

fn decode_json_unicode_escape<I>(chars: &mut std::iter::Peekable<I>) -> Option<String>
where
    I: Iterator<Item = char> + Clone,
{
    let first = decode_u16_hex(chars)?;
    if !(0xD800..=0xDBFF).contains(&first) {
        return char::from_u32(first as u32).map(|ch| ch.to_string());
    }

    let mut clone = chars.clone();
    if clone.next() != Some('\\') || clone.next() != Some('u') {
        return None;
    }
    let second = decode_u16_hex(&mut clone)?;
    if !(0xDC00..=0xDFFF).contains(&second) {
        return None;
    }

    let high = (first as u32) - 0xD800;
    let low = (second as u32) - 0xDC00;
    let code = 0x10000 + ((high << 10) | low);
    let ch = char::from_u32(code)?;

    // 正式消费 surrogate pair
    chars.next();
    chars.next();
    for _ in 0..4 {
        chars.next();
    }
    Some(ch.to_string())
}

fn decode_u16_hex<I>(chars: &mut std::iter::Peekable<I>) -> Option<u16>
where
    I: Iterator<Item = char>,
{
    let mut value = 0u16;
    for _ in 0..4 {
        let ch = chars.next()?;
        value = (value << 4) | hex_value(ch)? as u16;
    }
    Some(value)
}

fn hex_value(ch: char) -> Option<u8> {
    match ch {
        '0'..='9' => Some((ch as u8) - b'0'),
        'a'..='f' => Some((ch as u8) - b'a' + 10),
        'A'..='F' => Some((ch as u8) - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursively_expands_nested_object_string_fields() {
        let value = serde_json::json!({
            "msg": "{\"code\":\"ok\",\"data\":\"{\\\"nested\\\":true}\"}"
        });

        let expanded = expand_nested_json_strings(value, jsonw::ParseOptions::default());

        assert_eq!(
            expanded,
            serde_json::json!({
                "msg": {
                    "code": "ok",
                    "data": {
                        "nested": true
                    }
                }
            })
        );
    }

    #[test]
    fn recursively_expands_nested_array_string_fields() {
        let value = serde_json::json!([
            "{\"items\":[\"{\\\"id\\\":1}\",\"plain\"]}"
        ]);

        let expanded = expand_nested_json_strings(value, jsonw::ParseOptions::default());

        assert_eq!(
            expanded,
            serde_json::json!([
                {
                    "items": [
                        {"id": 1},
                        "plain"
                    ]
                }
            ])
        );
    }

    #[test]
    fn keeps_non_json_strings_unchanged() {
        let value = serde_json::json!({
            "msg": "plain text only"
        });

        let expanded = expand_nested_json_strings(value.clone(), jsonw::ParseOptions::default());

        assert_eq!(expanded, value);
    }

    #[test]
    fn unescapes_non_json_string_content() {
        let value = serde_json::json!({
            "msg": "请求 dataService 错误: {\"code\":\"forbidden\",\"msg\":\"\\u5728\\u6237\\u672a\\u767b\\u9646\"}\n"
        });

        let expanded = expand_nested_json_strings(value, jsonw::ParseOptions::default());

        assert_eq!(
            expanded,
            serde_json::json!({
                "msg": "请求 dataService 错误: {\"code\":\"forbidden\",\"msg\":\"在户未登陆\"}\n"
            })
        );
    }
}
