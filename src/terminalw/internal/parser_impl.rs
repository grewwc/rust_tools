use std::collections::{HashSet, VecDeque};

use crate::collections::trie::Trie;
use crate::strw::{indices, split};

use super::{EncodedArgs, FlagDef, FlagType, Parser};

const SEP: char = '\x00';
const QUOTE: char = '\x05';
const DASH: char = '\x01';
const SPACE: char = '\x02';

pub(crate) fn normalize_flag_key(flag_name: &str) -> String {
    let s = flag_name.trim();
    if s.starts_with('-') {
        s.to_string()
    } else {
        format!("-{s}")
    }
}

pub(crate) fn parse_args_cmd(p: &mut Parser, bool_optionals: &[&str]) {
    let argv = std::env::args().collect::<Vec<_>>();
    let mut start = 1usize;
    if argv.len() > 1 {
        if p.groups.get(&argv[1]).is_some() {
            start += 1;
            if let Some(sub) = p.groups.get_mut(&argv[1]) {
                *p = sub.clone();
            }
        }
    }
    let rest = argv.into_iter().skip(start).collect::<Vec<_>>();
    parse_argv(p, &rest, bool_optionals);
}

pub(crate) fn parse_args(p: &mut Parser, cmd: &str, bool_optionals: &[&str]) {
    p.cmd = cmd.to_string();
    p.num_arg = None;
    p.optional.clear();
    p.positional.clear();

    for opt in bool_optionals {
        let o = opt.trim_start_matches('-').to_string();
        if !o.is_empty() {
            p.bool_option_set.insert(o);
        }
    }

    process_alias_defs(p);

    let bool_opts = if bool_optionals.is_empty() {
        p.bool_option_set.iter().cloned().collect::<Vec<_>>()
    } else {
        bool_optionals
            .iter()
            .map(|s| s.trim_start_matches('-').to_string())
            .collect::<Vec<_>>()
    };

    let mut trie = Trie::new();
    for name in &bool_opts {
        trie.insert(name);
    }

    let cmd_slice =
        split::split_by_str_keep_quotes(cmd, " ", format!("\"'{}", QUOTE).as_str(), true);
    let mut args = Vec::with_capacity(cmd_slice.len());
    for mut arg in cmd_slice {
        if arg.starts_with("--") && arg.len() > 2 {
            arg = format!("-{}", arg.trim_start_matches("--"));
        }
        if arg.starts_with('-') && arg.len() > 1 {
            let mut segments = Vec::new();
            if test_bool_cluster(arg.trim_start_matches('-'), &trie, &mut segments) {
                for seg in segments {
                    args.push(format!("{}{}{}{}", QUOTE, DASH, seg, QUOTE));
                }
            } else {
                args.push(format!("{}{}{}", QUOTE, arg, QUOTE));
            }
        } else {
            args.push(format!("{}{}{}", QUOTE, arg, QUOTE));
        }
    }

    let mut encoded = args.join(&SEP.to_string());
    if p.enable_parse_num {
        if let Some((new_encoded, num)) = extract_num_arg(&encoded) {
            encoded = new_encoded;
            p.num_arg = Some(num);
        }
    }
    encoded = format!("{SEP}{encoded}{SEP}");
    parse_args_encoded(p, &encoded, &bool_opts);
}

pub(crate) fn parse_argv(p: &mut Parser, argv: &[String], bool_optionals: &[&str]) {
    p.cmd = argv.join(" ");
    p.num_arg = None;
    p.optional.clear();
    p.positional.clear();

    for opt in bool_optionals {
        let o = opt.trim_start_matches('-').to_string();
        if !o.is_empty() {
            p.bool_option_set.insert(o);
        }
    }

    process_alias_defs(p);

    let bool_opts = if bool_optionals.is_empty() {
        p.bool_option_set.iter().cloned().collect::<Vec<_>>()
    } else {
        bool_optionals
            .iter()
            .map(|s| s.trim_start_matches('-').to_string())
            .collect::<Vec<_>>()
    };

    let mut trie = Trie::new();
    for name in &bool_opts {
        trie.insert(name);
    }

    let mut flat = Vec::with_capacity(argv.len());
    for raw in argv {
        let trimmed = raw.trim_matches(['"', '\'']).to_string();
        if (trimmed.starts_with('-') || trimmed.starts_with("--")) && trimmed.contains('=') {
            if let Some((k, v)) = trimmed.split_once('=') {
                flat.push(k.to_string());
                flat.push(v.to_string());
                continue;
            }
        }
        flat.push(trimmed);
    }

    let mut args = Vec::with_capacity(flat.len());
    for mut arg in flat {
        arg = arg.replace(' ', &SPACE.to_string());
        if arg.starts_with("--") && arg.len() > 2 {
            arg = format!("-{}", arg.trim_start_matches("--"));
        }
        if arg.starts_with('-') && arg.len() > 1 {
            let mut segments = Vec::new();
            if test_bool_cluster(arg.trim_start_matches('-'), &trie, &mut segments) {
                for seg in segments {
                    args.push(format!("{}{}{}{}", QUOTE, DASH, seg, QUOTE));
                }
            } else {
                args.push(format!("{}{}{}", QUOTE, arg, QUOTE));
            }
        } else {
            args.push(format!("{}{}{}", QUOTE, arg, QUOTE));
        }
    }

    let mut encoded = args.join(&SEP.to_string());
    if p.enable_parse_num {
        if let Some((new_encoded, num)) = extract_num_arg(&encoded) {
            encoded = new_encoded;
            p.num_arg = Some(num);
        }
    }
    encoded = format!("{SEP}{encoded}{SEP}");
    parse_args_encoded(p, &encoded, &bool_opts);
}

fn process_alias_defs(p: &mut Parser) {
    let defs = p.flags.values().cloned().collect::<Vec<_>>();
    for def in defs {
        let Some(target) = p.alias_map.get(&def.name).cloned() else {
            continue;
        };
        if p.flags.contains_key(&target) {
            continue;
        }
        let usage = if def.usage.trim().is_empty() {
            format!("(alias for {:?})", def.name)
        } else {
            format!("{} (alias for {:?})", def.usage, def.name)
        };
        define_flag(p, &target, def.ty, def.default_value.clone(), &usage);
        if def.ty == FlagType::Bool {
            p.bool_option_set.insert(target);
        }
    }
}

fn define_flag(p: &mut Parser, name: &str, ty: FlagType, default_value: String, usage: &str) {
    let name = name.trim_start_matches('-').to_string();
    p.default_val_map
        .insert(name.clone(), default_value.clone());
    p.flags.insert(
        name.clone(),
        FlagDef {
            name,
            usage: usage.to_string(),
            default_value,
            ty,
        },
    );
}

fn parse_args_encoded(p: &mut Parser, encoded: &str, bool_opts: &[String]) {
    let supported = p.flags.keys().cloned().collect::<HashSet<_>>();

    let mut cmd = encoded.to_string();
    for name in p.flags.keys().cloned().collect::<Vec<_>>() {
        p.default_val_map
            .entry(name.clone())
            .or_insert_with(|| String::new());
        let key = format!("{QUOTE}-{name}{QUOTE}");
        let positions = indices::find_all(&cmd, &key);
        if positions.is_empty() {
            continue;
        }
        for idx in positions.into_iter().rev() {
            let end = idx + key.len();
            if end > cmd.len() {
                continue;
            }
            cmd.replace_range(idx..end, &format!("{DASH}{name}{SEP}"));
        }
    }

    let (mut positionals, bool_keys, keys, mut vals) = classify_arguments(&cmd, bool_opts);
    let mut opt = crate::collections::ordered_map::OrderedMap::new();

    for (i, key) in keys.into_iter().enumerate() {
        if i < vals.len() {
            vals[i] = vals[i].replace(SPACE, " ");
        }
        let mut actual_key = key.replace(DASH, "-");
        if actual_key.starts_with("--") {
            actual_key = format!("-{}", actual_key.trim_start_matches("--"));
        }
        let bare = actual_key.trim_start_matches('-').to_string();
        if !supported.contains(&bare) {
            if i < vals.len() {
                positionals.push_back(format!("{} {}", actual_key, vals[i]));
            } else {
                positionals.push_back(actual_key);
            }
            continue;
        }
        if i < vals.len() {
            opt.insert(actual_key, vals[i].clone());
        } else {
            let dv = p.default_val_map.get(&bare).cloned().unwrap_or_default();
            opt.insert(actual_key, dv);
        }
    }

    for key in bool_keys {
        let actual_key = key.replace(DASH, "-");
        let bare = actual_key.trim_start_matches('-');
        let default_val = p.default_val_map.get(bare).cloned().unwrap_or_default();
        if default_val == "false" {
            opt.insert(actual_key, "true".to_string());
        } else if !default_val.is_empty() {
            opt.insert(actual_key, default_val);
        } else {
            opt.insert(actual_key, "true".to_string());
        }
    }

    p.positional = crate::collections::deque_list::DequeList::from(positionals);
    p.optional = opt;
}

fn extract_num_arg(encoded: &str) -> Option<(String, i32)> {
    let bytes = encoded.as_bytes();
    let mut i = 0usize;
    while i + 2 < bytes.len() {
        if bytes[i] == QUOTE as u8 && bytes[i + 1] == b'-' {
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 2 && j < bytes.len() && bytes[j] == QUOTE as u8 {
                let num_str = &encoded[i + 2..j];
                if let Ok(n) = num_str.parse::<i32>() {
                    let mut out = encoded.to_string();
                    out.replace_range(i..=j, "");
                    return Some((out, n));
                }
            }
        }
        i += 1;
    }
    None
}

fn test_bool_cluster(cmd: &str, trie: &Trie, out: &mut Vec<String>) -> bool {
    if cmd.is_empty() {
        return false;
    }
    if trie.contains(cmd) {
        out.push(cmd.to_string());
        return true;
    }
    for i in 1..cmd.len() {
        let curr = &cmd[..i];
        if trie.contains(curr) && test_bool_cluster(&cmd[i..], trie, out) {
            out.push(curr.to_string());
            return true;
        }
    }
    false
}

fn can_construct_by_bool_optionals(key: &str, bool_opts: &[String]) -> bool {
    let key = key.trim_start_matches(DASH);
    if key.is_empty() {
        return true;
    }
    for (i, opt) in bool_opts.iter().enumerate() {
        if key.starts_with(opt) {
            let mut rest = bool_opts.to_vec();
            rest.remove(i);
            if can_construct_by_bool_optionals(&key[opt.len()..], &rest) {
                return true;
            }
        }
    }
    false
}

fn classify_arguments(cmd: &str, bool_opts: &[String]) -> EncodedArgs {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Positional,
        OptionalKey,
        OptionalVal,
        Space,
        BoolOptional,
        Start,
    }

    let mut prev = Mode::Start;
    let mut mode = Mode::Space;
    let mut positionals: VecDeque<String> = VecDeque::new();
    let mut keys: Vec<String> = Vec::new();
    let mut bool_keys: Vec<String> = Vec::new();
    let mut vals: Vec<String> = Vec::new();
    let mut p_buf = String::new();
    let mut k_buf = String::new();
    let mut v_buf = String::new();

    for ch in cmd.chars() {
        if ch == QUOTE {
            continue;
        }
        match mode {
            Mode::Space => {
                if ch == SEP {
                    continue;
                }
                if ch == DASH {
                    mode = Mode::OptionalKey;
                    k_buf.push(ch);
                } else {
                    if matches!(
                        prev,
                        Mode::BoolOptional | Mode::Start | Mode::Positional | Mode::OptionalVal
                    ) {
                        mode = Mode::Positional;
                        p_buf.push(ch);
                    } else {
                        mode = Mode::OptionalVal;
                        v_buf.push(ch);
                    }
                    prev = Mode::Space;
                }
            }
            Mode::Positional => {
                if ch == SEP {
                    mode = Mode::Space;
                    positionals.push_back(p_buf.replace(SPACE, " "));
                    p_buf.clear();
                    prev = Mode::Positional;
                    continue;
                }
                p_buf.push(ch);
            }
            Mode::OptionalKey => {
                if ch == SEP {
                    let k_str = k_buf.clone();
                    if can_construct_by_bool_optionals(&k_str, bool_opts) {
                        prev = Mode::BoolOptional;
                        bool_keys.push(k_str);
                    } else {
                        prev = Mode::OptionalKey;
                        keys.push(k_str);
                    }
                    mode = Mode::Space;
                    k_buf.clear();
                    continue;
                }
                k_buf.push(ch);
            }
            Mode::OptionalVal => {
                if ch == SEP {
                    mode = Mode::Space;
                    vals.push(v_buf.clone());
                    v_buf.clear();
                    prev = Mode::OptionalVal;
                    continue;
                }
                v_buf.push(ch);
            }
            Mode::BoolOptional | Mode::Start => {}
        }
    }

    (positionals, bool_keys, keys, vals)
}
