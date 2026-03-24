use colored::Colorize;
use regex::Regex;
use rust_tools::strw::split::split_by_str_keep_quotes;

pub fn globs_to_regexes(raw: &str, verbose: bool) -> Vec<Regex> {
    let mut ignores = raw.replace(',', " ");
    ignores = ignores.trim().to_string();
    if ignores.is_empty() {
        return Vec::new();
    }

    let parts = split_by_str_keep_quotes(&ignores, " ", "\"", false);
    let mut out = Vec::new();
    for part in parts {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let mut temp = p.replace('.', r"\.");
        temp = temp.replace('?', ".");
        temp = temp.replace('*', ".*");
        match Regex::new(&temp) {
            Ok(re) => out.push(re),
            Err(e) => {
                if verbose {
                    eprintln!("{}", format!("invalid exclude pattern {p}: {e}").red());
                }
            }
        }
    }
    out
}

pub fn should_exclude(path: &str, excludes: &[Regex]) -> bool {
    if excludes.is_empty() {
        return false;
    }
    let s = path.replace('\\', "/");
    excludes.iter().any(|re| re.is_match(&s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_globs_to_regexes_basic() {
        let res = globs_to_regexes("a*.rs, b?.go", false);
        assert_eq!(res.len(), 2);
        assert!(res[0].is_match("a123.rs"));
        assert!(res[1].is_match("bx.go"));
    }
}
