use regex::Regex;
use rust_tools::strw::indices::substring_quiet;

#[derive(Clone)]
pub enum MatchMode {
    Contains { needle: String, needle_re: Option<Regex> },
    Strict { needle: String, ignore_case: bool },
    Regex { re: Regex, overlap_hint_len: usize },
}

impl MatchMode {
    pub fn overlap_hint_len(&self) -> usize {
        match self {
            MatchMode::Contains { needle, .. } => needle.len(),
            MatchMode::Strict { needle, .. } => needle.len(),
            MatchMode::Regex { overlap_hint_len, .. } => *overlap_hint_len,
        }
    }
}

pub fn build_matcher(
    mut target: String,
    is_regex: bool,
    ignore_case: bool,
    strict: bool,
    word: bool,
) -> Result<(String, MatchMode), String> {
    if word {
        let word_re = Regex::new(r"\w+").map_err(|e| e.to_string())?;
        if !word_re.is_match(&target) {
            return Err("You should pass in a word if set \"-w\" option".to_string());
        }
        target = format!(r"\b{target}\b");
    }

    if is_regex {
        let mut pat = target.clone();
        if ignore_case {
            pat = format!("(?i){pat}");
        }
        let re = Regex::new(&pat).map_err(|e| e.to_string())?;
        let overlap_hint_len = target.len();
        return Ok((target, MatchMode::Regex { re, overlap_hint_len }));
    }

    if strict {
        let needle = if ignore_case {
            target.to_lowercase()
        } else {
            target
        };
        return Ok((needle.clone(), MatchMode::Strict { needle, ignore_case }));
    }

    if ignore_case {
        let pat = format!("(?i){}", regex::escape(&target));
        let needle_re = Regex::new(&pat).map_err(|e| e.to_string())?;
        return Ok((
            target.clone(),
            MatchMode::Contains {
                needle: target,
                needle_re: Some(needle_re),
            },
        ));
    }

    Ok((
        target.clone(),
        MatchMode::Contains {
            needle: target,
            needle_re: None,
        },
    ))
}

pub fn match_line<'a>(line: &'a str, matcher: &MatchMode) -> Option<(&'a str, Vec<(usize, usize)>)> {
    match matcher {
        MatchMode::Contains { needle, needle_re } => {
            if let Some(re) = needle_re {
                let mut ranges = Vec::new();
                for m in re.find_iter(line) {
                    ranges.push((m.start(), m.end()));
                }
                if ranges.is_empty() {
                    None
                } else {
                    Some((line, ranges))
                }
            } else {
                let mut ranges = Vec::new();
                for (idx, part) in line.match_indices(needle) {
                    ranges.push((idx, idx + part.len()));
                }
                if ranges.is_empty() {
                    None
                } else {
                    Some((line, ranges))
                }
            }
        }
        MatchMode::Strict { needle, ignore_case } => {
            let trimmed = line.trim();
            let ok = if *ignore_case {
                trimmed.to_lowercase() == *needle
            } else {
                trimmed == needle
            };
            if ok {
                Some((trimmed, vec![(0, trimmed.len())]))
            } else {
                None
            }
        }
        MatchMode::Regex { re, .. } => {
            let mut ranges = Vec::new();
            for m in re.find_iter(line) {
                ranges.push((m.start(), m.end()));
            }
            if ranges.is_empty() {
                None
            } else {
                Some((line, ranges))
            }
        }
    }
}

pub fn crop_for_overlap(s: &str, overlap_len: usize) -> String {
    if overlap_len == 0 {
        return String::new();
    }
    let len = s.len();
    substring_quiet(s, len as isize - overlap_len as isize, len as isize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_contains_case_sensitive() {
        let (_, m) = build_matcher("ab".to_string(), false, false, false, false).unwrap();
        let res = match_line("xxabyyab", &m).unwrap();
        assert_eq!(res.1, vec![(2, 4), (6, 8)]);
    }

    #[test]
    fn test_match_contains_ignore_case() {
        let (_, m) = build_matcher("ab".to_string(), false, true, false, false).unwrap();
        let res = match_line("xxAbYYaB", &m).unwrap();
        assert_eq!(res.1, vec![(2, 4), (6, 8)]);
    }
}
