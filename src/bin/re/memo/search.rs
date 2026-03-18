use regex::Regex;

use crate::memo::model::MemoRecord;

pub fn highlight_pattern(query: &str) -> Option<Regex> {
    let q = query.trim();
    if q.is_empty() {
        return None;
    }
    Regex::new(&format!("(?i){}", regex::escape(q))).ok()
}

pub fn score_record(record: &MemoRecord, query: &str) -> f64 {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return 0.0;
    }
    let text = record.title.to_lowercase();
    let mut score = 0.0;
    let mut start = 0usize;
    while let Some(idx) = text[start..].find(&q) {
        score += 1.0;
        start += idx + q.len();
        if start >= text.len() {
            break;
        }
    }
    if score == 0.0 {
        return 0.0;
    }
    score / (text.len().max(1) as f64).sqrt()
}

pub fn build_preview_lines(record: &MemoRecord, query: &str, max_lines: usize) -> Vec<String> {
    let q = query.trim();
    if q.is_empty() {
        return Vec::new();
    }
    let lower_q = q.to_lowercase();
    let mut out = Vec::new();
    for line in record.title.lines() {
        if out.len() >= max_lines {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.to_lowercase().contains(&lower_q) {
            out.push(t.to_string());
        }
    }
    out
}

