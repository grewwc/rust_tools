pub(crate) fn glob_match_simple(pattern: &str, name: &str) -> bool {
    let pattern = pattern.trim_start_matches("**/");
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let name_chars: Vec<char> = name.chars().collect();
    let mut pattern_index = 0usize;
    let mut name_index = 0usize;
    let mut star_index = None;
    let mut star_name_index = 0usize;

    while name_index < name_chars.len() {
        if pattern_index < pattern_chars.len()
            && (pattern_chars[pattern_index] == '?'
                || pattern_chars[pattern_index] == name_chars[name_index])
        {
            pattern_index += 1;
            name_index += 1;
        } else if pattern_index < pattern_chars.len() && pattern_chars[pattern_index] == '*' {
            star_index = Some(pattern_index);
            star_name_index = name_index;
            pattern_index += 1;
        } else if let Some(star_index) = star_index {
            pattern_index = star_index + 1;
            star_name_index += 1;
            name_index = star_name_index;
        } else {
            return false;
        }
    }

    while pattern_index < pattern_chars.len() && pattern_chars[pattern_index] == '*' {
        pattern_index += 1;
    }
    pattern_index == pattern_chars.len()
}

pub(crate) fn truncate_tool_output(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let mut output: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_none() {
        return s.to_string();
    }
    output.push_str("\n... (output truncated)");
    output
}
