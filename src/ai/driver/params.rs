use crate::ai::types::LoopOverrides;
use crate::strw::split::split_space_keep_symbol;

pub fn loop_overrides(question: &str) -> LoopOverrides {
    parse_loop_overrides(question).1
}

pub fn parse_loop_overrides(question: &str) -> (String, LoopOverrides) {
    let mut tokens = split_space_keep_symbol(question, "\"'").peekable();
    let mut out_tokens = Vec::new();

    let mut short_output = false;
    let mut has_x = false;
    let mut history_count: Option<usize> = None;

    // First pass to check for -x flag
    let mut first_pass = split_space_keep_symbol(question, "\"'");
    while let Some(token) = first_pass.next() {
        if token == "-x" {
            has_x = true;
            break;
        }
    }
    if has_x {
        history_count = Some(0);
    }

    // Second pass to process tokens
    while let Some(token) = tokens.next() {
        match token {
            "-s" => {
                short_output = true;
            }
            "-x" => {
                // Already handled, skip
            }
            "--history" => {
                if !has_x {
                    if let Some(next) = tokens.peek()
                        && let Ok(value) = next.parse::<usize>()
                    {
                        history_count = Some(value);
                        tokens.next(); // Consume the value
                    } else {
                        out_tokens.push(token.to_string());
                    }
                }
            }
            _ => {
                out_tokens.push(token.to_string());
            }
        }
    }

    (
        out_tokens.join(" "),
        LoopOverrides {
            short_output,
            history_count,
        },
    )
}
