use crate::strw::split::split_space_keep_symbol;

pub fn run_cmd(cmd: &str) -> std::io::Result<String> {
    if cmd.len() == 0 {
        return Ok("".to_owned());
    }
    
    // Check if the command contains shell operators like pipes
    if cmd.contains('|') || cmd.contains('>') || cmd.contains('<') || cmd.contains('&') {
        // Use shell to execute the command
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()?;
        let mut result = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !stderr.is_empty() {
            result.push_str(&stderr);
        }
        return Ok(result);
    }
    
    // For simple commands, parse and execute directly
    let mut iter = split_space_keep_symbol(cmd, r#"""#).into_iter();
    let program = iter.next().unwrap();
    let mut cmd = std::process::Command::new(program);
    iter.for_each(|arg| {
        cmd.arg(arg);
    });
    let output = cmd.output()?;
    let mut result = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !stderr.is_empty() {
        result.push_str(&stderr);
    }
    Ok(result)
}


