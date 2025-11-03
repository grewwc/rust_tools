use crate::strw::split::split_space_keep_symbol;

pub fn run_cmd(cmd: &str) -> std::io::Result<String> {
    if cmd.len() == 0 {
        return Ok("".to_owned());
    }
    let mut iter = split_space_keep_symbol(cmd, r#"""#).into_iter();
    let program = iter.next().unwrap();
    let mut cmd = std::process::Command::new(program);
    iter.for_each(|arg| {
        cmd.arg(arg);
    });
    let output = cmd.output().map(|output| output.stdout)?;
    Ok(String::from_utf8_lossy(output.as_ref()).to_string())
}

