//! 密钥加密/解密工具
//!
//! 用法：
//! ```sh
//! secret encrypt "my-api-key"    # 输出 enc:xxxxx 格式
//! secret decrypt "enc:xxxxx"     # 输出明文
//! ```
//!
//! 加密后的值可以直接放入 models.json 的 api_key 字段，
//! 运行时会自动解密。

use rust_tools::commonw::secret;
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];
    let value = &args[2];

    match command.as_str() {
        "encrypt" | "enc" | "e" => {
            match secret::encrypt(value) {
                Ok(encrypted) => {
                    println!("{}", encrypted);
                }
                Err(e) => {
                    eprintln!("加密失败: {}", e);
                    process::exit(1);
                }
            }
        }
        "decrypt" | "dec" | "d" => {
            match secret::decrypt(value) {
                Ok(plaintext) => {
                    println!("{}", plaintext);
                }
                Err(e) => {
                    eprintln!("解密失败: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("未知命令: {}", command);
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!(
        r#"用法: secret <命令> <值>

命令:
  encrypt, enc, e    加密明文，输出 enc:<base64> 格式
  decrypt, dec, d    解密 enc:<base64> 格式，输出明文

示例:
  secret encrypt "my-api-key-12345"
  # 输出: enc:aBcDeFgHiJkLmNoPqRsT...

  secret decrypt "enc:aBcDeFgHiJkLmNoPqRsT..."
  # 输出: my-api-key-12345

加密后的值可直接放入 models.json:
  "api_key": "enc:aBcDeFgHiJkLmNoPqRsT..."
"#
    );
}
