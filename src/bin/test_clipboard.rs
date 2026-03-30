use arboard::Clipboard;

fn main() {
    println!("=== 测试 arboard 剪贴板图片读取 ===");

    match Clipboard::new() {
        Ok(mut clipboard) => {
            println!("✓ Clipboard 创建成功");

            // 先尝试获取文本
            match clipboard.get_text() {
                Ok(text) => println!(
                    "✓ 剪贴板文本：{} (前50字符)",
                    text.chars().take(50).collect::<String>()
                ),
                Err(e) => println!("✗ 获取文本失败：{:?}", e),
            }

            // 尝试获取图片
            match clipboard.get_image() {
                Ok(image) => {
                    println!("✓ 成功获取图片！");
                    println!("  尺寸：{}x{}", image.width, image.height);
                    println!("  字节数：{}", image.bytes.len());
                    println!(
                        "  前10字节：{:?}",
                        &image.bytes[0..10.min(image.bytes.len())]
                    );
                }
                Err(e) => {
                    println!("✗ 获取图片失败：{:?}", e);
                    println!("  错误类型：{}", std::any::type_name_of_val(&e));
                }
            }
        }
        Err(e) => {
            println!("✗ Clipboard 创建失败：{:?}", e);
        }
    }
}
