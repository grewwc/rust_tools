use unicode_width::UnicodeWidthChar;

#[test]
fn check_emoji_widths() {
    // Characters from the screenshot
    let test_chars = vec![
        ('🐛', "bug"),
        ('⚠', "warning sign"),
        ('📝', "memo"),
        ('✅', "check mark"),
        ('🌧', "cloud with rain"),
        ('💧', "droplet"),
        ('🍃', "leaf"),
        ('☀', "sun"),
        ('✂', "scissors"),
        ('→', "right arrow"),
        ('×', "multiplication sign"),
    ];

    for (ch, name) in test_chars {
        let uw = UnicodeWidthChar::width(ch);
        let uw_cjk = UnicodeWidthChar::width_cjk(ch);
        println!("{} {} U+{:04X}: unicode_width={:?}, unicode_width_cjk={:?}", name, ch, ch as u32, uw, uw_cjk);
    }
}
