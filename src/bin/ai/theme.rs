pub(in crate::ai) const RESET: &str = "\x1b[0m";
pub(in crate::ai) const BOLD: &str = "\x1b[1m";
pub(in crate::ai) const DIM: &str = "\x1b[2m";

pub(in crate::ai) const ACCENT_PRIMARY: &str = "\x1b[38;2;110;130;160m";
pub(in crate::ai) const ACCENT_TOOL_NAME: &str = "\x1b[38;2;235;140;130m";
pub(in crate::ai) const ACCENT_SECONDARY: &str = "\x1b[38;2;196;181;253m";
pub(in crate::ai) const ACCENT_COMMAND: &str = "\x1b[38;2;165;185;225m";
pub(in crate::ai) const ACCENT_MUTED: &str = "\x1b[38;2;148;163;184m";
/// 低饱和暖灰：用于用户输入正文，避免与蓝/青色状态信息争夺视觉注意力。
pub(in crate::ai) const ACCENT_INPUT_RGB: (u8, u8, u8) = (215, 212, 206);
pub(in crate::ai) const ACCENT_INPUT: &str = "\x1b[38;2;215;212;206m";
pub(in crate::ai) const ACCENT_SUCCESS: &str = "\x1b[38;2;134;194;166m";
pub(in crate::ai) const ACCENT_WARN: &str = "\x1b[38;2;245;158;11m";
pub(in crate::ai) const ACCENT_DANGER: &str = "\x1b[38;2;251;113;133m";
pub(in crate::ai) const ACCENT_RULE: &str = "\x1b[38;2;71;85;105m";
/// 已提交用户输入预览：深紫，在白底/深底上都清晰可辨，与编辑态的低饱和暖灰区分。
pub(in crate::ai) const ACCENT_SUBMITTED: &str = "\x1b[38;2;120;80;170m";
