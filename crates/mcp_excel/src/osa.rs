//! osascript 调用封装 + AppleScript 模板（本 crate 的核心，所有 Excel 交互坑都收敛在这）。
//!
//! # 为什么是 osascript 而非文件层库
//! 目标是「像浏览器那样驱动真实 Excel 软件」，因此走 AppleScript（osascript）
//! 对应浏览器方案里的 CDP 地位。每次调用都是独立的 `osascript -e` 子进程，
//! Excel 应用自身维持工作簿的打开态，跨调用共享。
//!
//! # 已验证的 AppleScript 铁律（踩坑实测，务必遵守）
//! 1. **单个 `tell worksheet` 块只做一类操作**：纯读或纯写。同一 tell 块内
//!    先 `set value` 写、再 `value of` 读会触发 `-10003 access not allowed`
//!    （写操作污染块内后续读引用）。故读写模板严格分离。
//! 2. **单元格一律用 `range "A1"` 语法**，不用 `cell "A1"`：跨调用引用已打开
//!    workbook 时，`cell` 关键字属性访问会 `-10003`；`range`（含单格）稳定。
//! 3. **worksheet 用数字索引 `worksheet N of ...` 稳定**（实测 OK）。
//! 4. **读整块用 `value of used range`**，再在 Rust 侧循环拼 TSV：osascript 默认
//!    把二维 list 拍平会丢行结构，必须靠 AppleScript 内层 repeat 保住二维。
//! 5. **`open` 是异步的**：冷启动（Excel 刚被拉起）时 `open POSIX file` 返回后
//!    workbook 尚未注册进 `workbooks` 集合，紧接着的任何访问都会 `-50`。必须
//!    轮询 `exists workbook` 直到就绪再访问（见 open_workbook）。另注：`open
//!    workbook workbook file name "..."` 这个 specifier 在冷启动时会抛不可被
//!    `on error` 捕获的 -50，一律改用通用动词 `open POSIX file`。
//! 6. **集合遍历偏好「批量属性」，厌恶「对象引用遍历」**：取 sheet 名用
//!    `name of every worksheet of workbook X`（一次性属性）稳定；而
//!    `repeat with ws in (every worksheet of X)` 遍历 worksheet **对象引用**
//!    在冷启动后会 `-50`。这与铁律 4 的 `value of used range` 同源——Excel 的
//!    AS 桥对「一次性属性访问」远比「逐对象引用」健壮。
//! 7. **存盘 `save workbook as` 是 Excel 沙盒版的系统性 `-50` 限制**（多方证实，
//!    非交互 osascript 环境 powerbox 无法授权），故数据落盘改由 Rust 侧写文件
//!    （见 tools.rs 的 export），彻底绕过。

use tokio::process::Command;

/// 运行一段 AppleScript（多行整体通过单个 `-e` 传入），返回 stdout（已 trim 尾换行）。
///
/// 失败时返回 `Err`，把 Excel 的执行错误（含 `-50`/`-10003`/`-1728` 等错误码）
/// 原样带回，便于模型/用户诊断。错误消息刻意不含宿主 transport 触发词。
pub async fn run(script: &str) -> Result<String, String> {
    let output = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .output()
        .await
        .map_err(|e| format!("failed to spawn osascript: {e}"))?;

    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout);
        Ok(s.trim_end_matches('\n').to_string())
    } else {
        let err = String::from_utf8_lossy(&output.stderr);
        let err = err.trim();
        // osascript 的报错形如 "execution error: ... (-50)"，原样回传即可。
        Err(if err.is_empty() {
            format!("osascript exited with status {}", output.status)
        } else {
            err.to_string()
        })
    }
}

/// AppleScript 字符串字面量转义：`\` → `\\`，`"` → `\"`。
/// 用于把用户提供的路径/sheet 名/单元格值安全注入脚本，避免破坏语法或注入。
pub fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// 由 POSIX 路径取出文件名（Excel 打开后以文件名作为 workbook 标识）。
pub fn file_name_of(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// 判断 Excel 是否可用（已安装且可脚本化）。返回版本号或错误。
pub async fn excel_version() -> Result<String, String> {
    run(r#"tell application "Microsoft Excel" to return version as text"#).await
}

/// 幂等打开工作簿：若同名 workbook 已打开则复用，否则 `open`。
/// 返回该 workbook 的所有 worksheet 名（换行分隔）。
///
/// `path` 为 POSIX 绝对路径。
pub async fn open_workbook(path: &str) -> Result<String, String> {
    let p = esc(path);
    let name = esc(&file_name_of(path));
    let script = format!(
        r#"tell application "Microsoft Excel"
  set wbName to "{name}"
  if not (exists workbook wbName) then
    open POSIX file "{p}"
    -- 冷启动时 open 是异步的：Excel 需要片刻把 workbook 注册进集合，
    -- 否则紧接着的访问会 -50。轮询等待其就绪（最多 ~10s）。
    set tries to 0
    repeat until (exists workbook wbName) or (tries > 100)
      delay 0.1
      set tries to tries + 1
    end repeat
  end if
  if not (exists workbook wbName) then
    error "could not open workbook {name}"
  end if
  -- 一次性取属性（`name of every worksheet`），不用 `repeat with ws in ...`
  -- 遍历 worksheet 对象——后者在冷启动后引用不稳会 -50。
  set nameList to name of every worksheet of workbook wbName
  set AppleScript's text item delimiters to linefeed
  return (nameList as text)
end tell"#
    );
    run(&script).await
}

/// 列出某个（已打开或即将打开的）工作簿的所有 worksheet 名。
pub async fn list_sheets(path: &str) -> Result<String, String> {
    // 复用 open 的幂等逻辑即可。
    open_workbook(path).await
}

/// tell 块内定位 worksheet 的引用片段：给定名字则按名字，否则用第一个 sheet。
fn sheet_ref(sheet: Option<&str>) -> String {
    match sheet {
        Some(name) if !name.is_empty() => {
            format!(r#"first worksheet of wb whose name is "{}""#, esc(name))
        }
        _ => "worksheet 1 of wb".to_string(),
    }
}

/// 读单个单元格的值（`range "A1"` 语法，纯读 tell 块）。
pub async fn read_cell(path: &str, sheet: Option<&str>, addr: &str) -> Result<String, String> {
    let name = esc(&file_name_of(path));
    let a = esc(addr);
    let sref = sheet_ref(sheet);
    let script = format!(
        r#"tell application "Microsoft Excel"
  set wb to workbook "{name}"
  tell ({sref})
    return (value of (range "{a}")) as text
  end tell
end tell"#
    );
    run(&script).await
}

/// 读一个区域（或整个 used range）→ TSV 文本。
/// `range_addr` 为空则读 used range；否则读指定 `A1:C10` 区域。
pub async fn read_range(
    path: &str,
    sheet: Option<&str>,
    range_addr: Option<&str>,
) -> Result<String, String> {
    let name = esc(&file_name_of(path));
    let sref = sheet_ref(sheet);
    // 关键：AppleScript 内层双重 repeat 保住二维结构，用 TAB/linefeed 拼 TSV。
    // 单格区域时 value 不是二维 list，用 (class of v) 兜底。
    // TAB 用 `ASCII character 9` 显式取制表符——`tab` 关键字在本模板上下文里
    // 会被解析成字面字符串 "tab"（实测），故不用它。
    let target = match range_addr {
        Some(r) if !r.is_empty() => format!(r#"range "{}""#, esc(r)),
        _ => "used range".to_string(),
    };
    let script = format!(
        r#"tell application "Microsoft Excel"
  set wb to workbook "{name}"
  tell ({sref})
    set v to value of ({target})
  end tell
  set colSep to (ASCII character 9)
  set out to ""
  if (class of v) is list then
    repeat with r in v
      set rowText to ""
      if (class of r) is list then
        repeat with c in r
          set rowText to rowText & (c as text) & colSep
        end repeat
      else
        set rowText to (r as text) & colSep
      end if
      set out to out & rowText & linefeed
    end repeat
  else
    set out to (v as text)
  end if
  return out
end tell"#
    );
    run(&script).await
}

/// 写单个单元格（纯写 tell 块）。`value_is_number` 决定注入数值还是字符串字面量。
pub async fn write_cell(
    path: &str,
    sheet: Option<&str>,
    addr: &str,
    value: &str,
    value_is_number: bool,
) -> Result<String, String> {
    let name = esc(&file_name_of(path));
    let a = esc(addr);
    let sref = sheet_ref(sheet);
    // 数值直接注入（AppleScript 识别为 number），文本加引号并转义。
    let val_literal = if value_is_number {
        value.to_string()
    } else {
        format!(r#""{}""#, esc(value))
    };
    let script = format!(
        r#"tell application "Microsoft Excel"
  set wb to workbook "{name}"
  tell ({sref})
    set value of (range "{a}") to {val_literal}
  end tell
  return "ok"
end tell"#
    );
    run(&script).await
}

/// 关闭工作簿（不存盘，避免弹窗）。
pub async fn close_workbook(path: &str) -> Result<String, String> {
    let name = esc(&file_name_of(path));
    let script = format!(
        r#"tell application "Microsoft Excel"
  if (exists workbook "{name}") then
    close workbook "{name}" saving no
    return "closed"
  else
    return "not open"
  end if
end tell"#
    );
    run(&script).await
}

/// 尝试用 Excel 原生 `save workbook as` 存盘（实验性：沙盒版常返回 -50）。
/// 成功返回 "ok"，失败把原始错误带回（调用方据此走 Rust 侧导出兜底）。
pub async fn save_as(path: &str, dest: &str) -> Result<String, String> {
    let name = esc(&file_name_of(path));
    let d = esc(dest);
    let script = format!(
        r#"tell application "Microsoft Excel"
  set wb to workbook "{name}"
  save workbook as wb filename "{d}" file format workbook normal file format with overwrite
  return "ok"
end tell"#
    );
    run(&script).await
}
