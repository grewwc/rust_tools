use std::{
    cmp::Ordering as CmpOrdering,
    collections::HashMap,
    io::{self, IsTerminal, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

const POLL_INTERVAL_MS: u64 = 500;
const MIN_ACTIVE_KIB_PER_SEC: f64 = 0.01;

#[derive(Clone, Copy, Debug, Default)]
struct NetTotals {
    sent: u64,
    recv: u64,
}

#[derive(Clone, Debug)]
struct InterfaceTotals {
    name: String,
    totals: NetTotals,
}

#[derive(Clone, Debug)]
struct Snapshot {
    interfaces: Vec<InterfaceTotals>,
    tp: Instant,
}

#[derive(Clone, Debug)]
struct InterfaceSpeed {
    name: String,
    upload_kib_per_sec: f64,
    download_kib_per_sec: f64,
}

impl InterfaceSpeed {
    fn total_kib_per_sec(&self) -> f64 {
        self.upload_kib_per_sec + self.download_kib_per_sec
    }
}

fn format_speed(kib_per_sec: f64) -> String {
    let units = ["k/s", "m/s", "g/s", "t/s"];
    let mut value = kib_per_sec;
    let mut unit_idx = 0;

    while value >= 1024.0 && unit_idx < units.len() - 1 {
        value /= 1024.0;
        unit_idx += 1;
    }

    format!("{value:.2} {}", units[unit_idx])
}

fn is_target_interface(name: &str) -> bool {
    ["en", "eth", "lo", "utun"]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

fn parse_counter(value: &str) -> Option<u64> {
    if value == "-" {
        return None;
    }
    value.parse::<u64>().ok()
}

#[cfg(target_os = "linux")]
fn read_snapshot() -> io::Result<Snapshot> {
    let content = std::fs::read_to_string("/proc/net/dev")?;
    let interfaces = parse_linux_interfaces(&content)?;
    Ok(Snapshot {
        interfaces,
        tp: Instant::now(),
    })
}

#[cfg(target_os = "linux")]
fn parse_linux_interfaces(content: &str) -> io::Result<Vec<InterfaceTotals>> {
    let mut interfaces = Vec::new();

    for line in content.lines().skip(2) {
        let Some((name, payload)) = line.split_once(':') else {
            continue;
        };
        let iface = name.trim();
        if !is_target_interface(iface) {
            continue;
        }

        let cols: Vec<&str> = payload.split_whitespace().collect();
        if cols.len() < 9 {
            continue;
        }

        let Some(recv) = parse_counter(cols[0]) else {
            continue;
        };
        let Some(sent) = parse_counter(cols[8]) else {
            continue;
        };

        interfaces.push(InterfaceTotals {
            name: iface.to_string(),
            totals: NetTotals { sent, recv },
        });
    }

    if interfaces.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no eligible interfaces in /proc/net/dev",
        ));
    }

    interfaces.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(interfaces)
}

#[cfg(not(target_os = "linux"))]
fn read_snapshot() -> io::Result<Snapshot> {
    let output = std::process::Command::new("netstat")
        .args(["-ibdnW"])
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::other(format!("netstat failed: {}", stderr.trim())));
    }
    let content = String::from_utf8_lossy(&output.stdout);
    let interfaces = parse_macos_interfaces(content.as_ref())?;
    Ok(Snapshot {
        interfaces,
        tp: Instant::now(),
    })
}

#[cfg(not(target_os = "linux"))]
fn parse_macos_interfaces(content: &str) -> io::Result<Vec<InterfaceTotals>> {
    let mut lines = content.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty netstat output"))?;
    let (recv_idx, sent_idx) = get_input_output_index(header)?;

    let mut interface_map: HashMap<String, NetTotals> = HashMap::new();

    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() <= recv_idx || parts.len() <= sent_idx {
            continue;
        }

        let name = parts[0];
        if !is_target_interface(name) {
            continue;
        }

        let Some(recv) = parse_counter(parts[recv_idx]) else {
            continue;
        };
        let Some(sent) = parse_counter(parts[sent_idx]) else {
            continue;
        };

        let entry = interface_map.entry(name.to_string()).or_default();
        // netstat can emit multiple rows per interface; keep max counters for stability.
        entry.recv = entry.recv.max(recv);
        entry.sent = entry.sent.max(sent);
    }

    if interface_map.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no eligible interfaces in netstat output",
        ));
    }

    let mut interfaces: Vec<InterfaceTotals> = interface_map
        .into_iter()
        .map(|(name, totals)| InterfaceTotals { name, totals })
        .collect();
    interfaces.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(interfaces)
}

#[cfg(not(target_os = "linux"))]
fn get_input_output_index(header: &str) -> io::Result<(usize, usize)> {
    let mut i_index = None;
    let mut o_index = None;

    for (idx, col) in header.split_whitespace().enumerate() {
        match col {
            "Ibytes" => i_index = Some(idx),
            "Obytes" => o_index = Some(idx),
            _ => {}
        }
    }

    let recv_idx = i_index
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Ibytes column missing"))?;
    let sent_idx = o_index
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Obytes column missing"))?;

    Ok((recv_idx, sent_idx))
}

fn compute_speeds(prev: &Snapshot, curr: &Snapshot) -> Option<(Vec<InterfaceSpeed>, f64)> {
    let dt = curr.tp.duration_since(prev.tp).as_secs_f64();
    if dt <= 0.0 {
        return None;
    }

    let mut prev_map = HashMap::new();
    for iface in &prev.interfaces {
        prev_map.insert(iface.name.as_str(), iface.totals);
    }

    let mut speeds = Vec::with_capacity(curr.interfaces.len());
    for iface in &curr.interfaces {
        let (upload_kib_per_sec, download_kib_per_sec) =
            if let Some(last) = prev_map.get(iface.name.as_str()) {
                (
                    iface.totals.sent.saturating_sub(last.sent) as f64 / 1024.0 / dt,
                    iface.totals.recv.saturating_sub(last.recv) as f64 / 1024.0 / dt,
                )
            } else {
                (0.0, 0.0)
            };

        speeds.push(InterfaceSpeed {
            name: iface.name.clone(),
            upload_kib_per_sec,
            download_kib_per_sec,
        });
    }

    speeds.sort_by(|a, b| {
        b.total_kib_per_sec()
            .partial_cmp(&a.total_kib_per_sec())
            .unwrap_or(CmpOrdering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });

    Some((speeds, dt))
}

fn render_waiting(curr: &Snapshot) {
    clear_screen();
    println!(
        "ns | grouped realtime network speed monitor | interval {} ms",
        POLL_INTERVAL_MS
    );
    println!();
    println!("Collecting first baseline sample...");
    let names = curr
        .interfaces
        .iter()
        .map(|iface| iface.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    println!("Detected interfaces: {names}");
    println!();
    println!("Press Ctrl-C to quit.");
    let _ = io::stdout().flush();
}

fn render_error(err: &str) {
    clear_screen();
    println!("ns | grouped realtime network speed monitor");
    println!();
    println!("Failed to read network stats:");
    println!("{err}");
    println!();
    println!("Retrying...");
    println!("Press Ctrl-C to quit.");
    let _ = io::stdout().flush();
}

fn render_dashboard(speeds: &[InterfaceSpeed], dt: f64) {
    let total_upload: f64 = speeds.iter().map(|s| s.upload_kib_per_sec).sum();
    let total_download: f64 = speeds.iter().map(|s| s.download_kib_per_sec).sum();

    let mut visible: Vec<InterfaceSpeed> = speeds
        .iter()
        .filter(|s| s.total_kib_per_sec() >= MIN_ACTIVE_KIB_PER_SEC)
        .cloned()
        .collect();

    let mut hidden_idle = speeds.len().saturating_sub(visible.len());
    if visible.is_empty() {
        visible = speeds.to_vec();
        hidden_idle = 0;
    }

    let name_width = visible
        .iter()
        .map(|s| s.name.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let table_width = name_width + 44;

    clear_screen();
    println!(
        "ns | grouped realtime network speed monitor | sample {:.0} ms",
        dt * 1000.0
    );
    println!(
        "Total  Ul: {:>12}  Dl: {:>12}  Sum: {:>12}",
        format_speed(total_upload),
        format_speed(total_download),
        format_speed(total_upload + total_download)
    );
    println!("{:-<width$}", "", width = table_width);
    println!(
        "{:<name_width$}  {:>12}  {:>12}  {:>12}",
        "Interface",
        "Upload",
        "Download",
        "Total",
        name_width = name_width
    );
    println!("{:-<width$}", "", width = table_width);

    for speed in &visible {
        println!(
            "{:<name_width$}  {:>12}  {:>12}  {:>12}",
            speed.name,
            format_speed(speed.upload_kib_per_sec),
            format_speed(speed.download_kib_per_sec),
            format_speed(speed.total_kib_per_sec()),
            name_width = name_width
        );
    }

    if hidden_idle > 0 {
        println!("{:-<width$}", "", width = table_width);
        println!("{hidden_idle} idle interface(s) hidden (< 0.01 k/s)");
    }

    println!();
    println!("Press Ctrl-C to quit.");
    let _ = io::stdout().flush();
}

#[cfg(target_os = "windows")]
fn main() {
    eprintln!("ns is not supported on windows");
    std::process::exit(1);
}

#[cfg(not(target_os = "windows"))]
fn main() {
    let terminal_guard = TerminalGuard::new();
    let running = Arc::new(AtomicBool::new(true));
    let signal = Arc::clone(&running);
    if let Err(err) = ctrlc::set_handler(move || {
        signal.store(false, Ordering::SeqCst);
    }) {
        eprintln!("failed to set Ctrl-C handler: {err}");
        std::process::exit(1);
    }

    let interval = Duration::from_millis(POLL_INTERVAL_MS);
    let mut prev: Option<Snapshot> = None;

    while running.load(Ordering::SeqCst) {
        let now = match read_snapshot() {
            Ok(snapshot) => snapshot,
            Err(err) => {
                render_error(&err.to_string());
                std::thread::sleep(interval);
                continue;
            }
        };

        if let Some(last) = prev {
            if let Some((speeds, dt)) = compute_speeds(&last, &now) {
                render_dashboard(&speeds, dt);
            } else {
                render_waiting(&now);
            }
        } else {
            render_waiting(&now);
        }

        prev = Some(now);
        std::thread::sleep(interval);
    }

    println!();
    drop(terminal_guard);
}

fn clear_screen() {
    if TerminalGuard::enabled() {
        print!("\x1b[H\x1b[J");
    }
}

struct TerminalGuard {
    enabled: bool,
}

impl TerminalGuard {
    fn enabled() -> bool {
        io::stdout().is_terminal()
    }

    fn new() -> Self {
        let enabled = Self::enabled();
        if enabled {
            print!("\x1b[?1049h\x1b[H\x1b[J\x1b[?25l");
            let _ = io::stdout().flush();
        }
        Self { enabled }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.enabled {
            print!("\x1b[?25h\x1b[?1049l");
            let _ = io::stdout().flush();
        }
    }
}
