use clap::Parser;
use rust_tools::commonw::configw;

#[derive(Parser, Debug)]
#[command(about = "Read ~/.configW (go_tools compatible)")]
struct Cli {
    #[arg(value_name = "KEY", default_value = "")]
    key: String,

    #[arg(long, default_value = "", help = "default value if key missing")]
    default: String,

    #[arg(long, default_value_t = false, help = "print all key/value pairs")]
    all: bool,

    #[arg(long, default_value_t = false, help = "print all config as JSON")]
    json: bool,

    #[arg(
        long,
        default_value = "",
        help = "override config file path (or set CONFIGW_PATH)"
    )]
    path: String,

    #[arg(long, default_value_t = false, help = "refresh cache before reading")]
    refresh: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.refresh {
        configw::refresh();
    }

    if cli.all || cli.json {
        let cfg = if cli.path.trim().is_empty() {
            configw::get_all_config()
        } else {
            configw::ConfigW::from_file(cli.path.trim()).unwrap_or_default()
        };
        if cli.json {
            let mut map = serde_json::Map::new();
            for (k, v) in cfg.entries() {
                map.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            let s =
                serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_default();
            println!("{s}");
        } else {
            for (k, v) in cfg.entries() {
                println!("{k}={v}");
            }
        }
        return;
    }

    if cli.key.trim().is_empty() {
        eprintln!("usage: configw <key> [--default xxx]  |  configw --all  |  configw --json");
        std::process::exit(2);
    }
    let v = if cli.path.trim().is_empty() {
        configw::get_config(&cli.key, &cli.default)
    } else {
        let cfg = configw::ConfigW::from_file(cli.path.trim()).unwrap_or_default();
        cfg.get(&cli.key, &cli.default)
    };
    println!("{v}");
}
