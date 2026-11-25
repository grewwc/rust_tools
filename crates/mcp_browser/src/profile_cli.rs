//! Standalone commands for preparing a persistent browser profile.

use crate::profile::{DATA_DIR_ENV, Keychain};
use crate::profile_sync::{self, SyncOptions};
use std::ffi::OsString;
use std::path::PathBuf;

const HELP: &str = "Usage:
  mcp_browser                         Run the stdio MCP server
  mcp_browser --sync-profile --help   Copy one Chrome profile to a dedicated directory
  mcp_browser --login --help          Log in using a separate Chrome window";

const SYNC_HELP: &str = "Usage: mcp_browser --sync-profile [--profile-dir DIR] [--from DIR]
           [--profile NAME_OR_DIRECTORY] [--force] [--keychain native|mock]

Copy one Chrome profile and its Local State, without taking control of the source browser.
  --profile-dir DIR             Dedicated destination (or MCP_BROWSER_USER_DATA_DIR)
  --from DIR                    User-data root or its profile subdirectory (auto-detect by default)
  --profile NAME_OR_DIRECTORY   Source display name or directory (direct path, last_used, or Default)
  --force                       Replace an existing idle destination after copying succeeds
  --keychain native|mock        Keychain for future launches (default: native)";

const LOGIN_HELP: &str = "Usage: mcp_browser --login [--profile-dir DIR] [--url URL]

Log in in a separate Chrome window and press Enter to save and close it.
  --profile-dir DIR   Dedicated persistent directory (or MCP_BROWSER_USER_DATA_DIR)
  --url URL           Initial page (default: about:blank)";

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Command {
    Serve,
    Help(&'static str),
    Sync(SyncOptions),
    Login { destination: PathBuf, url: String },
}

pub(super) fn parse(
    args: impl IntoIterator<Item = OsString>,
    env_directory: Option<PathBuf>,
) -> Result<Command, String> {
    let mut args = args.into_iter();
    let Some(mode) = args.next() else {
        return Ok(Command::Serve);
    };
    let sync = match mode.to_str() {
        Some("--help" | "-h") => return Ok(Command::Help(HELP)),
        Some("--sync-profile") => true,
        Some("--login") => false,
        _ => {
            return Err(
                "expected --sync-profile or --login, or no arguments for the MCP server"
                    .to_string(),
            );
        }
    };
    let mut args = args.peekable();
    let mut destination = None;
    let mut from = None;
    let mut source_profile = None;
    let mut keychain = None;
    let mut force = false;
    let mut url = None;
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next_if(|value| !value.is_empty() && !value.to_string_lossy().starts_with("--"))
                .ok_or_else(|| format!("{} requires a value", arg.to_string_lossy()))
        };
        let mut text = || {
            value()?
                .into_string()
                .ok()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    format!(
                        "{} requires a nonempty Unicode value",
                        arg.to_string_lossy()
                    )
                })
        };
        match arg.to_str() {
            Some("--help" | "-h") => {
                return Ok(Command::Help(if sync { SYNC_HELP } else { LOGIN_HELP }));
            }
            Some("--profile-dir") if destination.is_none() => {
                destination = Some(PathBuf::from(text()?))
            }
            Some("--from") if sync && from.is_none() => from = Some(PathBuf::from(value()?)),
            Some("--profile") if sync && source_profile.is_none() => source_profile = Some(text()?),
            Some("--force") if sync && !force => force = true,
            Some("--keychain") if sync && keychain.is_none() => {
                keychain =
                    Some(Keychain::parse(&text()?).ok_or("--keychain must be native or mock")?)
            }
            Some("--url") if !sync && url.is_none() => url = Some(text()?),
            _ => return Err("unexpected or repeated option for this command".to_string()),
        }
    }
    let destination = destination.or(env_directory).ok_or_else(|| {
        format!("specify a dedicated Chrome profile with --profile-dir or {DATA_DIR_ENV}")
    })?;
    Ok(if sync {
        Command::Sync(SyncOptions {
            destination,
            from,
            profile: source_profile,
            force,
            keychain: keychain.unwrap_or(Keychain::Native),
        })
    } else {
        Command::Login {
            destination,
            url: url.unwrap_or_else(|| "about:blank".to_string()),
        }
    })
}

pub(super) async fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Serve => unreachable!("serve is dispatched in main"),
        Command::Help(help) => {
            println!("{help}");
            Ok(())
        }
        Command::Sync(options) => profile_sync::sync(options),
        Command::Login { destination, url } => crate::browser::login(&destination, &url).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(args: &[&str], env: Option<&str>) -> Result<Command, String> {
        parse(args.iter().map(OsString::from), env.map(PathBuf::from))
    }

    #[test]
    fn cli_routes_only_explicit_one_shot_commands() {
        assert_eq!(command(&[], Some("agent")).unwrap(), Command::Serve);
        assert!(matches!(
            command(&["--help"], None),
            Ok(Command::Help(HELP))
        ));
        assert!(matches!(
            command(&["--sync-profile", "--help"], None),
            Ok(Command::Help(SYNC_HELP))
        ));
        assert!(matches!(
            command(&["--login", "--help"], None),
            Ok(Command::Help(LOGIN_HELP))
        ));
        assert_eq!(
            command(&["--login"], Some("agent")).unwrap(),
            Command::Login {
                destination: "agent".into(),
                url: "about:blank".to_string()
            }
        );
        assert_eq!(
            command(
                &[
                    "--login",
                    "--profile-dir",
                    "agent directory",
                    "--url",
                    "https://example.test"
                ],
                Some("not-used")
            )
            .unwrap(),
            Command::Login {
                destination: "agent directory".into(),
                url: "https://example.test".to_string()
            }
        );
        let Command::Sync(defaults) = command(&["--sync-profile"], Some("agent")).unwrap() else {
            panic!("not sync")
        };
        assert_eq!(defaults.keychain, Keychain::Native);
        assert_eq!(
            command(
                &[
                    "--sync-profile",
                    "--profile-dir",
                    "agent",
                    "--from",
                    "source",
                    "--profile",
                    "Team work",
                    "--force",
                    "--keychain",
                    "mock"
                ],
                None
            )
            .unwrap(),
            Command::Sync(SyncOptions {
                destination: "agent".into(),
                from: Some("source".into()),
                profile: Some("Team work".into()),
                force: true,
                keychain: Keychain::Mock,
            })
        );
        for invalid in [
            &["--login"][..],
            &["--sync-profile"],
            &["--login", "--profile-dir", ""],
            &["--login", "--force"],
            &["--sync-profile", "--from", "--force"],
            &["--sync-profile", "--keychain", "unknown"],
            &["--sync-profile", "--force", "--force"],
        ] {
            assert!(command(invalid, None).is_err(), "{invalid:?}");
        }
    }
}
