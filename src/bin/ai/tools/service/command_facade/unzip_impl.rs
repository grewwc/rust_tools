//! Internal `unzip` implementation used when Info-ZIP's `unzip` is not installed.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use zip::{CompressionMethod, ZipArchive};

use super::{
    COPY_BUFFER_BYTES, CopyError, InternalOutcome, MAX_INTERNAL_OUTPUT_BYTES,
    copy_with_deadline, deadline_passed, failure, member_matches, resolve_path,
};

#[derive(Default)]
struct Options {
    list: bool,
    test: bool,
    pipe: bool,
    overwrite: bool,
    never_overwrite: bool,
    quiet: bool,
    junk_paths: bool,
    destination: Option<String>,
    exclude: Vec<String>,
    archive: String,
    members: Vec<String>,
    help: bool,
    version: bool,
}

struct PlannedEntry {
    index: usize,
    name: String,
    relative: PathBuf,
    out_path: PathBuf,
    is_dir: bool,
    mode: u32,
    compression: CompressionMethod,
}

pub(super) fn run(args: &[String], base: &Path, deadline: Instant) -> InternalOutcome {
    let options = match parse_args(args) {
        Ok(options) => options,
        Err(message) => return failure(message),
    };
    if options.help {
        return InternalOutcome::Success(help_text().to_string());
    }
    if options.version {
        return InternalOutcome::Success("internal unzip (rust_tools zip fallback)".to_string());
    }
    let archive_path = resolve_archive_path(base, &options.archive);
    let file = match File::open(&archive_path) {
        Ok(file) => file,
        Err(err) => {
            return failure(format!(
                "unzip: cannot open {}: {err}",
                archive_path.display()
            ));
        }
    };
    let mut archive = match ZipArchive::new(file) {
        Ok(archive) => archive,
        Err(err) => {
            return failure(format!(
                "unzip: cannot read {}: {err}",
                archive_path.display()
            ));
        }
    };
    if options.list {
        return list(&mut archive, &archive_path, &options, deadline);
    }
    if options.test {
        return test(&mut archive, &archive_path, &options, deadline);
    }
    if options.pipe {
        return pipe(&mut archive, &options, deadline);
    }
    extract(&mut archive, &archive_path, &options, base, deadline)
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut positional = Vec::new();
    let mut end_of_options = false;
    let mut index = 0;
    while index < args.len() {
        let token = &args[index];
        if !end_of_options {
            if token == "--" {
                end_of_options = true;
                index += 1;
                continue;
            }
            if token == "--help" {
                options.help = true;
                index += 1;
                continue;
            }
            if token == "--version" {
                options.version = true;
                index += 1;
                continue;
            }
            if token.starts_with('-') && token.len() > 1 {
                let chars: Vec<char> = token[1..].chars().collect();
                let mut position = 0;
                while position < chars.len() {
                    match chars[position] {
                        'l' => options.list = true,
                        't' => options.test = true,
                        'p' => options.pipe = true,
                        'o' => options.overwrite = true,
                        'n' => options.never_overwrite = true,
                        'q' => options.quiet = true,
                        'j' => options.junk_paths = true,
                        'v' => options.list = true,
                        'h' => options.help = true,
                        'd' => {
                            let attached: String =
                                chars[position + 1..].iter().copied().collect();
                            if attached.is_empty() {
                                index += 1;
                                let Some(value) = args.get(index) else {
                                    return Err(
                                        "unzip: option '-d' requires a directory".to_string()
                                    );
                                };
                                options.destination = Some(value.clone());
                            } else {
                                options.destination = Some(attached);
                            }
                            position = chars.len();
                            continue;
                        }
                        'x' => {
                            if position + 1 != chars.len() {
                                return Err(
                                    "unzip: '-x' must be the last option in its token and must be followed by exclude patterns"
                                        .to_string(),
                                );
                            }
                            if index + 1 >= args.len() {
                                return Err(
                                    "unzip: '-x' requires at least one exclude pattern".to_string()
                                );
                            }
                            options.exclude.extend(args[index + 1..].iter().cloned());
                            index = args.len();
                            position = chars.len();
                            continue;
                        }
                        other => {
                            return Err(format!(
                                "unzip option '-{other}' is not supported by the internal fallback; install Info-ZIP unzip for the full option set"
                            ));
                        }
                    }
                    position += 1;
                }
                index += 1;
                continue;
            }
        }
        positional.push(token.clone());
        index += 1;
    }
    if options.help || options.version {
        return Ok(options);
    }
    let Some(archive) = positional.first().cloned() else {
        return Err("unzip: missing archive name".to_string());
    };
    options.archive = archive;
    options.members = positional.into_iter().skip(1).collect();
    if options.overwrite && options.never_overwrite {
        return Err("unzip: '-o' and '-n' are mutually exclusive".to_string());
    }
    let modes = [options.list, options.test, options.pipe]
        .into_iter()
        .filter(|enabled| *enabled)
        .count();
    if modes > 1 {
        return Err("unzip: '-l', '-t', and '-p' are mutually exclusive".to_string());
    }
    Ok(options)
}

fn resolve_archive_path(base: &Path, raw: &str) -> PathBuf {
    let path = resolve_path(base, raw);
    if path.exists() {
        return path;
    }
    let has_zip_extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("zip"));
    if !has_zip_extension {
        let mut name = path.as_os_str().to_os_string();
        name.push(".zip");
        let candidate = PathBuf::from(name);
        if candidate.exists() {
            return candidate;
        }
    }
    path
}

fn selected(name: &str, options: &Options) -> bool {
    let included = options.members.is_empty()
        || options
            .members
            .iter()
            .any(|pattern| member_matches(pattern, name));
    included
        && !options
            .exclude
            .iter()
            .any(|pattern| member_matches(pattern, name))
}

fn list(
    archive: &mut ZipArchive<File>,
    path: &Path,
    options: &Options,
    deadline: Instant,
) -> InternalOutcome {
    let mut output = format!("Archive:  {}\n", path.display());
    output.push_str("  Length      Date    Time    Name\n");
    output.push_str("---------  ---------- -----   ----\n");
    let mut total = 0u64;
    let mut count = 0usize;
    for index in 0..archive.len() {
        if deadline_passed(deadline) {
            return InternalOutcome::TimedOut;
        }
        let file = match archive.by_index(index) {
            Ok(file) => file,
            Err(err) => {
                return failure(format!("unzip: cannot read entry {index}: {err}"));
            }
        };
        let name = file.name().to_string();
        if !selected(&name, options) {
            continue;
        }
        let modified = file.last_modified().unwrap_or_default();
        output.push_str(&format!(
            "{:>9}  {:04}-{:02}-{:02} {:02}:{:02}   {}\n",
            file.size(),
            modified.year(),
            modified.month(),
            modified.day(),
            modified.hour(),
            modified.minute(),
            name
        ));
        total += file.size();
        count += 1;
    }
    output.push_str("---------                     -------\n");
    output.push_str(&format!(
        "{total:>9}                     {count} file{}\n",
        if count == 1 { "" } else { "s" }
    ));
    InternalOutcome::Success(output)
}

fn test(
    archive: &mut ZipArchive<File>,
    path: &Path,
    options: &Options,
    deadline: Instant,
) -> InternalOutcome {
    let mut output = format!("Archive:  {}\n", path.display());
    let mut bad = 0usize;
    for index in 0..archive.len() {
        if deadline_passed(deadline) {
            return InternalOutcome::TimedOut;
        }
        let mut file = match archive.by_index(index) {
            Ok(file) => file,
            Err(err) => {
                return failure(format!("unzip: cannot read entry {index}: {err}"));
            }
        };
        let name = file.name().to_string();
        if !selected(&name, options) {
            continue;
        }
        let mut sink = io::sink();
        match copy_with_deadline(&mut file, &mut sink, deadline) {
            Ok(_) => {}
            Err(CopyError::TimedOut) => return InternalOutcome::TimedOut,
            Err(CopyError::Io(_)) => {
                bad += 1;
                output.push_str(&format!("  bad CRC: {name}\n"));
            }
        }
    }
    if bad > 0 {
        return InternalOutcome::Failed {
            code: 1,
            stdout: output,
            stderr: format!("unzip: {bad} file(s) failed the CRC check"),
        };
    }
    output.push_str(&format!(
        "No errors detected in compressed data of {}.\n",
        path.display()
    ));
    InternalOutcome::Success(output)
}

fn pipe(
    archive: &mut ZipArchive<File>,
    options: &Options,
    deadline: Instant,
) -> InternalOutcome {
    let mut output = Vec::new();
    for index in 0..archive.len() {
        if deadline_passed(deadline) {
            return InternalOutcome::TimedOut;
        }
        let mut file = match archive.by_index(index) {
            Ok(file) => file,
            Err(err) => {
                return failure(format!("unzip: cannot read entry {index}: {err}"));
            }
        };
        let name = file.name().to_string();
        if file.is_dir() || !selected(&name, options) {
            continue;
        }
        let mut buffer = [0u8; COPY_BUFFER_BYTES];
        loop {
            if deadline_passed(deadline) {
                return InternalOutcome::TimedOut;
            }
            let read = match file.read(&mut buffer) {
                Ok(read) => read,
                Err(err) => {
                    return failure(format!("unzip: cannot read '{name}': {err}"));
                }
            };
            if read == 0 {
                break;
            }
            let remaining = MAX_INTERNAL_OUTPUT_BYTES.saturating_sub(output.len());
            if remaining == 0 {
                break;
            }
            output.extend_from_slice(&buffer[..read.min(remaining)]);
        }
        if output.len() >= MAX_INTERNAL_OUTPUT_BYTES {
            break;
        }
    }
    InternalOutcome::Success(String::from_utf8_lossy(&output).into_owned())
}

/// Return the first component of `relative` that exists as a symlink under
/// `base`. Extraction must not follow a pre-existing symlink out of the
/// destination directory.
fn symlink_component(base: &Path, relative: &Path) -> Option<PathBuf> {
    let mut current = base.to_path_buf();
    for component in relative.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = fs::symlink_metadata(&current) {
            if metadata.file_type().is_symlink() {
                return Some(current);
            }
        }
    }
    None
}

fn extract(
    archive: &mut ZipArchive<File>,
    archive_path: &Path,
    options: &Options,
    base: &Path,
    deadline: Instant,
) -> InternalOutcome {
    let destination = match &options.destination {
        Some(raw) => resolve_path(base, raw),
        None => base.to_path_buf(),
    };
    if destination.exists() && !destination.is_dir() {
        return failure(format!(
            "unzip: destination {} is not a directory",
            destination.display()
        ));
    }
    if let Err(err) = fs::create_dir_all(&destination) {
        return failure(format!(
            "unzip: cannot create destination {}: {err}",
            destination.display()
        ));
    }
    let mut planned = Vec::new();
    let mut warnings = Vec::new();
    for index in 0..archive.len() {
        if deadline_passed(deadline) {
            return InternalOutcome::TimedOut;
        }
        let file = match archive.by_index(index) {
            Ok(file) => file,
            Err(err) => {
                return failure(format!("unzip: cannot read entry {index}: {err}"));
            }
        };
        let name = file.name().to_string();
        if !selected(&name, options) {
            continue;
        }
        let is_dir = file.is_dir();
        let raw_mode = file.unix_mode().unwrap_or(0);
        let file_type = raw_mode & 0o170000;
        if file_type == 0o120000 {
            warnings.push(format!("unzip warning: skipping symlink entry '{name}'"));
            continue;
        }
        if file_type != 0 && file_type != 0o100000 && file_type != 0o040000 {
            warnings.push(format!(
                "unzip warning: skipping special file entry '{name}'"
            ));
            continue;
        }
        if options.junk_paths && is_dir {
            continue;
        }
        let Some(enclosed) = file.enclosed_name() else {
            warnings.push(format!("unzip warning: skipping unsafe path '{name}'"));
            continue;
        };
        let relative = if options.junk_paths {
            match enclosed.file_name() {
                Some(file_name) => PathBuf::from(file_name),
                None => {
                    warnings.push(format!("unzip warning: skipping path '{name}'"));
                    continue;
                }
            }
        } else {
            enclosed
        };
        let out_path = destination.join(&relative);
        if !out_path.starts_with(&destination) {
            warnings.push(format!(
                "unzip warning: skipping path outside destination '{name}'"
            ));
            continue;
        }
        // A pre-existing symlink inside the destination would redirect writes
        // outside it. Archive-created symlinks are skipped above; this catches
        // links that were already on disk before extraction.
        if let Some(link) = symlink_component(&destination, &relative) {
            warnings.push(format!(
                "unzip warning: skipping path through symlink '{}' (entry '{name}')",
                link.display()
            ));
            continue;
        }
        let mode = if raw_mode & 0o777 == 0 {
            if is_dir { 0o755 } else { 0o644 }
        } else {
            raw_mode & 0o777
        };
        planned.push(PlannedEntry {
            index,
            name,
            relative,
            out_path,
            is_dir,
            mode,
            compression: file.compression(),
        });
    }

    let mut seen = BTreeSet::new();
    let mut duplicates = Vec::new();
    for entry in &planned {
        if !seen.insert(entry.out_path.clone()) {
            duplicates.push(entry.out_path.clone());
        }
    }
    if !duplicates.is_empty() && !options.overwrite && !options.never_overwrite {
        return conflict_failure(
            duplicates
                .iter()
                .map(|path| format!("{} (duplicate archive entry)", path.display()))
                .collect(),
            options,
        );
    }

    let mut conflicts = Vec::new();
    for entry in &planned {
        if !entry.out_path.exists() {
            continue;
        }
        if entry.is_dir {
            if !entry.out_path.is_dir() {
                conflicts.push(format!(
                    "{} exists and is not a directory",
                    entry.out_path.display()
                ));
            }
        } else if entry.out_path.is_dir() {
            conflicts.push(format!(
                "{} exists and is a directory",
                entry.out_path.display()
            ));
        } else if !options.overwrite && !options.never_overwrite {
            conflicts.push(format!("{} already exists", entry.out_path.display()));
        }
    }
    if !conflicts.is_empty() {
        return conflict_failure(conflicts, options);
    }

    let mut output = String::new();
    if !options.quiet {
        output.push_str(&format!("Archive:  {}\n", archive_path.display()));
    }
    for warning in &warnings {
        output.push_str(warning);
        output.push('\n');
    }
    for entry in &planned {
        if deadline_passed(deadline) {
            return InternalOutcome::TimedOut;
        }
        // Re-check immediately before writing: the destination could have
        // gained a symlink after the planning pass.
        if let Some(link) = symlink_component(&destination, &entry.relative) {
            return failure(format!(
                "unzip: refusing to extract '{}' through symlink {}",
                entry.name,
                link.display()
            ));
        }
        if entry.is_dir {
            if let Err(err) = fs::create_dir_all(&entry.out_path) {
                return failure(format!(
                    "unzip: cannot create directory {}: {err}",
                    entry.out_path.display()
                ));
            }
            continue;
        }
        if entry.out_path.exists() {
            if options.never_overwrite {
                continue;
            }
            if options.overwrite {
                if let Err(err) = fs::remove_file(&entry.out_path) {
                    return failure(format!(
                        "unzip: cannot replace {}: {err}",
                        entry.out_path.display()
                    ));
                }
            }
        }
        if let Some(parent) = entry.out_path.parent() {
            if let Err(err) = fs::create_dir_all(parent) {
                return failure(format!(
                    "unzip: cannot create directory {}: {err}",
                    parent.display()
                ));
            }
        }
        let mut file = match archive.by_index(entry.index) {
            Ok(file) => file,
            Err(err) => {
                return failure(format!("unzip: cannot read '{}': {err}", entry.name));
            }
        };
        let mut out = match File::create(&entry.out_path) {
            Ok(out) => out,
            Err(err) => {
                return failure(format!(
                    "unzip: cannot create {}: {err}",
                    entry.out_path.display()
                ));
            }
        };
        match copy_with_deadline(&mut file, &mut out, deadline) {
            Ok(_) => {}
            Err(CopyError::TimedOut) => return InternalOutcome::TimedOut,
            Err(CopyError::Io(err)) => {
                return failure(format!(
                    "unzip: cannot extract {}: {err}",
                    entry.out_path.display()
                ));
            }
        }
        super::apply_permissions(&entry.out_path, entry.mode);
        if !options.quiet {
            let verb = if entry.compression == CompressionMethod::Stored {
                " extracting"
            } else {
                "  inflating"
            };
            output.push_str(&format!("{verb}: {}\n", entry.out_path.display()));
        }
    }
    InternalOutcome::Success(output)
}

fn conflict_failure(conflicts: Vec<String>, options: &Options) -> InternalOutcome {
    let mut message =
        String::from("unzip: cannot extract because these paths already exist or conflict:\n");
    for conflict in conflicts {
        message.push_str("  ");
        message.push_str(&conflict);
        message.push('\n');
    }
    if options.never_overwrite {
        message.push_str("Resolve the directory/file conflicts manually.\n");
    } else {
        message.push_str("Pass -o to overwrite existing files or -n to skip them.\n");
    }
    InternalOutcome::Failed {
        code: 1,
        stdout: String::new(),
        stderr: message,
    }
}

fn help_text() -> &'static str {
    "Usage: unzip [-lptqonj] [-d dir] archive.zip [members...] [-x patterns...]\n\
     Internal fallback used when Info-ZIP unzip is not installed."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflicting_modes_are_rejected() {
        let outcome = run(
            &["-l".to_string(), "-p".to_string(), "a.zip".to_string()],
            Path::new("."),
            Instant::now() + std::time::Duration::from_secs(30),
        );
        match outcome {
            InternalOutcome::Failed { stderr, .. } => {
                assert!(stderr.contains("mutually exclusive"), "{stderr}");
            }
            other => panic!("expected failure, got {other:?}"),
        }
    }
}
