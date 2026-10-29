//! Internal `zip` implementation used when Info-ZIP's `zip` is not installed.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use chrono::{Datelike, Timelike};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use super::{
    CopyError, InternalOutcome, copy_with_deadline, deadline_passed, failure, replace_file,
    resolve_path, temp_path_for, unix_mode, wildcard_match,
};

struct Options {
    recursive: bool,
    quiet: bool,
    junk_paths: bool,
    delete: bool,
    method: CompressionMethod,
    level: Option<i64>,
    archive: String,
    inputs: Vec<String>,
    exclude: Vec<String>,
    help: bool,
    version: bool,
}

enum EntrySource {
    File(PathBuf),
    Directory(PathBuf),
}

struct CollectedEntries {
    entries: BTreeMap<String, EntrySource>,
    warnings: Vec<String>,
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
        return InternalOutcome::Success("internal zip (rust_tools zip fallback)".to_string());
    }
    let archive = archive_path(base, &options.archive);
    if options.delete {
        run_delete(&archive, &options, deadline)
    } else {
        run_add(&archive, &options, base, deadline)
    }
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        recursive: false,
        quiet: false,
        junk_paths: false,
        delete: false,
        method: CompressionMethod::Deflated,
        level: None,
        archive: String::new(),
        inputs: Vec::new(),
        exclude: Vec::new(),
        help: false,
        version: false,
    };
    let option_end = args
        .iter()
        .position(|token| token == "--")
        .unwrap_or(args.len());
    let (args, exclude) = match args[..option_end]
        .iter()
        .position(|token| token == "-x")
    {
        Some(index) => {
            if index + 1 >= args.len() {
                return Err("zip: '-x' requires at least one exclude pattern".to_string());
            }
            (&args[..index], args[index + 1..].to_vec())
        }
        None => (args, Vec::new()),
    };
    options.exclude = exclude;
    let mut positional = Vec::new();
    let mut end_of_options = false;
    for token in args {
        if !end_of_options {
            if token == "--" {
                end_of_options = true;
                continue;
            }
            if token == "--help" {
                options.help = true;
                continue;
            }
            if token == "--version" {
                options.version = true;
                continue;
            }
            if token.starts_with('-') && token.len() > 1 {
                for option in token[1..].chars() {
                    match option {
                        'r' => options.recursive = true,
                        'q' => options.quiet = true,
                        'j' => options.junk_paths = true,
                        'd' => options.delete = true,
                        'h' => options.help = true,
                        'v' => options.version = true,
                        '0'..='9' => {
                            let level = option.to_digit(10).unwrap_or(0) as i64;
                            options.level = Some(level);
                            options.method = if level == 0 {
                                CompressionMethod::Stored
                            } else {
                                CompressionMethod::Deflated
                            };
                        }
                        other => {
                            return Err(format!(
                                "zip option '-{other}' is not supported by the internal fallback; install Info-ZIP zip for the full option set"
                            ));
                        }
                    }
                }
                continue;
            }
        }
        positional.push(token.clone());
    }
    if options.help || options.version {
        return Ok(options);
    }
    let Some(archive) = positional.first().cloned() else {
        return Err("zip: missing archive name".to_string());
    };
    options.archive = archive;
    options.inputs = positional.into_iter().skip(1).collect();
    if options.delete {
        if options.inputs.is_empty() {
            return Err("zip -d: missing entry names to delete".to_string());
        }
    } else if options.inputs.is_empty() {
        return Err("zip: nothing to do (no input files)".to_string());
    }
    Ok(options)
}

fn archive_path(base: &Path, raw: &str) -> PathBuf {
    let path = resolve_path(base, raw);
    // Info-ZIP appends `.zip` only when the name has no extension at all;
    // `zip backup.tar` keeps `backup.tar` rather than creating `backup.tar.zip`.
    if path.extension().is_some() {
        path
    } else {
        let mut name = path.into_os_string();
        name.push(".zip");
        PathBuf::from(name)
    }
}

fn run_add(
    archive: &Path,
    options: &Options,
    base: &Path,
    deadline: Instant,
) -> InternalOutcome {
    let collected = match collect_entries(options, base, archive, deadline) {
        Ok(collected) => collected,
        Err(outcome) => return outcome,
    };
    let mut warnings = collected.warnings;
    if collected.entries.is_empty() {
        warnings.push("zip error: Nothing to do! (no matching files)".to_string());
        return InternalOutcome::Failed {
            code: 12,
            stdout: String::new(),
            stderr: warnings.join("\n"),
        };
    }
    let temp_path = temp_path_for(archive);
    if let Err(outcome) = write_archive(archive, &temp_path, &collected.entries, options, deadline) {
        let _ = fs::remove_file(&temp_path);
        return outcome;
    }
    if let Err(err) = replace_file(&temp_path, archive) {
        let _ = fs::remove_file(&temp_path);
        return failure(format!(
            "zip: cannot write archive {}: {err}",
            archive.display()
        ));
    }
    let mut output = warnings.join("\n");
    if !options.quiet {
        if !output.is_empty() {
            output.push('\n');
        }
        for name in collected.entries.keys() {
            output.push_str(&format!("  adding: {name}\n"));
        }
    }
    InternalOutcome::Success(output)
}

fn run_delete(archive: &Path, options: &Options, deadline: Instant) -> InternalOutcome {
    let old_file = match File::open(archive) {
        Ok(file) => file,
        Err(err) => {
            return failure(format!(
                "zip: cannot open archive {}: {err}",
                archive.display()
            ));
        }
    };
    let mut old = match ZipArchive::new(old_file) {
        Ok(archive) => archive,
        Err(err) => {
            return failure(format!(
                "zip: cannot read archive {}: {err}",
                archive.display()
            ));
        }
    };
    let temp_path = temp_path_for(archive);
    let result = (|| -> Result<String, InternalOutcome> {
        let temp_file = File::create(&temp_path)
            .map_err(|err| failure(format!("zip: cannot create temporary archive: {err}")))?;
        let mut writer = ZipWriter::new(temp_file);
        let mut removed = 0usize;
        for index in 0..old.len() {
            if deadline_passed(deadline) {
                return Err(InternalOutcome::TimedOut);
            }
            let file = old
                .by_index_raw(index)
                .map_err(|err| failure(format!("zip: cannot read archive entry: {err}")))?;
            let name = file.name().to_string();
            if options
                .inputs
                .iter()
                .any(|pattern| wildcard_match(pattern, &name))
            {
                removed += 1;
                continue;
            }
            writer
                .raw_copy_file(file)
                .map_err(|err| failure(format!("zip: cannot copy archive entry: {err}")))?;
        }
        writer
            .finish()
            .map_err(|err| failure(format!("zip: cannot finish archive: {err}")))?;
        Ok(format!(
            "zip: removed {removed} entr{}",
            if removed == 1 { "y" } else { "ies" }
        ))
    })();
    // Release the source archive before replacing it; Windows refuses to
    // delete a file that still has an open handle.
    drop(old);
    match result {
        Ok(output) => {
            if let Err(err) = replace_file(&temp_path, archive) {
                let _ = fs::remove_file(&temp_path);
                failure(format!(
                    "zip: cannot write archive {}: {err}",
                    archive.display()
                ))
            } else {
                InternalOutcome::Success(output)
            }
        }
        Err(outcome) => {
            let _ = fs::remove_file(&temp_path);
            outcome
        }
    }
}

fn excluded(name: &str, options: &Options) -> bool {
    options
        .exclude
        .iter()
        .any(|pattern| wildcard_match(pattern, name))
}

/// Whether `path` is the archive being written. Both sides are canonicalized
/// when possible so `./out.zip` and `out.zip` compare equal during an update.
fn is_archive_path(path: &Path, archive: &Path) -> bool {
    if path == archive {
        return true;
    }
    // Most children cannot be the archive; skip the two canonicalize calls
    // unless the file names match.
    if path.file_name() != archive.file_name() {
        return false;
    }
    match (fs::canonicalize(path), fs::canonicalize(archive)) {
        (Ok(path), Ok(archive)) => path == archive,
        _ => false,
    }
}

fn collect_entries(
    options: &Options,
    base: &Path,
    archive: &Path,
    deadline: Instant,
) -> Result<CollectedEntries, InternalOutcome> {
    let mut entries = BTreeMap::new();
    let mut warnings = Vec::new();
    let mut ancestors = Vec::new();
    for input in &options.inputs {
        if deadline_passed(deadline) {
            return Err(InternalOutcome::TimedOut);
        }
        let resolved = resolve_path(base, input);
        // An update run walks the same directory that holds the archive. The
        // real `zip` never stores its own output file, so skip it explicitly;
        // otherwise the rebuilt archive would contain the previous archive.
        if is_archive_path(&resolved, archive) {
            continue;
        }
        let metadata = match fs::symlink_metadata(&resolved) {
            Ok(metadata) => metadata,
            Err(err) => {
                warnings.push(format!("zip warning: name not matched: {input} ({err})"));
                continue;
            }
        };
        let entry_root = match entry_root_for_path(base, &resolved, input, options.junk_paths) {
            Ok(name) => name,
            Err(message) => return Err(failure(message)),
        };
        if excluded(&entry_root, options) {
            continue;
        }
        if metadata.file_type().is_symlink() {
            match fs::metadata(&resolved) {
                Ok(target) if target.is_file() => {
                    entries.insert(entry_root, EntrySource::File(resolved));
                }
                Ok(target) if target.is_dir() => {
                    if !options.recursive {
                        warnings.push(format!("zip warning: {input} is a directory (use -r)"));
                        continue;
                    }
                    collect_dir(
                        &resolved,
                        &entry_root,
                        options,
                        archive,
                        &mut entries,
                        &mut warnings,
                        &mut ancestors,
                        deadline,
                    )?;
                }
                _ => warnings.push(format!("zip warning: skipping unsupported symlink {input}")),
            }
        } else if metadata.is_dir() {
            if !options.recursive {
                warnings.push(format!("zip warning: {input} is a directory (use -r)"));
                continue;
            }
            collect_dir(
                &resolved,
                &entry_root,
                options,
                archive,
                &mut entries,
                &mut warnings,
                &mut ancestors,
                deadline,
            )?;
        } else if metadata.is_file() {
            entries.insert(entry_root, EntrySource::File(resolved));
        } else {
            warnings.push(format!(
                "zip warning: skipping unsupported file type: {input}"
            ));
        }
    }
    Ok(CollectedEntries { entries, warnings })
}

fn collect_dir(
    dir: &Path,
    entry_root: &str,
    options: &Options,
    archive: &Path,
    entries: &mut BTreeMap<String, EntrySource>,
    warnings: &mut Vec<String>,
    ancestors: &mut Vec<PathBuf>,
    deadline: Instant,
) -> Result<(), InternalOutcome> {
    if deadline_passed(deadline) {
        return Err(InternalOutcome::TimedOut);
    }
    let canonical = fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    if ancestors.contains(&canonical) {
        warnings.push(format!(
            "zip warning: skipping recursive directory link {}",
            dir.display()
        ));
        return Ok(());
    }
    ancestors.push(canonical);
    let prefix = if entry_root == "." {
        String::new()
    } else {
        format!("{entry_root}/")
    };
    if !options.junk_paths && !prefix.is_empty() {
        entries.insert(prefix.clone(), EntrySource::Directory(dir.to_path_buf()));
    }
    let read_dir = match fs::read_dir(dir) {
        Ok(read_dir) => read_dir,
        Err(err) => {
            return Err(failure(format!(
                "zip: cannot read directory {}: {err}",
                dir.display()
            )));
        }
    };
    let mut children = Vec::new();
    for child in read_dir {
        match child {
            Ok(child) => children.push(child),
            Err(err) => {
                return Err(failure(format!(
                    "zip: cannot read directory {}: {err}",
                    dir.display()
                )));
            }
        }
    }
    children.sort_by_key(|child| child.file_name());
    for child in children {
        if deadline_passed(deadline) {
            return Err(InternalOutcome::TimedOut);
        }
        let file_name = child.file_name();
        let Some(file_name) = file_name.to_str() else {
            warnings.push(format!(
                "zip warning: skipping non-UTF-8 file name under {}",
                dir.display()
            ));
            continue;
        };
        let child_path = child.path();
        if is_archive_path(&child_path, archive) {
            continue;
        }
        let metadata = match fs::symlink_metadata(&child_path) {
            Ok(metadata) => metadata,
            Err(err) => {
                warnings.push(format!(
                    "zip warning: cannot stat {}: {err}",
                    child_path.display()
                ));
                continue;
            }
        };
        let entry_name = if options.junk_paths {
            file_name.to_string()
        } else {
            format!("{prefix}{file_name}")
        };
        if excluded(&entry_name, options) {
            continue;
        }
        if metadata.file_type().is_symlink() {
            match fs::metadata(&child_path) {
                Ok(target) if target.is_file() => {
                    entries.insert(entry_name, EntrySource::File(child_path));
                }
                Ok(target) if target.is_dir() => {
                    collect_dir(
                        &child_path,
                        &entry_name,
                        options,
                        archive,
                        entries,
                        warnings,
                        ancestors,
                        deadline,
                    )?;
                }
                _ => warnings.push(format!(
                    "zip warning: skipping unsupported symlink {}",
                    child_path.display()
                )),
            }
        } else if metadata.is_dir() {
            collect_dir(
                &child_path,
                &entry_name,
                options,
                archive,
                entries,
                warnings,
                ancestors,
                deadline,
            )?;
        } else if metadata.is_file() {
            entries.insert(entry_name, EntrySource::File(child_path));
        } else {
            warnings.push(format!(
                "zip warning: skipping unsupported file type: {}",
                child_path.display()
            ));
        }
    }
    ancestors.pop();
    Ok(())
}

fn entry_root_for_path(
    base: &Path,
    resolved: &Path,
    raw: &str,
    junk: bool,
) -> Result<String, String> {
    if junk {
        return resolved
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .ok_or_else(|| format!("zip: cannot determine a file name for '{raw}'"));
    }
    let name = match resolved.strip_prefix(base) {
        Ok(relative) => relative.to_path_buf(),
        Err(_) => {
            // Absolute input outside the working directory: store it without
            // the leading root, like Info-ZIP does.
            let mut components = resolved.components();
            match components.next() {
                Some(Component::RootDir) => components.as_path().to_path_buf(),
                #[cfg(windows)]
                Some(Component::Prefix(_)) => components.as_path().to_path_buf(),
                _ => resolved.to_path_buf(),
            }
        }
    };
    if name
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(format!(
            "zip: refusing to store a path containing '..': '{raw}'"
        ));
    }
    let text = name.to_string_lossy().replace('\\', "/");
    let text = text.trim_matches('/');
    if text.is_empty() || text == "." {
        Ok(".".to_string())
    } else {
        Ok(text.to_string())
    }
}

/// Convert a filesystem mtime into the DOS timestamp a zip entry stores.
fn modified_time(metadata: &fs::Metadata) -> Option<zip::DateTime> {
    let system_time = metadata.modified().ok()?;
    // ZIP stores wall-clock time without a time zone; using UTC here shifts
    // every entry by the local offset compared with Info-ZIP's output.
    let datetime: chrono::DateTime<chrono::Local> = system_time.into();
    zip::DateTime::from_date_and_time(
        u16::try_from(datetime.year()).ok()?,
        u8::try_from(datetime.month()).ok()?,
        u8::try_from(datetime.day()).ok()?,
        u8::try_from(datetime.hour()).ok()?,
        u8::try_from(datetime.minute()).ok()?,
        u8::try_from(datetime.second()).ok()?,
    )
    .ok()
}

fn write_archive(
    archive: &Path,
    temp_path: &Path,
    entries: &BTreeMap<String, EntrySource>,
    options: &Options,
    deadline: Instant,
) -> Result<(), InternalOutcome> {
    let temp_file = File::create(temp_path).map_err(|err| {
        failure(format!(
            "zip: cannot create temporary archive {}: {err}",
            temp_path.display()
        ))
    })?;
    let mut writer = ZipWriter::new(temp_file);
    if archive.exists() {
        let old_file = File::open(archive).map_err(|err| {
            failure(format!(
                "zip: cannot open archive {}: {err}",
                archive.display()
            ))
        })?;
        let mut old = ZipArchive::new(old_file).map_err(|err| {
            failure(format!(
                "zip: cannot read archive {}: {err}",
                archive.display()
            ))
        })?;
        for index in 0..old.len() {
            if deadline_passed(deadline) {
                return Err(InternalOutcome::TimedOut);
            }
            let file = old.by_index_raw(index).map_err(|err| {
                failure(format!(
                    "zip: cannot read archive {}: {err}",
                    archive.display()
                ))
            })?;
            if entries.contains_key(file.name()) {
                continue;
            }
            writer
                .raw_copy_file(file)
                .map_err(|err| failure(format!("zip: cannot copy archive entry: {err}")))?;
        }
    }
    for (name, source) in entries {
        if deadline_passed(deadline) {
            return Err(InternalOutcome::TimedOut);
        }
        match source {
            EntrySource::Directory(path) => {
                let metadata = fs::metadata(path).ok();
                let mut directory_options = SimpleFileOptions::default()
                    .compression_method(CompressionMethod::Stored)
                    .unix_permissions(
                        metadata
                            .as_ref()
                            .map(|metadata| unix_mode(metadata, 0o755))
                            .unwrap_or(0o755),
                    );
                if let Some(modified) = metadata.as_ref().and_then(modified_time) {
                    directory_options = directory_options.last_modified_time(modified);
                }
                writer.add_directory(name.as_str(), directory_options).map_err(|err| {
                    failure(format!("zip: cannot add directory '{name}': {err}"))
                })?;
            }
            EntrySource::File(path) => {
                let metadata = fs::metadata(path).map_err(|err| {
                    failure(format!("zip: cannot stat {}: {err}", path.display()))
                })?;
                let mut file_options = SimpleFileOptions::default()
                    .compression_method(options.method)
                    .unix_permissions(unix_mode(&metadata, 0o644))
                    .large_file(metadata.len() >= u32::MAX as u64)
                    .last_modified_time(modified_time(&metadata).unwrap_or_default());
                if let Some(level) = options.level {
                    file_options = file_options.compression_level(Some(level));
                }
                writer.start_file(name.as_str(), file_options).map_err(|err| {
                    failure(format!("zip: cannot add '{name}': {err}"))
                })?;
                let mut file = File::open(path).map_err(|err| {
                    failure(format!("zip: cannot open {}: {err}", path.display()))
                })?;
                match copy_with_deadline(&mut file, &mut writer, deadline) {
                    Ok(_) => {}
                    Err(CopyError::TimedOut) => return Err(InternalOutcome::TimedOut),
                    Err(CopyError::Io(err)) => {
                        return Err(failure(format!("zip: cannot write '{name}': {err}")));
                    }
                }
            }
        }
    }
    writer
        .finish()
        .map_err(|err| failure(format!("zip: cannot finish archive: {err}")))?;
    Ok(())
}

fn help_text() -> &'static str {
    "Usage: zip [-rqjd] [-0-9] archive.zip files... [-x patterns...]\n\
     Internal fallback used when Info-ZIP zip is not installed."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_option_is_reported() {
        let outcome = run(
            &["-Z".to_string(), "a.zip".to_string(), "a".to_string()],
            Path::new("."),
            Instant::now() + std::time::Duration::from_secs(30),
        );
        match outcome {
            InternalOutcome::Failed { stderr, .. } => {
                assert!(stderr.contains("not supported"), "{stderr}");
            }
            other => panic!("expected failure, got {other:?}"),
        }
    }
}
