//! Shared `/history rewind` grammar and rewindable-user traversal.
//!
//! The local REPL and the serve client resolve targets against different
//! history shapes (full messages, remote messages, and roles-only metadata).
//! The target grammar and the definition of the user ordinal must stay
//! identical across those shapes.

/// Parsed `/history rewind` target, independent of where history is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HistoryRewindTarget {
    UserOrdinal(usize),
    LatestUser,
    Grep(String),
}

/// Parse the shared `/history rewind` grammar.
///
/// This performs syntax and ordinal validation only; target existence is
/// checked by each history source.
pub(crate) fn parse_history_rewind_target(args: &[&str]) -> Result<HistoryRewindTarget, String> {
    let Some(first) = args.first().copied() else {
        return Err(
            "missing rewind target. try: /history rewind last or /history rewind u3".into(),
        );
    };
    if first == "last" || first == "latest" {
        return Ok(HistoryRewindTarget::LatestUser);
    }
    if first == "grep" {
        let keyword = args[1..].join(" ");
        if keyword.trim().is_empty() {
            return Err("`/history rewind grep` requires a keyword".into());
        }
        return Ok(HistoryRewindTarget::Grep(keyword));
    }
    let raw = first
        .strip_prefix('u')
        .or_else(|| first.strip_prefix('U'))
        .unwrap_or(first);
    let ordinal = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid rewind target: {first}. try: /history rewind u3"))?;
    if ordinal == 0 {
        return Err("user ordinal must be >= 1".into());
    }
    Ok(HistoryRewindTarget::UserOrdinal(ordinal))
}

/// Enumerate rewindable user rows as `(1-based ordinal, canonical index, item)`.
///
/// Keeping the ordinal increment and the canonical index in one iterator
/// prevents callers from counting a row they later exclude (or vice versa).
pub(crate) fn real_user_ordinal_entries<'a, T>(
    items: &'a [T],
    mut is_real_user: impl FnMut(&T) -> bool,
) -> impl Iterator<Item = (usize, usize, &'a T)> {
    let mut ordinal = 0usize;
    items.iter().enumerate().filter_map(move |(index, item)| {
        if !is_real_user(item) {
            return None;
        }
        ordinal += 1;
        Some((ordinal, index, item))
    })
}

/// Count rewindable user rows using the same predicate as
/// [`real_user_ordinal_entries`].
pub(crate) fn count_real_users<T>(items: &[T], is_real_user: impl Fn(&T) -> bool) -> usize {
    items.iter().filter(|item| is_real_user(item)).count()
}

/// Canonical index of the Nth rewindable user row.
pub(crate) fn real_user_index<T>(
    items: &[T],
    ordinal: usize,
    is_real_user: impl Fn(&T) -> bool,
) -> Option<usize> {
    if ordinal == 0 {
        return None;
    }
    real_user_ordinal_entries(items, is_real_user)
        .find_map(|(seen, index, _)| (seen == ordinal).then_some(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_shared_rewind_grammar() {
        assert_eq!(
            parse_history_rewind_target(&["last"]).unwrap(),
            HistoryRewindTarget::LatestUser
        );
        assert_eq!(
            parse_history_rewind_target(&["U7"]).unwrap(),
            HistoryRewindTarget::UserOrdinal(7)
        );
        assert_eq!(
            parse_history_rewind_target(&["grep", "wrong", "turn"]).unwrap(),
            HistoryRewindTarget::Grep("wrong turn".to_string())
        );
        assert_eq!(
            parse_history_rewind_target(&[]).unwrap_err(),
            "missing rewind target. try: /history rewind last or /history rewind u3"
        );
        assert_eq!(
            parse_history_rewind_target(&["u0"]).unwrap_err(),
            "user ordinal must be >= 1"
        );
        assert_eq!(
            parse_history_rewind_target(&["grep"]).unwrap_err(),
            "`/history rewind grep` requires a keyword"
        );
    }

    #[test]
    fn ordinal_entries_skip_non_users() {
        let items = [("user", 1), ("assistant", 2), ("user", 3)];
        let entries: Vec<_> = real_user_ordinal_entries(&items, |item| item.0 == "user")
            .map(|(ordinal, index, item)| (ordinal, index, item.1))
            .collect();
        assert_eq!(entries, vec![(1, 0, 1), (2, 2, 3)]);
        assert_eq!(count_real_users(&items, |item| item.0 == "user"), 2);
        assert_eq!(real_user_index(&items, 2, |item| item.0 == "user"), Some(2));
        assert_eq!(real_user_index(&items, 0, |item| item.0 == "user"), None);
    }
}
