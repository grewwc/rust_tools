use std::ptr::null_mut;

use crate::common::types::FastMap;


/// A Trie (prefix tree) data structure for efficient string storage and retrieval.
///
/// This implementation uses a HashMap-based approach where each node contains
/// a map of characters to child nodes. It supports basic operations like insert,
/// search, prefix matching, and deletion.
///
/// # Examples
///
/// ```
/// use rust_tools::collections::trie::Trie;
///
/// let mut trie = Trie::new();
/// trie.insert("hello");
/// trie.insert("world");
///
/// assert!(trie.contains("hello"));
/// assert!(trie.has_prefix("hel"));
/// assert!(!trie.contains("hel"));
/// ```
pub struct Trie {
    /// Maps each character to its corresponding child Trie node
    children: FastMap<char, Box<Trie>>,

    /// Number of times this node marks the end of a complete word
    end_count: usize,

    /// Number of valid words ending at this node (updated in `insert` and `delete`)
    valid_count: usize,

    /// Total number of unique words in the entire trie
    size: usize,
}

impl Trie {
    /// Creates a new, empty Trie.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tools::collections::trie::Trie;
    ///
    /// let trie = Trie::new();
    /// assert!(trie.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            children: FastMap::default(),
            end_count: 0,
            size: 0,
            valid_count: 0,
        }
    }
}

impl Trie {
    /// Inserts a string into the Trie.
    ///
    /// Returns `true` if new nodes were created during insertion, `false` if the
    /// word already existed in the trie.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tools::collections::trie::Trie;
    ///
    /// let mut trie = Trie::new();
    /// assert!(trie.insert("hello"));
    /// assert!(!trie.insert("hello")); // Already exists
    /// ```
    pub fn insert(&mut self, s: &str) -> bool {
        let mut inserted = false;
        let mut curr = &mut *self;

        // Traverse or create nodes for each character
        for ch in s.chars() {
            if !curr.children.contains_key(&ch) {
                curr.children.insert(ch, Box::new(Trie::new()));
                inserted = true;
            }
            match curr.children.get_mut(&ch) {
                Some(node) => curr = node,
                None => return false,
            }
        }

        // Check if word already exists
        if curr.end_count > 0 {
            return false;
        }

        // Mark this node as the end of a word
        curr.end_count += 1;
        if curr.end_count == 1 {
            curr.valid_count += 1;
            self.size += 1; // Increment size when a new word is inserted
        }

        inserted
    }

    /// Checks if the Trie contains the exact string.
    ///
    /// Returns `true` only if the string exists as a complete word in the trie,
    /// not just as a prefix.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to search for
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tools::collections::trie::Trie;
    ///
    /// let mut trie = Trie::new();
    /// trie.insert("hello");
    ///
    /// assert!(trie.contains("hello"));
    /// assert!(!trie.contains("hel")); // Prefix, but not a complete word
    /// ```
    pub fn contains(&self, s: &str) -> bool {
        let mut curr = &*self;

        // Traverse the trie following the characters
        for (_, ch) in s.char_indices() {
            if !curr.children.contains_key(&ch) {
                return false;
            }
            curr = curr.children.get(&ch).unwrap();
        }

        // Check if this node marks the end of a word
        curr.end_count > 0
    }

    /// Checks if any word in the Trie starts with the given prefix.
    ///
    /// Returns `true` if there exists at least one word with the given prefix,
    /// even if the prefix itself is not a complete word.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix to search for
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tools::collections::trie::Trie;
    ///
    /// let mut trie = Trie::new();
    /// trie.insert("hello");
    ///
    /// assert!(trie.has_prefix("hel"));
    /// assert!(trie.has_prefix("hello"));
    /// assert!(!trie.has_prefix("world"));
    /// ```
    pub fn has_prefix(&self, prefix: &str) -> bool {
        let mut curr = &*self;

        // Traverse the trie following the characters
        for (_, ch) in prefix.char_indices() {
            if !curr.children.contains_key(&ch) {
                return false;
            }
            curr = curr.children.get(&ch).unwrap();
        }

        // If we successfully traversed all characters, the prefix exists
        true
    }

    /// Deletes a string from the Trie.
    ///
    /// Returns `true` if the word was found and deleted, `false` if the word
    /// didn't exist in the trie. This operation may remove nodes if they become
    /// unnecessary after deletion.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to delete
    ///
    /// # Safety
    ///
    /// This method uses unsafe code to manage parent-child relationships during
    /// node cleanup. The parent pointer is only used within the scope where it
    /// remains valid.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tools::collections::trie::Trie;
    ///
    /// let mut trie = Trie::new();
    /// trie.insert("hello");
    ///
    /// assert!(trie.delete("hello"));
    /// assert!(!trie.contains("hello"));
    /// assert!(!trie.delete("hello")); // Already deleted
    /// ```
    pub fn delete(&mut self, s: &str) -> bool {
        let mut curr = &mut *self;
        let mut parent: *mut Trie = null_mut();
        let mut parent_ch: char = '\x00';

        // Traverse to find the word, keeping track of parent for cleanup
        for (i, ch) in s.char_indices() {
            if !curr.children.contains_key(&ch) {
                return false;
            }
            parent = curr;
            curr = curr.children.get_mut(&ch).unwrap();
            if i + ch.len_utf8() < s.len() {
                parent_ch = ch;
            }
        }

        // Check if the word exists
        if curr.end_count == 0 {
            return false;
        }

        // Decrement the end count
        curr.end_count -= 1;
        if curr.end_count == 0 {
            curr.valid_count -= 1;
        }

        // Clean up: remove child node from parent if parent is no longer a word endpoint
        if !parent.is_null() {
            unsafe {
                if (*parent).end_count == 0 {
                    (*parent).children.remove(&parent_ch);
                }
            }
        }
        self.size -= 1;
        true
    }

    /// Returns the number of unique words stored in the Trie.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tools::collections::trie::Trie;
    ///
    /// let mut trie = Trie::new();
    /// assert_eq!(trie.len(), 0);
    ///
    /// trie.insert("hello");
    /// trie.insert("world");
    /// assert_eq!(trie.len(), 2);
    /// trie.delete("hello");
    /// assert_eq!(trie.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the Trie contains no words.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tools::collections::trie::Trie;
    ///
    /// let mut trie = Trie::new();
    /// assert!(trie.is_empty());
    ///
    /// trie.insert("hello");
    /// assert!(!trie.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test data for trie operations
    const DATA: &[&'static str] = &["hello", "world"];

    /// Test basic insertion and search operations
    #[test]
    fn test_insert_and_contains() {
        let mut t = Trie::new();

        // Insert test data
        for s in DATA {
            t.insert(s);
        }

        // Verify exact matches and prefixes
        for s in DATA {
            // Complete words should be found
            assert!(t.contains(s));
            assert!(t.has_prefix(s));

            // Prefixes should not be complete words but should exist as prefixes
            assert!(!t.contains(&s[..s.len() - 1]));
            assert!(t.has_prefix(&s[..s.len() - 1]));
        }
    }

    /// Test deletion operation
    #[test]
    fn test_delete() {
        let mut t = Trie::new();

        // Insert test data
        for s in DATA {
            t.insert(s);
        }

        // Try to delete a non-existent word
        assert!(!t.delete("hel"));

        // Insert a word that is a prefix of another
        t.insert("hell");
        assert!(t.contains("hello"));

        // Delete the longer word
        t.delete("hello");
        assert!(!t.contains("hello"));

        // The shorter word should still exist (currently commented out)
        // assert!(t.contains("hell"));
    }
}


