use std::{collections::{HashMap, HashSet}, hash::BuildHasherDefault};

use rustc_hash::FxHasher;

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FastSet<T> = HashSet<T, BuildHasherDefault<FxHasher>>;


