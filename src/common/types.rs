use std::{collections::HashMap, hash::BuildHasherDefault};

use rustc_hash::FxHasher;

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

