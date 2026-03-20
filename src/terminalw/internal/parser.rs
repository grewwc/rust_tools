use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use rustc_hash::FxHashMap;

use crate::collections::{deque_list::DequeList, ordered_map::OrderedMap, ordered_set::OrderedSet};
use crate::common::types::FastSet;

use super::actiontype::ActionList;

#[path = "parser_impl.rs"]
mod parser_impl;

#[derive(Clone)]
struct ActionEntry {
    cond: Arc<dyn Fn(&Parser) -> bool + Send + Sync + 'static>,
    actions: Arc<Mutex<Vec<Arc<dyn Fn() + Send + Sync + 'static>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlagType {
    Bool,
    String,
    Int,
    Int64,
    Float64,
}

#[derive(Debug, Clone)]
struct FlagDef {
    name: String,
    usage: String,
    default_value: String,
    ty: FlagType,
}

#[derive(Clone)]
pub struct Parser {
    pub optional: OrderedMap<String, String>,
    pub positional: DequeList<String>,

    groups: OrderedMap<String, Parser>,
    flags: FxHashMap<String, FlagDef>,
    default_val_map: FxHashMap<String, String>,
    bool_option_set: FastSet<String>,
    alias_map: FxHashMap<String, String>,

    cmd: String,
    enable_parse_num: bool,
    num_arg: Option<i32>,

    actions: Vec<ActionEntry>,
    executed: bool,
}

pub type ParserOption = fn(&mut Parser);

pub fn disable_parser_number(p: &mut Parser) {
    p.enable_parse_num = false;
}

impl Parser {
    pub fn new_with_options(options: &[ParserOption]) -> Self {
        let mut p = Self::new();
        p.apply_options(options);
        p
    }

    pub fn new() -> Self {
        Self {
            optional: OrderedMap::new(),
            positional: DequeList::new(),
            groups: OrderedMap::new(),
            flags: FxHashMap::default(),
            default_val_map: FxHashMap::default(),
            bool_option_set: FastSet::default(),
            alias_map: FxHashMap::default(),
            cmd: String::new(),
            enable_parse_num: true,
            num_arg: None,
            actions: Vec::new(),
            executed: false,
        }
    }

    pub fn apply_options(&mut self, options: &[ParserOption]) {
        for op in options {
            op(self);
        }
    }

    pub fn add_group(&mut self, group_name: &str) -> &mut Parser {
        let sub = Parser::new();
        self.groups.insert(group_name.to_string(), sub);
        self.groups.get_mut(group_name).unwrap()
    }

    pub fn group_by_name(&mut self, group_name: &str) -> Option<&mut Parser> {
        self.groups.get_mut(group_name)
    }

    pub fn groups(&self) -> Vec<&Parser> {
        self.groups.values()
    }

    pub fn alias(&mut self, target: &str, original: &str) {
        self.alias_map
            .insert(original.to_string(), target.to_string());
        self.alias_map
            .insert(target.to_string(), original.to_string());
    }

    pub fn add_bool(&mut self, name: &str, value: bool, usage: &str) -> &mut Parser {
        self.define_flag(name, FlagType::Bool, value.to_string(), usage);
        self.bool_option_set.insert(name.to_string());
        self
    }

    pub fn add_string(&mut self, name: &str, value: &str, usage: &str) -> &mut Parser {
        self.define_flag(name, FlagType::String, value.to_string(), usage);
        self
    }

    pub fn add_int(&mut self, name: &str, value: i32, usage: &str) -> &mut Parser {
        self.define_flag(name, FlagType::Int, value.to_string(), usage);
        self
    }

    pub fn add_i64(&mut self, name: &str, value: i64, usage: &str) -> &mut Parser {
        self.define_flag(name, FlagType::Int64, value.to_string(), usage);
        self
    }

    pub fn add_f64(&mut self, name: &str, value: f64, usage: &str) -> &mut Parser {
        self.define_flag(name, FlagType::Float64, value.to_string(), usage);
        self
    }

    fn define_flag(&mut self, name: &str, ty: FlagType, default_value: String, usage: &str) {
        let name = name.trim_start_matches('-').to_string();
        self.default_val_map
            .insert(name.clone(), default_value.clone());
        self.flags.insert(
            name.clone(),
            FlagDef {
                name,
                usage: usage.to_string(),
                default_value,
                ty,
            },
        );
    }

    pub fn print_defaults(&self) {
        for (k, sub) in self.groups.iter() {
            println!("{k}");
            sub.print_defaults();
        }
        let mut names = self.flags.keys().cloned().collect::<Vec<_>>();
        names.sort();
        for name in names {
            let Some(def) = self.flags.get(&name) else {
                continue;
            };
            if def.usage.trim().is_empty() {
                println!("-{} (default: {})", def.name, def.default_value);
            } else {
                println!(
                    "-{} (default: {}) {}",
                    def.name, def.default_value, def.usage
                );
            }
        }
    }

    pub fn cmd(&self) -> &str {
        &self.cmd
    }

    pub fn num_args(&self) -> i32 {
        self.num_arg.unwrap_or(-1)
    }

    pub fn is_empty(&self) -> bool {
        self.optional.is_empty() && self.positional.is_empty() && self.num_arg.is_none()
    }

    pub fn flag_value(&self, flag_name: &str) -> Result<String, String> {
        let key = parser_impl::normalize_flag_key(flag_name);
        if let Some(v) = self.optional.get(&key) {
            return Ok(v.clone());
        }
        if let Some(alias) = self.alias_map.get(key.trim_start_matches('-')) {
            return self.flag_value(alias);
        }
        Err(format!("GetFlagVal: flagName ({}) not exist", key))
    }

    pub fn flag_value_or_default(&self, flag_name: &str) -> String {
        self.flag_value(flag_name)
            .unwrap_or_else(|_| self.default_value(flag_name))
    }

    pub fn flag_value_with_default(&self, flag_name: &str, default_val: &str) -> String {
        let key = parser_impl::normalize_flag_key(flag_name);
        let v = self
            .optional
            .get(&key)
            .cloned()
            .unwrap_or_else(|| default_val.to_string());
        if v != default_val {
            return v;
        }
        let Some(alias) = self.alias_map.get(key.trim_start_matches('-')) else {
            return v;
        };
        let alias_key = parser_impl::normalize_flag_key(alias);
        self.optional
            .get(&alias_key)
            .cloned()
            .unwrap_or_else(|| default_val.to_string())
    }

    pub fn default_value(&self, key: &str) -> String {
        let name = key.trim_start_matches('-');
        if let Some(v) = self.default_val_map.get(name) {
            return v.clone();
        }
        if let Some(alias) = self.alias_map.get(name) {
            return self.default_val_map.get(alias).cloned().unwrap_or_default();
        }
        String::new()
    }

    pub fn set_flag_value(&mut self, flag_name: &str, val: &str) {
        let key = parser_impl::normalize_flag_key(flag_name);
        self.optional.insert(key, val.to_string());
    }

    pub fn remove_flag_value(&mut self, flag_name: &str) {
        let key = parser_impl::normalize_flag_key(flag_name);
        self.optional.remove(&key);
        if let Some(alias) = self.alias_map.get(key.trim_start_matches('-')).cloned() {
            let alias_key = parser_impl::normalize_flag_key(&alias);
            self.optional.remove(&alias_key);
        }
    }

    pub fn multi_flag_value_with_default(&self, flag_names: &[&str], default_val: &str) -> String {
        for name in flag_names {
            if self.contains_flag_strict(name) {
                return self.flag_value_or_default(name);
            }
        }
        default_val.to_string()
    }

    pub fn flag_value_i32(&self, flag_name: &str) -> i32 {
        self.flag_value_or_default(flag_name)
            .parse::<i32>()
            .unwrap()
    }

    pub fn flag_value_i64(&self, flag_name: &str) -> i64 {
        self.flag_value_or_default(flag_name)
            .parse::<i64>()
            .unwrap()
    }

    pub fn flag_value_int(&self, flag_name: &str) -> i32 {
        self.flag_value_i32(flag_name)
    }

    pub fn flag_value_int_or(&self, flag_name: &str, val: i32) -> i32 {
        if self.contains_flag_strict(flag_name) {
            return self.flag_value_i32(flag_name);
        }
        val
    }

    pub fn positional_args(&mut self, exclude_num_arg: bool) -> Vec<String> {
        if exclude_num_arg {
            if let Some(n) = self.num_arg {
                let remove = format!("-{}", n);
                self.positional.remove_first(|x| x == &remove);
            }
        }
        self.positional.to_vec()
    }

    pub fn contains_flag(&self, flag_name: &str) -> bool {
        let needle = flag_name.trim_start_matches('-');
        let mut buf = String::new();
        for (k, _) in self.optional.iter() {
            buf.push_str(k);
        }
        let alias = self
            .alias_map
            .get(needle)
            .cloned()
            .unwrap_or_else(|| needle.to_string());
        buf.contains(needle) || buf.contains(&alias)
    }

    pub fn contains_flag_strict(&self, flag_name: &str) -> bool {
        let key = parser_impl::normalize_flag_key(flag_name);
        if self.optional.contains_key(&key) {
            return true;
        }
        let Some(alias) = self.alias_map.get(key.trim_start_matches('-')) else {
            return false;
        };
        let alias_key = parser_impl::normalize_flag_key(alias);
        self.optional.contains_key(&alias_key)
    }

    pub fn contains_any_flag_strict(&self, flag_names: &[&str]) -> bool {
        flag_names.iter().any(|f| self.contains_flag_strict(f))
    }

    pub fn contains_all_flag_strict(&self, flag_names: &[&str]) -> bool {
        flag_names.iter().all(|f| self.contains_flag_strict(f))
    }

    pub fn coexists(&self, args: &[&str]) -> bool {
        for arg in args {
            if arg.trim().is_empty() {
                continue;
            }
            let key = parser_impl::normalize_flag_key(arg);
            if !self.optional.contains_key(&key) {
                return false;
            }
        }
        true
    }

    pub fn flags(&self) -> OrderedSet<String> {
        let mut out = OrderedSet::new();
        for k in self.optional.keys() {
            out.insert(k.clone());
        }
        out
    }

    pub fn boolean_args(&self) -> OrderedSet<String> {
        let mut out = OrderedSet::new();
        for (k, v) in self.optional.iter() {
            if v.is_empty() {
                out.insert_str(k.trim_start_matches('-'));
                out.insert(k.clone());
            }
        }
        out
    }

    pub fn on<F>(&mut self, condition: F) -> ActionList
    where
        F: Fn(&Parser) -> bool + Send + Sync + 'static,
    {
        let actions = Arc::new(Mutex::new(Vec::new()));
        self.actions.push(ActionEntry {
            cond: Arc::new(condition),
            actions: Arc::clone(&actions),
        });
        ActionList { actions }
    }

    pub fn execute(&mut self) {
        if self.executed {
            return;
        }
        self.executed = true;
        for entry in self.actions.iter() {
            if !(entry.cond)(self) {
                continue;
            }
            let actions = entry
                .actions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            for a in actions {
                a();
            }
        }
    }

    pub fn execute_first(&mut self) -> bool {
        if self.executed {
            return false;
        }
        self.executed = true;
        for entry in self.actions.iter() {
            if !(entry.cond)(self) {
                continue;
            }
            let actions = entry
                .actions
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            for a in actions {
                a();
            }
            return true;
        }
        false
    }

    pub fn parse_args_cmd(&mut self, bool_optionals: &[&str]) {
        parser_impl::parse_args_cmd(self, bool_optionals);
    }

    pub fn parse_args(&mut self, cmd: &str, bool_optionals: &[&str]) {
        parser_impl::parse_args(self, cmd, bool_optionals);
    }

    pub fn parse_argv(&mut self, argv: &[String], bool_optionals: &[&str]) {
        parser_impl::parse_argv(self, argv, bool_optionals);
    }
}

pub(crate) type EncodedArgs = (VecDeque<String>, Vec<String>, Vec<String>, Vec<String>);
