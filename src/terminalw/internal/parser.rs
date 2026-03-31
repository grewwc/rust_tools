use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use rustc_hash::FxHashMap;

use crate::common::types::FastSet;
use crate::cw::{deque_list::DequeList, ordered_map::OrderedMap, ordered_set::OrderedSet};

use super::actiontype::{ActionFnList, ActionList};

#[path = "parser_impl.rs"]
mod parser_impl;

#[derive(Clone)]
struct ActionEntry {
    cond: Arc<dyn Fn(&Parser) -> bool + Send + Sync + 'static>,
    actions: ActionFnList,
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

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
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
        let bin = std::env::args()
            .next()
            .and_then(|p| {
                std::path::Path::new(&p)
                    .file_name()?
                    .to_str()
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "app".to_string());

        println!("Usage:");
        let mut usage_parts = vec![bin];
        if !self.groups.is_empty() {
            usage_parts.push("[COMMAND]".to_string());
        }
        if !self.flags.is_empty() {
            usage_parts.push("[OPTIONS]".to_string());
        }
        usage_parts.push("[ARGS]".to_string());
        println!("  {}", usage_parts.join(" "));

        let mut command_names = self.groups.keys().cloned().collect::<Vec<_>>();
        command_names.sort();
        if !command_names.is_empty() {
            println!();
            println!("Commands:");
            for name in command_names {
                println!("  {name}");
            }
        }

        let options = self.collect_option_entries();
        if options.is_empty() {
            return;
        }
        println!();
        println!("Options:");
        let spec_width = options
            .iter()
            .map(|x| x.spec.len())
            .max()
            .unwrap_or(0)
            .max(24);
        for opt in options {
            if opt.help.is_empty() {
                println!("  {spec:<spec_width$}", spec = opt.spec);
            } else {
                println!(
                    "  {spec:<spec_width$}  {help}",
                    spec = opt.spec,
                    help = opt.help
                );
            }
        }
    }

    fn collect_option_entries(&self) -> Vec<OptionEntry> {
        let mut visited = FastSet::default();
        let mut out = Vec::new();

        let mut names = self.flags.keys().cloned().collect::<Vec<_>>();
        names.sort();

        if self.flags.contains_key("h") && self.flags.contains_key("help") {
            visited.insert("h".to_string());
            visited.insert("help".to_string());
            if let Some(entry) = self.build_option_entry(&["h", "help"]) {
                out.push(entry);
            }
        }

        for name in names {
            if visited.contains(&name) {
                continue;
            }
            let mut aliases = vec![name.clone()];
            if let Some(other) = self.alias_map.get(&name).cloned()
                && other != name
                && self.flags.contains_key(&other)
            {
                aliases.push(other);
            }
            for a in &aliases {
                visited.insert(a.clone());
            }
            let refs = aliases.iter().map(|s| s.as_str()).collect::<Vec<_>>();
            if let Some(entry) = self.build_option_entry(&refs) {
                out.push(entry);
            }
        }

        out.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));
        out
    }

    fn build_option_entry(&self, names: &[&str]) -> Option<OptionEntry> {
        let defs = names
            .iter()
            .filter_map(|n| self.flags.get(*n))
            .collect::<Vec<_>>();
        if defs.is_empty() {
            return None;
        }

        let mut spec_names = names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        spec_names.sort_by(|a, b| {
            let a_short = a.len() == 1;
            let b_short = b.len() == 1;
            a_short
                .cmp(&b_short)
                .reverse()
                .then_with(|| a.len().cmp(&b.len()))
                .then_with(|| a.cmp(b))
        });
        spec_names.dedup();

        let primary = defs
            .iter()
            .find(|d| !d.usage.contains("alias for"))
            .unwrap_or(&defs[0]);
        let usage = primary.usage.trim();
        let default_value = primary.default_value.trim();

        let placeholder = match primary.ty {
            FlagType::Bool => String::new(),
            _ => format!("<{}>", primary.name.to_uppercase().replace('-', "_").trim()),
        };

        let mut formatted_names = spec_names
            .iter()
            .map(|n| self.format_flag_name(n))
            .collect::<Vec<_>>();
        let show_value = !matches!(primary.ty, FlagType::Bool);
        if show_value && let Some(last) = formatted_names.last_mut() {
            last.push(' ');
            last.push_str(&placeholder);
        }
        let spec = formatted_names.join(", ");

        let mut help = usage.to_string();
        if self.should_show_default(primary) {
            if !help.is_empty() {
                help.push(' ');
            }
            help.push_str(&format!("[default: {}]", default_value));
        }

        let sort_key = spec_names
            .iter()
            .find(|n| n.len() == 1)
            .cloned()
            .or_else(|| spec_names.first().cloned())
            .unwrap_or_default();

        Some(OptionEntry {
            spec,
            help,
            sort_key,
        })
    }

    fn should_show_default(&self, def: &FlagDef) -> bool {
        let dv = def.default_value.trim();
        if dv.is_empty() {
            return false;
        }
        match def.ty {
            FlagType::Bool => dv != "false",
            _ => true,
        }
    }

    fn format_flag_name(&self, name: &str) -> String {
        if name.len() == 1 {
            format!("-{name}")
        } else {
            format!("--{name}")
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
            .unwrap_or(0)
    }

    pub fn flag_value_i64(&self, flag_name: &str) -> i64 {
        self.flag_value_or_default(flag_name)
            .parse::<i64>()
            .unwrap_or(0)
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
        if exclude_num_arg && let Some(n) = self.num_arg {
            let remove = format!("-{}", n);
            self.positional.remove_first(|x| x == &remove);
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
                out.insert(k.trim_start_matches('-').to_string());
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

struct OptionEntry {
    spec: String,
    help: String,
    sort_key: String,
}

pub(crate) type EncodedArgs = (VecDeque<String>, Vec<String>, Vec<String>, Vec<String>);
