use super::super::inline_recovery::normalize_tool_call_arguments;
use super::*;
use crate::ai::{
    cli::ParsedCli,
    ports::stream::StreamFilter,
    tools::os_tools::{GLOBAL_OS, init_os_tools_globals},
    types::{App, AppConfig},
};
use std::io::Read as _;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool, mpsc};

const REPORTED_FULLWIDTH_DSML_TOOL_CALL: &str = r#"<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="read_file">
<｜｜DSML｜｜parameter name="file_path" string="true">/Users/bytedance/rust_tools/src/bin/ai/driver/turn_runtime/iteration.rs</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="limit" string="false">80</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="offset" string="false">110</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>"#;

struct PrefixStreamFilter;

impl StreamFilter for PrefixStreamFilter {
    fn filter(&self, chunk: &str) -> Option<String> {
        Some(format!("filtered:{chunk}"))
    }

    fn name(&self) -> &str {
        "prefix"
    }
}


fn test_app() -> App {
    App {
        cli: ParsedCli::default(),
        scoped_preflight_required: Vec::new(),
        hooks: Default::default(),
        config: AppConfig {
            api_key: String::new(),
            base_history_file: PathBuf::new(),
            history_file: PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: String::new(),
            history_max_chars: 0,
            history_keep_last: 0,
            history_summary_max_chars: 0,
            intent_model: None,
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::builder().build().unwrap(),
        current_model: String::new(),
        current_agent: String::new(),
        current_agent_manifest: None,
        pending_files: None,
        forced_skills: Vec::new(),
        forced_skill_source: None,
        pending_skill_continuation: None,
        forced_question: None,
        attached_image_files: Vec::new(),
        shutdown: Arc::new(AtomicBool::new(false)),
        streaming: Arc::new(AtomicBool::new(false)),
        cancel_stream: Arc::new(AtomicBool::new(false)),
        ignore_next_prompt_interrupt: false,
        prompt_editor: None,
        agent_context: None,
        last_skill_bias: None,
        os: crate::ai::driver::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(
            crate::ai::driver::thinking::ThinkingOrchestrator::new(),
        )],
        last_known_prompt_tokens: None,
        last_known_cached_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
        turn_reasoning_items: Default::default(),
        stale_patch_targets: Default::default(),
        tool_middlewares: Vec::new(),
        llm_middlewares: Vec::new(),
    }
}

fn write_http_chunk(stream: &mut std::net::TcpStream, payload: &str) -> std::io::Result<()> {
    write!(stream, "{:X}\r\n", payload.len())?;
    stream.write_all(payload.as_bytes())?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

struct SavedColumns(Option<std::ffi::OsString>);
impl Drop for SavedColumns {
    fn drop(&mut self) {
        unsafe {
            match &self.0 {
                Some(value) => std::env::set_var("COLUMNS", value),
                None => std::env::remove_var("COLUMNS"),
            }
        }
    }
}

mod metrics;
mod terminal_dedupe;
mod degenerate;
mod interrupt;
mod closing_marker;
mod inline_recovery;
mod payload_processing;
mod thinking_fold;
mod fold_display;
mod cancelled_stream;
mod snapshot_done;
mod stream_response;
mod demuxer;
mod golden_wire;
