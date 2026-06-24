use uuid::Uuid;

use crate::ai::{
    history::SessionStore,
    persona::{self, PersonaDeleteResult, PersonaProfile, PersonaStore},
    types::App,
};
use crate::commonw::prompt::{prompt_yes_or_no_interruptible, read_line};

pub fn try_handle_persona_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "personas" && cmd != "persona" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");
    let remainder = parts.collect::<Vec<_>>().join(" ");
    let store = PersonaStore::new();

    match action {
        "help" | "h" => {
            println!("Persona management commands:");
            println!();
            println!("  /personas                 list personas");
            println!("  /personas list            list personas");
            println!("  /personas current         show current persona");
            println!("  /personas create          interactively create a persona");
            println!("  /personas use <name|id>   switch to a persona");
            println!("  /personas delete <name|id> delete a persona and its data");
            println!();
            println!("Notes:");
            println!("  - Each persona uses isolated session history and memory.");
            println!("  - Avatar is optional, but if set it must be unique.");
            println!();
        }
        "list" | "ls" | "" => match store.list_personas() {
            Ok(personas) => print_persona_list(app, &personas),
            Err(err) => eprintln!("Failed to list personas: {}", err),
        },
        "current" | "cur" => {
            print_current_persona(app);
        }
        "create" | "new" => {
            create_persona_interactively(app, &store);
        }
        "use" | "select" | "switch" => {
            let selector = remainder.trim();
            if selector.is_empty() {
                println!("missing persona selector. try: /personas use <name|id>");
                return Ok(true);
            }
            match store.set_active_persona(selector) {
                Ok(persona) => {
                    if persona.id == app.active_persona.id {
                        println!("Persona already active: {}", persona.name);
                    } else {
                        let old_name = app.active_persona.name.clone();
                        switch_to_persona(app, persona);
                        println!(
                            "Switched persona: {} -> {}",
                            old_name, app.active_persona.name
                        );
                    }
                }
                Err(err) => {
                    eprintln!("Failed to switch persona: {}", err);
                }
            }
        }
        "delete" | "del" | "rm" => {
            let selector = remainder.trim();
            if selector.is_empty() {
                println!("missing persona selector. try: /personas delete <name|id>");
                return Ok(true);
            }
            let confirm = prompt_yes_or_no_interruptible(&format!(
                "Delete persona '{selector}' and all its session/memory data? (y/n): ",
            ));
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }
            match store.delete_persona(selector) {
                Ok(result) => {
                    handle_deleted_persona(app, result);
                }
                Err(err) => {
                    eprintln!("Failed to delete persona: {}", err);
                }
            }
        }
        _ => {
            println!("unknown action: '{}'. try: /personas help", action);
        }
    }
    Ok(true)
}

fn create_persona_interactively(app: &mut App, store: &PersonaStore) {
    println!("Create persona:");
    let name = read_line("  Name: ");
    if name.trim().is_empty() {
        println!("canceled: empty persona name.");
        return;
    }

    let avatar = read_line("  Avatar (optional, unique): ");
    println!("  Persona prompt (multi-line). Submit to save; empty cancels.");
    app.refresh_prompt_editor_for_current_session();
    if let Some(editor) = app.prompt_editor.as_mut() {
        editor.set_prefill(format!(
            "You are {}.\n\nCore traits:\n- \n\nResponse style:\n- \n\nBoundaries:\n- \n",
            name.trim()
        ));
    }
    let prompt = app
        .prompt_editor
        .as_mut()
        .and_then(|editor| editor.read_multi_line().ok().flatten())
        .unwrap_or_default();
    if prompt.trim().is_empty() {
        println!("canceled: empty persona prompt.");
        return;
    }

    match store.create_persona(&name, Some(&avatar), &prompt) {
        Ok(persona) => match store.set_active_persona(&persona.id) {
            Ok(active_persona) => {
                let old_name = app.active_persona.name.clone();
                switch_to_persona(app, active_persona);
                println!("Created persona: {}", app.active_persona.name);
                println!(
                    "Switched persona: {} -> {}",
                    old_name, app.active_persona.name
                );
            }
            Err(err) => eprintln!("Created persona but failed to activate it: {}", err),
        },
        Err(err) => eprintln!("Failed to create persona: {}", err),
    }
}

fn handle_deleted_persona(app: &mut App, result: PersonaDeleteResult) {
    let removed_is_current = result.removed.id == app.active_persona.id;
    if let Err(err) =
        persona::cleanup_persona_storage(app.config.base_history_file.as_path(), &result.removed.id)
    {
        eprintln!(
            "[persona] deleted '{}' from registry, but failed to clean files: {}",
            result.removed.name, err
        );
    }

    if removed_is_current {
        let removed_name = result.removed.name.clone();
        switch_to_persona(app, result.active_after);
        println!("Deleted persona: {}", removed_name);
        println!("Switched to fallback persona: {}", app.active_persona.name);
    } else {
        println!("Deleted persona: {}", result.removed.name);
    }
}

fn switch_to_persona(app: &mut App, persona: PersonaProfile) {
    crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
    crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.tools.clear();
    }
    app.attached_image_files.clear();
    app.forced_skill = None;
    app.last_skill_bias = None;
    app.active_persona = persona.clone();
    app.config.history_file =
        persona::history_file_for_persona(app.config.base_history_file.as_path(), &persona.id);

    let session_id = persona
        .last_session_id
        .clone()
        .filter(|id| !id.trim().is_empty())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let session_store = SessionStore::new(app.config.history_file.as_path());
    app.session_id = session_id.clone();
    app.session_history_file = session_store.session_history_file(&session_id);
    app.sync_persona_session_binding();
}

fn print_persona_list(app: &App, personas: &[PersonaProfile]) {
    println!("Available personas:\n");
    for persona in personas {
        let mark = if persona.id == app.active_persona.id {
            "*"
        } else {
            " "
        };
        let avatar = if persona.avatar.trim().is_empty() {
            "-"
        } else {
            persona.avatar.trim()
        };
        println!(
            "{} {} [{}] - {}",
            mark,
            persona.name,
            avatar,
            persona.prompt_summary()
        );
        if !persona.is_default() {
            println!("    id: {}", persona.id);
        }
    }
    println!();
}

fn print_current_persona(app: &App) {
    println!("Current persona: {}", app.active_persona.name);
    println!("id: {}", app.active_persona.id);
    println!(
        "avatar: {}",
        if app.active_persona.avatar.trim().is_empty() {
            "-"
        } else {
            app.active_persona.avatar.trim()
        }
    );
    println!("history root: {}", app.config.history_file.display());
    println!("session: {}", app.session_id);
    println!("memory: {}", app.current_persona_memory_file().display());
    if app.active_persona.prompt.trim().is_empty() {
        println!("prompt: -");
    } else {
        println!("\nprompt:\n{}\n", app.active_persona.prompt.trim());
    }
}
