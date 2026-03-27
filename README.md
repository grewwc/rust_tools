# rust_tools
A collection of Rust utilities and tools.

## Overview

This repository contains various Rust-based tools and utilities for common development tasks.

## Prerequisites

- Rust 1.70 or higher
- Cargo (comes with Rust)

## Installation

```bash
git clone https://github.com/yourusername/rust_tools.git
cd rust_tools
cargo build --release
```

## Usage

```bash
cargo run --bin <tool_name>
```

## AI Skill And Memory

The AI agent now supports writing reusable skills and persistent memory notes by tool calls.

- `save_skill`: writes a `.skill` file into external skills directory.
- `memory_append`: appends notes to agent memory store (`jsonl`).
- `memory_search`: searches memory by keyword.
- `memory_recent`: reads recent notes.

Config keys in `~/.configW`:

- `ai.skills.dir` (optional): override skills directory. Default: `~/.config/rust_tools/skills`
- `ai.memory.file` (optional): override memory file. Default: `~/.config/rust_tools/agent_memory.jsonl`

## Project Structure

```
rust_tools/
├── src/
│   ├── main.rs
│   └── lib.rs
├── Cargo.toml
└── README.md
```

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.