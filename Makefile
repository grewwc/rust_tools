# Standalone MCP server crates live in crates/, not src/bin/*.rs, so the
# wildcard below cannot see them; install them alongside the regular bins.
MCP_BINS := mcp_browser mcp_computer mcp_excel mcp_pdf
INSTALL_BINS ?= $(sort $(patsubst src/bin/%.rs,%,$(wildcard src/bin/*.rs)) $(MCP_BINS))
ALL_BINS ?= $(INSTALL_BINS) c

# 允许 `make install fk` / `make install fk ff` 语法
.PHONY: $(INSTALL_BINS)
$(INSTALL_BINS): install ; @:

# Release features enabled by `install-all`. `agent-hang-debug` is
# deliberately excluded: Cargo.toml marks it as off-limits for normal and
# release builds.
RELEASE_FEATURES := serve

# Features for a plain `install`. `a` needs serve-client for `a --serve-chat`;
# the feature has cfg gates only inside src/bin/ai, so all other bins build
# identical binaries and the whole install fits in one cargo invocation.
INSTALL_FEATURES := serve-client

RELEASE_DIR := target/release

$(RELEASE_DIR)/%: src/bin/%.rs
	cargo build --release --bin $*

# `all` is a manual convenience target; install/install-all do not use these
# targets — their recipes build everything in one cargo invocation, so cargo
# itself decides what is stale (no mtime prerequisites, so a touched
# Cargo.toml no longer re-fires the recipes). mcp_* have no src/bin/*.rs
# prerequisite; each target watches only its own crate's sources, plus the
# root crate's sources for mcp_pdf/mcp_computer, which link the root lib
# (mcp_browser/mcp_excel do not).
$(RELEASE_DIR)/mcp_browser: $(shell find crates/mcp_browser -type f -name '*.rs')
	cargo build --release -p mcp_browser --bin mcp_browser

$(RELEASE_DIR)/mcp_computer: $(shell find crates/mcp_computer -type f -name '*.rs') $(shell find src -type f -name '*.rs')
	cargo build --release -p mcp_computer --bin mcp_computer

$(RELEASE_DIR)/mcp_excel: $(shell find crates/mcp_excel -type f -name '*.rs')
	cargo build --release -p mcp_excel --bin mcp_excel

$(RELEASE_DIR)/mcp_pdf: $(shell find crates/mcp_pdf -type f -name '*.rs') $(shell find src -type f -name '*.rs')
	cargo build --release -p mcp_pdf --bin mcp_pdf

all: $(addprefix $(RELEASE_DIR)/,$(ALL_BINS))

# Every install builds everything in ONE release cargo invocation:
# - cargo itself decides what is stale (fingerprints), so nothing rebuilds on
#   a no-op run and there are no mtime prerequisites to go stale (a touched
#   Cargo.toml used to re-fire the mcp_* targets).
# - one invocation means the root lib compiles once per install; the old
#   split (a with serve-client, then re/mcp_* and the other bins without)
#   compiled it twice per install. `--bins` selects all rust_tools bins, so
#   no `--bin`/`-p` mixing is needed; `rust_tools/<feature>` is the syntax
#   cargo accepts for features of a package other than the first `-p`.
# - re and the mcp_* crates have no src/bin/*.rs, so their staleness (e.g. a
#   changed Cargo.lock) is only detectable by cargo; that is why the
#   invocation runs even when nothing else is stale.
# A targeted `make install fk` keeps its own (root-bins-only) invocation so
# it builds just the requested bins; the requested mcp_* crates build in a
# second invocation because `--bin` (rust_tools) and `-p` (crate) cannot mix.
.PHONY: install
install:
	$(eval REQUESTED := $(filter-out install,$(MAKECMDGOALS)))
	@set -e; \
	if [ -n "$(REQUESTED)" ]; then \
		root_args=""; mcp_args=""; \
		for b in $(REQUESTED); do \
			case "$$b" in mcp_browser|mcp_computer|mcp_excel|mcp_pdf) mcp_args="$$mcp_args -p $$b";; *) root_args="$$root_args --bin $$b";; esac; \
		done; \
		if [ -n "$$root_args" ]; then cargo build --release --features rust_tools/$(INSTALL_FEATURES) $$root_args; fi; \
		if [ -n "$$mcp_args" ]; then cargo build --release --features rust_tools/$(INSTALL_FEATURES) $$mcp_args; fi; \
		sh ./move_executable.sh --force $(REQUESTED); \
	else \
		cargo build --release --features rust_tools/$(INSTALL_FEATURES) --bins -p rust_tools -p re -p mcp_browser -p mcp_computer -p mcp_excel -p mcp_pdf; \
		sh ./move_executable.sh $(INSTALL_BINS); \
		sh ./move_executable.sh re $(MCP_BINS); \
	fi

	@$(MAKE) install-completions
# -- shell completions --
.PHONY: install-completions
install-completions:
	@SHELL_NAME=$$(basename "$${SHELL:-/bin/bash}"); \
	INSTALL_DIR="$${INSTALL_DIR:-$$(pwd)/bin}"; \
	for bin in a fk re; do \
		BIN="$${INSTALL_DIR}/$$bin"; \
		if [ ! -x "$$BIN" ]; then \
			echo "  skip completions: $$bin not found at $$BIN"; \
			continue; \
		fi; \
		case "$$SHELL_NAME" in \
		  zsh) \
			DST="$${HOME}/.zfunc"; \
			mkdir -p "$$DST"; \
			"$$BIN" --generate-completions zsh > "$$DST/_$$bin" && \
			line="fpath=($$DST \$$fpath)"; \
			if ! grep -qF "fpath=($$DST " "$${HOME}/.zshrc" 2>/dev/null; then \
				{ echo ""; echo "# $$bin 命令补全"; echo "$$line"; echo "autoload -U compinit && compinit"; } >> "$${HOME}/.zshrc"; \
				echo "  added fpath to ~/.zshrc for $$bin"; \
				if ! grep -qF "rehash true" "$${HOME}/.zshrc" 2>/dev/null; then \
					{ echo ""; echo "zstyle '"'"':completion:*'"'"' rehash true"; } >> "$${HOME}/.zshrc"; \
					echo "  added rehash style to ~/.zshrc"; \
				fi; \
			else \
				if ! grep -qF "rehash true" "$${HOME}/.zshrc" 2>/dev/null; then \
					{ echo ""; echo "zstyle '"'"':completion:*'"'"' rehash true"; } >> "$${HOME}/.zshrc"; \
					echo "  added rehash style to ~/.zshrc"; \
				fi; \
			fi; \
			;; \
		  fish) \
			DST="$${HOME}/.config/fish/completions"; \
			mkdir -p "$$DST"; \
			"$$BIN" --generate-completions fish > "$$DST/$$bin.fish"; \
			echo "  completions -> $$DST/$$bin.fish"; \
			;; \
		  bash|*) \
			DST="$${HOME}/.bash_completion.d"; \
			mkdir -p "$$DST"; \
			"$$BIN" --generate-completions bash > "$$DST/$$bin" && \
			line='source '"$$DST/$$bin"; \
			for rc in "$${HOME}/.bashrc" "$${HOME}/.bash_profile"; do \
				if [ -f "$$rc" ] || [ "$$rc" = "$${HOME}/.bashrc" ]; then \
					if ! grep -qF "$$DST/$$bin" "$$rc" 2>/dev/null; then \
						{ echo ""; echo "# $$bin 命令补全"; echo "$$line"; } >> "$$rc"; \
						echo "  added source to $$rc for $$bin"; \
					else \
						echo "  $$rc already configured for $$bin"; \
					fi; \
				fi; \
			done; \
			echo "  completions -> $$DST/$$bin"; \
			echo "  add to ~/.bashrc: source $$DST/$$bin"; \
			;; \
		esac; \
	done
.PHONY: test test-a test-fk clean

# Like `install`, but builds everything with all release features (incl.
# `serve`) in one invocation, so the root lib still compiles only once. It
# runs unconditionally: a feature change does not touch source mtimes, so an
# mtime-based staleness check would wrongly skip the rebuild.
.PHONY: install-all
install-all:
	cargo build --release --features rust_tools/$(RELEASE_FEATURES) --bins -p rust_tools -p re -p mcp_browser -p mcp_computer -p mcp_excel -p mcp_pdf
	sh ./move_executable.sh --force $(INSTALL_BINS)
	sh ./move_executable.sh --force re
	@$(MAKE) install-completions
