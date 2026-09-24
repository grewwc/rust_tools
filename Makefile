# Standalone MCP server crates live in crates/, not src/bin/*.rs, so the
# wildcard below cannot see them; install them alongside the regular bins.
MCP_BINS := mcp_browser mcp_computer mcp_excel mcp_pdf
INSTALL_BINS ?= $(sort $(patsubst src/bin/%.rs,%,$(wildcard src/bin/*.rs)) $(MCP_BINS))
ALL_BINS ?= $(INSTALL_BINS) c

# 允许 `make install fk` / `make install fk ff` 语法
.PHONY: $(INSTALL_BINS)
$(INSTALL_BINS): install ; @:

RELEASE_DIR := target/release
DEBUG_DIR := target/debug
INSTALLW := $(DEBUG_DIR)/installw
CargoLock := $(wildcard Cargo.lock)
INSTALLW_DEPS := $(shell find src -type f -name '*.rs') Cargo.toml $(CargoLock)

$(RELEASE_DIR)/%: src/bin/%.rs
	cargo build --release --bin $*

# mcp_* have no src/bin/*.rs prerequisite; build them from the workspace root.
# Give each target its own crate's sources only, so touching one crate does not
# re-trigger the recipes of the others (cargo still skips no-op rebuilds).
# mcp_pdf links the root lib (crates/mcp_pdf/Cargo.toml), so it also watches the
# root crate's sources; mcp_browser/mcp_excel do not depend on the root lib.
$(RELEASE_DIR)/mcp_browser: $(shell find crates/mcp_browser -type f -name '*.rs') Cargo.toml
	cargo build --release -p mcp_browser --bin mcp_browser

$(RELEASE_DIR)/mcp_computer: $(shell find crates/mcp_computer -type f -name '*.rs') Cargo.toml
	cargo build --release -p mcp_computer --bin mcp_computer

$(RELEASE_DIR)/mcp_excel: $(shell find crates/mcp_excel -type f -name '*.rs') Cargo.toml
	cargo build --release -p mcp_excel --bin mcp_excel

$(RELEASE_DIR)/mcp_pdf: $(shell find crates/mcp_pdf -type f -name '*.rs') $(shell find src -type f -name '*.rs') Cargo.toml
	cargo build --release -p mcp_pdf --bin mcp_pdf

all: $(addprefix $(RELEASE_DIR)/,$(ALL_BINS))

$(INSTALLW): $(INSTALLW_DEPS)
	cargo build --bin installw

# `re` lives in crates/re, not src/bin/, so installw's build mode (which
# resolves src/bin/<bin>.rs dependencies) never emits it; build it explicitly
# and let move_executable.sh's install mode decide whether to re-copy.
.PHONY: install
install: $(INSTALLW) $(addprefix $(RELEASE_DIR)/,$(MCP_BINS))
	$(eval REQUESTED := $(filter-out install,$(MAKECMDGOALS)))
	$(eval BINS := $(or $(REQUESTED),$(INSTALL_BINS)))
	@set -e; \
	if [ -n "$(REQUESTED)" ]; then \
		args=""; \
		for b in $(BINS); do \
			case "$$b" in mcp_browser|mcp_computer|mcp_excel|mcp_pdf) args="$$args -p $$b --bin $$b";; *) args="$$args --bin $$b";; esac; \
		done; \
		cargo build --release $$args; \
		sh ./move_executable.sh --force $(BINS); \
	else \
		bins=$$($(INSTALLW) -- $(BINS)); \
		if [ -n "$$bins" ]; then \
			args=""; \
			for b in $$bins; do \
				case "$$b" in mcp_browser|mcp_computer|mcp_excel|mcp_pdf) args="$$args -p $$b --bin $$b";; *) args="$$args --bin $$b";; esac; \
			done; \
			cargo build --release $$args; \
		fi; \
		cargo build --release -p re; \
		sh ./move_executable.sh $(BINS); \
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
