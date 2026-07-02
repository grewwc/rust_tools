INSTALL_BINS ?= $(sort $(patsubst src/bin/%.rs,%,$(wildcard src/bin/*.rs)))
ALL_BINS ?= $(INSTALL_BINS) c

# 允许 `make install fk` / `make install fk ff` 语法
.PHONY: $(INSTALL_BINS)
$(INSTALL_BINS): install

RELEASE_DIR := target/release
DEBUG_DIR := target/debug
INSTALLW := $(DEBUG_DIR)/installw
CargoLock := $(wildcard Cargo.lock)
INSTALLW_DEPS := $(shell find src -type f -name '*.rs') Cargo.toml $(CargoLock)

$(RELEASE_DIR)/%: src/bin/%.rs
	cargo build --release --bin $*

all: $(addprefix $(RELEASE_DIR)/,$(ALL_BINS))

$(INSTALLW): $(INSTALLW_DEPS)
	cargo build --bin installw

RUSTFLAGS_INSTALL ?= -Awarnings
install: export RUSTFLAGS := $(strip $(RUSTFLAGS) $(RUSTFLAGS_INSTALL))
.PHONY: install
install:
	$(eval REQUESTED := $(filter-out install,$(MAKECMDGOALS)))
	$(eval BINS := $(or $(REQUESTED),$(INSTALL_BINS)))
	@set -e; \
	args=""; \
	for b in $(BINS); do args="$$args --bin $$b"; done; \
	echo "building $$args"; \
	cargo build --release $$args; \
	sh ./move_executable.sh $(BINS)

	@$(MAKE) install-completions
# -- shell completions --
.PHONY: install-completions
install-completions:
	@SHELL_NAME=$$(basename "$${SHELL:-/bin/bash}"); \
	INSTALL_DIR="$${INSTALL_DIR:-$$(pwd)/bin}"; \
	for bin in a fk; do \
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
