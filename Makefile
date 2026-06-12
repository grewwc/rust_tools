INSTALL_BINS ?= $(sort $(patsubst src/bin/%.rs,%,$(wildcard src/bin/*.rs)))
ALL_BINS ?= $(INSTALL_BINS) c

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
install: $(INSTALLW)
	@bins=$$($(INSTALLW) -- $(INSTALL_BINS)); \
	if [ -n "$$bins" ]; then \
		args=""; \
		for b in $$bins; do args="$$args --bin $$b"; done; \
		cargo build --release $$args; \
	fi; \
	sh ./move_executable.sh $(INSTALL_BINS)
	@$(MAKE) install-completions
# -- shell completions --
.PHONY: install-completions
install-completions:
	@SHELL_NAME=$$(basename "$${SHELL:-/bin/bash}"); \
	A_BIN="$${INSTALL_DIR:-$$(pwd)/bin}/a"; \
	if [ ! -x "$$A_BIN" ]; then \
		echo "  skip completions: a not found at $$A_BIN"; \
		exit 0; \
	fi; \
	case "$$SHELL_NAME" in \
	  zsh) \
		DST="$${HOME}/.zfunc"; \
		mkdir -p "$$DST"; \
		"$$A_BIN" --generate-completions zsh > "$$DST/_a" && \
		line="fpath=($$DST \$$fpath)"; \
		if ! grep -qF "fpath=($$DST " "$${HOME}/.zshrc" 2>/dev/null; then \
			{ echo ""; echo "# a 命令补全"; echo "$$line"; echo "autoload -U compinit && compinit"; } >> "$${HOME}/.zshrc"; \
			echo "  added fpath to ~/.zshrc"; \
		else \
			echo "  ~/.zshrc already configured"; \
		fi; \
		echo "  completions -> $$DST/_a"; \
		echo "  add to ~/.zshrc: $$line"; \
		;; \
	  fish) \
		DST="$${HOME}/.config/fish/completions"; \
		mkdir -p "$$DST"; \
		"$$A_BIN" --generate-completions fish > "$$DST/a.fish"; \
		echo "  completions -> $$DST/a.fish"; \
		;; \
	  bash|*) \
		DST="$${HOME}/.bash_completion.d"; \
		mkdir -p "$$DST"; \
		"$$A_BIN" --generate-completions bash > "$$DST/a" && \
		line='source '"$$DST/a"; \
		for rc in "$${HOME}/.bashrc" "$${HOME}/.bash_profile"; do \
			if [ -f "$$rc" ] || [ "$$rc" = "$${HOME}/.bashrc" ]; then \
				if ! grep -qF "$$DST/a" "$$rc" 2>/dev/null; then \
					{ echo ""; echo "# a 命令补全"; echo "$$line"; } >> "$$rc"; \
					echo "  added source to $$rc"; \
				else \
					echo "  $$rc already configured"; \
				fi; \
			fi; \
		done; \
		echo "  completions -> $$DST/a"; \
		echo "  add to ~/.bashrc: source $$DST/a"; \
		;; \
	esac
