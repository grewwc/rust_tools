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
		"$$A_BIN" --generate-completions zsh > "$$DST/_a"; \
		echo "  completions -> $$DST/_a"; \
		echo "  add to ~/.zshrc: fpath=($$DST \$$fpath) && autoload -Uz compinit && compinit"; \
		;; \
	  fish) \
		DST="$${HOME}/.config/fish/completions"; \
		mkdir -p "$$DST"; \
		"$$A_BIN" --generate-completions fish > "$$DST/a.fish"; \
		echo "  completions -> $$DST/a.fish"; \
		;; \
	  *) \
		DST="$${HOME}/.bash_completion.d"; \
		mkdir -p "$$DST"; \
		"$$A_BIN" --generate-completions bash > "$$DST/a"; \
		echo "  completions -> $$DST/a"; \
		echo "  add to ~/.bashrc: source $$DST/a"; \
		;; \
	esac
