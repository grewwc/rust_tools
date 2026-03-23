INSTALL_BINS ?= $(sort $(patsubst src/bin/%.rs,%,$(wildcard src/bin/*.rs)))
ALL_BINS ?= $(INSTALL_BINS) c

RELEASE_DIR := target/release
DEBUG_DIR := target/debug
INSTALLW := $(DEBUG_DIR)/installw
CargoLock := $(wildcard Cargo.lock)

$(RELEASE_DIR)/%: src/bin/%.rs
	cargo build --release --bin $*

all: $(addprefix $(RELEASE_DIR)/,$(ALL_BINS))

$(INSTALLW): src/bin/installw.rs Cargo.toml $(CargoLock)
	cargo build -q --bin installw

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
