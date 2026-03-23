INSTALL_BINS ?= a configw his j ns oo re tt
ALL_BINS ?= $(INSTALL_BINS) c

RELEASE_DIR := target/release

$(RELEASE_DIR)/%: src/bin/%.rs
	cargo build --release --bin $*

all: $(addprefix $(RELEASE_DIR)/,$(ALL_BINS))

RUSTFLAGS_INSTALL ?= -Awarnings
install: export RUSTFLAGS := $(strip $(RUSTFLAGS) $(RUSTFLAGS_INSTALL))
.PHONY: install
install:
	@bins=$$(cargo run --bin installw -- $(INSTALL_BINS)); \
	if [ -n "$$bins" ]; then \
		args=""; \
		for b in $$bins; do args="$$args --bin $$b"; done; \
		cargo build --release $$args; \
	fi; \
	sh ./move_executable.sh $(INSTALL_BINS)
