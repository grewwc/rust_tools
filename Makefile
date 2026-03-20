BINS = a c configw his j ns oo re tt

all:
	cargo build --release $(addprefix --bin ,$(BINS))

install:
	cargo build --release $(addprefix --bin ,$(BINS))
	sh ./move_executable.sh

install-a:
	cargo build --release --bin a
	sh ./move_executable.sh a
