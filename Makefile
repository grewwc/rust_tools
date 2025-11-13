all:
	cargo build --release

install:
	cargo build --release
	sh ./move_executable.sh

