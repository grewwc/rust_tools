#!/bin/sh

set -e

DEFAULT_BINS="a configw his j ns oo re tt"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

if [ "$#" -gt 0 ]; then
    BINS="$*"
else
    BINS="$DEFAULT_BINS"
fi

BIN_DIR="$(pwd)/target/release"

if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR" 2>/dev/null || sudo mkdir -p "$INSTALL_DIR"
fi

for bin in $BINS; do
    src="$BIN_DIR/$bin"
    if [ ! -f "$src" ] || [ ! -x "$src" ]; then
        echo "skip $bin (not built)" >&2
        continue
    fi

    dst="$INSTALL_DIR/$bin"
    if [ -w "$INSTALL_DIR" ]; then
        ln -sf "$src" "$dst"
    else
        sudo ln -sf "$src" "$dst"
    fi
    echo "linked $dst -> $src"
done
