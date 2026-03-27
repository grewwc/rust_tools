#!/bin/sh

set -e

DEFAULT_BINS="a configw his j ns oo re tt"
INSTALL_DIR="${INSTALL_DIR:-$(pwd)/bin}"

if [ "$#" -gt 0 ]; then
    BINS="$*"
else
    BINS="$DEFAULT_BINS"
fi

BIN_DIR="$(pwd)/target/release"

if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
fi

NEED="$(
    INSTALLW_BIN="${INSTALLW_BIN:-$(pwd)/target/debug/installw}"
    if [ -x "$INSTALLW_BIN" ]; then
        INSTALL_DIR="$INSTALL_DIR" "$INSTALLW_BIN" --mode install -- $BINS
    else
        INSTALL_DIR="$INSTALL_DIR" cargo run -q --bin installw -- --mode install -- $BINS
    fi
)"

for bin in $NEED; do
    src="$BIN_DIR/$bin"
    if [ ! -f "$src" ] || [ ! -x "$src" ]; then
        echo "skip $bin (not built)" >&2
        continue
    fi

    dst="$INSTALL_DIR/$bin"

    if [ -e "$dst" ] || [ -L "$dst" ]; then
        if [ "$src" -ef "$dst" ]; then
            rm -f "$dst"
        fi
    fi

    cp "$src" "$dst"
    echo "copied $dst <- $src"
done
