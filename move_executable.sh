#!/bin/sh

for file in target/release/*; do
    if [ -f "$file" ] && [ -x "$file" ]; then
        ln -sf "$(pwd)/$file" /usr/local/bin/
    fi
done
