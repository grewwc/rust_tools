#!/bin/bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test_dir=$(mktemp -d)
old_pid=""
cleanup() {
    if [[ -n "$old_pid" ]]; then
        kill -KILL "$old_pid" 2>/dev/null || true
    fi
    rm -rf "$test_dir"
}
trap cleanup EXIT

mkdir -p "$test_dir/bin" "$test_dir/repo/run"
cp "$repo_dir/run/dev_run.sh" "$test_dir/repo/run/dev_run.sh"

cat >"$test_dir/bin/uv" <<'EOF'
#!/bin/bash
printf '%s\n' "$@" >"$DEV_RUN_ARGS_FILE"
printf '%s\n' "${DEV_RUN_ENV_PROBE:-}" >"$DEV_RUN_ENV_FILE"
printf 'dev stdout\n'
printf 'dev stderr\n' >&2
EOF
chmod +x "$test_dir/bin/uv"

args_file="$test_dir/args"
env_file="$test_dir/env"
printf 'DEV_RUN_ENV_PROBE=loaded\n' >"$test_dir/repo/.env"
mkdir -p "$test_dir/repo/logs"
printf 'stale traceback\n' >"$test_dir/repo/logs/app.log"
bash -c 'exec -a "python -m uvicorn conciliator.main:app --host 0.0.0.0 --port 8765" sleep 300' &
old_pid=$!
sleep 0.1

(
    cd "$test_dir/repo"
    PATH="$test_dir/bin:$PATH" DEV_RUN_ARGS_FILE="$args_file" \
        DEV_RUN_ENV_FILE="$env_file" PORT0=8765 \
        ./run/dev_run.sh
) >"$test_dir/terminal.stdout" 2>"$test_dir/terminal.stderr"

old_state=$(ps -o stat= -p "$old_pid" 2>/dev/null | xargs || true)
if [[ -n "$old_state" && "$old_state" != Z* ]]; then
    printf 'old uvicorn process was not stopped: %s\n' "$old_pid" >&2
    exit 1
fi
wait "$old_pid" 2>/dev/null || true
old_pid=""

log_file="$test_dir/repo/logs/app.log"
if [[ ! -f "$log_file" ]]; then
    printf 'missing log file: %s\n' "$log_file" >&2
    exit 1
fi
grep -Fx 'dev stdout' "$log_file" >/dev/null
grep -Fx 'dev stderr' "$log_file" >/dev/null
if grep -F 'stale traceback' "$log_file" >/dev/null; then
    printf 'stale log content was not cleared\n' >&2
    exit 1
fi
grep -Fx 'loaded' "$env_file" >/dev/null

if [[ -s "$test_dir/terminal.stdout" || -s "$test_dir/terminal.stderr" ]]; then
    printf 'dev_run output unexpectedly reached the terminal\n' >&2
    exit 1
fi

expected=$(cat <<'EOF'
run
python
-m
uvicorn
conciliator.main:app
--reload
--host
0.0.0.0
--port
8765
EOF
)
actual=$(cat "$args_file")

if [[ "$actual" != "$expected" ]]; then
    printf 'unexpected dev_run command:\n%s\n' "$actual" >&2
    exit 1
fi
