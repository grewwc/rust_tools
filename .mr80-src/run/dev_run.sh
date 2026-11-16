#!/bin/bash
# 本地开发启动入口(带热重载,不走 bytedunicorn)
#
# 用法:
#   ./run/dev_run.sh
#
# 前置:
#   - .env 配置 ENV=dev、MYSQL_DSN 和本地凭据

set -euo pipefail

APP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$APP_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$APP_DIR/.env"
    set +a
fi

export ENV="${ENV:-dev}"
export PORT0="${PORT0:-8000}"

if [[ ! "$PORT0" =~ ^[0-9]+$ ]] || ((PORT0 < 1 || PORT0 > 65535)); then
    printf 'invalid PORT0: %s\n' "$PORT0" >&2
    exit 1
fi

LOG_DIR="$APP_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/app.log"

process_pattern="python.*-m[[:space:]]+uvicorn[[:space:]]+conciliator[.]main:app.*--port[[:space:]]+$PORT0([[:space:]]|$)"
mapfile -t old_pids < <(pgrep -f "$process_pattern" || true)

if ((${#old_pids[@]} > 0)); then
    kill -TERM "${old_pids[@]}" 2>/dev/null || true

    for _ in {1..10}; do
        remaining_pids=()
        for pid in "${old_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining_pids+=("$pid")
            fi
        done
        ((${#remaining_pids[@]} == 0)) && break
        sleep 0.2
    done

    if ((${#remaining_pids[@]} > 0)); then
        kill -KILL "${remaining_pids[@]}" 2>/dev/null || true
    fi
fi

: >"$LOG_FILE"
exec uv run python -m uvicorn conciliator.main:app \
    --reload \
    --reload-dir "$APP_DIR/src" \
    --host 0.0.0.0 \
    --port "$PORT0" \
    >>"$LOG_FILE" 2>&1
