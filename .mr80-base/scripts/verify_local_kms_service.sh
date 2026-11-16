#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
python_bin="${CONCILIATOR_PYTHON:-$repo_dir/.venv/bin/python}"
artifact_dir="${KMS_SMOKE_ARTIFACT_DIR:-$repo_dir/.local/kms}"
service_log="$artifact_dir/local-service.log"
service_pid_file="$artifact_dir/local-service.pid"
service_status="$artifact_dir/local-service-status.txt"
port="${PORT0:-8000}"

mkdir -p "$artifact_dir"
chmod 700 "$artifact_dir"

if ! test -x "$python_bin"; then
    printf 'service_start=failed\nreason=python_not_found\n' >"$service_status"
    chmod 600 "$service_status"
    printf 'Python executable not found: %s\n' "$python_bin" >&2
    exit 1
fi

if command -v lsof >/dev/null 2>&1 && \
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    printf 'service_start=failed\nreason=port_in_use\nport=%s\n' "$port" \
        >"$service_status"
    chmod 600 "$service_status"
    printf 'Port %s is already in use.\n' "$port" >&2
    exit 1
fi

cleanup() {
    if test -f "$service_pid_file"; then
        service_pid=$(cat "$service_pid_file")
        if kill -0 "$service_pid" 2>/dev/null; then
            kill "$service_pid" 2>/dev/null || true
            wait "$service_pid" 2>/dev/null || true
        fi
        rm -f "$service_pid_file"
    fi
}
trap cleanup EXIT INT TERM

export KMS_ZONE="${KMS_ZONE:-cn}"
export ENV=prod
# 完整服务探针：使用 Settings 默认值（包括 ByteCloud AK），需具备所有 KMS 权限。

cd "$repo_dir"
"$python_bin" -m uvicorn conciliator.main:app \
    --host 127.0.0.1 \
    --port "$port" \
    >"$service_log" 2>&1 &
service_pid=$!
printf '%s\n' "$service_pid" >"$service_pid_file"
chmod 600 "$service_log" "$service_pid_file"

service_ready=false
for _ in $(seq 1 40); do
    if curl --fail --silent "http://127.0.0.1:$port/healthz" >/dev/null; then
        service_ready=true
        break
    fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
        break
    fi
    sleep 0.25
done

if test "$service_ready" != true; then
    printf 'service_start=failed\nhealthz=unavailable\nlog=%s\n' "$service_log" \
        >"$service_status"
    chmod 600 "$service_status"
    printf 'Conciliator did not become healthy. See %s\n' "$service_log" >&2
    exit 1
fi

printf 'service_start=ok\nhealthz=ok\nlog=%s\n' "$service_log" \
    >"$service_status"
chmod 600 "$service_status"

"$python_bin" scripts/verify_kms_secrets.py \
    --kms-zone "$KMS_ZONE" \
    "$@"
