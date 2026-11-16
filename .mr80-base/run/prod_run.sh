#!/bin/bash
# Conciliator 启动入口(由 atum.yaml 的 run_command 调用)
#
# 约定(对齐字节 TCE + Atum):
#   - TCE 注入 PORT0 环境变量作为监听端口
#   - Conciliator 是 FastAPI/ASGI 服务,统一使用 Uvicorn 启动
#
# 数据库迁移在发布前独立完成；本脚本不跑 alembic，避免多实例迁移冲突。

set -euo pipefail

# 切到部署目录(atum.yaml work_dir)
APP_DIR="${APP_DIR:-/opt/tiger/conciliator}"
cd "$APP_DIR"

export LANG="${LANG:-en_US.UTF-8}"
export DEPLOY_ENV="${DEPLOY_ENV:-PROD}"
# TCE 的 toutiao/load 会预先设置 PYTHONPATH；必须显式把 src 放到最前面，
# 不能只在 PYTHONPATH 为空时兜底。
export PYTHONPATH="$APP_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# ulimit 对齐字节服务约定(core dump 用于排查)
ulimit -c 10485760 || true

PORT0="${PORT0:-8000}"

# TCE 可通过环境变量覆盖进程数和超时。
# 多进程父进程负责回收达到请求上限的 worker；业务任务并发另由 Settings 控制。
WORKER_NUM="${WORKER_NUM:-4}"
if [[ ! "$WORKER_NUM" =~ ^[1-9][0-9]*$ ]] || ((WORKER_NUM < 2)); then
    printf 'WORKER_NUM must be an integer >= 2 for production\n' >&2
    exit 1
fi
MAX_TIMEOUT="${MAX_TIMEOUT:-300}"
MAX_REQUESTS="${MAX_REQUESTS:-10000}"

# TCE 偶发端口占用,启动前先清理
if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti:$PORT0 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[prod_run] killing processes on port $PORT0: $pids"
        kill -9 $pids || true
    fi
fi

BIND_ADDR="${BIND_ADDR:-0.0.0.0}"

echo "[prod_run] starting uvicorn on ${BIND_ADDR}:${PORT0}"
exec python -m uvicorn \
    conciliator.main:app \
    --host "$BIND_ADDR" \
    --port "$PORT0" \
    --workers "$WORKER_NUM" \
    --timeout-keep-alive "$MAX_TIMEOUT" \
    --limit-max-requests "$MAX_REQUESTS"
