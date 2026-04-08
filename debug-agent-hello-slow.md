# [OPEN] agent-hello-slow

## 背景

- 现象：执行 `a hello` 体感明显偏慢
- 目标：确认慢点具体落在启动期、skill/mcp 准备、请求前判定，还是主模型请求本身

## 初始假设

1. `run()` 启动阶段会先加载 skills / 初始化 MCP，导致简单问候也被启动成本拖慢
2. `prepare_skill_for_turn()` 内部的意图识别或 skill 路由在 `hello` 场景仍然执行，产生不必要耗时
3. `do_request_messages()` 前存在额外的 thinking gate / 控制面请求，导致首包前多一次或多次网络 RTT
4. 主模型请求本身不是主要瓶颈，真正慢的是请求前串行准备链路
5. 终端/UI 或历史构建开销不是主因，只是次要噪声

## 当前计划

1. 给关键路径加最小埋点，记录阶段耗时
2. 复现 `a hello`
3. 根据日志确认瓶颈位置
4. 基于证据做最小优化

## 证据

- `driver::run:load_all_skills:end`：约 `59ms` / `6ms` / `14ms`
- `driver::run:init_mcp:end`：约 `68ms` / `6ms` / `12ms`
- `skill_runtime::prepare_skill_for_turn:intent:end`：约 `3ms`
- `request::select_skill_via_model:end`：约 `820ms` / `638ms`
- `request::decide_thinking_via_model:end`：约 `399ms`
- `request::do_request_messages:http_success`：约 `2888ms`

## 结论

1. 启动期 `skills/MCP` 有成本，但不是主因
2. `hello` 仍然触发了 skill router，平白多出约 `0.6s ~ 0.8s`
3. thinking gate 额外再增加约 `0.4s`
4. 主模型请求本身仍是最大头，约 `2.9s`
5. 对 `a hello` 这种问候语，如果想明显提速，必须做本地 fast-path

## 已做修复

1. 对 `short + casual` 输入跳过 skill router
2. 对 `query_concept/casual` 这类轻请求，本地跳过 skill router
3. 对 `query_concept/casual` 且较短的请求，本地跳过 thinking gate
4. 对较短 `casual` 请求，跳过知识召回

## 待确认

- post-fix 日志已确认 `skip_router_for_local_intent=true`，`prepare_skill_for_turn` 从约 `638ms~820ms` 降到约 `18ms`
- 当前环境又触发了独立问题：`unable to open database file ... .history_file.sessions/...sqlite`，导致本次未跑到完整主请求阶段
- 用户验证 `a hello` 在其正常环境下是否已达到可接受速度
