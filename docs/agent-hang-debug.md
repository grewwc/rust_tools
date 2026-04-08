# Agent Hang Debug

## 背景

`turn_runtime.rs` 中有一组用于排查 agent 卡顿的埋点函数 `report_agent_hang_debug()`。

这些埋点会在关键阶段上报调试事件，例如：

- 开始处理 turn
- 开始/结束准备 skill
- 开始/结束模型请求
- 开始/结束 stream
- 开始/结束 tool 执行

旧实现中，每次调用 `report_agent_hang_debug()` 都会启动一个新的线程，并尝试把调试数据发送到本地 debug server。

这对排查 hang 有帮助，但默认开启会带来额外开销。

## 编译期开关

现在这套埋点受 Cargo feature 控制：

- feature 名称：`agent-hang-debug`
- 默认状态：关闭

默认编译时：

- `report_agent_hang_debug()` 会编译成 no-op
- 不会创建线程
- 不会发送 HTTP 调试事件

显式开启 feature 时：

- 恢复原来的调试上报逻辑
- 每个埋点会创建后台线程发送事件

## 如何开启

### 运行 agent

```bash
cargo run --bin a --features agent-hang-debug
```

### 运行测试

```bash
cargo test --bin a --features agent-hang-debug
```

### 只做编译检查

```bash
cargo check --bin a --features agent-hang-debug
```

## 代码位置

- Cargo feature 定义：`Cargo.toml`
- 埋点实现：`src/bin/ai/driver/turn_runtime.rs`

## 适用场景

建议只在以下场景开启：

- 正在排查 agent 卡死、假死、长时间无响应
- 想确认卡在 `prepare_skill_for_turn`、模型请求还是 tool 执行
- 需要把运行阶段事件发送到本地 debug server

平时日常开发、普通测试、性能观察时，建议保持关闭。

## 注意事项

- 这是编译期开关，不是运行时配置
- 如果你已经编译过一版默认二进制，需要重新带 `--features agent-hang-debug` 编译或运行
- 如果本地没有对应 debug server，埋点发送失败不会影响主流程，但仍会有线程创建和额外请求开销

## 调试链路概览

`report_agent_hang_debug()` 会读取本地调试环境配置，并向本地服务上报 JSON 事件。

常见阶段包括：

- `turn_runtime::run_turn:begin`
- `turn_runtime::run_turn:prepare_skill_for_turn:begin`
- `turn_runtime::run_turn:prepare_skill_for_turn:end`
- `turn_runtime::run_turn:do_request_messages:begin`
- `turn_runtime::run_turn:do_request_messages:end`

如果需要进一步精简这套逻辑，下一步可以考虑：

- 用单线程队列代替“一次调用一个线程”
- 把 HTTP client 复用起来
- 增加运行时环境变量开关，和编译期开关配合使用
