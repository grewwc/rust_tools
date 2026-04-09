# `code_discovery_policy.json` 调参说明

## 目标

这份文档说明如何调节项目中的 `code_discovery_policy.json`，让 agent 在代码调试时更准确地：

- 识别什么是高价值代码发现
- 决定哪些发现需要持久化
- 决定哪些发现应该优先召回到后续上下文

对应的样例文件位于：

- [code_discovery_policy.json](file:///Users/bytedance/rust_tools/.rust_tools/code_discovery_policy.json)

策略实现代码位于：

- [code_discovery_policy.rs](file:///Users/bytedance/rust_tools/src/bin/ai/code_discovery_policy.rs)

## 生效路径

当前支持两个外部配置路径，按顺序加载：

1. `~/.config/rust_tools/code_discovery_policy.json`
2. `./.rust_tools/code_discovery_policy.json`

后加载的配置会覆盖先加载的配置。

如果这两个文件都不存在，或者解析失败，系统会自动回退到内建默认策略。

## 配置结构

完整结构分成 3 部分：

```json
{
  "classification": {
    "rules": []
  },
  "recall": {
    "max_items": 8,
    "confidence_weight": {},
    "kind_weight": {}
  },
  "persistence": {
    "max_persist_per_turn": 3,
    "min_confidence": "medium",
    "priority_weight": {}
  }
}
```

3 个部分分别控制：

- `classification`: 一条工具发现如何被分类成 `kind` 和 `confidence`
- `recall`: session 级 recall 时，哪些发现排得更靠前
- `persistence`: 每轮最多持久化多少条，以及哪些发现值得落盘

## 一、`classification.rules`

### 作用

`classification.rules` 决定一条工具输出是否被识别为 `code_discovery`，以及它属于什么类型。

系统会按顺序检查这些规则，第一条命中的规则生效。

### 单条规则结构

```json
{
  "enabled": true,
  "tool_names": ["read_file", "read_file_lines", "code_search"],
  "match": {
    "any_contains": ["root cause", "caused by"],
    "all_contains": [],
    "none_contains": []
  },
  "kind": "root_cause",
  "confidence": "high"
}
```

### 字段说明

| 字段 | 类型 | 是否必须 | 说明 |
|------|------|----------|------|
| `enabled` | `bool` | 否 | 是否启用该规则，默认 `true` |
| `tool_names` | `string[]` | 否 | 仅当工具名在列表中时才应用此规则 |
| `match.any_contains` | `string[]` | 否 | 命中任意一个子串即可 |
| `match.all_contains` | `string[]` | 否 | 必须同时命中所有子串 |
| `match.none_contains` | `string[]` | 否 | 只要命中任意一个子串就排除 |
| `kind` | `string` | 是 | 发现类型 |
| `confidence` | `string` | 是 | 发现置信度 |

### 匹配规则

匹配是大小写不敏感的，内部会统一转成小写后进行 `contains` 检查。

规则命中条件是：

- `tool_names` 为空，或者当前工具名在 `tool_names` 中
- `any_contains` 为空，或者至少命中一项
- `all_contains` 中所有项都命中
- `none_contains` 中没有任何一项命中

### `kind` 可选值

| 值 | 含义 | 典型场景 |
|----|------|----------|
| `root_cause` | 根因 | 找到真正导致 bug 的条件或原因 |
| `error_site` | 错误点 | 找到 panic / error / failed 等报错位置 |
| `entry_point` | 入口点 | 找到 `main`、bootstrap、启动入口 |
| `call_chain` | 调用链 | 找到 caller/callee 或清晰调用路径 |
| `symbol` | 符号定义 | 找到函数、结构体、类、impl 等 |
| `config` | 配置相关 | 找到 feature flag、配置项、toml 等 |
| `code_path` | 代码路径线索 | 找到文件/行号，但价值弱于根因和符号 |
| `todo` | TODO/FIXME | 找到待修复线索 |

### `confidence` 可选值

| 值 | 含义 |
|----|------|
| `high` | 高置信度，通常是明确结论 |
| `medium` | 中置信度，通常是强线索 |
| `low` | 低置信度，通常是弱线索或辅助路径 |

### 推荐调法

- 想让某类发现更容易被识别：
  - 在靠前的位置增加更具体的规则
- 想避免误判：
  - 使用 `none_contains`
  - 或把更严格的规则放在前面
- 想限定某条规则只作用于 `code_search`：
  - 把 `tool_names` 写成 `["code_search"]`

### 顺序非常重要

因为规则是“**按顺序命中第一条**”，所以建议：

- 更具体的规则放前面
- 更泛的规则放后面

例如：

- `root_cause`
- `call_chain`
- `entry_point`
- `error_site`
- `symbol`
- `code_path`

这样的顺序通常比把 `code_path` 放前面更合理。

## 二、`recall`

### 作用

`recall` 决定 session 级持久化发现被重新注入上下文时的优先级。

### `max_items`

```json
"max_items": 8
```

表示 recall 最多向上下文注入多少条 `code_discovery`。

调参建议：

- 代码库较复杂、需要跨多轮 debug:
  - 可以提高到 `10` 或 `12`
- 想减少上下文噪音：
  - 可以降低到 `4` 或 `6`

### `confidence_weight`

```json
"confidence_weight": {
  "low": 100,
  "medium": 200,
  "high": 300
}
```

表示不同置信度的基础分。

通常建议保持：

- `high > medium > low`

如果你希望系统极度偏好高置信度发现，可以把 `high` 拉得更高，例如：

```json
"confidence_weight": {
  "low": 80,
  "medium": 180,
  "high": 400
}
```

### `kind_weight`

```json
"kind_weight": {
  "error_site": 60,
  "root_cause": 70,
  "entry_point": 50,
  "call_chain": 40,
  "symbol": 30,
  "code_path": 10,
  "config": 20,
  "todo": 0
}
```

表示不同 `kind` 的附加分。

最终 recall 排序大致等于：

```text
confidence_weight + kind_weight
```

### 推荐排序思路

如果你的目标是偏调试，推荐：

- `root_cause`
- `error_site`
- `entry_point`
- `call_chain`
- `symbol`
- `config`
- `code_path`
- `todo`

如果你的目标是偏代码理解和重构，推荐提高：

- `symbol`
- `call_chain`
- `entry_point`

## 三、`persistence`

### 作用

`persistence` 决定当前 turn 内有哪些发现会被持久化到：

- session history
- `MemoryStore`

### `max_persist_per_turn`

```json
"max_persist_per_turn": 3
```

表示每轮最多持久化多少条 discovery。

调参建议：

- 想减少噪音：降低到 `1` 或 `2`
- 想让跨轮记忆更强：提高到 `4` 或 `5`

### `min_confidence`

```json
"min_confidence": "medium"
```

表示低于这个置信度的 discovery 不持久化。

可选值：

- `low`
- `medium`
- `high`

调参建议：

- 如果你觉得当前持久化噪音偏多：
  - 改成 `"high"`
- 如果你想让代码路径线索也保留：
  - 改成 `"low"`

### `priority_weight`

```json
"priority_weight": {
  "low": 120,
  "medium": 160,
  "high": 200
}
```

这是写入 `MemoryStore` 时的优先级映射。

它不直接影响分类，而是影响后续 memory 系统对这些记录的相对重要性。

通常建议保持：

- `high > medium > low`

## 常见调参目标

### 1. 更偏 root cause

适用于你主要用它来 debug。

建议：

- 提高 `recall.kind_weight.root_cause`
- 提高 `recall.kind_weight.error_site`
- 降低 `code_path`
- 将 `persistence.min_confidence` 保持在 `medium` 或 `high`

示例：

```json
"recall": {
  "kind_weight": {
    "root_cause": 120,
    "error_site": 100,
    "entry_point": 50,
    "call_chain": 40,
    "symbol": 20,
    "code_path": 5,
    "config": 15,
    "todo": 0
  }
}
```

### 2. 更偏调用链分析

适用于排查复杂调用路径、路由链、框架入口。

建议：

- 提高 `call_chain`
- 提高 `entry_point`
- 适度提高 `symbol`

### 3. 减少 recall 噪音

建议：

- 降低 `recall.max_items`
- 提高 `persistence.min_confidence`
- 降低 `todo` / `code_path` 权重

### 4. 让 `TODO/FIXME` 更容易被保留下来

建议：

- 提高 `kind_weight.todo`
- 把 `min_confidence` 调低
- 或把 TODO 规则的 `confidence` 从 `medium` 提到 `high`

## 推荐调参流程

建议按下面顺序调：

1. 先调 `recall.kind_weight`
2. 再调 `classification.rules`
3. 最后调 `persistence`

原因是：

- `recall` 调整最安全，只影响排序
- `classification` 调整会改变 discovery 的类型
- `persistence` 调整会改变哪些发现被永久保留下来，影响最大

## 调参原则

### 优先做小步修改

不要一次性重写整份规则，优先：

- 先改 1 到 2 个 `kind_weight`
- 观察几轮效果
- 再决定要不要改分类规则

### 优先提高具体规则，不要过早放宽泛规则

例如优先加：

- `"root cause"`
- `"call chain"`
- `"entry point"`

而不是直接用过于泛化的：

- `"calls"`
- `"start"`
- `"load"`

因为这些词很容易误判。

### 尽量让 `code_path` 保持弱权重

`code_path` 很容易命中，但语义价值通常不如：

- `root_cause`
- `error_site`
- `symbol`
- `entry_point`

它更适合作为辅助线索，而不是 recall 主角。

## 失败与回退机制

如果配置文件：

- 不存在
- JSON 语法错误
- 字段值非法
- 覆盖规则为空

系统会打印 warning，并尽量回退到内建默认策略。

所以调参时建议：

1. 先小改
2. 保存
3. 运行一轮 agent
4. 观察行为是否符合预期

## 一个更偏调试的参考版本

如果你主要想服务“定位根因”，可以把下面这部分作为起点：

```json
{
  "recall": {
    "max_items": 6,
    "confidence_weight": {
      "low": 80,
      "medium": 220,
      "high": 420
    },
    "kind_weight": {
      "error_site": 100,
      "root_cause": 140,
      "entry_point": 60,
      "call_chain": 50,
      "symbol": 25,
      "code_path": 5,
      "config": 20,
      "todo": 0
    }
  },
  "persistence": {
    "max_persist_per_turn": 2,
    "min_confidence": "high",
    "priority_weight": {
      "low": 100,
      "medium": 160,
      "high": 220
    }
  }
}
```

## 相关文件

- 样例配置：[code_discovery_policy.json](file:///Users/bytedance/rust_tools/.rust_tools/code_discovery_policy.json)
- 策略实现：[code_discovery_policy.rs](file:///Users/bytedance/rust_tools/src/bin/ai/code_discovery_policy.rs)
- 分类接入：[messaging.rs](file:///Users/bytedance/rust_tools/src/bin/ai/driver/turn_runtime/tool_result/messaging.rs)
- recall 接入：[prepare.rs](file:///Users/bytedance/rust_tools/src/bin/ai/driver/turn_runtime/prepare.rs)
