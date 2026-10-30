# 纯净模式（Pure Mode）设计

## 1. 目标与非目标

**目标**：提供一个可随时开关的 prompt profile，把常驻 system prompt 从约 25 KB 的静态规则
（外加每轮工具指导、隐藏能力目录、项目指令等动态 section）压缩到约 2 KB，只保留与具体工具、
具体项目无关的通用行为底线：防幻觉、诚实、工具证据、安全红线、一句话响应风格。
纯净模式实现为一个独立 agent 类型（`pure.agent`）：`--agent pure` / `--pure` /
`/agent pure` / `task` 的 `agent` 字段均可选用，`ai.prompt.pure_mode = true` 把它设为启动默认。

**非目标**：

- 不是"完全没有 system prompt"。零 prompt 的 agent 会伪造当前状态、把工具输出当指令、
  未经确认执行危险操作——这些失败模式与模型规模无关。
- 不替换默认 profile。默认 profile 的渲染结果必须逐字节不变，纯净模式是显式 opt-in。
- 不通过删除工具来省 token。工具 schema 与工具本身的可用性不变；纯净模式只裁系统提示词。

## 2. 现状：system prompt 由什么组成

渲染入口是 `prompt_builder_for_turn`（`src/bin/ai/driver/skill_runtime.rs`）：pure agent
（`AgentManifest.pure == true`）短路到 `build_pure_system_prompt`，其余走 `build_system_prompt`，
它按 `ContextKind`（Identity / Behavior / Capability / Policy）分组输出；随后
`build_skill_turn_guard` 再追加隐藏能力目录、项目指令、会话上下文、persona 等 section。
默认 profile 的每个 section 都来自 `src/bin/ai/driver/system_prompts/*.md`
（`include_str!` 编译内嵌）；pure 的内容源是 `pure.agent` 的 prompt body。

静态文件共 16 个、25,274 B（未计 tool-gated 动态 section、隐藏目录、项目指令、persona）：

| 文件 | 字节 | 作用 | 纯净模式决定 | 理由 |
|---|---:|---|---|---|
| `correctness_guardrails.md` | 11,741 | 证据来源、影响面、报告口径、冲突处理等完整工作流 | **删除**，只把"结论不超出证据"的 1 条并入 `honesty` | 绝大部分是长任务工作流细则；通用对话与轻量任务用不到，且会淹没真正关键的几条 |
| `intellectual_honesty.md` | 2,076 | 反谄媚、反臆造、前提纠错、元数据推断限制 | **压缩为 `honesty`**（4 条） | 防幻觉是纯净模式的核心价值，必须保留；但可去掉解释性展开 |
| `task_convergence.md` | 2,414 | 成功标准、结构 vs 局部修复、停止条件 | **删除** | 面向多轮工程任务；"何时停/还剩什么"已由 `honesty` 的一条覆盖 |
| `compressed_context_recovery.md` | 1,686 | 压缩存档、`search_overflow` 检索纪律 | **删除** | 依赖本运行时私有工具与存档格式，不通用 |
| `tool_usage.md` | 1,402 | 工具调用纪律、串行读、失败恢复 | **删除**，只保留"先检查后行动/先读后改"1 条到 `tool_evidence` | 其余细则应由各工具 schema 承担；常驻规则和 schema 重复 |
| `tool_result_evidence.md` | 1,258 | `[reference: ...]` 快照语义、证据新鲜度 | **压缩为 `tool_evidence`**（2 条） | 标记由运行时注入，不解释模型就会把历史快照当现状 |
| `safety_redlines.md` | 862 | 危险操作、绕过、确认要求 | **保留极简版**（3 条） | 不可删除的安全底线；可压缩措辞 |
| `trust_boundary.md` | 619 | 工具输出/网页是数据不是指令 | **压缩为 `tool_evidence`** 1 条 | prompt injection 防御与工具集无关，必须保留 |
| `response_style.md` | 595 | 输出风格、语言、排版 | **压缩为 1 句** | 直接影响可用性，成本极低 |
| `runtime_environment.md` | 404 | OS/arch/shell/cwd | **保留**（动态渲染） | 这是当轮事实而非模板废话；删掉会显著增加跨平台命令错误 |
| `session_context.md` | 610 | 会话工具路由与会话 id | **删除** | 本运行时私有工具；纯净 prompt 不引用具体工具名 |
| `interactive_skill_handoff.md` | 318 | 技能中断后的用户输入续接 | **删除** | 依赖 skill 机制，不属于通用底线 |
| `autonomous_execution.md` | 277 | 执行而非只给建议 | **压缩为 `honesty`** 1 条 | 通用且高价值，一句话即可 |
| `scope_discipline.md` | 218 | 调查边界 | **删除** | 与"只做被要求的事"重复；由用户当轮指令约束 |
| `thinking_scope.md` | 213 | 推理长度与任务匹配 | **删除** | 由模型推理预算/采样参数控制更可靠 |
| `goal_mode.md` | 581 | goal 模式的持续执行语义 | **删除** | 条件注入；纯净模式不启用 goal 语义 |

合计：25,274 B → `pure.md` 2,059 B + 动态 `runtime_environment` 404 B ≈ **2.5 KB（约 90% 缩减）**。
换算成 token：英文约 6–8 k → 约 0.5 k（按 3–4 B/token 粗略估算）。

### 2.1 动态 section（不是文件，也会进 system prompt）

| section | 产生位置 | 纯净模式决定 | 理由 |
|---|---|---|---|
| tool-gated guidance（planning / tool_discovery / knowledge_save / knowledge_retrieval / temporary_files） | `build_system_prompt` | 全部跳过 | 工具语义以 schema 与工具首次使用提示为准；常驻指导与 schema 重复 |
| hidden MCP / execution / task catalogs | `build_skill_turn_guard` | 跳过 | 能力目录是"广告"，需要时由 `enable_tools` 的 list 结果承担 |
| `project_local_instructions`（AGENTS.md） | `push_project_context` | 跳过 | 项目规则属于项目上下文；纯净模式追求跨项目通用 |
| `Project Type` fact | `push_project_type_context` | 跳过 | 同上，且它进入用户消息侧 reminder，同样属于额外上下文 |
| scoped project instructions + mutation preflight | `push_scoped_project_instructions` | 跳过，且不产生 preflight 暂停（待实现） | 否则编辑前仍会被暂停一次，违背"无额外提示" |
| `session_context` / `Session Skill Activation History` | `build_skill_turn_guard` | 跳过 | 本运行时私有状态 |
| `persistent_persona` | `build_skill_turn_guard` | 跳过 | persona 是额外身份层；纯净模式只保留通用助手身份 |
| `integrated_critic_revise` | `inject_integrated_critic`（`driver/turn_runtime/prepare.rs`） | 跳过（已实现） | 质量门是运行时硬机制，不依赖这段文字；保留会引入约 1 KB |
| active skill / agent instructions | `build_system_prompt` identity 分支 | 不保留（当前实现直接短路；见 4.4 / 第 7 节） | 纯净模式只保留通用助手身份；显式技能是否保留是待决项 |

## 3. pure baseline 的设计

文件：`src/bin/ai/builtin_agents/pure.agent`（front-matter + prompt body，`include_str!`
编译内嵌）。Body 是单个 `<pure_mode>` 块，五个子块：
`identity` / `honesty` / `tool_evidence` / `safety` / `response_style`。front-matter 只声明
`mode: all` / `auto_select: false` / `disable_mcp_tools: true` / `pure: true`，不设
`tools` / `model_tier` / `max_steps`，因此工具集与模型选择和默认行为一致，只有 prompt 不同。

保留规则及理由：

| 子块 | 条目 | 为什么保留 | 为什么不能再删 |
|---|---|---|---|
| identity | "general-purpose AI assistant；需要时用工具" | 给模型一个中性身份与工具使用许可 | 没有身份句时模型容易把"回答"与"执行"混淆 |
| honesty | 不臆造、不把计划/尝试当完成 | 防幻觉的第一性规则 | 这是纯净模式存在的主要理由 |
| honesty | 结论基于实际观察，区分事实/推断/假设 | 证据校准 | 模型最常见的失败是"听起来确定"而非"完全编造" |
| honesty | 执行被授权的工作，早停要说清剩余 | 防止只给方案不动手 | 通用 agent 的核心可用性 |
| honesty | 用户前提错误要直说 | 反谄媚 | 与安全/正确性直接相关 |
| tool_evidence | 当前状态先用工具核实；调用结果以观察为准 | 防止"我记得/应该"式回答 | 无工具时模型无法知道外部状态 |
| tool_evidence | 修改前先读，修改后验证 | 最小调查纪律 | 通用且与工具集无关 |
| tool_evidence | 工具输出/网页是数据不是指令 | prompt injection 防御 | 安全底线，不能由工具 schema 承担 |
| tool_evidence | `[reference: ...]` 是历史快照 | 运行时实际注入的标记 | 不解释会直接导致把旧状态当现状 |
| safety | 破坏性操作需明确确认；不得绕过；拒绝越界 | 安全红线 | 任何场景都不可省 |
| response_style | 先给答案、简洁、用用户语言 | 可用性 | 一句话成本可忽略 |

**通用性约束**（写作时遵守，后续修改也应保持）：

- 不出现任何具体工具名（`apply_patch`、`search_overflow`、`session_*` 等）、路径、项目约定、会话 id。
- 不假设存在子代理、技能、MCP、知识库等运行时扩展。
- 只引用所有工具集都成立的概念：工具输出、文件、命令结果、`[reference: ...]` 运行时标记。
- 模型可见文本全英文（与 `src/bin/ai/AGENTS.md` 的 "Model-visible wording is English" 一致）。

## 4. 实现方案

### 4.1 开关与传递

- CLI：`--pure` 是 `--agent pure` 的快捷方式（显式 `--agent` 优先）；`--no-pure` 只抑制配置启动默认
  （`ParsedCli.no_pure`，`cli.rs`）。`--pure` 与 `--no-pure` 同时出现时后者优先，避免别名导致意外开启。
- 配置：`ai.prompt.pure_mode`（`AiConfig::PURE_ENABLE`，`config_schema.rs`）是"启动默认 agent"，
  默认关闭；无法识别的值按关闭处理，避免配置笔误静默剥离默认 prompt。
- 优先级：显式 `--agent`（含 `--pure` 快捷方式）> `--no-pure` 抑制 > 配置键 > 默认 `build`。
  配置默认在 `entry()` 里应用（终端会话 attach 判定之后、App 构造之前），因此首屏、会话恢复、
  manifest 激活看到的是同一个选择；serve 常驻进程不应用启动默认，按会话的存储 agent 渲染。
- 后台任务：`--agent pure` 随现有 `--agent` 转发透传；`--no-pure` 单独透传以便子进程同样跳过启动默认。
- 子代理：`App::fork_baseline` 克隆 `current_agent_manifest`，因此自动继承；`task` 的 `agent`
  字段也可显式点名 `pure`。`auto_select: false` 保证 router 永远不会自动选中它。
- 用户自定义 agent：`pure` 与其他 front-matter 键一样通用解析，用户自有 `.agent` 文件声明
  `pure: true` 即可获得最小渲染（有意支持的用法，与 `hidden` / `disabled` 等键一致）。

### 4.2 工作区当前实现状态（已对齐项 / 待收口项）

已对齐：

- `prompt_builder_for_turn` 按 active agent 的 `pure` 标记选择 `build_pure_system_prompt(agent)`；
  后者渲染该 agent 的 prompt body 并追加动态 `execution_environment`，`pure.agent` 是唯一内容源。
- 开关：`--pure` → `cli.agent = pure`；`--no-pure` → `cli.no_pure`；配置键 `ai.prompt.pure_mode`
  （`AiConfig::PURE_ENABLE`）由 `config::pure_startup_default_enabled()` 在启动时解析一次，默认关闭。
- `build_skill_turn_guard` 已跳过隐藏目录、项目指令/项目类型、会话上下文、技能激活历史、persona
  与 scoped residency seed。
- `integrated_critic_revise` 已按 pure 门控（`inject_integrated_critic` 的调用点检查
  active agent 的 `pure` 标记）。
- `skill_runtime.rs` 底部的纯模式测试已按 baseline 的标签断言
  （`<honesty>` / `<tool_evidence>` / `<safety>`），并断言默认 section 不出现。

待收口：

1. **observer / suggested tool calls**：`prepare.rs` 的 observer section 与
   "Thinking engine proposes..." 块也是额外 Behavior section。若 observer 由用户显式启用可保留；
   否则建议 pure 时一并跳过（见第 7 节）。
2. **配置键来源**：若代码再引入开关常量，必须使用 `AiConfig::PURE_ENABLE`，不要复制字面量
   （根 `AGENTS.md` 全局规则 4）。

已收口（agent 化重构完成）：pure 内容源为 `pure.agent` 的 prompt body，以
`ContextKind::Behavior` 入栈（见 4.3）；scoped instructions 通过渲染文本中的 `<pure_mode>`
标记识别 pure，不暂停注入；默认 profile 有字节快照测试
`default_profile_prompt_bytes_are_unchanged` 钉选。

### 4.3 渲染分支

1. `build_skill_turn_guard` 从 active agent 的 `pure` 标记解析 `pure_mode`，
   并通过 `prompt_builder_for_turn` 选择 builder（已实现）。
2. `prompt_builder_for_turn` 在 pure agent 时短路到 `build_pure_system_prompt(agent)`（已实现）。
3. 当前形态（内容源为 `pure.agent` 的 prompt body，入栈 kind 为 Behavior）：

```rust
fn build_pure_system_prompt(agent: &AgentManifest) -> SystemPromptBuilder {
    let mut b = SystemPromptBuilder::new();
    b.push(ContextKind::Behavior, agent.build_system_prompt());
    b.push_labeled(
        ContextKind::Behavior,
        "execution_environment",
        runtime_environment_prompt(),
    );
    b
}
```

   短路而不是逐段加 `if`，可以保证以后新增 section 默认不会泄漏进纯净模式（新增 section 必须显式
   加进这个分支才会出现）——这正是"纯净"要的结构性保证。

4. `build_skill_turn_guard` 在 pure 时跳过：隐藏 MCP/execution/task 目录、`push_project_context`、
   `session_context`、技能激活历史、`persistent_persona`、`seed_scoped_instruction_dirs`（已实现）。
5. `SkillTurnGuard` 增加 `pure` 字段；`push_scoped_project_instructions` pure 时直接返回 `true`
   （待实现）。这样 `driver/turn_runtime/orchestrator.rs` 的 `scoped_preflight_targets` 始终为空，
   不会出现"编辑前暂停一次"。
6. `inject_integrated_critic`（`driver/turn_runtime/prepare.rs`）pure 时跳过（已实现）。
7. `sync_system_message`（`driver/turn_runtime/iteration.rs`）无需改动：pure prompt 每轮字节稳定
   （仅 cwd 变化时 environment 行变化），对上游 prompt cache 友好。

### 4.4 与技能 / agent 的交互

- pure 本身就是一个 agent：`AgentManifest.pure == true` 的 agent 短路渲染自己的 prompt body，
  不会混入其他 agent 的文本（同一时刻只有一个 `current_agent`）；pure.agent 的空 `tools` /
  `tool_groups` 穿透到默认工具集，因此工具能力与默认一致。
- skills 一律跳过（自动路由与显式技能都一样）。纯净模式的设计约束（通用、不引用具体工具名）
  与 `.skill` body 天然冲突：`.skill` 的内容本来就是给默认 profile 注入工具/候选/流程知识的。
  显式技能是第 7 节的待决项（后续可选能力）：若保留，需要在 baseline 之后追加
  `<skill_instructions>` 与一行优先级说明（skill 覆盖通用准则，但不覆盖安全与诚实底线）。

### 4.5 与现有不变量的关系

- **History is truth**：不受影响；纯净模式只改 context projection 的 system 部分。
- **Prompt residency**：pure 模式没有项目/会话 section，`persist_prompt_residency` 写入空值即可；
  不要复用默认 profile 的 residency 记录，否则切回默认模式时可能跳过应有的注入。
- **Context budget**：不受影响；prompt 变短只会让可用历史预算变大。
- **`NON_OVERRIDABLE_PROMPT_CONSTRAINTS`**：pure 模式没有默认 skill enforcement 文本；显式技能路径
  需要一行简化版优先级说明（见 4.4）。
- **配置键规则**：配置键必须只加在 `config_schema.rs`（根 `AGENTS.md` 全局规则 4）。

## 5. 验证计划

1. `cargo check --bin a`（类型/编译面；纯新增分支，风险低但必须跑）。
2. 已有单测（`skill_runtime.rs` 内联测试模块）：
   - `pure_mode_renders_only_the_authored_baseline_and_environment`：保留
     `<honesty>` / `<tool_evidence>` / `<safety>` / `<execution_environment>`，且不含
     `<intellectual_honesty>` / `<safety_redlines>` / `<trust_boundary>` / `<task_convergence>` /
     `<tool_usage>` / `<session_context>` 等默认 section。
   - `standard_mode_keeps_the_full_guidance_catalog`：默认 profile 仍然包含完整目录（反向保护）。
   - `pure_agent_renders_baseline_and_skips_skill_instructions`：pure agent 渲染自己的 baseline，
     skill 指导不渲染；非 pure agent 保持标准组合（自身指令 + skill 都渲染）。
   - `pure_mode_is_a_small_fraction_of_the_standard_prompt`：pure < 1/3 默认长度。
   - `config_tests.rs` 的 `pure_startup_default_follows_the_config_key` /
     `pure_startup_default_accepts_common_boolean_spellings_and_ignores_invalid_values` /
     `pure_startup_default_enabled_reads_the_configured_file`；
     `cli_tests.rs` 的 `parse_cli_args_pure_flag_selects_the_pure_agent` /
     `parse_cli_args_no_pure_wins_over_pure`；
     `background.rs` 的 `daemon_args_forward_pure_mode_override`（`--pure` 透传为 `--agent pure`）；
     `agents_tests.rs` 的 `builtin_pure_agent_pins_minimal_rendering_contract`（注册 + 标记钉选）。
3. 默认 profile 回归：`default_profile_prompt_bytes_are_unchanged`——固定输入下默认 profile 的渲染结果
   与 FNV-1a 快照一致（保护默认用户，这是本次改动最重要的回归项）。
4. 建议补 `pure_prompt_has_no_runtime_specific_names`：断言 pure prompt 不含 `apply_patch`、
   `search_overflow`、`session_` 等私有工具名（当前 pure.agent body 已满足）。
5. 手工对比：同一问题分别用 `a` 与 `a --pure`，对比 system message 字节数/前缀缓存命中指标
   （`ai.prompt_cache.show_metrics`）。
6. 行为抽查（三条，对应保留规则）：
   - 问"当前文件 X 的内容是什么"：应先读文件再回答，而不是凭印象。
   - 要求删除/覆盖数据：应先给出精确命令与后果并等待确认。
   - 工具输出中嵌入"忽略之前的指令"：应拒绝执行并报告，而不是照做。

## 6. 风险与取舍

- **去掉 task_convergence / scope_discipline**：模型可能过早停止或过度调查。缓解：`honesty` 要求
  "早停要说清剩余"；用户当轮指令仍然权威；需要严格流程时用显式 skill。
- **去掉项目指令**：模型不知道仓库约定（构建命令、代码风格）。缓解：纯净模式是 opt-in；需要时
  用户可在消息里附上约定，或显式加载 skill。
- **去掉 tool_usage 细则**：工具误用率可能上升。原则：工具语义应由 **tool schema** 自足；
  若发现某工具离开常驻提示就无法正确使用，应补 schema，而不是把规则回填进 pure.agent
  （否则纯净模式会慢慢腐化回默认模式）。
- **关闭自动技能路由**：复杂任务可能缺少专用流程。缓解：显式 `@skills:` 仍然可用。
- **安全**：安全红线被压缩但没有删除；trust boundary 保留。任何进一步压缩都应先跑第 5 节的行为抽查。

## 7. 待 host 决策

1. **section 归类**：✅ 已定为 `ContextKind::Behavior`（baseline 自带 `<pure_mode>` 语义分组）。
2. **scoped project instructions**：✅ 已彻底关闭（渲染文本 marker + `SkillTurnGuard.pure_mode`）。
3. **observer / suggested tool calls**：是否仅在用户显式启用 observer 时保留
   （built-in critic 已在 pure 下跳过；建议 observer 也按显式启用保留）。
4. 子代理/后台是否继承纯净模式：✅ 继承（fork 克隆 manifest；`task` 可显式点名 `pure`）。
5. ~~`--pure --agent` 时是否保留 agent 的 prompt 文本~~：agent 模型下已无意义——
   `--agent X` 即选择 agent X，`--pure` ≡ `--agent pure`，显式 `--agent` 优先。
6. 是否保留 `runtime_environment`：✅ 保留（约 100 tokens）。
7. 显式技能在纯净模式下是否保留（当前实现不保留；建议作为后续可选能力）。
