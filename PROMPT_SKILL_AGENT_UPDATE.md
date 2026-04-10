# Prompt-Skill Agent 更新说明

## 更新内容

为 `a.rs` agent 添加了新的内置 agent：**prompt-skill**，专门用于优化和生成 prompt/skill。

## 新增文件

### 1. 核心 Agent 定义
- **文件**: `src/bin/ai/builtin_agents/prompt-skill.agent`
- **功能**: 定义了 prompt-skill agent 的元数据和系统提示词
- **特性**:
  - 专注于 prompt 工程和 skill 设计
  - 支持中英文路由标签
  - 配置了必要的工具权限（save_skill, read_file_lines, write_file, apply_patch 等）
  - 使用 heavy model tier 以获得更好的推理能力

### 2. Agent 注册
- **文件**: `src/bin/ai/agents.rs`
- **修改**: 在 `BUILTIN_AGENTS` 常量中添加了 prompt-skill.agent 的引用

### 3. 文档
- **文件**: `docs/prompt-skill-agent-guide.md`
- **内容**: 详细的使用指南、最佳实践和示例

### 4. 示例文件
- **文件**: `examples/before-optimization.txt`
- **用途**: 展示需要优化的 prompt 示例

## 使用方法

### 基本用法

```bash
# 使用 prompt-skill agent
cargo run --bin a -- --agent prompt-skill "帮我优化这个 prompt"

# 创建新的 skill
cargo run --bin a -- --agent prompt-skill "创建一个用于 API 设计的 skill"

# 分析现有 skill
cargo run --bin a -- --agent prompt-skill "分析这个 skill 文件的质量"
```

### 典型应用场景

1. **优化现有 Prompt**
   ```
   请优化以下 prompt，使其更清晰和结构化：
   [粘贴需要优化的内容]
   ```

2. **生成新 Skill**
   ```
   我需要创建一个用于代码审查的 skill，要求：
   - 检查代码风格
   - 发现潜在 bug
   - 提供改进建议
   ```

3. **Skill 质量评估**
   ```
   请分析这个 skill 文件的优缺点：
   [粘贴 skill 文件内容]
   ```

## 技术细节

### Agent 配置

```yaml
name: prompt-skill
description: Specialized agent for optimizing and generating prompts and skills
mode: primary
color: info
model_tier: heavy
routing_tags:
  - prompt
  - skill
  - optimize
  - generate
  - engineer
  - 提示词
  - 技能
  - 优化
  - 生成
tools:
  - save_skill
  - read_file_lines
  - write_file
  - apply_patch
  - code_search
  - search_files
```

补充说明：当 `code_search` 用于结构化搜索时，应写成 `operation=structural` 并配合 `intent=find_functions|find_classes|find_methods|find_calls`，不要把 `find_functions`、`find_classes`、`find_methods`、`find_calls` 直接写到 `operation` 中。

### 核心能力

1. **Prompt 优化**
   - 应用提示词工程最佳实践
   - 提高清晰度和可执行性
   - 优化结构和组织方式

2. **Skill 生成**
   - 创建符合规范的 skill 文件
   - 设计准确的描述用于路由
   - 配置适当的工具权限

3. **模板设计**
   - 提供常见模式的模板
   - 生成示例和最佳实践
   - 确保 YAML front matter 语法正确

### 设计原则

- **最小改动**: 优先使用 apply_patch 进行精确修改
- **验证驱动**: 每次修改后验证 YAML 语法和结构
- **用户导向**: 从最终用户角度审视 prompt/skill 的有效性
- **一致性**: 保持与现有 agents/skills 的风格一致

## 验证

编译测试通过：
```bash
$ cargo build --bin a
Compiling rust_tools v0.1.0
Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.28s
```

Agent 已成功注册到系统中，可以通过 `--agent prompt-skill` 参数调用。

## 后续改进建议

1. 添加更多的 skill 模板示例
2. 创建 prompt 质量评估指标
3. 实现自动化的 prompt 测试框架
4. 添加多语言支持的最佳实践
5. 集成 A/B 测试功能来比较不同 prompt 的效果

## 相关文件

- Agent 定义: `src/bin/ai/builtin_agents/prompt-skill.agent`
- Agent 注册: `src/bin/ai/agents.rs`
- 使用指南: `docs/prompt-skill-agent-guide.md`
- 示例文件: `examples/before-optimization.txt`
