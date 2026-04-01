# a.rs Agent Skills 使用指南

我已经为 a.rs 项目创建了 5 个定制的 Rust 开发 Skills，这些 Skills 基于 Anthropic 官方的 Skills 仓库，并针对 Rust 开发场景进行了优化。

## 📦 已安装的 Skills

### 1. **rust-code-review** - Rust 代码审查专家
**用途**: 进行全面的 Rust 代码质量检查

**触发词**:
- "review this rust code"
- "rust 代码审查"
- "check this rust code"
- "rust code quality"
- "rust 最佳实践"

**功能**:
- 🔴 安全性检查（内存安全、并发安全、错误处理）
- 🟡 性能优化（零成本抽象、迭代器、数据结构）
- 🟢 代码质量（Rust 惯用法、类型系统、模式匹配）
- 📝 文档和测试审查

**示例**:
```
请审查这段 Rust 代码的安全性问题：
[粘贴代码]
```

---

### 2. **rust-testing** - Rust 测试专家
**用途**: 编写全面的 Rust 测试套件

**触发词**:
- "write rust tests"
- "rust 测试"
- "test this rust code"
- "rust unit test"
- "rust integration test"
- "rust benchmark"

**功能**:
- ✅ 单元测试（`#[cfg(test)]` 模块）
- ✅ 集成测试（`tests/` 目录）
- ✅ 属性测试（proptest/quickcheck）
- ✅ 基准测试（criterion）
- ✅ 文档测试

**示例**:
```
为这个函数编写单元测试和属性测试：
pub fn parse_int(s: &str) -> Result<i32, ParseError> { ... }
```

---

### 3. **rust-documentation** - Rust 文档专家
**用途**: 编写清晰的 Rust API 文档和技术文档

**触发词**:
- "write rust documentation"
- "rust 文档"
- "rust doc comments"
- "rust readme"
- "rust api documentation"

**功能**:
- 📖 API 文档注释（`///` 和 `//!`）
- 📄 README 编写
- 💡 示例代码
- 📚 技术文档

**示例**:
```
为这个模块编写完整的 API 文档：
pub mod parser { ... }
```

---

### 4. **mcp-builder-rust** - Rust MCP 服务器开发
**用途**: 使用 Rust 创建 Model Context Protocol 服务器

**触发词**:
- "create mcp server rust"
- "rust mcp"
- "model context protocol rust"
- "rust ai tools"

**功能**:
- 🛠️ MCP 服务器架构
- 🔌 工具设计和实现
- 📡 传输层（stdio/HTTP）
- ✅ 测试和部署

**示例**:
```
帮我创建一个文件操作的 MCP 服务器，支持读取和写入文件
```

---

### 5. **rust-project-helper** - Rust 项目开发助手
**用途**: 项目规划、依赖管理、CI/CD 配置

**触发词**:
- "rust project structure"
- "rust 项目结构"
- "cargo configuration"
- "rust ci/cd"
- "rust workspace"
- "rust dependencies"

**功能**:
- 📁 项目结构规划
- 📦 Cargo.toml 配置
- 🔧 构建优化
- 🚀 CI/CD 配置（GitHub Actions）
- ✅ 发布检查清单

**示例**:
```
帮我规划一个支持 WASM 的 Rust 库项目结构
```

---

## 🎯 使用技巧

### 1. 组合使用 Skills

可以组合多个 Skills 来完成复杂任务：

```
1. 先用 rust-project-helper 规划项目结构
2. 用 rust-code-review 审查代码质量
3. 用 rust-testing 编写测试
4. 用 rust-documentation 编写文档
```

### 2. 提供上下文

使用 Skills 时，提供足够的上下文会得到更好的结果：

```
✅ 好：请审查这个 HTTP 客户端的代码，重点关注错误处理和超时机制
[代码]

❌ 差：审查这段代码
[代码]
```

### 3. 指定关注点

明确告诉 Skill 你关心的方面：

```
请重点检查这个函数的：
- 内存安全性
- 并发安全性
- 性能瓶颈
```

---

## 📂 Skill 文件位置

所有 Skills 存储在：
```
/Users/bytedance/.config/rust_tools/skills/
```

文件列表：
- `rust-code-review.skill`
- `rust-testing.skill`
- `rust-documentation.skill`
- `mcp-builder-rust.skill`
- `rust-project-helper.skill`

---

## 🔧 自定义 Skills

如果需要修改某个 Skill，可以直接编辑对应的 `.skill` 文件：

```bash
# 编辑代码审查 Skill
nano /Users/bytedance/.config/rust_tools/skills/rust-code-review.skill

# 重新加载 Skills（如果需要）
# 重启 a.rs agent 或发送重载命令
```

---

## 📚 参考资源

### Anthropic 官方 Skills
- 仓库：https://github.com/anthropics/skills
- 文档：https://support.claude.com/en/articles/12512198-creating-custom-skills

### Rust 开发资源
- Rust Book: https://doc.rust-lang.org/book/
- Rust by Example: https://doc.rust-lang.org/rust-by-example/
- Cargo Book: https://doc.rust-lang.org/cargo/

### MCP 协议
- 规范：https://modelcontextprotocol.io/
- TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
- Python SDK: https://github.com/modelcontextprotocol/python-sdk

---

## 💡 最佳实践

1. **定期更新 Skills**: 根据实际使用情况优化 Skills
2. **分享反馈**: 如果某个 Skill 特别好用或需要改进，记录下来
3. **创建项目特定 Skills**: 为特定项目创建定制 Skills
4. **保持简洁**: Skills 应该专注解决特定问题

---

## 🆘 故障排除

### Skill 不触发？
- 检查触发词是否匹配
- 确保 Skill 文件在正确位置
- 重启 agent

### Skill 响应不符合预期？
- 提供更详细的上下文
- 明确指定需求
- 考虑修改 Skill 的 prompt

---

**创建时间**: 2025-04-01
**版本**: 1.0.0
**作者**: a.rs Agent
