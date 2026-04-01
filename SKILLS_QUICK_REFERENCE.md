# 🚀 a.rs Agent Skills 快速参考

## 📦 5 个已安装的 Rust 开发 Skills

### 1️⃣ rust-code-review
```
触发词：review this rust code | rust 代码审查 | check this rust code
功能：代码审查、安全性检查、性能优化、最佳实践
```

### 2️⃣ rust-testing
```
触发词：write rust tests | rust 测试 | rust unit test
功能：单元测试、集成测试、属性测试、基准测试
```

### 3️⃣ rust-documentation
```
触发词：write rust documentation | rust 文档 | rust api documentation
功能：API 文档、README、示例代码、技术文档
```

### 4️⃣ mcp-builder-rust
```
触发词：create mcp server rust | rust mcp | model context protocol rust
功能：MCP 服务器开发、工具设计、传输层配置
```

### 5️⃣ rust-project-helper
```
触发词：rust project structure | rust 项目结构 | cargo configuration
功能：项目规划、依赖管理、CI/CD、构建优化
```

---

## 💡 常用场景

### 场景 1: 新项目启动
```
1. rust-project-helper → 规划项目结构
2. rust-project-helper → 配置 Cargo.toml
3. rust-project-helper → 设置 CI/CD
```

### 场景 2: 代码开发
```
1. 编写代码
2. rust-code-review → 审查代码质量
3. rust-testing → 编写测试
4. rust-documentation → 编写文档
```

### 场景 3: 性能优化
```
1. rust-code-review → 识别性能问题
2. rust-testing → 添加基准测试
3. rust-project-helper → 优化构建配置
```

### 场景 4: AI 工具开发
```
1. mcp-builder-rust → 设计 MCP 工具
2. mcp-builder-rust → 实现服务器
3. rust-testing → 编写集成测试
4. rust-documentation → 编写 API 文档
```

---

## 🎯 使用技巧

### ✅ 好的提问方式
```
"请审查这个函数的内存安全性，特别是生命周期标注：
[代码]"

"为这个 Parser 模块编写完整的单元测试，包括边界情况：
[代码]"

"帮我规划一个支持插件系统的 Rust 项目结构"
```

### ❌ 避免的提问方式
```
"审查代码"  ← 太模糊
"写测试"     ← 缺少上下文
"怎么做？"   ← 问题不明确
```

---

## 📊 Skill 统计

| Skill | 大小 | 触发词 | 工具数 |
|-------|------|--------|--------|
| rust-code-review | 3.5K | 5 | 5 |
| rust-testing | 7.1K | 6 | 5 |
| rust-documentation | 8.2K | 5 | 4 |
| mcp-builder-rust | 15K | 4 | 5 |
| rust-project-helper | 8.8K | 6 | 6 |

**总计**: 42.6K 代码，26 个触发词

---

## 🔗 快速链接

- 📖 [完整使用指南](./SKILLS_GUIDE.md)
- 📝 [安装总结](./SKILLS_INSTALLATION_SUMMARY.md)
- 🏠 [Anthropic Skills 仓库](https://github.com/anthropics/skills)

---

## 🆘 快速故障排除

**Skill 不响应？**
- 检查触发词是否准确
- 确保提供足够上下文
- 重启 agent

**响应不符合预期？**
- 明确指定需求
- 提供代码示例
- 说明关注重点

---

**版本**: 1.0.0 | **更新时间**: 2025-04-01
