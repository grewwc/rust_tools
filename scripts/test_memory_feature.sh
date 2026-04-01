#!/bin/bash
# 记忆与知识库检索功能快速测试脚本

set -e

echo "=========================================="
echo "记忆与知识库检索功能测试"
echo "=========================================="
echo

# 测试 1: 编译检查
echo "✓ 测试 1: 编译检查"
cd /Users/bytedance/rust_tools
cargo check --bin a --quiet
echo "  编译通过 ✓"
echo

# 测试 2: 保存记忆
echo "✓ 测试 2: 保存记忆"
echo "  保存测试记忆 1..."
cargo run --bin a -- /memory save "测试：Rust 项目使用 cargo fmt 格式化代码" \
    --category coding_guideline \
    --tags rust,format \
    --source test_script 2>&1 | grep -E "(Memory saved|Error)" || true

echo "  保存测试记忆 2..."
cargo run --bin a -- /memory save "测试：用户偏好英文代码注释" \
    --category user_preference \
    --tags code_style,communication \
    --source test_script 2>&1 | grep -E "(Memory saved|Error)" || true

echo "  保存测试记忆 3..."
cargo run --bin a -- /memory save "测试：项目 API 端点配置" \
    --category project_config \
    --tags api,config \
    --source test_script 2>&1 | grep -E "(Memory saved|Error)" || true
echo

# 测试 3: 查看最近记忆
echo "✓ 测试 3: 查看最近记忆"
cargo run --bin a -- /memory recent 3 2>&1 | head -20 || true
echo

# 测试 4: 搜索记忆
echo "✓ 测试 4: 搜索记忆"
echo "  搜索 'Rust'..."
cargo run --bin a -- /memory search Rust --limit 5 2>&1 | head -20 || true
echo

echo "  搜索 'cargo'..."
cargo run --bin a -- /memory search cargo --limit 5 2>&1 | head -20 || true
echo

# 测试 5: 查看帮助
echo "✓ 测试 5: 查看帮助"
cargo run --bin a -- /memory 2>&1 | head -15 || true
echo

echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo
echo "📝 提示："
echo "1. 手动测试自动检索功能：cargo run --bin a"
echo "2. 提问与保存的记忆相关的问题，如：'Rust 代码格式化用什么工具？'"
echo "3. 观察 agent 是否自动检索并使用保存的记忆"
echo
echo "⚙️  配置选项（添加到 ~/.configW）："
echo "  ai.knowledge_retrieval.enable = true"
echo "  ai.knowledge_retrieval.max_results = 8"
echo "  ai.knowledge_retrieval.max_age_days = 90"
echo
