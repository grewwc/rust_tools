#!/bin/bash
# 清理临时测试文件脚本

set -e

echo "清理临时测试文件..."

cd /Users/bytedance/rust_tools

# 删除根目录的测试源代码文件
echo "删除 test_*.rs 文件..."
rm -f test_chars.rs
rm -f test_row.rs
rm -f test_stream.rs
rm -f test_stream2.rs
rm -f test_table.rs
rm -f test_table2.rs

# 删除编译后的二进制文件
echo "删除编译后的二进制文件..."
rm -f test_chars
rm -f test_row
rm -f test_stream
rm -f test_stream2
rm -f test_table
rm -f test_table2

# 删除旧的 clipboard 测试
echo "删除旧的 clipboard 测试..."
rm -f bin/test_clipboard

# 移动功能测试脚本到 scripts 目录
echo "移动功能测试脚本到 scripts/..."
mv -f test_memory_feature.sh scripts/

echo "清理完成！"
echo
echo "已删除的文件:"
echo "  - test_chars.rs, test_row.rs, test_stream.rs, test_stream2.rs, test_table.rs, test_table2.rs"
echo "  - test_chars, test_row, test_stream, test_stream2, test_table, test_table2 (二进制)"
echo "  - bin/test_clipboard (旧的 clipboard 测试)"
echo
echo "已移动的文件:"
echo "  - test_memory_feature.sh -> scripts/test_memory_feature.sh"
