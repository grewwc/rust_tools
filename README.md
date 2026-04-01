# Rust Tools

[![License](https://img.shields.io/crates/l/rust_tools.svg)](LICENSE)

一个多功能的 Rust 工具库，提供常用的数据结构、算法和实用工具。

## 特性

- 🚀 **高性能** - 使用 `rustc-hash` 提供快速的哈希表实现
- 📦 **模块化** - 按需使用各个模块，灵活的 API 设计
- 🔧 **实用** - 提供日常开发中常用的工具函数
- 📚 **文档完善** - 每个模块都有详细的使用示例和说明
- ✅ **测试覆盖** - 完善的单元测试确保代码质量

## 模块概览

### 数据结构 (`cw`)

提供多种常用的数据结构和容器实现：

- **基础数据结构**: `Queue`, `Stack`, `DequeList`
- **映射和集合**: `OrderedMap`, `OrderedSet`, `TreeMap`, `TreeSet`, `ConcurrentHashMap`
- **高级数据结构**: `LruCache`, `PriorityQueue`, `BloomFilter`, `Counter`, `Trie`, `SkipList`, `UF` (并查集)
- **图结构**: `DirectedGraph`, `UndirectedGraph`, `WeightedGraph`, `Mst` (最小生成树)

### 字符串处理 (`strw`)

提供丰富的字符串操作工具：

- 字符串修剪和清理
- 字符串分割和合并
- 文本格式化和换行
- 字符串搜索和查找
- 数值计算

### JSON 处理 (`jsonw`)

提供 JSON 数据处理功能：

- JSON 解析和格式化
- JSON 差异比较
- JSON 输入清理
- JSON 排序

### 算法工具 (`algow`)

提供常用算法实现：

- 二分查找 (`bisect_left`, `bisect_right`)
- 更多算法持续添加中...

### 剪贴板操作 (`clipboard`)

提供跨平台剪贴板访问：

- 文本读写
- 图片读写
- 二进制数据读写
- SSH 会话支持 (OSC52 协议)

### 终端工具 (`terminalw`)

提供终端相关工具：

- 文件查找（支持并发、过滤、排除）
- Glob 模式匹配
- 命令解析
- 路径处理

### 命令执行 (`cmd`)

提供系统命令执行功能：

- 命令执行和输出捕获
- 超时控制
- 工作目录设置
- 自动 Shell 检测

### PDF 处理 (`pdfw`)

提供 PDF 文件处理功能：

- PDF 文本提取
- PDF 转图片
- OCR 支持

### 排序工具 (`sortw`)

提供多种排序算法：

- TimSort
- 计数排序
- 基数排序
- Top-K 选择

## 安装

### 作为库使用

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
rust_tools = { path = "./rust_tools" }  # 本地路径
# 或
# rust_tools = "0.1.0"  # 发布到 crates.io 后
```

### 构建二进制工具

```bash
# 构建所有二进制文件
cargo build --release

# 构建特定的二进制文件
cargo build --release --bin pdf
cargo build --release --bin ai
```

## 快速开始

### 使用数据结构

```rust
use rust_tools::cw::{Queue, Counter, BloomFilter};

// 队列
let mut queue: Queue<i32> = Queue::new();
queue.enqueue(1);
queue.enqueue(2);
assert_eq!(queue.dequeue(), Some(1));

// 计数器
let mut counter: Counter<char> = Counter::new();
for c in "hello world".chars() {
    counter.inc(c);
}
assert_eq!(counter.get(&'l'), 3);

// 布隆过滤器
let mut bf = BloomFilter::with_rate(1000, 0.01);
bf.insert("hello");
assert!(bf.contains("hello"));
```

### 字符串处理

```rust
use rust_tools::strw::{trim_cutset, split_no_empty};

// 字符串修剪
let trimmed = trim_cutset("xxxhelloxxx", "x");
assert_eq!(trimmed, "hello");

// 字符串分割
let parts: Vec<&str> = split_no_empty("a,,b,,,c", ",");
assert_eq!(parts, vec!["a", "b", "c"]);
```

### 算法工具

```rust
use rust_tools::algow::{bisect_left, bisect_right};

let arr = [1, 3, 5, 7, 9];
let pos = bisect_left(&arr, &5);
assert_eq!(pos, 2);

// 处理重复元素
let arr_with_dups = [1, 3, 3, 3, 5];
let left = bisect_left(&arr_with_dups, &3);
let right = bisect_right(&arr_with_dups, &3);
assert_eq!(&arr_with_dups[left..right], &[3, 3, 3]);
```

### 剪贴板操作

```rust
use rust_tools::clipboard::{get_clipboard_content, set_clipboard_content};

// 读取剪贴板
let text = get_clipboard_content();
println!("剪贴板内容：{}", text);

// 写入剪贴板
set_clipboard_content("Hello, World!").expect("设置失败");
```

### 命令执行

```rust
use rust_tools::cmd::{run_cmd, run_cmd_output_with_timeout};
use std::time::Duration;

// 基本命令执行
let output = run_cmd("echo Hello").expect("命令失败");
println!("{}", output);

// 带超时控制
match run_cmd_output_with_timeout(
    "sleep 5",
    Default::default(),
    Duration::from_secs(2),
) {
    Ok(_) => println!("完成"),
    Err(e) if e.kind() == std::io::ErrorKind::TimedOut => println!("超时"),
    Err(e) => println!("错误：{}", e),
}
```

## 文档

### 生成文档

```bash
# 生成文档
cargo doc

# 生成并打开文档
cargo doc --open

# 包含私有 items（开发时使用）
cargo doc --document-private-items
```

生成的文档位于 `target/doc/rust_tools/index.html`

### 在线文档

文档发布后可以通过 docs.rs 访问。

## 测试

```bash
# 运行所有测试
cargo test

# 运行库测试
cargo test --lib

# 运行文档测试
cargo test --doc

# 运行特定模块的测试
cargo test cw::queue
```

## 项目结构

```
rust_tools/
├── src/
│   ├── lib.rs              # 库入口
│   ├── algow/              # 算法工具
│   ├── clipboard/          # 剪贴板操作
│   ├── cmd/                # 命令执行
│   ├── common/             # 通用工具
│   ├── cw/                 # 数据结构
│   ├── jsonw/              # JSON 处理
│   ├── pdfw/               # PDF 处理
│   ├── sortw/              # 排序工具
│   ├── strw/               # 字符串处理
│   └── terminalw/          # 终端工具
├── src/bin/                # 二进制工具
│   ├── pdf.rs
│   ├── ai/
│   └── ...
├── tests/                  # 集成测试
├── Cargo.toml
└── README.md
```

## 二进制工具

本项目还提供多个实用的命令行工具：

- `pdf` - PDF 处理工具
- `ai` - AI 助手工具
- `ff` - 快速文件查找
- `fk` - 文件搜索
- `re` - 备忘录工具
- 更多工具见 `src/bin/` 目录

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- [rustc-hash](https://github.com/rust-lang/rustc-hash) - 快速哈希函数
- [arboard](https://github.com/1Password/arboard) - 跨平台剪贴板访问
- 其他优秀的 Rust 开源项目
