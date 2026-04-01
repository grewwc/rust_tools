# 意图识别重构：使用意图修饰符模式

## 重构背景

原代码将 `SearchForSkills` 作为一个特定的 `UserIntent` 枚举值，这种设计存在以下问题：

1. **不够通用**：`SearchForSkills` 是一个特定的业务场景，不应该硬编码在意图类型中
2. **扩展性差**：如果将来有 "SearchForTools"、"SearchForDocs" 等类似场景，需要不断添加新的枚举值
3. **职责混淆**：意图识别应该关注用户的**交流目的**（询问概念、请求行动、寻求方案等），而不是具体的**操作对象**

## 重构方案：意图修饰符模式

采用**核心意图 + 修饰符**的设计模式：

### 核心意图（CoreIntent）

只关注用户的交流目的，保持稳定和通用：

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreIntent {
    /// 询问概念/定义（"这是什么"、"是什么意思"）
    QueryConcept,
    /// 请求操作（"帮我做 X"）
    RequestAction,
    /// 寻求解决方案（"怎么处理"、"如何解决"）
    SeekSolution,
    /// 闲聊/其他
    Casual,
}
```

### 意图修饰符（IntentModifiers）

可扩展的元数据，用于描述意图的特殊属性：

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntentModifiers {
    /// 是否是搜索/查找类查询（"找几个"、"收集"、"有哪些"）
    pub is_search_query: bool,
    /// 目标资源类型（"skills", "tools", "docs" 等）
    pub target_resource: Option<String>,
    /// 是否包含否定词
    pub negation: bool,
}
```

### 完整的用户意图（UserIntent）

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserIntent {
    pub core: CoreIntent,
    pub modifiers: IntentModifiers,
}
```

## 主要改动

### 1. 意图检测逻辑

**重构前**：
```rust
let search_skill_patterns = [
    "找几个", "找一些", "有什么技能", "有哪些 skill",
    "推荐几个 skill", "搜索 skill", "搜索技能",
];

if search_skill_patterns.iter().any(|p| input.contains(p)) {
    return UserIntent::SearchForSkills;  // 特定的枚举值
}
```

**重构后**：
```rust
let search_patterns = [
    "找几个", "找一些", "找些", "收集",
    "有什么", "有哪些", "推荐几个", "推荐一些",
    "搜索", "查找",
];

if search_patterns.iter().any(|p| input.contains(p)) {
    modifiers.is_search_query = true;
    modifiers.target_resource = extract_target_resource(input);  // 提取资源类型
}
```

### 2. 资源类型提取

新增 `extract_target_resource` 函数，自动识别用户想要查找的资源类型：

```rust
fn extract_target_resource(input: &str) -> Option<String> {
    let resource_keywords = [
        ("技能", "skill"),
        ("工具", "tool"),
        ("文档", "doc"),
        ("文件", "file"),
        // ... 可扩展
    ];

    for (cn, en) in resource_keywords {
        if input.contains(cn) || input.contains(en) {
            return Some(en.to_string());
        }
    }

    None
}
```

### 3. 意图排除逻辑

**重构前**：
```rust
match intent {
    UserIntent::SearchForSkills => {
        // 特殊处理技能搜索
    }
    _ => false,
}
```

**重构后**：
```rust
match &intent.core {
    CoreIntent::RequestAction | CoreIntent::SeekSolution | CoreIntent::Casual => {
        // 如果用户是在搜索资源（如"找几个 skill"）
        if intent.is_searching_resource("skill") {
            // 排除执行类技能
        } else {
            false
        }
    }
    _ => false,
}
```

### 4. 意图匹配加分

**重构前**：
```rust
match intent {
    UserIntent::SearchForSkills => {
        // 给执行类技能负分
    }
    // 其他意图...
}
```

**重构后**：
```rust
// 先处理搜索资源的情况（独立于核心意图）
if intent.is_searching_resource("skill") {
    // 给执行类技能负分
}

// 再处理核心意图
match &intent.core {
    CoreIntent::QueryConcept => { ... }
    CoreIntent::RequestAction => { ... }
    // ...
}
```

## 优势

### 1. 更好的扩展性

现在可以轻松支持新的资源类型，无需修改 `CoreIntent` 枚举：

```rust
// 用户说"帮我找几个工具"
// → core: RequestAction, modifiers: { is_search_query: true, target_resource: Some("tool") }

// 用户说"有什么文档可以参考"
// → core: QueryConcept, modifiers: { is_search_query: true, target_resource: Some("doc") }
```

### 2. 更清晰的职责分离

- `CoreIntent`：关注**为什么**（用户的交流目的）
- `IntentModifiers`：关注**什么**和**如何**（操作对象和方式）

### 3. 更灵活的查询处理

支持组合查询，例如：
- "帮我找个工具来审查代码" 
  - `core: RequestAction`
  - `is_search_query: true`
  - `target_resource: Some("tool")`

### 4. 更容易维护

核心意图类型保持稳定（4 个），修饰符可以按需扩展，不会导致枚举爆炸。

## 使用示例

```rust
let intent = detect_intent("帮我找几个 review skill");

// 检查是否是搜索查询
if intent.is_search_query() {
    println!("用户在查找资源");
}

// 检查是否在查找特定类型的资源
if intent.is_searching_resource("skill") {
    println!("用户在查找技能");
}

// 访问核心意图
match &intent.core {
    CoreIntent::RequestAction => println!("用户请求操作"),
    CoreIntent::QueryConcept => println!("用户询问概念"),
    // ...
}
```

## 测试验证

所有现有测试通过（226 个测试），包括：
- 技能匹配逻辑测试
- 意图识别测试
- 工具调用测试

## 后续可扩展的修饰符

未来可以根据需要添加更多修饰符：

```rust
pub struct IntentModifiers {
    pub is_search_query: bool,
    pub target_resource: Option<String>,
    pub negation: bool,
    
    // 未来可扩展：
    // pub urgency: Option<UrgencyLevel>,      // 紧急程度
    // pub formality: FormalityLevel,          // 正式程度
    // pub context_hints: Vec<String>,         // 上下文提示
    // pub preferred_tool: Option<String>,     // 偏好的工具
}
```

## 总结

通过使用**意图修饰符模式**，我们成功地将特定的业务场景（`SearchForSkills`）从核心意图类型中解耦，使系统更加灵活、可扩展和易于维护。这种设计模式可以很好地平衡通用性和特殊性，是处理类似场景的良好实践。
