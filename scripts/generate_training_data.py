#!/usr/bin/env python3
"""Augment training corpus by generating variations from seed samples."""
import json
import sys
from pathlib import Path

INTENT_SEEDS = {
    "query_concept": [
        "Rust 的所有权是什么", "什么是 trait object", "borrow checker 是什么意思",
        "请解释一下 tokio runtime", "Option 和 Result 的区别是什么",
        "what is a rust crate", "what is the meaning of lifetime elision",
        "define async await in rust", "serde derive 是啥",
        "这个字段代表什么", "yaml front matter 是什么", "为什么叫 mcp server",
        "什么是闭包", "什么是迭代器", "泛型是什么",
        "什么是生命周期", "trait 和 interface 有什么区别",
        "什么是零成本抽象", "什么是模式匹配", "什么是所有权转移",
        "什么是借用", "什么是可变引用", "什么是不可变引用",
        "什么是智能指针", "Box 和 Rc 有什么区别", "Arc 是干什么的",
        "什么是并发", "什么是并行", "什么是异步",
        "Future 是什么", "什么是 Pin", "什么是 Unpin",
        "什么是宏", "什么是过程宏", "什么是声明宏",
        "什么是模块", "什么是 crate", "什么是 workspace",
        "什么是特征对象", "什么是动态分发", "什么是静态分发",
        "什么是 unsafe", "什么是裸指针", "什么是 FFI",
        "什么是内存安全", "什么是数据竞争", "什么是死锁",
        "什么是通道", "什么是互斥锁", "什么是读写锁",
        "什么是条件变量", "什么是原子操作", "什么是内存序",
        "什么是 Cargo", "什么是 Cargo.toml", "什么是 Cargo.lock",
        "什么是 feature flag", "什么是条件编译", "什么是交叉编译",
        "什么是内联", "什么是尾调用优化", "什么是零大小类型",
        "什么是 newtype 模式", "什么是建造者模式", "什么是类型状态模式",
        "what is ownership in rust", "what is borrowing", "what is a lifetime",
        "what is a trait", "what is an associated type", "what is a generic",
        "what is an enum", "what is a struct", "what is an impl block",
        "what is pattern matching", "what is a match expression", "what is if let",
        "what is a closure", "what is a fn trait", "what is a higher ranked trait bound",
        "what is a smart pointer", "what is a box", "what is rc",
        "what is arc", "what is mutex", "what is rwlock",
        "what is a channel", "what is mpsc", "what is oneshot",
        "what is async await", "what is a future", "what is a runtime",
        "what is tokio", "what is async-std", "what is smol",
        "what is a macro", "what is a proc macro", "what is a declarative macro",
        "what is unsafe rust", "what is raw pointer", "what is ffi",
        "what is cargo", "what is a workspace", "what is a feature flag",
        "what is conditional compilation", "what is cross compilation",
        "what is inline", "what is zero cost abstraction", "what is zero sized type",
        "what is newtype pattern", "what is builder pattern", "what is type state pattern",
        "什么是 Deref", "什么是 Drop", "什么是 From 和 Into",
        "什么是 Display 和 Debug", "什么是 Error trait", "什么是 FromStr",
        "什么是 Default trait", "什么是 Clone 和 Copy", "什么是 Hash",
        "什么是 Ord 和 PartialOrd", "什么是 Eq 和 PartialEq",
        "什么是 Send 和 Sync", "什么是 UnwindSafe", "什么是 RefUnwindSafe",
        "解释一下这个函数", "这个模块是干什么的", "这个类型是什么意思",
        "这个参数有什么用", "这个返回值是什么", "这个错误是什么意思",
        "这个警告是什么意思", "这个注解是干什么的", "这个属性是什么",
        "这个 trait bound 是什么意思", "这个 where 子句是什么",
        "这个生命周期标注是什么意思", "这个泛型约束是什么",
        "explain this function", "what does this module do", "what does this type mean",
        "what is this parameter for", "what does this return value mean",
        "what does this error mean", "what does this warning mean",
        "what is this attribute for", "what is this trait bound",
        "什么是 serde", "什么是 rayon", "什么是 crossbeam",
        "什么是 hyper", "什么是 actix", "什么是 axum",
        "什么是 diesel", "什么是 sqlx", "什么是 sea-orm",
        "什么是 tonic", "什么是 prost", "什么是 capnp",
        "什么是 tracing", "什么是 log", "什么是 env_logger",
        "什么是 clap", "什么是 structopt", "什么是 dialoguer",
        "什么是 reqwest", "什么是 tower", "什么是 warp",
        "什么是 tokio-util", "什么是 tokio-stream", "什么是 bytes",
    ],
    "request_action": [
        "帮我审查这段代码", "请帮我重构这个函数", "给我写一个 shell 脚本",
        "帮我修一下这个 panic", "review this patch", "please refactor this function",
        "fix this bug for me", "帮我改一下这个正则", "帮我写个命令行工具",
        "请执行这个命令", "帮我优化这段逻辑", "给我整理一下这个 prompt",
        "帮我写一个单元测试", "请添加错误处理", "帮我实现这个接口",
        "修改这个配置文件", "更新这个依赖版本", "帮我添加日志",
        "写一个解析器", "实现一个缓存机制", "帮我创建一个新模块",
        "添加类型转换", "帮我写一个中间件", "实现一个连接池",
        "帮我修改这个 API", "写一个批处理脚本", "帮我添加认证逻辑",
        "实现文件上传功能", "帮我写一个排序算法", "添加输入验证",
        "帮我实现分页功能", "写一个数据导出工具", "帮我添加重试逻辑",
        "实现一个定时任务", "帮我写一个配置加载器", "添加健康检查接口",
        "帮我实现限流", "写一个消息队列消费者", "帮我添加监控指标",
        "实现一个简单的 ORM", "帮我写一个代码生成器", "添加连接超时处理",
        "帮我实现幂等性", "写一个数据迁移脚本", "帮我添加压缩功能",
        "实现一个简单的模板引擎", "帮我写一个 HTTP 客户端", "添加请求重定向",
        "帮我实现 WebSocket", "写一个简单的代理服务器", "帮我添加 CORS 支持",
        "实现一个简单的负载均衡", "帮我写一个日志收集器", "添加请求限速",
        "帮我实现断路器模式", "写一个简单的服务发现", "帮我添加优雅关闭",
        "实现一个简单的配置中心", "帮我写一个健康检查", "添加请求追踪",
        "帮我实现 API 网关", "写一个简单的消息总线", "帮我添加链路追踪",
        "实现一个简单的分布式锁", "帮我写一个事件溯源", "添加请求去重",
        "帮我实现 CQRS 模式", "写一个简单的 Saga 编排器", "帮我添加补偿事务",
        "实现一个简单的分布式事务", "帮我写一个幂等键管理器", "添加请求熔断",
        "帮我写个 Dockerfile", "帮我配置 CI/CD", "写一个 Makefile",
        "帮我写个 docker-compose", "添加 GitHub Actions", "帮我写个部署脚本",
        "帮我写个迁移脚本", "添加数据库索引", "帮我优化 SQL 查询",
        "写一个数据清洗脚本", "帮我实现数据校验", "添加字段加密",
        "帮我写个导入导出工具", "实现数据脱敏", "帮我添加审计日志",
        "write a unit test", "add error handling", "implement this interface",
        "update this dependency", "add logging to this module", "write a parser",
        "implement a cache mechanism", "create a new module", "add type conversion",
        "write a middleware", "implement a connection pool", "modify this API",
        "write a batch script", "add authentication logic", "implement file upload",
        "write a sorting algorithm", "add input validation", "implement pagination",
        "write a data export tool", "add retry logic", "implement a scheduled task",
        "write a config loader", "add health check endpoint", "implement rate limiting",
        "write a message queue consumer", "add monitoring metrics", "implement idempotency",
        "write a data migration script", "add compression", "implement a template engine",
        "write an HTTP client", "add request redirect", "implement WebSocket",
        "write a proxy server", "add CORS support", "implement load balancing",
        "write a log collector", "add request throttling", "implement circuit breaker",
        "write a service discovery", "add graceful shutdown", "implement config center",
        "write a health check", "add request tracing", "implement API gateway",
        "write a message bus", "add distributed tracing", "implement distributed lock",
        "write a Dockerfile", "configure CI/CD", "write a Makefile",
        "write a docker-compose", "add GitHub Actions", "write a deploy script",
        "write a migration script", "add database index", "optimize SQL query",
        "write a data cleaning script", "implement data validation", "add field encryption",
        "write an import/export tool", "implement data masking", "add audit logging",
        "有什么工具可以调试", "有哪些 skill 可以做 review", "推荐一些好用的工具",
        "what tools are available", "what skills do you have",
        "recommend some tools", "show me what you can do",
        "what are your capabilities", "what can you help with",
        "list available tools", "list available skills",
        "有什么功能", "能做什么", "你会什么",
        "有什么能力", "支持什么", "有哪些功能",
        "你能帮我做什么", "你的功能是什么",
        "帮我找几个 review skill", "帮我找几个调试工具",
        "推荐几个好用的 skill", "有哪些可用的工具",
    ],
    "seek_solution": [
        "怎么处理这个报错", "如何解决 borrow checker error", "why does this panic",
        "how to fix this exception", "这个编译错误怎么修复", "为什么这里会 crash",
        "程序卡住了怎么办", "这个超时问题怎么查", "请求失败的原因是什么",
        "how do i debug this request timeout", "为什么 tool call 会失败",
        "这个问题应该怎么定位", "内存泄漏怎么排查", "CPU 占用过高怎么处理",
        "死锁怎么检测", "如何避免数据竞争", "怎么处理并发问题",
        "如何优雅地处理错误", "怎么避免 unwrap panic", "如何正确使用生命周期",
        "怎么解决循环依赖", "如何处理异步错误", "怎么避免内存溢出",
        "如何优化启动速度", "怎么减少编译时间", "如何处理大文件读取",
        "怎么解决 OOM", "如何处理网络超时", "怎么避免 SQL 注入",
        "如何防止 XSS 攻击", "怎么处理 CSRF", "如何实现安全认证",
        "怎么处理跨域问题", "如何解决乱码问题", "怎么处理时区问题",
        "如何避免浮点精度问题", "怎么处理编码问题", "如何解决依赖冲突",
        "怎么处理版本兼容性", "如何解决环境差异", "怎么排查性能瓶颈",
        "如何定位内存问题", "怎么分析 CPU profile", "如何排查网络问题",
        "怎么处理磁盘 IO 瓶颈", "如何解决连接池耗尽", "怎么处理队列积压",
        "如何避免雪崩效应", "怎么处理级联故障", "如何实现故障恢复",
        "怎么处理数据不一致", "如何解决分布式事务问题", "怎么处理消息丢失",
        "如何避免重复消费", "怎么处理消息顺序问题", "如何实现幂等消费",
        "怎么处理超时重试", "如何避免无限重试", "怎么实现退避策略",
        "如何处理部分失败", "怎么实现最终一致性", "如何处理数据迁移",
        "怎么避免停机迁移", "如何实现灰度发布", "怎么处理回滚",
        "how to debug memory leak", "how to handle high CPU usage",
        "how to detect deadlock", "how to avoid data race",
        "how to handle concurrency issues", "how to handle errors gracefully",
        "how to avoid unwrap panic", "how to use lifetimes correctly",
        "how to resolve circular dependencies", "how to handle async errors",
        "how to avoid memory overflow", "how to optimize startup speed",
        "how to reduce compile time", "how to handle large file reads",
        "how to solve OOM", "how to handle network timeout",
        "how to avoid SQL injection", "how to prevent XSS attacks",
        "how to handle CSRF", "how to implement secure authentication",
        "how to handle CORS issues", "how to solve encoding issues",
        "how to handle timezone issues", "how to avoid floating point precision issues",
        "how to resolve dependency conflicts", "how to handle version compatibility",
        "how to troubleshoot performance bottlenecks", "how to analyze CPU profile",
        "how to troubleshoot network issues", "how to handle disk IO bottlenecks",
        "how to solve connection pool exhaustion", "how to handle queue backlog",
        "how to avoid cascade failures", "how to implement fault recovery",
        "how to handle data inconsistency", "how to solve distributed transaction issues",
        "how to handle message loss", "how to avoid duplicate consumption",
        "how to handle message ordering", "how to implement idempotent consumption",
        "how to handle timeout retries", "how to avoid infinite retries",
        "how to implement backoff strategy", "how to handle partial failures",
        "how to implement eventual consistency", "how to handle data migration",
        "how to implement canary deployment", "how to handle rollback",
        "这个错误怎么解决", "这个异常怎么处理", "这个 bug 怎么修",
        "这个崩溃怎么排查", "这个卡顿怎么优化", "这个延迟怎么降低",
        "这个内存问题怎么解决", "这个并发 bug 怎么修", "这个竞态条件怎么处理",
        "这个死锁怎么解开", "这个性能问题怎么优化", "这个安全漏洞怎么修补",
    ],
    "casual": [
        "hello", "hi", "hey there", "你好", "早上好", "在吗", "谢谢", "thanks",
        "辛苦了", "最近怎么样", "good morning", "下午好",
        "晚上好", "周末好", "好久不见", "怎么样", "还行吧",
        "不错", "好的", "明白了", "了解", "收到", "ok",
        "好的谢谢", "没问题", "可以的", "行", "嗯", "哦",
        "哈哈", "厉害", "牛", "赞", "棒", "酷",
        "再见", "拜拜", "下次见", "回见", "bye", "see you",
        "goodbye", "good night", "晚安", "明天见",
        "你是谁", "你叫什么",
        "讲个笑话", "说个故事", "聊聊天",
        "随便聊聊", "闲聊", "打发时间",
        "无聊", "没事做", "不知道干嘛",
        "今天天气怎么样",
        "最近有什么新闻", "有什么新鲜事",
        "how are you", "what's up", "how's it going",
        "nice to meet you", "glad to see you",
        "long time no see", "how have you been",
        "what's new", "anything interesting",
        "just saying hi", "checking in",
        "thought I'd drop by", "how's everything",
        "what's happening", "what's going on",
        "any updates", "any news",
        "tell me something interesting", "surprise me",
        "entertain me", "amuse me",
        "I'm bored", "got nothing to do",
        "what's trending", "what's popular",
    ],
}

AGENT_ROUTE_SEEDS = {
    "build": [
        "帮我修一下这个 bug", "修改这个函数的返回值", "重构这个模块",
        "添加一个新的接口", "实现这个功能", "修复这个编译错误",
        "改一下这个正则表达式", "帮我写个命令行工具", "优化这段逻辑",
        "debug 这个 panic", "fix this bug", "refactor this function",
        "implement this feature", "add a new endpoint", "update the config file",
        "修改配置文件", "帮我改一下这个文件", "把这个函数改成异步的",
        "给这个结构体加个字段", "修一下这个测试", "帮我实现这个接口",
        "修复这个 panic", "build 这个项目", "coding 一个新的模块",
        "执行这个命令", "这个文件有几行", "看一下这个文件的内容",
        "这个函数是干什么的", "这个变量在哪里被使用", "帮我看看这个报错",
        "这个文件第 50 行是什么", "这个问题为什么会这样", "为什么不行",
        "这个怎么用", "这是什么意思", "帮我看看", "这个报错是什么意思",
        "这个模块怎么工作的", "这个类是干啥的", "这段代码什么意思",
        "这个函数返回什么", "这个配置是干什么的", "这个参数有什么用",
        "这个文件在哪里", "这个依赖是做什么的", "这个接口怎么调用",
        "这个类型是什么", "这个宏是干什么的", "帮我看看这个",
        "看一下", "这个为什么报错", "这个怎么解决", "这个怎么修",
        "帮我处理一下", "这个需要改吗", "这个能不能这样改",
        "这个文件改一下", "这个函数需要优化", "帮我检查一下",
        "这个逻辑对不对", "这个写法有没有问题", "what does this do",
        "how does this work", "why is this happening", "can you check this",
        "look at this error", "what is this for", "help me understand this",
        "is this correct", "show me this file", "where is this defined",
        "how to use this", "这个怎么跑", "跑一下测试", "看一下日志",
        "你好", "hello", "谢谢", "早上好", "在吗", "hi", "hey",
        "@/path/to/file.rs 这个文件有几行", "@/src/main.rs 看一下这个文件",
        "@/Users/test/project/src/lib.rs 这个文件", "@config.yaml 这个配置",
        "这个文件有多少行", "看一下这个路径下的文件", "列出这个目录",
        "这个项目结构是怎样的", "这个仓库有多少文件", "git log 看一下",
        "git status", "最近改了什么", "这个分支有什么变化", "diff 看一下",
        "帮我写个 Dockerfile", "帮我配置 CI/CD", "写一个 Makefile",
        "帮我写个 docker-compose", "添加 GitHub Actions", "帮我写个部署脚本",
        "帮我写个迁移脚本", "添加数据库索引", "帮我优化 SQL 查询",
        "写一个数据清洗脚本", "帮我实现数据校验", "添加字段加密",
        "帮我写个导入导出工具", "实现数据脱敏", "帮我添加审计日志",
        "write a unit test", "add error handling", "implement this interface",
        "update this dependency", "add logging to this module", "write a parser",
        "implement a cache mechanism", "create a new module", "add type conversion",
        "write a middleware", "implement a connection pool", "modify this API",
        "write a batch script", "add authentication logic", "implement file upload",
        "write a sorting algorithm", "add input validation", "implement pagination",
        "write a data export tool", "add retry logic", "implement a scheduled task",
        "write a config loader", "add health check endpoint", "implement rate limiting",
        "write a message queue consumer", "add monitoring metrics", "implement idempotency",
        "write a data migration script", "add compression", "implement a template engine",
        "write an HTTP client", "add request redirect", "implement WebSocket",
        "write a proxy server", "add CORS support", "implement load balancing",
        "write a log collector", "add request throttling", "implement circuit breaker",
        "write a service discovery", "add graceful shutdown", "implement config center",
        "write a health check", "add request tracing", "implement API gateway",
        "write a message bus", "add distributed tracing", "implement distributed lock",
               "这个方法有 bug", "修一下这个逻辑", "帮我改改这段代码",
        "这个函数太慢了", "帮我优化性能", "这段代码有内存泄漏",
        "帮我修复安全漏洞", "这个接口返回了错误", "帮我处理异常",
        "添加输入校验", "帮我写个工具函数", "实现一个 helper 方法",
        "帮我加个注释", "写个文档", "帮我整理一下代码",
        "这个文件太乱了", "帮我拆分这个模块", "提取公共逻辑",
        "帮我消除重复代码", "这个函数太长了", "帮我拆分函数",
        "这个类太大了", "帮我拆分类", "提取接口",
        "帮我实现适配器", "写个装饰器", "帮我实现代理模式",
        "这个配置不对", "帮我修改配置", "更新环境变量",
        "这个依赖有冲突", "帮我解决依赖问题", "升级这个包",
        "这个测试挂了", "帮我修测试", "添加集成测试",
        "帮我写个 benchmark", "添加性能测试", "写个压力测试",
        "这个脚本跑不了", "帮我修脚本", "这个命令不工作",
        "帮我调试一下", "这个进程挂了", "帮我重启服务",
        "这个服务起不来", "帮我排查启动问题", "这个端口被占用了",
        "帮我杀掉这个进程", "这个文件权限不对", "帮我修改权限",
    ],
    "executor": [
        "帮我实现这个功能，然后跑测试并修掉所有报错", "自动修复所有编译错误",
        "端到端完成这个任务", "自动执行这个流程", "完整实现这个需求",
        "帮我一步步完成这个重构", "自动修复所有 lint 错误然后跑测试",
        "implement this end-to-end and fix all errors", "autonomously complete this task",
        "run the full pipeline and repair failures", "fix all the test failures automatically",
        "然后跑检查并修掉相关报错", "分步骤完成这个迁移", "自动排查并修复这个问题",
        "闭环完成这个需求", "帮我实现并验证", "自动执行所有步骤",
        "同时修改多个文件", "顺便把那个也改了",
        "implement and run tests to verify", "repair all build errors",
        "debug and fix this end to end", "run tests and fix failures",
        "帮我完成整个功能的开发和测试", "自动实现并验证这个需求",
        "一步步完成这个需求的实现和测试", "完整实现这个功能包括测试",
        "自动修复所有问题并确保通过", "帮我从头到尾完成这个任务",
        "实现这个功能并修复所有报错", "自动完成代码修改和验证",
        "帮我完成这个重构并确保测试通过", "自动修复 lint 和编译错误",
        "完整实现这个接口包括错误处理", "帮我实现并跑通所有测试",
        "自动完成这个迁移并验证功能正确", "帮我完成这个需求的端到端实现",
        "实现这个功能并处理所有边界情况", "自动修复并验证所有测试用例",
        "帮我完成这个模块的开发和集成测试", "自动实现并确保代码质量",
        "完整完成这个功能的开发和部署", "帮我实现并优化性能",
        "自动修复所有警告并确保零警告", "帮我完成这个 API 的实现和文档",
        "实现这个功能并添加完整的测试覆盖", "自动完成重构并确保行为不变",
        "帮我实现这个需求并处理兼容性", "自动完成所有修改并验证",
        "implement the full feature with tests", "automatically fix all issues and verify",
        "complete the entire task end to end", "implement and validate the full solution",
        "fix all errors and run the test suite", "implement this with full test coverage",
        "automatically resolve all compilation errors", "complete the migration and verify",
        "implement and deploy this feature", "fix all lint warnings and ensure clean build",
        "implement this with error handling and tests", "automatically complete all changes",
        "refactor and ensure all tests still pass", "implement with backward compatibility",
        "complete the implementation with documentation", "fix and verify all test cases",
        "implement with edge case handling", "automatically fix and validate everything",
        "然后验证一下", "跑完测试再提交", "修完所有错误再继续",
        "自动完成所有步骤并验证", "帮我实现并确保质量", "自动修复并跑通",
        "完整实现并测试", "帮我做完整个流程", "自动完成端到端",
        "实现并验证所有功能", "修复并确认通过", "完成并检查",
        "实现这个需求然后验证", "帮我做完并确认没问题",
        "自动修复并确保通过所有测试", "完整实现这个功能包括测试和文档",
        "帮我完成这个任务的所有步骤", "自动实现并处理所有错误",
        "实现这个功能并确保代码质量", "帮我完成并验证结果",
        "自动修复所有问题并确认", "完整实现并确保正确性",
    ],
    "plan": [
        "帮我规划一下这个项目的架构", "分析一下这个模块的设计",
        "review 这段代码", "总结一下这个文件的功能", "帮我做个方案",
        "规划一下重构步骤", "分析一下性能瓶颈", "review this code",
        "analyze the architecture", "summarize this module", "plan the migration steps",
        "帮我分析一下这个方案", "梳理一下这个模块的依赖关系",
        "review 一下这个 PR", "总结一下这次改动", "分析一下这个问题的根因",
        "规划一下接下来的开发计划", "这个架构有什么问题",
        "帮我看看这个设计合不合理", "分析一下这段代码的质量",
        "planning the next sprint", "architecture review for this service",
        "帮我规划一下架构方案", "帮我做个技术方案", "帮我分析一下架构",
        "帮我梳理一下模块关系", "帮我总结一下这个项目", "帮我 review 一下代码",
        "帮我规划重构方案", "帮我分析性能瓶颈", "规划一下这个需求",
        "做个方案出来", "分析一下可行性", "总结一下这个模块",
        "帮我看看架构合不合理", "帮我分析一下这个设计",
        "帮我规划一下开发计划", "帮我 review 这个 PR", "帮我总结一下改动",
        "帮我分析根因", "分析一下这个系统的瓶颈", "规划一下微服务拆分",
        "帮我评估一下技术选型", "梳理一下数据流", "分析一下调用链路",
        "规划一下数据库设计", "帮我设计一下 API", "分析一下安全性",
        "评估一下性能影响", "规划一下灰度方案", "帮我设计一下监控方案",
        "分析一下成本", "规划一下容量", "帮我设计一下容灾方案",
        "分析一下可用性", "规划一下扩展方案", "帮我设计一下缓存策略",
        "分析一下一致性", "规划一下数据同步方案", "帮我设计一下消息方案",
        "analyze this system's bottleneck", "plan the microservice decomposition",
        "evaluate technology choices", "map out the data flow",
        "analyze the call chain", "plan the database design",
        "design the API", "analyze the security", "evaluate performance impact",
        "plan the canary deployment", "design the monitoring strategy",
        "analyze the cost", "plan the capacity", "design the disaster recovery",
        "analyze the availability", "plan the scaling strategy",
        "design the caching strategy", "analyze the consistency",
        "plan the data sync strategy", "design the messaging solution",
        "review the codebase structure", "analyze the dependency graph",
        "plan the refactoring roadmap", "evaluate the architecture patterns",
        "design the deployment pipeline", "analyze the error handling strategy",
        "plan the testing strategy", "design the logging strategy",
        "analyze the resource utilization", "plan the optimization roadmap",
        "design the retry strategy", "analyze the failure modes",
        "plan the rollback strategy", "design the circuit breaker pattern",
        "analyze the scalability", "plan the observability strategy",
        "design the rate limiting approach", "analyze the throughput",
        "plan the data migration strategy", "design the idempotency approach",
        "这个设计有什么缺陷", "帮我分析一下风险", "规划一下迭代计划",
        "帮我评估一下工作量", "分析一下技术债务", "规划一下重构优先级",
        "帮我设计一下错误处理方案", "分析一下这个方案的优缺点",
        "规划一下测试策略", "帮我设计一下日志方案",
        "分析一下资源使用情况", "规划一下优化路线",
        "帮我设计一下重试策略", "分析一下故障模式",
        "规划一下回滚方案", "帮我设计一下熔断方案",
        "分析一下可扩展性", "规划一下可观测性方案",
        "帮我设计一下限流方案", "分析一下吞吐量",
    ],
    "prompt-skill": [
        "帮我优化这个 prompt", "生成一个新的 skill", "优化这个提示词",
        "帮我写一个 skill", "设计一个 prompt 模板", "optimize this prompt",
        "generate a new skill", "engineer a better prompt for this task",
        "帮我改进这个 skill 的描述", "提示词工程优化", "技能生成和优化",
        "帮我设计一个提示词", "优化一下这个 agent 的 prompt", "创建一个新的提示词模板",
        "帮我写个 skill 文件", "帮我优化这个 agent 的指令",
        "改进这个 skill 的触发条件", "帮我设计一个更好的 prompt 策略",
        "优化这个 skill 的工具配置", "帮我写一个专业的提示词",
        "帮我生成一个代码审查的 skill", "优化这个 prompt 的结构",
        "帮我设计一个多步骤的提示词", "改进这个 skill 的输出格式",
        "帮我写一个调试用的 skill", "优化这个 agent 的系统提示词",
        "帮我创建一个自动化 skill", "设计一个 prompt chain",
        "帮我优化这个 skill 的优先级", "改进这个提示词的清晰度",
        "帮我写一个重构 skill", "优化这个 prompt 的指令部分",
        "帮我设计一个代码生成的 skill", "改进这个 skill 的描述和标签",
        "帮我写一个测试生成的提示词", "优化这个 agent 的行为指令",
        "帮我创建一个文档生成的 skill", "设计一个 prompt 模板用于代码审查",
        "帮我优化这个 skill 的路由标签", "改进这个提示词的示例部分",
        "帮我写一个部署 skill", "优化这个 prompt 的约束条件",
        "帮我设计一个监控 skill", "改进这个 skill 的工具权限",
        "帮我写一个安全审计的提示词", "优化这个 agent 的角色定义",
        "帮我创建一个性能优化的 skill", "设计一个 prompt 用于错误分析",
        "帮我优化这个 skill 的执行步骤", "改进这个提示词的输出规范",
        "optimize the prompt structure", "improve the skill description",
        "design a prompt template for code review", "create a debugging skill",
        "optimize the agent's system prompt", "create an automation skill",
        "design a prompt chain", "optimize the skill priority",
        "improve the prompt clarity", "write a refactoring skill",
        "optimize the prompt instructions", "design a code generation skill",
        "improve the skill tags and description", "write a test generation prompt",
        "optimize the agent behavior instructions", "create a documentation skill",
        "design a prompt for code review", "optimize the skill routing tags",
        "improve the prompt examples", "write a deployment skill",
        "optimize the prompt constraints", "design a monitoring skill",
        "improve the skill tool permissions", "write a security audit prompt",
        "optimize the agent role definition", "create a performance optimization skill",
        "design a prompt for error analysis", "optimize the skill execution steps",
        "improve the prompt output specification", "write a prompt engineering guide",
        "帮我写一个提示词模板", "优化这个 skill 的参数配置",
        "帮我设计一个多 agent 协作的 skill", "改进这个提示词的上下文管理",
        "帮我写一个代码风格检查的 skill", "优化这个 prompt 的 few-shot 示例",
        "帮我创建一个 API 设计的 skill", "设计一个 prompt 用于需求分析",
        "帮我优化这个 skill 的错误处理指令", "改进这个提示词的格式要求",
        "帮我写一个性能测试的 skill", "优化这个 agent 的工具使用指令",
        "帮我设计一个代码迁移的 skill", "改进这个 skill 的触发条件",
        "帮我写一个安全扫描的提示词", "优化这个 prompt 的角色设定",
        "帮我创建一个日志分析的 skill", "设计一个 prompt 用于架构评审",
        "帮我优化这个 skill 的输出模板", "改进这个提示词的约束说明",
    ],
}


def augment_intent_query_concept(seeds: list) -> list:
    prefixes = [
        "什么是", "解释一下", "请解释", "帮我理解",
        "what is", "what are", "explain",
    ]
    subjects = [
        "所有权", "借用", "生命周期", "trait", "泛型", "闭包", "迭代器",
        "智能指针", "模式匹配", "零成本抽象", "并发", "异步", "宏",
        "模块", "crate", "workspace", "特征对象", "动态分发",
        "unsafe", "裸指针", "FFI", "内存安全", "数据竞争",
        "通道", "互斥锁", "读写锁", "原子操作", "内存序",
        "Cargo", "feature flag", "条件编译", "交叉编译",
        "内联", "零大小类型", "newtype 模式", "建造者模式",
        "Deref", "Drop", "From 和 Into", "Display 和 Debug",
        "Error trait", "FromStr", "Default trait", "Clone 和 Copy",
        "Send 和 Sync", "Hash", "Ord 和 PartialOrd",
        "serde", "rayon", "crossbeam", "hyper", "actix", "axum",
        "diesel", "sqlx", "tonic", "tracing", "clap", "reqwest",
        "tokio", "async-std", "Future", "Pin", "Unpin",
        "过程宏", "声明宏", "关联类型", "泛型约束",
    ]
    suffixes = ["", "？", "是什么意思", "是干什么的"]
    result = list(seeds)
    for prefix in prefixes:
        for subject in subjects:
            for suffix in suffixes[:2]:
                result.append(f"{prefix}{subject}{suffix}")
    return result


def augment_intent_request_action(seeds: list) -> list:
    prefixes = ["帮我", "请帮我", "帮我来", "请"]
    verbs = [
        "写", "实现", "添加", "修改", "修复", "重构", "优化", "创建",
        "更新", "删除", "替换", "调整", "完善", "改进", "扩展",
        "write", "implement", "add", "modify", "fix", "refactor",
        "optimize", "create", "update", "delete", "replace", "adjust",
    ]
    objects = [
        "一个单元测试", "错误处理", "这个接口", "日志",
        "一个解析器", "缓存机制", "新模块", "类型转换",
        "中间件", "连接池", "这个 API", "批处理脚本",
        "认证逻辑", "文件上传", "排序算法", "输入验证",
        "分页功能", "数据导出", "重试逻辑", "定时任务",
        "配置加载器", "健康检查", "限流", "消息队列消费者",
        "监控指标", "幂等性", "数据迁移脚本", "压缩功能",
        "模板引擎", "HTTP 客户端", "请求重定向", "WebSocket",
        "代理服务器", "CORS 支持", "负载均衡", "日志收集器",
        "请求限速", "断路器", "服务发现", "优雅关闭",
        "配置中心", "请求追踪", "API 网关", "消息总线",
        "链路追踪", "分布式锁", "事件溯源", "补偿事务",
        "Dockerfile", "CI/CD", "Makefile", "docker-compose",
        "部署脚本", "数据库索引", "SQL 查询", "数据清洗",
        "数据校验", "字段加密", "导入导出", "数据脱敏",
        "审计日志", "a unit test", "error handling", "this interface",
        "logging", "a parser", "cache mechanism", "new module",
        "type conversion", "middleware", "connection pool", "this API",
        "batch script", "authentication", "file upload", "sorting algorithm",
        "input validation", "pagination", "data export", "retry logic",
        "scheduled task", "config loader", "health check", "rate limiting",
    ]
    result = list(seeds)
    for prefix in prefixes:
        for verb in verbs[:8]:
            for obj in objects[:30]:
                result.append(f"{prefix}{verb}{obj}")
    return result


def augment_intent_seek_solution(seeds: list) -> list:
    prefixes = ["怎么", "如何", "为什么", "怎样", "怎么才能", "如何才能"]
    verbs = ["处理", "解决", "修复", "排查", "避免", "定位", "调试", "消除"]
    problems = [
        "这个报错", "这个错误", "这个异常", "这个 bug", "这个崩溃",
        "内存泄漏", "CPU 过高", "死锁", "数据竞争", "编译错误",
        "超时问题", "请求失败", "连接池耗尽", "队列积压",
        "this error", "this bug", "memory leak", "deadlock",
        "compilation error", "timeout", "request failure",
    ]
    result = list(seeds)
    for prefix in prefixes:
        for verb in verbs:
            for problem in problems:
                result.append(f"{prefix}{verb}{problem}")
    return result


def augment_intent_casual(seeds: list) -> list:
    greetings_cn = [
        "你好", "早上好", "下午好", "晚上好", "周末好", "好久不见",
        "在吗", "怎么样", "还行吧", "不错", "好的", "明白了",
        "了解", "收到", "好的谢谢", "没问题", "可以的", "行",
        "嗯", "哦", "哈哈", "厉害", "牛", "赞", "棒", "酷",
        "再见", "拜拜", "下次见", "回见", "晚安", "明天见",
        "谢谢", "辛苦了", "最近怎么样", "你好呀", "嗨", "嘿",
        "早上好啊", "下午好啊", "晚上好啊", "你好啊",
    ]
    greetings_en = [
        "hello", "hi", "hey", "hey there", "good morning", "good afternoon",
        "good evening", "good night", "bye", "see you", "goodbye",
        "thanks", "thank you", "ok", "okay", "sure", "nice",
        "cool", "great", "awesome", "got it", "understood",
        "how are you", "what's up", "how's it going",
        "nice to meet you", "long time no see", "how have you been",
        "good to see you", "take care", "have a good day",
        "no worries", "my pleasure", "you're welcome",
        "what's happening", "what's going on", "any news",
        "hey man", "yo", "sup", "greetings",
    ]
    result = list(seeds)
    for g in greetings_cn:
        for suffix in ["", "！", "。", "呀", "啊", "呢", "哦", "嘛"]:
            result.append(f"{g}{suffix}")
    for g in greetings_en:
        for suffix in ["", "!", ".", " :)", " :D", " ;)"]:
            result.append(f"{g}{suffix}")
    chat_phrases = [
        "讲个笑话", "说个故事", "聊聊天", "随便聊聊", "闲聊",
        "打发时间", "无聊", "没事做", "不知道干嘛",
        "今天天气怎么样", "最近有什么新闻", "有什么新鲜事",
        "你是谁", "你叫什么", "介绍一下你自己",
        "tell me a joke", "tell me a story", "let's chat",
        "I'm bored", "got nothing to do", "what's trending",
        "surprise me", "entertain me", "amuse me",
        "who are you", "what's your name", "introduce yourself",
        "早上好啊今天", "今天心情不错", "天气真好",
        "周末愉快", "假期快乐", "新年快乐",
        "节日快乐", "生日快乐", "恭喜",
        "加油", "继续", "努力", "坚持",
        "慢慢来", "别着急", "放心",
        "可以的", "没问题", "一定行",
        "真棒", "太好了", "不错不错",
        "挺好的", "还行", "一般般",
        "就这样吧", "先这样", "暂时这样",
        "好的呢", "知道了", "明白", "了解啦",
    ]
    for phrase in chat_phrases:
        for suffix in ["", "？", "!", "呢", "吧", "啊"]:
            result.append(f"{phrase}{suffix}")
    return result


def augment_executor(seeds: list) -> list:
    prefixes = [
        "帮我", "请", "自动", "完整", "端到端", "一步步", "分步骤",
        "闭环", "从头到尾", "全流程", "全自动",
    ]
    suffixes = [
        "并验证", "并修掉报错", "并跑测试", "并确保通过", "并确认正确",
        "然后验证", "然后跑测试", "然后修掉报错", "然后确保通过",
        "然后检查", "并修复所有问题", "并确认功能正确",
    ]
    actions = [
        "实现这个功能", "完成这个任务", "修复所有错误", "完成这个需求",
        "实现这个接口", "完成这个重构", "修复所有编译错误", "完成这个迁移",
        "实现这个模块", "完成这个开发", "修复所有测试失败", "完成这个集成",
        "实现这个特性", "完成这个优化", "修复所有警告", "完成这个部署",
        "实现这个服务", "完成这个改造", "修复所有 lint", "完成这个升级",
    ]
    result = list(seeds)
    for prefix in prefixes:
        for action in actions:
            result.append(f"{prefix}{action}")
    for action in actions:
        for suffix in suffixes:
            result.append(f"{action}{suffix}")
    for prefix in prefixes[:7]:
        for action in actions[:12]:
            for suffix in suffixes[:6]:
                result.append(f"{prefix}{action}{suffix}")
    return result


def augment_plan(seeds: list) -> list:
    prefixes = ["帮我", "请帮我", "帮我来", "一起来"]
    verbs = ["分析", "规划", "梳理", "评估", "设计", "总结", "review", "审查"]
    objects = [
        "这个架构", "这个模块", "这个系统", "这个方案", "这个设计",
        "这个代码", "这个 PR", "这个需求", "这个接口", "这个流程",
        "性能瓶颈", "依赖关系", "数据流", "调用链路", "安全性",
        "可行性", "技术选型", "重构方案", "迁移方案", "测试策略",
        "这个服务", "这个组件", "这个配置", "这个实现",
        "扩展性", "可用性", "可靠性", "可维护性",
        "监控方案", "容灾方案", "灰度方案", "缓存策略",
    ]
    result = list(seeds)
    for prefix in prefixes:
        for verb in verbs:
            for obj in objects:
                result.append(f"{prefix}{verb}{obj}")
    return result


def augment_prompt_skill(seeds: list) -> list:
    prefixes = [
        "帮我", "请帮我", "帮我设计", "帮我创建", "帮我写",
        "帮我生成", "帮我改进", "帮我优化", "帮我构建", "帮我制作",
    ]
    targets = [
        "一个代码审查的 skill", "一个调试的 skill", "一个部署的 skill",
        "一个测试的 skill", "一个文档的 skill", "一个监控的 skill",
        "一个安全审计的 skill", "一个性能优化的 skill", "一个日志分析的 skill",
        "一个代码生成的 skill", "一个重构的 skill", "一个迁移的 skill",
        "一个 prompt 模板", "一个提示词", "一个 skill 文件",
        "一个 API 设计的 skill", "一个数据库的 skill", "一个前端的 skill",
        "一个后端的 skill", "一个运维的 skill", "一个安全扫描的 skill",
        "一个代码风格检查的 skill", "一个性能测试的 skill", "一个集成测试的 skill",
        "一个代码迁移的 skill", "一个日志收集的 skill", "一个配置管理的 skill",
        "一个需求分析的 skill", "一个架构评审的 skill", "一个代码搜索的 skill",
        "一个代码格式化的 skill", "一个依赖检查的 skill", "一个版本管理的 skill",
        "一个环境配置的 skill", "一个数据备份的 skill", "一个服务编排的 skill",
        "一个代码审查的提示词", "一个调试的提示词", "一个部署的提示词",
        "一个测试的提示词", "一个文档的提示词", "一个监控的提示词",
        "一个 prompt", "一个提示词模板", "一个指令模板",
        "一个代码补全的 skill", "一个代码解释的 skill", "一个代码摘要的 skill",
        "一个 bug 分析的 skill", "一个代码评审的提示词", "一个自动修复的 skill",
        "一个代码审查的 prompt", "一个调试的 prompt", "一个部署的 prompt",
        "一个测试的 prompt", "一个文档的 prompt", "一个监控的 prompt",
        "一个安全审计的 prompt", "一个性能优化的 prompt", "一个日志分析的 prompt",
        "一个重构的 prompt", "一个迁移的 prompt", "一个代码生成的 prompt",
        "一个 prompt chain", "一个多步骤 prompt", "一个条件 prompt",
        "一个递归 prompt", "一个模板化 prompt", "一个参数化 prompt",
        "一个对话式 prompt", "一个指令式 prompt", "一个角色扮演 prompt",
        "一个 few-shot prompt", "一个 zero-shot prompt", "一个 CoT prompt",
        "一个思维链 prompt", "一个自省 prompt", "一个反思 prompt",
        "一个自我纠错 prompt", "一个多轮对话 prompt", "一个上下文学习 prompt",
        "一个元 prompt", "一个系统 prompt", "一个用户 prompt",
    ]
    actions = [
        "优化这个 prompt", "改进这个提示词", "优化这个 skill",
        "改进这个 skill 的描述", "优化这个 agent 的指令",
        "改进这个 skill 的触发条件", "优化这个 skill 的工具配置",
        "改进这个 skill 的参数", "优化这个提示词的结构",
        "改进这个 skill 的输出格式", "优化这个 agent 的角色定义",
        "改进这个 skill 的标签", "优化这个提示词的示例",
        "改进这个 skill 的路由", "优化这个提示词的指令",
    ]
    result = list(seeds)
    for prefix in prefixes:
        for target in targets:
            result.append(f"{prefix}{target}")
    for action in actions:
        for suffix in ["的结构", "的清晰度", "的格式", "的约束", "的示例", "的规范"]:
            result.append(f"{action}{suffix}")
    return result


def augment_build(seeds: list) -> list:
    prefixes = ["帮我", "请帮我", "帮我来", "请"]
    verbs = ["修", "改", "修一下", "改一下", "看一下", "检查一下", "处理一下", "调试一下"]
    objects = [
        "这个 bug", "这个函数", "这个文件", "这个模块", "这个接口",
        "这个配置", "这个测试", "这个错误", "这个逻辑", "这个类",
        "这个方法", "这个参数", "这个返回值", "这个类型", "这个字段",
        "这个依赖", "这个脚本", "这个服务", "这个组件", "这个工具",
        "这个正则", "这个查询", "这个路由", "这个中间件", "这个缓存",
    ]
    result = list(seeds)
    for prefix in prefixes:
        for verb in verbs:
            for obj in objects:
                result.append(f"{prefix}{verb}{obj}")
    questions = [
        "这个文件有几行", "看一下这个文件", "这个函数是干什么的",
        "这个变量在哪里被使用", "帮我看看这个报错", "这个为什么报错",
        "这个怎么用", "这是什么意思", "帮我看看", "看一下",
        "这个怎么跑", "跑一下测试", "看一下日志",
        "这个文件有多少行", "列出这个目录", "这个项目结构是怎样的",
        "git log 看一下", "git status", "最近改了什么",
    ]
    for q in questions:
        for suffix in ["", "？", "呢", "啊", "吧"]:
            result.append(f"{q}{suffix}")
    actions = [
        "写个", "实现一个", "添加一个", "创建一个", "写一个",
    ]
    targets = [
        "单元测试", "集成测试", "工具函数", "helper 方法",
        "Dockerfile", "Makefile", "部署脚本", "迁移脚本",
        "配置文件", "环境变量", "数据校验", "输入验证",
        "错误处理", "重试逻辑", "超时处理", "日志记录",
    ]
    for action in actions:
        for target in targets:
            result.append(f"{action}{target}")
    return result


def build_corpus(seeds: dict, label_key: str, version: int, labels: list,
                 feature_config: dict, extra_fields: dict = None) -> dict:
    samples = []
    for label, texts in seeds.items():
        for text in texts:
            sample = {"text": text, label_key: label}
            samples.append(sample)
    return _build_corpus_from_samples(samples, version, labels, feature_config, extra_fields)


def build_augmented_corpus(seeds: dict, label_key: str, version: int, labels: list,
                           feature_config: dict, extra_fields: dict = None,
                           augment_fns: dict = None) -> dict:
    augmented = dict(seeds)
    if augment_fns:
        for label, fn in augment_fns.items():
            if label in augmented:
                augmented[label] = fn(augmented[label])
    samples = []
    for label, texts in augmented.items():
        for text in texts:
            sample = {"text": text, label_key: label}
            samples.append(sample)
    return _build_corpus_from_samples(samples, version, labels, feature_config, extra_fields)


def _build_corpus_from_samples(samples, version, labels, feature_config, extra_fields):
    corpus = {
        "version": version,
        "labels": labels,
        "feature_config": feature_config,
    }
    if extra_fields:
        corpus.update(extra_fields)
    corpus["samples"] = samples
    return corpus


def main():
    base = Path(__file__).resolve().parent.parent / "src" / "bin" / "ai" / "config"

    intent_corpus = build_augmented_corpus(
        INTENT_SEEDS,
        label_key="core",
        version=1,
        labels=["query_concept", "request_action", "seek_solution", "casual"],
        feature_config={"char_ngram_min": 2, "char_ngram_max": 4, "max_features": 768},
        extra_fields={
            "runtime_rules": {
                "search_patterns": [
                    "找几个", "找一些", "找些", "收集", "有什么", "有哪些",
                    "推荐几个", "推荐一些", "搜索", "查找",
                ],
                "negation_patterns": [
                    "不", "别", "不要", "无需", "不需要", "not", "don't", "no",
                ],
                "resource_keywords": [
                    {"pattern": "技能", "resource": "skill"},
                    {"pattern": "skill", "resource": "skill"},
                    {"pattern": "skills", "resource": "skill"},
                    {"pattern": "工具", "resource": "tool"},
                    {"pattern": "tool", "resource": "tool"},
                    {"pattern": "tools", "resource": "tool"},
                    {"pattern": "文档", "resource": "doc"},
                    {"pattern": "doc", "resource": "doc"},
                    {"pattern": "docs", "resource": "doc"},
                    {"pattern": "文件", "resource": "file"},
                    {"pattern": "file", "resource": "file"},
                ],
            },
        },
        augment_fns={
            "query_concept": augment_intent_query_concept,
            "request_action": augment_intent_request_action,
            "seek_solution": augment_intent_seek_solution,
            "casual": augment_intent_casual,
        },
    )

    agent_route_corpus = build_augmented_corpus(
        AGENT_ROUTE_SEEDS,
        label_key="agent",
        version=1,
        labels=["build", "executor", "plan", "prompt-skill"],
        feature_config={"char_ngram_min": 2, "char_ngram_max": 4, "max_features": 768},
        augment_fns={
            "build": augment_build,
            "executor": augment_executor,
            "plan": augment_plan,
            "prompt-skill": augment_prompt_skill,
        },
    )

    intent_path = base / "intent" / "training_corpus.json"
    agent_route_path = base / "agent_route" / "training_corpus.json"

    intent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_route_path.parent.mkdir(parents=True, exist_ok=True)

    intent_path.write_text(
        json.dumps(intent_corpus, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    agent_route_path.write_text(
        json.dumps(agent_route_corpus, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    intent_count = len(intent_corpus["samples"])
    agent_count = len(agent_route_corpus["samples"])
    print(f"[intent] {intent_count} samples -> {intent_path}")
    print(f"[agent_route] {agent_count} samples -> {agent_route_path}")

    for label in intent_corpus["labels"]:
        count = sum(1 for s in intent_corpus["samples"] if s["core"] == label)
        print(f"  intent/{label}: {count}")
    for label in agent_route_corpus["labels"]:
        count = sum(1 for s in agent_route_corpus["samples"] if s["agent"] == label)
        print(f"  agent_route/{label}: {count}")


if __name__ == "__main__":
    main()
