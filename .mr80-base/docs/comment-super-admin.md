# 评论指令超级管理员

名单直接保存在 `src/conciliator/config.py` 的 `COMMENT_SUPER_ADMIN_EMAILS` 中：

```python
COMMENT_SUPER_ADMIN_EMAILS: Final[tuple[str, ...]] = (
    "cuijiawei.2026@bytedance.com",
    "huaguojin@bytedance.com",
)
```

服务启动时使用该名单，无需配置环境变量或执行数据库迁移。增加或移除超管时修改这个元组，提交代码并部署新版本。

Comment webhook 优先读取 `Comment.CreatedBy.Email`，缺失时使用 `Sender.Email`。服务用这个邮箱和白名单逐项匹配，忽略大小写与首尾空格；空邮箱不匹配。不使用用户名、用户 ID、MR 作者或评论正文里的邮箱授予超管权限。

超级管理员可执行 `/resolve cancel`、`/resolve approve all`、`/resolve retry`，也可在冲突块 Discussion 中批准或拒绝。取消可从原 MR 或解决冲突的 MR 发起；原 MR 有多个活动任务时，使用 `/resolve cancel <任务 ID>` 指定任务。审批、重解和取消的审计记录仍保存实际操作人的邮箱。

超管只放开身份限制。任务状态、MR 与 Discussion 的关联、重解条件以及并发版本检查仍然生效。Webhook 验签和机器人自身评论过滤保持原有行为。`/resolve` 与 `/resolve status` 原本没有上述身份限制。

## 部署后验证

当前已配置 `cuijiawei.2026@bytedance.com` 和 `huaguojin@bytedance.com`，部署后可用其中任一账号在原 MR 上取消另一个人发起的非终态任务。确认任务变成 `CANCELLED`，取消记录中的邮箱是测试者，再发起 `/resolve`。新任务应读完 Coco 事件分页，发布 MR 后继续创建 Discussion，进入等待审批；不再因为 merge blocked 判定发布失败。

分页修复上线后，已有 `COCO_RUNNING` 任务可能被后台轮询自动推进。执行取消前先确认任务当前状态；已进入终态的任务不能再次取消。
