# 原 MR 的任务状态查询

本方案基于 `fix/resolve-mr-publication` 分支的 `0483b3f`，按确认后的方案在 `feat/original-mr-status` 分支实现。无需数据库迁移；线上评论验证需部署后执行。

## 预期行为

用户在原 MR 评论 `/resolve status`，Conciliator 在这条评论所在的讨论串中回复关联任务的当前状态。用户在 resolve MR 使用同一指令时，继续返回该任务的冲突块审批情况。

状态取自数据库在处理查询时的快照，不等待 Coco，不触发执行、取消、重解或状态迁移。查询沿用现有权限：能在 MR 评论的用户均可查询，不要求是任务发起人或超级管理员。

## 代码现状

- `api/webhooks.py` 已通过 `GetMergeRequest` 取得评论所属 MR，包含 source 分支；`_normalize_trigger_payload()` 也已保留 `source_branch`。
- `/resolve status` 通过 `CommentRouter.route_approval_command()` 进入 `ApprovalService.handle_comment()`。
- 当前先调用 `find_active_for_resolve_mr()`，按 `repo_id + new_mr_id` 查找非终态任务。原 MR 不匹配时直接返回，没有回复。
- `_reply_status()` 读取冲突块和审批记录，回复任务状态、审批数量及冲突块表格。已结束的 resolve MR 任务也会被前面的非终态过滤挡住。
- 仓储已有 `list_active_for_mr()` 和 `find_latest_for_original_mr()`，后者包含终态任务。表上已有原 MR、resolve MR 的查询索引。

## 区分两类 MR

建议以数据库中的 MR 关联为准：先按 `repo_id + new_mr_id` 查找任务，找到则按 resolve MR 处理；否则按 `repo_id + original_mr_id` 查询原 MR 的任务。所有查询都带仓库 ID，避免不同仓库的 MR 编号相同造成串任务。

可以使用 source 分支辅助核对。当前生成规则是 `fix/<source>-to-<target>-conflict-<YYYYMMDD>-t<task_id>`，对应实现为 `domain/resolve_branch.py`。但 `fix/` 也是普通业务分支常见的前缀，完整格式也可能被人工使用，因此不能仅凭分支名认定 MR 类型或授权访问某个任务。若核对分支，应与任务持久化的完整 `resolve_branch` 比较，不能只匹配前缀或提取末尾任务 ID。

这样无需为状态查询新增 NextCode 请求，也无需把分支信息继续透传到审批服务。

## 原 MR 如何选择任务

同一个原 MR 在 source 或 target 提交变化后，可以同时有多个活动任务。因此不假定一个原 MR 永远只有一个“当前任务”。建议规则如下：

| 查询结果 | 回复内容 |
|---|---|
| 只有一个非终态任务 | 返回该任务的当前状态 |
| 有多个非终态任务 | 在一条回复中列出全部活动任务，按任务 ID 倒序，并展示各自冻结的 source/target SHA |
| 没有非终态任务，但有历史任务 | 返回最新任务，明确写“当前没有进行中的任务，最近一次任务……” |
| 从未创建任务 | 回复“该 MR 暂无解冲突任务，可评论 `/resolve` 发起。” |

查询范围是该原 MR，不局限于触发 `/resolve` 的旧讨论串，也不要求任务冻结的 SHA 等于分支现在的 HEAD。否则，分支有新提交后，仍在处理旧版本的任务反而会查不到。

每个任务展示任务 ID、中文状态和原始状态值、冻结的 source/target 短 SHA、最后更新时间；已经持久化新 MR 链接时附上链接。状态中文文案集中定义，覆盖全部 `TaskStatus`，未知值直接显示原值。`ALL_APPROVED` 写“全部冲突块已批准”，不表示 MR 已合并。

单任务回复示例，以下编号和版本仅作展示：

```text
任务 #123：Coco 处理中（COCO_RUNNING）
处理版本：source a1b2c3d4 → target e5f6a7b8
最后更新：2026-09-16 14:30:00（北京时间）
```

终态示例：

```text
当前没有进行中的任务。
最近一次任务 #123：已取消（CANCELLED）
最后更新：2026-09-16 14:35:00（北京时间）
```

原 MR 回复不展开冲突块审批表；用户可通过 resolve MR 查询审批明细。首版不新增 `/resolve status <任务 ID>` 语法。

## 回复位置与异常

使用本次 `/resolve status` 评论的 `ThreadId`，调用 `reply_discussion()`。回复的仓库和 MR 必须是用户此次评论的位置，不能默认使用任务的 `new_mr_id`，也不能使用旧 `/resolve` 评论的讨论串。

即使没有匹配任务，也在本次查询的讨论串下回复。实现和验收均要求正确的 `ThreadId`；缺失时记录可定位的处理错误，不静默另起顶层评论冒充“评论下方回复”。数据库读取失败不能回复“没有任务”，评论发送失败也不能改变任务状态，沿用现有 webhook 错误记录流程。

## 实现改动

1. 在 `ApprovalService.handle_comment()` 中将 `Status` 分支移到通用的 `find_active_for_resolve_mr()` 查询之前，交给新增的 `_route_status()`。其他指令继续走原有路由和权限校验。
2. 在同步及异步仓储增加 `find_for_resolve_mr()`，按仓库及新 MR 编号查询，包含终态。原 MR 查询复用已有的活动任务列表和最新任务查询。
3. 新增原 MR 状态回复格式化函数，统一中文状态文案、时区和版本展示；回复函数显式接收当前评论的 MR 编号及讨论串 ID，也能处理“没有任务”的情况。
4. resolve MR 继续调用现有 `_reply_status()` 展示审批信息，并允许查询终态任务的最终审批记录。现有冲突块字段和表格内容保持不变。
5. 补充命令说明与回归测试。无需数据库迁移，无需额外运行时配置。

## 验证

- 原 MR 在尚未创建新 MR 的 `PENDING`、`COCO_RUNNING` 等阶段，可以查询状态。
- 一个原 MR 有多个活动版本时，回复列出全部任务，不遗漏较早的活动任务。
- 没有活动任务时，返回最新的失败、取消、超时或全部批准状态；完全没有任务时给出明确提示。
- 原 MR 的普通 `fix/...` 分支、长分支名以及不同仓库的相同 MR 编号不会导致错误分类或串任务。
- resolve MR 的活动和终态任务均返回对应审批表；其他审批、重解、取消指令及超管权限不受影响。
- 回复必须使用本次查询的 MR 编号和 `ThreadId`；查询失败、发送失败不修改任务状态。
- 单元测试覆盖路由、文案及状态映射；SQLite 仓储测试覆盖任务筛选和排序；Webhook 测试覆盖实际评论载荷到线程回复的链路。
- 部署后在原 MR 和 resolve MR 各评论一次 `/resolve status`，分别确认任务状态、审批表以及回复位置，并核对数据库状态。
