# `/resolve retry` 重解流程实现与端到端验证方案

## 结论

本轮可以实现并验证“部分冲突块被 reject 后，只重解这些块”的完整链路，但当前仓库还不能直接开始 TCE 验收。现有代码已经能识别 `/reject <reason>` 和 `/resolve retry`，也有 `PARTIAL_REJECTED`、`RETRYING`、`retry_count` 等基础结构；实际业务只做到把任务改成 `RETRYING`，worker 不会领取它，后续 Coco、飞书、MR 和 Discussion 更新均未实现。

本轮需要完成以下能力：

- reject 状态、原因和任务状态必须在同一事务内落库，并立即刷新 MR 描述。
- `/resolve retry` 只能从 `PARTIAL_REJECTED` 发起，原子冻结本轮 rejected block 和 reason，并防止多个 worker 重复执行。
- Coco 输入只包含 rejected block；approved 和 pending block 只能作为锁定集合，不能作为待重解内容发送给 Coco。
- retry commit 必须是上一次已验收 resolve commit 的直接后继，不能重新创建 resolve 分支，也不能再次创建 MR。
- Coco 完成后，只更新 rejected block 的方案；approved 和原 pending block 的审批与方案保持不变。
- 重新生成完整 Markdown，上传成一个新的飞书文件，更新同一个 MR 的描述和飞书链接。
- 在原 Discussion 中追加新版方案，不创建新的 Discussion。
- 只有新版方案、飞书链接、MR 描述和 Discussion 全部发布并回读成功后，才把本轮 rejected approval 重置为 `PENDING`。
- 最终所有 12 个块完成 approval 后，沿用已验证的飞书私聊通知链路。

按下面的实现方案，**本轮不需要新增 DDL**。`resolve_tasks.retry_count`、当前 Coco/commit/飞书字段、`approvals.comment` 和 `task_events.payload` 已能保存当前状态与每轮审计快照；`resolve_tasks.status` 是 `VARCHAR(32)`，增加 retry 中间状态也不需要改表。若以后要求用 SQL 直接查询每一轮的完整 Coco 结果和每条外部写入状态，再单独引入 `resolve_attempts` 表，不把这项扩展塞进本轮。

审计基线：

- 当前分支：`feat/lark-all-approved-notification`
- 当前 commit：`e94f843`
- 工作区在审计开始时干净。
- 上一阶段只真实验证了 `/resolve` 后四块全部 `/approval`；`/reject` 和 retry 尚无真实链路证据。

## 本轮范围

### 要实现和验证

- 两个测试分支、四个文件、12 个冲突块。
- 每个冲突块在 source 和 target 中的物理行号不同。
- 原始冲突 MR 经 `/resolve` 生成同一个 resolve 分支和 resolve MR。
- 首轮在 12 个块中完成以下操作：
  - 4 个块 `/approval`；
  - 5 个块 `/reject <reason>`；
  - 3 个块保持 `PENDING`。
- `/resolve retry` 只把 5 个 rejected block 及其 reason 发给 Coco。
- Coco 在已有 resolve 分支上新增一个普通 commit；这个 commit 的 parent 是上一次 `resolve_commit_sha`。
- resolve MR 不变，只增加一个版本和一个 commit；`target..resolve` 仍有符合 fixture 预期的有效改动。
- 重新上传一份完整 Markdown；新文件 token 和链接与首轮不同，MR 描述只指向新链接。
- 只在 5 个 rejected block 的原 Discussion 中追加新版方案。
- retry 成功后，首轮 4 个 approved block 仍为 `APPROVED`，原 3 个 pending block 仍为 `PENDING`，5 个 rejected block 重置为 `PENDING`。
- 最后再 approve 8 个 pending block，使任务进入 `ALL_APPROVED`。
- RDS、TCE、Coco、NextCode、飞书云盘和 Chrome 页面形成同一条证据链。
- 用户自行确认最终飞书私聊是否收到；服务侧仍检查 outbox 是否成功发送。

### 本轮不做

- `/resolve approve all`、`/resolve status`、`/resolve cancel`。
- 自动合入 resolve MR。
- 用户 OAuth、多用户 token 表、团队知识库归档、L2、SHA-256 和文件下载。
- target 在审批期间发生变化后的自动 rebase 或重新合并。本轮遇到 source、target 或 resolve HEAD 漂移时直接拒绝 retry，保留现场。
- rename/delete、二进制文件、submodule 和跨仓库 MR。
- 仓库 admin 权限。当前 NextCode App OpenAPI 没有已验证的成员/admin Action，本轮只用原 MR 作者执行全局指令。
- 失败后自动回滚已经 push 的 Git commit。Git 结果一旦与冻结基线不符，任务应停在可诊断状态，不能 force-push 掩盖现场。

## spec 状态机需要补清楚

现有 spec 给出的主路径是：

```text
PARTIAL_REJECTED
  -> RETRYING
  -> COCO_SUBMITTED
  -> COCO_RUNNING
  -> DOC_CREATED
  -> MR_CREATED
  -> AWAITING_APPROVAL
```

这条路径没有区分首次执行和 retry，也没有表达“已有 MR 只更新、不再创建”。实现时建议保留 `RETRYING` 作为排队态，增加 retry 专用中间态：

```text
PARTIAL_REJECTED
  │ /resolve retry：冻结 rejected block、reason，retry_count + 1
  ▼
RETRYING
  │ worker CAS 领取
  ▼
RETRY_PREPARING
  │ 校验 source、target、resolve MR 和 resolve HEAD 未漂移
  ▼
RETRY_COCO_SUBMITTED
  ▼
RETRY_COCO_RUNNING
  │ Coco 完成，服务端回读 branch、commit、文件和锁定块
  ▼
RETRY_RESULT_VERIFIED
  │ 生成完整 Markdown，上传新文件并保存新 URL
  ▼
RETRY_DOC_UPLOADED
  │ 更新同一个 MR；在原 Discussion 追加新版方案
  ▼
RETRY_PUBLISHING
  │ 原子更新 5 个 block，且只把这 5 条 approval 重置为 PENDING
  ▼
AWAITING_APPROVAL
  │ 8 个 PENDING 块逐个 approval
  ▼
ALL_APPROVED
```

`RETRYING` 不能继续复用当前的“只改 version、状态保持不变”的领取方式，否则另一个 worker 仍能再次领取。领取动作必须通过条件更新把 `RETRYING` 改成 `RETRY_PREPARING`。

### retry commit 的 Git 约束

首次 resolve 结果仍使用双 parent merge commit：

```text
parents = [frozen_source_commit, frozen_target_commit]
```

retry 不再 merge target。它从上一次已验收的 resolve commit 开始，只提交本轮修订：

```text
expected_resolve_head = previous_resolve_commit
retry_commit.parents = [previous_resolve_commit]
resolve_branch.head = retry_commit
```

开始 retry 前还要回读并确认：

- 原 source branch HEAD 仍等于 `source_commit_sha`。
- target branch HEAD 仍等于 `target_commit_sha`。
- resolve branch HEAD 仍等于任务里保存的 `resolve_commit_sha`。
- resolve MR 的 source/target 仍是任务记录的 resolve/target 分支。
- resolve MR 仍为 open，且没有被人工合入或关闭。

任一条件不满足时，不调用 Coco，任务保持 `PARTIAL_REJECTED`，在 MR 中回复具体原因。自动 rebase 会改变未 rejected block 的语义和坐标，不属于本轮范围。

### retry 失败后的状态

“任何异常都把整个任务置为 `FAILED`”不适合 retry。首轮产物和审批记录已经存在，retry 在 Coco push 之前失败时，应回到 `PARTIAL_REJECTED`，保留 rejected 状态和 reason，允许用户再次执行 `/resolve retry`。

失败分四类处理：

| 失败位置 | 处理 |
|---|---|
| worker 领取、基线回读、Coco Session 创建之前 | 回到 `PARTIAL_REJECTED`，记录 `RETRY_FAILED`；可以再次 retry |
| Coco 未 push，Run 明确失败或超时 | 回到 `PARTIAL_REJECTED`，保留 reason；可以再次 retry |
| resolve branch 已被推到无法验收的 commit | 进入 `FAILED`，保留 commit、Session、Run 和错误；禁止自动 force-push |
| commit 已验收，飞书或 NextCode 发布暂时失败 | 停在对应 retry 中间态，由 worker 从该检查点继续；不能再次调用 Coco |

## 用户给出的验证步骤还缺三项

### 必须故意留下 pending block

如果首轮把所有块都分成 approved 和 rejected，就无法证明 pending block 没有被发给 Coco。12 个块应固定分成 4 个 approved、5 个 rejected、3 个 pending。

retry 完成后，要完成全部确认，除了重新 approve 5 个曾 rejected 的块，还要 approve 首轮留下的 3 个 pending 块。最终需要补 8 次 `/approval`，不能只补 5 次。

### 必须检查 source 的有效改动仍在

上一轮出现过 resolve MR `Commits: 2`、`Changes: 0`：merge commit 合法，但最终 tree 与 target 完全相同。新 fixture 必须加入一处 source-only、非冲突且必须保留的改动，并在首次 resolve 和 retry 后都检查：

```bash
git diff --exit-code <target_commit>..<resolve_commit> -- <must-preserve-file>
```

这里预期命令返回非零，因为应存在差异；验收脚本还要直接断言 must-preserve 内容存在，不能只看 `Changes` 数字。

### 必须核对旧链接和旧审批未被误改

retry 不只是检查新文档和新评论，还要记录并对比首轮快照：

- 首轮 `resolve_commit_sha`、`lark_doc_url`、12 个 `discussion_id`。
- 4 个 approved block 的 reviewer 和 reviewed_at。
- 3 个 pending block 的空 reviewer、空 reviewed_at。
- 5 个 rejected block 的 reason、reviewer 和 reviewed_at。

retry 后，新 commit 和新飞书 URL 必须变化；MR IID、resolve branch 和 12 个 discussion ID 必须不变。approved 和原 pending 的审批字段必须逐字段保持不变。

## 当前仓库能力审计

| 能力 | 当前代码 | 判断 |
|---|---|---|
| `/reject <reason>` 解析 | `domain/commands.py` 支持 `/reject`、`reject`、emoji 和 `-1`，reason 使用普通 Unicode 字符串 | 部分实现；reason 允许为空且没有长度上限 |
| `/resolve retry` 路由 | webhook 能把 resolve MR 主线评论路由到 `ApprovalService` | 已有基础能力 |
| App 自己评论不回声 | webhook 已忽略当前 NextCode App 创建的评论 | 已验证 |
| reject 持久化 | 写 `approvals.status/comment/reviewer/reviewed_at` | 部分实现；与 task 状态分属两个事务，没有事件，也不刷新 MR 描述 |
| retry 请求 | `_handle_retry()` 只检查是否存在 REJECTED，然后把 task 改成 `RETRYING` | 阻断项 |
| retry worker | `TaskRunner.run_once()` 只调用 `claim_pending()` | 未实现，`RETRYING` 会永久卡住 |
| worker 并发领取 | 首次任务已有 `PENDING -> PREPARING` CAS | retry 未实现 CAS 领取 |
| retry Coco 输入 | manifest 里虽有 `locked_conflict_ids`、`retry_reasons`，但始终是空数组 | 阻断项 |
| retry Coco Git 动作 | prompt 总是要求从 source HEAD merge target，并生成双 parent commit | 阻断项；不适用于已有 resolve 分支的第二次提交 |
| retry 结果 schema | parser 强制 `conciliator.resolve.result.v2` 且必须恰好两个 parent | 阻断项 |
| rejected 子集校验 | `_verify_coco_result()` 要求结果 conflict ID 与全部 frozen block 完全一致 | 阻断项 |
| 锁定块保护 | 没有校验 approved/pending 方案在 retry commit 中保持不变 | 阻断项 |
| retry 文档 | `_prepare_report_data()` 要求 Coco 返回全部 block；文件名也没有 attempt | 阻断项 |
| 更新同一个 MR | post-Coco 路径总是调用“创建或恢复 MR” | 阻断项 |
| 原 Discussion 续写 | 只有创建 Discussion 和普通 reply，没有 retry 模板、comment ID 回读和幂等恢复 | 阻断项 |
| approval 子集重置 | 没有“只把本轮 rejected 重置为 PENDING”的仓储方法 | 阻断项 |
| MR 描述状态 | 模板只有 approved 布尔值；REJECTED 与 PENDING 显示相同；reject 后不刷新 | 阻断项 |
| 飞书新文件 | `upload_markdown()` 每次都会创建新文件 | 能力可复用；调用方要使用 attempt 文件名并回填新 URL |
| 当前状态字段 | `retry_count`、Coco 当前字段、`resolve_commit_sha`、`lark_doc_url` 均已存在 | 可复用 |
| 审计历史 | `task_events` 可存 JSON，但 reject/retry 事件尚未接线 | 代码缺口，不需要新表 |
| 测试 | 只有 parser 和状态迁移的少量覆盖，没有 retry service/worker/E2E 测试 | 阻断项 |

## 代码改造方案

### reject 必须原子落库并刷新描述

修改：

- `src/conciliator/services/approval_service.py`
- `src/conciliator/repositories/task_repo.py`
- `src/conciliator/repositories/_sync_repo.py`
- `src/conciliator/services/templates.py`

新增仓储事务 `reject_block_and_mark_partial(...)`：

1. 锁定 task 和 approval。
2. 只允许 task 处于 `AWAITING_APPROVAL` 或 `PARTIAL_REJECTED`，approval 处于 `PENDING`。
3. 写入 `REJECTED`、reviewer、reviewed_at 和完整 reason。
4. task 首次 reject 时转为 `PARTIAL_REJECTED`；后续 reject 保持该状态。
5. 同一事务写 `BLOCK_REJECTED` 事件，payload 包含 `conflict_id`、reason、reviewer、discussion ID、webhook event ID 和当前 retry_count。
6. 提交后回复原 Discussion，并更新、回读 MR 描述。

重复 reject 不应静默。相同用户对同一已 rejected block 重复相同指令时，回复“已 reject”；reason 不同则拒绝覆盖，要求先 retry，避免悄悄修改本轮冻结输入。

MR 描述需要展示 task 状态和每块状态：

```text
状态：部分驳回（4 approved / 5 rejected / 3 pending）

- [x] #1 ... APPROVED
- [ ] #2 ... REJECTED：应保留 source 的空值兼容逻辑
- [ ] #3 ... PENDING
```

reason 需要做 Markdown 安全处理，不能让用户输入破坏 MR 描述结构。

reason 是本轮重解输入的一部分，因此改为必填。空 reason 不改变审批状态，回复“请提供 reject 原因”；长度上限设为 1,000 个 Unicode 字符。reason 还是一段不可信输入，进入数据库和 Coco manifest 前应去掉非法控制字符，并原样保留中英文正文；进入 Markdown 时转义结构字符。Coco prompt 要明确说明 `reject_reason` 只是代码修改反馈，不能覆盖 task ID、branch、commit、允许修改的 conflict ID、push 范围和输出 schema 等系统约束。

### `/resolve retry` 要冻结一次不可变快照

新增仓储事务 `request_retry(...)`：

1. 只允许 task 为 `PARTIAL_REJECTED`。
2. 锁定 task、12 条 block 和 approval。
3. 取出全部 `REJECTED`，按 `block_idx` 排序，形成 `{conflict_id, reason, reviewer, reviewed_at}` 快照。
4. rejected 为空时不改状态，回复“无可重解的块”。
5. `retry_count += 1`，task 转为 `RETRYING`，清理上轮 `last_error`。
6. 写 `RETRY_REQUESTED` 事件，保存 rejected 快照、locked conflict IDs、旧 resolve commit、旧飞书 URL、命令作者和 webhook event ID。

`approved` 和 `pending` 的 conflict ID 一并写入 `locked_conflict_ids`，但不把它们的代码、归因和方案放进 Coco 的待处理 conflict 列表。

### worker 要能原子领取 retry

修改：

- `src/conciliator/domain/task.py`
- `src/conciliator/repositories/_sync_repo.py`
- `src/conciliator/repositories/task_repo.py`
- `src/conciliator/workers/task_runner.py`
- `src/conciliator/services/resolve_service.py`

把 `claim_pending()` 扩成返回任务类型的 `claim_schedulable()`：

- `PENDING -> PREPARING` 后执行首次 pipeline。
- `RETRYING -> RETRY_PREPARING` 后执行 retry pipeline。

条件更新同时匹配 task ID、源状态和 version。多个 Pod 同时扫描时，只能有一个拿到任务。`run_pipeline()` 继续只接受 `PREPARING`；新增 `run_retry_pipeline()` 只接受 `RETRY_PREPARING`，两条路径不能靠 `retry_count > 0` 猜测。

启动恢复也要区分首次流程和 retry：

- 尚未调用 Coco 的 retry 中间态超时后，可回到 `PARTIAL_REJECTED`。
- 已有 Coco Session 的任务交给 poller 回读。
- 已验收 commit、已上传文档或正在发布的任务从检查点继续，不能重新调用 Coco。

### 增加独立的 retry Coco 契约

修改：

- `src/conciliator/integrations/coco/client.py`
- `src/conciliator/services/resolve_service.py`
- `tests/unit/test_coco_client.py`
- `tests/unit/test_resolve_service.py`

不要放宽首次 `conciliator.resolve.result.v2` 的双 parent 约束。新增 retry 专用输入和结果 schema，例如：

```json
{
  "schema_version": "conciliator.resolve.retry.input.v1",
  "task_id": 17,
  "attempt": 2,
  "resolve_branch": "fix/...-t17",
  "expected_resolve_head": "上一轮 resolve commit",
  "frozen_target_commit": "任务冻结的 target commit",
  "rejected_conflicts": [
    {
      "conflict_id": "C0002",
      "path": "validation/.../routing.py",
      "source_text": "...",
      "target_text": "...",
      "previous_resolution": "...",
      "reject_reason": "应保留 source 的空值兼容逻辑"
    }
  ],
  "locked_conflict_ids": ["C0001", "C0003"]
}
```

retry prompt 必须明确：

- 当前挂载分支就是已有 resolve branch。
- HEAD 必须等于 `expected_resolve_head`。
- 只修改 `rejected_conflicts`，不得重新 merge target。
- approved/pending 决策不可改。
- 新建普通 commit，只 push resolve branch。
- 每个 rejected conflict 必须返回一次，不能返回 locked conflict。
- 输出新的公开决策摘要，禁止输出原始 AgentThinking。

retry 结果使用独立 schema：

```json
{
  "schema_version": "conciliator.resolve.retry.result.v1",
  "task_id": 17,
  "status": "succeeded",
  "attempt": 2,
  "resolve_branch": "fix/...-t17",
  "base_commit_sha": "上一轮 resolve commit",
  "commit_sha": "本次新 commit",
  "parent_commits": ["上一轮 resolve commit"],
  "conflicts": ["只包含 rejected 子集的完整结构"],
  "tests": [],
  "unresolved_items": []
}
```

服务端回读必须验证：

- task ID、attempt、branch、base commit 精确匹配。
- result conflict ID 集合恰好等于本轮 rejected 快照。
- result 不包含任何 locked conflict ID。
- commit 存在且 parent 只有上一次 resolve commit。
- resolve branch HEAD 等于新 commit；source 和 target HEAD 未变化。
- 新 commit 中没有冲突 marker，测试全部通过。
- 每个新 `resolved.text` 与 `GetFile(new_commit)` 的真实行切片一致。
- approved/pending block 的旧 `ai_resolution` 在新文件中仍保持不变；找不到、出现多处歧义或内容改变都应失败。
- fixture 的 source-only must-preserve 内容仍在。

### 合并新旧方案后再生成完整文档

修改：

- `src/conciliator/services/resolve_service.py`
- `src/conciliator/services/templates.py`
- `src/conciliator/integrations/lark/drive.py`，仅在需要补充恢复查询时修改

retry 结果只包含 5 个块，不能继续使用当前要求 `result_by_id[全部 block]` 的 `_prepare_report_data()`。新路径应先更新内存中的 rejected 方案，再与数据库中的 7 个锁定方案合并，生成包含 12 个块的完整 Markdown。

retry 文件名带 attempt：

```text
conciliator-mr-<original_mr_iid>-task-<task_id>-attempt-2.md
```

文档中增加：

- 当前 attempt 和新 resolve commit。
- 本轮重解的 conflict ID。
- 每个重解块的上次 reject reason。
- approved/pending 块沿用的方案。
- 本轮测试结果。

`upload_all` 正常成功后，将新 URL 和旧 URL 一起写入 `RETRY_LARK_MARKDOWN_UPLOADED` 事件，并把 task 的 `lark_doc_url` 更新为新链接。旧文件不删除。

Drive 上传没有业务幂等键。正常路径不自动重试写请求；若出现“请求可能成功但响应丢失”，只有在官方契约确认后，才允许按确定文件名查询指定文件夹做恢复。查到恰好一个文件则回收 token；查不到或同名多于一个都停下，不猜测。

### 更新原 MR，并在原 Discussion 中追加方案

修改：

- `src/conciliator/services/resolve_service.py`
- `src/conciliator/services/approval_service.py`
- `src/conciliator/integrations/nextcode/discussions.py`
- `src/conciliator/integrations/nextcode/mrs.py`
- `src/conciliator/services/templates.py`

retry 发布阶段不能调用 `CreateMergeRequest` 或 `create_discussion()`。操作顺序固定为：

1. 回读已有 resolve MR，确认 IID、source、target 和状态。
2. 等待最新 MR Version 出现，确认 source commit 是 retry commit。
3. 调 `UpdateMergeRequest` 写入新 resolve commit、新飞书链接、retry 状态和完整进度；随后 `GetMergeRequest` 回读。
4. 对 5 个 rejected block，按原 `discussion_id` 调 `CreateComment(ThreadId=...)` 追加新版方案。
5. 每条回复带确定的隐藏标记，并通过 `GetThread` 或 `GetComment` 回读，避免进程重启后重复评论。
6. 5 条回复全部可见后，原子更新这 5 个 block 的新方案和坐标，只把对应 approval 从 `REJECTED` 重置为 `PENDING`。
7. task 进入 `AWAITING_APPROVAL`，再次刷新并回读 MR 描述。

新版 Discussion 回复沿用 7.8 的主体结构，增加 attempt 和 reject reason：

```markdown
<!-- conciliator:task=17:block=C0002:attempt=2 -->
**🔄 冲突块 #2 重新方案 v2**

**上次 reject 原因**：应保留 source 的空值兼容逻辑

**Ours（源分支）**：
...

**Theirs（目标分支）**：
...

**🤖 AI 解冲突方案**：
...

**决策摘要**：...
**额外修复**：...

---
👤 回复 `/approval`、`approve` 或 `reject <reason>` 即可确认
```

原 Discussion ID 不变。即使 Codebase 把旧行位置标为 outdated，Activity 中也必须能看到同一 Thread 的完整历史；本轮不创建替代 Discussion。

NextCode 的回复模型是向同一个 `ThreadId` 追加评论，没有“回复某条回复”的二级嵌套。用户所说的“在 `/reject` comment 下方”在契约层应解释为：新版方案与 `/reject`、reject 回执都出现在同一 Thread，并按创建时间排在它们之后。

### 只重置本轮 rejected approval

新增仓储事务 `publish_retry_result(...)`：

- 条件：task 为 `RETRY_PUBLISHING`，retry_count 与本轮 attempt 匹配。
- 更新对象：只允许本轮快照中的 block ID。
- 对 rejected block 更新 AI 方案、决策摘要、额外修复和 resolved 坐标。
- 对这些 block 的 approval 执行 `REJECTED -> PENDING`，清空当前 reviewer、reviewed_at 和 comment。reject 历史已在 `BLOCK_REJECTED`、`RETRY_REQUESTED` 事件和旧 Discussion 中保留。
- approved block 的 status、reviewer、reviewed_at 和方案逐字段不变。
- 原 pending block 的 status、reviewer、reviewed_at 和方案逐字段不变。
- 更新 task 的 `resolve_commit_sha`、Coco 当前字段、`coco_result_json`、`lark_doc_url` 和状态。
- 同一事务写 `RETRY_PUBLISHED`，保存旧/新 commit、旧/新飞书 URL、5 个 block ID 和 reply comment ID。

若条件更新影响的 rejected 行数不是 5，整个事务回滚。

## DDL 结论

本轮必须使用的字段已经存在：

| 数据 | 现有位置 |
|---|---|
| 当前重解轮次 | `resolve_tasks.retry_count` |
| 上一次和当前 resolve commit | `resolve_tasks.resolve_commit_sha`，切换记录写 `task_events` |
| 当前 Coco Session/Run/result | `resolve_tasks.coco_*`、`coco_result_json` |
| 当前飞书链接 | `resolve_tasks.lark_doc_url` |
| 当前审批状态和 reject reason | `approvals.status/comment` |
| 每块当前方案和 resolved 坐标 | `conflict_blocks` |
| 每轮冻结快照和历史 | `task_events.payload` |

因此不新增 `0006` migration，也不提交 RDS DDL 工单。实现时需要更新 `TaskStatus` 和代码，但数据库中的 status 列没有 enum/check 约束。

如果以后有以下需求，再设计 `resolve_attempts` 和 `conflict_resolution_versions`：

- 高频多轮 retry，需要直接按 attempt 做报表。
- 需要长期保存每轮完整 Coco JSON，而不只保留事件摘要和飞书文件。
- 需要对每条 Discussion 发布状态做关系型查询。
- 需要后台自动恢复跨多天的半完成外部写入。

## 12 块 fixture 设计

新增 `scripts/create_resolve_retry_fixture.py`，只负责创建并推送两个分支，不创建 MR。支持 `--dry-run` 和唯一 `--run-id`。

### 文件与冲突分布

使用四个小型 Python 文件，每个文件三个互相分隔的冲突块，共 12 个：

| 文件 | conflict ID | 首轮动作 |
|---|---|---|
| `routing.py` | C0001 / C0002 / C0003 | approve / reject / pending |
| `policy.py` | C0004 / C0005 / C0006 | reject / approve / reject |
| `formatter.py` | C0007 / C0008 / C0009 | pending / reject / approve |
| `retry.py` | C0010 / C0011 / C0012 | reject / pending / approve |

每两个冲突点之间加入不冲突的稳定函数，防止 Git 把相邻修改合并成一个 marker block。source 和 target 在每个块之前插入不同数量的独有注释，使 12 个块的两侧物理行号全部不同。

source 另加一处 target 没有的非冲突文件或代码段，例如 `source_contract.py`。fixture 中的测试必须断言它存在，防止再次出现“解冲突成功但 Changes 为 0”。

### reject reason

固定使用五条可识别原因，覆盖中文、英文和中英混合：

| block | reason |
|---|---|
| C0002 | `应保留 source 的空值兼容逻辑` |
| C0004 | `keep target validation before fallback` |
| C0006 | `需要合并两侧条件，不能整段选择一边` |
| C0008 | `preserve source metric name，但使用 target threshold` |
| C0010 | `重试次数沿用 source，上限使用 target` |

fixture 自带测试，使这五条 reason 对应到可判定的目标代码语义。否则 Coco 即使输出另一种“看起来合理”的结果，也无法自动判断是否真的响应了 reject reason。

### fixture 脚本必须输出

- run ID、source/target 分支名。
- 公共 base、source commit、target commit。
- 四个文件和 12 个稳定 conflict ID。
- 每块 source/target 行范围，且逐块断言两者不同。
- 预期首轮动作与 reason。
- must-preserve 文件和最终业务断言。
- 本地 `git merge --no-commit` 后解析得到恰好四个文件、12 个 marker block。
- 远端分支不存在检查和 push 后 `ls-remote` 回读。

## 自动化测试清单

### 领域与指令

- `/reject 中文原因`、`/reject English reason`、首尾空格和大小写。
- reason 长度上限、控制字符、Markdown 结构字符和 prompt 注入文本。
- `/resolve retry` 在 MR 主线和 Discussion 都能识别。
- `PARTIAL_REJECTED -> RETRYING -> RETRY_PREPARING` 合法。
- 所有 retry 中间态的继续、取消和失败迁移明确。
- `APPROVED`、原 `PENDING` 不允许被 retry 仓储方法重置。

### 仓储与并发

- reject 和 task `PARTIAL_REJECTED` 同事务提交；任一步失败全部回滚。
- `request_retry()` 快照恰好五个 rejected，reason 字节级一致。
- 八个并发 worker 只能领取一次 `RETRYING`。
- 重复 webhook delivery 不增加 retry_count。
- 两条不同的 `/resolve retry` 评论在任务已离开 `PARTIAL_REJECTED` 后，第二条只得到当前状态回复。
- `publish_retry_result()` 只重置五个 rejected；4 个 approved 和 3 个 pending 的字段完全不变。
- 每轮事件包含 attempt、旧/新 commit、旧/新飞书 URL 和 block 集合。

### Coco 契约

- retry 输入只含五个 rejected conflict，reason 完整。
- `locked_conflict_ids` 恰好七个，不包含待重解块。
- `expected_resolve_head` 是上一次 resolve commit，不是 frozen source。
- prompt 不再要求 merge target 或生成双 parent commit。
- retry result 只接受一个 parent。
- 多返回 locked ID、少返回 rejected ID、重复 ID、attempt 不匹配、branch 不匹配全部拒绝。
- branch HEAD、commit parent、source/target HEAD、resolved text、marker 和 tests 全部回读。
- protected block 发生变化时拒绝发布。

### 飞书、MR 和 Discussion

- retry Markdown 包含 12 个当前方案和五条 reject reason。
- retry 文件名带 attempt，新 token 与旧 token 不同。
- 同一个 MR 只调用 `UpdateMergeRequest`，绝不再次调用 `CreateMergeRequest`。
- MR 描述包含 retry_count、新 commit、新飞书链接和 `4/5/3` 状态分布。
- 只对五个 rejected discussion ID 调用 reply。
- reply 模板能正确渲染多行代码块和中英文 reason。
- 相同 task/block/attempt 的隐藏 marker 已存在时不重复回复。
- 发布完成后描述变成 `4 approved / 0 rejected / 8 pending`。

### 故障恢复

至少做以下注入测试：

- Coco Session 创建成功、DB 更新前进程退出。
- Coco push 后结果回读暂时不完整。
- commit 验收成功、飞书上传前退出。
- 飞书上传成功、MR 描述更新前退出。
- 三条 Discussion 已回复、第四条回复时失败。
- 所有回复成功、approval 重置事务前退出。

恢复后不得重复调用 Coco，不得创建第二个 MR；同一 retry attempt 不得产生重复 Discussion 回复。Drive 的“请求成功但响应丢失”必须按前述文件夹精确查询规则处理，不能盲目重传。

## 本地门禁

实现完成后先执行：

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q
.venv/bin/mypy src scripts/create_resolve_retry_fixture.py scripts/verify_resolve_retry_state.py
.venv/bin/python scripts/check_schema.py
.venv/bin/python scripts/create_resolve_retry_fixture.py --dry-run --run-id local-check
.venv/bin/python -m py_compile scripts/create_resolve_retry_fixture.py scripts/verify_resolve_retry_state.py
git diff --check
```

MySQL Testcontainers 因本机 Docker 不可用而 skip 时，要明确记录 skip 数；不能把 SQLite 单测当成 MySQL 事务证据。DDL 结论为“不需要 migration”，仍要执行 schema 检查，确认模型没有误改。

## 部署前检查

代码合入新分支、推送远端后：

1. 构建新的 SCM 版本。
2. 更新 TCE 小流量和生产实例。
3. 从实例页读取每个 Pod 当时的随机 access port，不把主端口 8000 当作 Pod 实际端口。
4. 在每个 Pod WebShell 中使用 `127.0.0.1:${PORT0}` 检查 `/healthz` 和 `/readyz`。
5. 回读运行镜像中的 retry 状态、Coco schema 常量和 commit，确认不是旧版本。
6. 检查 RDS 当前 `alembic_version` 未变化；本轮不提交 DDL 工单。
7. 确认 NextCode、Coco、飞书 KMS 和 tenant token 基础契约仍通过。
8. 开始业务写入前确认没有其他活跃验证任务。

## TCE 端到端验收步骤

### 创建 fixture 分支

本地运行：

```bash
.venv/bin/python scripts/create_resolve_retry_fixture.py --run-id <YYYYMMDD-HHMM>
```

保存脚本输出的 manifest。使用 Chrome 创建 source -> target MR，检查：

- MR 明确显示合入冲突。
- 冲突文件恰好四个，marker block 恰好 12 个。
- source/target 行号与 manifest 对应且逐块不同。
- MR 作者是执行 `/resolve retry` 的测试用户。

### 首次 `/resolve`

在原 MR 主线提交：

```text
/resolve
```

等待任务进入 `AWAITING_APPROVAL`，记录：

- task ID、Coco Session/Run、首次 resolve commit。
- resolve branch、新 MR IID 和 URL。
- 首份飞书 file token 和 URL。
- 12 个 conflict ID、discussion ID 和 resolved 坐标。

页面与 Git 验收：

- 新 MR 无冲突，source 是 resolve branch，target 是测试 target。
- 首次 resolve commit 是 frozen source/target 的双 parent merge commit。
- `Changes` 非零，must-preserve source 内容存在。
- MR 描述是 `0 approved / 0 rejected / 12 pending`。
- 飞书文件包含 12 个冲突块、方案、归因和测试结果。
- 12 个 Discussion 均符合 7.8，代码块正常渲染。

### 制造 approved、rejected 和 pending 三种状态

按 fixture manifest 执行：

- `/approval`：C0001、C0005、C0009、C0012。
- `/reject <reason>`：C0002、C0004、C0006、C0008、C0010。
- 不操作：C0003、C0007、C0011。

每次操作后检查 Discussion 回执和 MR 描述。第五次 reject 后，预期为：

```text
task.status = PARTIAL_REJECTED
approved = 4
rejected = 5
pending = 3
```

记录 12 条 approval 的完整快照，并确认五条 reason 在页面和 RDS 中字节级一致。

### 触发 `/resolve retry`

在 resolve MR 主线提交：

```text
/resolve retry
```

检查 webhook delivery 只处理一次，task 依次进入 retry 状态，`retry_count=1`。Coco 输入应满足：

- attempt 为 2。
- rejected conflicts 只有 C0002、C0004、C0006、C0008、C0010。
- 五条 reason 完整。
- locked IDs 是其余七个。
- expected HEAD 等于首次 resolve commit。

等待 Coco 在同一 resolve branch push 一个新 commit。检查：

- 新 commit 与首次 resolve commit 不同。
- 新 commit 只有一个 parent，且 parent 是首次 resolve commit。
- source/target HEAD 未变化。
- resolve MR IID 不变，MR Version 和 commit 数增加。
- 新 diff 只改变预期的重解语义；4 个 approved 和 3 个原 pending 方案保持不变。
- must-preserve source 内容仍存在，MR `Changes` 非零。

### 验收重新发布

等待任务回到 `AWAITING_APPROVAL`，检查：

- 飞书文件夹出现 attempt 2 的新 Markdown。
- 新旧 file token、URL 不同，旧文件仍存在。
- MR 描述里的飞书链接已经替换成新 URL。
- MR 描述中的 resolve commit 是 retry commit。
- MR 状态为 `4 approved / 0 rejected / 8 pending`。
- 4 个 approved block 仍勾选；其 reviewer 和 reviewed_at 不变。
- 3 个原 pending block 仍未勾选，且没有收到新方案评论。
- 5 个曾 rejected 的原 Discussion 中各增加一条 attempt 2 方案；discussion ID 不变。
- 这五条 approval 已重置为 `PENDING`，reason 已转存到事件和 Discussion 历史。
- 没有新的 MR，没有新的顶层 conflict Discussion，没有 Conciliator 自己评论触发的权限回声。

在 TCE Pod WebShell 中运行只读核对：

```bash
python scripts/verify_resolve_retry_state.py \
  --task-id <task_id> \
  --expect-status AWAITING_APPROVAL \
  --expect-retry-count 1 \
  --expect-approved C0001,C0005,C0009,C0012 \
  --expect-pending C0002,C0003,C0004,C0006,C0007,C0008,C0010,C0011 \
  --expect-retry-replies 5
```

### 完成全部确认

对八个 pending block 逐个回复 `/approval`：

- 五个重解块：C0002、C0004、C0006、C0008、C0010。
- 三个原 pending 块：C0003、C0007、C0011。

确认 MR 描述从 `4/12` 逐步更新到 `12/12`，最终 task 为 `ALL_APPROVED`。服务侧检查飞书通知 outbox 只产生一条 `ALL_APPROVED` 通知；用户自行确认私聊内容。

最终核对：

```bash
python scripts/verify_resolve_retry_state.py \
  --task-id <task_id> \
  --expect-status ALL_APPROVED \
  --expect-retry-count 1 \
  --expect-approved C0001,C0002,C0003,C0004,C0005,C0006,C0007,C0008,C0009,C0010,C0011,C0012 \
  --expect-retry-replies 5
```

## RDS 验收 SQL

在 RDS AutoSQL 使用实际 task ID：

```sql
SELECT
  id, status, retry_count, original_mr_id, new_mr_id, new_mr_url,
  source_commit_sha, target_commit_sha, resolve_commit_sha,
  coco_session_id, coco_run_id, lark_doc_url,
  trigger_author_email, active_key, last_error, version,
  created_at, updated_at
FROM resolve_tasks
WHERE id = <task_id>;
```

```sql
SELECT
  b.block_idx, b.conflict_id, b.file_path,
  b.discussion_id, b.ai_resolution, b.decision_summary,
  b.resolved_path, b.resolved_start_line, b.resolved_end_line,
  a.status, a.reviewer_email, a.reviewed_at, a.comment, a.version
FROM conflict_blocks b
JOIN approvals a ON a.block_id = b.id
WHERE b.task_id = <task_id>
ORDER BY b.block_idx;
```

```sql
SELECT id, event_type, payload, coco_event_id, occurred_at
FROM task_events
WHERE task_id = <task_id>
ORDER BY id;
```

关键事件至少包括：

```text
BLOCK_REJECTED x 5
RETRY_REQUESTED x 1
RETRY_CLAIMED x 1
RETRY_COCO_SUBMITTED x 1
RETRY_COCO_VERIFIED x 1
RETRY_LARK_MARKDOWN_UPLOADED x 1
RETRY_DISCUSSION_REPLIED x 5
RETRY_PUBLISHED x 1
BLOCK_APPROVED x 12
ALL_APPROVED x 1
LARK_NOTIFICATION_QUEUED x 1
LARK_NOTIFICATION_SENT x 1
```

`BLOCK_APPROVED x 12` 指每个 block 最终只发生一次成功的 approval；首轮已经通过的四块在 retry 后不得再次产生 approval 事件。

```sql
SELECT
  task_id, notification_type, recipient_email, recipient_open_id,
  mr_url, request_uuid, status, attempts, message_id, last_error,
  created_at, updated_at
FROM lark_notification_outbox
WHERE task_id = <task_id>;
```

最终预期：

- `resolve_tasks.status = 'ALL_APPROVED'`。
- `retry_count = 1`。
- `active_key IS NULL`。
- 12 条 approval 全部为 `APPROVED`。
- `resolve_commit_sha` 是 retry commit。
- `lark_doc_url` 是 attempt 2 文件链接。
- outbox 恰好一条，`mr_url` 等于同一个 resolve MR URL。
- `last_error IS NULL`。

## 验收失败的判定

出现以下任一情况，本轮都不能记为通过：

- worker 把 task 留在 `RETRYING`，没有启动 Coco。
- Coco 输入包含 approved 或 pending block 的待解内容。
- 任一 reject reason 丢失、截断或乱码。
- retry 再次 merge target，或生成的 commit parent 不是上一次 resolve commit。
- source、target 或 resolve 基线漂移后仍继续执行。
- 创建了第二个 resolve MR或新的顶层 conflict Discussion。
- approved/pending block 的方案、状态、reviewer 或 reviewed_at 被改动。
- retry 后 MR `Changes: 0`，或 must-preserve source 内容消失。
- 飞书没有新文件，MR 仍指向旧链接，或 RDS URL 与页面不一致。
- 新方案评论没有留在原 Thread，少于五条或重复出现。
- approval 在新版方案发布完成前提前从 `REJECTED` 变成 `PENDING`。
- 最终通知出现零条或多条。
- 只看页面或只看 RDS 就宣布成功，没有对齐 task、commit、MR、file token、discussion 和 event。

## 交付顺序

实现阶段建议按以下提交边界拆分，便于回滚和评审：

1. `feat: add atomic reject and retry task transitions`
2. `feat: add coco rejected-block retry contract`
3. `feat: publish retried resolutions to mr and lark`
4. `test: add resolve retry fixture and verification`
5. `docs: align resolve retry state machine`

完成代码和本地门禁后推送远端。用户构建 SCM、更新 TCE；部署完成后再按本计划操作 Chrome、Pod WebShell、飞书云盘和 RDS，不在旧镜像上提前制造真实业务数据。

## 依据和证据边界

- 本地 spec：`docs/superpowers/specs/2026-07-21-conciliator-design.md`，重点是 6.3、6.4、7.3、7.5、7.6、7.7 和 7.8。
- 上一阶段计划：`docs/superpowers/plans/2026-09-03-resolve-all-blocks-approval-e2e-validation.md`。
- 产品文档：<https://bytedance.larkoffice.com/wiki/WYhowUp2DiwbttkT1ENcZdN8nZb>。
- Codebase MR：<https://code.byted.org/xiahaitao/Conciliator/merge_requests>。
- 飞书文件夹：<https://bytedance.larkoffice.com/drive/folder/K0zefVjqUlsicAdu4TjccNhjnjh>。
- RDS AutoSQL：<https://cloud.bytedance.net/rds/detail/db/global/aeolus_govern/autoSQL?x-global-site=cn&x-global-vregion=China-North&x-global-vdc=cn&x-global-detail-tab=China-North&x-resource-account=public&x-bc-region-id=bytedance&x-bc-vregion=ChinaSinf-North&x-bc-vdc=multicloud>。

证据边界保持不变：单元测试证明本地逻辑；TCE WebShell 证明当前镜像和运行身份；Chrome 证明用户可见结果；RDS 证明状态与审计持久化；真实业务链路只有五层证据对齐后才算通过。

## 本轮实现结果

已在 `feat/resolve-retry-e2e` 完成本计划对应的业务代码和复用工具：

- reject reason 改为必填，保存 approval、task 状态和 `BLOCK_REJECTED` 事件时使用同一事务。
- `/resolve retry` 原子冻结 rejected/locked 集合和每条 reason，增加重解轮次后进入可调度队列。
- worker 用 `RETRYING → RETRY_PREPARING` 条件更新认领任务，未认领到的 worker 直接退出。
- Coco 使用独立的 retry 输入和输出 schema，只接收被拒块，并强制新提交以旧 resolve commit 为唯一 parent。
- 服务端校验分支基线、commit parent、rejected 精确集合、locked 方案、最终文件、测试结果和 blame。
- 重解成功后上传新的 attempt Markdown，更新原 MR 描述，在原 Discussion 追加新方案；不创建第二个 MR 或顶层 Discussion。
- 全部外部发布完成后，只把本轮 rejected approval 重置为 `PENDING`，保留原 approved 和原 pending 状态。
- post-Coco 发布步骤支持按状态和 `task_events` 续跑；已记录或能从 Thread marker 回读到的回复不会重复发送。
- 新增 4 文件、12 冲突块夹具；本地 dry-run 已确认冲突数量、文件数量及两侧物理行号均符合验收要求。
- 新增只读核对脚本，对齐 RDS task/approval/event、NextCode 分支和 commit、原 MR、Discussion、飞书 URL 及 must-preserve source 内容。

本轮没有新增 DDL。当前模型和五份 migration 的静态一致性检查已通过。

本地门禁结果：

```text
pytest: 187 passed, 11 skipped
ruff: passed
mypy: passed (57 source and validation-script files)
schema static checks: passed
fixture dry-run: 4 conflict files / 12 conflict blocks
```

这些结果只证明本地实现和夹具成立。SCM 构建和 TCE 更新完成后，仍需按上文步骤做一次真实 `/resolve → approval/reject → /resolve retry → approval` 验收。
