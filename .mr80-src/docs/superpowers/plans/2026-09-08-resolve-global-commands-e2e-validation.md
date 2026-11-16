# `/resolve` 全局指令、评论别名与发起人权限端到端验证规划

## 本轮要证明什么

本轮用两条相互隔离的真实任务，同时验证以下能力：

- 原始 MR 已有活跃任务时，再次评论 `/resolve` 会在该评论下方明确提示，不会静默结束，也不会创建第二条任务。
- `/resolve status`、`/resolve approve all`、`/resolve retry` 和 `/resolve cancel` 仍按既有状态机工作。
- 方括号形式的单块通过、拒绝指令可以驱动真实 Discussion 和 RDS 状态变化。
- 全局写指令只允许 `/resolve` 发起人执行。
- 单块指令只允许 block author 或 `/resolve` 发起人执行，原 MR author 不再天然拥有权限。
- 全部冲突块通过后，Resolve MR 的完成评论和飞书私信都通知 `/resolve` 发起人。

任务 A 使用四块夹具，验证重复 `/resolve`、`status`、全局权限、`approve all` 和完成通知。任务 B 使用十二块夹具，覆盖全部方括号别名、单块权限、rejected 拦截、`retry` 权限和 `cancel`。

Codebase 页面负责确认用户能看到的 MR、Discussion、回复位置、权限提示和通知对象。TCE Pod 与服务日志负责确认部署版本、服务健康和命令处理分支。RDS 负责确认 task、approval、event 和飞书通知 outbox。

当前实现没有新增字段，本轮不需要 DDL。验证只创建测试分支、测试 MR、评论和业务任务，不合入测试 MR。

## 账号前提和证据边界

完整验证发起人权限需要两个不同的真实 Codebase 账号：

- 账号 I：在原始 MR 评论 `/resolve`，是任务的 `trigger_author_email`，也是预期收到完成通知的人。
- 账号 M：创建原始 MR，是 `mr_author_email`，但不发起 `/resolve`。
- 夹具作者 F：用于提交 source、target 夹具代码的非交互邮箱，必须与 I、M 都不同。

账号 I 和 M 必须都有仓库评论权限。账号 I 的邮箱还必须能被飞书应用解析成 open_id。若只有一个真实账号，只能验证指令功能，无法证明“MR author 与 `/resolve` 发起人不同时”的授权边界，不能把发起人权限标为端到端通过。

创建夹具时使用只对该命令生效的独立作者身份，避免账号 M 因为同时是 block author 而绕过单块权限校验，也避免污染后续 Git 提交：

```bash
GIT_AUTHOR_NAME="Conciliator Fixture" \
GIT_AUTHOR_EMAIL="conciliator-fixture@invalid.example" \
GIT_COMMITTER_NAME="Conciliator Fixture" \
GIT_COMMITTER_EMAIL="conciliator-fixture@invalid.example" \
<夹具脚本命令>
```

夹具推送后，根据脚本输出的 source、target commit SHA 检查作者邮箱。任一 commit 的 author email 与账号 I 或 M 相同，都要停止验证并重建夹具。

## 验证基线

- 代码分支：`fix/resolve-initiator-ownership`
- 重复 `/resolve` 提示提交：`c491b34374b4bd5bdb1d8a7c69986c6b5415fa6e`
- 方括号评论别名提交：`a63d3a885ebba183c3bb7b78bd477f20fc456609`
- `/resolve` 发起人权限提交：`393df0244647683685495c95734d9b795930b2b4`
- Codebase 仓库：`xiahaitao/Conciliator`
- NextCode repo id：`1106267`
- TCE 服务 id：`210017612`
- TCE 服务：`data.aeolus.conciliator`
- 飞书 Markdown 目录 token：`K0zefVjqUlsicAdu4TjccNhjnjh`

SCM 构建版本必须包含上面三个功能提交。后续单独增加验证文档的 commit 不改变运行时代码。

TCE Pod 的监听端口读取该 Pod 的 `$PORT0`，不把主端口 `8000` 当作 Pod 内实际端口。

## 部署预检

在 TCE 实例页选择当前 Pod 并进入 WebShell：

```bash
echo "$PORT0"
curl -fsS "http://127.0.0.1:${PORT0}/healthz"
curl -fsS "http://127.0.0.1:${PORT0}/readyz"
```

预期 `/healthz` 和 `/readyz` 都返回 `ok`。随后确认容器包含三项新增实现和原有全局指令：

```bash
grep -R "ACTIVE_TASK_REPLY" -n /opt/tiger/conciliator/src/conciliator
grep -R "active_task_reply_created" -n /opt/tiger/conciliator/src/conciliator
grep -R "thumbsup" -n /opt/tiger/conciliator/src/conciliator/domain/commands.py
grep -R "minusone" -n /opt/tiger/conciliator/src/conciliator/domain/commands.py
grep -R "_is_trigger_author" -n /opt/tiger/conciliator/src/conciliator/services/approval_service.py
grep -R '限 `/resolve` 发起人' -n /opt/tiger/conciliator/src/conciliator
grep -R "approve_all_pending_and_enqueue" -n /opt/tiger/conciliator/src/conciliator
grep -R "TASK_CANCELLED" -n /opt/tiger/conciliator/src/conciliator
```

部署代码、健康检查或 RDS 就绪检查任一失败时停止业务验证，先修部署问题。

## 任务 A：重复触发、status、approve all 和完成通知

### 创建四块夹具和原始 MR

使用有推送权限的本地凭证运行夹具脚本，保留脚本输出：

```bash
GIT_AUTHOR_NAME="Conciliator Fixture" \
GIT_AUTHOR_EMAIL="conciliator-fixture@invalid.example" \
GIT_COMMITTER_NAME="Conciliator Fixture" \
GIT_COMMITTER_EMAIL="conciliator-fixture@invalid.example" \
.venv/bin/python scripts/create_resolve_approval_fixture.py \
  --run-id <YYYYMMDD-HHMM>-initiator-success \
  --output /tmp/conciliator-initiator-success.json
```

检查夹具满足两个冲突文件、四个冲突块，source 与 target 对应块的物理行号不同。账号 M 在 Codebase 页面创建 source 合入 target 的原始 MR，确认页面显示合入冲突。

切换到账号 I，在原始 MR 主线评论：

```text
/resolve
```

等待 RDS 出现一条新 task，先确认 `trigger_author_email` 是账号 I。任务进入 `BRANCH_CREATED` 或后续状态后，再确认：

- `mr_author_email` 是账号 M。
- 两个邮箱不同。

等待服务创建 resolve 分支和新的 Resolve MR。开始执行后续命令前确认：

- Resolve MR 没有合入冲突。
- Resolve MR 有四个独立冲突块 Discussion。
- MR 描述显示 `0 approved / 0 rejected / 4 pending`。
- MR 描述包含飞书 Markdown 链接。
- MR 描述中三条全局写指令都标明“限 `/resolve` 发起人”。

### 验证重复 `/resolve` 会回复用户

任务仍未进入终态时，账号 I 回到原始 MR，再创建一条新的顶层评论：

```text
/resolve
```

预期 Conciliator 在这条评论的下方回复：

```text
当前 MR 已有解冲突任务正在运行，请等待该任务结束后再重新执行 `/resolve`。
```

同时确认：

- 回复与第二条 `/resolve` 属于同一个 Thread，不是无关的顶层评论。
- 原始 MR 仍只有一条活跃 task。
- `TASK_CREATED` 仍只有一条，没有第二个 task id。
- 没有第二个 resolve 分支、Resolve MR、Coco Session 或飞书文档。
- 服务日志包含 `trigger_dropped_active_exists` 和 `active_task_reply_created`。
- Conciliator 自己创建的提示评论被 webhook 入口忽略，不产生回声回复。

第二条用户评论本身会产生一条正常的 webhook delivery 记录；“没有新增数据”仅指不新增业务 task、task event 和外部解冲突产物。

### 验证 status 仍允许任何人执行

账号 M 在 Resolve MR 主线评论：

```text
/resolve status
```

预期 Conciliator 在该评论线程中回复：

- 任务状态 `AWAITING_APPROVAL`。
- `0/4 APPROVED，0 REJECTED，4 PENDING`。
- 按 `#1` 到 `#4` 排序的四行明细。
- 每行都有文件区间、source author、target author 和审批状态。

RDS 中的 task、approval 和 task event 在执行前后完全相同。`status` 是只读指令，MR author 即使不是 `/resolve` 发起人也可以查询。

### 验证 MR author 不能执行全局写指令

账号 M 依次在 Resolve MR 主线评论：

```text
/resolve approve all
```

```text
/resolve retry
```

```text
/resolve cancel
```

三次都应得到以下权限提示，其中邮箱是账号 M：

```text
@<账号M邮箱> 无权限执行此指令（限 `/resolve` 发起人）
```

每次操作后都要确认 task 仍为 `AWAITING_APPROVAL`，四个 approval 仍为 `PENDING`，没有新增 `BLOCK_APPROVED`、`RETRY_REQUESTED`、`TASK_CANCELLED`、`ALL_APPROVED` 或飞书通知 outbox。

### 验证发起人可以 approve all

账号 I 在 Resolve MR 主线评论：

```text
/resolve approve all
```

页面预期：

- Conciliator 在该评论线程中回复“已将 4 个 PENDING 冲突块设为 APPROVED”。
- MR 描述更新为 `4 approved / 0 rejected / 0 pending`，并显示全部确认完成。
- 四个冲突块全部显示 APPROVED。
- 没有新建 Resolve MR，没有新增 Coco commit，也没有重新上传解冲突文档。
- Resolve MR 新增完成评论，第一行只 `@账号I邮箱`，不 `@账号M邮箱`；正文为“所有冲突块已 approve，可以合入。”

RDS 预期：

- `resolve_tasks.status = 'ALL_APPROVED'`。
- `resolve_tasks.active_key IS NULL`。
- 四条 approval 全部为 `APPROVED`，`reviewer_email` 都是账号 I。
- 本次命令新增四条 `BLOCK_APPROVED`，每条 payload 的 `approved_via` 为 `approve_all`。
- `ALL_APPROVED` 和 `LARK_NOTIFICATION_QUEUED` 各一条。
- `lark_notification_outbox.recipient_email` 是账号 I，不是账号 M。
- 该 task 只有一条 outbox，最终进入 `SENT`；重复 webhook 不得产生第二条。

账号 I 最后确认收到飞书私信，消息内的 Resolve MR 链接可访问。账号 M 不应收到这条完成通知。

## 任务 B：方括号别名、单块权限、retry 和 cancel

### 创建十二块夹具和第二个原始 MR

继续使用独立夹具作者 F，创建四个冲突文件、十二个冲突块：

```bash
GIT_AUTHOR_NAME="Conciliator Fixture" \
GIT_AUTHOR_EMAIL="conciliator-fixture@invalid.example" \
GIT_COMMITTER_NAME="Conciliator Fixture" \
GIT_COMMITTER_EMAIL="conciliator-fixture@invalid.example" \
.venv/bin/python scripts/create_resolve_retry_fixture.py \
  --run-id <YYYYMMDD-HHMM>-aliases-cancel \
  --output /tmp/conciliator-aliases-cancel.json
```

账号 M 创建 source 合入 target 的原始 MR，账号 I 在原始 MR 主线评论 `/resolve`。等待新的 Resolve MR 进入 `AWAITING_APPROVAL`，确认有十二个独立 Discussion，RDS 中 `trigger_author_email = 账号I`、`mr_author_email = 账号M`。

### 验证 MR author 不能操作不属于自己的冲突块

账号 M 在 C0011 的 Discussion 回复：

```text
[LGTM]
```

由于 C0011 的 source、target author 都是夹具作者 F，账号 M 既不是 block author，也不是 `/resolve` 发起人。预期下方新增回复：

```text
无权限，仅 block author 或 `/resolve` 发起人可操作
```

C0011 必须保持 `PENDING`，不得新增 `BLOCK_APPROVED` 事件，MR 描述中的进度不变。

### 验证四种通过别名

账号 I 按顺序在 C0001 至 C0004 的 Discussion 回复：

| Conflict | 评论内容 | 预期 |
| --- | --- | --- |
| C0001 | `[LGTM] looks good` | `APPROVED`，证明仍按第一个单词识别 |
| C0002 | `[JIAYI]` | `APPROVED` |
| C0003 | `[CheckMark]` | `APPROVED` |
| C0004 | `[THUMBSUP]` | `APPROVED` |

每条评论都必须在原 Discussion 下收到 `Block #N 已 approve` 回复。MR 描述依次推进到 `1/12`、`2/12`、`3/12`、`4/12`。四条 approval 的 `reviewer_email` 都是账号 I，这可以证明账号 I 即使不是 block author，也能以 `/resolve` 发起人身份操作。

### 验证六种拒绝别名

账号 I 按下表回复 C0005 至 C0010：

| Conflict | 评论内容 | RDS 保存的 reason |
| --- | --- | --- |
| C0005 | `[MinusOne]保留 source fallback` | `保留 source fallback` |
| C0006 | `[CrossMark]keep target validation` | `keep target validation` |
| C0007 | `[ThumbsDown]不要删除超时保护` | `不要删除超时保护` |
| C0008 | `[MinusOne] 保留 access allowlist` | `保留 access allowlist` |
| C0009 | `[CrossMark] keep output formatting` | `keep output formatting` |
| C0010 | `[ThumbsDown] 保留 runtime guard` | `保留 runtime guard` |

前三条验证标记与 reason 之间没有空格，后三条验证有空格。每条评论都应在原 Discussion 下收到 reject 回复，保存的 reason 不包含方括号标记和开头空格。

全部执行后预期：

- task 为 `PARTIAL_REJECTED`。
- `4 approved / 6 rejected / 2 pending`。
- C0011、C0012 仍为 `PENDING`。
- 六条 `BLOCK_REJECTED` 的 payload 和 approval.comment 与上表一致。

空原因不在真实任务上提交，避免制造无意义评论；该边界由单元测试覆盖：`[MinusOne]`、`[CrossMark]` 和 `[ThumbsDown]` 单独出现时均被忽略。

### 验证权限检查先于全局写操作

账号 M 在存在六个 REJECTED 块时依次执行：

```text
/resolve approve all
/resolve retry
/resolve cancel
```

三次都必须先得到“限 `/resolve` 发起人”的权限提示，不能因为当前有 rejected 块而返回其他业务提示。执行前后的 task version、approval、task event 和 outbox 必须一致。

账号 I 随后执行：

```text
/resolve approve all
```

预期进入业务校验并回复当前有六个 REJECTED 冲突块，需要先执行 `/resolve retry`。RDS 中 task 仍为 `PARTIAL_REJECTED`，4 个 APPROVED、6 个 REJECTED、2 个 PENDING 全部不变，没有新增 `BLOCK_APPROVED`、`ALL_APPROVED` 或 outbox。

### 验证发起人可以 retry

账号 I 在 Resolve MR 主线评论：

```text
/resolve retry
```

预期命令被接受，只把 C0005 至 C0010 及各自 reason 交给 Coco。等待 retry 完成后确认：

- Resolve MR 不变，只新增 retry commit 和新版本。
- 飞书目录中新增一份重写后的 Markdown 文档，MR 描述更新为新链接。
- C0001 至 C0004 仍为 `APPROVED`。
- C0005 至 C0010 重置为 `PENDING`，reviewer、reviewed_at 和 comment 清空。
- C0011、C0012 始终为 `PENDING`。
- MR 描述为 `4 approved / 0 rejected / 8 pending`。
- task 回到 `AWAITING_APPROVAL`，`retry_count = 1`。
- task events 包含一条 `RETRY_REQUESTED`，其中 `requested_by` 是账号 I，rejected 集合恰好为 C0005 至 C0010。

这一步只复核 retry 的发起人权限和基本发布结果，不重复评估 Coco 每个重解方案的业务质量；详细 retry 语义仍按独立重解验收计划执行。

### 验证发起人可以 cancel

retry 发布完成后，账号 M 先执行一次 `/resolve status`。预期可以读取 `4/12 APPROVED，0 REJECTED，8 PENDING`，且 RDS 零写入。

账号 I 再执行：

```text
/resolve cancel
```

页面预期：

- Conciliator 在该评论线程中回复“任务已取消，分支与 MR 保留，系统不再响应”。
- MR、resolve 分支、已有 commit、Discussion 和飞书文档都保留。
- MR 描述中的任务状态更新为 `CANCELLED`，审批进度保持 `4 approved / 0 rejected / 8 pending`。

RDS 预期：

- `resolve_tasks.status = 'CANCELLED'`。
- `resolve_tasks.active_key IS NULL`。
- 十二条 approval 的状态和审批字段保持不变。
- 新增且只新增一条 `TASK_CANCELLED`，payload 保存取消前状态、账号 I、resolve branch 和 Resolve MR id。
- 不产生 `ALL_APPROVED` 和飞书通知 outbox。

取消后再评论 `/resolve status`，等待至少一个 webhook 处理周期。预期没有 Conciliator 回复；服务日志记录找不到活跃任务，RDS 不再变化。

## RDS 核对方法

从两个原始 MR number 定位 task：

```sql
SELECT id, original_mr_id, new_mr_id, status, active_key, version,
       mr_author_email, trigger_author_email, retry_count,
       new_mr_url, lark_doc_url
FROM resolve_tasks
WHERE repo_id = '1106267'
  AND original_mr_id IN (<任务A原始MR>, <任务B原始MR>)
ORDER BY id;

SELECT cb.task_id, cb.block_idx, cb.conflict_id, cb.discussion_id,
       cb.source_author_email, cb.target_author_email,
       a.status, a.reviewer_email, a.comment, a.reviewed_at, a.version
FROM conflict_blocks cb
JOIN approvals a ON a.block_id = cb.id
WHERE cb.task_id IN (<任务A task id>, <任务B task id>)
ORDER BY cb.task_id, cb.block_idx;

SELECT task_id, event_type, payload, occurred_at
FROM task_events
WHERE task_id IN (<任务A task id>, <任务B task id>)
ORDER BY task_id, id;

SELECT task_id, notification_type, recipient_email, mr_url,
       status, attempts, message_id, last_error
FROM lark_notification_outbox
WHERE task_id IN (<任务A task id>, <任务B task id>)
ORDER BY task_id, id;
```

这些查询只读，不提交 DML 或 DDL 工单。每次权限拒绝、重复 `/resolve` 和 rejected 拦截前后都要保存查询快照，不能只看最终状态。

## 服务日志核对

按原始 MR、Resolve MR、task id 和评论时间窗口检索 TCE 日志，至少确认：

- 第二次 `/resolve` 命中 `trigger_dropped_active_exists`，随后成功记录 `active_task_reply_created`。
- 没有第二条 `task_created` 对应同一个原始 MR。
- 方括号指令进入 approval/reject 处理，没有被记录为 ignored comment。
- 三次账号 M 的全局写指令都没有进入 approve-all、retry 或 cancel 仓储调用。
- 账号 I 的 approve-all、retry 和 cancel 分别产生对应 task event。
- Conciliator 自己发送的所有回复均命中应用评论忽略逻辑，没有产生回声。

日志只能证明服务执行分支，不能替代 Codebase 页面和 RDS 的结果核对。

## 通过标准

以下条件全部满足才算通过：

- TCE Pod 确认运行包含三个功能提交的版本，`healthz`、`readyz` 均正常。
- 重复 `/resolve` 在原评论下方得到明确提示，且没有第二条 task 或外部产物。
- 四种通过别名和六种拒绝别名全部驱动正确的 Discussion 回复与 RDS 状态，reason 提取准确。
- 账号 I 不是 block author 时仍能执行单块指令；账号 M 不是 block author、也不是发起人时被拒绝。
- 账号 M 可以执行 `status`，但不能执行 `approve all`、`retry` 或 `cancel`。
- 账号 I 可以执行 `approve all`、`retry` 和 `cancel`，各自状态转换与副作用正确。
- 存在 REJECTED 块时，`approve all` 明确报错，不误改 task、approval、event 或 outbox。
- 任务 A 全部通过后，Resolve MR 完成评论只 @账号 I，飞书 outbox 和私信收件人也是账号 I。
- 任务 B 取消后进入 `CANCELLED`、释放 `active_key`、保留外部产物，并停止响应后续命令。
- Codebase、TCE 和 RDS 的 task id、原始 MR、Resolve MR、操作者邮箱、状态和事件可以一一对应。

本轮不验证仓库 admin 权限；当前权限模型不再把 admin 作为全局写指令的授权角色。测试 MR 保持未合入状态，完成后按测试资源清理流程处理分支、MR 和飞书文档。
