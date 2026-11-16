from collections.abc import Sequence
from datetime import UTC
from zoneinfo import ZoneInfo

from conciliator.domain.task import TERMINAL_STATUSES, TaskStatus
from conciliator.repositories.models import ResolveTask
from conciliator.utils.task_links import task_trace_url

TASK_STATUS_LABELS: dict[str, str] = {
    TaskStatus.PENDING: "等待处理",
    TaskStatus.PREPARING: "准备中",
    TaskStatus.BRANCH_CREATED: "分支已创建",
    TaskStatus.CONFLICTS_FETCHED: "冲突信息已获取",
    TaskStatus.COCO_SUBMITTED: "已提交 Coco",
    TaskStatus.COCO_RUNNING: "Coco 处理中",
    TaskStatus.COCO_PROCESSING: "正在处理 Coco 结果",
    TaskStatus.DOC_CREATED: "报告已创建",
    TaskStatus.MR_CREATED: "解决冲突的 MR 已创建",
    TaskStatus.AWAITING_APPROVAL: "等待冲突块审批",
    TaskStatus.PARTIAL_REJECTED: "部分冲突块被拒绝",
    TaskStatus.RETRYING: "等待重新处理",
    TaskStatus.RETRY_PREPARING: "准备重新处理",
    TaskStatus.RETRY_COCO_SUBMITTED: "重解已提交 Coco",
    TaskStatus.RETRY_COCO_RUNNING: "Coco 重新处理中",
    TaskStatus.RETRY_COCO_PROCESSING: "正在处理 Coco 重解结果",
    TaskStatus.RETRY_RESULT_VERIFIED: "重解结果已验证",
    TaskStatus.RETRY_DOC_UPLOADED: "重解报告已上传",
    TaskStatus.RETRY_PUBLISHING: "正在发布重解结果",
    TaskStatus.ALL_APPROVED: "需要人工审批的冲突块已全部批准",
    TaskStatus.CANCELLED: "已取消",
    TaskStatus.COCO_TIMEOUT: "Coco 处理超时",
    TaskStatus.FAILED: "处理失败",
}
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def original_mr_status_reply(tasks: Sequence[ResolveTask]) -> str:
    if not tasks:
        return "该 MR 暂无解冲突任务，可评论 `/resolve` 发起。"
    latest_finished = len(tasks) == 1 and tasks[0].status in TERMINAL_STATUSES
    lines = []
    if latest_finished:
        lines.append("当前没有进行中的任务。")
    elif len(tasks) > 1:
        lines.append(f"当前 MR 有 {len(tasks)} 个活动任务：")
    for task in sorted(tasks, key=lambda item: item.id, reverse=True):
        label = TASK_STATUS_LABELS.get(task.status)
        if task.status == TaskStatus.ALL_APPROVED and (task.trigger_payload or {}).get(
            "_blank_approval_publication_verified"
        ):
            label = "纯空行冲突已处理，无需人工审批"
        state = f"{label}（`{task.status}`）" if label else f"`{task.status}`"
        prefix = "最近一次任务" if latest_finished else "任务"
        updated_at = task.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        lines.extend([
            f"[{prefix} {task.id}]({task_trace_url(task.id)})：{state}",
            f"处理版本：source `{(task.source_commit_sha or '未记录')[:8]}` → "
            f"target `{(task.target_commit_sha or '未记录')[:8]}`",
            f"最后更新：{updated_at.astimezone(_SHANGHAI):%Y-%m-%d %H:%M:%S}（北京时间）",
        ])
        if task.new_mr_url:
            lines.append(f"[解决冲突的 MR](<{task.new_mr_url}>)")
        lines.append("")
    return "\n".join(lines).rstrip()
