"""Demo bot cards: reuse task transitions, never expose the task action token."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import time
from typing import TYPE_CHECKING, Any

from conciliator.domain.aeolus_task import (
    AeolusState,
    AeolusTask,
    can_retry_coding,
    can_retry_completion,
    current_verification_targets,
)
from conciliator.integrations.coco.aeolus import PLATFORM_ERROR_PREFIX
from conciliator.integrations.lark.contacts import get_user_open_id_by_email
from conciliator.integrations.meegle.ai_node import AINodeRef, MeegleAINodeClient
from conciliator.services.aeolus_registry import InvalidTaskTransition, TaskNotFound
from conciliator.services.deployment_counts import devbox_result_kind
from conciliator.services.ppe_deployment_service import ppe_result_kind
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import error_fields, workflow_step
from conciliator.utils.redaction import safe_text

if TYPE_CHECKING:
    from conciliator.integrations.lark.client import LarkClient
    from conciliator.services.aeolus_service import AeolusService

log = get_logger(__name__)
NOTIFIABLE = {
    AeolusState.EVALUATION_REVIEW,
    AeolusState.VALIDATION_FAILED,
    AeolusState.REVIEW,
    AeolusState.CODING_BLOCKED,
    AeolusState.FAILED,
    AeolusState.COMPLETED,
    AeolusState.CODING,
    AeolusState.RECODING,
    AeolusState.COMPLETING,
}
LABELS = {
    AeolusState.EVALUATION_REVIEW: "输入评估待确认",
    AeolusState.VALIDATION_FAILED: "输入评估未通过",
    AeolusState.REVIEW: "编码结果待确认",
    AeolusState.CODING_BLOCKED: "编码阻塞，待人工处理",
    AeolusState.CODING: "正在编码",
    AeolusState.RECODING: "正在根据反馈修改",
    AeolusState.COMPLETING: "正在完成节点",
    AeolusState.COMPLETED: "节点已完成",
    AeolusState.FAILED: "处理失败，请在 Meegle 查看详情",
}
# Demo space mapping confirmed by the user-provided work item URL.
# Other spaces stay plain text until their actual URL prefix is known.
WORK_ITEM_URL_PREFIXES = {
    "6aabe23024218882059f74a4": "https://meego.larkoffice.com/4ebse5/story/detail/",
}


class InvalidCardCallback(ValueError):
    """Untrusted callback; keep its body out of logs and error responses."""


class AeolusLarkService:
    def __init__(
        self,
        client: LarkClient,
        *,
        app_id: str,
        meegle: MeegleAINodeClient,
        verification_token: str,
        encrypt_key: str = "",
    ) -> None:
        self.client = client
        self.app_id = app_id
        self.meegle = meegle
        self.token = verification_token
        self.encrypt_key = encrypt_key

    @staticmethod
    def _ref(task: AeolusTask) -> AINodeRef:
        return AINodeRef(task.project_key, task.work_item_id, task.node_uuid, task.user_key)

    @staticmethod
    def _task_id_footer(task: AeolusTask) -> dict:
        return {"tag": "markdown", "content": f"<font color='grey'>任务 ID：`{task.task_id}`</font>"}

    def _value(self, task: AeolusTask, action: str, owner_key: str, recipient: str) -> dict:
        value: dict[str, Any] = {
            "task_id": task.task_id,
            "state": task.state.value,
            "evaluation_round": len(task.evaluation_rounds),
            "coding_round": len(task.rounds),
            "action": action,
            "owner_key": owner_key,
            "recipient": recipient,
        }
        ppe = task.ppe_deployment
        if action == "confirm" and (
            task.deployment and task.deployment.get("resource_owned") or ppe and not ppe.get("released")
        ):
            value["release_resources"] = True
        if action == "complete_review":
            value["verification_targets"] = [
                {"target": item.target, "deployment_id": item.deployment_id}
                for item in current_verification_targets(task)
            ]
        if action in {"confirm_verified", "complete_review", "release_environment"} and task.deployment:
            value["deployment_id"] = task.deployment["deployment_id"]
            value["generation"] = task.deployment["generation"]
        if action == "deploy":
            round_ = task.rounds[-1] if task.rounds else None
            value.update(
                round_no=round_.round_no if round_ else None,
                commit_sha=round_.commit_sha if round_ else "",
                deployment_id=task.deployment.get("deployment_id") if task.deployment else None,
            )
        if action in {"deploy_ppe", "retry_ppe", "release_ppe", "complete_review"} and task.ppe_capability:
            round_ = task.rounds[-1] if task.rounds else None
            value.update(
                ppe_deployment_id=ppe.get("deployment_id") if ppe else None,
                ppe_environment_id=ppe.get("environment_id") if ppe else None,
                ppe_generation=ppe.get("generation") if ppe else None,
                ppe_profile_revision=task.ppe_capability["profile_revision"],
                ppe_round_no=round_.round_no if round_ else None,
                ppe_commit_sha=round_.commit_sha if round_ else "",
            )
            if action == "deploy_ppe":
                value["ppe_runtime_profiles"] = {
                    name: profile["profile_revision"]
                    for name, profile in task.ppe_capability.get("runtime_profiles", {}).items()
                    if profile.get("compatible") is not False
                }
        value["signature"] = hmac.new(
            task.action_token.encode(),
            json.dumps(value, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()
        return value

    def card(
        self, task: AeolusTask, owner_key: str, recipient: str,
        *, evaluation_snapshot: dict | None = None, selected_worker_id: str | None = None,
    ) -> dict:
        # source: https://open.feishu.cn/document/feishu-cards/card-json-v2-components/interactive-components/button
        def button(
            action: str, label: str, *, submit: bool = False, name: str = "submit_feedback", style: str = "default",
        ) -> dict:
            result: dict[str, Any] = {
                "tag": "button",
                "type": style,
                "text": {"tag": "plain_text", "content": label},
                "behaviors": [{"type": "callback", "value": self._value(task, action, owner_key, recipient)}],
            }
            if submit:
                result.update(form_action_type="submit", name=name)
            return result

        def select(name: str, placeholder: str, options: list[tuple[str, str]]) -> dict:
            # source: official single-select-dropdown-menu and form-container JSON 2.0 specifications
            return {
                "tag": "select_static", "name": name, "required": True,
                "placeholder": {"tag": "plain_text", "content": placeholder},
                "options": [
                    {"text": {"tag": "plain_text", "content": label}, "value": value}
                    for label, value in options
                ],
            }

        def text(content: str) -> dict:
            # source: https://open.feishu.cn/document/feishu-cards/card-json-v2-components/content-components/rich-text
            return {"tag": "markdown", "content": content}

        def runtime_evidence(detection: dict | None) -> list[dict]:
            if not detection:
                return []
            lines = [f"识别提交：`{detection.get('commit_sha', '')[:12]}`"]
            lines.extend(f"{item['path']}：{item['detail']}" for item in detection.get("evidence", []))
            lines.extend(detection.get("warnings", []))
            if detection.get("branch_hints"):
                lines.append("基础分支提示：" + "、".join(detection["branch_hints"]) + "（仅供参考）")
            return [text(safe_text("\n".join(lines)))]

        def panel(title: str, items: list[dict], *, expanded: bool = False, color: str = "blue") -> dict:
            # source: https://open.feishu.cn/document/feishu-cards/card-json-v2-components/containers/collapsible-panel
            # Form containers must remain at the card root, never inside this panel.
            caption = title if expanded else f"{title} · 展开查看"
            header: dict[str, Any] = {
                "title": {"tag": "markdown", "content": f"<font color='{color}'>**{caption}**</font>"},
            }
            if not expanded:
                header.update(background_color="grey", padding="6px 8px 6px 8px")
            return {"tag": "collapsible_panel", "expanded": expanded, "header": header, "elements": items}

        def detail(title: str, content: str) -> dict:
            # Short content reads better inline; count non-empty Markdown lines, not blank spacing.
            short = len(content.strip()) <= 160 and sum(bool(line.strip()) for line in content.splitlines()) <= 3
            if short:
                return text(f"**{title}**\n{content}")
            return panel(title, [text(content)])

        work_item = f"工作项 {task.work_item_id}"
        url_prefix = WORK_ITEM_URL_PREFIXES.get(task.project_key)
        if url_prefix:
            work_item = f"[{work_item}]({url_prefix}{task.work_item_id})"
        latest_evaluation = task.evaluation_rounds[-1] if task.evaluation_rounds else {}
        evaluation_error = latest_evaluation.get("error") or task.last_error
        evaluation_failed = task.state == AeolusState.VALIDATION_FAILED and bool(
            evaluation_error or latest_evaluation.get("status") == "error"
        )
        platform_failed = evaluation_failed and (
            latest_evaluation.get("error_kind") == "platform"
            # Persisted errors are stored as "<ExceptionType>: <message>", so the
            # prefix can appear after the type name; rounds written before
            # ``error_kind`` existed only carry that string.
            or PLATFORM_ERROR_PREFIX in str(evaluation_error or "")
        )
        stage_label = "评估执行失败" if evaluation_failed else LABELS.get(task.state, "处理中")
        round_labels = [
            f"评估第 {len(task.evaluation_rounds)} 轮" if task.evaluation_rounds else "尚未开始评估",
            f"Coding 第 {len(task.rounds)} 轮" if task.rounds else "尚未开始 Coding",
        ]
        elements = [
            text(
                f"{work_item} · {stage_label}\n"
                + " / ".join(round_labels)
            )
        ]
        next_steps = {
            AeolusState.CODING: "正在依据已确认需求编码；完成后会推送代码 Review 卡片。",
            AeolusState.RECODING: "正在处理本轮修改建议；新代码完成后可手动重新部署，旧部署不代表最新代码。",
            AeolusState.CODING_BLOCKED: "请 Review 已有结果并提供处理意见；阻塞解决前不能完成节点或新发起部署。",
            AeolusState.COMPLETING: "正在同步节点完成状态；环境释放以 worker 停服确认为准。",
            AeolusState.COMPLETED: "代码结果已由用户确认；需求共识和部署记录可按需展开查看。",
        }
        if task.state in next_steps:
            next_step = next_steps[task.state]
            if task.state == AeolusState.COMPLETING and task.ppe_deployment:
                next_step = "正在同步节点完成状态；PPE 释放以平台资源回查为准。"
            elements.append(text(next_step))
        evaluation = evaluation_snapshot if evaluation_snapshot is not None else (task.evaluation or {})
        evaluation_start = len(elements)
        if evaluation_snapshot is not None or task.state in {
            AeolusState.EVALUATION_REVIEW, AeolusState.VALIDATION_FAILED,
        }:
            if evaluation_failed:
                elements.append(text(
                    "**失败原因**\n"
                    f"{safe_text(evaluation_error or '未记录具体异常，请联系服务维护者检查后端日志。')[:1000]}"
                ))
                if platform_failed:
                    retried = bool(latest_evaluation.get("platform_retry_attempted"))
                    elements.append(text(
                        "本次失败由 Coco 平台侧导致（资源受限，或运行未产生任何输出），"
                        "与需求文档、仓库配置和文档权限无关。"
                        + (
                            "服务已按退避策略自动重试一次；若仍失败，请稍后再次点击重试。"
                            if retried else
                            "本次没有执行自动重试，请稍后再次点击重试。"
                        )
                    ))
                else:
                    elements.append(text(
                        "本轮未完成评估，不代表需求不合理。请根据失败原因检查文档访问权限、"
                        "Agent 输出或仓库配置；修复后可直接重试，无需强制修改需求文档。"
                    ))
            else:
                summary = str(evaluation.get("summary") or "暂无评估结论")[:3000]
                elements.append(text(summary[:400] + ("…" if len(summary) > 400 else "")))
                if len(summary) > 400:
                    elements.append(detail("完整评估结论", summary))
            feedback = latest_evaluation.get("feedback")
            if feedback:
                elements.append(detail("本轮评估输入反馈", str(feedback)[:2000]))
            if evaluation.get("missing_items"):
                items = "\n".join(f"- {item}" for item in evaluation["missing_items"])[:2000]
                elements.append(text(f"**待补充 / 阻断问题**\n{items}"))
            contract = evaluation.get("requirement_contract")
            if contract:
                elements.append(detail("已确认需求共识" if evaluation_snapshot is not None else "需求共识 · 待确认",
                                       str(contract)[:5000]))
            elif evaluation_failed:
                for previous in reversed(task.evaluation_rounds):
                    previous_contract = (previous.get("result") or {}).get("requirement_contract")
                    if previous_contract:
                        elements.append(detail(
                            f"上次需求共识（第 {previous.get('round_no', '?')} 轮，非本轮结论）",
                            str(previous_contract)[:5000],
                        ))
                        break
            if evaluation.get("risks"):
                items = "\n".join(f"- {item}" for item in evaluation["risks"])[:2000]
                elements.append(detail("风险提示", items))
            if not evaluation_failed and evaluation.get("valid") is False:
                elements.append(text("请补充上述阻断问题后提交反馈，或更新需求文档后重新评估。"))
        if len(elements) > evaluation_start:
            result = elements[evaluation_start:]
            title = "评估执行失败" if evaluation_failed else (
                "需求文档需要补充" if evaluation.get("valid") is False else "需求评估结果"
            )
            elements[evaluation_start:] = [panel(title, result, expanded=evaluation_snapshot is None)]
        coding_start = len(elements)
        coding_details: list[dict] = []
        if task.rounds and task.state in {
            AeolusState.CODING, AeolusState.RECODING, AeolusState.REVIEW, AeolusState.CODING_BLOCKED,
            AeolusState.COMPLETING, AeolusState.COMPLETED, AeolusState.FAILED,
        }:
            round_ = task.rounds[-1]
            elements.append(text(f"**第 {round_.round_no} 轮编码 · {LABELS[task.state]}**"))
            if round_.feedback:
                target = (
                    f"（针对第 {round_.reviewed_round_no} 轮）"
                    if round_.reviewed_round_no is not None else ""
                )
                coding_details.append(detail(f"用户反馈{target}", round_.feedback[:2000]))
            if round_.summary:
                summary = round_.summary[:3000]
                elements.append(text(f"**处理结果**\n{summary[:400]}" + ("…" if len(summary) > 400 else "")))
                if len(summary) > 400:
                    coding_details.append(detail("完整编码结果", summary))
            if round_.unresolved_items:
                items = "\n".join(f"{index}. {item}" for index, item in enumerate(round_.unresolved_items, 1))[:3000]
                elements.append(text(f"**阻塞 / 未完成项 · {len(round_.unresolved_items)} 项**\n{items}"))
            if round_.failure_code:
                elements.append(text(f"**错误码**：{safe_text(round_.failure_code)[:200]}"))
            if round_.commit_sha:
                elements.append(text(f"**代码提交**：`{round_.commit_sha[:12]}`"))
                coding_details.append(text(f"**完整代码提交**：`{round_.commit_sha}`"))
            if round_.tests:
                tests = "\n".join(f"- {item}" for item in round_.tests)[:2000]
                coding_details.append(detail("测试记录（Agent 报告，非 CI 状态）", tests))
            if task.state == AeolusState.FAILED:
                error = round_.error or task.last_error
                if error:
                    elements.append(text(f"**失败原因**\n{safe_text(error)[:1000]}"))
        if task.state == AeolusState.FAILED and not task.rounds and task.last_error:
            elements.append(text(f"**失败原因**\n{safe_text(task.last_error)[:1000]}"))
        code_url = task.mr_url or (task.rounds[-1].code_url if task.rounds else "")
        if code_url.startswith("https://code.byted.org/"):
            elements.append(
                {
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": "Review 已有代码" if task.state == AeolusState.CODING_BLOCKED else "查看代码 MR",
                    },
                    "behaviors": [{"type": "open_url", "default_url": code_url}],
                }
            )
        if task.state in {AeolusState.REVIEW, AeolusState.COMPLETED} and evaluation.get("requirement_contract"):
            coding_details.append(detail(
                "查看编码依据 · 已确认需求共识", str(evaluation["requirement_contract"])[:5000],
            ))
        if len(elements) > coding_start:
            elements[coding_start:] = [panel("代码结果" if task.state in {AeolusState.REVIEW, AeolusState.COMPLETED}
                                            else "本轮编码进度", elements[coding_start:], expanded=True)]
        completion_elements: list[dict] = []
        feedback_elements: list[dict] = []
        deployment = task.deployment
        deployment_elements: list[dict] = []
        deployment_controls: list[dict] = []
        deployment_details: list[dict] = []
        ppe_items: list[dict] = []
        ppe_details: list[dict] = []
        def machine_label(worker, *, detailed: bool = False):
            counts = worker.get("deployment_counts")
            if counts is None:
                return worker["display_name"]
            if not detailed:
                return f"{worker['display_name']} · 在线 · 部署 {counts['total']}"
            return (f"{worker['display_name']} · 部署 {counts['total']} · 运行 {counts['running']}"
                    f" · 部署中 {counts['deploying']} · 停止待确认 {counts['stopping']}"
                    f" · 未知 {counts['unknown']} · 清理待处理 {counts['cleanup_pending']}")
        if deployment:
            deployment_status = deployment.get("status_label", deployment["status"])
            if deployment.get("current"):
                deployment_status = "已验证" if deployment.get("verified") else "待验证"
            deployment_elements.append(text(f"<font color='blue'>**Devbox · {deployment_status}**</font>"))
            if deployment.get("error"):
                deployment_elements.append(text("**部署错误**\n" + safe_text(deployment["error"])))
            if deployment.get("status_note"):
                deployment_elements.append(text(deployment["status_note"]))
            if deployment.get("resource_releasing"):
                deployment_elements.append(text("释放请求已提交，等待 worker 确认；无需重复点击。"))
            if task.state == AeolusState.COMPLETED:
                deployment_elements.append(text(
                    "验证结果：" + ("已记录业务验证通过" if deployment.get("verified") else "未记录业务验证通过")
                ))
            deployment_elements.append(text(
                f"{deployment.get('display_name', '')} · 提交 `{deployment['commit_sha'][:12]}`"
            ))
            deployment_details = [text(f"完整部署提交：`{deployment['commit_sha']}`")]
            if deployment.get("log_path"):
                deployment_details.append(text(f"部署日志：`{deployment['log_path']}`"))
            selected = next((worker for worker in task.deployment_workers or []
                             if worker["worker_id"] == deployment["worker_id"]), None)
            if selected and selected.get("deployment_counts"):
                deployment_details.append(text(machine_label(selected, detailed=True)))
                if not selected.get("online"):
                    deployment_elements.append(text("非实时，上次统计：" + selected.get("counts_observed_at", "未知")))
            if deployment.get("runtime_profile_id"):
                deployment_details.append(text(
                    f"运行配置：{deployment['runtime_profile_id']} · Python {deployment.get('python_version', '')}"
                    f" · {deployment.get('runtime_revision', '')}"
                ))
            if deployment.get("current") or (
                deployment.get("stage") == "maat_entry_pending" and deployment.get("resource_owned")
                and not deployment.get("resource_releasing") and not deployment.get("superseded")
            ):
                deployment_elements.append({
                    "tag": "button", "type": "primary", "text": {"tag": "plain_text", "content": (
                        "打开 MAAT" if deployment.get("current") else "打开 MAAT（入口待验证）"
                    )},
                    "behaviors": [{"type": "open_url", "default_url": deployment["maat_url"]}],
                })
                if not deployment.get("current"):
                    deployment_elements.append(text("入口尚未验证，暂不能确认验证通过。"))
            if deployment.get("can_release"):
                deployment_details.append(button("release_environment", "释放 Devbox"))
        ppe = task.ppe_deployment
        if ppe:
            ppe_status = ppe["status_label"]
            if (ppe["status"] == "deployed" and ppe.get("can_verify")
                    and not ppe.get("superseded") and not ppe.get("released")
                    and not ppe.get("release_requested")):
                ppe_status = "已验证" if ppe.get("verified") else "待验证"
            ppe_items.append(text(f"<font color='purple'>**PPE · {ppe_status}**</font>"))
            current = ppe.get("current_deployment")
            if current and not ppe.get("released") and not ppe.get("release_requested") and (
                current["deployment_id"] != ppe["deployment_id"] or ppe.get("superseded")
            ):
                ppe_items.append(text(
                    "**⚠ 保留上一版 PPE 环境，本轮更新尚未完成**\n"
                    f"上一版就绪镜像：**{current.get('image_version') or '未知'}**\n"
                    f"上一版提交：`{current['commit_sha'][:12]}`\n"
                    "请等待目标版本部署完成后再验证。"
                ))
            if ppe.get("error"):
                ppe_items.append(text("**部署错误**\n" + safe_text(ppe["error"])))
            for approval in ppe.get("approvals", []):
                ppe_items.append(text(
                    f"{approval['cluster_name']} 等待 TCE 资源审批，工单 {approval['tce_ticket_id']}。"
                ))
            version_label = "部署" if ppe["status"] == "deployed" and not ppe.get("superseded") else "目标"
            ppe_items.append(text(
                f"{version_label}提交：`{ppe['commit_sha'][:12]}`\n"
                f"{version_label}镜像：{ppe.get('image_version') or '待构建'}"
            ))
            if ppe.get("can_view_environment"):
                ppe_items.append({
                    "tag": "button", "type": "primary",
                    "text": {"tag": "plain_text", "content": "查看 PPE 环境"},
                    "behaviors": [{"type": "open_url", "default_url": ppe["environment_url"]}],
                })
                if ppe.get("can_access_environment"):
                    ppe_items.append(text(f"沿用 Aeolus 前端；请求头 X-TT-ENV：`{ppe['environment_name']}`"))
                else:
                    ppe_items.append(text("环境已创建，可查看部署进度；实例核验就绪后再进行业务验证。"))
            ppe_details = [text(
                f"环境：`{ppe['environment_name']}`\n"
                f"完整目标提交：`{ppe['commit_sha']}`\nSCM：{ppe.get('scm_version') or '待构建'}"
            )]
            if current:
                ppe_details.append(text(f"上一版完整提交：`{current['commit_sha']}`"))
            if ppe.get("can_access_environment"):
                ppe_details.append(text(f"**环境访问方式（更新时保持不变）**\n{ppe['access_hint']}"))
            for cluster in ppe.get("clusters", []):
                service_name = {"data.vqs.api": "VQS · ", "data.aeolus.api_v3": "Aeolus · "}.get(cluster.get("psm"), "")
                ppe_details.append(text(
                    f"{service_name}{cluster['name']} · 镜像 {cluster.get('image_version') or '未知'} · "
                    f"{'实例已就绪' if cluster.get('ready') else '未就绪'}"
                ))
            build_links = [
                f"[{label}]({ppe[key]})"
                for key, label in (
                    ("bits_run_url", "BITS 构建运行"),
                    ("scm_url", "SCM 制品"),
                    ("image_url", "ICM 镜像构建"),
                )
                if ppe.get(key)
            ]
            if build_links:
                ppe_details.append(text(" · ".join(build_links)))
            if ppe.get("due_time"):
                ppe_details.append(text(f"平台租期：{ppe['due_time']}"))
            if ppe.get("can_release"):
                ppe_details.append(button("release_ppe", "释放 PPE"))
            if ppe.get("can_retry") and task.state == AeolusState.REVIEW:
                ppe_items.append(button("retry_ppe", "重试 PPE 部署"))
        actions = []
        if task.state == AeolusState.EVALUATION_REVIEW and evaluation.get("valid") is True:
            actions.append(button("confirm_evaluation", "确认需求，开始编码", style="primary"))
        if task.state == AeolusState.REVIEW:
            if (task.ppe_capability and task.ppe_capability.get("enabled")
                    and task.rounds and task.rounds[-1].commit_sha
                    and (not ppe or ppe.get("can_deploy"))):
                if not ppe:
                    ppe_items.append(text("<font color='purple'>**PPE · 未部署**</font>"))
                capability = task.ppe_capability
                profiles = capability.get("runtime_profiles", {})
                recommended = capability.get("recommended_runtime_profile_id")
                options = [
                    (f"{name} · Python {profile['python_version']}"
                     + (" · 推荐" if name == recommended else ""),
                     "::".join((name, profile["profile_revision"])))
                    for name, profile in profiles.items() if profile.get("compatible") is not False
                ]
                if options:
                    runtime_select = select("ppe_runtime_profile_id", "请选择 PPE 运行配置", options)
                    if (isinstance(recommended, str) and recommended in profiles
                            and profiles[recommended].get("compatible") is not False):
                        runtime_select["initial_option"] = "::".join((
                            recommended, profiles[recommended]["profile_revision"],
                        ))
                    ppe_items.append({
                        "tag": "form", "name": "ppe_deployment_form", "elements": [
                            runtime_select,
                            button("deploy_ppe", "一键部署 PPE · LF / IPv6",
                                   submit=True, name="submit_ppe_deployment"),
                        ],
                    })
                else:
                    ppe_items.append(text("暂无与当前代码兼容的 PPE 运行配置，请检查声明与依赖。"))
                ppe_items.extend(runtime_evidence(capability.get("runtime_detection")))
            workers = task.deployment_workers
            if workers is not None and task.rounds and task.rounds[-1].commit_sha:
                if not deployment:
                    deployment_elements.append(text("<font color='blue'>**Devbox · 未部署**</font>"))
                executing = deployment and deployment["status"] in {"running", "unknown", "cancel_requested"}
                candidates = [w for w in workers if w.get("online") and w.get("supports_repository", True)
                              and not (deployment and deployment.get("resource_releasing"))]
                if not executing and candidates:
                    deployment_controls.append(text("**重新部署设置**" if deployment else "**部署设置**"))
                    selected = next((w for w in candidates if w["worker_id"] == selected_worker_id), candidates[0])
                    machine_select = select("worker_id", "请选择 Devbox", [
                        (machine_label(worker), worker["worker_id"]) for worker in candidates
                    ])
                    machine_select.pop("required")  # This selector is outside the submit form.
                    machine_select.update(
                        initial_option=selected["worker_id"],
                        behaviors=[{"type": "callback", "value": self._value(
                            task, "choose_deployment_worker", owner_key, recipient,
                        )}],
                    )
                    deployment_controls.append(machine_select)
                    profiles = selected.get("runtime_profiles", {})
                    options = [
                        (f"{profile_id} · Python {profile['python_version']} · {profile.get('runtime_revision', '')}"
                         + (" · 推荐" if profile_id == selected.get("recommended_runtime_profile_id") else ""),
                         "::".join((selected["worker_id"], profile_id, profile.get("profile_revision", ""))))
                        for profile_id, profile in profiles.items() if profile.get("compatible") is not False
                    ]
                    runtime_select = select("runtime_profile_id", "请选择 Python / 依赖配置", options)
                    recommended = selected.get("recommended_runtime_profile_id")
                    if (isinstance(recommended, str) and recommended in profiles
                            and profiles[recommended].get("compatible") is not False):
                        runtime_select["initial_option"] = "::".join((
                            selected["worker_id"], recommended, profiles[recommended].get("profile_revision", ""),
                        ))
                    if options:
                        deployment_controls.append({
                            "tag": "form", "name": "deployment_form", "elements": [
                                runtime_select,
                                button("deploy", "部署当前代码", submit=True, name="submit_deployment"),
                            ],
                        })
                    else:
                        deployment_controls.append(text("所选 Devbox 暂无可用运行配置，请选择其他机器。"))
                    deployment_controls.extend(runtime_evidence(selected.get("runtime_detection")))
                    if selected.get("deployment_counts"):
                        deployment_controls.append(panel(
                            "所选机器部署统计", [text(machine_label(selected, detailed=True))],
                        ))
                elif executing:
                    deployment_controls.append(text("部署正在执行或停止中。"))
                elif not candidates:
                    deployment_controls.append(text("暂无可部署的在线 Devbox。"))
            owns_environment = (deployment and deployment.get("resource_owned") or ppe and not ppe.get("released"))
            completion_note = ""
            if targets := current_verification_targets(task):
                names = ["Devbox" if item.target == "devbox" else "PPE" for item in targets]
                completion_note = "验证通过将确认当前 " + " 和 ".join(names) + "。"
            if owns_environment:
                completion_note += "完成后释放环境。"
            completion_elements.append(text("**完成确认**" + ("\n" + completion_note if completion_note else "")))
            completion_elements.append({
                    "tag": "form", "name": "completion_form", "elements": [
                        # source: official JSON 2.0 column-set; stack on narrow screens.
                        {
                            "tag": "column_set", "flex_mode": "stretch",
                            "columns": [
                                {"tag": "column", "width": "auto", "elements": [
                                    select("verification_mode", "请选择验证结果", [
                                        *([("验证通过", "all_verified")] if targets else []),
                                        ("跳过验证", "skip"),
                                    ]),
                                ]},
                                {"tag": "column", "width": "auto", "elements": [
                                    button("complete_review", "完成节点并释放环境" if owns_environment
                                           else "确认代码，完成节点", submit=True,
                                           name="submit_completion", style="primary"),
                                ]},
                            ],
                        },
                    ],
            })
        if can_retry_completion(task):
            actions.append(button("retry_completion", "重试完成节点", style="primary"))
        if can_retry_coding(task):
            actions.append(button("retry_coding", "重试编码", style="primary"))
        evaluation_review = task.state in {
            AeolusState.EVALUATION_REVIEW,
            AeolusState.VALIDATION_FAILED,
        }
        if evaluation_review:
            retry_label = "重试评估" if evaluation_failed else "文档已更新，重新评估"
            actions.append(button(
                "retry_validation", retry_label,
                style="primary" if evaluation_failed or evaluation.get("valid") is False else "default",
            ))
        if evaluation_review or task.state in {AeolusState.REVIEW, AeolusState.CODING_BLOCKED}:
            feedback_elements.append(text("**补充需求**" if evaluation_review else "**需要修改？**"))
            # source: https://open.feishu.cn/document/feishu-cards/card-json-v2-components/containers/form-container
            feedback_elements.append(
                {
                    "tag": "form",
                    "name": "feedback_form",
                    "elements": [
                        {
                            "tag": "input",
                            "name": "feedback",
                            "max_length": 1000,
                            "required": True,
                            "placeholder": {"tag": "plain_text", "content": "填写意见；长反馈请在 Meego 填写"},
                        },
                        button(
                            "evaluation_feedback" if evaluation_review else "submit_feedback",
                            "提交并重新评估" if evaluation_review else (
                                "提交意见并继续 Coding"
                                if task.state == AeolusState.CODING_BLOCKED else "提交并重新 Coding"
                            ),
                            submit=True,
                        ),
                    ],
                }
            )
        # Keep forms at root and each environment's state and controls in one contiguous section.
        elements = (
            elements + completion_elements + actions + feedback_elements
            + deployment_elements + deployment_controls
            + ([panel("Devbox 详情", deployment_details)] if deployment else [])
            + ppe_items + ([panel("PPE 详情", ppe_details)] if ppe else [])
            + ([panel("编码详情", coding_details)] if coding_details else [])
            + [self._task_id_footer(task)]
        )
        color = "red" if evaluation_failed or task.state == AeolusState.FAILED else {
            AeolusState.EVALUATION_REVIEW: "orange", AeolusState.VALIDATION_FAILED: "orange",
            AeolusState.REVIEW: "orange", AeolusState.CODING_BLOCKED: "orange", AeolusState.COMPLETED: "green",
        }.get(task.state, "blue")
        return {
            "schema": "2.0",
            "header": {
                "template": color, "title": {"tag": "plain_text", "content": f"aeolus 编码助手 · {stage_label}"},
            },
            "body": {"elements": elements},
        }

    def ppe_result_card(self, task: AeolusTask, owner_key: str, recipient: str) -> dict:
        ppe = task.ppe_deployment
        assert ppe is not None
        kind = ppe_result_kind(ppe)
        success = kind == "success"
        title = {
            "success": "PPE 部署成功", "released": "PPE 释放完成",
            "failure": "PPE 部署失败", "release_failure": "PPE 释放失败",
            "unknown": "PPE 部署结果待确认", "release_unknown": "PPE 释放结果待确认",
        }.get(kind, "PPE 部署失败")
        prefix = WORK_ITEM_URL_PREFIXES.get(task.project_key)
        work_item = (
            f"[工作项 {task.work_item_id}]({prefix}{task.work_item_id})" if prefix else f"工作项 {task.work_item_id}"
        )
        summary = f"{work_item} · 编码第 {ppe.get('round_no', len(task.rounds))} 轮\n"
        summary += f"提交：`{ppe['commit_sha'][:8]}`"
        if ppe.get("image_version"):
            summary += f" · 镜像：{ppe['image_version']}"
        elements: list[dict[str, Any]] = [{"tag": "markdown", "content": summary}]
        if kind == "released":
            elements.append({"tag": "markdown", "content": "PPE 环境及所属资源已确认回收，无需继续释放。"})
        elif success and ppe.get("can_access_environment"):
            elements.append({
                "tag": "markdown",
                "content": f"环境：`{ppe['environment_name']}`\n请求头：`X-TT-ENV: {ppe['environment_name']}`\n"
                           "请使用测试账号验证，并在 Meego 卡片确认结果。",
            })
        elif not success:
            reason = safe_text(ppe.get("error") or "请打开 Meego 卡片查看详情")
            if len(reason) > 200:
                reason = reason[:200] + "…"
            elements.append({"tag": "markdown", "content": reason})
            if kind in {"release_failure", "release_unknown"}:
                elements.append({"tag": "markdown", "content": "资源尚未确认回收，请核对原工单和实际资源后再重试。"})
            elif ppe.get("current_deployment"):
                elements.append({"tag": "markdown", "content": "上一版 PPE 环境仍保留；本次更新尚未就绪。"})
        if kind != "released" and ppe.get("can_view_environment"):
            elements.append({
                "tag": "button", "text": {"tag": "plain_text", "content": "查看 PPE 环境"},
                "behaviors": [{"type": "open_url", "default_url": ppe["environment_url"]}],
            })
        if success and ppe.get("can_release"):
            elements.append({
                "tag": "button", "text": {"tag": "plain_text", "content": "释放 PPE"},
                "behaviors": [{"type": "callback", "value": self._value(task, "release_ppe", owner_key, recipient)}],
            })
        elements.append(self._task_id_footer(task))
        return {
            "schema": "2.0",
            "header": {"template": "green" if success or kind == "released" else
                       "orange" if ppe["status"] == "unknown" else "red",
                       "title": {"tag": "plain_text", "content": f"aeolus 编码助手 · {title}"}},
            "body": {"elements": elements},
        }

    def devbox_result_card(self, task: AeolusTask, owner_key: str, recipient: str) -> dict:
        deployment = task.deployment
        assert deployment is not None
        kind = devbox_result_kind(deployment)
        title = {
            "success": "Devbox 部署成功",
            "failure": "Devbox 部署失败",
            "unknown": "Devbox 部署结果待确认",
            "cleanup": "Devbox 释放待人工处理",
            "released": "Devbox 释放完成",
            "release_unknown": "Devbox 释放结果待确认",
        }[kind]
        prefix = WORK_ITEM_URL_PREFIXES.get(task.project_key)
        work_item = (
            f"[工作项 {task.work_item_id}]({prefix}{task.work_item_id})" if prefix else f"工作项 {task.work_item_id}"
        )
        summary = (
            f"{work_item} · 编码第 {deployment.get('round_no', len(task.rounds))} 轮\n"
            f"Devbox：{deployment.get('display_name') or deployment['worker_id']}"
            f" · 提交：`{deployment['commit_sha'][:8]}`"
        )
        elements: list[dict[str, Any]] = [{"tag": "markdown", "content": summary}]
        if kind == "success":
            elements.append({
                "tag": "markdown", "content": "部署已完成，请打开 MAAT 进行业务验证，再在 Meego 卡片确认结果。",
            })
            elements.append({
                "tag": "button", "text": {"tag": "plain_text", "content": "访问部署环境（MAAT）"},
                "behaviors": [{"type": "open_url", "default_url": deployment["maat_url"]}],
            })
            if deployment.get("can_release"):
                elements.append({
                    "tag": "button", "text": {"tag": "plain_text", "content": "释放 Devbox"},
                    "behaviors": [{
                        "type": "callback", "value": self._value(task, "release_environment", owner_key, recipient),
                    }],
                })
        elif kind == "released":
            elements.append({
                "tag": "markdown", "content": (
                    "服务和实例代码、MAAT 环境及代理记录已清理。旧端口继续保留预留。"
                    if deployment.get("stage") == "maat_cleanup_records_deleted"
                    else "服务和实例代码已清理，端口预留已释放。"
                ),
            })
        else:
            reason = safe_text(deployment.get("error") or "请在 Meego 卡片查看部署详情。")
            elements.append({"tag": "markdown", "content": reason[:200] + ("…" if len(reason) > 200 else "")})
            if kind in {"cleanup", "release_unknown"}:
                elements.append({"tag": "markdown", "content": "旧端口继续保留预留，需维护者核对资源后处理。"})
            elif kind == "failure" and deployment.get("resource_releasing"):
                elements.append({"tag": "markdown", "content": "正在释放本次部署；完成后会另行通知。"})
        elements.append(self._task_id_footer(task))
        return {
            "schema": "2.0",
            "header": {
                "template": "green" if kind in {"success", "released"} else "red" if kind == "failure" else "orange",
                "title": {"tag": "plain_text", "content": f"aeolus 编码助手 · {title}"},
            },
            "body": {"elements": elements},
        }

    @workflow_step("lark_notification", level="debug")
    async def notify(
        self, task: AeolusTask, *, strict: bool = False, ppe_only: bool = False, devbox_only: bool = False,
    ) -> None:
        if ppe_only:
            ppe = task.ppe_deployment
            if ppe_result_kind(ppe) is None:
                return
        elif devbox_only:
            if devbox_result_kind(task.deployment) is None:
                return
        elif task.state not in NOTIFIABLE:
            return
        ref = self._ref(task)
        owner_keys = await self.meegle.get_node_owner_keys(ref)
        if not owner_keys:
            log.warning("aeolus_lark_node_has_no_owner", task_id=task.task_id)
            if strict:
                raise RuntimeError("PPE notification recipient unavailable")
            return
        users = await self.meegle.query_users(ref, owner_keys)
        users_by_key = {user.get("user_key"): user for user in users}

        async def send_owner(owner_key: str) -> None:
            email = (users_by_key.get(owner_key) or {}).get("email")
            if not email:
                log.warning("aeolus_lark_owner_email_missing", task_id=task.task_id)
                if strict:
                    raise RuntimeError("PPE notification recipient unavailable")
                return
            try:
                recipient = await get_user_open_id_by_email(self.client, email)
                await self._send(task, owner_key, recipient, ppe_only=ppe_only, devbox_only=devbox_only)
            except Exception as exc:
                log.warning(
                    "aeolus_notification_failed", task_id=task.task_id, error_type=type(exc).__name__,
                    http_status=getattr(exc, "http_status", None), code=getattr(exc, "code", None),
                )
                if strict:
                    raise

        await asyncio.gather(*(send_owner(owner_key) for owner_key in owner_keys))

    @workflow_step("lark_card_send")
    async def _send(
        self, task: AeolusTask, owner_key: str, recipient: str, *, ppe_only: bool = False, devbox_only: bool = False,
    ) -> None:
        if ppe_only or devbox_only:
            deployment = task.ppe_deployment if ppe_only else task.deployment
            assert deployment is not None
            result = ppe_result_kind(deployment) if ppe_only else devbox_result_kind(deployment)
            target = "ppe" if ppe_only else "devbox"
            identity_source = (
                f"{task.task_id}:{deployment['deployment_id']}:{result}:"
                f"{owner_key}:{recipient}:{target}-result-layout-3"
            )
            if ppe_only and result in {"release_failure", "release_unknown"}:
                identity_source += f":{deployment.get('release_ticket_id') or ''}"
            identity = hashlib.sha256(identity_source.encode()).hexdigest()
            await self.client._request(
                "POST", "/im/v1/messages", params={"receive_id_type": "open_id"},
                json={
                    "receive_id": recipient, "msg_type": "interactive",
                    "content": json.dumps(
                        self.ppe_result_card(task, owner_key, recipient) if ppe_only
                        else self.devbox_result_card(task, owner_key, recipient), ensure_ascii=False,
                    ), "uuid": identity[:50],
                },
            )
            return
        # Same snapshot uses the same message UUID for Feishu's send deduplication.
        identity = self._value(task, "notification", owner_key, recipient)["signature"]
        if task.ppe_deployment:
            ppe = task.ppe_deployment
            identity = hashlib.sha256(f"{identity}:{ppe['deployment_id']}:{ppe['status']}:{ppe['stage']}:"
                                      f"{ppe.get('resource_state')}:{ppe.get('error')}:{ppe.get('status_label')}".encode()).hexdigest()
        if task.deployment:
            identity = hashlib.sha256(
                f"{identity}:{task.deployment['deployment_id']}:{task.deployment['status']}:"
                f"{task.deployment.get('stage')}:{task.deployment.get('maat_environment_status')}".encode()
            ).hexdigest()
        if task.rounds:
            identity = hashlib.sha256(f"{identity}:{len(task.rounds[-1].attempts)}".encode()).hexdigest()
        # A layout change must not deduplicate a requested new card against an old layout.
        identity = hashlib.sha256(
            f"{identity}:{task.deployment.get('resource_releasing') if task.deployment else ''}:"
            f"{task.deployment.get('released') if task.deployment else ''}:message-layout-9".encode()
        ).hexdigest()
        await self.client._request(
            "POST",
            "/im/v1/messages",
            params={"receive_id_type": "open_id"},
            json={
                "receive_id": recipient,
                "msg_type": "interactive",
                "content": json.dumps(self.card(task, owner_key, recipient), ensure_ascii=False),
                "uuid": identity[:50],
            },
        )

    def _decode(self, raw: bytes, headers) -> dict:
        # source: lark_oapi.event.dispatcher_handler.EventDispatcherHandler
        try:
            payload = json.loads(raw)
            if payload.get("encrypt"):
                from lark_oapi.core.utils import AESCipher

                if not self.encrypt_key:
                    raise ValueError("encrypt key required")
                payload = json.loads(AESCipher(self.encrypt_key).decrypt_str(payload["encrypt"]))
            token = payload.get("token") or payload.get("header", {}).get("token")
            if not self.token or not isinstance(token, str) or not hmac.compare_digest(token, self.token):
                raise ValueError("invalid token")
            if self.encrypt_key and payload.get("type") != "url_verification":
                timestamp = headers.get("x-lark-request-timestamp", "")
                nonce = headers.get("x-lark-request-nonce", "")
                expected = hashlib.sha256((timestamp + nonce + self.encrypt_key).encode() + raw).hexdigest()
                if abs(time.time() - int(timestamp)) > 300 or not hmac.compare_digest(
                    expected, headers.get("x-lark-signature", "")
                ):
                    raise ValueError("invalid signature")
            return payload
        except (ValueError, TypeError, KeyError, AttributeError):
            raise InvalidCardCallback("invalid card callback") from None

    async def callback(self, service: AeolusService, raw: bytes, headers) -> dict:
        payload = self._decode(raw, headers)
        if payload.get("type") == "url_verification":
            challenge = payload.get("challenge")
            if not isinstance(challenge, str) or not challenge:
                raise InvalidCardCallback("invalid challenge")
            return {"challenge": challenge}
        return await self.handle_event(service, payload)

    async def handle_event(self, service: AeolusService, payload: dict) -> dict:
        """Only HTTP-verified or app-authenticated WebSocket events may enter here."""
        header = payload.get("header", {})
        if not isinstance(header, dict):
            raise InvalidCardCallback("invalid event header")
        if header.get("event_type") != "card.action.trigger" or header.get("app_id") != self.app_id:
            raise InvalidCardCallback("unexpected card event")
        request_id = header.get("event_id")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise InvalidCardCallback("invalid event id")
        event = payload.get("event", {})
        if not isinstance(event, dict):
            raise InvalidCardCallback("invalid event")
        action = event.get("action", {})
        if not isinstance(action, dict) or not isinstance(event.get("operator"), dict):
            raise InvalidCardCallback("invalid operator or action")
        value = action.get("value", {})
        if not isinstance(value, dict):
            raise InvalidCardCallback("invalid action")
        try:
            task = await service.get(str(value.get("task_id", "")))
        except TaskNotFound:
            raise InvalidCardCallback("unknown task") from None
        recipient = event["operator"].get("open_id")
        owner_key = value.get("owner_key")
        if (
            not isinstance(recipient, str)
            or not recipient
            or not isinstance(owner_key, str)
            or not owner_key
            or value.get("recipient") != recipient
        ):
            raise InvalidCardCallback("unexpected operator")
        operation = value.get("action")
        allowed = {
            "confirm_evaluation", "retry_validation", "evaluation_feedback",
            "confirm", "confirm_verified", "complete_review", "deploy", "choose_deployment_worker",
            "release_environment", "submit_feedback",
            "retry_coding", "retry_completion",
            "deploy_ppe", "retry_ppe", "release_ppe",
        }
        if not isinstance(operation, str) or operation not in allowed:
            raise InvalidCardCallback("unknown action")
        return await self._handle_action(service, task, operation, request_id, value, action, owner_key, recipient)

    @workflow_step("lark_card_action")
    async def _handle_action(self, service, task, operation, request_id, value, action, owner_key, recipient):
        expected = self._value(task, operation, owner_key, recipient)
        if value != expected:
            log.info(
                "meegle_card_action_stale", workflow="meegle", task_id=task.task_id,
                action=operation, event_id=request_id, state=task.state.value,
            )
            return {"toast": {"type": "info", "content": "此卡片已过期，请使用最新卡片"}}
        log.info(
            "meegle_card_action_received", workflow="meegle", task_id=task.task_id,
            action=operation, event_id=request_id, state=task.state.value,
        )
        # A signed recipient was resolved at send time. Recheck current ownership
        # with one bounded API call, without repeating email/open_id queries in the callback.
        try:
            owners = await asyncio.wait_for(self.meegle.get_node_owner_keys(self._ref(task)), timeout=1.5)
        except Exception as exc:
            log.warning("aeolus_lark_owner_check_failed", task_id=task.task_id, error_type=type(exc).__name__)
            return {"toast": {"type": "error", "content": "暂时无法核实节点负责人，请稍后重试"}}
        if owner_key not in owners:
            raise InvalidCardCallback("operator is no longer a node owner")
        evaluation_review = task.state in {AeolusState.EVALUATION_REVIEW, AeolusState.VALIDATION_FAILED}
        if (
            (operation in {"retry_validation", "evaluation_feedback"} and not evaluation_review)
            or (
                operation == "confirm_evaluation"
                and (task.state != AeolusState.EVALUATION_REVIEW or (task.evaluation or {}).get("valid") is not True)
            )
            or (operation in {"confirm", "confirm_verified", "complete_review", "deploy", "choose_deployment_worker",
                              "deploy_ppe", "retry_ppe"}
                and task.state != AeolusState.REVIEW)
            or (operation == "submit_feedback" and task.state not in {AeolusState.REVIEW, AeolusState.CODING_BLOCKED})
            or (operation == "retry_coding" and not can_retry_coding(task))
            or (operation == "retry_completion" and not can_retry_completion(task))
        ):
            raise InvalidCardCallback("action is not available")
        evaluation_snapshot = (
            copy.deepcopy(task.evaluation or {}) if operation == "confirm_evaluation" else None
        )
        selected_worker_id = None
        try:
            if operation in {"deploy_ppe", "retry_ppe", "release_ppe"}:
                ppe_service = service.ppe_deployments
                if ppe_service is None:
                    raise InvalidTaskTransition("PPE deployment is disabled")
                if operation == "deploy_ppe":
                    form = action.get("form_value")
                    selection = form.get("ppe_runtime_profile_id") if isinstance(form, dict) else None
                    if not isinstance(selection, str) or len(parts := selection.split("::")) != 2:
                        raise ValueError("请选择 PPE 运行配置")
                    runtime_profile_id, revision = parts
                    if not revision or value["ppe_runtime_profiles"].get(runtime_profile_id) != revision:
                        raise ValueError("PPE 运行配置已变更或不兼容，请刷新卡片")
                    await ppe_service.create(task.task_id, request_id, value["ppe_round_no"],
                                             value["ppe_commit_sha"], revision, runtime_profile_id)
                elif operation == "retry_ppe":
                    await ppe_service.retry(task.task_id, value["ppe_deployment_id"], request_id)
                else:
                    await ppe_service.release(task.task_id, value["ppe_deployment_id"], value["ppe_environment_id"],
                                              value["ppe_generation"])
                task = await service.get(task.task_id)
            elif operation == "choose_deployment_worker":
                selected_worker_id = action.get("option")
                selected = next((
                    w for w in task.deployment_workers or [] if w["worker_id"] == selected_worker_id
                    and w.get("online") and w.get("supports_repository", True)
                ), None)
                if (not isinstance(selected_worker_id, str) or selected is None
                        or task.deployment and (
                            task.deployment.get("resource_releasing")
                            or task.deployment.get("status") in {"running", "unknown", "cancel_requested"}
                        )):
                    raise ValueError("请选择当前可部署的 Devbox")
            elif operation == "release_environment":
                if service.deployments is None or not task.deployment or not task.deployment.get("can_release"):
                    raise InvalidTaskTransition("environment cannot be released")
                await service.deployments.run("release", task.task_id, value["deployment_id"], value["generation"])
                task = await service.get(task.task_id)
            elif operation == "deploy":
                form = action.get("form_value")
                # New cards submit a machine-bound runtime value; accept previously sent combined selectors too.
                worker_id = (
                    (form.get("runtime_profile_id") or form.get("worker_id")) if isinstance(form, dict) else None
                )
                if not isinstance(worker_id, str) or not worker_id or service.deployments is None:
                    raise ValueError("请选择可部署的 Devbox")
                runtime_profile_id = None
                expected_profile_revision = None
                if "::" in worker_id:
                    parts = worker_id.split("::")
                    if len(parts) not in {2, 3}:
                        raise ValueError("请选择有效的运行配置")
                    worker_id, runtime_profile_id = parts[:2]
                    expected_profile_revision = parts[2] if len(parts) == 3 else None
                await service.deployments.run(
                    "enqueue", task.task_id, worker_id, request_id, value["round_no"], value["commit_sha"],
                    *([runtime_profile_id, expected_profile_revision] if runtime_profile_id else []),
                )
                task = await service.get(task.task_id)
            elif operation == "complete_review":
                form = action.get("form_value")
                mode = form.get("verification_mode") if isinstance(form, dict) else None
                if mode == "all_verified":
                    targets = current_verification_targets(task)
                    if not targets:
                        raise ValueError("当前没有可验证环境")
                    task = await service.confirm(
                        task.task_id, request_id, release_authorized=True, verification_targets=targets,
                    )
                elif (not isinstance(mode, str) or mode not in {"verified", "ppe_verified", "skip"}
                        or mode == "verified" and (not task.deployment or not task.deployment.get("current"))
                        or mode == "ppe_verified" and (
                            not task.ppe_deployment or not task.ppe_deployment.get("can_verify"))):
                    raise ValueError("请选择有效的验证结果")
                elif mode == "ppe_verified":
                    assert task.ppe_deployment is not None
                    task = await service.confirm(
                        task.task_id, request_id, task.ppe_deployment["deployment_id"], True, "ppe",
                    )
                else:
                    if mode == "verified":
                        assert task.deployment is not None
                    task = await service.confirm(
                        task.task_id, request_id,
                        task.deployment["deployment_id"] if mode == "verified" and task.deployment else "", True,
                    )
            elif operation in {"evaluation_feedback", "submit_feedback"}:
                form = action.get("form_value")
                feedback = form.get("feedback", "") if isinstance(form, dict) else ""
                if not isinstance(feedback, str) or not 1 <= len(feedback.strip()) <= 4000:
                    raise ValueError("请填写 1 至 4000 字的反馈")
                task = await getattr(service, operation)(task.task_id, feedback, request_id)
            elif operation == "confirm" and value.get("release_resources"):
                task = await service.confirm(task.task_id, request_id, "", True)
            elif operation == "retry_completion":
                task = await service.confirm(task.task_id, request_id)
            elif operation == "confirm_verified":
                if not task.deployment or not task.deployment.get("current"):
                    raise InvalidTaskTransition("deployment is no longer current")
                task = await service.confirm(task.task_id, request_id, task.deployment["deployment_id"])
            elif operation == "retry_validation":
                task = await service.retry_validation(task.task_id)
            else:
                task = await getattr(service, operation)(task.task_id, request_id)
        except (InvalidTaskTransition, TaskNotFound, ValueError) as exc:
            log.warning(
                "meegle_card_action_rejected", workflow="meegle", task_id=task.task_id,
                action=operation, event_id=request_id, **error_fields(exc),
            )
            return {"toast": {"type": "error", "content": "当前状态无法操作或表单内容无效，请刷新卡片后重试"}}
        log.info(
            "meegle_card_action_accepted", workflow="meegle", task_id=task.task_id,
            action=operation, event_id=request_id, state=task.state.value,
        )
        return {
            "toast": {"type": "success", "content": "已更新运行配置选项" if selected_worker_id else {
                "deploy": "Devbox 部署请求已受理", "deploy_ppe": "PPE 部署请求已受理",
                "retry_ppe": "PPE 重试部署请求已受理", "release_environment": "Devbox 释放请求已受理",
                "release_ppe": "PPE 释放请求已受理",
            }.get(operation, "已接收，正在处理")},
            "card": {"type": "raw", "data": self.card(
                task, owner_key, recipient, evaluation_snapshot=evaluation_snapshot,
                selected_worker_id=selected_worker_id,
            )},
        }
