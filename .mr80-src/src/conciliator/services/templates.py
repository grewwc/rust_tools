import re

from conciliator.integrations.git.conflicts import blank_description

NEXTCODE_COMMENT_MAX_LENGTH = 6000


def _bounded_discussion(body: str, header: list[str], instruction: str, lark_doc_url: str) -> str:
    if len(body) <= NEXTCODE_COMMENT_MAX_LENGTH:
        return body
    # Keep the marker and approval command intact; the report contains the full code.
    return "\n".join([
        *header,
        "",
        f"冲突块内容过长，详情见：[飞书文档链接]({lark_doc_url})",
        "",
        instruction,
        "",
    ])


def build_lark_doc_xml(
    *,
    source_branch: str,
    target_branch: str,
    resolve_branch: str,
    blocks: list[dict],
    original_mr_number: int | None = None,
    source_commit: str = "",
    target_commit: str = "",
    resolve_commit: str = "",
) -> str:
    """用任务信息和 Coco 结果生成飞书文档内容。"""
    parts = ["<h1>MR 自动解冲突报告</h1>"]
    if original_mr_number is not None:
        parts.append(f"<p>原 MR：!{original_mr_number}</p>")
    parts.extend(
        [
            f"<p>源分支 → 目标分支：<code>{_escape(source_branch)}</code> → <code>{_escape(target_branch)}</code></p>",
            f"<p>resolve 分支：<code>{_escape(resolve_branch)}</code></p>",
            f"<p>冻结 source commit：<code>{_escape(source_commit)}</code></p>",
            f"<p>冻结 target commit：<code>{_escape(target_commit)}</code></p>",
            f"<p>resolve commit：<code>{_escape(resolve_commit)}</code></p>",
            f"<p>冲突块数：{len(blocks)}</p>",
            "<h2>冲突块详情</h2>",
        ]
    )
    for block in blocks:
        conflict_id = str(block.get("conflict_id") or f"C{block['idx']:04d}")
        file_path = str(block["file_path"])
        parts.append(f"<h3>{_escape(conflict_id)} · {_escape(file_path)}</h3>")
        if block.get("approval_status") == "NOT_REQUIRED":
            parts.append("<p>纯空行冲突，无需人工审批；保留 source 侧原始空白。</p>")
            for side in ("source", "target"):
                parts.append(f"<p>{side.title()}：{blank_description(block[f'{side}_text'])}</p>")
            continue
        for side in ("source", "target"):
            author = str(block.get(f"{side}_author_email") or "（空）")
            location = _location(
                str(block.get(f"{side}_path") or file_path),
                block.get(f"{side}_start_line"),
                block.get(f"{side}_end_line"),
            )
            parts.append(f"<p>{side.title()} 作者和文件：{_escape(author)}；<code>{_escape(location)}</code></p>")
        source_text = str(block.get("source_text", block.get("ours", "")))
        target_text = str(block.get("target_text", block.get("theirs", "")))
        parts.append(f"<p>Source 侧冲突块内容：</p><pre><code>{_escape(source_text)}</code></pre>")
        parts.append(f"<p>Target 侧冲突块内容：</p><pre><code>{_escape(target_text)}</code></pre>")
        resolution = block.get("ai_resolution")
        rendered_resolution = str(resolution) if resolution is not None else "(待生成)"
        parts.append(f"<p>解决后的代码：</p><pre><code>{_escape(rendered_resolution)}</code></pre>")
        parts.append(f"<p>解决决策：{_escape(str(block.get('decision_summary') or '(无)'))}</p>")
    return "".join(parts)


def build_lark_doc_markdown(
    *,
    original_mr_number: int,
    source_branch: str,
    source_commit: str,
    target_branch: str,
    target_commit: str,
    resolve_branch: str,
    resolve_commit: str,
    blocks: list[dict],
    resolved_files: dict[str, str] | None = None,
) -> str:
    """Render the human-reviewable content used by the TCE validation flow."""
    lines = [
        "# MR 自动解冲突报告",
        "",
        f"- 原 MR：!{original_mr_number}",
        f"- 源分支：`{source_branch}`",
        f"- 目标分支：`{target_branch}`",
        f"- resolve 分支：`{resolve_branch}`",
        f"- 冻结 source commit：`{source_commit}`",
        f"- 冻结 target commit：`{target_commit}`",
        f"- resolve commit：`{resolve_commit}`",
        f"- 冲突块数：{len(blocks)}",
        "",
        "## 冲突块详情",
    ]
    for block in blocks:
        conflict_id = str(block.get("conflict_id") or f"C{block['idx']:04d}")
        if block.get("approval_status") == "NOT_REQUIRED":
            lines.extend([
                "", f"### {conflict_id} `{block['file_path']}`", "",
                "纯空行冲突，无需人工审批；保留 source 侧原始空白。", "",
                f"- Source：{blank_description(block['source_text'])}",
                f"- Target：{blank_description(block['target_text'])}",
                f"- 解决后：{blank_description(block['ai_resolution'])}",
            ])
            continue
        lines.extend(
            [
                "",
                f"### {conflict_id} `{block['file_path']}`",
                "",
                (
                    "- Source 作者和文件："
                    f"`{block.get('source_author_email') or '（空）'}`，"
                    f"`{block.get('source_path') or block['file_path']}:{block.get('source_start_line') or '?'}-"
                    f"{block.get('source_end_line') or '?'}`"
                ),
                (
                    "- Target 作者和文件："
                    f"`{block.get('target_author_email') or '（空）'}`，"
                    f"`{block.get('target_path') or block['file_path']}:{block.get('target_start_line') or '?'}-"
                    f"{block.get('target_end_line') or '?'}`"
                ),
                "",
                "Source 侧冲突块内容：",
                "",
                _code_block(str(block.get("source_text") or "")),
                "",
                "Target 侧冲突块内容：",
                "",
                _code_block(str(block.get("target_text") or "")),
                "",
                "解决后的代码：",
                "",
                _code_block(str(block.get("ai_resolution") or "")),
                "",
                f"解决决策：{block.get('decision_summary') or '(无)'}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def build_mr_description(
    *,
    source_branch: str,
    target_branch: str,
    resolve_branch: str,
    lark_doc_url: str,
    blocks: list[dict],
    original_mr_number: int | None = None,
    source_commit: str = "",
    target_commit: str = "",
    resolve_commit: str = "",
    retry_count: int = 0,
    task_status: str = "AWAITING_APPROVAL",
) -> str:
    approval_statuses = [
        str(block.get("approval_status") or ("APPROVED" if block.get("approved") else "PENDING")) for block in blocks
    ]
    approved_count = approval_statuses.count("APPROVED")
    rejected_count = approval_statuses.count("REJECTED")
    exempt_count = approval_statuses.count("NOT_REQUIRED")
    pending_count = approval_statuses.count("PENDING")
    manual_count = approved_count + rejected_count + pending_count
    unknown_count = len(blocks) - exempt_count - manual_count
    completed = bool(blocks) and approved_count == manual_count and not unknown_count
    status = (
        f"全部确认完成（{approved_count}/{manual_count}）"
        if completed
        else f"等待 owner 确认（{approved_count}/{manual_count}）"
    )
    if exempt_count:
        status = ("纯空行冲突已处理，无需人工审批" if not manual_count and not unknown_count else
                  f"人工审批 {approved_count}/{manual_count}，另有 {exempt_count} 块无需人工审批")
    if unknown_count:
        status += f"；{unknown_count} 块审批记录异常，尚未完成"
    lines = ["## 🤖 自动解冲突报告", ""]
    if original_mr_number is not None:
        lines.append(f"- **原 MR**：!{original_mr_number}")
    lines.extend(
        [
            f"- **源分支** → **目标分支**：`{source_branch}` → `{target_branch}`",
            f"- **resolve 分支**：`{resolve_branch}`",
            f"- **冻结 source commit**：`{source_commit}`",
            f"- **冻结 target commit**：`{target_commit}`",
            f"- **resolve commit**：`{resolve_commit}`",
            f"- **冲突块数**：{len(blocks)}",
            f"- **状态**：{status}",
            f"- **任务状态**：`{task_status}`",
            f"- **重解次数**：{retry_count}",
            (f"- **确认分布**：{approved_count} approved / {rejected_count} rejected / {pending_count} pending"),
            "",
            "本任务只处理上述冻结版本。分支后续提交由新任务处理，合入前请确认 MR 的最新状态。",
            "",
            "### 📄 详细报告",
            f"[飞书文档链接]({lark_doc_url})",
            "",
        ]
    )
    lines.extend(["", "### ✅ 确认进度"])
    for block in blocks:
        approval_status = str(block.get("approval_status") or ("APPROVED" if block.get("approved") else "PENDING"))
        if approval_status == "NOT_REQUIRED":
            lines.append(f"- #{block['idx']}（{block['conflict_id']}） `{block['file_path']}` "
                         "— 纯空行冲突，无需人工审批（NOT_REQUIRED）")
            continue
        check = "x" if approval_status == "APPROVED" else " "
        conflict_id = block.get("conflict_id") or f"C{block['idx']:04d}"
        authors = list(
            dict.fromkeys(
                email
                for email in (
                    block.get("source_author_email"),
                    block.get("target_author_email"),
                    block.get("author_email"),
                )
                if email
            )
        )
        rendered_authors = ", ".join(authors) or "unknown"
        file_path = block.get("resolved_path") or block["file_path"]
        start_line = block.get("resolved_start_line") or block["start_line"]
        end_line = block.get("resolved_end_line") or block["end_line"]
        if approval_status == "APPROVED":
            status_suffix = " ✓ APPROVED"
        elif approval_status == "REJECTED":
            reason = _inline_text(str(block.get("reject_reason") or "未提供原因"))
            status_suffix = f" ✗ REJECTED：{reason}"
        elif approval_status == "PENDING":
            status_suffix = " · PENDING"
        else:
            status_suffix = " · 审批记录异常"
        lines.append(
            f"- [{check}] #{block['idx']}（{conflict_id}） "
            f"`{file_path}:{start_line}-{end_line}` — "
            f"{rendered_authors}{status_suffix}"
        )
    lines.extend(
        [
            "",
            "### 📜 指令",
            "- 在对应冲突块 Discussion 里回复 `/approval`、`approve` 或 `reject <reason>`",
            "- MR 主线回复 `/resolve approve all` — 全部同意（限 `/resolve` 发起人）",
            "- MR 主线回复 `/resolve retry` — 重解被拒块（限 `/resolve` 发起人）",
            "- MR 主线回复 `/resolve status` — 查看进度",
            "- MR 主线回复 `/resolve cancel` — 取消任务（限 `/resolve` 发起人）",
        ]
    )
    return "\n".join(lines)


def build_discussion_body(
    *,
    task_id: int,
    conflict_id: str,
    idx: int,
    file_path: str,
    start_line: int,
    end_line: int,
    ours: str,
    theirs: str,
    ai_resolution: str,
    lark_doc_url: str,
    author_email: str | None = None,
    source_author_email: str | None = None,
    target_author_email: str | None = None,
    source_path: str | None = None,
    source_start_line: int | None = None,
    source_end_line: int | None = None,
    target_path: str | None = None,
    target_start_line: int | None = None,
    target_end_line: int | None = None,
    decision_summary: str = "",
    thinking: str = "",
) -> str:
    del thinking  # 不传播 Coco 的原始 AgentThinking。
    source_author = source_author_email or author_email or "unknown"
    target_author = target_author_email or "unknown"
    source_location = _location(
        source_path or file_path,
        source_start_line,
        source_end_line,
    )
    target_location = _location(
        target_path or file_path,
        target_start_line,
        target_end_line,
    )
    lines = [
        f"<!-- conciliator:task={task_id}:block={conflict_id} -->",
        f"**冲突块 #{idx}** · `{file_path}:{start_line}-{end_line}`  ",
        f"**Source 作者和文件**：{source_author} · `{source_location}`  ",
        f"**Target 作者和文件**：{target_author} · `{target_location}`",
        "",
        "**Source 侧冲突块内容**：",
        "",
        _code_block(ours),
        "",
        "**Target 侧冲突块内容**：",
        "",
        _code_block(theirs),
        "",
        "**🤖 解决后的代码**：",
        "",
        _code_block(ai_resolution),
        "",
        f"**解决决策**：{decision_summary or '(无)'}",
        "",
        "---",
        "👤 回复 `/approval`、`approve` 或 `reject <reason>` 即可确认",
    ]
    return _bounded_discussion("\n".join(lines).rstrip() + "\n", lines[:4], lines[-1], lark_doc_url)


def build_retry_discussion_body(
    *,
    task_id: int,
    attempt: int,
    conflict_id: str,
    idx: int,
    file_path: str,
    start_line: int,
    end_line: int,
    ai_resolution: str,
    decision_summary: str,
    reject_reason: str,
    lark_doc_url: str,
    source_author_email: str | None = None,
    target_author_email: str | None = None,
    source_location: str = "",
    target_location: str = "",
) -> str:
    """Render the new solution appended to an existing conflict thread."""
    lines = [
        (f"<!-- conciliator:task={task_id}:block={conflict_id}:attempt={attempt} -->"),
        f"**第 {attempt} 轮重解 · 冲突块 #{idx}** · `{file_path}:{start_line}-{end_line}`",
        f"**上次 reject 原因**：{_inline_text(reject_reason)}",
        "",
        "**重解后的代码**：",
        "",
        _code_block(ai_resolution),
        "",
        f"**解决决策**：{decision_summary or '(无)'}",
        "",
        "---",
        "👤 本块已重置为 PENDING，请重新回复 `/approval` 或 `reject <reason>`",
    ]
    header = [
        lines[0],
        lines[1] + "  ",
        f"**Source 作者和文件**：{source_author_email or 'unknown'} · `{source_location or file_path}`  ",
        f"**Target 作者和文件**：{target_author_email or 'unknown'} · `{target_location or file_path}`",
    ]
    return _bounded_discussion("\n".join(lines).rstrip() + "\n", header, lines[-1], lark_doc_url)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _code_block(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _location(
    path: str,
    start_line: int | None,
    end_line: int | None,
) -> str:
    if start_line is None or end_line is None:
        return path
    return f"{path}:{start_line}-{end_line}"


def _inline_text(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").replace("|", "\\|")
