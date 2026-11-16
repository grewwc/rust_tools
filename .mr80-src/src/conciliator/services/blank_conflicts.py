"""Verify automatic whitespace resolutions against frozen input and committed files."""

from conciliator.domain.approval import approval_not_required
from conciliator.integrations.git.conflicts import is_blank_conflict, parse_conflict_file


def verify_blank_files(task, results: dict, files: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Rebuild files containing automatic blocks, including their unchanged context.

    Manual replacements come from this result or the previously verified result
    on retries. Automatic replacements always come from the frozen source side.
    Exact readback prevents another blank line elsewhere from satisfying a match.
    """
    paths = {b.file_path for b in task.blocks if approval_not_required(b)}
    if not paths:
        return {}
    snapshots = {item["file_path"]: item["content"] for item in (task.conflict_files_snapshot or [])}
    positions = {}
    for path in paths:
        if path not in snapshots or path not in files:
            raise ValueError(f"{path}: blank conflict has no frozen input or committed file")
        snapshot = snapshots[path]
        parsed = parse_conflict_file(path, snapshot)
        blocks = sorted((b for b in task.blocks if b.file_path == path), key=lambda b: b.block_idx)
        if len(parsed) != len(blocks):
            raise ValueError(f"{path}: frozen conflict count differs")
        lines = snapshot.splitlines(keepends=True)
        parts: list[str] = []
        blank_offsets = []
        cursor = 0
        for original, block in zip(parsed, blocks, strict=True):
            if (original.start_line, original.end_line, original.ours_text, original.theirs_text) != (
                block.start_line,
                block.end_line,
                block.ours_text,
                block.theirs_text,
            ):
                raise ValueError(f"{block.conflict_id}: frozen input differs")
            parts.append("".join(lines[cursor : original.start_line - 1]))
            result = results.get(block.conflict_id)
            if approval_not_required(block):
                if not is_blank_conflict(block.ours_text, block.theirs_text):
                    raise ValueError(f"{block.conflict_id}: invalid blank classification")
                replacement = block.ours_text
                if result is not None and result.resolved.text != replacement:
                    raise ValueError(f"{block.conflict_id}: blank result must preserve exact source whitespace")
                blank_offsets.append((block, len("".join(parts).splitlines()), replacement, result))
            else:
                replacement = result.resolved.text if result is not None else block.ai_resolution
                if replacement is None:
                    raise ValueError(f"{block.conflict_id}: missing verified manual resolution")
                # Coco may omit the final line terminator for a code excerpt.
                if replacement and not replacement.endswith("\n"):
                    replacement += "\r\n" if lines[original.start_line - 1].endswith("\r\n") else "\n"
            parts.append(replacement)
            cursor = original.end_line
        parts.append("".join(lines[cursor:]))
        if "".join(parts) != files[path]:
            raise ValueError(f"{path}: committed content differs from frozen context and conflict resolutions")
        line_count = len(files[path].splitlines())
        for block, offset, replacement, result in blank_offsets:
            if replacement:
                position = (offset + 1, offset + len(replacement.splitlines()))
            elif line_count:
                anchor = min(offset + 1, line_count)
                position = (anchor, anchor)
            else:
                position = (0, 0)
            if result is not None and (result.resolved.start_line, result.resolved.end_line) != position:
                raise ValueError(f"{block.conflict_id}: blank result position differs from committed context")
            positions[block.conflict_id] = position
    return positions
