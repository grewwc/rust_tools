import re
from dataclasses import dataclass

_START_RE = re.compile(r"^<{7} \S+$")
_SEP_RE = re.compile(r"^={7}$")
_END_RE = re.compile(r"^>{7} \S+$")


@dataclass(frozen=True)
class ParsedConflict:
    file_path: str
    block_idx: int  # 在该文件内的 0-based 序号
    start_line: int  # ours 块在原文件中的起始行号(1-based)
    end_line: int  # end_re 的行号(1-based)
    ours_text: str
    theirs_text: str


class ConflictParseError(Exception):
    def __init__(self, file_path: str, line_no: int, detail: str):
        super().__init__(f"{file_path}:{line_no}: {detail}")
        self.file_path = file_path
        self.line_no = line_no
        self.detail = detail


def parse_conflict_file(file_path: str, content: str) -> list[ParsedConflict]:
    """解析一个含冲突标记的文件内容。

    冲突块格式:
        <<<<<<< HEAD   (或 <<<<<<< ours/branch-name)
        ours content
        =======
        theirs content
        >>>>>>> target
    """
    lines = content.splitlines(keepends=True)
    result: list[ParsedConflict] = []
    i = 0
    block_idx = 0

    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        if _START_RE.match(line):
            start_line = i + 1  # 1-based
            ours_lines: list[str] = []
            theirs_lines: list[str] = []
            i += 1

            # 收集 ours
            while i < len(lines) and not _SEP_RE.match(lines[i].rstrip("\r\n")):
                if _START_RE.match(lines[i].rstrip("\r\n")) or _END_RE.match(lines[i].rstrip("\r\n")):
                    raise ConflictParseError(
                        file_path, i + 1, "unexpected marker in ours section"
                    )
                ours_lines.append(lines[i])
                i += 1

            if i >= len(lines) or not _SEP_RE.match(lines[i].rstrip("\r\n")):
                raise ConflictParseError(
                    file_path, i + 1, "missing ======= separator"
                )
            i += 1  # skip =======

            # 收集 theirs
            while i < len(lines) and not _END_RE.match(lines[i].rstrip("\r\n")):
                if _START_RE.match(lines[i].rstrip("\r\n")) or _SEP_RE.match(lines[i].rstrip("\r\n")):
                    raise ConflictParseError(
                        file_path, i + 1, "unexpected marker in theirs section"
                    )
                theirs_lines.append(lines[i])
                i += 1

            if i >= len(lines) or not _END_RE.match(lines[i].rstrip("\r\n")):
                raise ConflictParseError(
                    file_path, i + 1, "missing >>>>>>> end marker"
                )
            end_line = i + 1  # 1-based
            i += 1  # skip >>>>>>>

            result.append(ParsedConflict(
                file_path=file_path,
                block_idx=block_idx,
                start_line=start_line,
                end_line=end_line,
                ours_text="".join(ours_lines),
                theirs_text="".join(theirs_lines),
            ))
            block_idx += 1
        else:
            i += 1

    return result


def is_blank_text(text: object) -> bool:
    """Only known ASCII horizontal whitespace and line endings qualify."""
    return isinstance(text, str) and not text.strip(" \t\r\n")


def is_blank_conflict(source: object, target: object) -> bool:
    return is_blank_text(source) and is_blank_text(target)


def blank_description(text: str) -> str:
    if not text:
        return "无内容"
    count = len(text.splitlines())
    if not text.strip("\r\n"):
        return f"{count} 行空行"
    return f"仅含空白字符，共 {count} 行"
