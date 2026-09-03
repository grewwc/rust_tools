#!/usr/bin/env python3
"""Post-process the agent's final answer text before it is printed to the
terminal: Chinese (fullwidth / ideographic) punctuation that appears inside
code or file-location contexts is converted to its ASCII equivalent, as are
fullwidth parentheses in plain prose and fullwidth colons / full stops that
directly abut an ASCII letter.  A full stop immediately before an inline-code
span that starts with an ASCII letter is treated the same way.  Other Chinese
punctuation in plain prose is left untouched.

Contexts that are translated (Chinese -> ASCII punctuation):

  1. Fenced code blocks (``` ... ```): every line inside the block.
  2. Inline code spans (`...`): everything between the backticks.
  3. Prose "file-location" spans: within a non-space run, substrings that
     look like a file path or file reference (start with a path indicator
     such as `~` `.` `/` `\` or a drive letter, or match
     `name.ext[:line[:col]]` / a dotted word followed by a separator).
     Chinese punctuation inside such a span is converted -- but a span stops
     at punctuation followed by a CJK ideograph, marking the transition back
     into prose (e.g. `src/main，rs` becomes `src/main,rs` while
     `src/main，请查看` keeps its prose comma).
  4. Fullwidth parentheses `（` `）` in plain prose: converted to halfwidth
     with a separating space when they abut prose, so rendered Markdown does
     not jam CJK text against the paren (e.g. `动态指引（code 段）` becomes
     `动态指引 (code 段)`; `函数（输入）报错` becomes `函数 (输入) 报错`).
     The space is inserted only when the neighbour is a non-space character:
     an already-present space is reused (`）（` -> `) (`), and a paren at the
     line end stays bare (no trailing whitespace).  This is a deliberate
     exception to the "prose is untouched" rule: technical prose mixes
     halfwidth parens around code/file references, and a lone fullwidth pair
     reads as a rendering glitch.  Parentheses inside fenced blocks, inline
     code spans, and path spans still map to bare `(` `)` -- a space there
     would corrupt tokens.
  5. Fullwidth colon `：` in plain prose: converted to `: ` (halfwidth colon
     plus one space) only when it directly abuts an ASCII letter -- e.g.
     `使用 rustc：完成` becomes `使用 rustc: 完成`.  Pure-Chinese labels keep
     the fullwidth form (`他说：你好` stays unchanged).  Colons inside fenced
     blocks, inline code spans, and path spans still map to a bare `:` -- a
     space there would corrupt tokens such as `main.rs:10` or `C:\\Users`.
     All other prose punctuation (`，` `、` ...) still stays fullwidth.
  6. Fullwidth period `。` in plain prose: converted to `. ` (halfwidth
     period plus one space) only when it directly abuts an ASCII letter --
     either the sentence ends on ASCII (`使用 rustc。下一步` becomes
     `使用 rustc. 下一步`) or ASCII follows (`完成。Next step` becomes
     `完成. Next step`).  A period immediately before an inline-code span
     that starts with ASCII is also converted, with the space outside the
     code delimiter (`提示规则。`src/main.rs`` becomes
     `提示规则. `src/main.rs``).  Pure-Chinese sentences keep their fullwidth
     `。` (`已处理完成。` stays unchanged), so CJK-only prose is never mangled.
     As with the colon, periods inside fenced blocks, inline code spans, and
     path spans still map to a bare `.` -- a space there would corrupt
     tokens such as `main。rs` (a typo for `main.rs`).

ANSI escape sequences are preserved verbatim and act as token boundaries, so
the script can also be used as a pipe filter directly on rendered terminal
output.

Usage:
    python3 postprocess_terminal.py < input.txt > output.txt
    python3 postprocess_terminal.py --selftest   # run the built-in checks

Integration: the agent runs this via the `ai.output.postprocess_command`
config key (stdin -> stdout filter).  See scripts/AGENTS.md.
"""

import re
import sys

# Chinese / ideographic punctuation -> ASCII mapping.  Applied only inside the
# code / file-location contexts described above.
CJK_TO_ASCII = {
    "\u3002": ".",   # 。 ideographic full stop
    "\uff0c": ", ",   # ， fullwidth comma
    "\u3001": ", ",   # 、 ideographic comma
    "\uff1a": ": ",   # ： fullwidth colon (code/path contexts; prose uses ': ')
    "\uff1b": "; ",   # ； fullwidth semicolon
    "\uff01": "! ",   # ！ fullwidth exclamation
    "\uff1f": "? ",   # ？ fullwidth question
    "\uff08": " (",   # （ fullwidth left paren
    "\uff09": ") ",   # ） fullwidth right paren
    "\u3010": " [",   # 【 left corner bracket
    "\u3011": "] ",   # 】 right corner bracket
    "\uff5e": "~",   # ～ fullwidth tilde
    "\u201c": '"',   # “ left double quote
    "\u201d": '"',   # ” right double quote
    "\u2018": "'",   # ‘ left single quote
    "\u2019": "'",   # ’ right single quote
    "\u300a": "<",   # 《 left double angle bracket
    "\u300b": ">",   # 》 right double angle bracket
    "\u3008": "<",   # 〈 left angle bracket
    "\u3009": ">",   # 〉 right angle bracket
}

# ANSI escape sequences: preserved verbatim, never treated as token content.
# Covers CSI (e.g. SGR `\x1b[...m`, cursor moves), OSC (e.g. `\x1b]...\x07`)
# and the two-char feeds (e.g. `\x1b(B`).
_ANSI_RE = re.compile(
    r"(?:\x1b\[[0-9;?]*[A-Za-z])|(?:\x1b\][^\x07]*(?:\x07|\x1b\\))|(?:\x1b[()][0-9A-Z])"
)

# Inline code span: backtick-delimited, no nested backticks or newlines.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

# A non-space "run" of prose that may be a file path / file reference.
_RUN_RE = re.compile(r"\S+")

# ASCII word-ish chars usable in file names / extensions.
_ASCII_WORD = r"[A-Za-z0-9_]"

# Fullwidth colon `：` and period `。` in plain prose: converted to `: ` /
# `. ` only when they directly abut an ASCII letter on either side -- a
# label or sentence that ends on ASCII, or ASCII that follows.  A period
# immediately before an ASCII-leading inline-code span is visually adjacent
# after Markdown rendering, so it also receives the prose replacement.  The
# inserted space stays outside the code delimiter.  Pure-Chinese prose keeps
# both fullwidth.  Fenced blocks, inline code, and path spans are handled
# upstream and use a bare `:` / `.`.
_PROSE_ASCII_ADJACENT_RE = re.compile(
    r"(?<=[A-Za-z])[\u3002\uff1a]|[\u3002\uff1a](?=[A-Za-z])"
    r"|\u3002(?=`[A-Za-z][^`\n]*`)"
)


def _prose_ascii_adjacent_sub(m):
    """Replacement for _PROSE_ASCII_ADJACENT_RE: `：` -> `: `, `。` -> `. `."""
    return ": " if m.group(0) == "\uff1a" else ". "


def _translate(s):
    """Map every Chinese punctuation char in `s` to its ASCII equivalent."""
    return "".join(CJK_TO_ASCII.get(ch, ch) for ch in s)


def _split_ansi(text):
    """Split into (visible, ansi) alternating segments, starting and ending
    with a visible segment (which may be empty).  ANSI sequences are returned
    verbatim so callers can translate only the visible parts."""
    parts = []
    pos = 0
    for m in _ANSI_RE.finditer(text):
        parts.append((text[pos : m.start()], text[m.start() : m.end()]))
        pos = m.end()
    parts.append((text[pos:], ""))
    return parts


def _translate_visible(text):
    """Translate CJK punctuation in the visible parts, keeping ANSI codes."""
    out = []
    for visible, ansi in _split_ansi(text):
        out.append(_translate(visible))
        out.append(ansi)
    return "".join(out)


# ASCII characters that can appear inside a path (segments, separators, dots,
# hyphens, tildes, underscores).
_PATH_CHAR = "[A-Za-z0-9_./\\\\~-]"

# A path span starts at one of:
#   1. a path indicator: `~/`, `./`, `../`, `/`, `\`, or a drive prefix like
#      `C:\` / `C：\` (fullwidth colon allowed);
#   2. a file reference `name[.name...].ext` (ASCII dot or `。` before the
#      extension) with an optional `:line[:col]` / `：line[:col]` suffix;
#   3. a dotted ASCII word followed by a path separator (a relative
#      multi-segment path that does not start with `./`, e.g. `src/main.rs`).
_SPAN_START_RE = re.compile(
    r"~/|\./|\.\./|/|\\\\|[A-Za-z][:：][\\\\/]"
    r"|(?:"
    + _ASCII_WORD
    + r"+[.\-])*"
    + _ASCII_WORD
    + r"+(?:[.:。]"
    + _ASCII_WORD
    + r"{1,12}(?:[:：]\d+(?:[:：]\d+)?)?)"
    r"|(?:"
    + _ASCII_WORD
    + r"+[.\-])*"
    + _ASCII_WORD
    + r"+(?=[/\\\\])"
)


def _path_spans(run):
    """Yield (start, end) char spans of path-like substrings inside `run`.

    A span continues while the next char is a path char, or is CJK punctuation
    immediately followed by a path char or a digit (path-internal, e.g. the
    `，` in `src/main，rs` or the `：` in `main.rs：10`).  A CJK punctuation
    followed by a CJK ideograph ends the span: prose has resumed (e.g. the
    `，` in `src/main.rs，请确认` stays untouched)."""
    n = len(run)
    i = 0
    while i < n:
        m = _SPAN_START_RE.match(run, i)
        if not m:
            i += 1
            continue
        start = m.start()
        j = m.end()
        while j < n:
            ch = run[j]
            if re.match(_PATH_CHAR, ch):
                j += 1
                continue
            if ch in CJK_TO_ASCII:
                nxt = run[j + 1] if j + 1 < n else ""
                if re.match(_PATH_CHAR + r"|\d", nxt):
                    j += 1  # CJK punctuation inside the path -> translate it
                    continue
            break
        yield (start, j)
        i = j


def _process_path_run(run):
    """Translate CJK punctuation inside the path-like spans of a non-space
    run, leaving any surrounding prose untouched."""
    if not any(ch in CJK_TO_ASCII for ch in run):
        return run
    spans = list(_path_spans(run))
    if not spans:
        return run
    out = []
    prev = 0
    for start, end in spans:
        out.append(run[prev:start])
        out.append(_translate(run[start:end]))
        prev = end
    out.append(run[prev:])
    return "".join(out)


def _translate_prose_punct(text):
    """Convert remaining fullwidth parentheses to ASCII, and the fullwidth
    colon / period to `: ` / `. ` in prose when they directly abut an ASCII
    letter.  These inside fenced blocks, inline code spans, and path spans are
    already translated by the upstream passes with a bare `.`/`:` (a space
    would corrupt tokens such as `main.rs:10`, `C:\\Users`, or `main。rs`);
    whatever `（`/`）`/`：`/`。` is left is prose-level and gets converted
    here (see module docstring, exceptions 4-6)."""
    # Fullwidth parens to halfwidth with a separating space, but only when the
    # neighbour is a non-space character: an already-present space is reused so
    # `）（` becomes `) (` (never `)  (`), and a paren at the line end / before
    # whitespace stays bare (no trailing space). Order matters: `）` is spaced
    # first, so the following `（` sees that inserted space and adds none.
    text = re.sub(r"\uff09(?=\S)", ") ", text)
    text = re.sub(r"(?<=\S)\uff08", " (", text)
    text = text.replace("\uff08", " (").replace("\uff09", ") ")
    out = []
    # Apply the adjacency rule per visible segment: ANSI codes are token
    # boundaries, so a char never "sees" an escape byte (e.g. the trailing
    # `m` of a SGR code) as its neighbour.
    for visible, ansi in _split_ansi(text):
        out.append(_PROSE_ASCII_ADJACENT_RE.sub(_prose_ascii_adjacent_sub, visible))
        out.append(ansi)
    return "".join(out)


def _process_prose_line(visible):
    """Translate a prose line: full translation inside inline code spans,
    path-context translation everywhere else, plus fullwidth parens and the
    ASCII-adjacent colon / period in prose (module docstring, exceptions 4-6)."""
    out = []
    pos = 0
    for m in _INLINE_CODE_RE.finditer(visible):
        out.append(visible[pos : m.start()])
        out.append("`" + _translate(m.group(1)) + "`")
        pos = m.end()
    out.append(visible[pos:])
    return _translate_prose_punct(_translate_path_runs("".join(out)))


def _translate_path_runs(text):
    out = []
    pos = 0
    for m in _RUN_RE.finditer(text):
        run = m.group(0)
        out.append(text[pos : m.start()])
        # Translate only visible chars; ANSI segments are restored verbatim.
        translated = []
        for visible, ansi in _split_ansi(run):
            translated.append(_process_path_run(visible))
            translated.append(ansi)
        out.append("".join(translated))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def process_text(text):
    """Main entry point: translate CJK punctuation in code / file-location
    contexts, preserving ANSI escape sequences and prose punctuation."""
    lines = text.split("\n")
    in_fence = False
    out_lines = []
    for line in lines:
        visible = "".join(v for v, _ in _split_ansi(line))
        stripped = visible.strip()
        if stripped.startswith("```"):
            # Toggle fenced code block (```, ```lang, ...).  Fence lines are
            # left verbatim.
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(_translate_visible(line))
        else:
            out_lines.append(_process_prose_line(line))
    return "\n".join(out_lines)


def _selftest():
    cases = [
        # (input, expected)
        # Inline code: everything translated.
        ("调用 `println！（\"你好\"）` 即可", "调用 `println!(\"你好\")` 即可"),
        # Fenced code block: everything translated.
        ("```python\nprint（1）\n```\n结束。", "```python\nprint(1)\n```\n结束。"),
        # Path separators: translate CJK punct inside the run.
        ("在 src/main，rs 中", "在 src/main,rs 中"),
        ("打开 C：\\Users\\文档\\a。rs", "打开 C:\\Users\\文档\\a.rs"),
        ("路径 /tmp/foo。bar.txt 已写入", "路径 /tmp/foo.bar.txt 已写入"),
        # File reference with line/col.
        ("错误在 main.rs：10：5", "错误在 main.rs:10:5"),
        ("错误在 main。rs：10", "错误在 main.rs:10"),
        # Prose transition: comma after a path stays Chinese.
        ("查看 src/main.rs，请确认", "查看 src/main.rs，请确认"),
        # Prose colon -> ': ' only when it directly abuts an ASCII letter
        # (either side); pure-Chinese labels keep the fullwidth form.
        ("他说：你好，世界。请确认！", "他说：你好，世界。请确认！"),
        ("文件：/tmp/x，处理完成。", "文件：/tmp/x，处理完成。"),
        ("使用 rustc：完成", "使用 rustc: 完成"),
        ("完成：Next step", "完成: Next step"),
        # Colon at end of line: converted only after ASCII, else fullwidth.
        ("运行 cargo check：", "运行 cargo check: "),
        ("步骤如下：", "步骤如下："),
        # Prose period -> '. ' only when it directly abuts an ASCII letter
        # (either side); pure-Chinese periods keep the fullwidth form.
        ("使用 rustc。下一步继续", "使用 rustc. 下一步继续"),
        ("完成。Next step 继续", "完成. Next step 继续"),
        # Markdown delimiters do not render, so a period before an
        # ASCII-leading inline-code span needs its prose space outside code.
        ("提示规则。`src/main.rs:10`", "提示规则. `src/main.rs:10`"),
        # A CJK-leading inline-code span is not ASCII-adjacent and stays
        # untouched.
        ("提示规则。`路径`", "提示规则。`路径`"),
        ("已完成。下一步继续", "已完成。下一步继续"),
        ("已完成。", "已完成。"),
        # Digits are not letters: a period after a number stays fullwidth.
        ("版本 1.0。请确认", "版本 1.0。请确认"),
        # Prose comma stays fullwidth while the sentence-ending period is
        # converted.
        ("查看 src/main.rs，请确认。", "查看 src/main.rs，请确认。"),
        # Prose colon directly before an inline code span.
        ("全仓 8000 有三处：`patch_tools.rs:2648` 拦截（源头，功能性）",
         "全仓 8000 有三处：`patch_tools.rs:2648` 拦截 (源头，功能性)"),
        # Colons inside inline code stay bare: a space would corrupt tokens.
        ("`配置：main.rs：10` 已修复", "`配置:main.rs:10` 已修复"),
        # ANSI codes preserved.
        ("\x1b[31m错误在 main。rs：5\x1b[0m 结束", "\x1b[31m错误在 main.rs:5\x1b[0m 结束"),
        # Quoted file location in prose.
        ('请打开 "src/foo，bar.rs"', '请打开 "src/foo,bar.rs"'),
        # Prose fullwidth parens -> halfwidth (exception 4); inner code span
        # and prose comma stay as-is; a separating space keeps prose and
        # paren apart, and the line-ending paren stays bare (no trailing
        # whitespace).
        ("动态指引（`temporary_files` 段，注入 system prompt）",
         "动态指引 (`temporary_files` 段，注入 system prompt)"),
        ("查看 src/main.rs（请确认）", "查看 src/main.rs (请确认)"),
        # Parens converted with separating spaces; adjacent pairs reuse one
        # space (`）（` -> `) (`) instead of a double space; colon/period
        # stay fullwidth in pure-Chinese prose.
        ("函数（输入）报错：请重试。", "函数 (输入) 报错：请重试。"),
        ("嵌套（甲）（乙）结束", "嵌套 (甲) (乙) 结束"),
        # A fullwidth paren next to an existing space reuses it instead of
        # adding another, and the line-ending paren stays bare.
        ("说明 （注）", "说明 (注)"),
        # ANSI codes are token boundaries: a reset's trailing `m` is not a
        # letter, so a pure-Chinese period after it stays fullwidth.
        ("\x1b[32m成功\x1b[0m。", "\x1b[32m成功\x1b[0m。"),
    ]
    failures = 0
    for i, (inp, expected) in enumerate(cases, 1):
        got = process_text(inp)
        if got != expected:
            failures += 1
            print(f"FAIL case {i}:")
            print(f"  input:    {inp!r}")
            print(f"  expected: {expected!r}")
            print(f"  got:      {got!r}")
    if failures:
        print(f"{failures}/{len(cases)} selftest cases failed")
        return 1
    print(f"selftest: all {len(cases)} cases passed")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    data = sys.stdin.buffer.read()
    text = data.decode("utf-8", errors="replace")
    sys.stdout.write(process_text(text))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
