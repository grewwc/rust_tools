#!/usr/bin/env python3
"""DDL 导出工具:从 alembic 迁移渲染纯 SQL,供运维执行。

本地使用(开发者改完 models.py + autogenerate 迁移后跑):

    # 1. 起临时 MySQL(可选;工具自己起 testcontainer)
    docker run -d --name mysql -e MYSQL_ROOT_PASSWORD=root \
      -e MYSQL_DATABASE=conciliator_ddl -p 3306:3306 mysql:8.0
    export MYSQL_DSN="mysql+pymysql://root:root@127.0.0.1:3306/conciliator_ddl"

    # 2. 渲染
    uv run python scripts/export_ddl.py

产物:
  ddl/<rev>_<desc>.sql   每个 revision 的增量 SQL(base → 该 revision 累积)
  ddl/all.sql            空库初始化快照

设计:
  - 工具完全在本地跑,不连 RDS
  - 用临时 MySQL 容器作渲染环境
  - SQL 内容由 alembic upgrade --sql 单向渲染,.py 是 source of truth
  - all.sql 用 IF NOT EXISTS 包裹,新环境可一次跑完
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DDL_DIR = REPO_ROOT / "ddl"
VERSIONS_DIR = REPO_ROOT / "migrations" / "versions"


def run(cmd: list[str], **kwargs) -> str:
    """跑命令,返回 stdout;失败抛异常。"""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        **kwargs,
    )
    return result.stdout


def _alembic_cmd(*args: str) -> list[str]:
    """构造 alembic 命令,用当前 python 解释器(确保找到 alembic)。"""
    import sys as _sys

    return [_sys.executable, "-m", "alembic", *args]


def alembic_env() -> dict[str, str]:
    """构造 alembic 命令的环境变量。"""
    env = os.environ.copy()
    env.setdefault("ENV", "dev")
    return env


def get_revisions() -> list[tuple[str, str, str, str | None]]:
    """返回 [(revision_id, description, filename_stem, down_revision)]。

    从 migrations/versions/*.py 静态解析(不依赖 alembic),顺序按依赖链。
    """
    revisions: list[tuple[str, str, str, str | None]] = []
    for py_file in sorted(VERSIONS_DIR.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        text = py_file.read_text(encoding="utf-8")
        rev_match = re.search(r"^revision\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        down_match = re.search(r"^down_revision\s*=\s*(['\"][^'\"]*['\"]|None)", text, re.M)
        _desc_match = re.search(
            r"^#\s*\d{4}-\d{2}-\d{2}.*?\n.*?\n.*?\n",
            text,
        )
        if not rev_match:
            continue
        rev = rev_match.group(1)
        down = down_match.group(1).strip("'\"") if down_match and down_match.group(1) != "None" else None
        # 描述:从 migration docstring 或文件名提取
        doc_match = re.search(
            r'^"""(.*?)"""',
            text,
            re.S,
        )
        desc = doc_match.group(1).strip().split("\n")[0] if doc_match else py_file.stem
        revisions.append((rev, desc, py_file.stem, down))
    # 按依赖链排序(从 base 到 head)
    return sort_by_chain(revisions)


def sort_by_chain(
    revisions: list[tuple[str, str, str, str | None]],
) -> list[tuple[str, str, str, str | None]]:
    """按 down_revision 拓扑排序。"""
    by_rev = {r[0]: r for r in revisions}
    sorted_revs: list[tuple[str, str, str, str | None]] = []
    visited: set[str] = set()

    def visit(rev: str) -> None:
        if rev in visited or rev not in by_rev:
            return
        data = by_rev[rev]
        if data[3] is not None:  # 先访问 down_revision
            visit(data[3])
        visited.add(rev)
        sorted_revs.append(data)

    for rev in by_rev:
        visit(rev)
    return sorted_revs


def _revision_is_idempotent(stem: str) -> bool:
    """True when the migration opts into rerunnable incremental DDL.

    Only migrations whose operations are limited to CREATE TABLE / CREATE INDEX
    can be made rerunnable safely; ALTER TABLE is not idempotent on MySQL.
    """
    path = VERSIONS_DIR / f"{stem}.py"
    return "IDEMPOTENT_DDL = True" in path.read_text(encoding="utf-8")


def export_revision_sql(
    rev: str,
    desc: str,
    stem: str,
    *,
    down_revision: str | None,
) -> str:
    """渲染单个 revision 的增量 SQL。

    base revision 从空库渲染；后续 revision 使用 ``down:rev`` 范围，避免
    运维在已有数据库上执行增量文件时重复建表。
    alembic_version 表的 DDL 被 _strip_alembic_version 移除(由 alembic CLI 管理)。
    """
    target = rev if down_revision is None else f"{down_revision}:{rev}"
    sql = run(
        _alembic_cmd("upgrade", target, "--sql"),
        env=alembic_env(),
    )
    # 移除 alembic_version 相关语句(由 alembic CLI 自动管理,不需要运维执行,
    # 且 VARCHAR PK 无法通过 SQL 审核)
    sql = _fix_alembic_version(sql)
    idempotent = _revision_is_idempotent(stem)
    if idempotent:
        # MySQL does not support CREATE INDEX IF NOT EXISTS, so inline the index
        # into the table definition before adding IF NOT EXISTS.
        sql = _inline_indexes(_make_idempotent(sql))
    return _wrap_sql_file(sql, rev, desc, stem, idempotent=idempotent)


def export_full_schema_sql(revisions: list) -> str:
    """生成供空库一次执行的全量 schema 快照。"""
    if not revisions:
        return "-- no revisions found\n"
    head = revisions[-1][0]
    sql = run(
        _alembic_cmd("upgrade", head, "--sql"),
        env=alembic_env(),
    )
    # 移除 alembic_version(由 alembic CLI 管理,VARCHAR PK 过不了审核)
    sql = _fix_alembic_version(sql)
    idempotent = _make_idempotent(sql)
    # 把 CREATE INDEX 内联到 CREATE TABLE(目标 RDS 要求)
    inlined = _inline_indexes(idempotent)
    return _wrap_all_sql(inlined, head)


def _wrap_sql_file(
    sql: str, rev: str, desc: str, stem: str, *, idempotent: bool = False
) -> str:
    rerun_note = (
        "-- ✅ 本文件幂等,可重复执行。\n"
        if idempotent
        else "-- ⚠️ 本文件非幂等,不要直接重跑!依赖 alembic_version 表去重。\n"
    )
    header = (
        f"-- ddl/{stem}.sql\n"
        f"-- 由 scripts/export_ddl.py 从 migrations/versions/{stem}.py 渲染\n"
        f"-- revision: {rev}\n"
        f"-- description: {desc}\n"
        f"-- generated at: {datetime.now(UTC).isoformat()}\n"
        f"--\n"
        f"-- 运维执行前先查:\n"
        f"--   SELECT version_num FROM alembic_version;\n"
        f"-- 若 version_num 已是 {rev} 或更新 → 跳过本文件\n"
        f"--\n"
        f"{rerun_note}"
        f"\n"
    )
    return _normalize_output(header + sql)


def _wrap_all_sql(sql: str, head_rev: str) -> str:
    header = (
        f"-- ddl/all.sql — 全量 schema 快照\n"
        f"-- 由 scripts/export_ddl.py 渲染,对应 head revision: {head_rev}\n"
        f"-- generated at: {datetime.now(UTC).isoformat()}\n"
        f"--\n"
        f"-- 用途:\n"
        f"--   - 空库初始化(新机房 / 本地 docker / 测试环境)\n"
        f"--   - 不用于增量!增量走 ddl/<rev>.sql\n"
        f"--\n"
        f"-- 仅限空库执行一次:CREATE 语句带 IF NOT EXISTS,后续 ALTER 不保证可重跑\n"
        f"\n"
    )
    return _normalize_output(header + sql)


def _normalize_output(value: str) -> str:
    """移除 Alembic 离线渲染产生的行尾空格，并只保留一个末尾换行。"""
    return "\n".join(line.rstrip() for line in value.splitlines()).rstrip() + "\n"


def _make_idempotent(sql: str) -> str:
    """让全量快照中的建表、建索引和基线版本写入可安全处理空库。"""
    sql = re.sub(r"\bCREATE TABLE\s+", "CREATE TABLE IF NOT EXISTS ", sql, flags=re.I)
    sql = re.sub(r"\bCREATE INDEX\s+", "CREATE INDEX IF NOT EXISTS ", sql, flags=re.I)
    sql = re.sub(r"\bCREATE UNIQUE INDEX\s+", "CREATE UNIQUE INDEX IF NOT EXISTS ", sql, flags=re.I)
    sql = re.sub(
        r"INSERT\s+INTO\s+alembic_version\s*\(version_num\)\s*"
        r"VALUES\s*\(([^)]+)\)\s*;",
        r"INSERT INTO alembic_version (version_num) "
        r"SELECT \1 WHERE NOT EXISTS (SELECT 1 FROM alembic_version);",
        sql,
        flags=re.I,
    )
    return sql


def _fix_alembic_version(sql: str) -> str:
    """重写 alembic_version 表,满足 SQL 审核规则。

    审核要求:
      1. 表必须有 BIGINT UNSIGNED 主键
      2. 表必须有 COMMENT
      3. 列必须有 COMMENT

    原始(alembic 自动生成):
      CREATE TABLE alembic_version (
          version_num VARCHAR(32) NOT NULL,
          CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
      );

    重写为:
      CREATE TABLE alembic_version (
          id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,
          version_num VARCHAR(32) NOT NULL COMMENT '当前迁移版本号',
          PRIMARY KEY (id),
          UNIQUE KEY uk_alembic_version_version_num (version_num)
      ) COMMENT='alembic迁移版本跟踪表';

    注意:加 id 自增列不影响 alembic 的 SELECT version_num FROM alembic_version 逻辑
    (alembic 只读 version_num 列,不关心是否有额外 PK)。
    """
    # 匹配 CREATE TABLE alembic_version ... ); (含 IF NOT EXISTS)
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF NOT EXISTS\s+)?alembic_version\s*\([^;]*?\);",
        re.I | re.DOTALL,
    )

    replacement = (
        "CREATE TABLE{exists} alembic_version (\n"
        "    id BIGINT UNSIGNED NOT NULL COMMENT '主键ID' AUTO_INCREMENT,\n"
        "    version_num VARCHAR(32) NOT NULL COMMENT '当前迁移版本号',\n"
        "    PRIMARY KEY (id),\n"
        "    UNIQUE KEY uk_alembic_version_version_num (version_num)\n"
        ") COMMENT='alembic迁移版本跟踪表';"
    )

    def _replace(m: re.Match) -> str:
        exists = " IF NOT EXISTS" if "IF NOT EXISTS" in m.group(0).upper() else ""
        return replacement.format(exists=exists)

    sql = pattern.sub(_replace, sql)
    return sql


def _inline_indexes(sql: str) -> str:
    """把独立的 CREATE INDEX 合并到对应的 CREATE TABLE 语句内。

    目标 RDS 不支持独立的 CREATE INDEX(或运维要求归并),
    把 `CREATE INDEX xxx ON table (cols);` 转成
    `KEY xxx (cols)` 内联到 CREATE TABLE 的 `)` 之前。

    用于 all.sql(全量 schema 快照)。
    """
    import re as _re

    # 找所有 CREATE INDEX ... ON <table> (cols);
    # 支持 IF NOT EXISTS(all.sql 里 _make_idempotent 加的)
    index_pattern = _re.compile(
        r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF NOT EXISTS\s+)?(\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)\s*;",
        _re.IGNORECASE,
    )
    # 找所有 CREATE TABLE <table> ( ... ) [COMMENT='...'];
    # MySQL 方言会把表注释放在右括号与分号之间，必须一并保留。
    table_pattern = _re.compile(
        r"(CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(\w+)\s*\()"
        r"([^;]*?)(\)\s*(?:(?:COMMENT\s*=\s*'[^']*'|"
        r"(?:DEFAULT\s+)?(?:CHARSET|CHARACTER\s+SET|COLLATE|ENGINE)\s*=\s*\w+)\s*)*;)",
        _re.IGNORECASE | _re.DOTALL,
    )

    # 收集每个表的索引
    indexes_by_table: dict[str, list[tuple[str, str, bool]]] = {}  # table -> [(name, cols, unique)]
    for m in index_pattern.finditer(sql):
        idx_name, table, cols = m.group(1), m.group(2), m.group(3)
        is_unique = bool(_re.search(r"UNIQUE", m.group(0), _re.IGNORECASE))
        indexes_by_table.setdefault(table, []).append((idx_name, cols, is_unique))

    if not indexes_by_table:
        return sql

    # 从 SQL 中删除独立的 CREATE INDEX 语句
    matched_tables = {m.group(2).lower() for m in table_pattern.finditer(sql)}
    sql = index_pattern.sub(lambda m: "" if m.group(2).lower() in matched_tables else m.group(0), sql)

    # 把索引内联到 CREATE TABLE 的 `)` 之前
    def _inject_indexes(m: _re.Match) -> str:
        table_def = m.group(1)
        table_name = m.group(2)
        body = m.group(3)
        close = m.group(4)

        idxs = indexes_by_table.get(table_name, [])
        if not idxs:
            return table_def + body + close

        # 在 body 末尾(PRIMARY KEY 之后,`)` 之前)加 KEY/UNIQUE KEY
        idx_lines = []
        for idx_name, cols, is_unique in idxs:
            kw = "UNIQUE KEY" if is_unique else "KEY"
            idx_lines.append(f"  {kw} {idx_name} ({cols})")

        # body 末尾可能有逗号,也可能没有;确保有逗号分隔
        body_stripped = body.rstrip()
        if body_stripped and not body_stripped.endswith(","):
            body_stripped += ","
        body_stripped += "\n" + ",\n".join(idx_lines) + "\n"

        return table_def + body_stripped + close

    sql = table_pattern.sub(_inject_indexes, sql)
    # 清理多余空行
    sql = _re.sub(r"\n{3,}", "\n\n", sql)
    return sql


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DDL_DIR),
        help=f"输出目录(默认 {DDL_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    revisions = get_revisions()
    if not revisions:
        print("❌ 未找到任何 migration 文件", file=sys.stderr)
        return 1

    print(f"找到 {len(revisions)} 个 revision:")
    for rev, _desc, stem, down in revisions:
        print(f"  {rev} ({stem}) ← {down}")

    # 渲染每个 revision 的增量 SQL
    for rev, _desc, stem, down in revisions:
        print(f"  渲染 ddl/{stem}.sql ...")
        sql = export_revision_sql(
            rev,
            _desc,
            stem,
            down_revision=down,
        )
        (output_dir / f"{stem}.sql").write_text(sql, encoding="utf-8")

    # 渲染全量 schema
    print("  渲染 ddl/all.sql ...")
    all_sql = export_full_schema_sql(revisions)
    (output_dir / "all.sql").write_text(all_sql, encoding="utf-8")

    print(f"\n✓ DDL 已导出到 {output_dir}/")
    print(f"  共 {len(revisions)} 个增量 SQL + 1 个全量快照")
    print("\n下一步:")
    print('  git add ddl/ && git commit -m "chore(ddl): regenerate DDL"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
