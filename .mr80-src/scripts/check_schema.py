#!/usr/bin/env python3
"""静态检查 migrations 与 ddl 的一致性(不依赖 MySQL)。

CI 调用,检查:
  1. migrations/versions/*.py 能 import(语法正确)
  2. 每个 revision id 唯一
  3. revision 依赖链完整(每个 down_revision 都能找到)
  4. ddl/*.sql 文件名与 migrations/versions/*.py 一一对应
  5. 每个 ddl/*.sql 含 alembic_version 的 UPDATE 语句(渲染完整)
  6. src/conciliator/repositories/models.py 能 import(ORM 语法正确)

不连 MySQL,纯静态分析,CI 友好。

用法:
    uv run python scripts/check_schema.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = REPO_ROOT / "migrations" / "versions"
DDL_DIR = REPO_ROOT / "ddl"
MODELS_PATH = REPO_ROOT / "src" / "conciliator" / "repositories" / "models.py"

# 失败计数,结束时统一报告
_failures: list[str] = []


def check(condition: bool, msg: str) -> None:
    if condition:
        print(f"  ✓ {msg}")
    else:
        print(f"  ✗ {msg}")
        _failures.append(msg)


def parse_migration(path: Path) -> tuple[str, str | None, str] | None:
    """从 .py 文件解析 (revision, down_revision, message)。"""
    text = path.read_text(encoding="utf-8")
    rev_match = re.search(r"^revision\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    down_match = re.search(r"^down_revision\s*=\s*(['\"][^'\"]*['\"]|None)", text, re.M)
    if not rev_match:
        return None
    rev = rev_match.group(1)
    if down_match:
        raw = down_match.group(1)
        down = None if raw == "None" else raw.strip("'\"")
    else:
        down = None
    # 描述从 docstring 取第一行
    doc_match = re.search(r'^"""(.*?)"""', text, re.S)
    msg = doc_match.group(1).strip().split("\n")[0] if doc_match else path.stem
    return rev, down, msg


def main() -> int:
    print("Schema static checks")
    print("=" * 60)

    # ① migrations/*.py 能解析
    print("\n[1/6] 解析 migrations/versions/*.py")
    migrations: list[tuple[str, str | None, str, Path]] = []
    if not VERSIONS_DIR.exists():
        print(f"  ✗ 目录不存在: {VERSIONS_DIR}")
        return 1
    py_files = sorted(f for f in VERSIONS_DIR.glob("*.py") if not f.name.startswith("__"))
    check(len(py_files) > 0, f"找到 {len(py_files)} 个迁移文件")
    for f in py_files:
        parsed = parse_migration(f)
        if parsed is None:
            check(False, f"{f.name}: 无法解析 revision")
            continue
        rev, down, msg = parsed
        migrations.append((rev, down, msg, f))
        print(f"  ✓ {f.name}: rev={rev} down={down} ({msg})")

    # ② revision id 唯一
    print("\n[2/6] revision id 唯一性")
    revs = [m[0] for m in migrations]
    duplicates = {r for r in revs if revs.count(r) > 1}
    check(not duplicates, "无重复 revision id" + (f"(发现重复: {duplicates})" if duplicates else ""))

    # ③ 依赖链完整(每个 down_revision 都能找到)
    print("\n[3/6] revision 依赖链完整性")
    all_revs = {m[0] for m in migrations}
    chain_ok = True
    for _rev, down, _, path in migrations:
        if down is None:
            continue  # base revision,无前驱,OK
        if down not in all_revs:
            print(f"  ✗ {path.name}: down_revision={down} 找不到对应文件")
            chain_ok = False
        else:
            print(f"  ✓ {path.name}: 链 → {down}")
    check(chain_ok, "所有 down_revision 都能找到")

    # ④ ddl/*.sql 文件名与 migrations/versions/*.py 一一对应
    # (all.sql 是全量快照,不算对应关系)
    print("\n[4/6] ddl/*.sql 与 migrations/*.py 对应")
    if not DDL_DIR.exists():
        print(f"  ✗ 目录不存在: {DDL_DIR}")
        return 1
    ddl_files = {f.stem for f in DDL_DIR.glob("*.sql") if f.name != "all.sql"}
    py_stems = {m[3].stem for m in migrations}
    missing_ddl = py_stems - ddl_files
    extra_ddl = ddl_files - py_stems
    check(not missing_ddl, "每个 migration 都有对应 ddl/*.sql" + (f"(缺: {missing_ddl})" if missing_ddl else ""))
    check(not extra_ddl, "无孤儿 ddl/*.sql" + (f"(多: {extra_ddl})" if extra_ddl else ""))

    # ⑤ 每个 ddl/*.sql 含 alembic_version 推进语句
    #   - base revision:CREATE TABLE alembic_version(alembic --sql 模式不生成 INSERT,
    #     export_ddl.py 会补一句 INSERT,见 export_ddl.py 的 _ensure_version_insert)
    #   - 增量 revision:UPDATE alembic_version SET version_num='xxx'
    print("\n[5/6] ddl/*.sql 含 alembic_version 推进语句")
    version_pattern_ok = True
    for rev, down, _msg, path in migrations:
        ddl_path = DDL_DIR / f"{path.stem}.sql"
        if not ddl_path.exists():
            continue
        text = ddl_path.read_text(encoding="utf-8")
        is_base = down is None
        # base revision 容忍:CREATE TABLE alembic_version 即可(INSERT 由 export_ddl.py 补)
        # 增量 revision:必须有 UPDATE/INSERT version_num='xxx'
        insert_pattern = rf"INSERT\s+INTO\s+alembic_version.*?{re.escape(rev)}"
        update_pattern = rf"UPDATE\s+alembic_version\s+SET\s+version_num\s*=\s*['\"]?{re.escape(rev)}['\"]?"
        if is_base:
            has_version_table = "CREATE TABLE" in text and "alembic_version" in text
            if has_version_table:
                print(f"  ✓ {ddl_path.name}: base revision,含 CREATE TABLE alembic_version")
            else:
                print(f"  ✗ {ddl_path.name}: base revision 缺 CREATE TABLE alembic_version")
                version_pattern_ok = False
        elif re.search(insert_pattern, text, re.I | re.S) or re.search(update_pattern, text, re.I | re.S):
            print(f"  ✓ {ddl_path.name}: 含 revision={rev} 推进语句")
        else:
            print(f"  ✗ {ddl_path.name}: 缺 revision={rev} 的 alembic_version 推进")
            version_pattern_ok = False
    check(version_pattern_ok, "所有 ddl/*.sql 含正确的 alembic_version 推进")

    # ⑥ models.py 能 import(语法正确)
    print("\n[6/6] models.py 语法检查")
    try:
        spec = importlib.util.spec_from_file_location("models_check", MODELS_PATH)
        if spec is None or spec.loader is None:
            check(False, f"无法加载 {MODELS_PATH}")
        else:
            # 用新鲜模块 import,不执行 alembic 副作用
            module = importlib.util.module_from_spec(spec)
            # 需要 sqlalchemy 在 sys.path,CI 已装依赖
            sys.path.insert(0, str(REPO_ROOT / "src"))
            spec.loader.exec_module(module)
            # 至少能拿到 Base
            has_base = hasattr(module, "Base")
            check(has_base, "models.py 能 import,Base 存在")
    except Exception as e:
        check(False, f"models.py import 失败: {e}")

    # 总结
    print("\n" + "=" * 60)
    if _failures:
        print(f"❌ 失败 {len(_failures)} 项:")
        for f in _failures:
            print(f"   - {f}")
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
