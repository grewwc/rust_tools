"""Token binding invariants and offline migration/DDL parity (no MySQL required)."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import re
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from threading import Barrier, Lock

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.exc import IntegrityError, MultipleResultsFound
from sqlalchemy.orm import sessionmaker

from conciliator.repositories.lark_user_token_repo import (
    LarkUserTokenRepository,
    StoredUserToken,
)
from conciliator.repositories.models import LarkUserToken

ROOT = Path(__file__).resolve().parents[2]


def _token(user_key="user-a", open_id="open-a", suffix="old"):
    return StoredUserToken(
        user_key=user_key,
        open_id=open_id,
        access_token=f"access-{suffix}",
        refresh_token=f"refresh-{suffix}",
        access_expires_at=datetime(2030, 1, 1),
        refresh_expires_at=datetime(2030, 2, 1),
    )


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'tokens.db'}")
    LarkUserToken.__table__.create(engine)
    factory = sessionmaker(bind=engine)
    yield LarkUserTokenRepository(factory), factory
    engine.dispose()


async def test_save_updates_by_user_key_and_preserves_one_row(store):
    repo, factory = store
    assert await repo.find_by_user_key("user-a") is None
    original = _token()
    await repo.save(original)
    assert await repo.find_by_user_key("user-a") == original
    with factory() as session:
        original_id = session.scalar(select(LarkUserToken.id))

    renewed = replace(
        original,
        open_id="open-renewed",
        access_token="access-new",
        refresh_token="refresh-new",
        access_expires_at=datetime(2031, 1, 1),
        refresh_expires_at=None,
    )
    await repo.save(renewed)
    assert await repo.find_by_user_key("user-a") == renewed
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(LarkUserToken)) == 1
        assert session.scalar(select(LarkUserToken.id)) == original_id


@pytest.mark.parametrize("existing_user", [False, True])
async def test_open_id_conflict_does_not_transfer_or_overwrite_binding(store, existing_user):
    repo, factory = store
    victim = _token()
    other = _token(user_key="user-b", open_id="open-b", suffix="other")
    await repo.save(victim)
    if existing_user:
        await repo.save(other)

    with pytest.raises(IntegrityError):
        await repo.save(replace(other, open_id=victim.open_id, access_token="wrong-user"))

    assert await repo.find_by_user_key(victim.user_key) == victim
    assert await repo.find_by_user_key(other.user_key) == (other if existing_user else None)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(LarkUserToken)) == 1 + existing_user


@pytest.mark.parametrize(
    "duplicate", [_token(open_id="open-other"), _token(user_key="user-other")],
    ids=["user-key", "open-id"],
)
async def test_database_enforces_both_unique_keys(store, duplicate):
    repo, factory = store
    await repo.save(_token())
    with factory() as session:
        session.add(LarkUserToken(**asdict(duplicate)))
        with pytest.raises(IntegrityError):
            session.commit()
    assert await repo.find_by_user_key("user-a") == _token()


async def test_delete_only_removes_requested_user(store):
    repo, _ = store
    other = _token(user_key="user-b", open_id="open-b")
    await repo.save(_token())
    await repo.save(other)
    await repo.delete_by_user_key("missing")
    await repo.delete_by_user_key("user-a")
    assert await repo.find_by_user_key("user-a") is None
    assert await repo.find_by_user_key("user-b") == other


@pytest.mark.parametrize("same_user", [True, False], ids=["same-user", "open-id-conflict"])
async def test_concurrent_first_save_cannot_duplicate_or_transfer_binding(store, same_user):
    repo, factory = store
    barrier = Barrier(2)
    counter_lock = Lock()
    insert_attempts = 0

    def synchronize_first_inserts(session, flush_context, instances):
        nonlocal insert_attempts
        if not any(isinstance(row, LarkUserToken) for row in session.new):
            return
        with counter_lock:
            insert_attempts += 1
            attempt = insert_attempts
        # Both transactions have observed a missing user_key before either INSERT.
        # A conflict retry must not wait on a writer that already finished.
        if attempt <= 2:
            barrier.wait(timeout=5)

    first = _token(suffix="first")
    second = _token(user_key="user-a" if same_user else "user-b", suffix="second")
    event.listen(factory, "before_flush", synchronize_first_inserts)
    try:
        results = await asyncio.gather(repo.save(first), repo.save(second), return_exceptions=True)
    finally:
        event.remove(factory, "before_flush", synchronize_first_inserts)

    if same_user:
        assert results == [None, None]
        assert await repo.find_by_user_key("user-a") in (first, second)
    else:
        assert sum(result is None for result in results) == 1
        assert sum(isinstance(result, IntegrityError) for result in results) == 1
        for token, result in zip((first, second), results, strict=True):
            assert await repo.find_by_user_key(token.user_key) == (token if result is None else None)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(LarkUserToken)) == 1


@pytest.mark.parametrize("refresh_expiry", [None, datetime(2030, 2, 1)])
async def test_authorization_save_replaces_full_snapshot_after_interleaved_refresh(
    store, refresh_expiry,
):
    repo, factory = store
    original = replace(_token(), refresh_expires_at=refresh_expiry)
    renewed = replace(original, access_token="access-new", refresh_token="refresh-new")
    refreshed = replace(_token(suffix="refreshed"), refresh_expires_at=datetime(2032, 2, 1))
    await repo.save(original)
    refresh_writes = []

    def refresh_after_authorization_read(session, record):
        if isinstance(record, LarkUserToken):
            # The consent writer has read the old row but has not written yet.
            refresh_writes.append(repo._replace_if_current(original, refreshed))

    event.listen(factory, "loaded_as_persistent", refresh_after_authorization_read)
    try:
        await repo.save(renewed)
    finally:
        event.remove(factory, "loaded_as_persistent", refresh_after_authorization_read)

    assert refresh_writes == [True]
    assert await repo.find_by_user_key(original.user_key) == renewed


@pytest.mark.parametrize("nullable_expiry", [False, True])
async def test_conditional_replacement_matches_complete_snapshot(store, nullable_expiry):
    repo, _ = store
    original = _token()
    if nullable_expiry:
        original = replace(original, access_expires_at=None, refresh_expires_at=None)
    await repo.save(original)
    renewed = _token(suffix="renewed")
    assert await repo.replace_if_current(original, renewed) is True
    assert await repo.find_by_user_key(original.user_key) == renewed
    assert await repo.replace_if_current(original, _token(suffix="stale")) is False
    assert await repo.find_by_user_key(original.user_key) == renewed


@pytest.mark.parametrize("changed", [
    {"access_token": "ACCESS-old"},
    {"refresh_token": "REFRESH-old"},
    {"access_expires_at": None},
    {"refresh_expires_at": None},
    {"open_id": "open-new"},
])
async def test_conditional_replacement_preserves_newer_authorization(store, changed):
    repo, _ = store
    original = _token()
    await repo.save(original)
    newer = replace(original, **changed)
    await repo.save(newer)
    invalidated = replace(original, access_token="", refresh_token="")
    assert await repo.replace_if_current(original, invalidated) is False
    assert await repo.find_by_user_key(original.user_key) == newer


@pytest.mark.parametrize("changed", [{"user_key": "other"}, {"open_id": "other"}])
async def test_conditional_replacement_rejects_identity_changes(store, changed):
    repo, _ = store
    original = _token()
    await repo.save(original)
    with pytest.raises(ValueError, match="cannot change identity"):
        await repo.replace_if_current(original, replace(original, **changed))
    assert await repo.find_by_user_key(original.user_key) == original


async def test_concurrent_conditional_replacements_have_one_winner(store):
    repo, factory = store
    other = LarkUserTokenRepository(factory)
    original = _token()
    await repo.save(original)
    replacements = [_token(suffix="first"), _token(suffix="second")]
    results = await asyncio.gather(
        repo.replace_if_current(original, replacements[0]),
        other.replace_if_current(original, replacements[1]),
    )
    assert sorted(results) == [False, True]
    assert await repo.find_by_user_key(original.user_key) == replacements[results.index(True)]


async def test_conditional_replacement_does_not_recreate_removed_authorization(store):
    repo, _ = store
    original = _token()
    assert await repo.replace_if_current(original, _token(suffix="new")) is False
    assert await repo.find_by_user_key(original.user_key) is None


def _migration(revision):
    path = ROOT / "migrations" / "versions" / f"{revision}.py"
    spec = importlib.util.spec_from_file_location(revision, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_legacy_duplicate_user_keys_fail_closed_without_deleting_data(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    try:
        with (
            engine.begin() as connection,
            Operations.context(MigrationContext.configure(connection)),
        ):
            _migration("0015_lark_user_tokens").upgrade()
        factory = sessionmaker(bind=engine)
        with factory() as session:
            session.add_all([
                LarkUserToken(**asdict(_token())),
                LarkUserToken(**asdict(_token(open_id="open-other"))),
            ])
            session.commit()
        repo = LarkUserTokenRepository(factory)
        with pytest.raises(MultipleResultsFound):
            await repo.find_by_user_key("user-a")
        with pytest.raises(MultipleResultsFound):
            await repo.save(_token())
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(LarkUserToken)) == 2
    finally:
        engine.dispose()


def _normalize_sql(text):
    return re.sub(r"\s+", "", re.sub(r"--[^\n]*", "", text)).lower()


@pytest.mark.parametrize("revision", ["0015_lark_user_tokens", "0016_lark_token_user_unique"])
def test_incremental_ddl_matches_mysql_offline_migration(revision):
    migration = _migration(revision)
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="mysql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        migration.upgrade()
    expected = output.getvalue() + (
        f"UPDATE alembic_version SET version_num='{revision}' "
        f"WHERE alembic_version.version_num = '{migration.down_revision}';"
    )
    assert _normalize_sql((ROOT / "ddl" / f"{revision}.sql").read_text()) == _normalize_sql(expected)


def test_unique_user_migration_is_additive_and_downgrade_preserves_open_id():
    migration = _migration("0016_lark_token_user_unique")
    assert migration.down_revision == "0015_lark_user_tokens"
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="mysql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        migration.upgrade()
        migration.downgrade()
    assert _normalize_sql(output.getvalue()) == _normalize_sql(
        "ALTER TABLE lark_user_tokens ADD CONSTRAINT uk_lark_user_tokens_user_key UNIQUE (user_key);"
        "ALTER TABLE lark_user_tokens DROP INDEX uk_lark_user_tokens_user_key;"
    )
