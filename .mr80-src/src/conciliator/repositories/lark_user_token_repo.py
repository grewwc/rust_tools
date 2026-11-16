from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from sqlalchemy import LargeBinary, cast, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from conciliator.repositories.models import LarkUserToken


@dataclass(frozen=True)
class StoredUserToken:
    """Detached snapshot of one stored Feishu user token."""

    open_id: str
    user_key: str
    refresh_token: str
    access_token: str
    access_expires_at: datetime | None
    refresh_expires_at: datetime | None


def _snapshot(record: LarkUserToken) -> StoredUserToken:
    return StoredUserToken(
        open_id=record.open_id,
        user_key=record.user_key,
        refresh_token=record.refresh_token,
        access_token=record.access_token,
        access_expires_at=record.access_expires_at,
        refresh_expires_at=record.refresh_expires_at,
    )


class LarkUserTokenRepository:
    """Stores one token per Meegle user_key, with an exclusive open_id binding."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sf = session_factory

    def _find(self, user_key: str) -> StoredUserToken | None:
        with self._sf() as session:
            record = session.scalars(
                select(LarkUserToken).where(LarkUserToken.user_key == user_key)
            ).one_or_none()
            return _snapshot(record) if record is not None else None

    async def find_by_user_key(self, user_key: str) -> StoredUserToken | None:
        return await run_in_threadpool(self._find, user_key)

    def _save(self, token: StoredUserToken) -> None:
        for attempt in range(2):
            with self._sf() as session:
                record = session.scalars(
                    select(LarkUserToken).where(LarkUserToken.user_key == token.user_key)
                ).one_or_none()
                inserting = record is None
                try:
                    if record is None:
                        session.add(LarkUserToken(**asdict(token)))
                    else:
                        # A refresh can commit after this row was read. Write
                        # the complete consent snapshot, including unchanged
                        # or unknown expiries, rather than only ORM dirty fields.
                        session.execute(
                            update(LarkUserToken)
                            .where(LarkUserToken.id == record.id)
                            .values(**asdict(token))
                            .execution_options(synchronize_session=False)
                        )
                    session.commit()
                    return
                except IntegrityError:
                    if not inserting or attempt:
                        raise
                    # Concurrent first consent may have inserted this user_key.
                    # Retry in a fresh transaction; never look up by open_id,
                    # which could transfer another user's existing binding.

    async def save(self, token: StoredUserToken) -> None:
        await run_in_threadpool(self._save, token)

    def _replace_if_current(self, expected: StoredUserToken, token: StoredUserToken) -> bool:
        if (token.user_key, token.open_id) != (expected.user_key, expected.open_id):
            raise ValueError("Conditional token replacement cannot change identity")
        # Compare opaque credentials byte-for-byte even with MySQL's default
        # case-insensitive text collation. One UPDATE makes the check atomic.
        statement = (
            update(LarkUserToken)
            .where(
                LarkUserToken.user_key == expected.user_key,
                cast(LarkUserToken.open_id, LargeBinary) == expected.open_id.encode(),
                cast(LarkUserToken.access_token, LargeBinary) == expected.access_token.encode(),
                cast(LarkUserToken.refresh_token, LargeBinary) == expected.refresh_token.encode(),
                LarkUserToken.access_expires_at == expected.access_expires_at,
                LarkUserToken.refresh_expires_at == expected.refresh_expires_at,
            )
            .values(
                access_token=token.access_token,
                refresh_token=token.refresh_token,
                access_expires_at=token.access_expires_at,
                refresh_expires_at=token.refresh_expires_at,
            )
            .execution_options(synchronize_session=False)
        )
        with self._sf() as session:
            result = session.execute(statement)
            session.commit()
            return result.rowcount == 1  # type: ignore[attr-defined]

    async def replace_if_current(self, expected: StoredUserToken, token: StoredUserToken) -> bool:
        """Replace credentials only if no authorization or refresh superseded them."""
        return await run_in_threadpool(self._replace_if_current, expected, token)

    def _delete(self, user_key: str) -> None:
        with self._sf() as session:
            record = session.scalar(
                select(LarkUserToken).where(LarkUserToken.user_key == user_key)
            )
            if record is not None:
                session.delete(record)
                session.commit()

    async def delete_by_user_key(self, user_key: str) -> None:
        await run_in_threadpool(self._delete, user_key)
