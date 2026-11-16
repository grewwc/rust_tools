from conciliator.integrations.lark.client import LarkClient


class LarkUserNotFoundError(LookupError):
    pass


async def get_user_email_by_open_id(
    client: LarkClient, open_id: str,
) -> str | None:
    data = await client._request(
        "GET",
        f"/contact/v3/users/{open_id}",
        params={"user_id_type": "open_id"},
    )
    user = data.get("user", {})
    return user.get("email") or user.get("enterprise_email")


async def get_user_open_id_by_email(
    client: LarkClient,
    email: str,
) -> str:
    """Resolve one active tenant user email to an open_id with app identity."""
    normalized = email.strip().lower()
    if not normalized or "@" not in normalized:
        raise ValueError("email must be a non-empty email address")
    data = await client._request(
        "POST",
        "/contact/v3/users/batch_get_id",
        params={"user_id_type": "open_id"},
        json={"emails": [normalized], "include_resigned": False},
    )
    users = data.get("user_list") or []
    matched = next(
        (
            user
            for user in users
            if str(user.get("email") or "").strip().lower() == normalized
        ),
        users[0] if len(users) == 1 else None,
    )
    open_id = str((matched or {}).get("user_id") or "").strip()
    if not open_id:
        raise LarkUserNotFoundError(
            f"no active Lark user found for email {normalized}"
        )
    return open_id
