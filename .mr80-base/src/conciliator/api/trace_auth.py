"""Optional read-only HTTP Basic boundary; independent of MySQL authentication.

With no configured viewers, this service relies on its existing internal R&D
network ingress. Never deploy this mode on a public ingress.
"""

import base64
import binascii
import secrets

from fastapi import HTTPException, Request


def require_viewer(request: Request):
    viewers = request.app.state.settings.trace_viewers
    if not viewers:
        return
    try:
        scheme, encoded = request.headers.get("authorization", "").split(" ", 1)
        if scheme.lower() != "basic":
            raise ValueError()
        username, password = base64.b64decode(encoded, validate=True).decode().split(":", 1)
        expected = viewers.get(username)
        if expected is not None and secrets.compare_digest(password.encode(), expected.get_secret_value().encode()):
            return
    except (ValueError, UnicodeError, binascii.Error):
        pass
    raise HTTPException(
        401, "需要只读查看权限", headers={"WWW-Authenticate": 'Basic realm="Conciliator trace", charset="UTF-8"'}
    )
