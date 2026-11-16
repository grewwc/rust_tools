from fastapi.middleware.cors import CORSMiddleware

from conciliator.main import create_app


def test_plugin_asset_origin_is_allowed() -> None:
    cors = next(
        middleware
        for middleware in create_app().user_middleware
        if middleware.cls is CORSMiddleware
    )

    assert "https://meegoplg.larkofficepkg.com" in cors.kwargs[
        "allow_origins"
    ]
