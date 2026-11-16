"""Shared service defaults for multi-instance runtime profiles."""

ALIASES = {"aeolus": "dp/aeolus", "vqs": "dp/VizQueryService"}
SERVICE_DEFAULTS = {
    "dp/aeolus": {"service_identity": "data.aeolus.api_v3"},
    "dp/VizQueryService": {
        "service_identity": "data.vqs.api",
        "readiness_json": {"code": "aeolus/ok"},
    },
}
VQS_DEPENDENCIES = ("aeolus_expr_parse", "aeolus_kms", "sysconf", "envmanager")
