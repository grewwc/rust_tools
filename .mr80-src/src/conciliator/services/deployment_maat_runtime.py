"""Application lifecycle for public-OpenAPI MAAT creation and record cleanup."""

import asyncio
import contextlib
import json
from pathlib import Path

from conciliator.integrations.maat.auth import SdkMaatTokenProvider, application_http_client
from conciliator.integrations.maat.client import (
    MAAT_GATEWAY_ORIGIN,
    MaatError,
    service_environment_plan,
    template_revision,
)
from conciliator.integrations.maat.control_plane import (
    MaatOpenAPIControlPlaneClient,
    environment_body,
    environment_url,
)
from conciliator.services.deployment_maat import MaatDeploymentCoordinator
from conciliator.utils.logging import get_logger

log = get_logger(__name__)
SERVICE_PATHS = {
    "dp/aeolus": ("api/v3/misc", "aeolus/api/v3", "aeolus/openApi", "aeolus/llm/api/v2"),
    "dp/VizQueryService": ("aeolus/vqs",),
}


class DeploymentMaatRuntime:
    def __init__(self, coordinator, http, *, interval=5, notifier=None):
        self.coordinator = coordinator
        self.http = http
        self.notifier = notifier
        self.interval = interval
        self._task = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="maat:deployment")

    async def _loop(self):
        while True:
            try:
                results = await self.coordinator.run_once()
                for result in results:
                    if isinstance(result, Exception):
                        log.warning("maat_deployment_operation_failed", error_type=type(result).__name__)
                if self.notifier:
                    await asyncio.gather(*(
                        self._notify(task_id) for task_id in {r for r in results if isinstance(r, str)}
                    ))
            except Exception as exc:
                log.warning("maat_deployment_scan_failed", error_type=type(exc).__name__)
            await asyncio.sleep(self.interval)

    async def _notify(self, task_id):
        try:
            await asyncio.wait_for(self.notifier(task_id), 5)
        except Exception as exc:
            log.warning("maat_deployment_notification_failed", task_id=task_id, error_type=type(exc).__name__)

    async def aclose(self):
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.http.aclose()


async def build_maat_runtime(settings, service, secrets):
    operator = settings.maat_action_user.strip()
    if not operator:
        raise MaatError("MAAT_ACTION_USER must identify a project-authorized operator")
    provider = SdkMaatTokenProvider(settings, secrets, proxy_url=settings.maat_auth_proxy or None)
    http = application_http_client(provider, MAAT_GATEWAY_ORIGIN, trust_env=False)
    try:
        control = MaatOpenAPIControlPlaneClient(http, operator)
        snapshot = json.loads(await asyncio.to_thread(Path(settings.maat_baseline_file).read_text))
        template, project_config = snapshot["environment"], snapshot["project_config"]
        if (
            template.get("name") != settings.maat_baseline_environment
            or template.get("projectName") != "aeolus" or template.get("region") != "CN"
            or template.get("isDeleted")
        ):
            raise MaatError("MAAT local baseline identity does not match")
        if template_revision(template) != settings.maat_baseline_revision:
            raise MaatError("MAAT baseline changed; review before enabling creation")
        environment_url(template, project_config)
        bindings = {}
        for repository, paths in SERVICE_PATHS.items():
            selected = []
            for path in paths:
                rules = [r for r in template["apiProxyList"] if r.get("pathname") == path]
                # Older reference environments do not expose the optional LLM API.
                if not rules and path == "aeolus/llm/api/v2":
                    continue
                if len(rules) != 1 or rules[0].get("pathMatchType") not in {"prefix", "regexp"}:
                    raise MaatError("MAAT service binding is ambiguous")
                selected.append((path, rules[0]["pathMatchType"], rules[0]["apiId"]))
            bindings[repository] = selected
        for repository in bindings:
            plan = service_environment_plan(
                template, repository, "preflight-only", bindings=bindings,
                expected_template_revision=settings.maat_baseline_revision,
            )
            environment_body(template, plan, name="hua-ai-preflight", action_user=operator)

        async def baseline_provider():
            # Keep the reviewed snapshot stable; never silently adopt live baseline changes.
            return template, project_config

        coordinator = MaatDeploymentCoordinator(
            service, openapi=control, control_plane=control, baseline_provider=baseline_provider,
            baseline_revision=settings.maat_baseline_revision, service_bindings=bindings,
            proxy_options={r: {"protocol": "http", "is_socket": False} for r in bindings},
            openapi_only=True,
        )
        return DeploymentMaatRuntime(coordinator, http)
    except BaseException:
        await http.aclose()
        raise
