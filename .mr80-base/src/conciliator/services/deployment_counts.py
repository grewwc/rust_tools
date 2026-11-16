"""Machine-wide deployment counts; cleanup is an overlapping indicator."""

from datetime import UTC, datetime


def devbox_result_kind(deployment):
    if not deployment:
        return None
    if deployment.get("cleanup_state") == "succeeded" and (
        deployment.get("released") or deployment.get("stage") == "maat_cleanup_records_deleted"
    ):
        return "released"
    if deployment.get("stage") == "maat_cleanup_unknown":
        return "release_unknown"
    if deployment.get("cleanup_state") == "manual_required" and deployment.get("stage") in {
        "maat_cleanup_todo", "cleanup_pending",
    }:
        return "cleanup"
    if deployment.get("superseded"):
        return None
    if deployment.get("deployment_failed") or deployment.get("result") or deployment.get("status") == "failed":
        return "failure"
    if deployment.get("stage") in {"maat_write_unknown", "maat_name_conflict"}:
        return "unknown"
    if deployment.get("released") or deployment.get("resource_releasing") or deployment.get("release_requested"):
        return None
    if deployment.get("maat_url") and (
        deployment.get("status") == "succeeded" and deployment.get("current", True)
        or deployment.get("stage") == "maat_entry_pending"
        and deployment.get("maat_environment_status") in {"deploy_file_ok", "deploy_static_ok"}
    ):
        return "success"
    return None


def local_deployment_released(record):
    data = record.data or {}
    return bool(data.get("released") or (
        data.get("protocol_version") == 2 and data.get("release_requested")
        and data.get("stop_ack") == record.generation
    ))


def v2_entry_ready(data, now):
    if (
        not data.get("maat_entry_verified") or data.get("maat_write_intent")
        or data.get("release_requested") or data.get("released")
    ):
        return False
    observation = data.get("observation") or {}
    if observation.get("service_status") != "healthy":
        return False
    try:
        stamp = datetime.fromisoformat(observation["observed_at"])
        stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
        clock = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
        age = (clock - stamp).total_seconds()
        return -5 <= age <= 30
    except (KeyError, TypeError, ValueError):
        return False


def deployment_counts(records, *, observations_fresh=True, now=None):
    counts = dict(deploying=0, running=0, stopping=0, unknown=0, cleanup_pending=0, total=0)
    services: dict[str, int] = {}
    for record in records:
        data = record.data or {}
        if data.get("protocol_version") != 2:
            continue
        stopped = data.get("stop_ack") == record.generation
        cleanup = data.get("cleanup_state")
        if cleanup in {"pending", "running", "unknown", "manual_required"}:
            counts["cleanup_pending"] += 1
        if stopped or data.get("released") or record.status == "cancelled" and not data.get("claimed"):
            continue
        observed = data.get("observation") or {}
        service_status = observed.get("service_status", "unknown")
        if now is not None and observed.get("observed_at"):
            try:
                if (now - datetime.fromisoformat(observed["observed_at"])).total_seconds() > 30:
                    service_status = "unknown"
            except (TypeError, ValueError):
                service_status = "unknown"
        if not data.get("claimed"):
            category = "deploying" if record.status == "running" else None
        elif record.status == "cancel_requested" or data.get("release_requested"):
            category = "stopping"
        elif not observations_fresh or service_status == "unknown":
            category = "unknown"
        elif service_status in {"healthy", "unhealthy"} and data.get("endpoint"):
            category = "running"
        elif record.status == "running":
            category = "deploying"
        elif record.status == "succeeded":
            category = "running"
        elif record.status == "failed":
            category = "unknown"
        else:
            category = None
        if category:
            counts[category] += 1
            counts["total"] += 1
            repository = data.get("repository", "dp/aeolus")
            services[repository] = services.get(repository, 0) + 1
    return {**counts, "by_repository": services}
