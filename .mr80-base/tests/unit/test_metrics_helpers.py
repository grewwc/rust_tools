from conciliator.utils import metrics


def test_emit_counter_without_agent_does_not_raise():
    # 测试环境无 bytedance.metrics,应静默 no-op
    metrics.emit_counter("foo", 1, {"a": "b"})
    metrics.emit_timer("bar", 1.5, {})
    metrics.emit_gauge("baz", 10, {})


def test_metric_names_centralized():
    assert metrics.MetricName.TASKS_TOTAL == "conciliator.tasks"
    assert metrics.MetricName.APPROVAL_TOTAL == "conciliator.approval"
