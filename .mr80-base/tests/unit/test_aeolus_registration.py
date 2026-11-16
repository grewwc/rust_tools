from conciliator.main import create_app


def test_aeolus_routes_are_registered():
    def paths(routes):
        result = set()
        for route in routes:
            if hasattr(route, "path"):
                result.add(route.path)
            nested = getattr(route, "original_router", None)
            if nested is not None:
                result.update(paths(nested.routes))
        return result

    paths = paths(create_app().routes)
    assert "/hooks/meegle/ai-node" in paths
    assert "/api/aeolus/tasks/by-node" in paths
