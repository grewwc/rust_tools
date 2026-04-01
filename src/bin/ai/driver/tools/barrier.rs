use crate::ai::types::ToolCall;

use super::ToolRoute;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BarrierRule {
    Always,
    OnSuccessNonEmptyOutput,
    Never,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolRouteKind {
    Builtin,
    Mcp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BarrierSpec {
    route: ToolRouteKind,
    tool_name: &'static str,
    rule: BarrierRule,
}

const BARRIER_SPECS: &[BarrierSpec] = &[
    BarrierSpec {
        route: ToolRouteKind::Builtin,
        tool_name: "search_files",
        rule: BarrierRule::OnSuccessNonEmptyOutput,
    },
    BarrierSpec {
        route: ToolRouteKind::Builtin,
        tool_name: "list_directory",
        rule: BarrierRule::Always,
    },
    BarrierSpec {
        route: ToolRouteKind::Builtin,
        tool_name: "web_search",
        rule: BarrierRule::Always,
    },
];

fn route_kind(route: &ToolRoute) -> ToolRouteKind {
    match route {
        ToolRoute::Builtin => ToolRouteKind::Builtin,
        ToolRoute::Mcp { .. } => ToolRouteKind::Mcp,
    }
}

fn barrier_rule(route: &ToolRoute, tool_name: &str) -> BarrierRule {
    if matches!(route, ToolRoute::Mcp { .. }) {
        return BarrierRule::Always;
    }

    for spec in BARRIER_SPECS {
        if spec.route == route_kind(route) && spec.tool_name == tool_name {
            return spec.rule;
        }
    }
    BarrierRule::Never
}

pub(super) fn should_barrier_after(
    route: &ToolRoute,
    tool_call: &ToolCall,
    ok: bool,
    content: &str,
) -> bool {
    match barrier_rule(route, &tool_call.function.name) {
        BarrierRule::Always => true,
        BarrierRule::OnSuccessNonEmptyOutput => ok && !content.trim().is_empty(),
        BarrierRule::Never => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    #[test]
    fn barrier_builtin_search_files_requires_non_empty_success_output() {
        let tc = tool_call("search_files");
        assert!(!should_barrier_after(&ToolRoute::Builtin, &tc, true, "   "));
        assert!(!should_barrier_after(
            &ToolRoute::Builtin,
            &tc,
            false,
            "/tmp/a.rs"
        ));
        assert!(should_barrier_after(
            &ToolRoute::Builtin,
            &tc,
            true,
            "/tmp/a.rs"
        ));
    }

    #[test]
    fn barrier_builtin_and_mcp_rules_match_existing_behavior() {
        assert!(should_barrier_after(
            &ToolRoute::Builtin,
            &tool_call("list_directory"),
            true,
            ""
        ));
        assert!(!should_barrier_after(
            &ToolRoute::Builtin,
            &tool_call("read_file"),
            true,
            "content"
        ));
        assert!(should_barrier_after(
            &ToolRoute::Mcp {
                server_name: "foo".to_string(),
                tool_name: "bar".to_string(),
            },
            &tool_call("mcp_foo_bar"),
            false,
            ""
        ));
    }
}
