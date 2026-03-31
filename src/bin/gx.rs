use std::fs;
use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};

use clap::{CommandFactory, Parser};
use rust_tools::cw::{
    DirectedGraph, Edge, Mst, SkipSet, SkipMap, UndirectedGraph, WeightedDirectedGraph,
    WeightedUndirectedGraph,
};
use serde::Serialize;

#[derive(Parser)]
#[command(
    about = "Graph analysis utilities (go_tools graphx/gx compatible subset)",
    after_help = "modes:\n  cycle      detect cycle\n  scc        strongly connected components (directed)\n  topo       topological sort (directed DAG)\n  sp         shortest path: requires -from and -to\n  cc         connected components\n  mst        minimum spanning tree (undirected weighted)\n\ninput edge format:\n  unweighted: <from> <to>\n  weighted:   <from> <to> <weight>\n  comments:   lines starting with # or //\n\nextra output:\n  --fmt text|json|csv   result output format\n  --viz dot|mermaid     include graph visualization text\n  --out <file>          write output to file\n  --dir <path>          batch analyze all graph files in directory\n"
)]
struct Cli {
    #[arg(short = 'f', value_name = "FILE", num_args = 0..=1, default_missing_value = "")]
    file: Option<String>,

    #[arg(long, default_value = "", value_name = "DIR")]
    dir: String,

    #[arg(
        long,
        default_value = ".txt,.graph,.edgelist,.csv",
        value_name = "EXTS"
    )]
    ext: String,

    #[arg(
        short = 'r',
        default_value_t = false,
        help = "recursive scan for --dir"
    )]
    recursive: bool,

    #[arg(long, default_value = "", value_name = "NODE")]
    from: String,

    #[arg(long, default_value = "", value_name = "NODE")]
    to: String,

    #[arg(short = 'u', default_value_t = false, help = "undirected graph")]
    undirected: bool,

    #[arg(
        short = 'w',
        default_value_t = false,
        help = "weighted edges: <from> <to> <weight>"
    )]
    weighted: bool,

    #[arg(
        long,
        default_value = "",
        value_name = "SEP",
        help = "field separator, default whitespace"
    )]
    sep: String,

    #[arg(long = "fmt", default_value = "text", value_name = "FMT")]
    format: String,

    #[arg(
        long,
        default_value = "",
        value_name = "VIZ",
        help = "visualization output: dot|mermaid"
    )]
    viz: String,

    #[arg(
        long,
        default_value = "",
        value_name = "FILE",
        help = "write result to file"
    )]
    out: String,

    #[arg(value_name = "MODE [FILE]", num_args = 0..=2)]
    positional: Vec<String>,
}

#[derive(Clone, Debug)]
struct GraphEdge {
    from: String,
    to: String,
    weight: f64,
}

#[derive(Clone, Debug, Default)]
struct AnalysisOptions {
    mode: String,
    from: String,
    to: String,
    weighted: bool,
    undirected: bool,
    viz: String,
    source: String,
}

#[derive(Clone, Debug, Serialize, Default)]
struct EdgeOut {
    from: String,
    to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Default)]
struct AnalysisResult {
    #[serde(skip_serializing_if = "String::is_empty")]
    source: String,
    mode: String,
    nodes: usize,
    edges: usize,
    weighted: bool,
    undirected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    has_cycle: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cycle: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    components: Vec<Vec<String>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    order: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    path: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hops: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_weight: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    mst_edges: Vec<EdgeOut>,
    #[serde(skip_serializing_if = "String::is_empty")]
    warning: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    error: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    viz: String,
}

fn main() {
    let cli = Cli::parse();

    let Some(mode) = cli.positional.first().cloned() else {
        let mut cmd = Cli::command();
        cmd.print_help().ok();
        println!();
        std::process::exit(0);
    };

    let mode = mode.to_lowercase();
    let (input_file, file_flag_set) = resolve_input_file(cli.file, &cli.positional);
    if file_flag_set && input_file.is_empty() {
        eprintln!("flag -f requires a file path; remove -f to read from stdin");
        std::process::exit(1);
    }

    let mut viz = cli.viz.to_lowercase();
    if viz == "mmd" {
        viz = "mermaid".to_string();
    }
    if !viz.is_empty() && viz != "dot" && viz != "mermaid" {
        eprintln!("unsupported viz {:?}, use dot|mermaid", viz);
        std::process::exit(1);
    }

    let format = cli.format.to_lowercase();
    if format != "text" && format != "json" && format != "csv" {
        eprintln!("unsupported format {:?}, use text|json|csv", format);
        std::process::exit(1);
    }

    let weighted = cli.weighted || mode == "mst";
    let undirected = cli.undirected || mode == "cc" || mode == "components" || mode == "mst";

    let opts = AnalysisOptions {
        mode: mode.clone(),
        from: cli.from.clone(),
        to: cli.to.clone(),
        weighted,
        undirected,
        viz,
        source: String::new(),
    };

    let files = match collect_input_files(&input_file, &cli.dir, &cli.ext, cli.recursive) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("failed to collect input files: {err}");
            std::process::exit(1);
        }
    };

    let mut results = Vec::new();
    if files.is_empty() {
        let source = if input_file.is_empty() {
            "<stdin>".to_string()
        } else {
            input_file.clone()
        };

        if input_file.is_empty() && stdin_is_tty() {
            eprintln!("reading edge list from stdin, press Ctrl-D to finish");
        }

        let input = match open_input(&input_file) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("failed to open input: {err}");
                std::process::exit(1);
            }
        };
        let mut local_opts = opts.clone();
        local_opts.source = source;
        results.push(analyze_single_source(input, &local_opts, &cli.sep));
    } else {
        for path in files {
            match fs::File::open(&path) {
                Ok(f) => {
                    let mut local_opts = opts.clone();
                    local_opts.source = path.to_string_lossy().to_string();
                    results.push(analyze_single_source(Box::new(f), &local_opts, &cli.sep));
                }
                Err(err) => {
                    results.push(AnalysisResult {
                        source: path.to_string_lossy().to_string(),
                        mode: mode.clone(),
                        weighted,
                        undirected,
                        error: err.to_string(),
                        ..AnalysisResult::default()
                    });
                }
            }
        }
    }

    let output = match render_results(&results, &format) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("failed to render output: {err}");
            std::process::exit(1);
        }
    };

    if !cli.out.is_empty() {
        if let Err(err) = fs::write(&cli.out, output.as_bytes()) {
            eprintln!("failed to write output file: {err}");
            std::process::exit(1);
        }
    } else {
        print!("{output}");
    }

    if results.iter().any(|r| !r.error.is_empty()) {
        std::process::exit(1);
    }
}

fn resolve_input_file(file: Option<String>, positional: &[String]) -> (String, bool) {
    if let Some(v) = file {
        return (v.trim().to_string(), true);
    }
    let from_pos = positional
        .get(1)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    (from_pos, false)
}

fn open_input(path: &str) -> io::Result<Box<dyn Read>> {
    if path.trim().is_empty() {
        Ok(Box::new(io::stdin()))
    } else {
        Ok(Box::new(fs::File::open(path)?))
    }
}

fn stdin_is_tty() -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn analyze_single_source(
    mut reader: Box<dyn Read>,
    opts: &AnalysisOptions,
    sep: &str,
) -> AnalysisResult {
    let mut buf = String::new();
    if let Err(err) = reader.read_to_string(&mut buf) {
        return AnalysisResult {
            source: opts.source.clone(),
            mode: opts.mode.clone(),
            weighted: opts.weighted,
            undirected: opts.undirected,
            error: err.to_string(),
            ..AnalysisResult::default()
        };
    }

    let edges = match parse_graph_edges(buf.as_bytes(), opts.weighted, sep) {
        Ok(v) => v,
        Err(err) => {
            return AnalysisResult {
                source: opts.source.clone(),
                mode: opts.mode.clone(),
                weighted: opts.weighted,
                undirected: opts.undirected,
                error: err,
                ..AnalysisResult::default()
            };
        }
    };

    if edges.is_empty() {
        return AnalysisResult {
            source: opts.source.clone(),
            mode: opts.mode.clone(),
            weighted: opts.weighted,
            undirected: opts.undirected,
            error: "no edges found in input".to_string(),
            ..AnalysisResult::default()
        };
    }

    analyze_graph(&edges, opts)
}

fn parse_graph_edges<R: Read>(
    reader: R,
    weighted: bool,
    sep: &str,
) -> Result<Vec<GraphEdge>, String> {
    let mut res = Vec::new();
    let br = io::BufReader::new(reader);
    for (idx, line) in br.lines().enumerate() {
        let line_no = idx + 1;
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let parts = split_fields(line, sep);
        if parts.len() < 2 {
            return Err(format!(
                "line {line_no}: expected at least 2 columns, got {}",
                parts.len()
            ));
        }

        let mut edge = GraphEdge {
            from: parts[0].clone(),
            to: parts[1].clone(),
            weight: 1.0,
        };

        if weighted {
            if parts.len() < 3 {
                return Err(format!(
                    "line {line_no}: expected 3 columns for weighted graph"
                ));
            }
            edge.weight = parts[2]
                .parse::<f64>()
                .map_err(|e| format!("line {line_no}: invalid weight {:?}: {e}", parts[2]))?;
        }

        res.push(edge);
    }
    Ok(res)
}

fn split_fields(line: &str, sep: &str) -> Vec<String> {
    if sep.is_empty() {
        return line.split_whitespace().map(|s| s.to_string()).collect();
    }
    line.split(sep)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn count_nodes(edges: &[GraphEdge]) -> usize {
    let mut set = SkipSet::new(16);
    for e in edges {
        set.insert(e.from.clone());
        set.insert(e.to.clone());
    }
    set.len()
}

fn sorted_nodes(edges: &[GraphEdge]) -> Vec<String> {
    let mut set = SkipSet::new(16);
    for e in edges {
        set.insert(e.from.clone());
        set.insert(e.to.clone());
    }
    set.to_vec()
}

fn build_directed(edges: &[GraphEdge]) -> DirectedGraph<String> {
    let mut g = DirectedGraph::new();
    for e in edges {
        g.add_edge(e.from.clone(), e.to.clone());
    }
    g
}

fn build_undirected(edges: &[GraphEdge]) -> UndirectedGraph<String> {
    let mut g = UndirectedGraph::new();
    for e in edges {
        g.add_edge(e.from.clone(), e.to.clone());
    }
    g
}

fn build_weighted_directed(edges: &[GraphEdge], undirected: bool) -> WeightedDirectedGraph<String> {
    let mut g = WeightedDirectedGraph::new();
    for e in edges {
        g.add_edge(e.from.clone(), e.to.clone(), e.weight);
        if undirected {
            g.add_edge(e.to.clone(), e.from.clone(), e.weight);
        }
    }
    g
}

fn build_weighted_undirected(edges: &[GraphEdge]) -> WeightedUndirectedGraph<String> {
    let mut g = WeightedUndirectedGraph::new();
    for e in edges {
        g.add_edge(e.from.clone(), e.to.clone(), e.weight);
    }
    g
}

fn normalize_components(mut groups: Vec<Vec<String>>) -> Vec<Vec<String>> {
    groups.retain(|g| !g.is_empty());
    for g in &mut groups {
        g.sort();
    }
    groups.sort_by(|a, b| a[0].cmp(&b[0]).then_with(|| a.len().cmp(&b.len())));
    groups
}

fn analyze_graph(edges: &[GraphEdge], opts: &AnalysisOptions) -> AnalysisResult {
    let mut result = AnalysisResult {
        source: opts.source.clone(),
        mode: opts.mode.clone(),
        nodes: count_nodes(edges),
        edges: edges.len(),
        weighted: opts.weighted,
        undirected: opts.undirected,
        ..AnalysisResult::default()
    };

    match opts.mode.as_str() {
        "cycle" | "cy" => {
            if opts.undirected {
                let g = build_undirected(edges);
                result.has_cycle = Some(g.has_cycle());
            } else {
                let g = build_directed(edges);
                let has = g.has_cycle();
                result.has_cycle = Some(has);
                if has {
                    result.cycle = g.cycle().unwrap_or_default();
                }
            }
        }
        "scc" => {
            if opts.undirected {
                result.error = "scc is for directed graph only".to_string();
            } else {
                let g = build_directed(edges);
                let comps = g.strong_components();
                result.components = normalize_components(
                    comps
                        .into_iter()
                        .map(|mut v| {
                            v.sort();
                            v
                        })
                        .collect(),
                );
            }
        }
        "cc" | "components" => {
            let g = build_undirected(edges);
            let comps = g.groups();
            result.components = normalize_components(comps);
        }
        "topo" | "tsort" => {
            if opts.undirected {
                result.error = "topo is for directed graph only".to_string();
            } else {
                let g = build_directed(edges);
                match g.sorted() {
                    Some(order) => result.order = order,
                    None => {
                        result.error =
                            "graph has cycle, topological order not available".to_string();
                        if let Some(cycle) = g.cycle()
                            && !cycle.is_empty()
                        {
                            result.cycle = cycle;
                        }
                    }
                }
            }
        }
        "sp" | "path" => {
            if opts.from.trim().is_empty() || opts.to.trim().is_empty() {
                result.error = "sp mode requires -from and -to".to_string();
            } else if opts.from == opts.to {
                result.path = vec![opts.from.clone()];
                result.hops = Some(0);
                result.total_weight = Some(0.0);
            } else if !opts.weighted {
                let path = if opts.undirected {
                    build_undirected(edges).path(&opts.from, &opts.to)
                } else {
                    build_directed(edges).path(&opts.from, &opts.to)
                };
                match path {
                    Some(p) => {
                        result.hops = Some(p.len().saturating_sub(1));
                        result.path = p;
                    }
                    None => result.error = format!("no path from {:?} to {:?}", opts.from, opts.to),
                }
            } else if opts.undirected {
                let mut g = build_weighted_undirected(edges);
                let path_edges = g.shortest_path(&opts.from, &opts.to);
                if path_edges.is_empty() {
                    result.error = format!("no path from {:?} to {:?}", opts.from, opts.to);
                } else {
                    let (path, total) = edges_to_node_path(&opts.from, &path_edges);
                    result.path = path;
                    result.total_weight = Some(total);
                    if g.has_negative_cycle() {
                        result.warning =
                            "negative cycle detected, shortest-path result may be unreliable"
                                .to_string();
                    }
                }
            } else {
                let mut g = build_weighted_directed(edges, false);
                let path_edges = g.shortest_path(&opts.from, &opts.to);
                if path_edges.is_empty() {
                    result.error = format!("no path from {:?} to {:?}", opts.from, opts.to);
                } else {
                    let (path, total) = edges_to_node_path(&opts.from, &path_edges);
                    result.path = path;
                    result.total_weight = Some(total);
                    if g.has_negative_cycle() {
                        result.warning =
                            "negative cycle detected, shortest-path result may be unreliable"
                                .to_string();
                    }
                }
            }
        }
        "mst" => {
            let g = build_weighted_undirected(edges);
            let mst = g.mst();
            result.mst_edges = mst_edges_out(&mst);
            result.total_weight = Some(mst.total_weight());
            if result.nodes > 0 && mst.edges().len() < result.nodes.saturating_sub(1) {
                result.warning =
                    "graph is disconnected, result is a minimum spanning forest".to_string();
            }
        }
        other => result.error = format!("unknown mode {:?}", other),
    }

    if !opts.viz.is_empty() {
        result.viz = build_viz(edges, &opts.viz, opts.undirected, opts.weighted);
    }

    result
}

fn edges_to_node_path(from: &str, path_edges: &[Edge<String>]) -> (Vec<String>, f64) {
    let mut path = Vec::with_capacity(path_edges.len().saturating_add(1));
    path.push(from.to_string());
    let mut total = 0.0;
    for e in path_edges {
        path.push(e.v2().clone());
        total += e.weight();
    }
    (path, total)
}

fn mst_edges_out(mst: &Mst<String>) -> Vec<EdgeOut> {
    let mut edges = mst.edges().to_vec();
    edges.sort_by(|a, b| {
        let (a1, a2) = normalize_pair(a.v1(), a.v2());
        let (b1, b2) = normalize_pair(b.v1(), b.v2());
        a1.cmp(b1).then_with(|| a2.cmp(b2)).then_with(|| {
            a.weight()
                .partial_cmp(&b.weight())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    edges
        .into_iter()
        .map(|e| EdgeOut {
            from: e.v1().clone(),
            to: e.v2().clone(),
            weight: Some(e.weight()),
        })
        .collect()
}

fn normalize_pair<'a>(a: &'a String, b: &'a String) -> (&'a String, &'a String) {
    if a <= b { (a, b) } else { (b, a) }
}

fn build_viz(edges: &[GraphEdge], viz: &str, undirected: bool, weighted: bool) -> String {
    match viz {
        "dot" => build_dot(edges, undirected, weighted),
        "mermaid" => build_mermaid(edges, undirected, weighted),
        _ => String::new(),
    }
}

fn escape_dot_label(s: &str) -> String {
    s.replace('"', "\\\"")
}

fn format_weight(weight: f64) -> String {
    let rounded = weight.round();
    if (weight - rounded).abs() < 1e-9 {
        return format!("{}", rounded as i64);
    }
    let s = format!("{weight}");
    if s.contains('e') || s.contains('E') {
        format!("{weight:.12}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    } else {
        s
    }
}

fn build_dot(edges: &[GraphEdge], undirected: bool, weighted: bool) -> String {
    let mut lines = Vec::with_capacity(edges.len().saturating_add(8));
    let (header, op) = if undirected {
        ("graph G {", "--")
    } else {
        ("digraph G {", "->")
    };
    lines.push(header.to_string());
    for node in sorted_nodes(edges) {
        lines.push(format!("  \"{}\";", escape_dot_label(&node)));
    }
    for e in edges {
        let from = escape_dot_label(&e.from);
        let to = escape_dot_label(&e.to);
        if weighted {
            lines.push(format!(
                "  \"{}\" {} \"{}\" [label=\"{}\"] ;",
                from,
                op,
                to,
                format_weight(e.weight)
            ));
        } else {
            lines.push(format!("  \"{}\" {} \"{}\";", from, op, to));
        }
    }
    lines.push("}".to_string());
    lines.join("\n")
}

fn mermaid_id(idx: usize) -> String {
    format!("N{idx}")
}

fn escape_mermaid_label(s: &str) -> String {
    s.replace('"', "\\\"")
}

fn build_mermaid(edges: &[GraphEdge], undirected: bool, weighted: bool) -> String {
    let nodes = sorted_nodes(edges);
    let mut id_map: Box<SkipMap<String, String>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    let mut lines = Vec::new();
    lines.push("graph TD".to_string());
    for (idx, node) in nodes.iter().enumerate() {
        let id = mermaid_id(idx);
        id_map.insert(node.clone(), id.clone());
        lines.push(format!("    {id}[\"{}\"]", escape_mermaid_label(node)));
    }
    let op = if undirected { "---" } else { "-->" };
    for e in edges {
        let from = id_map
            .get_ref(&e.from)
            .cloned()
            .unwrap_or_else(|| e.from.clone());
        let to = id_map
            .get_ref(&e.to)
            .cloned()
            .unwrap_or_else(|| e.to.clone());
        if weighted {
            lines.push(format!("    {from} {op}|{}| {to}", format_weight(e.weight)));
        } else {
            lines.push(format!("    {from} {op} {to}"));
        }
    }
    lines.join("\n")
}

fn render_results(results: &[AnalysisResult], format: &str) -> Result<String, String> {
    match format {
        "text" | "txt" | "" => Ok(render_text_results(results)),
        "json" => {
            if results.len() == 1 {
                serde_json::to_string_pretty(&results[0])
                    .map(|s| format!("{s}\n"))
                    .map_err(|e| e.to_string())
            } else {
                serde_json::to_string_pretty(results)
                    .map(|s| format!("{s}\n"))
                    .map_err(|e| e.to_string())
            }
        }
        "csv" => render_csv_results(results),
        other => Err(format!("unsupported format {:?}, use text|json|csv", other)),
    }
}

fn render_text_results(results: &[AnalysisResult]) -> String {
    if results.is_empty() {
        return String::new();
    }
    let mut chunks = Vec::with_capacity(results.len());
    for r in results {
        chunks.push(format_text_result(r).trim_end_matches('\n').to_string());
    }
    format!("{}\n", chunks.join("\n\n---\n\n"))
}

fn format_text_result(r: &AnalysisResult) -> String {
    let mut out = String::new();
    if !r.source.is_empty() {
        out.push_str(&format!("source: {}\n", r.source));
    }
    out.push_str(&format!("mode: {}\n", r.mode));
    out.push_str(&format!(
        "graph: nodes={} edges={} weighted={} undirected={}\n",
        r.nodes, r.edges, r.weighted, r.undirected
    ));
    if !r.error.is_empty() {
        out.push_str(&format!("error: {}\n", r.error));
        if !r.viz.is_empty() {
            out.push_str("viz:\n");
            out.push_str(&r.viz);
            out.push('\n');
        }
        return out;
    }
    if let Some(v) = r.has_cycle {
        out.push_str(&format!("has_cycle: {v}\n"));
    }
    if !r.cycle.is_empty() {
        out.push_str(&format!("cycle: {}\n", r.cycle.join(" -> ")));
    }
    if !r.components.is_empty() {
        out.push_str(&format!("components: {}\n", r.components.len()));
        for (idx, group) in r.components.iter().enumerate() {
            out.push_str(&format!(
                "{} ({}): {}\n",
                idx + 1,
                group.len(),
                group.join(", ")
            ));
        }
    }
    if !r.order.is_empty() {
        out.push_str(&format!("order: {}\n", r.order.join(" -> ")));
    }
    if !r.path.is_empty() {
        out.push_str(&format!("path: {}\n", r.path.join(" -> ")));
    }
    if let Some(hops) = r.hops {
        out.push_str(&format!("hops: {hops}\n"));
    }
    if let Some(total) = r.total_weight {
        out.push_str(&format!("total_weight: {:.6}\n", total));
    }
    if !r.mst_edges.is_empty() {
        out.push_str(&format!("mst_edges: {}\n", r.mst_edges.len()));
        for (idx, e) in r.mst_edges.iter().enumerate() {
            let w = e.weight.unwrap_or_default();
            out.push_str(&format!("{}. ({}-{}) {:.6}\n", idx + 1, e.from, e.to, w));
        }
    }
    if !r.warning.is_empty() {
        out.push_str(&format!("warning: {}\n", r.warning));
    }
    if !r.viz.is_empty() {
        out.push_str("viz:\n");
        out.push_str(&r.viz);
        out.push('\n');
    }
    out
}

fn render_csv_results(results: &[AnalysisResult]) -> Result<String, String> {
    let headers = [
        "source",
        "mode",
        "nodes",
        "edges",
        "weighted",
        "undirected",
        "has_cycle",
        "cycle",
        "components_count",
        "components",
        "order",
        "path",
        "hops",
        "total_weight",
        "mst_edges_count",
        "mst_edges",
        "warning",
        "error",
        "viz",
    ];

    let mut out = String::new();
    out.push_str(&headers.join(","));
    out.push('\n');

    for r in results {
        let has_cycle = r.has_cycle.map(|v| v.to_string()).unwrap_or_default();
        let hops = r.hops.map(|v| v.to_string()).unwrap_or_default();
        let total_weight = r.total_weight.map(|v| format!("{v}")).unwrap_or_default();

        let row = [
            r.source.clone(),
            r.mode.clone(),
            r.nodes.to_string(),
            r.edges.to_string(),
            r.weighted.to_string(),
            r.undirected.to_string(),
            has_cycle,
            csv_join(&r.cycle),
            r.components.len().to_string(),
            flatten_components(&r.components),
            csv_join(&r.order),
            csv_join(&r.path),
            hops,
            total_weight,
            r.mst_edges.len().to_string(),
            flatten_mst_edges(&r.mst_edges),
            r.warning.clone(),
            r.error.clone(),
            r.viz.clone(),
        ];

        let escaped: Vec<String> = row.into_iter().map(csv_escape).collect();
        out.push_str(&escaped.join(","));
        out.push('\n');
    }

    Ok(out)
}

fn csv_join(items: &[String]) -> String {
    items.join(";")
}

fn flatten_components(groups: &[Vec<String>]) -> String {
    if groups.is_empty() {
        return String::new();
    }
    groups
        .iter()
        .map(|g| g.join(","))
        .collect::<Vec<_>>()
        .join("|")
}

fn flatten_mst_edges(edges: &[EdgeOut]) -> String {
    if edges.is_empty() {
        return String::new();
    }
    edges
        .iter()
        .map(|e| {
            let w = e.weight.unwrap_or_default();
            format!("{}-{}:{}", e.from, e.to, format_weight(w))
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn csv_escape(s: String) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s
    }
}

fn normalize_ext_set(raw: &str) -> (SkipSet<String>, bool) {
    let raw = raw.trim();
    if raw.is_empty() || raw == "*" {
        return (SkipSet::new(16), true);
    }

    let mut set = SkipSet::new(16);
    for part in raw.split(',') {
        let mut p = part.trim().to_lowercase();
        if p.is_empty() {
            continue;
        }
        if p == "*" {
            return (SkipSet::new(16), true);
        }
        if !p.starts_with('.') {
            p = format!(".{p}");
        }
        set.insert(p);
    }

    if set.is_empty() {
        (SkipSet::new(16), true)
    } else {
        (set, false)
    }
}

fn collect_input_files(
    single_file: &str,
    dir_path: &str,
    ext_filter: &str,
    recursive: bool,
) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();

    let single_file = single_file.trim();
    if !single_file.is_empty() {
        let meta = fs::metadata(single_file).map_err(|e| e.to_string())?;
        if meta.is_dir() {
            return Err(format!(
                "-f expects a file, got directory {:?}",
                single_file
            ));
        }
        files.push(PathBuf::from(single_file));
    }

    let dir_path = dir_path.trim();
    if dir_path.is_empty() {
        files.sort();
        return Ok(files);
    }

    let (exts, allow_all) = normalize_ext_set(ext_filter);
    let mut add_if_match = |path: &Path| {
        if allow_all {
            files.push(path.to_path_buf());
            return;
        }
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        let ext = if ext.is_empty() {
            "".to_string()
        } else {
            format!(".{ext}")
        };
        if exts.contains(&ext) {
            files.push(path.to_path_buf());
        }
    };

    if !recursive {
        for entry in fs::read_dir(dir_path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_file() {
                add_if_match(&path);
            }
        }
        files.sort();
        return Ok(files);
    }

    fn walk_dir(path: &Path, f: &mut dyn FnMut(&Path)) -> Result<(), String> {
        for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let p = entry.path();
            if p.is_dir() {
                walk_dir(&p, f)?;
            } else if p.is_file() {
                f(&p);
            }
        }
        Ok(())
    }

    walk_dir(Path::new(dir_path), &mut add_if_match)?;
    files.sort();
    Ok(files)
}
