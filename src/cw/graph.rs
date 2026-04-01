use std::cmp::Ordering;
use std::collections::VecDeque;
use std::hash::Hash;

use crate::commonw::types::{FastMap, FastSet};

const INF: f64 = f64::INFINITY;

#[derive(Clone, Debug, PartialEq)]
pub struct Edge<T>
where
    T: Clone,
{
    v1: T,
    v2: T,
    weight: f64,
    directed: bool,
}

impl<T> Edge<T>
where
    T: Clone,
{
    pub fn new(v1: T, v2: T, weight: f64, directed: bool) -> Self {
        Self {
            v1,
            v2,
            weight,
            directed,
        }
    }

    pub fn v1(&self) -> &T {
        &self.v1
    }

    pub fn v2(&self) -> &T {
        &self.v2
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub fn directed(&self) -> bool {
        self.directed
    }

    pub fn other(&self, u: &T) -> Option<T>
    where
        T: Eq,
    {
        if &self.v1 == u {
            return Some(self.v2.clone());
        }
        if &self.v2 == u {
            return Some(self.v1.clone());
        }
        None
    }
}

pub struct DirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    adj: FastMap<T, FastSet<T>>,
    edge_cnt: usize,
}

impl<T> DirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            adj: FastMap::default(),
            edge_cnt: 0,
        }
    }

    pub fn add_node(&mut self, u: T) -> bool {
        match self.adj.entry(u) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(FastSet::default());
                true
            }
        }
    }

    pub fn delete_node(&mut self, u: &T) -> bool {
        let Some(outgoing) = self.adj.remove(u) else {
            return false;
        };

        self.edge_cnt = self.edge_cnt.saturating_sub(outgoing.len());

        let mut incoming_removed = 0usize;
        for neighbors in self.adj.values_mut() {
            if neighbors.remove(u) {
                incoming_removed += 1;
            }
        }
        self.edge_cnt = self.edge_cnt.saturating_sub(incoming_removed);
        true
    }

    pub fn add_edge(&mut self, u: T, v: T) -> bool {
        self.add_node(u.clone());
        self.add_node(v.clone());
        let inserted = self.adj.get_mut(&u).is_some_and(|set| set.insert(v));
        if inserted {
            self.edge_cnt += 1;
        }
        inserted
    }

    pub fn delete_edge(&mut self, u: &T, v: &T) -> bool {
        if let Some(neighbors) = self.adj.get_mut(u)
            && neighbors.remove(v)
        {
            self.edge_cnt = self.edge_cnt.saturating_sub(1);
            return true;
        }
        false
    }

    pub fn adj(&self, u: &T) -> Vec<T> {
        self.adj
            .get(u)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn nodes(&self) -> Vec<T> {
        self.adj.keys().cloned().collect()
    }

    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edge_cnt
    }

    pub fn degree(&self, u: &T, incoming: bool) -> usize {
        if !incoming {
            return self.adj.get(u).map(|set| set.len()).unwrap_or(0);
        }
        self.adj.values().filter(|set| set.contains(u)).count()
    }

    pub fn reachable(&self, from: &T, to: &T) -> bool {
        if !self.adj.contains_key(from) || !self.adj.contains_key(to) {
            return false;
        }
        if from == to {
            return true;
        }
        let (visited, _) = self.bfs_prev(from);
        visited.contains(to)
    }

    pub fn path(&self, from: &T, to: &T) -> Option<Vec<T>> {
        if !self.adj.contains_key(from) || !self.adj.contains_key(to) {
            return None;
        }
        if from == to {
            return Some(vec![from.clone()]);
        }

        let (visited, prev) = self.bfs_prev(from);
        if !visited.contains(to) {
            return None;
        }
        Self::reconstruct_node_path(from, to, &prev, self.num_nodes())
    }

    pub fn has_cycle(&self) -> bool {
        self.detect_cycle().is_some()
    }

    pub fn cycle(&self) -> Option<Vec<T>> {
        self.detect_cycle()
    }

    pub fn sorted(&self) -> Option<Vec<T>> {
        let nodes = self.nodes();
        let mut indegree: FastMap<T, usize> = FastMap::default();
        for node in &nodes {
            indegree.insert(node.clone(), 0);
        }

        for neighbors in self.adj.values() {
            for next in neighbors {
                *indegree.entry(next.clone()).or_insert(0) += 1;
            }
        }

        let mut q = VecDeque::new();
        for node in &nodes {
            if indegree.get(node).copied().unwrap_or(0) == 0 {
                q.push_back(node.clone());
            }
        }

        let mut order = Vec::with_capacity(nodes.len());
        while let Some(curr) = q.pop_front() {
            order.push(curr.clone());
            for next in self.adj(&curr) {
                if let Some(x) = indegree.get_mut(&next) {
                    *x = x.saturating_sub(1);
                    if *x == 0 {
                        q.push_back(next);
                    }
                }
            }
        }

        if order.len() == self.num_nodes() {
            Some(order)
        } else {
            None
        }
    }

    pub fn reverse(&self) -> DirectedGraph<T> {
        let mut g = DirectedGraph::new();
        for node in self.nodes() {
            g.add_node(node);
        }
        for (u, neighbors) in &self.adj {
            for v in neighbors {
                g.add_edge(v.clone(), u.clone());
            }
        }
        g
    }

    pub fn strongly_connected(&self, u: &T, v: &T) -> bool {
        let id_map = self.component_id_map();
        match (id_map.get(u), id_map.get(v)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    pub fn strong_components(&self) -> Vec<Vec<T>> {
        let reverse_graph = self.reverse();

        let mut visited: FastSet<T> = FastSet::default();
        let mut order: Vec<T> = Vec::with_capacity(self.num_nodes());
        for node in reverse_graph.nodes() {
            if !visited.contains(&node) {
                Self::dfs_finish_order(&reverse_graph, &node, &mut visited, &mut order);
            }
        }

        visited.clear();
        let mut components = Vec::new();
        while let Some(node) = order.pop() {
            if visited.contains(&node) {
                continue;
            }
            let mut component = Vec::new();
            Self::dfs_collect_component(self, &node, &mut visited, &mut component);
            components.push(component);
        }

        components
    }

    pub fn num_strong_components(&self) -> usize {
        self.strong_components().len()
    }

    fn bfs_prev(&self, from: &T) -> (FastSet<T>, FastMap<T, T>) {
        let mut visited: FastSet<T> = FastSet::default();
        let mut prev: FastMap<T, T> = FastMap::default();

        if !self.adj.contains_key(from) {
            return (visited, prev);
        }

        let mut q = VecDeque::new();
        visited.insert(from.clone());
        q.push_back(from.clone());

        while let Some(curr) = q.pop_front() {
            for next in self.adj(&curr) {
                if visited.insert(next.clone()) {
                    prev.insert(next.clone(), curr.clone());
                    q.push_back(next);
                }
            }
        }

        (visited, prev)
    }

    fn reconstruct_node_path(
        from: &T,
        to: &T,
        prev: &FastMap<T, T>,
        node_limit: usize,
    ) -> Option<Vec<T>> {
        let mut path = vec![to.clone()];
        let mut curr = to.clone();

        let max_steps = node_limit.saturating_add(1);
        for _ in 0..max_steps {
            if &curr == from {
                path.reverse();
                return Some(path);
            }
            let parent = prev.get(&curr)?.clone();
            path.push(parent.clone());
            curr = parent;
        }

        None
    }

    fn detect_cycle(&self) -> Option<Vec<T>> {
        fn dfs<T>(
            g: &DirectedGraph<T>,
            node: &T,
            state: &mut FastMap<T, u8>,
            stack: &mut Vec<T>,
            stack_pos: &mut FastMap<T, usize>,
        ) -> Option<Vec<T>>
        where
            T: Eq + Hash + Clone,
        {
            state.insert(node.clone(), 1);
            stack_pos.insert(node.clone(), stack.len());
            stack.push(node.clone());

            for next in g.adj(node) {
                let st = state.get(&next).copied().unwrap_or(0);
                if st == 0 {
                    if let Some(cycle) = dfs(g, &next, state, stack, stack_pos) {
                        return Some(cycle);
                    }
                } else if st == 1 {
                    let start = *stack_pos.get(&next).unwrap_or(&0);
                    let mut cycle = stack[start..].to_vec();
                    cycle.push(next.clone());
                    return Some(cycle);
                }
            }

            stack.pop();
            stack_pos.remove(node);
            state.insert(node.clone(), 2);
            None
        }

        let mut state: FastMap<T, u8> = FastMap::default();
        let mut stack = Vec::new();
        let mut stack_pos: FastMap<T, usize> = FastMap::default();

        for node in self.nodes() {
            if state.get(&node).copied().unwrap_or(0) == 0
                && let Some(cycle) = dfs(self, &node, &mut state, &mut stack, &mut stack_pos)
            {
                return Some(cycle);
            }
        }

        None
    }

    fn dfs_finish_order(
        g: &DirectedGraph<T>,
        node: &T,
        visited: &mut FastSet<T>,
        order: &mut Vec<T>,
    ) {
        visited.insert(node.clone());
        for next in g.adj(node) {
            if !visited.contains(&next) {
                Self::dfs_finish_order(g, &next, visited, order);
            }
        }
        order.push(node.clone());
    }

    fn dfs_collect_component(
        g: &DirectedGraph<T>,
        node: &T,
        visited: &mut FastSet<T>,
        component: &mut Vec<T>,
    ) {
        visited.insert(node.clone());
        component.push(node.clone());
        for next in g.adj(node) {
            if !visited.contains(&next) {
                Self::dfs_collect_component(g, &next, visited, component);
            }
        }
    }

    fn component_id_map(&self) -> FastMap<T, usize> {
        let mut id_map = FastMap::default();
        for (idx, component) in self.strong_components().into_iter().enumerate() {
            for node in component {
                id_map.insert(node, idx);
            }
        }
        id_map
    }
}

impl<T> Default for DirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct UndirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    adj: FastMap<T, FastSet<T>>,
    edge_cnt: usize,
}

impl<T> UndirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            adj: FastMap::default(),
            edge_cnt: 0,
        }
    }

    pub fn add_node(&mut self, v: T) -> bool {
        match self.adj.entry(v) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(FastSet::default());
                true
            }
        }
    }

    pub fn add_edge(&mut self, u: T, v: T) -> bool {
        self.add_node(u.clone());
        self.add_node(v.clone());

        if self.adj.get(&u).is_some_and(|set| set.contains(&v)) {
            return false;
        }

        self.adj
            .get_mut(&u)
            .expect("node must exist")
            .insert(v.clone());
        self.adj
            .get_mut(&v)
            .expect("node must exist")
            .insert(u.clone());
        self.edge_cnt += 1;
        true
    }

    pub fn delete_edge(&mut self, u: &T, v: &T) -> bool {
        if !self.adj.contains_key(u) || !self.adj.contains_key(v) {
            return false;
        }

        let removed1 = self.adj.get_mut(u).is_some_and(|set| set.remove(v));
        if !removed1 {
            return false;
        }

        if u != v {
            let _ = self.adj.get_mut(v).is_some_and(|set| set.remove(u));
        }

        self.edge_cnt = self.edge_cnt.saturating_sub(1);
        true
    }

    pub fn delete_node(&mut self, u: &T) -> bool {
        let Some(neighbors) = self.adj.remove(u) else {
            return false;
        };

        for v in neighbors {
            if let Some(set) = self.adj.get_mut(&v)
                && set.remove(u)
            {
                self.edge_cnt = self.edge_cnt.saturating_sub(1);
            }
        }
        true
    }

    pub fn adj(&self, u: &T) -> Vec<T> {
        self.adj
            .get(u)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn nodes(&self) -> Vec<T> {
        self.adj.keys().cloned().collect()
    }

    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edge_cnt
    }

    pub fn degree(&self, u: &T) -> usize {
        self.adj.get(u).map(|set| set.len()).unwrap_or(0)
    }

    pub fn connected(&self, u: &T, v: &T) -> bool {
        if !self.adj.contains_key(u) || !self.adj.contains_key(v) {
            return false;
        }
        if u == v {
            return true;
        }

        let mut visited: FastSet<T> = FastSet::default();
        let mut q = VecDeque::new();
        visited.insert(u.clone());
        q.push_back(u.clone());

        while let Some(curr) = q.pop_front() {
            for next in self.adj(&curr) {
                if next == *v {
                    return true;
                }
                if visited.insert(next.clone()) {
                    q.push_back(next);
                }
            }
        }

        false
    }

    pub fn groups(&self) -> Vec<Vec<T>> {
        let mut groups = Vec::new();
        let mut visited: FastSet<T> = FastSet::default();

        for node in self.nodes() {
            if visited.contains(&node) {
                continue;
            }

            let mut component = Vec::new();
            let mut q = VecDeque::new();
            visited.insert(node.clone());
            q.push_back(node.clone());

            while let Some(curr) = q.pop_front() {
                component.push(curr.clone());
                for next in self.adj(&curr) {
                    if visited.insert(next.clone()) {
                        q.push_back(next);
                    }
                }
            }

            groups.push(component);
        }

        groups
    }

    pub fn num_groups(&self) -> usize {
        self.groups().len()
    }

    pub fn group(&self, u: &T) -> Option<usize> {
        for (idx, component) in self.groups().iter().enumerate() {
            if component.contains(u) {
                return Some(idx);
            }
        }
        None
    }

    pub fn path(&self, from: &T, to: &T) -> Option<Vec<T>> {
        if !self.adj.contains_key(from) || !self.adj.contains_key(to) {
            return None;
        }
        if from == to {
            return Some(vec![from.clone()]);
        }

        let mut visited: FastSet<T> = FastSet::default();
        let mut prev: FastMap<T, T> = FastMap::default();
        let mut q = VecDeque::new();

        visited.insert(from.clone());
        q.push_back(from.clone());

        while let Some(curr) = q.pop_front() {
            for next in self.adj(&curr) {
                if visited.insert(next.clone()) {
                    prev.insert(next.clone(), curr.clone());
                    q.push_back(next);
                }
            }
        }

        if !visited.contains(to) {
            return None;
        }

        DirectedGraph::<T>::reconstruct_node_path(from, to, &prev, self.num_nodes())
    }

    pub fn has_cycle(&self) -> bool {
        fn dfs<T>(
            g: &UndirectedGraph<T>,
            node: &T,
            parent: Option<&T>,
            visited: &mut FastSet<T>,
        ) -> bool
        where
            T: Eq + Hash + Clone,
        {
            visited.insert(node.clone());
            for next in g.adj(node) {
                if !visited.contains(&next) {
                    if dfs(g, &next, Some(node), visited) {
                        return true;
                    }
                } else if parent.is_none_or(|p| p != &next) {
                    return true;
                }
            }
            false
        }

        let mut visited: FastSet<T> = FastSet::default();
        for node in self.nodes() {
            if !visited.contains(&node) && dfs(self, &node, None, &mut visited) {
                return true;
            }
        }
        false
    }
}

impl<T> Default for UndirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct WeightedDirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    graph: DirectedGraph<T>,
    weights: FastMap<T, FastMap<T, f64>>,
    has_negative_cycle: bool,
}

impl<T> WeightedDirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            graph: DirectedGraph::new(),
            weights: FastMap::default(),
            has_negative_cycle: false,
        }
    }

    pub fn add_edge(&mut self, u: T, v: T, weight: f64) -> bool {
        if !self.graph.add_edge(u.clone(), v.clone()) {
            return false;
        }
        self.weights.entry(u).or_default().insert(v, weight);
        true
    }

    pub fn delete_edge(&mut self, u: &T, v: &T) -> bool {
        if !self.graph.delete_edge(u, v) {
            return false;
        }
        if let Some(neighbors) = self.weights.get_mut(u) {
            neighbors.remove(v);
            if neighbors.is_empty() {
                self.weights.remove(u);
            }
        }
        true
    }

    pub fn edges(&self) -> Vec<Edge<T>> {
        let mut edges = Vec::new();
        for (u, neighbors) in &self.weights {
            for (v, weight) in neighbors {
                edges.push(Edge::new(u.clone(), v.clone(), *weight, true));
            }
        }
        edges
    }

    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    pub fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    pub fn has_negative_cycle(&self) -> bool {
        self.has_negative_cycle
    }

    pub fn shortest_path(&mut self, from: &T, to: &T) -> Vec<Edge<T>> {
        if !self.graph.reachable(from, to) {
            return Vec::new();
        }

        self.has_negative_cycle = false;
        let has_negative_edge = self.edges().iter().any(|e| e.weight() < 0.0);

        let (prev, _) = if has_negative_edge {
            self.bellman_ford(from)
        } else if self.graph.has_cycle() {
            self.dijkstra(from, to)
        } else {
            self.acyclic_shortest(from, to)
        };

        self.reconstruct_path(from, to, &prev)
    }

    fn weight(&self, u: &T, v: &T) -> Option<f64> {
        self.weights.get(u).and_then(|m| m.get(v)).copied()
    }

    fn reconstruct_path(&self, from: &T, to: &T, prev: &FastMap<T, T>) -> Vec<Edge<T>> {
        if from == to {
            return Vec::new();
        }

        let mut res = Vec::new();
        let mut curr = to.clone();
        let max_steps = self.graph.num_nodes().saturating_add(1);

        for _ in 0..max_steps {
            if &curr == from {
                break;
            }
            let Some(parent) = prev.get(&curr).cloned() else {
                return Vec::new();
            };
            let Some(weight) = self.weight(&parent, &curr) else {
                return Vec::new();
            };
            res.push(Edge::new(parent.clone(), curr.clone(), weight, true));
            curr = parent;
        }

        if &curr != from {
            return Vec::new();
        }

        res.reverse();
        res
    }

    fn dijkstra(&self, from: &T, to: &T) -> (FastMap<T, T>, FastMap<T, f64>) {
        let nodes = self.graph.nodes();
        let mut dist: FastMap<T, f64> = FastMap::default();
        let mut prev: FastMap<T, T> = FastMap::default();
        let mut visited: FastSet<T> = FastSet::default();

        dist.insert(from.clone(), 0.0);

        loop {
            let mut curr_node: Option<T> = None;
            let mut curr_dist = INF;

            for node in &nodes {
                if visited.contains(node) {
                    continue;
                }
                let d = dist.get(node).copied().unwrap_or(INF);
                if d < curr_dist {
                    curr_dist = d;
                    curr_node = Some(node.clone());
                }
            }

            let Some(curr) = curr_node else {
                break;
            };
            if !curr_dist.is_finite() {
                break;
            }
            if &curr == to {
                break;
            }

            visited.insert(curr.clone());

            for next in self.graph.adj(&curr) {
                let Some(weight) = self.weight(&curr, &next) else {
                    continue;
                };
                let candidate = curr_dist + weight;
                if candidate < dist.get(&next).copied().unwrap_or(INF) {
                    dist.insert(next.clone(), candidate);
                    prev.insert(next.clone(), curr.clone());
                }
            }
        }

        (prev, dist)
    }

    fn acyclic_shortest(&self, from: &T, to: &T) -> (FastMap<T, T>, FastMap<T, f64>) {
        let Some(order) = self.graph.sorted() else {
            return self.dijkstra(from, to);
        };

        let mut dist: FastMap<T, f64> = FastMap::default();
        let mut prev: FastMap<T, T> = FastMap::default();
        dist.insert(from.clone(), 0.0);

        for node in order {
            let base = dist.get(&node).copied().unwrap_or(INF);
            if !base.is_finite() {
                continue;
            }
            for next in self.graph.adj(&node) {
                let Some(weight) = self.weight(&node, &next) else {
                    continue;
                };
                let candidate = base + weight;
                if candidate < dist.get(&next).copied().unwrap_or(INF) {
                    dist.insert(next.clone(), candidate);
                    prev.insert(next.clone(), node.clone());
                }
            }
        }

        (prev, dist)
    }

    fn bellman_ford(&mut self, from: &T) -> (FastMap<T, T>, FastMap<T, f64>) {
        let mut dist: FastMap<T, f64> = FastMap::default();
        let mut prev: FastMap<T, T> = FastMap::default();
        let edges = self.edges();

        dist.insert(from.clone(), 0.0);

        for _ in 0..self.graph.num_nodes().saturating_sub(1) {
            let mut changed = false;
            for edge in &edges {
                let u = edge.v1();
                let v = edge.v2();
                let du = dist.get(u).copied().unwrap_or(INF);
                if !du.is_finite() {
                    continue;
                }
                let cand = du + edge.weight();
                if cand < dist.get(v).copied().unwrap_or(INF) {
                    dist.insert(v.clone(), cand);
                    prev.insert(v.clone(), u.clone());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        self.has_negative_cycle = false;
        for edge in &edges {
            let u = edge.v1();
            let v = edge.v2();
            let du = dist.get(u).copied().unwrap_or(INF);
            if du.is_finite() && du + edge.weight() < dist.get(v).copied().unwrap_or(INF) {
                self.has_negative_cycle = true;
                break;
            }
        }

        (prev, dist)
    }
}

impl<T> Default for WeightedDirectedGraph<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct Mst<T>
where
    T: Clone,
{
    edges: Vec<Edge<T>>,
}

impl<T> Mst<T>
where
    T: Clone,
{
    pub fn edges(&self) -> &[Edge<T>] {
        &self.edges
    }

    pub fn total_weight(&self) -> f64 {
        self.edges.iter().map(|e| e.weight()).sum()
    }
}

pub struct WeightedUndirectedGraph<T>
where
    T: Eq + Hash + Clone + Ord,
{
    graph: UndirectedGraph<T>,
    weights: FastMap<T, FastMap<T, f64>>,
    has_negative_cycle: bool,
}

impl<T> WeightedUndirectedGraph<T>
where
    T: Eq + Hash + Clone + Ord,
{
    pub fn new() -> Self {
        Self {
            graph: UndirectedGraph::new(),
            weights: FastMap::default(),
            has_negative_cycle: false,
        }
    }

    pub fn add_edge(&mut self, u: T, v: T, weight: f64) -> bool {
        let _inserted = self.graph.add_edge(u.clone(), v.clone());
        self.weights
            .entry(u.clone())
            .or_default()
            .insert(v.clone(), weight);
        self.weights.entry(v).or_default().insert(u, weight);
        true
    }

    pub fn delete_edge(&mut self, u: &T, v: &T) -> bool {
        if !self.graph.delete_edge(u, v) {
            return false;
        }
        if let Some(neighbors) = self.weights.get_mut(u) {
            neighbors.remove(v);
            if neighbors.is_empty() {
                self.weights.remove(u);
            }
        }
        if let Some(neighbors) = self.weights.get_mut(v) {
            neighbors.remove(u);
            if neighbors.is_empty() {
                self.weights.remove(v);
            }
        }
        true
    }

    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    pub fn num_edges(&self) -> usize {
        self.graph.num_edges()
    }

    pub fn has_negative_cycle(&self) -> bool {
        self.has_negative_cycle
    }

    pub fn edges(&self) -> Vec<Edge<T>> {
        let mut result = Vec::new();
        for (u, neighbors) in &self.weights {
            for (v, weight) in neighbors {
                if u <= v {
                    result.push(Edge::new(u.clone(), v.clone(), *weight, false));
                }
            }
        }
        result
    }

    pub fn mst(&self) -> Mst<T> {
        let mut edges = self.edges();
        edges.sort_by(|a, b| {
            a.weight()
                .partial_cmp(&b.weight())
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.v1().cmp(b.v1()))
                .then_with(|| a.v2().cmp(b.v2()))
        });

        let mut uf = SimpleUf::new(self.graph.nodes());
        let mut picked = Vec::new();

        for edge in edges {
            if uf.union(edge.v1().clone(), edge.v2().clone()) {
                picked.push(edge);
                if self.num_nodes() > 0 && picked.len() >= self.num_nodes() - 1 {
                    break;
                }
            }
        }

        Mst { edges: picked }
    }

    pub fn shortest_path(&mut self, from: &T, to: &T) -> Vec<Edge<T>> {
        if !self.graph.connected(from, to) {
            return Vec::new();
        }

        self.has_negative_cycle = false;
        let has_negative_edge = self.edges().iter().any(|e| e.weight() < 0.0);

        let (prev, _) = if has_negative_edge {
            self.bellman_ford(from)
        } else {
            self.dijkstra(from, to)
        };

        self.reconstruct_path(from, to, &prev)
    }

    fn weight(&self, u: &T, v: &T) -> Option<f64> {
        self.weights.get(u).and_then(|m| m.get(v)).copied()
    }

    fn reconstruct_path(&self, from: &T, to: &T, prev: &FastMap<T, T>) -> Vec<Edge<T>> {
        if from == to {
            return Vec::new();
        }

        let mut res = Vec::new();
        let mut curr = to.clone();
        let max_steps = self.graph.num_nodes().saturating_add(1);

        for _ in 0..max_steps {
            if &curr == from {
                break;
            }
            let Some(parent) = prev.get(&curr).cloned() else {
                return Vec::new();
            };
            let Some(weight) = self.weight(&parent, &curr) else {
                return Vec::new();
            };
            res.push(Edge::new(parent.clone(), curr.clone(), weight, false));
            curr = parent;
        }

        if &curr != from {
            return Vec::new();
        }

        res.reverse();
        res
    }

    fn dijkstra(&self, from: &T, to: &T) -> (FastMap<T, T>, FastMap<T, f64>) {
        let nodes = self.graph.nodes();
        let mut dist: FastMap<T, f64> = FastMap::default();
        let mut prev: FastMap<T, T> = FastMap::default();
        let mut visited: FastSet<T> = FastSet::default();

        dist.insert(from.clone(), 0.0);

        loop {
            let mut curr_node: Option<T> = None;
            let mut curr_dist = INF;

            for node in &nodes {
                if visited.contains(node) {
                    continue;
                }
                let d = dist.get(node).copied().unwrap_or(INF);
                if d < curr_dist {
                    curr_dist = d;
                    curr_node = Some(node.clone());
                }
            }

            let Some(curr) = curr_node else {
                break;
            };
            if !curr_dist.is_finite() {
                break;
            }
            if &curr == to {
                break;
            }

            visited.insert(curr.clone());

            for next in self.graph.adj(&curr) {
                let Some(weight) = self.weight(&curr, &next) else {
                    continue;
                };
                let candidate = curr_dist + weight;
                if candidate < dist.get(&next).copied().unwrap_or(INF) {
                    dist.insert(next.clone(), candidate);
                    prev.insert(next.clone(), curr.clone());
                }
            }
        }

        (prev, dist)
    }

    fn bellman_ford(&mut self, from: &T) -> (FastMap<T, T>, FastMap<T, f64>) {
        let mut dist: FastMap<T, f64> = FastMap::default();
        let mut prev: FastMap<T, T> = FastMap::default();
        let edges = self.edges();

        dist.insert(from.clone(), 0.0);

        for _ in 0..self.graph.num_nodes().saturating_sub(1) {
            let mut changed = false;
            for edge in &edges {
                let u = edge.v1();
                let v = edge.v2();
                let w = edge.weight();

                let du = dist.get(u).copied().unwrap_or(INF);
                if du.is_finite() {
                    let cand = du + w;
                    if cand < dist.get(v).copied().unwrap_or(INF) {
                        dist.insert(v.clone(), cand);
                        prev.insert(v.clone(), u.clone());
                        changed = true;
                    }
                }

                let dv = dist.get(v).copied().unwrap_or(INF);
                if dv.is_finite() {
                    let cand = dv + w;
                    if cand < dist.get(u).copied().unwrap_or(INF) {
                        dist.insert(u.clone(), cand);
                        prev.insert(u.clone(), v.clone());
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        self.has_negative_cycle = false;
        for edge in &edges {
            let u = edge.v1();
            let v = edge.v2();
            let w = edge.weight();

            let du = dist.get(u).copied().unwrap_or(INF);
            if du.is_finite() && du + w < dist.get(v).copied().unwrap_or(INF) {
                self.has_negative_cycle = true;
                break;
            }

            let dv = dist.get(v).copied().unwrap_or(INF);
            if dv.is_finite() && dv + w < dist.get(u).copied().unwrap_or(INF) {
                self.has_negative_cycle = true;
                break;
            }
        }

        (prev, dist)
    }
}

impl<T> Default for WeightedUndirectedGraph<T>
where
    T: Eq + Hash + Clone + Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

struct SimpleUf<T>
where
    T: Eq + Hash + Clone,
{
    parent: FastMap<T, T>,
    size: FastMap<T, usize>,
}

impl<T> SimpleUf<T>
where
    T: Eq + Hash + Clone,
{
    fn new<I>(nodes: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut uf = Self {
            parent: FastMap::default(),
            size: FastMap::default(),
        };
        for node in nodes {
            uf.parent.insert(node.clone(), node.clone());
            uf.size.insert(node, 1);
        }
        uf
    }

    fn ensure(&mut self, node: &T) {
        if !self.parent.contains_key(node) {
            self.parent.insert(node.clone(), node.clone());
            self.size.insert(node.clone(), 1);
        }
    }

    fn find(&mut self, node: T) -> T {
        self.ensure(&node);

        let mut root = node.clone();
        loop {
            let parent = self
                .parent
                .get(&root)
                .cloned()
                .unwrap_or_else(|| root.clone());
            if parent == root {
                break;
            }
            root = parent;
        }

        let mut curr = node;
        loop {
            let parent = self
                .parent
                .get(&curr)
                .cloned()
                .unwrap_or_else(|| curr.clone());
            if parent == root {
                self.parent.insert(curr, root.clone());
                break;
            }
            self.parent.insert(curr.clone(), root.clone());
            curr = parent;
        }

        root
    }

    fn union(&mut self, a: T, b: T) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }

        let sa = self.size.get(&ra).copied().unwrap_or(1);
        let sb = self.size.get(&rb).copied().unwrap_or(1);

        if sa < sb {
            self.parent.insert(ra.clone(), rb.clone());
            self.size.insert(rb, sa + sb);
        } else {
            self.parent.insert(rb.clone(), ra.clone());
            self.size.insert(ra, sa + sb);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::{DirectedGraph, UndirectedGraph, WeightedDirectedGraph, WeightedUndirectedGraph};

    fn round6(v: f64) -> f64 {
        (v * 1_000_000.0).round() / 1_000_000.0
    }

    fn sum_weight<T: Clone>(edges: &[super::Edge<T>]) -> f64 {
        edges.iter().map(|e| e.weight()).sum()
    }

    #[test]
    fn test_directed_graph_cycle_and_scc() {
        let mut g = DirectedGraph::new();
        g.add_edge("A", "B");
        g.add_edge("B", "C");
        g.add_edge("C", "A");
        g.add_edge("C", "D");

        assert!(g.has_cycle());
        let cycle = g.cycle().expect("cycle should exist");
        assert!(cycle.len() >= 4);
        assert_eq!(cycle.first(), cycle.last());

        assert!(g.strongly_connected(&"A", &"B"));
        assert!(!g.strongly_connected(&"A", &"D"));

        let mut groups = g.strong_components();
        for group in &mut groups {
            group.sort();
        }
        groups.sort_by(|a, b| a[0].cmp(b[0]));

        assert_eq!(groups, vec![vec!["A", "B", "C"], vec!["D"]]);
        assert_eq!(g.num_strong_components(), 2);
    }

    #[test]
    fn test_directed_graph_topo_and_path() {
        let mut g = DirectedGraph::new();
        g.add_edge("A", "B");
        g.add_edge("B", "C");
        g.add_edge("C", "D");

        assert!(!g.has_cycle());
        assert_eq!(g.sorted(), Some(vec!["A", "B", "C", "D"]));
        assert_eq!(g.path(&"A", &"D"), Some(vec!["A", "B", "C", "D"]));
    }

    #[test]
    fn test_undirected_graph_groups_path_cycle() {
        let mut g = UndirectedGraph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(4, 5);

        let mut sizes: Vec<usize> = g.groups().iter().map(|c| c.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![2, 3]);

        assert_eq!(g.path(&1, &3), Some(vec![1, 2, 3]));
        assert!(!g.has_cycle());

        g.add_edge(1, 3);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_weighted_directed_shortest_path() {
        let mut g = WeightedDirectedGraph::new();
        g.add_edge("A", "B", 1.0);
        g.add_edge("B", "C", 2.0);
        g.add_edge("A", "C", 10.0);
        g.add_edge("C", "D", 1.0);

        let path = g.shortest_path(&"A", &"D");
        assert_eq!(path.len(), 3);
        assert_eq!(round6(sum_weight(&path)), 4.0);
        assert!(!g.has_negative_cycle());
    }

    #[test]
    fn test_weighted_directed_negative_edge_shortest_path() {
        let mut g = WeightedDirectedGraph::new();
        g.add_edge("A", "B", 4.0);
        g.add_edge("A", "C", 5.0);
        g.add_edge("B", "C", -10.0);

        let path = g.shortest_path(&"A", &"C");
        assert_eq!(path.len(), 2);
        assert_eq!(round6(sum_weight(&path)), -6.0);
        assert!(!g.has_negative_cycle());
    }

    #[test]
    fn test_weighted_undirected_shortest_and_mst() {
        let mut g = WeightedUndirectedGraph::new();
        g.add_edge("A", "B", 1.0);
        g.add_edge("B", "C", 2.0);
        g.add_edge("C", "D", 1.0);
        g.add_edge("A", "D", 10.0);
        g.add_edge("B", "D", 5.0);

        let path = g.shortest_path(&"A", &"D");
        assert_eq!(path.len(), 3);
        assert_eq!(round6(sum_weight(&path)), 4.0);

        let mst = g.mst();
        assert_eq!(mst.edges().len(), 3);
        assert_eq!(round6(mst.total_weight()), 4.0);
    }
}
