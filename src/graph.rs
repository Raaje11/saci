//! CSR (Compressed Sparse Row) computational graph representation (Section 26).
//!
//! Replaces pointer-heavy graph structures with contiguous `offsets[]` and
//! `edges[]` arrays for predictable cache-friendly traversal.

use crate::rasp::{RaspOp, RaspProgram};
use crate::sys::{
    ArrayVec, ErrorCode, ExprId, NodeId, QueueFullPolicy, RingBuffer, FLAG_DISCONNECTED,
};

pub const MAX_GRAPH_NODES: usize = 256;

/// CSR directed graph of RASP expressions.
/// Forward edges go `dependency -> dependent` (matching `rasp_to_graph.py`).
/// Reverse edges go `dependent -> dependency` (for post-order DFS from sink).
pub struct CsrGraph {
    pub num_nodes: usize,
    pub sink: NodeId,
    pub sources: ArrayVec<NodeId, 64>,
    /// Mapping `NodeId -> ExprId`
    pub node_to_expr: Vec<ExprId>,
    /// Direct lookup `ExprId -> Option<NodeId>` (indexed by `expr_id.0`)
    pub expr_to_node: Vec<Option<NodeId>>,
    /// Forward CSR: successors of `u` are `fwd_edges[fwd_offsets[u]..fwd_offsets[u+1]]`
    pub fwd_offsets: Vec<u32>,
    pub fwd_edges: Vec<NodeId>,
    /// Reverse CSR: predecessors of `u` are `rev_edges[rev_offsets[u]..rev_offsets[u+1]]`
    pub rev_offsets: Vec<u32>,
    pub rev_edges: Vec<NodeId>,
    /// Per-node bitflags
    pub flags: Vec<u16>,
}

impl CsrGraph {
    #[inline]
    pub fn successors(&self, u: NodeId) -> &[NodeId] {
        let i = u.0 as usize;
        let start = self.fwd_offsets[i] as usize;
        let end = self.fwd_offsets[i + 1] as usize;
        &self.fwd_edges[start..end]
    }

    #[inline]
    pub fn predecessors(&self, u: NodeId) -> &[NodeId] {
        let i = u.0 as usize;
        let start = self.rev_offsets[i] as usize;
        let end = self.rev_offsets[i + 1] as usize;
        &self.rev_edges[start..end]
    }

    #[inline]
    pub fn node_of(&self, expr: ExprId) -> Option<NodeId> {
        self.expr_to_node.get(expr.0 as usize).copied().flatten()
    }

    #[inline]
    pub fn expr_of(&self, node: NodeId) -> ExprId {
        self.node_to_expr[node.0 as usize]
    }

    /// Post-order DFS traversal on the reverse graph starting at `sink`.
    /// Visits all dependencies before their dependents.
    pub fn reverse_postorder(&self, out: &mut Vec<NodeId>) {
        out.clear();
        let mut visited = vec![0u8; self.num_nodes];
        // Explicit iterative DFS stack: (node, next_child_cursor)
        let mut stack: ArrayVec<(NodeId, usize), MAX_GRAPH_NODES> = ArrayVec::new();
        visited[self.sink.0 as usize] = 1;
        stack.push((self.sink, 0));

        while let Some((u, cursor)) = stack.pop() {
            let preds = self.predecessors(u);
            if cursor < preds.len() {
                stack.push((u, cursor + 1));
                let v = preds[cursor];
                if visited[v.0 as usize] == 0 {
                    visited[v.0 as usize] = 1;
                    stack.push((v, 0));
                }
            } else {
                out.push(u);
            }
        }
    }
}

#[inline]
fn expr_children(op: RaspOp) -> ArrayVec<ExprId, 2> {
    let mut children = ArrayVec::new();
    match op {
        RaspOp::Tokens | RaspOp::Indices | RaspOp::Full(_) | RaspOp::ConstantSelector { .. } => {}
        RaspOp::Map { inner, .. } | RaspOp::SelectorNot { inner } => {
            children.push(inner);
        }
        RaspOp::SequenceMap { fst, snd, .. }
        | RaspOp::LinearSequenceMap { fst, snd, .. }
        | RaspOp::SelectorAnd { fst, snd }
        | RaspOp::SelectorOr { fst, snd } => {
            children.push(fst);
            children.push(snd);
        }
        RaspOp::Select { keys, queries, .. } => {
            children.push(keys);
            children.push(queries);
        }
        RaspOp::Aggregate { selector, sop, .. } => {
            children.push(selector);
            children.push(sop);
        }
        RaspOp::SelectorWidth { selector } => {
            children.push(selector);
        }
    }
    children
}

pub fn extract_rasp_graph(prog: &RaspProgram, sink_expr: ExprId) -> Result<CsrGraph, ErrorCode> {
    if (sink_expr.0 as usize) >= prog.len() {
        return Err(ErrorCode::InvalidExpr);
    }

    let mut expr_to_node = vec![None; prog.len()];
    let mut node_to_expr = Vec::with_capacity(prog.len());
    let mut raw_fwd_edges: Vec<(NodeId, NodeId)> = Vec::with_capacity(prog.len() * 2);
    let mut sources: ArrayVec<NodeId, 64> = ArrayVec::new();

    let mut queue: RingBuffer<ExprId, MAX_GRAPH_NODES> = RingBuffer::new(QueueFullPolicy::Reject);

    let ensure_node = |expr: ExprId,
                       expr_to_node: &mut Vec<Option<NodeId>>,
                       node_to_expr: &mut Vec<ExprId>|
     -> NodeId {
        if let Some(nid) = expr_to_node[expr.0 as usize] {
            nid
        } else {
            let nid = NodeId(node_to_expr.len() as u32);
            expr_to_node[expr.0 as usize] = Some(nid);
            node_to_expr.push(expr);
            nid
        }
    };

    let sink = ensure_node(sink_expr, &mut expr_to_node, &mut node_to_expr);
    if queue.push(sink_expr) != ErrorCode::None {
        return Err(ErrorCode::CapacityExceeded);
    }

    while let Some(expr) = queue.pop() {
        let parent_id = ensure_node(expr, &mut expr_to_node, &mut node_to_expr);
        let children = expr_children(prog.node(expr).op);
        if children.is_empty() {
            // Avoid duplicate sources if visited via multiple paths
            if !sources.as_slice().contains(&parent_id) {
                let _ = sources.push(parent_id);
            }
        }
        for &child_expr in children.as_slice() {
            let child_existed = expr_to_node[child_expr.0 as usize].is_some();
            let child_id = ensure_node(child_expr, &mut expr_to_node, &mut node_to_expr);
            // Deduplicate edge (child_id -> parent_id)
            if !raw_fwd_edges.contains(&(child_id, parent_id)) {
                raw_fwd_edges.push((child_id, parent_id));
            }
            if !child_existed && queue.push(child_expr) != ErrorCode::None {
                return Err(ErrorCode::CapacityExceeded);
            }
        }
    }

    let num_nodes = node_to_expr.len();
    let num_edges = raw_fwd_edges.len();

    // Build forward CSR
    let mut fwd_counts = vec![0u32; num_nodes];
    let mut rev_counts = vec![0u32; num_nodes];
    for &(u, v) in &raw_fwd_edges {
        fwd_counts[u.0 as usize] += 1;
        rev_counts[v.0 as usize] += 1;
    }

    let mut fwd_offsets = vec![0u32; num_nodes + 1];
    let mut rev_offsets = vec![0u32; num_nodes + 1];
    for i in 0..num_nodes {
        fwd_offsets[i + 1] = fwd_offsets[i] + fwd_counts[i];
        rev_offsets[i + 1] = rev_offsets[i] + rev_counts[i];
    }

    let mut fwd_edges = vec![NodeId(0); num_edges];
    let mut rev_edges = vec![NodeId(0); num_edges];
    let mut fwd_cursor = fwd_offsets[..num_nodes].to_vec();
    let mut rev_cursor = rev_offsets[..num_nodes].to_vec();

    for &(u, v) in &raw_fwd_edges {
        let fi = fwd_cursor[u.0 as usize] as usize;
        fwd_edges[fi] = v;
        fwd_cursor[u.0 as usize] += 1;

        let ri = rev_cursor[v.0 as usize] as usize;
        rev_edges[ri] = u;
        rev_cursor[v.0 as usize] += 1;
    }

    let mut flags = Vec::with_capacity(num_nodes);
    for &eid in &node_to_expr {
        flags.push(prog.node(eid).flags & !FLAG_DISCONNECTED);
    }

    Ok(CsrGraph {
        num_nodes,
        sink,
        sources,
        node_to_expr,
        expr_to_node,
        fwd_offsets,
        fwd_edges,
        rev_offsets,
        rev_edges,
        flags,
    })
}
