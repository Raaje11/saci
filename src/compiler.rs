//! Tracr Compiler Pipeline:
//!   static_validate -> extract_rasp_graph -> infer_bases ->
//!   expr_to_craft_graph -> craft_graph_to_model -> assemble -> CompiledTransformer.
//!
//! Emits structured enum-based `TelemetryEvent`s; performs zero string formatting.

use crate::craft::{
    self, AttentionHead, BasisDir, BasisRegistry, BasisVal, CraftBlock, DenseMat, DirNamespace,
    LayerBlock, LinearMap, Mlp, MultiAttentionHead, SeriesWithResiduals, VectorSpace,
};
use crate::graph::{extract_rasp_graph, CsrGraph};
use crate::rasp::{Encoding, RaspOp, RaspProgram, Value};
use crate::sys::{
    DirId, ErrorCode, EventKind, ExprId, Module, NodeId, Status, SymId, TelemetryBus,
    ValidationResult, FLAG_CATEGORICAL, FLAG_FAILED, FLAG_VALIDATED,
};
use crate::transformer::{CompiledLayer, CompiledTransformer};

pub const DEFAULT_BOS_SYM: SymId = SymId(0xFFFF_FFFE);
pub const DEFAULT_PAD_SYM: SymId = SymId(0xFFFF_FFFD);

#[derive(Clone, Copy, Debug)]
pub struct CompileConfig {
    pub max_seq_len: usize,
    pub causal: bool,
    pub bos_value: Value,
    pub pad_value: Value,
    pub mlp_exactness: f64,
    pub softmax_coldness: f64,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 16,
            causal: false,
            bos_value: Value::Sym(DEFAULT_BOS_SYM),
            pad_value: Value::Sym(DEFAULT_PAD_SYM),
            mlp_exactness: 100.0,
            softmax_coldness: 100.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidationOutcome {
    pub result: ValidationResult,
    pub code: ErrorCode,
    pub expr_id: Option<ExprId>,
}

// ---------------------------------------------------------------------------
// 1. Static Validation
// ---------------------------------------------------------------------------

pub fn static_validate(prog: &RaspProgram, sink: ExprId) -> ValidationOutcome {
    if (sink.0 as usize) >= prog.len() {
        return ValidationOutcome {
            result: ValidationResult::Invalid,
            code: ErrorCode::InvalidExpr,
            expr_id: Some(sink),
        };
    }

    let mut stack = Vec::with_capacity(prog.len());
    let mut visited = vec![0u8; prog.len()];
    stack.push(sink);
    visited[sink.0 as usize] = 1;

    while let Some(eid) = stack.pop() {
        let node = prog.node(eid);
        match node.op {
            RaspOp::Tokens | RaspOp::Indices => {
                if node.is_numerical() {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedEncoding,
                        expr_id: Some(eid),
                    };
                }
            }
            RaspOp::Select { keys, queries, .. } => {
                if !prog.node(keys).is_categorical() || !prog.node(queries).is_categorical() {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedEncoding,
                        expr_id: Some(eid),
                    };
                }
                for dep in [keys, queries] {
                    if visited[dep.0 as usize] == 0 {
                        visited[dep.0 as usize] = 1;
                        stack.push(dep);
                    }
                }
            }
            RaspOp::Aggregate {
                selector,
                sop,
                default,
            } => {
                let sop_node = prog.node(sop);
                if node.encoding != sop_node.encoding {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedAggregate,
                        expr_id: Some(eid),
                    };
                }
                if node.is_categorical() && default != Value::None {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedAggregate,
                        expr_id: Some(eid),
                    };
                }
                if node.is_numerical() && default.as_f64() != Some(0.0) {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedAggregate,
                        expr_id: Some(eid),
                    };
                }
                if !matches!(prog.node(selector).op, RaspOp::Select { .. }) {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedSelector,
                        expr_id: Some(eid),
                    };
                }
                for dep in [selector, sop] {
                    if visited[dep.0 as usize] == 0 {
                        visited[dep.0 as usize] = 1;
                        stack.push(dep);
                    }
                }
            }
            RaspOp::LinearSequenceMap { fst, snd, .. } => {
                if !prog.node(fst).is_numerical()
                    || !prog.node(snd).is_numerical()
                    || !node.is_numerical()
                {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedSequenceMap,
                        expr_id: Some(eid),
                    };
                }
                for dep in [fst, snd] {
                    if visited[dep.0 as usize] == 0 {
                        visited[dep.0 as usize] = 1;
                        stack.push(dep);
                    }
                }
            }
            RaspOp::SequenceMap { fst, snd, .. } => {
                if !prog.node(fst).is_categorical()
                    || !prog.node(snd).is_categorical()
                    || !node.is_categorical()
                {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedSequenceMap,
                        expr_id: Some(eid),
                    };
                }
                for dep in [fst, snd] {
                    if visited[dep.0 as usize] == 0 {
                        visited[dep.0 as usize] = 1;
                        stack.push(dep);
                    }
                }
            }
            RaspOp::Map { inner, .. } => {
                if visited[inner.0 as usize] == 0 {
                    visited[inner.0 as usize] = 1;
                    stack.push(inner);
                }
            }
            RaspOp::SelectorWidth { selector } => {
                if !matches!(prog.node(selector).op, RaspOp::Select { .. }) {
                    return ValidationOutcome {
                        result: ValidationResult::Invalid,
                        code: ErrorCode::UnsupportedSelector,
                        expr_id: Some(eid),
                    };
                }
                if visited[selector.0 as usize] == 0 {
                    visited[selector.0 as usize] = 1;
                    stack.push(selector);
                }
            }
            RaspOp::SelectorAnd { .. } | RaspOp::SelectorOr { .. } => {
                return ValidationOutcome {
                    result: ValidationResult::Invalid,
                    code: ErrorCode::UnsupportedSelector,
                    expr_id: Some(eid),
                };
            }
            RaspOp::SelectorNot { .. } | RaspOp::ConstantSelector { .. } => {
                return ValidationOutcome {
                    result: ValidationResult::Invalid,
                    code: ErrorCode::UnsupportedSelector,
                    expr_id: Some(eid),
                };
            }
            RaspOp::Full(_) => {}
        }
    }

    ValidationOutcome {
        result: ValidationResult::Valid,
        code: ErrorCode::None,
        expr_id: None,
    }
}

// ---------------------------------------------------------------------------
// 2. Basis Inference
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct NodeBasisInfo {
    value_set: Vec<Value>,
    output_space: Option<VectorSpace>,
}

#[inline]
fn expr_namespace(prog: &RaspProgram, eid: ExprId) -> DirNamespace {
    match prog.node(eid).op {
        RaspOp::Tokens => DirNamespace::Tokens,
        RaspOp::Indices => DirNamespace::Indices,
        _ => DirNamespace::Expr(eid),
    }
}

fn infer_bases(
    prog: &RaspProgram,
    graph: &CsrGraph,
    vocab: &[Value],
    max_seq_len: usize,
    reg: &mut BasisRegistry,
) -> Result<Vec<NodeBasisInfo>, ErrorCode> {
    let mut info = vec![
        NodeBasisInfo {
            value_set: Vec::new(),
            output_space: None,
        };
        graph.num_nodes
    ];

    let mut order = Vec::with_capacity(graph.num_nodes);
    graph.reverse_postorder(&mut order);

    for &nid in &order {
        let eid = graph.expr_of(nid);
        let rnode = prog.node(eid);
        if !rnode.is_sop() {
            continue;
        }

        let mut vs: Vec<Value> = match rnode.op {
            RaspOp::Tokens => vocab.to_vec(),
            RaspOp::Indices => (0..max_seq_len).map(|i| Value::Int(i as i32)).collect(),
            RaspOp::SelectorWidth { .. } => (0..=max_seq_len).map(|i| Value::Int(i as i32)).collect(),
            RaspOp::Full(v) => vec![v],
            RaspOp::Map { inner, func } => {
                let in_nid = graph.node_of(inner).ok_or(ErrorCode::InvalidGraph)?;
                let mut out = Vec::with_capacity(info[in_nid.0 as usize].value_set.len());
                for &x in &info[in_nid.0 as usize].value_set {
                    if let Some(y) = func(x) {
                        out.push(y);
                    }
                }
                out
            }
            RaspOp::SequenceMap { fst, snd, func } => {
                let fst_nid = graph.node_of(fst).ok_or(ErrorCode::InvalidGraph)?;
                let snd_nid = graph.node_of(snd).ok_or(ErrorCode::InvalidGraph)?;
                let fvs = &info[fst_nid.0 as usize].value_set;
                let svs = &info[snd_nid.0 as usize].value_set;
                let mut out = Vec::with_capacity(fvs.len() * svs.len());
                for &a in fvs {
                    for &b in svs {
                        if let Some(y) = func(a, b) {
                            out.push(y);
                        }
                    }
                }
                out
            }
            RaspOp::LinearSequenceMap {
                fst,
                snd,
                fst_fac,
                snd_fac,
            } => {
                let fst_nid = graph.node_of(fst).ok_or(ErrorCode::InvalidGraph)?;
                let snd_nid = graph.node_of(snd).ok_or(ErrorCode::InvalidGraph)?;
                let fvs = &info[fst_nid.0 as usize].value_set;
                let svs = &info[snd_nid.0 as usize].value_set;
                let mut out = Vec::with_capacity(fvs.len() * svs.len());
                for &a in fvs {
                    for &b in svs {
                        if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
                            out.push(Value::from_f64(fst_fac * fa + snd_fac * fb));
                        }
                    }
                }
                out
            }
            RaspOp::Aggregate { sop, .. } => {
                let sop_nid = graph.node_of(sop).ok_or(ErrorCode::InvalidGraph)?;
                let svs = &info[sop_nid.0 as usize].value_set;
                if rnode.is_categorical() {
                    svs.clone()
                } else {
                    for &v in svs {
                        let iv = v.as_int().ok_or(ErrorCode::NonBinaryNumericalAggregate)?;
                        if iv != 0 && iv != 1 {
                            return Err(ErrorCode::NonBinaryNumericalAggregate);
                        }
                    }
                    let mut out = Vec::with_capacity(svs.len() * max_seq_len);
                    for &v in svs {
                        let fv = v.as_f64().unwrap_or(0.0);
                        for len in 1..=max_seq_len {
                            out.push(Value::from_f64(fv / (len as f64)));
                        }
                    }
                    out
                }
            }
            _ => return Err(ErrorCode::InvalidExpr),
        };

        vs.sort();
        vs.dedup();

        let ns = expr_namespace(prog, eid);
        let space = match rnode.encoding {
            Encoding::Categorical => {
                let dirs: Vec<DirId> = vs
                    .iter()
                    .map(|&v| reg.intern(BasisDir::new(ns, BasisVal::Val(v))))
                    .collect();
                VectorSpace::new(dirs)
            }
            Encoding::Numerical => {
                let d = reg.intern(BasisDir::named(ns));
                VectorSpace::new(vec![d])
            }
        };

        info[nid.0 as usize].value_set = vs;
        info[nid.0 as usize].output_space = Some(space);
    }

    Ok(info)
}

// ---------------------------------------------------------------------------
// 3. Expr to Craft Graph
// ---------------------------------------------------------------------------

fn build_craft_blocks(
    prog: &RaspProgram,
    graph: &CsrGraph,
    basis_info: &[NodeBasisInfo],
    bos_dir: DirId,
    one_dir: DirId,
    cfg: &CompileConfig,
    reg: &mut BasisRegistry,
) -> Result<Vec<Option<CraftBlock>>, ErrorCode> {
    let mut blocks = vec![None; graph.num_nodes];

    for nid_idx in 0..graph.num_nodes {
        let nid = NodeId(nid_idx as u32);
        let eid = graph.expr_of(nid);
        let rnode = prog.node(eid);
        if !rnode.is_sop() {
            continue;
        }

        let block = match rnode.op {
            RaspOp::Tokens | RaspOp::Indices => None,
            RaspOp::Full(val) => {
                let out_space = basis_info[nid_idx]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let one_space = VectorSpace::new(vec![one_dir]);
                if rnode.is_categorical() {
                    let target = reg.intern(BasisDir::new(
                        expr_namespace(prog, eid),
                        BasisVal::Val(val),
                    ));
                    Some(CraftBlock::Mlp(craft::map_categorical_mlp(
                        &one_space,
                        out_space,
                        |_| Some(target),
                    )))
                } else {
                    let fv = val.as_f64().ok_or(ErrorCode::ArithmeticError)?;
                    let mut fst_mat = DenseMat::zeros(1, 1);
                    fst_mat.set(0, 0, fv);
                    let fst = LinearMap {
                        input_space: one_space,
                        output_space: out_space.clone(),
                        matrix: fst_mat,
                    };
                    let snd = LinearMap::project(out_space, out_space);
                    Some(CraftBlock::Mlp(Mlp::new(fst, snd)))
                }
            }
            RaspOp::Map { inner, func } => {
                let in_nid = graph.node_of(inner).ok_or(ErrorCode::InvalidGraph)?;
                let in_node = prog.node(inner);
                let in_space = basis_info[in_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let out_space = basis_info[nid_idx]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let out_ns = expr_namespace(prog, eid);

                let mlp = match (in_node.encoding, rnode.encoding) {
                    (Encoding::Categorical, Encoding::Categorical) => {
                        craft::map_categorical_mlp(in_space, out_space, |d| {
                            if let BasisVal::Val(v) = reg.get(d).val {
                                func(v).map(|res| reg.intern(BasisDir::new(out_ns, BasisVal::Val(res))))
                            } else {
                                None
                            }
                        })
                    }
                    (Encoding::Categorical, Encoding::Numerical) => {
                        craft::map_categorical_to_numerical_mlp(reg, in_space, out_space, |v| {
                            func(v).and_then(|res| res.as_f64())
                        })?
                    }
                    (Encoding::Numerical, Encoding::Categorical) => {
                        let in_vals: Vec<f64> = basis_info[in_nid.0 as usize]
                            .value_set
                            .iter()
                            .filter_map(|v| v.as_f64())
                            .collect();
                        craft::map_numerical_to_categorical_mlp(
                            reg,
                            |x| func(Value::from_f64(x)),
                            in_space,
                            out_space,
                            &in_vals,
                            one_dir,
                            cfg.mlp_exactness,
                        )?
                    }
                    (Encoding::Numerical, Encoding::Numerical) => {
                        let in_vals: Vec<f64> = basis_info[in_nid.0 as usize]
                            .value_set
                            .iter()
                            .filter_map(|v| v.as_f64())
                            .collect();
                        craft::map_numerical_mlp(
                            reg,
                            |x| func(Value::from_f64(x)).and_then(|res| res.as_f64()),
                            in_space,
                            out_space,
                            &in_vals,
                            one_dir,
                            cfg.mlp_exactness,
                        )?
                    }
                };
                Some(CraftBlock::Mlp(mlp))
            }
            RaspOp::LinearSequenceMap {
                fst,
                snd,
                fst_fac,
                snd_fac,
            } => {
                let fst_nid = graph.node_of(fst).ok_or(ErrorCode::InvalidGraph)?;
                let snd_nid = graph.node_of(snd).ok_or(ErrorCode::InvalidGraph)?;
                let fst_dir = basis_info[fst_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?
                    .basis[0];
                let snd_dir = basis_info[snd_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?
                    .basis[0];
                let out_dir = basis_info[nid_idx]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?
                    .basis[0];
                Some(CraftBlock::Mlp(craft::linear_sequence_map_numerical_mlp(
                    reg, fst_dir, snd_dir, out_dir, fst_fac, snd_fac,
                )))
            }
            RaspOp::SequenceMap { fst, snd, func } => {
                let fst_nid = graph.node_of(fst).ok_or(ErrorCode::InvalidGraph)?;
                let snd_nid = graph.node_of(snd).ok_or(ErrorCode::InvalidGraph)?;
                let fst_space = basis_info[fst_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let snd_space = basis_info[snd_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let out_space = basis_info[nid_idx]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let out_ns = expr_namespace(prog, eid);

                let mlp = if fst_space == snd_space {
                    craft::map_categorical_mlp(fst_space, out_space, |d| {
                        if let BasisVal::Val(v) = reg.get(d).val {
                            func(v, v)
                                .map(|res| reg.intern(BasisDir::new(out_ns, BasisVal::Val(res))))
                        } else {
                            None
                        }
                    })
                } else {
                    craft::sequence_map_categorical_mlp(
                        reg,
                        fst_space,
                        snd_space,
                        out_space,
                        one_dir,
                        |reg, d1, d2| {
                            if let (BasisVal::Val(v1), BasisVal::Val(v2)) =
                                (reg.get(d1).val, reg.get(d2).val)
                            {
                                func(v1, v2).map(|res| {
                                    reg.intern(BasisDir::new(out_ns, BasisVal::Val(res)))
                                })
                            } else {
                                None
                            }
                        },
                    )?
                };
                Some(CraftBlock::Mlp(mlp))
            }
            RaspOp::Aggregate { selector, sop, .. } => {
                let sel_node = prog.node(selector);
                let (keys, queries, pred) = match sel_node.op {
                    RaspOp::Select {
                        keys,
                        queries,
                        pred,
                    } => (keys, queries, pred),
                    _ => return Err(ErrorCode::UnsupportedSelector),
                };
                let q_nid = graph.node_of(queries).ok_or(ErrorCode::InvalidGraph)?;
                let k_nid = graph.node_of(keys).ok_or(ErrorCode::InvalidGraph)?;
                let v_nid = graph.node_of(sop).ok_or(ErrorCode::InvalidGraph)?;

                let q_space = basis_info[q_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let k_space = basis_info[k_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let v_space = basis_info[v_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let out_space = basis_info[nid_idx]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;

                let attn = craft::categorical_attn(
                    q_space,
                    k_space,
                    v_space,
                    out_space,
                    bos_dir,
                    one_dir,
                    |qd, kd| {
                        if let (BasisVal::Val(qv), BasisVal::Val(kv)) =
                            (reg.get(qd).val, reg.get(kd).val)
                        {
                            pred.eval(kv, qv)
                        } else {
                            false
                        }
                    },
                    cfg.causal,
                    false,
                    true,
                    cfg.softmax_coldness,
                )?;
                Some(CraftBlock::Attn(attn))
            }
            RaspOp::SelectorWidth { selector } => {
                let sel_node = prog.node(selector);
                let (keys, queries, pred) = match sel_node.op {
                    RaspOp::Select {
                        keys,
                        queries,
                        pred,
                    } => (keys, queries, pred),
                    _ => return Err(ErrorCode::UnsupportedSelector),
                };
                let q_nid = graph.node_of(queries).ok_or(ErrorCode::InvalidGraph)?;
                let k_nid = graph.node_of(keys).ok_or(ErrorCode::InvalidGraph)?;
                let q_space = basis_info[q_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let k_space = basis_info[k_nid.0 as usize]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;
                let out_space = basis_info[nid_idx]
                    .output_space
                    .as_ref()
                    .ok_or(ErrorCode::InvalidBasis)?;

                let out_vals: Vec<i32> = basis_info[nid_idx]
                    .value_set
                    .iter()
                    .filter_map(|v| v.as_int())
                    .collect();

                let (attn, mlp) = craft::selector_width(
                    reg,
                    eid,
                    q_space,
                    k_space,
                    out_space,
                    bos_dir,
                    one_dir,
                    |reg, qd, kd| {
                        if let (BasisVal::Val(qv), BasisVal::Val(kv)) =
                            (reg.get(qd).val, reg.get(kd).val)
                        {
                            pred.eval(kv, qv)
                        } else {
                            false
                        }
                    },
                    &out_vals,
                    rnode.is_categorical(),
                    cfg.causal,
                    cfg.softmax_coldness,
                    cfg.mlp_exactness,
                )?;
                Some(CraftBlock::ResidualPair(attn, mlp))
            }
            _ => return Err(ErrorCode::InvalidExpr),
        };

        blocks[nid_idx] = block;
    }

    Ok(blocks)
}

// ---------------------------------------------------------------------------
// 4. Craft Graph to Model (Layer Allocation & Condensation)
// ---------------------------------------------------------------------------

fn compute_computational_depth(prog: &RaspProgram, graph: &CsrGraph) -> Vec<i32> {
    let mut depth = vec![-1i32; graph.num_nodes];

    fn dfs(prog: &RaspProgram, graph: &CsrGraph, nid: NodeId, d: i32, depth: &mut [i32]) {
        if d > depth[nid.0 as usize] {
            depth[nid.0 as usize] = d;
        }
        for &succ in graph.successors(nid) {
            let succ_expr = graph.expr_of(succ);
            let next_d = if prog.node(succ_expr).is_sop() {
                d + 1
            } else {
                d
            };
            dfs(prog, graph, succ, next_d, depth);
        }
    }

    for &src in graph.sources.as_slice() {
        dfs(prog, graph, src, 0, &mut depth);
    }
    depth
}

fn craft_graph_to_model(
    prog: &RaspProgram,
    graph: &CsrGraph,
    mut blocks: Vec<Option<CraftBlock>>,
) -> SeriesWithResiduals {
    let comp_depth = compute_computational_depth(prog, graph);

    let mut max_depth = 0i32;
    let mut min_depth = i32::MAX;
    let mut node_depths: Vec<(NodeId, i32)> = Vec::new();

    for nid_idx in 0..graph.num_nodes {
        if blocks[nid_idx].is_some() {
            let d = comp_depth[nid_idx];
            if d > max_depth {
                max_depth = d;
            }
            if d < min_depth {
                min_depth = d;
            }
            node_depths.push((NodeId(nid_idx as u32), d));
        }
    }

    if node_depths.is_empty() {
        return SeriesWithResiduals {
            blocks: Vec::new(),
            residual_space: VectorSpace::empty(),
        };
    }

    // Condense adjacent (all-Attn at d, all-Mlp at d+1) levels
    let mut depth_map = vec![-1i32; graph.num_nodes];
    for &(nid, d) in &node_depths {
        depth_map[nid.0 as usize] = d;
    }

    let mut d = min_depth;
    while d < max_depth {
        let mut all_attn_d = true;
        let mut any_d = false;
        let mut all_mlp_d1 = true;
        let mut any_d1 = false;

        for nid_idx in 0..graph.num_nodes {
            if depth_map[nid_idx] == d {
                any_d = true;
                if !matches!(
                    blocks[nid_idx],
                    Some(CraftBlock::Attn(_)) | Some(CraftBlock::MultiAttn(_))
                ) {
                    all_attn_d = false;
                }
            } else if depth_map[nid_idx] == d + 1 {
                any_d1 = true;
                if !matches!(blocks[nid_idx], Some(CraftBlock::Mlp(_))) {
                    all_mlp_d1 = false;
                }
            }
        }

        if any_d && any_d1 && all_attn_d && all_mlp_d1 {
            for nid_idx in 0..graph.num_nodes {
                if depth_map[nid_idx] > d {
                    depth_map[nid_idx] -= 1;
                }
            }
            max_depth -= 1;
        }
        d += 1;
    }

    // Allocate to alternating layers: 0=attn, 1=mlp, 2=attn, 3=mlp, ...
    let mut layer_alloc = vec![-1i32; graph.num_nodes];
    let mut items: Vec<(NodeId, i32)> = (0..graph.num_nodes)
        .filter(|&i| depth_map[i] >= 0)
        .map(|i| (NodeId(i as u32), depth_map[i]))
        .collect();
    items.sort_by_key(|&(_, d)| d);

    let mut current_layer = 0i32;
    let mut current_depth = 1i32;
    for (nid, d) in items {
        while d > current_depth {
            current_depth += 1;
            current_layer += 2;
        }
        if d == current_depth {
            let is_mlp = matches!(blocks[nid.0 as usize], Some(CraftBlock::Mlp(_)));
            layer_alloc[nid.0 as usize] = current_layer + i32::from(is_mlp);
        }
    }

    let max_layer = layer_alloc.iter().copied().max().unwrap_or(-1);
    let num_half_layers = if max_layer >= 0 {
        (max_layer + 2) as usize // room for ResidualPair (+1)
    } else {
        0
    };

    let mut attn_by_layer: Vec<Vec<AttentionHead>> = vec![Vec::new(); num_half_layers];
    let mut mlp_by_layer: Vec<Vec<Mlp>> = vec![Vec::new(); num_half_layers];
    let mut space_refs: Vec<VectorSpace> = Vec::new();

    for nid_idx in 0..graph.num_nodes {
        let lno = layer_alloc[nid_idx];
        if lno < 0 {
            continue;
        }
        let lno = lno as usize;
        if let Some(blk) = blocks[nid_idx].take() {
            space_refs.push(blk.residual_space());
            match blk {
                CraftBlock::Attn(h) => attn_by_layer[lno].push(h),
                CraftBlock::MultiAttn(mh) => attn_by_layer[lno].extend(mh.heads),
                CraftBlock::Mlp(m) => mlp_by_layer[lno].push(m),
                CraftBlock::ResidualPair(a, m) => {
                    attn_by_layer[lno].push(a);
                    mlp_by_layer[lno + 1].push(m);
                }
            }
        }
    }

    let ref_slices: Vec<&VectorSpace> = space_refs.iter().collect();
    let residual_space = VectorSpace::join(&ref_slices);

    let mut model_blocks = Vec::new();
    for lno in 0..num_half_layers {
        if !attn_by_layer[lno].is_empty() {
            let heads = core::mem::take(&mut attn_by_layer[lno]);
            let mut mh = MultiAttentionHead::new(heads);
            mh.residual_space = residual_space.clone();
            model_blocks.push(LayerBlock::Attn(mh));
        }
        if !mlp_by_layer[lno].is_empty() {
            let mlps = core::mem::take(&mut mlp_by_layer[lno]);
            let refs: Vec<&Mlp> = mlps.iter().collect();
            let mut combined = Mlp::combine_in_parallel(&refs);
            combined.residual_space = residual_space.clone();
            model_blocks.push(LayerBlock::Mlp(combined));
        }
    }

    SeriesWithResiduals {
        blocks: model_blocks,
        residual_space,
    }
}

// ---------------------------------------------------------------------------
// 5. Assemble Craft Model into CompiledTransformer
// ---------------------------------------------------------------------------

fn assemble_transformer(
    craft_model: &SeriesWithResiduals,
    tokens_space: &VectorSpace,
    indices_space: &VectorSpace,
    output_space: &VectorSpace,
    vocab_with_specials: &[Value],
    output_vocab: &[Value],
    one_dir: DirId,
    categorical_output: bool,
    cfg: &CompileConfig,
) -> CompiledTransformer {
    // Map craft_model.blocks onto transformer/layer_L/{attn, mlp}
    let mut layer_pairs: Vec<(Option<&MultiAttentionHead>, Option<&Mlp>)> = Vec::new();
    let mut cur_attn: Option<&MultiAttentionHead> = None;

    for blk in &craft_model.blocks {
        match blk {
            LayerBlock::Attn(mh) => {
                if cur_attn.is_some() {
                    layer_pairs.push((cur_attn.take(), None));
                }
                cur_attn = Some(mh);
            }
            LayerBlock::Mlp(m) => {
                layer_pairs.push((cur_attn.take(), Some(m)));
            }
        }
    }
    if cur_attn.is_some() {
        layer_pairs.push((cur_attn, None));
    }
    if layer_pairs.is_empty() {
        layer_pairs.push((None, None));
    }

    let num_layers = layer_pairs.len();
    let num_heads = layer_pairs
        .iter()
        .filter_map(|(a, _)| a.map(|mh| mh.heads.len()))
        .max()
        .unwrap_or(1)
        .max(1);
    let key_size = layer_pairs
        .iter()
        .filter_map(|(a, _)| {
            a.and_then(|mh| {
                mh.heads
                    .iter()
                    .map(|h| h.w_qk.matrix.rows.max(h.w_qk.matrix.cols))
                    .max()
            })
        })
        .max()
        .unwrap_or(1)
        .max(1);
    let mlp_hidden_size = layer_pairs
        .iter()
        .filter_map(|(_, m)| m.map(|mlp| mlp.fst.output_space.num_dims()))
        .max()
        .unwrap_or(1)
        .max(1);

    let residual_space = VectorSpace::join(&[
        &craft_model.residual_space,
        tokens_space,
        indices_space,
        output_space,
    ]);
    let d = residual_space.num_dims();
    let hk = num_heads * key_size;

    // Token embeddings: [vocab_size, d]
    let mut tokens_to_res = LinearMap::project(tokens_space, &residual_space);
    if residual_space.contains(one_dir) {
        let one_col = residual_space.index_of(one_dir).unwrap();
        for r in 0..tokens_space.num_dims() {
            tokens_to_res.matrix.add_at(r, one_col, 1.0);
        }
    }
    // Note: tokens_space.basis is sorted by DirId, whereas vocab_with_specials[i]
    // maps to a specific DirId in tokens_space. We index token_embed by TokenId!
    let mut token_embed = vec![0.0; vocab_with_specials.len() * d];
    for (tid, _) in vocab_with_specials.iter().enumerate() {
        // Each token in vocab_with_specials corresponds to its BasisDir in tokens_space.
        // We copy row `r` of tokens_to_res where `tokens_space.basis[r]` matches!
        // Since tokens_space was built from vocab_with_specials, we pass the dir_ids or
        // look up via `tid`: we'll store token_embed ordered by `vocab_with_specials` index below.
        let _ = tid;
    }

    // Positional embeddings: row 0 = zeros (BOS), rows 1..=max_seq_len = indices_space
    let max_total_len = cfg.max_seq_len + 1;
    let index_to_res = LinearMap::project(indices_space, &residual_space);
    let mut pos_embed = vec![0.0; max_total_len * d];
    for pos in 0..cfg.max_seq_len {
        let dst_off = (pos + 1) * d;
        let src_off = pos * d;
        if src_off + d <= index_to_res.matrix.data.len() {
            pos_embed[dst_off..dst_off + d]
                .copy_from_slice(&index_to_res.matrix.data[src_off..src_off + d]);
        }
    }

    // Unembedding: [d, out_dim]
    let res_to_out = LinearMap::project(&residual_space, output_space);
    let out_dim = output_space.num_dims();
    let unembed = res_to_out.matrix.data;

    // Layers
    let mut layers = Vec::with_capacity(num_layers);
    for (opt_attn, opt_mlp) in layer_pairs {
        let mut w_q = vec![0.0; d * hk];
        let mut w_k = vec![0.0; d * hk];
        let mut w_v = vec![0.0; d * hk];
        let mut w_o = vec![0.0; hk * d];
        let mut w_1 = vec![0.0; d * mlp_hidden_size];
        let mut w_2 = vec![0.0; mlp_hidden_size * d];

        if let Some(mh) = opt_attn {
            for (h_idx, head) in mh.heads.iter().enumerate() {
                let head_col_off = h_idx * key_size;
                let res_to_q = LinearMap::project(&residual_space, &head.w_qk.left_space).matrix;
                let q_mat = res_to_q.matmul(&head.w_qk.matrix); // [d, head_k_sz]
                let head_k_sz = head.w_qk.matrix.cols;
                for r in 0..d {
                    for c in 0..head_k_sz {
                        w_q[r * hk + head_col_off + c] = q_mat.get(r, c);
                    }
                }

                let res_to_k = LinearMap::project(&residual_space, &head.w_qk.right_space).matrix;
                for r in 0..d {
                    for c in 0..head_k_sz {
                        w_k[r * hk + head_col_off + c] = res_to_k.get(r, c);
                    }
                }

                let res_to_v = LinearMap::project(&residual_space, &head.w_ov.input_space).matrix;
                let v_mat = res_to_v.matmul(&head.w_ov.matrix); // [d, head_v_sz]
                let head_v_sz = head.w_ov.matrix.cols;
                for r in 0..d {
                    for c in 0..head_v_sz {
                        w_v[r * hk + head_col_off + c] = v_mat.get(r, c);
                    }
                }

                let o_mat = LinearMap::project(&residual_space, &head.w_ov.output_space)
                    .matrix
                    .transpose(); // [head_v_sz, d]
                for r in 0..head_v_sz {
                    for c in 0..d {
                        w_o[(head_col_off + r) * d + c] = o_mat.get(r, c);
                    }
                }
            }
        }

        if let Some(mlp) = opt_mlp {
            let h_sz = mlp.fst.output_space.num_dims();
            let res_to_fst = LinearMap::project(&residual_space, &mlp.fst.input_space).matrix;
            let fst_mat = res_to_fst.matmul(&mlp.fst.matrix); // [d, h_sz]
            for r in 0..d {
                for c in 0..h_sz {
                    w_1[r * mlp_hidden_size + c] = fst_mat.get(r, c);
                }
            }

            let snd_to_res = LinearMap::project(&residual_space, &mlp.snd.output_space)
                .matrix
                .transpose(); // [out_d, d]
            let snd_mat = mlp.snd.matrix.matmul(&snd_to_res); // [h_sz, d]
            for r in 0..h_sz {
                for c in 0..d {
                    w_2[r * d + c] = snd_mat.get(r, c);
                }
            }
        }

        layers.push(CompiledLayer {
            w_q,
            w_k,
            w_v,
            w_o,
            w_1,
            w_2,
        });
    }

    // Fill token_embed placeholder (caller will populate exact rows by TokenId)
    let _ = &mut token_embed;

    CompiledTransformer {
        num_layers,
        num_heads,
        key_size,
        mlp_hidden_size,
        d_model: d,
        out_dim,
        max_total_len,
        causal: cfg.causal,
        categorical_output,
        bos_token_id: None,
        bos_value: cfg.bos_value,
        pad_token_id: None,
        vocab: vocab_with_specials.to_vec(),
        output_vocab: output_vocab.to_vec(),
        token_embed,
        pos_embed,
        unembed,
        layers,
    }
}

// ---------------------------------------------------------------------------
// Public Compiler Entrypoint
// ---------------------------------------------------------------------------

pub fn compile_rasp_to_model<const CAP: usize>(
    prog: &RaspProgram,
    sink: ExprId,
    vocab: &[Value],
    cfg: &CompileConfig,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<CompiledTransformer, ErrorCode> {
    telemetry.emit(
        Module::Compiler,
        EventKind::CompileStarted,
        Status::Running,
        ErrorCode::None,
        if cfg.causal { 1 } else { 0 },
        sink.0,
        cfg.max_seq_len as u32,
    );

    if vocab.contains(&cfg.bos_value) || vocab.contains(&cfg.pad_value) {
        telemetry.emit(
            Module::Compiler,
            EventKind::ValidationFailed,
            Status::Failed,
            ErrorCode::BosOrPadInVocab,
            FLAG_FAILED,
            sink.0,
            0,
        );
        return Err(ErrorCode::BosOrPadInVocab);
    }

    let val = static_validate(prog, sink);
    if val.result != ValidationResult::Valid {
        telemetry.emit(
            Module::Validator,
            EventKind::ValidationFailed,
            Status::Failed,
            val.code,
            FLAG_FAILED,
            val.expr_id.unwrap_or(sink).0,
            0,
        );
        return Err(val.code);
    }
    telemetry.emit(
        Module::Validator,
        EventKind::ValidationPassed,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        sink.0,
        0,
    );

    let graph = extract_rasp_graph(prog, sink)?;
    telemetry.emit(
        Module::Graph,
        EventKind::GraphExtracted,
        Status::Completed,
        ErrorCode::None,
        0,
        sink.0,
        graph.num_nodes as u32,
    );

    let tokens_expr = prog.tokens_id().ok_or(ErrorCode::MissingTokensNode)?;
    if graph.node_of(tokens_expr).is_none() {
        return Err(ErrorCode::MissingTokensNode);
    }

    let mut reg = BasisRegistry::new();
    let one_dir = reg.one_dir();
    let bos_dir = reg.intern(BasisDir::new(
        DirNamespace::Tokens,
        BasisVal::Val(cfg.bos_value),
    ));

    let basis_info = infer_bases(prog, &graph, vocab, cfg.max_seq_len, &mut reg)?;
    telemetry.emit(
        Module::BasisInference,
        EventKind::BasesInferred,
        Status::Completed,
        ErrorCode::None,
        0,
        sink.0,
        graph.num_nodes as u32,
    );

    let blocks = build_craft_blocks(prog, &graph, &basis_info, bos_dir, one_dir, cfg, &mut reg)?;
    telemetry.emit(
        Module::Craft,
        EventKind::CraftBlocksBuilt,
        Status::Completed,
        ErrorCode::None,
        0,
        sink.0,
        blocks.iter().filter(|b| b.is_some()).count() as u32,
    );

    let craft_model = craft_graph_to_model(prog, &graph, blocks);

    // Build tokens_space (vocab + bos + pad), indices_space (0..max_seq_len), output_space
    let mut full_vocab = vocab.to_vec();
    full_vocab.push(cfg.bos_value);
    full_vocab.push(cfg.pad_value);
    full_vocab.sort();
    full_vocab.dedup();

    let token_dirs: Vec<DirId> = full_vocab
        .iter()
        .map(|&v| reg.intern(BasisDir::new(DirNamespace::Tokens, BasisVal::Val(v))))
        .collect();
    let tokens_space = VectorSpace::new(token_dirs);

    let index_dirs: Vec<DirId> = (0..cfg.max_seq_len)
        .map(|i| {
            reg.intern(BasisDir::new(
                DirNamespace::Indices,
                BasisVal::Val(Value::Int(i as i32)),
            ))
        })
        .collect();
    let indices_space = VectorSpace::new(index_dirs);

    let sink_nid = graph.sink;
    let output_space = basis_info[sink_nid.0 as usize]
        .output_space
        .as_ref()
        .ok_or(ErrorCode::InvalidBasis)?;
    let categorical_output = prog.node(sink).is_categorical();

    let output_vocab: Vec<Value> = if categorical_output {
        output_space
            .basis
            .iter()
            .filter_map(|&d| match reg.get(d).val {
                BasisVal::Val(v) => Some(v),
                _ => None,
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut model = assemble_transformer(
        &craft_model,
        &tokens_space,
        &indices_space,
        output_space,
        &full_vocab,
        &output_vocab,
        one_dir,
        categorical_output,
        cfg,
    );

    // Populate token_embed and pos_embed with exact basis-direction matching
    let residual_space = VectorSpace::join(&[
        &craft_model.residual_space,
        &tokens_space,
        &indices_space,
        output_space,
    ]);
    let d = model.d_model;
    model.token_embed.fill(0.0);
    for (tid, &val) in full_vocab.iter().enumerate() {
        let dir = reg.intern(BasisDir::new(DirNamespace::Tokens, BasisVal::Val(val)));
        if let Some(col) = residual_space.index_of(dir) {
            model.token_embed[tid * d + col] = 1.0;
        }
        if let Some(one_col) = residual_space.index_of(one_dir) {
            model.token_embed[tid * d + one_col] = 1.0;
        }
    }

    model.pos_embed.fill(0.0);
    for pos in 0..cfg.max_seq_len {
        let dir = reg.intern(BasisDir::new(
            DirNamespace::Indices,
            BasisVal::Val(Value::Int(pos as i32)),
        ));
        if let Some(col) = residual_space.index_of(dir) {
            model.pos_embed[(pos + 1) * d + col] = 1.0;
        }
    }

    model.bos_token_id = full_vocab
        .iter()
        .position(|&v| v == cfg.bos_value)
        .map(|i| i as u32);
    model.pad_token_id = full_vocab
        .iter()
        .position(|&v| v == cfg.pad_value)
        .map(|i| i as u32);

    telemetry.emit(
        Module::Assembler,
        EventKind::ModelAssembled,
        Status::Completed,
        ErrorCode::None,
        if categorical_output { FLAG_CATEGORICAL } else { 0 },
        sink.0,
        model.num_layers as u32,
    );

    Ok(model)
}
