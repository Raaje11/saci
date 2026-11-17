//! RASP intermediate representation and reference evaluator.
//!
//! All expressions live in a contiguous `RaspProgram` arena indexed by `ExprId`.
//! Values are `Copy` (zero `String` / heap allocation).

use crate::sys::{ErrorCode, ExprId, SymId, FLAG_CATEGORICAL, FLAG_NUMERICAL};

/// Canonicalized f64 bits so `Value` implements `Eq`, `Ord`, `Hash`, `Copy`.
#[inline]
pub fn canonical_f64_bits(x: f64) -> u64 {
    if x == 0.0 {
        0.0_f64.to_bits()
    } else if x.is_nan() {
        f64::NAN.to_bits()
    } else {
        x.to_bits()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    None,
    Bool(bool),
    Int(i32),
    Float(u64),
    Sym(SymId),
}

impl Value {
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Value::Float(canonical_f64_bits(x))
    }

    #[inline]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(i as f64),
            Value::Float(bits) => Some(f64::from_bits(bits)),
            Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
            Value::None | Value::Sym(_) => None,
        }
    }

    #[inline]
    pub fn as_int(self) -> Option<i32> {
        match self {
            Value::Int(i) => Some(i),
            Value::Bool(b) => Some(if b { 1 } else { 0 }),
            Value::Float(bits) => {
                let f = f64::from_bits(bits);
                if f.is_finite() && f.fract() == 0.0 {
                    Some(f as i32)
                } else {
                    None
                }
            }
            Value::None | Value::Sym(_) => None,
        }
    }

    #[inline]
    pub fn as_bool(self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(b),
            Value::Int(i) => Some(i != 0),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Categorical = 0,
    Numerical = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    Eq = 0,
    Neq = 1,
    Lt = 2,
    Leq = 3,
    Gt = 4,
    Geq = 5,
}

impl Comparison {
    #[inline]
    pub fn eval(self, key: Value, query: Value) -> bool {
        match self {
            Comparison::Eq => key == query,
            Comparison::Neq => key != query,
            Comparison::Lt => match (key.as_f64(), query.as_f64()) {
                (Some(a), Some(b)) => a < b,
                _ => key < query,
            },
            Comparison::Leq => match (key.as_f64(), query.as_f64()) {
                (Some(a), Some(b)) => a <= b,
                _ => key <= query,
            },
            Comparison::Gt => match (key.as_f64(), query.as_f64()) {
                (Some(a), Some(b)) => a > b,
                _ => key > query,
            },
            Comparison::Geq => match (key.as_f64(), query.as_f64()) {
                (Some(a), Some(b)) => a >= b,
                _ => key >= query,
            },
        }
    }
}

#[derive(Clone, Copy)]
pub enum SelectorPred {
    Cmp(Comparison),
    Fn(fn(Value, Value) -> bool),
}

impl SelectorPred {
    #[inline]
    pub fn eval(self, key: Value, query: Value) -> bool {
        match self {
            SelectorPred::Cmp(c) => c.eval(key, query),
            SelectorPred::Fn(f) => f(key, query),
        }
    }
}

#[derive(Clone, Copy)]
pub enum RaspOp {
    Tokens,
    Indices,
    Full(Value),
    Map {
        inner: ExprId,
        func: fn(Value) -> Option<Value>,
    },
    SequenceMap {
        fst: ExprId,
        snd: ExprId,
        func: fn(Value, Value) -> Option<Value>,
    },
    LinearSequenceMap {
        fst: ExprId,
        snd: ExprId,
        fst_fac: f64,
        snd_fac: f64,
    },
    Select {
        keys: ExprId,
        queries: ExprId,
        pred: SelectorPred,
    },
    ConstantSelector {
        value: bool,
        causal: bool,
    },
    SelectorAnd {
        fst: ExprId,
        snd: ExprId,
    },
    SelectorOr {
        fst: ExprId,
        snd: ExprId,
    },
    SelectorNot {
        inner: ExprId,
    },
    Aggregate {
        selector: ExprId,
        sop: ExprId,
        default: Value,
    },
    SelectorWidth {
        selector: ExprId,
    },
}

#[derive(Clone, Copy)]
pub struct RaspNode {
    pub id: ExprId,
    pub op: RaspOp,
    pub encoding: Encoding,
    pub flags: u16,
}

impl RaspNode {
    #[inline]
    pub const fn is_sop(&self) -> bool {
        !matches!(
            self.op,
            RaspOp::Select { .. }
                | RaspOp::ConstantSelector { .. }
                | RaspOp::SelectorAnd { .. }
                | RaspOp::SelectorOr { .. }
                | RaspOp::SelectorNot { .. }
        )
    }

    #[inline]
    pub const fn is_categorical(&self) -> bool {
        matches!(self.encoding, Encoding::Categorical)
    }

    #[inline]
    pub const fn is_numerical(&self) -> bool {
        matches!(self.encoding, Encoding::Numerical)
    }
}

/// Contiguous arena holding all expressions of a RASP program.
/// Cold-path initialization structure; expressions are referenced by `ExprId`.
pub struct RaspProgram {
    nodes: Vec<RaspNode>,
    tokens_id: Option<ExprId>,
    indices_id: Option<ExprId>,
}

impl RaspProgram {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(32),
            tokens_id: None,
            indices_id: None,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn node(&self, id: ExprId) -> &RaspNode {
        &self.nodes[id.0 as usize]
    }

    #[inline]
    pub fn node_mut(&mut self, id: ExprId) -> &mut RaspNode {
        &mut self.nodes[id.0 as usize]
    }

    #[inline]
    pub fn tokens_id(&self) -> Option<ExprId> {
        self.tokens_id
    }

    fn push_node(&mut self, op: RaspOp, encoding: Encoding) -> ExprId {
        let id = ExprId(self.nodes.len() as u32);
        let flags = match encoding {
            Encoding::Categorical => FLAG_CATEGORICAL,
            Encoding::Numerical => FLAG_NUMERICAL,
        };
        self.nodes.push(RaspNode {
            id,
            op,
            encoding,
            flags,
        });
        id
    }

    pub fn tokens(&mut self) -> ExprId {
        if let Some(id) = self.tokens_id {
            return id;
        }
        let id = self.push_node(RaspOp::Tokens, Encoding::Categorical);
        self.tokens_id = Some(id);
        id
    }

    pub fn indices(&mut self) -> ExprId {
        if let Some(id) = self.indices_id {
            return id;
        }
        let id = self.push_node(RaspOp::Indices, Encoding::Categorical);
        self.indices_id = Some(id);
        id
    }

    pub fn full(&mut self, value: Value) -> ExprId {
        self.push_node(RaspOp::Full(value), Encoding::Categorical)
    }

    pub fn map(&mut self, func: fn(Value) -> Option<Value>, inner: ExprId) -> ExprId {
        let enc = self.node(inner).encoding;
        self.push_node(RaspOp::Map { inner, func }, enc)
    }

    pub fn sequence_map(
        &mut self,
        func: fn(Value, Value) -> Option<Value>,
        fst: ExprId,
        snd: ExprId,
    ) -> ExprId {
        let enc = self.node(fst).encoding;
        self.push_node(RaspOp::SequenceMap { fst, snd, func }, enc)
    }

    pub fn linear_sequence_map(
        &mut self,
        fst: ExprId,
        snd: ExprId,
        fst_fac: f64,
        snd_fac: f64,
    ) -> ExprId {
        self.push_node(
            RaspOp::LinearSequenceMap {
                fst,
                snd,
                fst_fac,
                snd_fac,
            },
            Encoding::Numerical,
        )
    }

    pub fn select(&mut self, keys: ExprId, queries: ExprId, cmp: Comparison) -> ExprId {
        self.push_node(
            RaspOp::Select {
                keys,
                queries,
                pred: SelectorPred::Cmp(cmp),
            },
            Encoding::Categorical,
        )
    }

    pub fn select_fn(
        &mut self,
        keys: ExprId,
        queries: ExprId,
        pred: fn(Value, Value) -> bool,
    ) -> ExprId {
        self.push_node(
            RaspOp::Select {
                keys,
                queries,
                pred: SelectorPred::Fn(pred),
            },
            Encoding::Categorical,
        )
    }

    pub fn constant_selector(&mut self, value: bool, causal: bool) -> ExprId {
        self.push_node(
            RaspOp::ConstantSelector { value, causal },
            Encoding::Categorical,
        )
    }

    pub fn selector_and(&mut self, fst: ExprId, snd: ExprId) -> ExprId {
        self.push_node(RaspOp::SelectorAnd { fst, snd }, Encoding::Categorical)
    }

    pub fn selector_or(&mut self, fst: ExprId, snd: ExprId) -> ExprId {
        self.push_node(RaspOp::SelectorOr { fst, snd }, Encoding::Categorical)
    }

    pub fn selector_not(&mut self, inner: ExprId) -> ExprId {
        self.push_node(RaspOp::SelectorNot { inner }, Encoding::Categorical)
    }

    pub fn aggregate(&mut self, selector: ExprId, sop: ExprId, default: Value) -> ExprId {
        let enc = self.node(sop).encoding;
        self.push_node(
            RaspOp::Aggregate {
                selector,
                sop,
                default,
            },
            enc,
        )
    }

    pub fn selector_width(&mut self, selector: ExprId) -> ExprId {
        self.push_node(RaspOp::SelectorWidth { selector }, Encoding::Categorical)
    }

    pub fn categorical(&mut self, id: ExprId) -> ExprId {
        let n = self.node_mut(id);
        n.encoding = Encoding::Categorical;
        n.flags = (n.flags & !FLAG_NUMERICAL) | FLAG_CATEGORICAL;
        id
    }

    pub fn numerical(&mut self, id: ExprId) -> ExprId {
        let n = self.node_mut(id);
        n.encoding = Encoding::Numerical;
        n.flags = (n.flags & !FLAG_CATEGORICAL) | FLAG_NUMERICAL;
        id
    }
}

impl Default for RaspProgram {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Preallocated RASP Reference Evaluator
// ---------------------------------------------------------------------------

pub struct EvalScratch {
    max_seq_len: usize,
    num_exprs: usize,
    /// Flat [num_exprs * max_seq_len] S-Op values
    sop_vals: Vec<Value>,
    /// Flat [num_exprs * max_seq_len * max_seq_len] selector matrices
    sel_vals: Vec<bool>,
    /// Track which exprs have been evaluated in the current pass
    computed: Vec<u8>,
}

impl EvalScratch {
    pub fn new(num_exprs: usize, max_seq_len: usize) -> Self {
        Self {
            max_seq_len,
            num_exprs,
            sop_vals: vec![Value::None; num_exprs * max_seq_len],
            sel_vals: vec![false; num_exprs * max_seq_len * max_seq_len],
            computed: vec![0; num_exprs],
        }
    }

    pub fn reset(&mut self, num_exprs: usize, max_seq_len: usize) {
        if num_exprs > self.num_exprs || max_seq_len > self.max_seq_len {
            self.num_exprs = num_exprs.max(self.num_exprs);
            self.max_seq_len = max_seq_len.max(self.max_seq_len);
            self.sop_vals.resize(self.num_exprs * self.max_seq_len, Value::None);
            self.sel_vals
                .resize(self.num_exprs * self.max_seq_len * self.max_seq_len, false);
            self.computed.resize(self.num_exprs, 0);
        }
        self.computed[..num_exprs].fill(0);
    }

    #[inline]
    fn sop_slice(&self, id: ExprId, seq_len: usize) -> &[Value] {
        let off = id.0 as usize * self.max_seq_len;
        &self.sop_vals[off..off + seq_len]
    }

    #[inline]
    fn sop_slice_mut(&mut self, id: ExprId, seq_len: usize) -> &mut [Value] {
        let off = id.0 as usize * self.max_seq_len;
        &mut self.sop_vals[off..off + seq_len]
    }

    #[inline]
    fn sel_slice(&self, id: ExprId, seq_len: usize) -> &[bool] {
        let off = id.0 as usize * self.max_seq_len * self.max_seq_len;
        &self.sel_vals[off..off + seq_len * seq_len]
    }

    #[inline]
    fn sel_slice_mut(&mut self, id: ExprId, seq_len: usize) -> &mut [bool] {
        let off = id.0 as usize * self.max_seq_len * self.max_seq_len;
        &mut self.sel_vals[off..off + seq_len * seq_len]
    }
}

pub fn evaluate_rasp<'a>(
    prog: &RaspProgram,
    sink: ExprId,
    xs: &[Value],
    causal: bool,
    scratch: &'a mut EvalScratch,
) -> Result<&'a [Value], ErrorCode> {
    if xs.is_empty() {
        return Err(ErrorCode::EmptyInput);
    }
    scratch.reset(prog.len(), xs.len());
    eval_expr(prog, sink, xs, causal, scratch)?;
    Ok(scratch.sop_slice(sink, xs.len()))
}

fn eval_expr(
    prog: &RaspProgram,
    id: ExprId,
    xs: &[Value],
    causal: bool,
    scratch: &mut EvalScratch,
) -> Result<(), ErrorCode> {
    let idx = id.0 as usize;
    if scratch.computed[idx] != 0 {
        return Ok(());
    }
    let seq_len = xs.len();
    let op = prog.node(id).op;

    match op {
        RaspOp::Tokens => {
            scratch.sop_slice_mut(id, seq_len).copy_from_slice(xs);
        }
        RaspOp::Indices => {
            let out = scratch.sop_slice_mut(id, seq_len);
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = Value::Int(i as i32);
            }
        }
        RaspOp::Full(v) => {
            scratch.sop_slice_mut(id, seq_len).fill(v);
        }
        RaspOp::Map { inner, func } => {
            eval_expr(prog, inner, xs, causal, scratch)?;
            for i in 0..seq_len {
                let v = scratch.sop_slice(inner, seq_len)[i];
                let res = func(v).ok_or(ErrorCode::ArithmeticError)?;
                scratch.sop_slice_mut(id, seq_len)[i] = res;
            }
        }
        RaspOp::SequenceMap { fst, snd, func } => {
            eval_expr(prog, fst, xs, causal, scratch)?;
            eval_expr(prog, snd, xs, causal, scratch)?;
            for i in 0..seq_len {
                let a = scratch.sop_slice(fst, seq_len)[i];
                let b = scratch.sop_slice(snd, seq_len)[i];
                let res = func(a, b).ok_or(ErrorCode::ArithmeticError)?;
                scratch.sop_slice_mut(id, seq_len)[i] = res;
            }
        }
        RaspOp::LinearSequenceMap {
            fst,
            snd,
            fst_fac,
            snd_fac,
        } => {
            eval_expr(prog, fst, xs, causal, scratch)?;
            eval_expr(prog, snd, xs, causal, scratch)?;
            for i in 0..seq_len {
                let a = scratch.sop_slice(fst, seq_len)[i]
                    .as_f64()
                    .ok_or(ErrorCode::ArithmeticError)?;
                let b = scratch.sop_slice(snd, seq_len)[i]
                    .as_f64()
                    .ok_or(ErrorCode::ArithmeticError)?;
                scratch.sop_slice_mut(id, seq_len)[i] =
                    Value::from_f64(fst_fac * a + snd_fac * b);
            }
        }
        RaspOp::Select {
            keys,
            queries,
            pred,
        } => {
            eval_expr(prog, keys, xs, causal, scratch)?;
            eval_expr(prog, queries, xs, causal, scratch)?;
            for q_idx in 0..seq_len {
                let q_val = scratch.sop_slice(queries, seq_len)[q_idx];
                let max_k = if causal { q_idx + 1 } else { seq_len };
                for k_idx in 0..seq_len {
                    let selected = if k_idx < max_k {
                        let k_val = scratch.sop_slice(keys, seq_len)[k_idx];
                        pred.eval(k_val, q_val)
                    } else {
                        false
                    };
                    scratch.sel_slice_mut(id, seq_len)[q_idx * seq_len + k_idx] = selected;
                }
            }
        }
        RaspOp::ConstantSelector {
            value,
            causal: sel_causal,
        } => {
            let is_causal = causal || sel_causal;
            let out = scratch.sel_slice_mut(id, seq_len);
            for q in 0..seq_len {
                for k in 0..seq_len {
                    out[q * seq_len + k] = if is_causal && k > q { false } else { value };
                }
            }
        }
        RaspOp::SelectorAnd { fst, snd } => {
            eval_expr(prog, fst, xs, causal, scratch)?;
            eval_expr(prog, snd, xs, causal, scratch)?;
            for idx_cell in 0..(seq_len * seq_len) {
                let a = scratch.sel_slice(fst, seq_len)[idx_cell];
                let b = scratch.sel_slice(snd, seq_len)[idx_cell];
                scratch.sel_slice_mut(id, seq_len)[idx_cell] = a && b;
            }
        }
        RaspOp::SelectorOr { fst, snd } => {
            eval_expr(prog, fst, xs, causal, scratch)?;
            eval_expr(prog, snd, xs, causal, scratch)?;
            for idx_cell in 0..(seq_len * seq_len) {
                let a = scratch.sel_slice(fst, seq_len)[idx_cell];
                let b = scratch.sel_slice(snd, seq_len)[idx_cell];
                scratch.sel_slice_mut(id, seq_len)[idx_cell] = a || b;
            }
        }
        RaspOp::SelectorNot { inner } => {
            eval_expr(prog, inner, xs, causal, scratch)?;
            for q in 0..seq_len {
                for k in 0..seq_len {
                    let cell = q * seq_len + k;
                    let v = !scratch.sel_slice(inner, seq_len)[cell];
                    scratch.sel_slice_mut(id, seq_len)[cell] =
                        if causal && k > q { false } else { v };
                }
            }
        }
        RaspOp::Aggregate {
            selector,
            sop,
            default,
        } => {
            eval_expr(prog, selector, xs, causal, scratch)?;
            eval_expr(prog, sop, xs, causal, scratch)?;
            let is_cat = prog.node(id).is_categorical();
            for q in 0..seq_len {
                let mut count = 0usize;
                let mut sum = 0.0f64;
                let mut first_val = Value::None;
                for k in 0..seq_len {
                    if scratch.sel_slice(selector, seq_len)[q * seq_len + k] {
                        let v = scratch.sop_slice(sop, seq_len)[k];
                        if count == 0 {
                            first_val = v;
                        }
                        if !is_cat {
                            sum += v.as_f64().unwrap_or(0.0);
                        }
                        count += 1;
                    }
                }
                let res = if count == 0 {
                    default
                } else if is_cat {
                    first_val
                } else {
                    Value::from_f64(sum / (count as f64))
                };
                scratch.sop_slice_mut(id, seq_len)[q] = res;
            }
        }
        RaspOp::SelectorWidth { selector } => {
            eval_expr(prog, selector, xs, causal, scratch)?;
            for q in 0..seq_len {
                let mut count = 0i32;
                for k in 0..seq_len {
                    if scratch.sel_slice(selector, seq_len)[q * seq_len + k] {
                        count += 1;
                    }
                }
                scratch.sop_slice_mut(id, seq_len)[q] = Value::Int(count);
            }
        }
    }

    scratch.computed[idx] = 1;
    Ok(())
}
