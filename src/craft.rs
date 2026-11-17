//! Craft: Vector-space linear/bilinear algebra, Transformer blocks, and Chamber
//! weight-space constructions.
//!
//! Uses compact `DirId(u32)` handles in a `BasisRegistry` and contiguous
//! row-major `DenseMat` buffers (zero `String`s, zero `Vec<Vec<f64>>`).

use crate::rasp::Value;
use crate::sys::{DirId, ErrorCode, ExprId};

// ---------------------------------------------------------------------------
// Basis Directions & Registry (Zero-String Symbolic Representation)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DirNamespace {
    One,
    Tokens,
    Indices,
    Expr(ExprId),
    Hidden(u32),
    HiddenStart(u32),
    HiddenX(u32),
    HiddenY(u32),
    SelectorWidthAttnOut(ExprId),
    CleanBosHidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BasisVal {
    None,
    Val(Value),
    Pair(DirId, DirId),
    Thresh(u32, u8),
    Sign(i8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BasisDir {
    pub ns: DirNamespace,
    pub val: BasisVal,
}

impl BasisDir {
    #[inline]
    pub const fn new(ns: DirNamespace, val: BasisVal) -> Self {
        Self { ns, val }
    }

    #[inline]
    pub const fn named(ns: DirNamespace) -> Self {
        Self {
            ns,
            val: BasisVal::None,
        }
    }
}

pub struct BasisRegistry {
    dirs: Vec<BasisDir>,
    next_hidden_id: u32,
}

impl BasisRegistry {
    pub fn new() -> Self {
        Self {
            dirs: Vec::with_capacity(128),
            next_hidden_id: 1,
        }
    }

    pub fn intern(&mut self, dir: BasisDir) -> DirId {
        if let Some(pos) = self.dirs.iter().position(|&d| d == dir) {
            return DirId(pos as u32);
        }
        let id = DirId(self.dirs.len() as u32);
        self.dirs.push(dir);
        id
    }

    #[inline]
    pub fn get(&self, id: DirId) -> BasisDir {
        self.dirs[id.0 as usize]
    }

    #[inline]
    pub fn alloc_hidden_tag(&mut self) -> u32 {
        let id = self.next_hidden_id;
        self.next_hidden_id += 1;
        id
    }

    pub fn one_dir(&mut self) -> DirId {
        self.intern(BasisDir::named(DirNamespace::One))
    }
}

impl Default for BasisRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VectorSpace (Sorted, Deduplicated DirIds)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorSpace {
    pub basis: Vec<DirId>,
}

impl VectorSpace {
    pub fn new(mut dirs: Vec<DirId>) -> Self {
        dirs.sort_unstable();
        dirs.dedup();
        Self { basis: dirs }
    }

    pub fn empty() -> Self {
        Self { basis: Vec::new() }
    }

    #[inline]
    pub fn num_dims(&self) -> usize {
        self.basis.len()
    }

    #[inline]
    pub fn contains(&self, dir: DirId) -> bool {
        self.basis.binary_search(&dir).is_ok()
    }

    #[inline]
    pub fn index_of(&self, dir: DirId) -> Option<usize> {
        self.basis.binary_search(&dir).ok()
    }

    pub fn join(spaces: &[&VectorSpace]) -> Self {
        let total: usize = spaces.iter().map(|s| s.basis.len()).sum();
        let mut all = Vec::with_capacity(total);
        for s in spaces {
            all.extend_from_slice(&s.basis);
        }
        Self::new(all)
    }

    pub fn is_disjoint(&self, other: &VectorSpace) -> bool {
        let (mut i, mut j) = (0, 0);
        while i < self.basis.len() && j < other.basis.len() {
            if self.basis[i] == other.basis[j] {
                return false;
            } else if self.basis[i] < other.basis[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Contiguous Dense Matrix
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct DenseMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    #[inline]
    pub fn add_at(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] += val;
    }

    /// Matrix multiply `self (M x K) * rhs (K x N) -> out (M x N)`
    pub fn matmul(&self, rhs: &DenseMat) -> DenseMat {
        debug_assert_eq!(self.cols, rhs.rows);
        let mut out = DenseMat::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..rhs.cols {
                    out.data[i * rhs.cols + j] += a * rhs.data[k * rhs.cols + j];
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> DenseMat {
        let mut out = DenseMat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Linear & Bilinear VectorSpace Functions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LinearMap {
    pub input_space: VectorSpace,
    pub output_space: VectorSpace,
    /// Shape: `[input_space.num_dims(), output_space.num_dims()]`
    pub matrix: DenseMat,
}

impl LinearMap {
    pub fn project(from: &VectorSpace, to: &VectorSpace) -> Self {
        let mut matrix = DenseMat::zeros(from.num_dims(), to.num_dims());
        for (i, &d) in from.basis.iter().enumerate() {
            if let Some(j) = to.index_of(d) {
                matrix.set(i, j, 1.0);
            }
        }
        Self {
            input_space: from.clone(),
            output_space: to.clone(),
            matrix,
        }
    }

    pub fn combine_in_parallel(fns: &[&LinearMap]) -> Self {
        let in_refs: Vec<&VectorSpace> = fns.iter().map(|f| &f.input_space).collect();
        let out_refs: Vec<&VectorSpace> = fns.iter().map(|f| &f.output_space).collect();
        let input_space = VectorSpace::join(&in_refs);
        let output_space = VectorSpace::join(&out_refs);
        let mut matrix = DenseMat::zeros(input_space.num_dims(), output_space.num_dims());

        for f in fns {
            for (r, &in_d) in f.input_space.basis.iter().enumerate() {
                let gr = input_space.index_of(in_d).unwrap();
                for (c, &out_d) in f.output_space.basis.iter().enumerate() {
                    let v = f.matrix.get(r, c);
                    if v != 0.0 {
                        let gc = output_space.index_of(out_d).unwrap();
                        matrix.add_at(gr, gc, v);
                    }
                }
            }
        }

        Self {
            input_space,
            output_space,
            matrix,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScalarBilinear {
    pub left_space: VectorSpace,
    pub right_space: VectorSpace,
    /// Shape: `[left_space.num_dims(), right_space.num_dims()]`
    pub matrix: DenseMat,
}

// ---------------------------------------------------------------------------
// Craft Blocks
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AttentionHead {
    pub w_qk: ScalarBilinear,
    pub w_ov: LinearMap,
    pub residual_space: VectorSpace,
    pub causal: bool,
}

impl AttentionHead {
    pub fn new(w_qk: ScalarBilinear, w_ov: LinearMap, causal: bool) -> Self {
        let residual_space = VectorSpace::join(&[
            &w_qk.left_space,
            &w_qk.right_space,
            &w_ov.input_space,
            &w_ov.output_space,
        ]);
        Self {
            w_qk,
            w_ov,
            residual_space,
            causal,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MultiAttentionHead {
    pub heads: Vec<AttentionHead>,
    pub residual_space: VectorSpace,
}

impl MultiAttentionHead {
    pub fn new(heads: Vec<AttentionHead>) -> Self {
        let refs: Vec<&VectorSpace> = heads.iter().map(|h| &h.residual_space).collect();
        let residual_space = VectorSpace::join(&refs);
        Self {
            heads,
            residual_space,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mlp {
    pub fst: LinearMap,
    pub snd: LinearMap,
    pub residual_space: VectorSpace,
}

impl Mlp {
    pub fn new(fst: LinearMap, snd: LinearMap) -> Self {
        let residual_space = VectorSpace::join(&[&fst.input_space, &snd.output_space]);
        Self {
            fst,
            snd,
            residual_space,
        }
    }

    pub fn combine_in_parallel(mlps: &[&Mlp]) -> Self {
        let fsts: Vec<&LinearMap> = mlps.iter().map(|m| &m.fst).collect();
        let snds: Vec<&LinearMap> = mlps.iter().map(|m| &m.snd).collect();
        let fst = LinearMap::combine_in_parallel(&fsts);
        let snd = LinearMap::combine_in_parallel(&snds);
        Self::new(fst, snd)
    }
}

#[derive(Clone, Debug)]
pub enum CraftBlock {
    Attn(AttentionHead),
    MultiAttn(MultiAttentionHead),
    Mlp(Mlp),
    ResidualPair(AttentionHead, Mlp),
}

impl CraftBlock {
    pub fn residual_space(&self) -> VectorSpace {
        match self {
            CraftBlock::Attn(h) => h.residual_space.clone(),
            CraftBlock::MultiAttn(m) => m.residual_space.clone(),
            CraftBlock::Mlp(m) => m.residual_space.clone(),
            CraftBlock::ResidualPair(a, m) => {
                VectorSpace::join(&[&a.residual_space, &m.residual_space])
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum LayerBlock {
    Attn(MultiAttentionHead),
    Mlp(Mlp),
}

#[derive(Clone, Debug)]
pub struct SeriesWithResiduals {
    pub blocks: Vec<LayerBlock>,
    pub residual_space: VectorSpace,
}

// ---------------------------------------------------------------------------
// Chamber: Categorical Attention
// ---------------------------------------------------------------------------

pub fn categorical_attn<F>(
    query_space: &VectorSpace,
    key_space: &VectorSpace,
    value_space: &VectorSpace,
    output_space: &VectorSpace,
    bos_dir: DirId,
    one_dir: DirId,
    mut attn_fn: F,
    causal: bool,
    always_attend_to_bos: bool,
    use_bos_for_default_output: bool,
    softmax_coldness: f64,
) -> Result<AttentionHead, ErrorCode>
where
    F: FnMut(DirId, DirId) -> bool,
{
    let bos_space = VectorSpace::new(vec![bos_dir]);
    let one_space = VectorSpace::new(vec![one_dir]);

    let q_space = VectorSpace::join(&[query_space, &bos_space, &one_space]);
    let k_space = VectorSpace::join(&[key_space, &bos_space]);
    let v_space = VectorSpace::join(&[value_space, &bos_space]);

    let value_basis: Vec<DirId> = if always_attend_to_bos {
        v_space.basis.clone()
    } else {
        v_space
            .basis
            .iter()
            .copied()
            .filter(|&d| d != bos_dir)
            .collect()
    };

    if value_basis.len() != output_space.num_dims() {
        return Err(ErrorCode::DimensionMismatch);
    }

    let mut qk_mat = DenseMat::zeros(q_space.num_dims(), k_space.num_dims());
    for (qi, &qd) in q_space.basis.iter().enumerate() {
        for (ki, &kd) in k_space.basis.iter().enumerate() {
            let val = if kd == bos_dir && qd == one_dir {
                let c = if always_attend_to_bos { 1.0 } else { 0.5 };
                c * softmax_coldness
            } else if kd == one_dir || kd == bos_dir || qd == one_dir || qd == bos_dir {
                0.0
            } else if attn_fn(qd, kd) {
                softmax_coldness
            } else {
                0.0
            };
            qk_mat.set(qi, ki, val);
        }
    }

    let mut ov_mat = DenseMat::zeros(v_space.num_dims(), output_space.num_dims());
    for (vi, &vd) in v_space.basis.iter().enumerate() {
        if use_bos_for_default_output && vd == bos_dir {
            // Default output is null_vector (all zeros).
            continue;
        }
        if let Some(pos) = value_basis.iter().position(|&d| d == vd) {
            let out_dir = output_space.basis[pos];
            let out_col = output_space.index_of(out_dir).unwrap();
            ov_mat.set(vi, out_col, 1.0);
        }
    }

    Ok(AttentionHead::new(
        ScalarBilinear {
            left_space: q_space,
            right_space: k_space,
            matrix: qk_mat,
        },
        LinearMap {
            input_space: v_space,
            output_space: output_space.clone(),
            matrix: ov_mat,
        },
        causal,
    ))
}

// ---------------------------------------------------------------------------
// Chamber: Categorical MLPs
// ---------------------------------------------------------------------------

pub fn map_categorical_mlp<F>(
    input_space: &VectorSpace,
    output_space: &VectorSpace,
    mut operation: F,
) -> Mlp
where
    F: FnMut(DirId) -> Option<DirId>,
{
    let mut fst_mat = DenseMat::zeros(input_space.num_dims(), output_space.num_dims());
    for (r, &in_d) in input_space.basis.iter().enumerate() {
        if let Some(out_d) = operation(in_d) {
            if let Some(c) = output_space.index_of(out_d) {
                fst_mat.set(r, c, 1.0);
            }
        }
    }
    let fst = LinearMap {
        input_space: input_space.clone(),
        output_space: output_space.clone(),
        matrix: fst_mat,
    };
    let snd = LinearMap::project(output_space, output_space);
    Mlp::new(fst, snd)
}

pub fn map_categorical_to_numerical_mlp<F>(
    reg: &BasisRegistry,
    input_space: &VectorSpace,
    output_space: &VectorSpace,
    mut operation: F,
) -> Result<Mlp, ErrorCode>
where
    F: FnMut(Value) -> Option<f64>,
{
    if output_space.num_dims() != 1 {
        return Err(ErrorCode::DimensionMismatch);
    }
    let mut fst_mat = DenseMat::zeros(input_space.num_dims(), 1);
    for (r, &in_d) in input_space.basis.iter().enumerate() {
        if let BasisVal::Val(v) = reg.get(in_d).val {
            if let Some(res) = operation(v) {
                fst_mat.set(r, 0, res);
            }
        }
    }
    let fst = LinearMap {
        input_space: input_space.clone(),
        output_space: output_space.clone(),
        matrix: fst_mat,
    };
    let snd = LinearMap::project(output_space, output_space);
    Ok(Mlp::new(fst, snd))
}

pub fn sequence_map_categorical_mlp<F>(
    reg: &mut BasisRegistry,
    input1_space: &VectorSpace,
    input2_space: &VectorSpace,
    output_space: &VectorSpace,
    one_dir: DirId,
    mut operation: F,
) -> Result<Mlp, ErrorCode>
where
    F: FnMut(&mut BasisRegistry, DirId, DirId) -> Option<DirId>,
{
    if !input1_space.is_disjoint(input2_space) {
        return Err(ErrorCode::InvalidBasis);
    }

    let one_space = VectorSpace::new(vec![one_dir]);
    let input_space = VectorSpace::join(&[input1_space, input2_space, &one_space]);

    let hidden_tag = reg.alloc_hidden_tag();
    let mut hidden_dirs = Vec::with_capacity(input1_space.num_dims() * input2_space.num_dims());
    for &d1 in &input1_space.basis {
        for &d2 in &input2_space.basis {
            let hd = reg.intern(BasisDir::new(
                DirNamespace::Hidden(hidden_tag),
                BasisVal::Pair(d1, d2),
            ));
            hidden_dirs.push(hd);
        }
    }
    let hidden_space = VectorSpace::new(hidden_dirs);

    let mut fst_mat = DenseMat::zeros(input_space.num_dims(), hidden_space.num_dims());
    for (r, &in_d) in input_space.basis.iter().enumerate() {
        if in_d == one_dir {
            for c in 0..hidden_space.num_dims() {
                fst_mat.set(r, c, -1.0);
            }
        } else if input1_space.contains(in_d) {
            for &d2 in &input2_space.basis {
                let hd = reg.intern(BasisDir::new(
                    DirNamespace::Hidden(hidden_tag),
                    BasisVal::Pair(in_d, d2),
                ));
                let c = hidden_space.index_of(hd).unwrap();
                fst_mat.set(r, c, 1.0);
            }
        } else {
            for &d1 in &input1_space.basis {
                let hd = reg.intern(BasisDir::new(
                    DirNamespace::Hidden(hidden_tag),
                    BasisVal::Pair(d1, in_d),
                ));
                let c = hidden_space.index_of(hd).unwrap();
                fst_mat.set(r, c, 1.0);
            }
        }
    }

    let mut snd_mat = DenseMat::zeros(hidden_space.num_dims(), output_space.num_dims());
    for (r, &hd) in hidden_space.basis.iter().enumerate() {
        if let BasisVal::Pair(d1, d2) = reg.get(hd).val {
            if let Some(out_d) = operation(reg, d1, d2) {
                if let Some(c) = output_space.index_of(out_d) {
                    snd_mat.set(r, c, 1.0);
                }
            }
        }
    }

    Ok(Mlp::new(
        LinearMap {
            input_space,
            output_space: hidden_space.clone(),
            matrix: fst_mat,
        },
        LinearMap {
            input_space: hidden_space,
            output_space: output_space.clone(),
            matrix: snd_mat,
        },
    ))
}

// ---------------------------------------------------------------------------
// Chamber: Numerical MLPs (Discretising Layer)
// ---------------------------------------------------------------------------

struct DiscretisingLayer {
    hidden_space: VectorSpace,
    start_dir: DirId,
    hidden_tag: u32,
    output_values: Vec<f64>,
    fst_mat: DenseMat,
    full_input_space: VectorSpace,
}

fn build_discretising_layer<F>(
    reg: &mut BasisRegistry,
    input_space: &VectorSpace,
    one_dir: DirId,
    input_value_set: &[f64],
    mut f: F,
    large_number: f64,
) -> Result<DiscretisingLayer, ErrorCode>
where
    F: FnMut(f64) -> Option<f64>,
{
    if input_space.num_dims() != 1 {
        return Err(ErrorCode::DimensionMismatch);
    }

    let mut sorted_in = input_value_set.to_vec();
    sorted_in.sort_by(|a, b| a.total_cmp(b));
    sorted_in.dedup();

    let mut sorted_vals = Vec::with_capacity(sorted_in.len());
    let mut output_values = Vec::with_capacity(sorted_in.len());
    for x in sorted_in {
        if let Some(y) = f(x) {
            sorted_vals.push(x);
            output_values.push(y);
        }
    }
    if sorted_vals.is_empty() {
        return Err(ErrorCode::InvalidBasis);
    }

    let num_vals = sorted_vals.len();
    let thresholds: Vec<f64> = (0..num_vals.saturating_sub(1))
        .map(|i| 0.5 * (sorted_vals[i] + sorted_vals[i + 1]))
        .collect();

    let hidden_tag = reg.alloc_hidden_tag();
    let start_dir = reg.intern(BasisDir::named(DirNamespace::HiddenStart(hidden_tag)));
    let mut hidden_dirs = Vec::with_capacity(1 + 2 * num_vals.saturating_sub(1));
    hidden_dirs.push(start_dir);
    for k in 1..num_vals {
        hidden_dirs.push(reg.intern(BasisDir::new(
            DirNamespace::Hidden(hidden_tag),
            BasisVal::Thresh(k as u32, 0),
        )));
        hidden_dirs.push(reg.intern(BasisDir::new(
            DirNamespace::Hidden(hidden_tag),
            BasisVal::Thresh(k as u32, 1),
        )));
    }
    let hidden_space = VectorSpace::new(hidden_dirs);

    let one_space = VectorSpace::new(vec![one_dir]);
    let full_input_space = VectorSpace::join(&[input_space, &one_space]);
    let mut fst_mat = DenseMat::zeros(full_input_space.num_dims(), hidden_space.num_dims());

    for (r, &in_d) in full_input_space.basis.iter().enumerate() {
        if in_d == one_dir {
            let start_c = hidden_space.index_of(start_dir).unwrap();
            fst_mat.set(r, start_c, 1.0);
            for k in 1..num_vals {
                let c0 = hidden_space
                    .index_of(reg.intern(BasisDir::new(
                        DirNamespace::Hidden(hidden_tag),
                        BasisVal::Thresh(k as u32, 0),
                    )))
                    .unwrap();
                let c1 = hidden_space
                    .index_of(reg.intern(BasisDir::new(
                        DirNamespace::Hidden(hidden_tag),
                        BasisVal::Thresh(k as u32, 1),
                    )))
                    .unwrap();
                let t = thresholds[k - 1];
                fst_mat.add_at(r, c0, 1.0 - large_number * t);
                fst_mat.add_at(r, c1, -large_number * t);
            }
        } else {
            for k in 1..num_vals {
                let c0 = hidden_space
                    .index_of(reg.intern(BasisDir::new(
                        DirNamespace::Hidden(hidden_tag),
                        BasisVal::Thresh(k as u32, 0),
                    )))
                    .unwrap();
                let c1 = hidden_space
                    .index_of(reg.intern(BasisDir::new(
                        DirNamespace::Hidden(hidden_tag),
                        BasisVal::Thresh(k as u32, 1),
                    )))
                    .unwrap();
                fst_mat.add_at(r, c0, large_number);
                fst_mat.add_at(r, c1, large_number);
            }
        }
    }

    Ok(DiscretisingLayer {
        hidden_space,
        start_dir,
        hidden_tag,
        output_values,
        fst_mat,
        full_input_space,
    })
}

pub fn map_numerical_mlp<F>(
    reg: &mut BasisRegistry,
    f: F,
    input_space: &VectorSpace,
    output_space: &VectorSpace,
    input_value_set: &[f64],
    one_dir: DirId,
    large_number: f64,
) -> Result<Mlp, ErrorCode>
where
    F: FnMut(f64) -> Option<f64>,
{
    if output_space.num_dims() != 1 {
        return Err(ErrorCode::DimensionMismatch);
    }
    let disc = build_discretising_layer(reg, input_space, one_dir, input_value_set, f, large_number)?;

    let mut snd_mat = DenseMat::zeros(disc.hidden_space.num_dims(), 1);
    for (r, &hd) in disc.hidden_space.basis.iter().enumerate() {
        if hd == disc.start_dir {
            snd_mat.set(r, 0, disc.output_values[0]);
        } else if let BasisVal::Thresh(k, i) = reg.get(hd).val {
            let k = k as usize;
            let sign = if i == 0 { 1.0 } else { -1.0 };
            let delta = disc.output_values[k] - disc.output_values[k - 1];
            snd_mat.set(r, 0, sign * delta);
        }
    }

    let _ = disc.hidden_tag;
    Ok(Mlp::new(
        LinearMap {
            input_space: disc.full_input_space,
            output_space: disc.hidden_space.clone(),
            matrix: disc.fst_mat,
        },
        LinearMap {
            input_space: disc.hidden_space,
            output_space: output_space.clone(),
            matrix: snd_mat,
        },
    ))
}

pub fn map_numerical_to_categorical_mlp<F>(
    reg: &mut BasisRegistry,
    f: F,
    input_space: &VectorSpace,
    output_space: &VectorSpace,
    input_value_set: &[f64],
    one_dir: DirId,
    large_number: f64,
) -> Result<Mlp, ErrorCode>
where
    F: FnMut(f64) -> Option<Value>,
{
    // Map f(x) -> Value, then look up corresponding output basis direction index
    let mut sorted_in = input_value_set.to_vec();
    sorted_in.sort_by(|a, b| a.total_cmp(b));
    sorted_in.dedup();

    let mut out_val_dirs = Vec::with_capacity(sorted_in.len());
    let mut f_eval = f;
    let mut filtered_in = Vec::with_capacity(sorted_in.len());

    for x in sorted_in {
        if let Some(v) = f_eval(x) {
            // Find matching direction in output_space
            let mut found = None;
            for (c, &od) in output_space.basis.iter().enumerate() {
                if reg.get(od).val == BasisVal::Val(v) {
                    found = Some(c);
                    break;
                }
            }
            let c = found.ok_or(ErrorCode::OutputValueMismatch)?;
            filtered_in.push(x);
            out_val_dirs.push(c);
        }
    }

    if filtered_in.is_empty() {
        return Err(ErrorCode::InvalidBasis);
    }

    let disc = build_discretising_layer(
        reg,
        input_space,
        one_dir,
        &filtered_in,
        |x| Some(x),
        large_number,
    )?;

    let mut snd_mat = DenseMat::zeros(disc.hidden_space.num_dims(), output_space.num_dims());
    for (r, &hd) in disc.hidden_space.basis.iter().enumerate() {
        if hd == disc.start_dir {
            snd_mat.set(r, out_val_dirs[0], 1.0);
        } else if let BasisVal::Thresh(k, i) = reg.get(hd).val {
            let k = k as usize;
            let sign = if i == 0 { 1.0 } else { -1.0 };
            snd_mat.add_at(r, out_val_dirs[k], sign);
            snd_mat.add_at(r, out_val_dirs[k - 1], -sign);
        }
    }

    Ok(Mlp::new(
        LinearMap {
            input_space: disc.full_input_space,
            output_space: disc.hidden_space.clone(),
            matrix: disc.fst_mat,
        },
        LinearMap {
            input_space: disc.hidden_space,
            output_space: output_space.clone(),
            matrix: snd_mat,
        },
    ))
}

pub fn linear_sequence_map_numerical_mlp(
    reg: &mut BasisRegistry,
    in1_dir: DirId,
    in2_dir: DirId,
    out_dir: DirId,
    in1_factor: f64,
    in2_factor: f64,
) -> Mlp {
    let input_space = VectorSpace::new(vec![in1_dir, in2_dir]);
    let output_space = VectorSpace::new(vec![out_dir]);

    let tag = reg.alloc_hidden_tag();
    let x_pos = reg.intern(BasisDir::new(DirNamespace::HiddenX(tag), BasisVal::Sign(1)));
    let x_neg = reg.intern(BasisDir::new(DirNamespace::HiddenX(tag), BasisVal::Sign(-1)));
    let y_pos = reg.intern(BasisDir::new(DirNamespace::HiddenY(tag), BasisVal::Sign(1)));
    let y_neg = reg.intern(BasisDir::new(DirNamespace::HiddenY(tag), BasisVal::Sign(-1)));
    let hidden_space = VectorSpace::new(vec![x_pos, x_neg, y_pos, y_neg]);

    let mut fst_mat = DenseMat::zeros(input_space.num_dims(), hidden_space.num_dims());
    for (r, &in_d) in input_space.basis.iter().enumerate() {
        if in_d == in1_dir {
            fst_mat.add_at(r, hidden_space.index_of(x_pos).unwrap(), 1.0);
            fst_mat.add_at(r, hidden_space.index_of(x_neg).unwrap(), -1.0);
        }
        if in_d == in2_dir {
            fst_mat.add_at(r, hidden_space.index_of(y_pos).unwrap(), 1.0);
            fst_mat.add_at(r, hidden_space.index_of(y_neg).unwrap(), -1.0);
        }
    }

    let mut snd_mat = DenseMat::zeros(hidden_space.num_dims(), 1);
    for (r, &hd) in hidden_space.basis.iter().enumerate() {
        let d = reg.get(hd);
        match (d.ns, d.val) {
            (DirNamespace::HiddenX(_), BasisVal::Sign(s)) => {
                snd_mat.set(r, 0, in1_factor * (s as f64));
            }
            (DirNamespace::HiddenY(_), BasisVal::Sign(s)) => {
                snd_mat.set(r, 0, in2_factor * (s as f64));
            }
            _ => {}
        }
    }

    Mlp::new(
        LinearMap {
            input_space,
            output_space: hidden_space.clone(),
            matrix: fst_mat,
        },
        LinearMap {
            input_space: hidden_space,
            output_space,
            matrix: snd_mat,
        },
    )
}

// ---------------------------------------------------------------------------
// Chamber: SelectorWidth
// ---------------------------------------------------------------------------

pub fn selector_width<F>(
    reg: &mut BasisRegistry,
    expr_id: ExprId,
    query_space: &VectorSpace,
    key_space: &VectorSpace,
    output_space: &VectorSpace,
    bos_dir: DirId,
    one_dir: DirId,
    mut attn_fn: F,
    out_value_set: &[i32],
    categorical_output: bool,
    causal: bool,
    softmax_coldness: f64,
    mlp_large_number: f64,
) -> Result<(AttentionHead, Mlp), ErrorCode>
where
    F: FnMut(&BasisRegistry, DirId, DirId) -> bool,
{
    let attn_out_dir = reg.intern(BasisDir::named(DirNamespace::SelectorWidthAttnOut(expr_id)));
    let attn_out_space = VectorSpace::new(vec![attn_out_dir]);
    let bos_space = VectorSpace::new(vec![bos_dir]);

    let attn = categorical_attn(
        query_space,
        key_space,
        &bos_space,
        &attn_out_space,
        bos_dir,
        one_dir,
        |qd, kd| attn_fn(reg, qd, kd),
        causal,
        true,
        false,
        softmax_coldness,
    )?;

    let in_value_set: Vec<f64> = out_value_set
        .iter()
        .map(|&v| 1.0 / ((v as f64) + 1.0))
        .collect();

    let mlp = if categorical_output {
        map_numerical_to_categorical_mlp(
            reg,
            |x| Some(Value::Int(((1.0 / x) - 1.0).round() as i32)),
            &attn_out_space,
            output_space,
            &in_value_set,
            one_dir,
            mlp_large_number,
        )?
    } else {
        map_numerical_mlp(
            reg,
            |x| Some(((1.0 / x) - 1.0).round()),
            &attn_out_space,
            output_space,
            &in_value_set,
            one_dir,
            mlp_large_number,
        )?
    };

    // BOS-scrubbing MLP: subtracts attn_out_dir (and zero_dir if categorical) at BOS position
    let clean_bos_out_space = VectorSpace::join(&[&attn_out_space, output_space]);
    let clean_hidden_dir = reg.intern(BasisDir::named(DirNamespace::CleanBosHidden));
    let clean_hidden_space = VectorSpace::new(vec![clean_hidden_dir]);

    let mut clean_fst_mat = DenseMat::zeros(1, 1);
    clean_fst_mat.set(0, 0, 1.0);
    let clean_fst = LinearMap {
        input_space: bos_space,
        output_space: clean_hidden_space.clone(),
        matrix: clean_fst_mat,
    };

    let mut clean_snd_mat = DenseMat::zeros(1, clean_bos_out_space.num_dims());
    let attn_out_col = clean_bos_out_space.index_of(attn_out_dir).unwrap();
    clean_snd_mat.set(0, attn_out_col, -1.0);

    if categorical_output {
        let zero_dir = output_space
            .basis
            .iter()
            .copied()
            .find(|&d| reg.get(d).val == BasisVal::Val(Value::Int(0)))
            .ok_or(ErrorCode::OutputValueMismatch)?;
        let zero_col = clean_bos_out_space.index_of(zero_dir).unwrap();
        clean_snd_mat.add_at(0, zero_col, -1.0);
    }

    let clean_snd = LinearMap {
        input_space: clean_hidden_space,
        output_space: clean_bos_out_space,
        matrix: clean_snd_mat,
    };
    let clean_bos_mlp = Mlp::new(clean_fst, clean_snd);

    let combined_mlp = Mlp::combine_in_parallel(&[&mlp, &clean_bos_mlp]);
    Ok((attn, combined_mlp))
}
