//! Superposition with Equiangular Tight Frame (ETF) features.
//!
//! Compresses a `CompiledTransformer`'s residual stream from `D` orthogonal
//! dimensions into `m <= D` superposed dimensions using deterministic ETF
//! feature vectors:
//! - `m == D`: exact identity (zero interference).
//! - `D == m + 1`: exact closed-form Helmert Simplex ETF (`coherence == 1/m`).
//! - `m < D - 1`: deterministic DCT-II initialized Alternating-Projection
//!   Tight Frame (APTF) via symmetric Jacobi eigendecomposition, converging
//!   toward the Welch bound `sqrt((D - m) / (m * (D - 1)))`.

use crate::craft::DenseMat;
use crate::sys::{ErrorCode, EventKind, Module, Status, TelemetryBus};
use crate::transformer::{CompiledLayer, CompiledTransformer};

#[derive(Clone, Copy, Debug)]
pub struct EtfConfig {
    pub target_dim: usize,
    pub max_iters: usize,
    pub tol: f64,
}

impl EtfConfig {
    pub const fn new(target_dim: usize) -> Self {
        Self {
            target_dim,
            max_iters: 80,
            tol: 1e-9,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EtfMetrics {
    pub orig_dim: usize,
    pub target_dim: usize,
    pub welch_bound: f64,
    pub max_coherence: f64,
    pub tightness_error: f64,
}

#[derive(Clone, Debug)]
pub struct EtfFrame {
    pub orig_dim: usize,
    pub target_dim: usize,
    /// `[orig_dim, target_dim]` — each row is a unit-norm feature vector in `R^target_dim`.
    pub features: DenseMat,
    pub metrics: EtfMetrics,
}

#[inline]
pub fn welch_bound(n: usize, m: usize) -> f64 {
    if n <= 1 || m >= n {
        return 0.0;
    }
    let nf = n as f64;
    let mf = m as f64;
    ((nf - mf) / (mf * (nf - 1.0))).sqrt()
}

/// Normalize every row of `mat` to unit L2 norm.
fn normalize_rows(mat: &mut DenseMat) {
    for r in 0..mat.rows {
        let mut sq = 0.0;
        let off = r * mat.cols;
        for c in 0..mat.cols {
            let v = mat.data[off + c];
            sq += v * v;
        }
        let norm = sq.sqrt();
        if norm > 1e-15 {
            let inv = 1.0 / norm;
            for c in 0..mat.cols {
                mat.data[off + c] *= inv;
            }
        } else if mat.cols > 0 {
            mat.data[off + (r % mat.cols)] = 1.0;
        }
    }
}

/// Symmetric Jacobi eigendecomposition for `n x n` symmetric matrix `a`.
/// Returns `(eigenvalues, eigenvectors)` where column `j` of `eigenvectors`
/// is the unit eigenvector for `eigenvalues[j]`.
fn jacobi_eigen_sym(a: &DenseMat, max_sweeps: usize) -> (Vec<f64>, DenseMat) {
    let n = a.rows;
    let mut d = a.clone();
    let mut v = DenseMat::zeros(n, n);
    for i in 0..n {
        v.set(i, i, 1.0);
    }

    for _ in 0..max_sweeps {
        let mut max_off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = d.get(p, q).abs();
                if apq > max_off {
                    max_off = apq;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = d.get(p, q);
                if apq.abs() < 1e-14 {
                    continue;
                }
                let app = d.get(p, p);
                let aqq = d.get(q, q);
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                d.set(p, p, app - t * apq);
                d.set(q, q, aqq + t * apq);
                d.set(p, q, 0.0);
                d.set(q, p, 0.0);

                for r in 0..n {
                    if r != p && r != q {
                        let drp = d.get(r, p);
                        let drq = d.get(r, q);
                        let nrp = c * drp - s * drq;
                        let nrq = s * drp + c * drq;
                        d.set(r, p, nrp);
                        d.set(p, r, nrp);
                        d.set(r, q, nrq);
                        d.set(q, r, nrq);
                    }
                    let vrp = v.get(r, p);
                    let vrq = v.get(r, q);
                    v.set(r, p, c * vrp - s * vrq);
                    v.set(r, q, s * vrp + c * vrq);
                }
            }
        }
    }

    let evals = (0..n).map(|i| d.get(i, i)).collect();
    (evals, v)
}

fn compute_metrics(f: &DenseMat) -> EtfMetrics {
    let n = f.rows;
    let m = f.cols;
    let wb = welch_bound(n, m);

    // Gram G = F * F^T [n, n]
    let ft = f.transpose();
    let gram = f.matmul(&ft);
    let mut max_coh = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let c = gram.get(i, j).abs();
            if c > max_coh {
                max_coh = c;
            }
        }
    }

    // Tightness error: ||F^T * F - (n/m) I_m||_F / sqrt(m)
    let frame_op = ft.matmul(f);
    let target_diag = (n as f64) / (m as f64);
    let mut sq_err = 0.0;
    for i in 0..m {
        for j in 0..m {
            let expected = if i == j { target_diag } else { 0.0 };
            let diff = frame_op.get(i, j) - expected;
            sq_err += diff * diff;
        }
    }
    let tightness_error = (sq_err / (m as f64)).sqrt();

    EtfMetrics {
        orig_dim: n,
        target_dim: m,
        welch_bound: wb,
        max_coherence: max_coh,
        tightness_error,
    }
}

/// Construct a deterministic ETF feature matrix `F` of shape `[orig_dim, target_dim]`.
pub fn build_etf(orig_dim: usize, cfg: &EtfConfig) -> Result<EtfFrame, ErrorCode> {
    let n = orig_dim;
    let m = cfg.target_dim;
    if m == 0 || m > n {
        return Err(ErrorCode::InvalidSuperpositionDim);
    }

    let mut f = DenseMat::zeros(n, m);

    if m == n {
        for i in 0..n {
            f.set(i, i, 1.0);
        }
    } else if n == m + 1 {
        // Exact closed-form Helmert Simplex ETF
        let scale = ((n as f64) / (m as f64)).sqrt();
        for k in 0..m {
            let kf = k as f64;
            let norm = ((kf + 1.0) * (kf + 2.0)).sqrt();
            let val = scale / norm;
            for r in 0..=k {
                f.set(r, k, val);
            }
            f.set(k + 1, k, -((kf + 1.0) * scale) / norm);
        }
    } else {
        // Deterministic DCT-II initialization (columns 1..=m, avoiding constant DC col 0)
        let pi = core::f64::consts::PI;
        for r in 0..n {
            for c in 0..m {
                let k = (c + 1) as f64;
                let theta = pi * k * ((r as f64) + 0.5) / (n as f64);
                f.set(r, c, theta.cos());
            }
        }
        normalize_rows(&mut f);

        let mu = welch_bound(n, m);
        let tight_scale = ((n as f64) / (m as f64)).sqrt();

        // Alternating-Projection Tight Frame (APTF)
        for _ in 0..cfg.max_iters {
            let ft = f.transpose();
            let mut gram = f.matmul(&ft);

            // 1. Project Gram matrix onto Welch-bound coherence ball
            let mut max_excess = 0.0f64;
            for i in 0..n {
                gram.set(i, i, 1.0);
                for j in (i + 1)..n {
                    let gij = gram.get(i, j);
                    let excess = gij.abs() - mu;
                    if excess > max_excess {
                        max_excess = excess;
                    }
                    let clamped = gij.clamp(-mu, mu);
                    gram.set(i, j, clamped);
                    gram.set(j, i, clamped);
                }
            }
            if max_excess <= cfg.tol {
                break;
            }

            // 2. Project onto rank-m tight frame via top-m eigenvectors of Gram
            let (evals, evecs) = jacobi_eigen_sym(&gram, 24);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));

            for r in 0..n {
                for c in 0..m {
                    let eig_idx = order[c];
                    f.set(r, c, evecs.get(r, eig_idx) * tight_scale);
                }
            }
            normalize_rows(&mut f);
        }
    }

    let metrics = compute_metrics(&f);
    Ok(EtfFrame {
        orig_dim: n,
        target_dim: m,
        features: f,
        metrics,
    })
}

/// Fold ETF feature matrix `F: [D, m]` into all weights of `CompiledTransformer`,
/// compressing `d_model` from `D` to `m`.
pub fn superpose_etf<const CAP: usize>(
    model: &CompiledTransformer,
    cfg: &EtfConfig,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<(CompiledTransformer, EtfMetrics), ErrorCode> {
    let etf = build_etf(model.d_model, cfg)?;
    apply_projection_matrix(model, &etf.features, etf.metrics, telemetry)
}

/// Fold a block-diagonal projection `F: [D, m]` where `protected[i] == true`
/// dimensions are kept orthogonal (identity) and all remaining dimensions are
/// packed into an ETF subspace of dimension `m - num_protected`.
pub fn superpose_etf_with_protected<const CAP: usize>(
    model: &CompiledTransformer,
    cfg: &EtfConfig,
    protected: &[bool],
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<(CompiledTransformer, EtfMetrics), ErrorCode> {
    let d = model.d_model;
    let m = cfg.target_dim;
    if protected.len() != d || m == 0 || m > d {
        return Err(ErrorCode::InvalidSuperpositionDim);
    }

    let prot_indices: Vec<usize> = (0..d).filter(|&i| protected[i]).collect();
    let free_indices: Vec<usize> = (0..d).filter(|&i| !protected[i]).collect();
    let num_prot = prot_indices.len();
    if m <= num_prot {
        return Err(ErrorCode::InvalidSuperpositionDim);
    }
    let free_m = m - num_prot;

    let mut f = DenseMat::zeros(d, m);
    for (col, &orig_r) in prot_indices.iter().enumerate() {
        f.set(orig_r, col, 1.0);
    }

    let metrics = if !free_indices.is_empty() {
        let sub_cfg = EtfConfig {
            target_dim: free_m,
            max_iters: cfg.max_iters,
            tol: cfg.tol,
        };
        let sub_etf = build_etf(free_indices.len(), &sub_cfg)?;
        for (sub_r, &orig_r) in free_indices.iter().enumerate() {
            for sub_c in 0..free_m {
                f.set(orig_r, num_prot + sub_c, sub_etf.features.get(sub_r, sub_c));
            }
        }
        EtfMetrics {
            orig_dim: d,
            target_dim: m,
            welch_bound: sub_etf.metrics.welch_bound,
            max_coherence: sub_etf.metrics.max_coherence,
            tightness_error: sub_etf.metrics.tightness_error,
        }
    } else {
        compute_metrics(&f)
    };

    apply_projection_matrix(model, &f, metrics, telemetry)
}

fn apply_projection_matrix<const CAP: usize>(
    model: &CompiledTransformer,
    f: &DenseMat,
    metrics: EtfMetrics,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<(CompiledTransformer, EtfMetrics), ErrorCode> {
    let d = model.d_model;
    let m = f.cols;

    telemetry.emit(
        Module::Etf,
        EventKind::EtfBuilt,
        Status::Completed,
        ErrorCode::None,
        0,
        d as u32,
        m as u32,
    );

    let ft = f.transpose(); // [m, d]

    // token_embed_sup = token_embed [vocab, d] * F [d, m] -> [vocab, m]
    let tok_mat = DenseMat {
        rows: model.vocab.len(),
        cols: d,
        data: model.token_embed.clone(),
    };
    let tok_sup = tok_mat.matmul(f);

    // pos_embed_sup = pos_embed [max_total_len, d] * F [d, m] -> [max_total_len, m]
    let pos_mat = DenseMat {
        rows: model.max_total_len,
        cols: d,
        data: model.pos_embed.clone(),
    };
    let pos_sup = pos_mat.matmul(f);

    // unembed_sup = F^T [m, d] * unembed [d, out_dim] -> [m, out_dim]
    let unembed_mat = DenseMat {
        rows: d,
        cols: model.out_dim,
        data: model.unembed.clone(),
    };
    let unembed_sup = ft.matmul(&unembed_mat);

    let hk = model.num_heads * model.key_size;
    let mlp_h = model.mlp_hidden_size;

    let mut layers_sup = Vec::with_capacity(model.layers.len());
    for layer in &model.layers {
        let w_q = DenseMat {
            rows: d,
            cols: hk,
            data: layer.w_q.clone(),
        };
        let w_k = DenseMat {
            rows: d,
            cols: hk,
            data: layer.w_k.clone(),
        };
        let w_v = DenseMat {
            rows: d,
            cols: hk,
            data: layer.w_v.clone(),
        };
        let w_o = DenseMat {
            rows: hk,
            cols: d,
            data: layer.w_o.clone(),
        };
        let w_1 = DenseMat {
            rows: d,
            cols: mlp_h,
            data: layer.w_1.clone(),
        };
        let w_2 = DenseMat {
            rows: mlp_h,
            cols: d,
            data: layer.w_2.clone(),
        };

        layers_sup.push(CompiledLayer {
            w_q: ft.matmul(&w_q).data,
            w_k: ft.matmul(&w_k).data,
            w_v: ft.matmul(&w_v).data,
            w_o: w_o.matmul(f).data,
            w_1: ft.matmul(&w_1).data,
            w_2: w_2.matmul(f).data,
        });
    }

    let sup_model = CompiledTransformer {
        num_layers: model.num_layers,
        num_heads: model.num_heads,
        key_size: model.key_size,
        mlp_hidden_size: model.mlp_hidden_size,
        d_model: m,
        out_dim: model.out_dim,
        max_total_len: model.max_total_len,
        causal: model.causal,
        categorical_output: model.categorical_output,
        bos_token_id: model.bos_token_id,
        bos_value: model.bos_value,
        pad_token_id: model.pad_token_id,
        vocab: model.vocab.clone(),
        output_vocab: model.output_vocab.clone(),
        token_embed: tok_sup.data,
        pos_embed: pos_sup.data,
        unembed: unembed_sup.data,
        numerical_mask: vec![false; m],
        layers: layers_sup,
    };

    telemetry.emit(
        Module::Etf,
        EventKind::ModelSuperposed,
        Status::Completed,
        ErrorCode::None,
        0,
        d as u32,
        m as u32,
    );

    Ok((sup_model, metrics))
}
