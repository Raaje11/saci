//! Compiled Transformer weights and zero-allocation deterministic forward pass.
//!
//! All weight matrices are stored in contiguous flat `f64` buffers.
//! `ForwardScratch` is allocated once during initialization and reused across
//! all `forward_*` calls with zero runtime heap allocation.

use crate::rasp::Value;
use crate::sys::{ErrorCode, EventKind, Module, Status, TelemetryBus};

#[derive(Clone, Debug)]
pub struct CompiledLayer {
    /// `[d_model * (num_heads * key_size)]`
    pub w_q: Vec<f64>,
    /// `[d_model * (num_heads * key_size)]`
    pub w_k: Vec<f64>,
    /// `[d_model * (num_heads * key_size)]`
    pub w_v: Vec<f64>,
    /// `[(num_heads * key_size) * d_model]`
    pub w_o: Vec<f64>,
    /// `[d_model * mlp_hidden_size]`
    pub w_1: Vec<f64>,
    /// `[mlp_hidden_size * d_model]`
    pub w_2: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CompiledTransformer {
    pub num_layers: usize,
    pub num_heads: usize,
    pub key_size: usize,
    pub mlp_hidden_size: usize,
    pub d_model: usize,
    pub out_dim: usize,
    /// Maximum sequence length INCLUDING BOS token.
    pub max_total_len: usize,
    pub causal: bool,
    pub categorical_output: bool,
    pub bos_token_id: Option<u32>,
    pub bos_value: Value,
    pub pad_token_id: Option<u32>,
    /// Maps TokenId -> Value
    pub vocab: Vec<Value>,
    /// Maps categorical output index -> Value (empty if numerical)
    pub output_vocab: Vec<Value>,
    /// `[vocab.len() * d_model]`
    pub token_embed: Vec<f64>,
    /// `[max_total_len * d_model]`
    pub pos_embed: Vec<f64>,
    /// `[d_model * out_dim]`
    pub unembed: Vec<f64>,
    /// Per-residual-dim flag: true for `one` and 1D numerical/scalar dims
    pub numerical_mask: Vec<bool>,
    pub layers: Vec<CompiledLayer>,
}

/// Preallocated scratchpad for zero-allocation Transformer inference.
pub struct ForwardScratch {
    pub max_total_len: usize,
    pub d_model: usize,
    pub hk_dim: usize,
    pub mlp_hidden_size: usize,
    pub out_dim: usize,
    pub encoded_tokens: Vec<u32>,
    pub residual: Vec<f64>,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub attn_logits: Vec<f64>,
    pub attn_weights: Vec<f64>,
    pub attn_concat: Vec<f64>,
    pub attn_out: Vec<f64>,
    pub mlp_hidden: Vec<f64>,
    pub mlp_out: Vec<f64>,
    pub unembed_logits: Vec<f64>,
    pub decoded_out: Vec<Value>,
}

impl ForwardScratch {
    pub fn for_model(model: &CompiledTransformer) -> Self {
        let t = model.max_total_len;
        let d = model.d_model;
        let hk = model.num_heads * model.key_size;
        let m = model.mlp_hidden_size;
        let o = model.out_dim;
        let h = model.num_heads;
        Self {
            max_total_len: t,
            d_model: d,
            hk_dim: hk,
            mlp_hidden_size: m,
            out_dim: o,
            encoded_tokens: vec![0; t],
            residual: vec![0.0; t * d],
            q: vec![0.0; t * hk],
            k: vec![0.0; t * hk],
            v: vec![0.0; t * hk],
            attn_logits: vec![0.0; h * t * t],
            attn_weights: vec![0.0; h * t * t],
            attn_concat: vec![0.0; t * hk],
            attn_out: vec![0.0; t * d],
            mlp_hidden: vec![0.0; t * m],
            mlp_out: vec![0.0; t * d],
            unembed_logits: vec![0.0; t * o],
            decoded_out: vec![Value::None; t],
        }
    }
}

impl CompiledTransformer {
    #[inline]
    pub fn lookup_token_id(&self, val: Value) -> Option<u32> {
        self.vocab
            .iter()
            .position(|&v| v == val)
            .map(|idx| idx as u32)
    }

    /// Zero-allocation forward pass on raw `Value` sequence (BOS is prepended automatically
    /// if configured). Returns slice of length `input.len() + 1` (including BOS at index 0)
    /// or `input.len()` if no BOS.
    pub fn forward_values<'a, const CAP: usize>(
        &self,
        input: &[Value],
        scratch: &'a mut ForwardScratch,
        telemetry: &mut TelemetryBus<CAP>,
    ) -> Result<&'a [Value], ErrorCode> {
        if input.is_empty() {
            return Err(ErrorCode::EmptyInput);
        }
        let has_bos = self.bos_token_id.is_some();
        let total_len = input.len() + usize::from(has_bos);
        if total_len > self.max_total_len || total_len > scratch.max_total_len {
            return Err(ErrorCode::SeqLenExceeded);
        }

        let mut cursor = 0;
        if let Some(bos_id) = self.bos_token_id {
            scratch.encoded_tokens[0] = bos_id;
            cursor = 1;
        }
        for &val in input {
            let tid = self.lookup_token_id(val).ok_or(ErrorCode::InvalidExpr)?;
            scratch.encoded_tokens[cursor] = tid;
            cursor += 1;
        }

        self.forward_encoded(total_len, scratch, telemetry)
    }

    /// Zero-allocation forward pass on pre-encoded token IDs in `scratch.encoded_tokens[..seq_len]`.
    pub fn forward_encoded<'a, const CAP: usize>(
        &self,
        seq_len: usize,
        scratch: &'a mut ForwardScratch,
        telemetry: &mut TelemetryBus<CAP>,
    ) -> Result<&'a [Value], ErrorCode> {
        if seq_len == 0 {
            return Err(ErrorCode::EmptyInput);
        }
        if seq_len > self.max_total_len || seq_len > scratch.max_total_len {
            return Err(ErrorCode::SeqLenExceeded);
        }

        telemetry.emit(
            Module::Transformer,
            EventKind::ForwardStarted,
            Status::Running,
            ErrorCode::None,
            0,
            0,
            seq_len as u32,
        );

        let d = self.d_model;
        let h = self.num_heads;
        let k_sz = self.key_size;
        let hk = h * k_sz;
        let mlp_h = self.mlp_hidden_size;
        let inv_sqrt_k = 1.0 / (k_sz as f64).sqrt();

        // 1. Embeddings: token_embed[tok] + pos_embed[pos]
        for t in 0..seq_len {
            let tok = scratch.encoded_tokens[t] as usize;
            let tok_off = tok * d;
            let pos_off = t * d;
            let res_off = t * d;
            for i in 0..d {
                scratch.residual[res_off + i] =
                    self.token_embed[tok_off + i] + self.pos_embed[pos_off + i];
            }
        }

        // 2. Transformer layers
        for layer in &self.layers {
            // Q, K, V projections: [seq_len, d] @ [d, hk] -> [seq_len, hk]
            matmul_zero_init(&scratch.residual[..seq_len * d], &layer.w_q, &mut scratch.q[..seq_len * hk], seq_len, d, hk);
            matmul_zero_init(&scratch.residual[..seq_len * d], &layer.w_k, &mut scratch.k[..seq_len * hk], seq_len, d, hk);
            matmul_zero_init(&scratch.residual[..seq_len * d], &layer.w_v, &mut scratch.v[..seq_len * hk], seq_len, d, hk);

            // Attention per head
            for head in 0..h {
                let head_off = head * k_sz;
                let logit_base = head * seq_len * seq_len;

                for qt in 0..seq_len {
                    let q_row = qt * hk + head_off;
                    let row_base = logit_base + qt * seq_len;

                    // Compute logits & find max for numerically stable softmax
                    let mut max_logit = f64::NEG_INFINITY;
                    for kt in 0..seq_len {
                        let is_pad = self
                            .pad_token_id
                            .map_or(false, |pid| scratch.encoded_tokens[kt] == pid);
                        let masked_out = is_pad || (self.causal && kt > qt);
                        let logit = if masked_out {
                            -1e30
                        } else {
                            let k_row = kt * hk + head_off;
                            let mut dot = 0.0;
                            for i in 0..k_sz {
                                dot += scratch.q[q_row + i] * scratch.k[k_row + i];
                            }
                            dot * inv_sqrt_k
                        };
                        scratch.attn_logits[row_base + kt] = logit;
                        if logit > max_logit {
                            max_logit = logit;
                        }
                    }

                    // Softmax
                    let mut sum_exp = 0.0;
                    for kt in 0..seq_len {
                        let w = (scratch.attn_logits[row_base + kt] - max_logit).exp();
                        scratch.attn_weights[row_base + kt] = w;
                        sum_exp += w;
                    }
                    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                    for kt in 0..seq_len {
                        scratch.attn_weights[row_base + kt] *= inv_sum;
                    }

                    // Weighted sum of V -> attn_concat[qt, head_off..head_off+k_sz]
                    let out_row = qt * hk + head_off;
                    for i in 0..k_sz {
                        scratch.attn_concat[out_row + i] = 0.0;
                    }
                    for kt in 0..seq_len {
                        let w = scratch.attn_weights[row_base + kt];
                        if w == 0.0 {
                            continue;
                        }
                        let v_row = kt * hk + head_off;
                        for i in 0..k_sz {
                            scratch.attn_concat[out_row + i] += w * scratch.v[v_row + i];
                        }
                    }
                }
            }

            // Output projection: attn_concat [seq_len, hk] @ w_o [hk, d] -> attn_out [seq_len, d]
            matmul_zero_init(
                &scratch.attn_concat[..seq_len * hk],
                &layer.w_o,
                &mut scratch.attn_out[..seq_len * d],
                seq_len,
                hk,
                d,
            );

            // Residual add
            for i in 0..(seq_len * d) {
                scratch.residual[i] += scratch.attn_out[i];
            }

            // MLP layer 1 + ReLU: residual [seq_len, d] @ w_1 [d, mlp_h] -> mlp_hidden [seq_len, mlp_h]
            matmul_zero_init(
                &scratch.residual[..seq_len * d],
                &layer.w_1,
                &mut scratch.mlp_hidden[..seq_len * mlp_h],
                seq_len,
                d,
                mlp_h,
            );
            for v in &mut scratch.mlp_hidden[..seq_len * mlp_h] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }

            // MLP layer 2: mlp_hidden [seq_len, mlp_h] @ w_2 [mlp_h, d] -> mlp_out [seq_len, d]
            matmul_zero_init(
                &scratch.mlp_hidden[..seq_len * mlp_h],
                &layer.w_2,
                &mut scratch.mlp_out[..seq_len * d],
                seq_len,
                mlp_h,
                d,
            );

            // Residual add
            for i in 0..(seq_len * d) {
                scratch.residual[i] += scratch.mlp_out[i];
            }
        }

        // 3. Unembed: residual [seq_len, d] @ unembed [d, out_dim] -> unembed_logits [seq_len, out_dim]
        let out_dim = self.out_dim;
        matmul_zero_init(
            &scratch.residual[..seq_len * d],
            &self.unembed,
            &mut scratch.unembed_logits[..seq_len * out_dim],
            seq_len,
            d,
            out_dim,
        );

        // 4. Decode
        for t in 0..seq_len {
            if t == 0 && self.bos_token_id.is_some() {
                scratch.decoded_out[0] = self.bos_value;
                continue;
            }
            if self.categorical_output {
                let row = &scratch.unembed_logits[t * out_dim..(t + 1) * out_dim];
                let mut best_idx = 0usize;
                let mut best_val = f64::NEG_INFINITY;
                for (i, &val) in row.iter().enumerate() {
                    if val > best_val {
                        best_val = val;
                        best_idx = i;
                    }
                }
                scratch.decoded_out[t] = self.output_vocab[best_idx];
            } else {
                scratch.decoded_out[t] = Value::from_f64(scratch.unembed_logits[t * out_dim]);
            }
        }

        telemetry.emit(
            Module::Transformer,
            EventKind::ForwardCompleted,
            Status::Completed,
            ErrorCode::None,
            0,
            0,
            seq_len as u32,
        );

        Ok(&scratch.decoded_out[..seq_len])
    }
}

#[inline]
fn matmul_zero_init(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    c[..m * n].fill(0.0);
    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;
        for p in 0..k {
            let av = a[a_row + p];
            if av == 0.0 {
                continue;
            }
            let b_row = p * n;
            for j in 0..n {
                c[c_row + j] += av * b[b_row + j];
            }
        }
    }
}
