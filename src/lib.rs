//! `tracr-rs`: Pure-Rust RASP-to-Transformer weight-space compiler.
//!
//! Designed in strict compliance with `D:\poirot\CLAUDE.md`:
//! - Compact integer IDs and `#[repr(u8)]` enums in all core paths.
//! - Contiguous flat `f64` weight/activation buffers and CSR graph layout.
//! - Preallocated `ForwardScratch` for zero-allocation Transformer inference.
//! - Structured `TelemetryEvent` ring buffer with presentation isolated in `observatory`.

pub mod compiler;
pub mod craft;
pub mod etf;
pub mod graph;
pub mod observatory;
pub mod rasp;
pub mod sys;
pub mod transformer;

#[cfg(test)]
mod tests {
    use crate::compiler::{compile_rasp_to_model, static_validate, CompileConfig};
    use crate::observatory;
    use crate::rasp::{evaluate_rasp, Comparison, EvalScratch, RaspProgram, Value};
    use crate::sys::{
        ArrayVec, ErrorCode, QueueFullPolicy, RingBuffer, SymId, TelemetryBus, ValidationResult,
    };
    use crate::transformer::ForwardScratch;

    fn sym(c: char) -> Value {
        Value::Sym(SymId(c as u32))
    }

    #[test]
    fn ring_buffer_and_array_vec_invariants() {
        let mut rb: RingBuffer<u32, 4> = RingBuffer::new(QueueFullPolicy::Reject);
        assert_eq!(rb.push(10), ErrorCode::None);
        assert_eq!(rb.push(20), ErrorCode::None);
        assert_eq!(rb.push(30), ErrorCode::None);
        assert_eq!(rb.push(40), ErrorCode::None);
        assert_eq!(rb.push(50), ErrorCode::QueueFull);
        assert_eq!(rb.pop(), Some(10));
        assert_eq!(rb.push(50), ErrorCode::None);
        assert_eq!(rb.pop(), Some(20));
        assert_eq!(rb.pop(), Some(30));
        assert_eq!(rb.pop(), Some(40));
        assert_eq!(rb.pop(), Some(50));
        assert_eq!(rb.pop(), None);

        let mut rb_ov: RingBuffer<u32, 4> = RingBuffer::new(QueueFullPolicy::OverwriteOldest);
        for i in 1..=6 {
            assert_eq!(rb_ov.push(i), ErrorCode::None);
        }
        assert_eq!(rb_ov.pop(), Some(3));
        assert_eq!(rb_ov.pop(), Some(4));
        assert_eq!(rb_ov.pop(), Some(5));
        assert_eq!(rb_ov.pop(), Some(6));

        let mut av: ArrayVec<u16, 3> = ArrayVec::new();
        assert_eq!(av.push(1), ErrorCode::None);
        assert_eq!(av.push(2), ErrorCode::None);
        assert_eq!(av.push(3), ErrorCode::None);
        assert_eq!(av.push(4), ErrorCode::CapacityExceeded);
        assert_eq!(av.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn compile_and_run_hist() {
        // hist(x): count occurrences of each token in the sequence
        let mut prog = RaspProgram::new();
        let tokens = prog.tokens();
        let same_tok = prog.select(tokens, tokens, Comparison::Eq);
        let hist = prog.selector_width(same_tok);

        let vocab = [sym('a'), sym('b'), sym('c')];
        let cfg = CompileConfig {
            max_seq_len: 6,
            causal: false,
            ..Default::default()
        };
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let model = compile_rasp_to_model(&prog, hist, &vocab, &cfg, &mut telemetry)
            .expect("hist compilation should succeed");

        let mut scratch = ForwardScratch::for_model(&model);
        let mut eval_scratch = EvalScratch::new(prog.len(), 6);

        let input = [sym('a'), sym('b'), sym('a'), sym('c'), sym('a')];
        let expected = evaluate_rasp(&prog, hist, &input, false, &mut eval_scratch).unwrap();
        let actual = model
            .forward_values(&input, &mut scratch, &mut telemetry)
            .unwrap();

        // actual[0] is BOS; actual[1..] matches RASP output
        assert_eq!(actual[0], cfg.bos_value);
        assert_eq!(&actual[1..], expected);

        // Verify telemetry was emitted and Observatory can render it
        assert!(!telemetry.is_empty());
        while let Some(ev) = telemetry.pop() {
            let rendered = observatory::format_event(&ev);
            assert!(!rendered.is_empty());
        }
    }

    #[test]
    fn compile_and_run_reverse() {
        // reverse(x):
        //   all_true = Select(tokens, tokens, |_, _| true)
        //   length = SelectorWidth(all_true)
        //   opp_idx = SequenceMap(|len, idx| len - idx - 1, length, indices)
        //   sel = Select(indices, opp_idx, ==)
        //   out = Aggregate(sel, tokens, None)
        let mut prog = RaspProgram::new();
        let tokens = prog.tokens();
        let indices = prog.indices();
        let all_true = prog.select_fn(tokens, tokens, |_, _| true);
        let length = prog.selector_width(all_true);
        let opp_idx = prog.sequence_map(
            |len, idx| Some(Value::Int(len.as_int()? - idx.as_int()? - 1)),
            length,
            indices,
        );
        let sel = prog.select(indices, opp_idx, Comparison::Eq);
        let rev = prog.aggregate(sel, tokens, Value::None);

        let vocab = [sym('a'), sym('b'), sym('c'), sym('d')];
        let cfg = CompileConfig {
            max_seq_len: 5,
            causal: false,
            ..Default::default()
        };
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let model = compile_rasp_to_model(&prog, rev, &vocab, &cfg, &mut telemetry)
            .expect("reverse compilation should succeed");

        let mut scratch = ForwardScratch::for_model(&model);
        let mut eval_scratch = EvalScratch::new(prog.len(), 5);

        let input = [sym('a'), sym('b'), sym('c'), sym('d')];
        let expected = evaluate_rasp(&prog, rev, &input, false, &mut eval_scratch).unwrap();
        let actual = model
            .forward_values(&input, &mut scratch, &mut telemetry)
            .unwrap();

        assert_eq!(expected, &[sym('d'), sym('c'), sym('b'), sym('a')]);
        assert_eq!(&actual[1..], expected);
    }

    #[test]
    fn compile_and_run_causal_frac_prevs() {
        // frac_prevs(x == 'a'): fraction of previous (causal) tokens equal to 'a'
        let mut prog = RaspProgram::new();
        let tokens = prog.tokens();
        let is_a = prog.map(
            |v| Some(Value::Int(i32::from(v == Value::Sym(SymId('a' as u32))))),
            tokens,
        );
        let is_a_num = prog.numerical(is_a);
        let all_true = prog.select_fn(tokens, tokens, |_, _| true);
        let frac = prog.aggregate(all_true, is_a_num, Value::Int(0));

        let vocab = [sym('a'), sym('b')];
        let cfg = CompileConfig {
            max_seq_len: 5,
            causal: true,
            ..Default::default()
        };
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let model = compile_rasp_to_model(&prog, frac, &vocab, &cfg, &mut telemetry)
            .expect("frac_prevs compilation should succeed");

        let mut scratch = ForwardScratch::for_model(&model);
        let mut eval_scratch = EvalScratch::new(prog.len(), 5);

        let input = [sym('a'), sym('b'), sym('a'), sym('b')];
        let expected = evaluate_rasp(&prog, frac, &input, true, &mut eval_scratch).unwrap();
        let actual = model
            .forward_values(&input, &mut scratch, &mut telemetry)
            .unwrap();

        assert_eq!(actual.len(), input.len() + 1);
        for (i, (&exp, &act)) in expected.iter().zip(actual[1..].iter()).enumerate() {
            let ef = exp.as_f64().unwrap();
            let af = act.as_f64().unwrap();
            assert!(
                (ef - af).abs() < 1e-3,
                "pos {}: expected {}, got {}",
                i,
                ef,
                af
            );
        }
    }

    #[test]
    fn compile_and_run_linear_sequence_map() {
        // Computes 2.0 * is_a + 3.0 * is_b
        let mut prog = RaspProgram::new();
        let tokens = prog.tokens();
        let is_a = prog.map(
            |v| Some(Value::Int(i32::from(v == Value::Sym(SymId('a' as u32))))),
            tokens,
        );
        let is_a_num = prog.numerical(is_a);
        let is_b = prog.map(
            |v| Some(Value::Int(i32::from(v == Value::Sym(SymId('b' as u32))))),
            tokens,
        );
        let is_b_num = prog.numerical(is_b);
        let lin = prog.linear_sequence_map(is_a_num, is_b_num, 2.0, 3.0);

        let vocab = [sym('a'), sym('b'), sym('c')];
        let cfg = CompileConfig {
            max_seq_len: 4,
            causal: false,
            ..Default::default()
        };
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let model = compile_rasp_to_model(&prog, lin, &vocab, &cfg, &mut telemetry)
            .expect("linear_sequence_map compilation should succeed");

        let mut scratch = ForwardScratch::for_model(&model);
        let input = [sym('a'), sym('b'), sym('c')];
        let actual = model
            .forward_values(&input, &mut scratch, &mut telemetry)
            .unwrap();

        assert!((actual[1].as_f64().unwrap() - 2.0).abs() < 1e-5);
        assert!((actual[2].as_f64().unwrap() - 3.0).abs() < 1e-5);
        assert!((actual[3].as_f64().unwrap() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn static_validation_rejects_invalid_programs() {
        // 1. Numerical tokens
        let mut prog1 = RaspProgram::new();
        let tok1 = prog1.tokens();
        prog1.numerical(tok1);
        let res1 = static_validate(&prog1, tok1);
        assert_eq!(res1.result, ValidationResult::Invalid);
        assert_eq!(res1.code, ErrorCode::UnsupportedEncoding);

        // 2. Composite selector
        let mut prog2 = RaspProgram::new();
        let tok2 = prog2.tokens();
        let s1 = prog2.select(tok2, tok2, Comparison::Eq);
        let s2 = prog2.select(tok2, tok2, Comparison::Neq);
        let sand = prog2.selector_and(s1, s2);
        let agg = prog2.aggregate(sand, tok2, Value::None);
        let res2 = static_validate(&prog2, agg);
        assert_eq!(res2.result, ValidationResult::Invalid);
        assert_eq!(res2.code, ErrorCode::UnsupportedSelector);
    }

    #[test]
    fn etf_frame_properties_simplex_and_general() {
        use crate::etf::{build_etf, welch_bound, EtfConfig};

        // 1. Simplex ETF: n = m + 1 -> exact Welch bound 1/m
        let simplex = build_etf(16, &EtfConfig::new(15)).unwrap();
        assert!((simplex.metrics.max_coherence - (1.0 / 15.0)).abs() < 1e-10);
        assert!(simplex.metrics.tightness_error < 1e-10);

        // 2. General APTF: n = 32, m = 24
        let general = build_etf(32, &EtfConfig::new(24)).unwrap();
        let wb = welch_bound(32, 24);
        assert!(
            general.metrics.max_coherence < wb + 0.08,
            "max_coherence={} too far from welch_bound={}",
            general.metrics.max_coherence,
            wb
        );

        // 3. Invalid dimensions
        assert_eq!(
            build_etf(8, &EtfConfig::new(0)).unwrap_err(),
            ErrorCode::InvalidSuperpositionDim
        );
        assert_eq!(
            build_etf(8, &EtfConfig::new(9)).unwrap_err(),
            ErrorCode::InvalidSuperpositionDim
        );
    }

    #[test]
    fn superposed_etf_shift_right_and_protected_hist_preserve_outputs() {
        use crate::etf::{superpose_etf, EtfConfig};

        // Test 1: Pure global ETF superposition on categorical shift-right (Select + Aggregate + Map)
        {
            let mut prog = RaspProgram::new();
            let tokens = prog.tokens();
            let indices = prog.indices();
            let prev_idx = prog.map(|v| Some(Value::Int(v.as_int()? - 1)), indices);
            let sel = prog.select(indices, prev_idx, Comparison::Eq);
            let shifted = prog.aggregate(sel, tokens, Value::None);

            let vocab = [sym('a'), sym('b'), sym('c'), sym('d')];
            let cfg = CompileConfig {
                max_seq_len: 5,
                causal: false,
                ..Default::default()
            };
            let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
            let model =
                compile_rasp_to_model(&prog, shifted, &vocab, &cfg, &mut telemetry).unwrap();

            let target_dim = model.d_model - 2;
            let (sup_model, metrics) =
                superpose_etf(&model, &EtfConfig::new(target_dim), &mut telemetry).unwrap();
            assert_eq!(sup_model.d_model, target_dim);
            assert!(metrics.max_coherence < 0.25);

            let mut scratch = ForwardScratch::for_model(&sup_model);
            let mut eval_scratch = EvalScratch::new(prog.len(), 5);
            let input = [sym('a'), sym('b'), sym('c'), sym('d')];
            let expected =
                evaluate_rasp(&prog, shifted, &input, false, &mut eval_scratch).unwrap();
            let actual = sup_model
                .forward_values(&input, &mut scratch, &mut telemetry)
                .unwrap();

            // Position 1 has no prev token so aggregate outputs default (BOS); positions 2.. match shifted
            assert_eq!(&actual[2..], &expected[1..]);
        }

        // Test 2: Pure global ETF superposition on multi-layer Map + SequenceMap + Aggregate
        {
            let mut prog = RaspProgram::new();
            let tokens = prog.tokens();
            let indices = prog.indices();
            // Swap adjacent pairs: even index i -> i + 1, odd index i -> i - 1
            let partner_idx = prog.map(
                |v| {
                    let i = v.as_int()?;
                    Some(Value::Int(if i % 2 == 0 { i + 1 } else { i - 1 }))
                },
                indices,
            );
            let sel = prog.select(indices, partner_idx, Comparison::Eq);
            let swapped = prog.aggregate(sel, tokens, Value::None);
            // Combine original token and swapped partner token via SequenceMap
            let paired = prog.sequence_map(
                |a, b| {
                    let ca = match a {
                        Value::Sym(s) => s.0,
                        _ => return None,
                    };
                    let cb = match b {
                        Value::Sym(s) => s.0,
                        _ => return None,
                    };
                    Some(Value::Int((ca == cb) as i32))
                },
                tokens,
                swapped,
            );

            let vocab = [sym('a'), sym('b'), sym('c')];
            let cfg = CompileConfig {
                max_seq_len: 4,
                causal: false,
                ..Default::default()
            };
            let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
            let model =
                compile_rasp_to_model(&prog, paired, &vocab, &cfg, &mut telemetry).unwrap();

            let target_dim = model.d_model - 3;
            assert!(target_dim < model.d_model);
            let (sup_model, metrics) =
                superpose_etf(&model, &EtfConfig::new(target_dim), &mut telemetry).unwrap();
            assert_eq!(sup_model.d_model, target_dim);
            assert!(metrics.max_coherence < 0.25);

            let mut scratch = ForwardScratch::for_model(&sup_model);
            let mut eval_scratch = EvalScratch::new(prog.len(), 4);
            let input = [sym('a'), sym('a'), sym('b'), sym('c')];
            let expected =
                evaluate_rasp(&prog, paired, &input, false, &mut eval_scratch).unwrap();
            let actual = sup_model
                .forward_values(&input, &mut scratch, &mut telemetry)
                .unwrap();

            assert_eq!(expected, &[Value::Int(1), Value::Int(1), Value::Int(0), Value::Int(0)]);
            assert_eq!(&actual[1..], expected);
        }
    }
}
