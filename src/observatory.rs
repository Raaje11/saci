//! Observatory: Presentation Layer Boundary (Sections 11, 12, 35).
//!
//! All human-readable string mappings and formatting live exclusively here.
//! Core runtime, compiler, and transformer modules never construct or format
//! human-readable strings.

use crate::sys::{ErrorCode, EventKind, Module, Status, TelemetryEvent};

pub fn module_name(m: Module) -> &'static str {
    match m {
        Module::Rasp => "RASP",
        Module::Validator => "Validator",
        Module::Graph => "Graph",
        Module::BasisInference => "BasisInference",
        Module::Craft => "Craft",
        Module::Compiler => "Compiler",
        Module::Assembler => "Assembler",
        Module::Transformer => "Transformer",
        Module::Harness => "Harness",
        Module::Etf => "ETF",
    }
}

pub fn status_name(s: Status) -> &'static str {
    match s {
        Status::Idle => "Idle",
        Status::Ready => "Ready",
        Status::Running => "Running",
        Status::Waiting => "Waiting",
        Status::Completed => "Completed",
        Status::Failed => "Failed",
    }
}

pub fn error_description(e: ErrorCode) -> &'static str {
    match e {
        ErrorCode::None => "No error",
        ErrorCode::QueueFull => "Ring buffer exhausted",
        ErrorCode::CapacityExceeded => "Fixed buffer capacity exceeded",
        ErrorCode::InvalidExpr => "Invalid RASP expression",
        ErrorCode::InvalidGraph => "Invalid computational graph",
        ErrorCode::InvalidBasis => "Invalid vector space basis",
        ErrorCode::DimensionMismatch => "Vector space dimension mismatch",
        ErrorCode::UnsupportedEncoding => "Unsupported S-Op encoding",
        ErrorCode::UnsupportedAggregate => "Unsupported Aggregate configuration",
        ErrorCode::UnsupportedSequenceMap => "Unsupported SequenceMap encoding",
        ErrorCode::UnsupportedSelector => "Composite selector compilation not supported",
        ErrorCode::MissingTokensNode => "Program must include rasp.tokens",
        ErrorCode::BosOrPadInVocab => "Compiler BOS or PAD token present in vocab",
        ErrorCode::EmptyInput => "Input token sequence is empty",
        ErrorCode::SeqLenExceeded => "Input sequence exceeds max_seq_len",
        ErrorCode::ArithmeticError => "Arithmetic error during evaluation",
        ErrorCode::NonBinaryNumericalAggregate => {
            "Numerical aggregate currently only supports binary values"
        }
        ErrorCode::OutputValueMismatch => "Output value missing from output basis",
        ErrorCode::InvalidSuperpositionDim => "Invalid ETF superposition dimension",
    }
}

pub fn event_description(ev: EventKind) -> &'static str {
    match ev {
        EventKind::CompileStarted => "Compilation started",
        EventKind::ValidationPassed => "Static validation passed",
        EventKind::ValidationFailed => "Static validation failed",
        EventKind::GraphExtracted => "RASP CSR graph extracted",
        EventKind::BasesInferred => "Basis inference completed",
        EventKind::CraftBlocksBuilt => "Craft blocks constructed",
        EventKind::ModelAssembled => "Transformer weights assembled",
        EventKind::ForwardStarted => "Transformer forward pass started",
        EventKind::ForwardCompleted => "Transformer forward pass completed",
        EventKind::QueueFull => "Telemetry or task ring buffer full",
        EventKind::ResourcePressure => "Resource pressure detected",
        EventKind::EtfBuilt => "Equiangular Tight Frame constructed",
        EventKind::ModelSuperposed => "Transformer weights compressed via ETF superposition",
    }
}

pub fn format_event(ev: &TelemetryEvent) -> String {
    format!(
        "[tick={}] [{}] {} ({}) code={} flags=0x{:04x} id={} val={}",
        ev.tick,
        module_name(ev.module),
        event_description(ev.event),
        status_name(ev.status),
        error_description(ev.code),
        ev.flags,
        ev.id,
        ev.value,
    )
}
