//! Core systems primitives: bounded buffers, ring buffers, compact IDs,
//! enum-driven state/errors, bitflags, and structured telemetry.
//!
//! Strictly zero heap allocation, zero string formatting, and deterministic O(1)
//! operations on hot paths.

use core::mem::MaybeUninit;

// ---------------------------------------------------------------------------
// Strongly typed compact IDs (Section 24)
// ---------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DirId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpaceId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenId(pub u32);

// ---------------------------------------------------------------------------
// Enum-Only Runtime Semantics (Sections 7, 8, 9, 10, 13)
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Module {
    Rasp = 0,
    Validator = 1,
    Graph = 2,
    BasisInference = 3,
    Craft = 4,
    Compiler = 5,
    Assembler = 6,
    Transformer = 7,
    Harness = 8,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Idle = 0,
    Ready = 1,
    Running = 2,
    Waiting = 3,
    Completed = 4,
    Failed = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    None = 0,
    QueueFull = 1,
    CapacityExceeded = 2,
    InvalidExpr = 3,
    InvalidGraph = 4,
    InvalidBasis = 5,
    DimensionMismatch = 6,
    UnsupportedEncoding = 7,
    UnsupportedAggregate = 8,
    UnsupportedSequenceMap = 9,
    UnsupportedSelector = 10,
    MissingTokensNode = 11,
    BosOrPadInVocab = 12,
    EmptyInput = 13,
    SeqLenExceeded = 14,
    ArithmeticError = 15,
    NonBinaryNumericalAggregate = 16,
    OutputValueMismatch = 17,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    CompileStarted = 0,
    ValidationPassed = 1,
    ValidationFailed = 2,
    GraphExtracted = 3,
    BasesInferred = 4,
    CraftBlocksBuilt = 5,
    ModelAssembled = 6,
    ForwardStarted = 7,
    ForwardCompleted = 8,
    QueueFull = 9,
    ResourcePressure = 10,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationResult {
    Valid = 0,
    Invalid = 1,
    Deferred = 2,
}

// ---------------------------------------------------------------------------
// Bitflags (Sections 7, 20)
// ---------------------------------------------------------------------------

pub const FLAG_CATEGORICAL: u16 = 1 << 0;
pub const FLAG_NUMERICAL: u16 = 1 << 1;
pub const FLAG_CAUSAL: u16 = 1 << 2;
pub const FLAG_HAS_BOS: u16 = 1 << 3;
pub const FLAG_HAS_ONE: u16 = 1 << 4;
pub const FLAG_VALIDATED: u16 = 1 << 5;
pub const FLAG_FAILED: u16 = 1 << 6;
pub const FLAG_ATTN_BLOCK: u16 = 1 << 7;
pub const FLAG_MLP_BLOCK: u16 = 1 << 8;
pub const FLAG_RESIDUAL_BLOCK: u16 = 1 << 9;
pub const FLAG_DISCONNECTED: u16 = 1 << 10;

// ---------------------------------------------------------------------------
// Structured Telemetry (Sections 10, 34)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TelemetryEvent {
    pub tick: u64,
    pub module: Module,
    pub event: EventKind,
    pub status: Status,
    pub code: ErrorCode,
    pub flags: u16,
    pub id: u32,
    pub value: u32,
}

// ---------------------------------------------------------------------------
// Fixed-Capacity Ring Buffer (Section 6)
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueFullPolicy {
    Reject = 0,
    OverwriteOldest = 1,
}

/// Fixed-capacity circular ring buffer with power-of-two masking.
/// Zero heap allocation; all operations are O(1).
pub struct RingBuffer<T: Copy, const CAP: usize> {
    buf: [MaybeUninit<T>; CAP],
    head: usize,
    tail: usize,
    len: usize,
    policy: QueueFullPolicy,
}

impl<T: Copy, const CAP: usize> RingBuffer<T, CAP> {
    const _ASSERT_POW2: () = assert!(CAP > 0 && (CAP & (CAP - 1)) == 0, "CAP must be a power of 2");
    const MASK: usize = CAP - 1;

    pub const fn new(policy: QueueFullPolicy) -> Self {
        let () = Self::_ASSERT_POW2;
        Self {
            // SAFETY: MaybeUninit<T> does not require initialization.
            buf: [const { MaybeUninit::uninit() }; CAP],
            head: 0,
            tail: 0,
            len: 0,
            policy,
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len == CAP
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        CAP
    }

    #[inline]
    pub fn push(&mut self, item: T) -> ErrorCode {
        if self.len == CAP {
            match self.policy {
                QueueFullPolicy::Reject => return ErrorCode::QueueFull,
                QueueFullPolicy::OverwriteOldest => {
                    self.head = (self.head + 1) & Self::MASK;
                    self.len -= 1;
                }
            }
        }
        self.buf[self.tail].write(item);
        self.tail = (self.tail + 1) & Self::MASK;
        self.len += 1;
        ErrorCode::None
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        // SAFETY: `head` points to an initialized slot because `len > 0`.
        let val = unsafe { self.buf[self.head].assume_init_read() };
        self.head = (self.head + 1) & Self::MASK;
        self.len -= 1;
        Some(val)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }
}

/// Structured telemetry bus backed by a fixed-capacity ring buffer.
pub struct TelemetryBus<const CAP: usize> {
    ring: RingBuffer<TelemetryEvent, CAP>,
    tick: u64,
    pub dropped_count: u64,
}

impl<const CAP: usize> TelemetryBus<CAP> {
    pub const fn new() -> Self {
        Self {
            ring: RingBuffer::new(QueueFullPolicy::OverwriteOldest),
            tick: 0,
            dropped_count: 0,
        }
    }

    #[inline]
    pub fn emit(
        &mut self,
        module: Module,
        event: EventKind,
        status: Status,
        code: ErrorCode,
        flags: u16,
        id: u32,
        value: u32,
    ) {
        self.tick = self.tick.wrapping_add(1);
        let ev = TelemetryEvent {
            tick: self.tick,
            module,
            event,
            status,
            code,
            flags,
            id,
            value,
        };
        if self.ring.is_full() {
            self.dropped_count += 1;
        }
        self.ring.push(ev);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<TelemetryEvent> {
        self.ring.pop()
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.ring.len()
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

impl<const CAP: usize> Default for TelemetryBus<CAP> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fixed-Capacity Inline ArrayVec (Sections 3, 5, 27)
// ---------------------------------------------------------------------------

/// Stack/inline fixed-capacity vector with zero heap allocation.
pub struct ArrayVec<T: Copy, const CAP: usize> {
    buf: [MaybeUninit<T>; CAP],
    len: usize,
}

impl<T: Copy, const CAP: usize> ArrayVec<T, CAP> {
    pub const fn new() -> Self {
        Self {
            buf: [const { MaybeUninit::uninit() }; CAP],
            len: 0,
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        CAP
    }

    #[inline]
    pub fn push(&mut self, item: T) -> ErrorCode {
        if self.len >= CAP {
            return ErrorCode::CapacityExceeded;
        }
        self.buf[self.len].write(item);
        self.len += 1;
        ErrorCode::None
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: slot `self.len` was initialized.
        Some(unsafe { self.buf[self.len].assume_init_read() })
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: first `self.len` elements are initialized.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr() as *const T, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: first `self.len` elements are initialized.
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut T, self.len) }
    }
}

impl<T: Copy, const CAP: usize> Default for ArrayVec<T, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const CAP: usize> core::ops::Deref for ArrayVec<T, CAP> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy, const CAP: usize> core::ops::DerefMut for ArrayVec<T, CAP> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
