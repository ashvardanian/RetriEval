//! A trait abstracting over NumKong's packed distance kernels so that one
//! generic top-K helper can serve both Hamming (`u1x8 → u32`) and Euclidean
//! (`f32 → f64`) workloads without duplicated code paths.
//!
//! NumKong itself has no super-trait that spans `Hammings` and `Euclideans`
//! with a uniform distance-type shape: `Hammings` hard-codes the output to
//! `u32`, while `Euclideans` returns `Self::SpatialResult`. This thin wrapper
//! bridges the two behind one associated-type-per-metric so the heap-based
//! top-K extractor can be written once and parameterised over the metric.
//!
//! Add new metrics (Jaccard, Angular, Cosine, …) by implementing
//! [`PackedDistance`] for the relevant scalar type and delegating to the
//! corresponding NumKong `try_*_packed_parallel_into` method.

use numkong::{Dots, Matrix, MatrixSpan, PackedMatrix, StorageElement, TensorError};

/// A distance metric that NumKong can compute via its packed-matrix kernels.
///
/// Implementations are expected to be zero-sized — all behaviour lives on
/// the scalar type itself via `numkong::Hammings` / `numkong::Euclideans`.
///
/// # Associated types
///
/// - [`Self::Distance`] is the type NumKong writes into the output tensor.
/// - [`Self::OrderingKey`] is the key used inside the top-K heap. It MUST
///   preserve the natural ordering of `Self::Distance` for values the
///   kernel produces. For integral distances it is the distance itself;
///   for `f64` we project via `f64::to_bits`, which is monotonic for
///   non-negative finite floats (the only values NumKong's Euclidean
///   kernel produces from finite non-negative input).
pub trait PackedDistance: StorageElement + Dots + Sized + Default + Copy + Send + Sync
where
    Self::Distance: StorageElement + Copy + Default + Send + Sync,
    Self::OrderingKey: Ord + Copy + Send + Sync,
{
    /// Scalar type the kernel writes into the output distance matrix.
    type Distance;

    /// Ordered key used inside the top-K heap.
    type OrderingKey;

    /// Project one raw distance into its ordering key. Cheap, branchless.
    fn ordering_key(distance: Self::Distance) -> Self::OrderingKey;

    /// Write the full query-vs-base distance matrix into `out` in parallel.
    ///
    /// `out` must be shape `[queries.shape()[0], packed_base.width()]`.
    /// NumKong's kernel overwrites every entry — callers need not
    /// pre-initialize beyond the allocation.
    fn distances_parallel_into(
        queries: &Matrix<Self>,
        packed_base: &PackedMatrix<Self>,
        out: &mut MatrixSpan<'_, Self::Distance>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError>;

    /// Human-readable label for progress-bar / log messages. Stable across
    /// runs so consumers can grep log output.
    fn metric_name() -> &'static str;
}

// ----------------------------------------------------------------------------
// Hamming (u1x8 → u32)
// ----------------------------------------------------------------------------

impl PackedDistance for numkong::u1x8 {
    type Distance = u32;
    type OrderingKey = u32;

    #[inline(always)]
    fn ordering_key(distance: u32) -> u32 {
        distance
    }

    fn distances_parallel_into(
        queries: &Matrix<Self>,
        packed_base: &PackedMatrix<Self>,
        out: &mut MatrixSpan<'_, u32>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError> {
        queries.try_hammings_packed_parallel_into(packed_base, out, pool)
    }

    fn metric_name() -> &'static str {
        "hamming"
    }
}

// ----------------------------------------------------------------------------
// Euclidean L2 (f32 → f64)
// ----------------------------------------------------------------------------

impl PackedDistance for f32 {
    /// NumKong's `Euclideans::SpatialResult` for `f32` input is `f64` —
    /// wider accumulator avoids precision loss summing many squared diffs
    /// across high dimensions.
    type Distance = f64;

    /// `f64::to_bits()` preserves ordering for non-negative finite floats:
    /// the IEEE-754 binary layout is lexicographic over (sign, exponent,
    /// mantissa), so for the `[0, +∞)` subrange the bit pattern rises
    /// monotonically with the value. NumKong's L2 kernel only produces
    /// non-negative finite results for finite non-negative input, so the
    /// projection is monotonic across all outputs we ever see.
    type OrderingKey = u64;

    #[inline(always)]
    fn ordering_key(distance: f64) -> u64 {
        debug_assert!(
            distance >= 0.0 && distance.is_finite(),
            "L2 distance must be non-negative and finite, got {distance}"
        );
        distance.to_bits()
    }

    fn distances_parallel_into(
        queries: &Matrix<Self>,
        packed_base: &PackedMatrix<Self>,
        out: &mut MatrixSpan<'_, f64>,
        pool: &mut fork_union::ThreadPool,
    ) -> Result<(), TensorError> {
        queries.try_euclideans_packed_parallel_into(packed_base, out, pool)
    }

    fn metric_name() -> &'static str {
        "l2"
    }
}
