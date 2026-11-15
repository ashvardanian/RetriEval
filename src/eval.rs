//! Recall, intersection, and ranking metrics over borrowed search results, judged by exact distance.

use std::{alloc::System, iter::Sum, ops::Add};

use numkong::{self as nk, Angular, Dot, Euclidean, Hamming, Jaccard};

use crate::{dataset::GroundTruth, Dataset, Key, Keys, Vectors};

/// How [`score`] measures the exact distance between two rows; smaller is nearer under every one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Inner,
    Angular,
    Euclidean,
    Manhattan,
    Hamming,
    Jaccard,
}

impl Metric {
    /// The metric a backend's `metric` label names, in any engine's spelling.
    pub fn from_label(label: &str) -> Option<Self> {
        match label.to_ascii_lowercase().as_str() {
            "ip" | "inner" | "dot" => Some(Self::Inner),
            "cos" | "cosine" | "angular" => Some(Self::Angular),
            "l2" | "l2sq" | "euclidean" | "sqeuclidean" => Some(Self::Euclidean),
            "manhattan" | "l1" => Some(Self::Manhattan),
            "hamming" => Some(Self::Hamming),
            "jaccard" | "tanimoto" => Some(Self::Jaccard),
            _ => None,
        }
    }
}

/// Whether `found` ranks no further than `exact`, within the rounding two kernels may differ by.
fn within(found: f64, exact: f64) -> bool {
    found <= exact + 1e-6 * exact.abs().max(1.0)
}

/// The exact distance between two rows of a numeric type under `metric`.
fn numeric_distance<Scalar>(metric: Metric, first: &[Scalar], second: &[Scalar]) -> Result<f64, String>
where
    Scalar: Dot + Angular + Euclidean + Copy + Into<f64>,
    <Scalar as Dot>::Output: Into<f64>,
    <Scalar as Angular>::Output: Into<f64>,
    <Scalar as Euclidean>::SqEuclideanOutput: Into<f64>,
{
    let kernel = |error: nk::Error| format!("exact {metric:?} distance: {error:?}");
    Ok(match metric {
        Metric::Inner => -Into::<f64>::into(Scalar::dot(first, second).map_err(kernel)?),
        Metric::Angular => Scalar::angular(first, second).map_err(kernel)?.into(),
        Metric::Euclidean => Scalar::sqeuclidean(first, second).map_err(kernel)?.into(),
        Metric::Manhattan => first
            .iter()
            .zip(second)
            .map(|(&first, &second)| (first.into() - second.into()).abs())
            .sum(),
        Metric::Hamming | Metric::Jaccard => return Err(format!("{metric:?} needs binary vectors")),
    })
}

/// The exact distance between two binary rows under `metric`.
fn binary_distance(metric: Metric, first: &[nk::u1x8], second: &[nk::u1x8]) -> Result<f64, String> {
    let kernel = |error: nk::Error| format!("exact {metric:?} distance: {error:?}");
    Ok(match metric {
        Metric::Hamming => f64::from(nk::u1x8::hamming(first, second).map_err(kernel)?),
        Metric::Jaccard => f64::from(nk::u1x8::jaccard(first, second).map_err(kernel)?),
        Metric::Inner => -f64::from(nk::u1x8::dot(first, second).map_err(kernel)?),
        Metric::Angular | Metric::Euclidean | Metric::Manhattan => {
            return Err(format!("{metric:?} needs numeric vectors"))
        }
    })
}

/// The exact distance between one-row batches `first` and `second` of one scalar type.
fn exact_distance(metric: Metric, first: &Vectors<'_>, second: &Vectors<'_>) -> Result<f64, String> {
    const ONE_ROW: &str = "a one-row batch is packed";
    match (first, second) {
        (Vectors::F32(first), Vectors::F32(second)) => numeric_distance(
            metric,
            first.as_packed_slice().expect(ONE_ROW),
            second.as_packed_slice().expect(ONE_ROW),
        ),
        (Vectors::I8(first), Vectors::I8(second)) => numeric_distance(
            metric,
            first.as_packed_slice().expect(ONE_ROW),
            second.as_packed_slice().expect(ONE_ROW),
        ),
        (Vectors::U8(first), Vectors::U8(second)) => numeric_distance(
            metric,
            first.as_packed_slice().expect(ONE_ROW),
            second.as_packed_slice().expect(ONE_ROW),
        ),
        (Vectors::B1x8(first), Vectors::B1x8(second)) => binary_distance(
            metric,
            first.as_packed_slice().expect(ONE_ROW),
            second.as_packed_slice().expect(ONE_ROW),
        ),
        _ => Err("queries and base hold different scalar types".into()),
    }
}

/// The rows and metric [`score`] rescores every query's truth and answers over.
pub struct ExactTruth<'a> {
    pub ground_truth: &'a GroundTruth,
    pub base: &'a Dataset,
    pub queries: &'a Dataset,
    pub base_keys: &'a Keys,
    pub metric: Metric,
}

/// One pass of a query set scored against its ground truth, every metric at the same `k`, every
/// distance recomputed exactly from the base, so a neighbor tied with a true one counts whichever
/// key the ground truth kept.
pub struct Scores {
    /// The first answer within the true nearest distance: 1-recall@1.
    pub recall_at_1: f64,
    /// Any of the top `k` within the true nearest distance: the "1-recall@K" / hit-rate convention
    /// of FAISS's `OneRecallAtRCriterion` and USearch's `mean_recall`. It is *not* the set-overlap
    /// "K-recall@K" that ann-benchmarks and cuVS report. Neither generally bounds the other;
    /// the two must not be compared as equivalent metrics, and they coincide at K=1.
    pub recall_at_k: f64,
    /// The mean share of each query's top `k` within the `k`-th true distance, at most `k` of them:
    /// K-recall@K, FAISS's `IntersectionCriterion`, and what ann-benchmarks and cuVS publish as
    /// recall. It ignores rank and does not saturate as K grows, which keeps it informative at K=100.
    pub intersection_at_k: f64,
    /// Discounted gain of the answers within the `k`-th true distance, normalized by the best possible.
    pub ndcg_at_k: f64,
}

/// Every metric summed over the queries scored. Sums add associatively, so the partial sums of any
/// split of the queries combine into the same total before [`ScoreSums::mean`] divides once.
#[derive(Clone, Copy, Default)]
struct ScoreSums {
    queries: usize,
    recall_at_1: f64,
    recall_at_k: f64,
    intersection_at_k: f64,
    ndcg_at_k: f64,
}

impl Add for ScoreSums {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            queries: self.queries + other.queries,
            recall_at_1: self.recall_at_1 + other.recall_at_1,
            recall_at_k: self.recall_at_k + other.recall_at_k,
            intersection_at_k: self.intersection_at_k + other.intersection_at_k,
            ndcg_at_k: self.ndcg_at_k + other.ndcg_at_k,
        }
    }
}

impl Sum for ScoreSums {
    fn sum<Iter: Iterator<Item = Self>>(sums: Iter) -> Self {
        sums.fold(Self::default(), Add::add)
    }
}

impl ScoreSums {
    /// One query's exact answer distances, best first and at most `k`, against its exact truth distances.
    fn of_query(found: &[f64], truth: &[f64], discount: &[f64]) -> Self {
        if truth.is_empty() {
            return Self {
                queries: 1,
                ..Self::default()
            };
        }
        let nearest = truth.iter().copied().fold(f64::INFINITY, f64::min);
        let farthest = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let hits = || (0..found.len()).filter(|&rank| within(found[rank], farthest));
        // Normalized by the truth available, so a ground truth narrower than `k` cannot depress it.
        let expected = truth.len();
        let gain: f64 = hits().take(expected).map(|rank| discount[rank]).sum();
        let ideal: f64 = discount[..expected].iter().sum();
        Self {
            queries: 1,
            recall_at_1: f64::from(found.first().is_some_and(|&distance| within(distance, nearest))),
            recall_at_k: f64::from(found.iter().any(|&distance| within(distance, nearest))),
            intersection_at_k: hits().count().min(expected) as f64 / expected as f64,
            ndcg_at_k: gain / ideal,
        }
    }

    fn mean(self) -> Scores {
        let mean = |total: f64| match self.queries {
            0 => 0.0,
            queries => total / queries as f64,
        };
        Scores {
            recall_at_1: mean(self.recall_at_1),
            recall_at_k: mean(self.recall_at_k),
            intersection_at_k: mean(self.intersection_at_k),
            ndcg_at_k: mean(self.ndcg_at_k),
        }
    }
}

/// Scores each query's `found` keys, best first, against its ground-truth rows, rescoring both
/// exactly over `truth`'s rows at `dimensions`.
pub fn score<'a>(
    found: impl Iterator<Item = &'a [Key]>,
    truth: &ExactTruth<'_>,
    dimensions: usize,
    k: usize,
) -> Result<Scores, String> {
    let discount = discount_table(k);
    let mut query_scratch = vec![0u8; truth.queries.vector_bytes()];
    let mut row_scratch = vec![0u8; truth.base.vector_bytes()];
    let (mut found_distances, mut truth_distances) = (Vec::new_in(System), Vec::new_in(System));
    let mut sums = ScoreSums::default();
    for (query_index, keys) in found.take(truth.ground_truth.queries()).enumerate() {
        let query = truth.queries.slice(query_index, 1, dimensions, &mut query_scratch);
        let mut exact = |row: usize| {
            exact_distance(
                truth.metric,
                &query,
                &truth.base.slice(row, 1, dimensions, &mut row_scratch),
            )
        };
        truth_distances.clear();
        for row in truth.ground_truth.neighbors(query_index).take(k) {
            truth_distances.push(exact(row)?);
        }
        found_distances.clear();
        for &key in &keys[..keys.len().min(k)] {
            let row = truth
                .base_keys
                .row_of(key)
                .ok_or_else(|| format!("query {query_index} found {key}, which no base row holds"))?;
            found_distances.push(exact(row)?);
        }
        sums = sums + ScoreSums::of_query(&found_distances, &truth_distances, &discount);
    }
    Ok(sums.mean())
}

/// Counts the queries whose own `expected` key came back at rank 1 and within the top `k`.
///
/// Self-recall needs no ground-truth file: a vector that is in the index is its own nearest
/// neighbor. Exact-duplicate vectors in the base can cost a hit, since either twin may take rank
/// 1, so self-recall below 1.0 bounds the index's error rate from above rather than measuring it.
pub fn self_recall_hits<'a>(
    found: impl Iterator<Item = &'a [Key]>,
    expected: impl Iterator<Item = Key>,
    k: usize,
) -> (usize, usize) {
    found.zip(expected).fold((0, 0), |(at_1, at_k), (found, expected)| {
        let results = &found[..found.len().min(k)];
        (
            at_1 + usize::from(results.first() == Some(&expected)),
            at_k + usize::from(results.contains(&expected)),
        )
    })
}

/// Precomputed log2 table for NDCG discount factors: 1/log2(rank+1) for rank 1..=K.
/// discount[0] = 1/log2(2) = 1.0, discount[1] = 1/log2(3) ≈ 0.63, etc.
fn discount_table(k: usize) -> Vec<f64, System> {
    let mut values = Vec::with_capacity_in(k, System);
    values.extend((0..k).map(|rank| 1.0 / ((rank + 2) as f64).log2()));
    values
}

// #region Tests

#[cfg(test)]
mod tests {
    use super::{discount_table, ScoreSums};

    /// Without ties, counting by exact distance agrees with counting the truth's own keys.
    #[test]
    fn distinct_distances_count_like_keys() {
        let discount = discount_table(3);
        let truth = [1.0, 2.0, 3.0];
        // The answers hold the first and third true neighbors, then one past the truth.
        let found = ScoreSums::of_query(&[1.0, 3.0, 5.0], &truth, &discount).mean();
        assert_eq!((found.recall_at_1, found.intersection_at_k), (1.0, 2.0 / 3.0));
        let missed = ScoreSums::of_query(&[2.0, 3.0, 5.0], &truth, &discount).mean();
        assert_eq!((missed.recall_at_1, missed.recall_at_k), (0.0, 0.0));
    }

    /// A neighbor tied with the truth's nearest or farthest counts, though its key is not the one
    /// the truth kept, and so does the rounding a kernel may add.
    #[test]
    fn ties_count_whichever_key_the_truth_kept() {
        let discount = discount_table(3);
        let truth = [0.0, 4.0, 9.0];
        let tied = ScoreSums::of_query(&[0.0, 4.0, 9.0 + 1e-7], &truth, &discount).mean();
        assert_eq!(
            (tied.recall_at_1, tied.intersection_at_k, tied.ndcg_at_k),
            (1.0, 1.0, 1.0)
        );
        let farther = ScoreSums::of_query(&[0.0, 4.0, 9.1], &truth, &discount).mean();
        assert_eq!(farther.intersection_at_k, 2.0 / 3.0);
    }
}
