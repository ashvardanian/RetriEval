//! Recall, intersection, and ranking metrics over borrowed search results.

use std::{alloc::System, iter::Sum, ops::Add};

use crate::{dataset::GroundTruth, Key, Keys};

/// Membership tester over one query's ground-truth prefix, reused across queries.
///
/// NDCG and intersection both ask "is this returned key in the truth set?" once
/// per returned key. Scanning the prefix per lookup costs `O(k²)` per query,
/// which was tolerable while `k` was hardcoded to 10 but is not now that
/// `--top-k` defaults to the ground-truth width — 100 on the BigANN sets,
/// i.e. 10,000 comparisons per query instead of 100. Sorting once into a reused
/// buffer and binary-searching makes it `O(k log k)` with no per-query
/// allocation. Ground-truth rows are ordered by distance, not by key, so the
/// sort cannot be skipped.
struct TruthSet {
    sorted: Vec<Key, System>,
    nearest: Option<Key>,
}

impl Default for TruthSet {
    fn default() -> Self {
        Self {
            sorted: Vec::new_in(System),
            nearest: None,
        }
    }
}

impl TruthSet {
    /// Replaces the set with `truth`, nearest first.
    fn load(&mut self, truth: impl Iterator<Item = Key>) {
        self.sorted.clear();
        self.sorted.extend(truth);
        self.nearest = self.sorted.first().copied();
        self.sorted.sort_unstable();
    }

    fn len(&self) -> usize {
        self.sorted.len()
    }

    fn contains(&self, key: Key) -> bool {
        self.sorted.binary_search(&key).is_ok()
    }
}

/// One pass of a query set scored against its ground truth, every metric at the same `k`.
pub struct Scores {
    /// Rank-1 truth found at rank 1.
    pub recall_at_1: f64,
    /// Rank-1 truth found anywhere in the top `k`: the "1-recall@K" / hit-rate convention of
    /// FAISS's `OneRecallAtRCriterion` and USearch's `mean_recall`. It is *not* the set-overlap
    /// "K-recall@K" that ann-benchmarks and cuVS report. Neither generally bounds the other;
    /// the two must not be compared as equivalent metrics, and they coincide at K=1.
    pub recall_at_k: f64,
    /// The mean share of each query's top-`k` truth that came back, `|top-K ∩ truth-K| / K`:
    /// FAISS's `IntersectionCriterion`, and what ann-benchmarks and cuVS publish as recall. It
    /// ignores rank and does not saturate as K grows, which keeps it informative at K=100.
    pub intersection_at_k: f64,
    /// Discounted gain of the truth found in the top `k`, normalized by the best possible.
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
    /// One query's `results`, best first and at most `k`, against its `truth`.
    fn of_query(results: &[Key], truth: &TruthSet, discount: &[f64]) -> Self {
        let Some(nearest) = truth.nearest else {
            return Self {
                queries: 1,
                ..Self::default()
            };
        };
        let matched = |rank: &usize| truth.contains(results[*rank]);
        let overlap = (0..results.len()).filter(matched).count();
        let gain: f64 = (0..results.len()).filter(matched).map(|rank| discount[rank]).sum();
        let ideal: f64 = discount[..truth.len()].iter().sum();
        Self {
            queries: 1,
            recall_at_1: f64::from(results.first() == Some(&nearest)),
            recall_at_k: f64::from(results.contains(&nearest)),
            // Normalized by the truth available, so a ground truth narrower than `k` cannot depress it.
            intersection_at_k: overlap as f64 / truth.len() as f64,
            // Clamped, since a key returned twice is credited twice.
            ndcg_at_k: (gain / ideal).min(1.0),
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

/// Scores each query's `found` keys, best first, against its ground-truth rows named by `base_keys`.
pub fn score<'a>(
    found: impl Iterator<Item = &'a [Key]>,
    ground_truth: &GroundTruth,
    base_keys: &Keys,
    k: usize,
) -> Scores {
    let discount = discount_table(k);
    let mut truth = TruthSet::default();
    found
        .take(ground_truth.queries())
        .enumerate()
        .map(|(query_index, found)| {
            truth.load(
                ground_truth
                    .neighbors(query_index)
                    .take(k)
                    .map(|row| base_keys.get(row)),
            );
            ScoreSums::of_query(&found[..found.len().min(k)], &truth, &discount)
        })
        .sum::<ScoreSums>()
        .mean()
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
    use super::{Key, TruthSet};

    /// The sorted-and-binary-searched set must answer exactly what a linear
    /// scan of the same prefix would, including for duplicate and absent keys.
    #[test]
    fn truth_set_matches_linear_scan() {
        let mut set = TruthSet::default();
        // Unsorted and with a duplicate, as ground-truth rows may be.
        let cases: [&[u32]; 5] = [
            &[],
            &[7],
            &[9, 3, 5, 1],
            &[4, 4, 2],
            &[u32::MAX, 0, 12345, 6, 6, 999_999],
        ];
        for truth in cases {
            set.load(truth.iter().map(|&key| Key::from(key)));
            assert_eq!(set.nearest, truth.first().map(|&key| Key::from(key)));
            for probe in 0..64u32 {
                assert_eq!(
                    set.contains(Key::from(probe)),
                    truth.contains(&probe),
                    "probe {probe} in {truth:?}"
                );
            }
            for &probe in truth {
                assert!(set.contains(Key::from(probe)), "member {probe} missing from {truth:?}");
            }
        }
    }

    /// `load` must fully replace the previous contents, not accumulate.
    #[test]
    fn truth_set_load_replaces() {
        let mut set = TruthSet::default();
        set.load([1u32, 2, 3].map(Key::from).into_iter());
        set.load([4u32, 5].map(Key::from).into_iter());
        assert!([4u32, 5].map(Key::from).iter().all(|&key| set.contains(key)));
        assert!(![1u32, 2, 3].map(Key::from).iter().any(|&key| set.contains(key)));
    }
}
