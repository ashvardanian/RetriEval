//! Recall, intersection, and ranking metrics over borrowed search results.

use std::{alloc::System, iter::Sum, ops::Add};

use crate::{dataset::GroundTruth, Distance, Key, Keys};

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
    /// Each truth key with its stored distance, NaN when the file stores none, sorted by key.
    sorted: Vec<(Key, Distance), System>,
    nearest: Option<Key>,
    /// The farthest truth distance, when the file stores distances.
    farthest: Option<Distance>,
}

impl Default for TruthSet {
    fn default() -> Self {
        Self {
            sorted: Vec::new_in(System),
            nearest: None,
            farthest: None,
        }
    }
}

impl TruthSet {
    /// Replaces the set with `truth`, nearest first, and its `distances` when the file stores them.
    fn load(&mut self, truth: impl Iterator<Item = Key>, distances: Option<impl Iterator<Item = Distance>>) {
        self.sorted.clear();
        match distances {
            Some(distances) => self.sorted.extend(truth.zip(distances)),
            None => self.sorted.extend(truth.map(|key| (key, Distance::NAN))),
        }
        self.nearest = self.sorted.first().map(|&(key, _)| key);
        self.farthest = self
            .sorted
            .last()
            .map(|&(_, distance)| distance)
            .filter(|distance| !distance.is_nan());
        self.sorted.sort_unstable_by_key(|&(key, _)| key);
    }

    fn len(&self) -> usize {
        self.sorted.len()
    }

    fn distance(&self, key: Key) -> Option<Distance> {
        let index = self.sorted.binary_search_by_key(&key, |&(key, _)| key).ok()?;
        Some(self.sorted[index].1)
    }

    fn contains(&self, key: Key) -> bool {
        self.distance(key).is_some()
    }
}

/// Whether an engine's distance and a stored one agree past rounding.
fn same_distance(first: Distance, second: Distance) -> bool {
    (first - second).abs() <= 1e-3 * first.abs().max(second.abs()) + 1e-6
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
    /// The share of the top `k` within the `k`-th true distance, so a neighbor tied with the
    /// truth's last counts whichever key the truth kept. Present only when the ground truth stores
    /// distances and the engine's distances match them on every shared key.
    pub distance_recall_at_k: Option<f64>,
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
    distance_recall_at_k: f64,
    /// Returned keys also in the truth, whose two distances were compared.
    distance_checks: usize,
    /// Compared pairs that disagree, as when an engine reports another scale.
    distance_mismatches: usize,
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
            distance_recall_at_k: self.distance_recall_at_k + other.distance_recall_at_k,
            distance_checks: self.distance_checks + other.distance_checks,
            distance_mismatches: self.distance_mismatches + other.distance_mismatches,
        }
    }
}

impl Sum for ScoreSums {
    fn sum<Iter: Iterator<Item = Self>>(sums: Iter) -> Self {
        sums.fold(Self::default(), Add::add)
    }
}

impl ScoreSums {
    /// One query's `results` and their `distances`, best first and at most `k`, against its `truth`.
    fn of_query(results: &[Key], distances: &[Distance], truth: &TruthSet, discount: &[f64]) -> Self {
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
        let (mut distance_checks, mut distance_mismatches) = (0, 0);
        let mut within = 0;
        if let Some(farthest) = truth.farthest {
            for (&key, &distance) in results.iter().zip(distances) {
                if let Some(stored) = truth.distance(key) {
                    distance_checks += 1;
                    distance_mismatches += usize::from(!same_distance(distance, stored));
                }
                within += usize::from(distance <= farthest || same_distance(distance, farthest));
            }
        }
        Self {
            queries: 1,
            recall_at_1: f64::from(results.first() == Some(&nearest)),
            recall_at_k: f64::from(results.contains(&nearest)),
            // Normalized by the truth available, so a ground truth narrower than `k` cannot depress it.
            intersection_at_k: overlap as f64 / truth.len() as f64,
            // Clamped, since a key returned twice is credited twice.
            ndcg_at_k: (gain / ideal).min(1.0),
            distance_recall_at_k: (within as f64 / truth.len() as f64).min(1.0),
            distance_checks,
            distance_mismatches,
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
            distance_recall_at_k: (self.distance_checks > 0 && self.distance_mismatches == 0)
                .then(|| mean(self.distance_recall_at_k)),
        }
    }
}

/// Scores each query's `found` keys and distances, best first, against its ground-truth rows named
/// by `base_keys`.
pub fn score<'a>(
    found: impl Iterator<Item = (&'a [Key], &'a [Distance])>,
    ground_truth: &GroundTruth,
    base_keys: &Keys,
    k: usize,
) -> Scores {
    let discount = discount_table(k);
    let mut truth = TruthSet::default();
    found
        .take(ground_truth.queries())
        .enumerate()
        .map(|(query_index, (keys, distances))| {
            truth.load(
                ground_truth
                    .neighbors(query_index)
                    .take(k)
                    .map(|row| base_keys.get(row)),
                ground_truth.distances(query_index).map(|distances| distances.take(k)),
            );
            let count = keys.len().min(k);
            ScoreSums::of_query(&keys[..count], &distances[..count], &truth, &discount)
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
    use super::{Distance, Key, ScoreSums, TruthSet};

    const NO_DISTANCES: Option<std::iter::Empty<Distance>> = None;

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
            set.load(truth.iter().map(|&key| Key::from(key)), NO_DISTANCES);
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
        set.load([1u32, 2, 3].map(Key::from).into_iter(), NO_DISTANCES);
        set.load([4u32, 5].map(Key::from).into_iter(), NO_DISTANCES);
        assert!([4u32, 5].map(Key::from).iter().all(|&key| set.contains(key)));
        assert!(![1u32, 2, 3].map(Key::from).iter().any(|&key| set.contains(key)));
    }

    /// A neighbor tied with the truth's farthest counts under distance recall, though its key is
    /// not the one the truth kept; distances on another scale leave the metric out.
    #[test]
    fn distance_recall_counts_ties_and_refuses_other_scales() {
        let mut truth = TruthSet::default();
        let keys = [1u32, 2, 3].map(Key::from);
        truth.load(keys.into_iter(), Some([0.0, 4.0, 9.0].into_iter()));
        let discount = super::discount_table(3);
        let tied = [1u32, 2, 7].map(Key::from);
        let sums = ScoreSums::of_query(&tied, &[0.0, 4.0, 9.0], &truth, &discount);
        assert_eq!(sums.mean().intersection_at_k, 2.0 / 3.0);
        assert_eq!(sums.mean().distance_recall_at_k, Some(1.0));
        let similarities = ScoreSums::of_query(&tied, &[-1.0, -0.5, -0.2], &truth, &discount);
        assert_eq!(similarities.mean().distance_recall_at_k, None);
    }
}
