//! Recall, intersection, and ranking metrics over borrowed search results.

use std::alloc::System;

use crate::{dataset::GroundTruth, Key};

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
}

impl Default for TruthSet {
    fn default() -> Self {
        Self {
            sorted: Vec::new_in(System),
        }
    }
}

impl TruthSet {
    fn load(&mut self, truth: &[Key]) {
        self.sorted.clear();
        self.sorted.extend_from_slice(truth);
        self.sorted.sort_unstable();
    }

    fn contains(&self, key: Key) -> bool {
        self.sorted.binary_search(&key).is_ok()
    }
}

/// Compute recall@K: the fraction of queries where the true nearest neighbor
/// (rank-1 ground truth) appears within the top-K search results.
///
/// This is the "1-recall@K" / hit-rate@K convention — FAISS's
/// `OneRecallAtRCriterion`, and identical to USearch's `mean_recall`. It equals
/// textbook recall@K when the relevant set is the single true nearest neighbor.
/// It is *not* the set-intersection "K-recall@K" that ann-benchmarks and cuVS
/// report (`|top-K ∩ ground-truth-K| / K`), and since 1-recall@K ≥ K-recall@K
/// for the same run, these numbers must not be compared across the two
/// conventions. The two coincide at K=1.
pub fn recall_at_k(
    out_keys: &[Key],
    out_counts: &[usize],
    neighbor_count: usize,
    ground_truth: &GroundTruth,
    k: usize,
) -> f64 {
    let num_queries = out_counts.len().min(ground_truth.queries());
    if num_queries == 0 || k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;

    for (query_index, &found_count) in out_counts[..num_queries].iter().enumerate() {
        let ground_truth_neighbors = ground_truth.neighbors(query_index);
        if ground_truth_neighbors.is_empty() {
            continue;
        }
        let true_nearest = ground_truth_neighbors[0];

        let offset = query_index * neighbor_count;
        let found = found_count.min(k);
        if out_keys[offset..offset + found].contains(&true_nearest) {
            hits += 1;
        }
    }

    hits as f64 / num_queries as f64
}

/// Compute intersection@K: the mean share of each query's top-K ground-truth
/// set that the search actually returned, `|top-K ∩ truth-K| / K`.
///
/// This is the set-overlap convention — FAISS's `IntersectionCriterion`, and
/// what ann-benchmarks and cuVS publish as "recall". Unlike [`recall_at_k`] it
/// ignores rank and does not saturate as K grows, which is what keeps it
/// informative on datasets whose ground truth is 100 wide.
pub fn intersection_at_k(
    out_keys: &[Key],
    out_counts: &[usize],
    neighbor_count: usize,
    ground_truth: &GroundTruth,
    k: usize,
) -> f64 {
    let num_queries = out_counts.len().min(ground_truth.queries());
    if num_queries == 0 || k == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut truth_set = TruthSet::default();

    for (query_index, &found_count) in out_counts[..num_queries].iter().enumerate() {
        let truth = ground_truth.neighbors(query_index);
        let truth_count = truth.len().min(k);
        if truth_count == 0 {
            continue;
        }
        truth_set.load(&truth[..truth_count]);
        let offset = query_index * neighbor_count;
        let found = found_count.min(k);
        let overlap = out_keys[offset..offset + found]
            .iter()
            .filter(|&&key| truth_set.contains(key))
            .count();
        // Normalized by the truth set actually available, so a ground-truth file
        // narrower than `k` cannot depress the score.
        total += overlap as f64 / truth_count as f64;
    }

    total / num_queries as f64
}

/// Count how many queries in one batch retrieved their own key within the top-K.
///
/// Self-recall needs no ground-truth file: a vector that is in the index is its
/// own nearest neighbor, so `expected_keys[query]` is the truth for that row.
/// Returns a count rather than a ratio so callers can stream the entire base
/// through in batches and accumulate, instead of materializing a
/// `queries × K` result matrix — at 100M vectors that matrix is gigabytes.
///
/// Exact-duplicate vectors in the base can cost a hit, since either twin may
/// take rank 1. Self-recall below 1.0 is therefore an upper bound on the
/// index's error rate, not a measurement of it alone.
pub fn self_recall_at_k(
    out_keys: &[Key],
    out_counts: &[usize],
    neighbor_count: usize,
    expected_keys: &[Key],
    k: usize,
) -> usize {
    if k == 0 {
        return 0;
    }

    let mut hits = 0usize;

    for (query_index, (&found_count, &expected)) in out_counts.iter().zip(expected_keys).enumerate() {
        let offset = query_index * neighbor_count;
        let found = found_count.min(k);
        if out_keys[offset..offset + found].contains(&expected) {
            hits += 1;
        }
    }

    hits
}

/// Precomputed log2 table for NDCG discount factors: 1/log2(rank+1) for rank 1..=K.
/// discount[0] = 1/log2(2) = 1.0, discount[1] = 1/log2(3) ≈ 0.63, etc.
fn discount_table(k: usize) -> Vec<f64, System> {
    let mut values = Vec::with_capacity_in(k, System);
    values.extend((0..k).map(|rank| 1.0 / ((rank + 2) as f64).log2()));
    values
}

/// Compute NDCG@K (Normalized Discounted Cumulative Gain).
/// Checks which of the top-K ground truth neighbors appear in our top-K results,
/// weighted by their position in the result list.
pub fn ndcg_at_k(
    out_keys: &[Key],
    out_counts: &[usize],
    neighbor_count: usize,
    ground_truth: &GroundTruth,
    k: usize,
) -> f64 {
    let num_queries = out_counts.len().min(ground_truth.queries());
    if num_queries == 0 || k == 0 {
        return 0.0;
    }

    let discount = discount_table(k);
    let mut total_ndcg = 0.0;
    let mut truth_set = TruthSet::default();

    for (query_index, &found_count) in out_counts[..num_queries].iter().enumerate() {
        let ground_truth_neighbors = ground_truth.neighbors(query_index);
        let ground_truth_count = ground_truth_neighbors.len().min(k);
        truth_set.load(&ground_truth_neighbors[..ground_truth_count]);
        let offset = query_index * neighbor_count;
        let found = found_count.min(k);
        let results = &out_keys[offset..offset + found];

        let mut dcg = 0.0;
        for (rank, &key) in results.iter().enumerate() {
            if truth_set.contains(key) {
                dcg += discount[rank];
            }
        }

        let query_idcg: f64 = discount[..ground_truth_count].iter().sum();
        if query_idcg > 0.0 {
            // Clamped because `dcg` credits every result position that matches
            // the truth set, with no dedup: if a backend returns the same key
            // twice (possible with a non-unique `--keys` file) and the truth
            // set is shorter than `k`, the ratio can exceed 1 and NDCG stops
            // being a normalized measure.
            total_ndcg += (dcg / query_idcg).min(1.0);
        }
    }

    total_ndcg / num_queries as f64
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
        // Deliberately unsorted, with a duplicate, since ground-truth rows are
        // ordered by distance and nothing forbids repeats.
        let cases: [&[Key]; 5] = [
            &[],
            &[7],
            &[9, 3, 5, 1],
            &[4, 4, 2],
            &[u32::MAX, 0, 12345, 6, 6, 999_999],
        ];
        for truth in cases {
            set.load(truth);
            for probe in 0..64u32 {
                assert_eq!(
                    set.contains(probe),
                    truth.contains(&probe),
                    "probe {probe} in {truth:?}"
                );
            }
            for &probe in truth {
                assert!(set.contains(probe), "member {probe} missing from {truth:?}");
            }
            assert!(!set.contains(u32::MAX - 1) || truth.contains(&(u32::MAX - 1)));
        }
    }

    /// `load` must fully replace the previous contents, not accumulate.
    #[test]
    fn truth_set_load_replaces() {
        let mut set = TruthSet::default();
        set.load(&[1, 2, 3]);
        set.load(&[4, 5]);
        assert!(set.contains(4) && set.contains(5));
        assert!(!set.contains(1) && !set.contains(2) && !set.contains(3));
    }
}
