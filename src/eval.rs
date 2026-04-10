use crate::dataset::GroundTruth;
use crate::Key;

/// Compute recall@K: the fraction of queries where the true nearest neighbor
/// (rank-1 ground truth) appears within the top-K search results.
pub fn recall_at_k(
    out_keys: &[Key],
    out_counts: &[usize],
    max_count: usize,
    ground_truth: &GroundTruth,
    k: usize,
) -> f64 {
    let num_queries = out_counts.len().min(ground_truth.queries());
    if num_queries == 0 || k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;

    for (query_idx, &found_count) in out_counts[..num_queries].iter().enumerate() {
        let ground_truth_neighbors = ground_truth.neighbors(query_idx);
        if ground_truth_neighbors.is_empty() {
            continue;
        }
        let true_nearest = ground_truth_neighbors[0];

        let offset = query_idx * max_count;
        let found = found_count.min(k);
        if out_keys[offset..offset + found].contains(&true_nearest) {
            hits += 1;
        }
    }

    hits as f64 / num_queries as f64
}

/// Precomputed log2 table for NDCG discount factors: 1/log2(rank+1) for rank 1..=K.
/// discount[0] = 1/log2(2) = 1.0, discount[1] = 1/log2(3) ≈ 0.63, etc.
fn discount_table(k: usize) -> Vec<f64> {
    (0..k).map(|rank| 1.0 / ((rank + 2) as f64).log2()).collect()
}

/// Compute NDCG@K (Normalized Discounted Cumulative Gain).
/// Checks which of the top-K ground truth neighbors appear in our top-K results,
/// weighted by their position in the result list.
/// K² comparisons per query — efficient for small K (typically 10).
pub fn ndcg_at_k(
    out_keys: &[Key],
    out_counts: &[usize],
    max_count: usize,
    ground_truth: &GroundTruth,
    k: usize,
) -> f64 {
    let num_queries = out_counts.len().min(ground_truth.queries());
    if num_queries == 0 || k == 0 {
        return 0.0;
    }

    let discount = discount_table(k);
    let mut total_ndcg = 0.0;

    for (query_idx, &found_count) in out_counts[..num_queries].iter().enumerate() {
        let ground_truth_neighbors = ground_truth.neighbors(query_idx);
        let ground_truth_count = ground_truth_neighbors.len().min(k);
        let offset = query_idx * max_count;
        let found = found_count.min(k);
        let results = &out_keys[offset..offset + found];

        let mut dcg = 0.0;
        for (rank, &key) in results.iter().enumerate() {
            if ground_truth_neighbors[..ground_truth_count].contains(&key) {
                dcg += discount[rank];
            }
        }

        let query_idcg: f64 = discount[..ground_truth_count].iter().sum();
        if query_idcg > 0.0 {
            total_ndcg += dcg / query_idcg;
        }
    }

    total_ndcg / num_queries as f64
}

/// Re-normalize a recall/NDCG metric for partial indexing.
/// When only `indexed` out of `total` vectors are in the index,
/// the expected recall for a perfect search is `indexed/total`.
/// Normalized = raw / (indexed/total), capped at 1.0.
pub fn normalize_metric(raw: f64, indexed: usize, total: usize) -> f64 {
    if indexed == 0 || total == 0 {
        return 0.0;
    }
    let fraction = indexed as f64 / total as f64;
    (raw / fraction).min(1.0)
}
