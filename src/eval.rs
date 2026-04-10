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

    for q in 0..num_queries {
        let gt = ground_truth.neighbors(q);
        if gt.is_empty() {
            continue;
        }
        let true_nearest = gt[0];

        let offset = q * max_count;
        let found = out_counts[q].min(k);
        if out_keys[offset..offset + found].contains(&true_nearest) {
            hits += 1;
        }
    }

    hits as f64 / num_queries as f64
}

/// Precomputed log2 table for NDCG discount factors: 1/log2(rank+1) for rank 1..=K.
/// discount[0] = 1/log2(2) = 1.0, discount[1] = 1/log2(3) ≈ 0.63, etc.
fn discount_table(k: usize) -> Vec<f64> {
    (0..k).map(|r| 1.0 / ((r + 2) as f64).log2()).collect()
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

    for q in 0..num_queries {
        let gt = ground_truth.neighbors(q);
        let gt_k = gt.len().min(k);
        let offset = q * max_count;
        let found = out_counts[q].min(k);
        let results = &out_keys[offset..offset + found];

        let mut dcg = 0.0;
        for (rank, &key) in results.iter().enumerate() {
            // Check if this result is among the top-K ground truth
            for g in 0..gt_k {
                if key == gt[g] {
                    dcg += discount[rank];
                    break;
                }
            }
        }

        // Ideal DCG for this query uses min(gt_k, k) relevant docs
        let query_idcg: f64 = discount[..gt_k].iter().sum();
        if query_idcg > 0.0 {
            total_ndcg += dcg / query_idcg;
        }
    }

    total_ndcg / num_queries as f64
}
