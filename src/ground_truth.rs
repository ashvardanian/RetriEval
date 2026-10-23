//! Exact top-K ground truth over NumKong's packed distance kernels, and the BigANN-style files it
//! lands in, shared by `retri-generate` and the dataset downloaders under `scripts/`.

use std::{
    alloc::System,
    fs::File,
    io::{self, BufWriter, Write},
    ops::Range,
    path::Path,
};

use forkunion::{IntoParallelIterator, ParallelIteratorExt, ThreadPool};
use numkong::{self as nk, MatrixSpan, MatrixView};
use rand::{rngs::StdRng, seq::index::sample as sample_without_replacement, SeedableRng};

use crate::{
    error::GroundTruthError, packed_distance::PackedDistance, pod_slice_as_bytes, row_ranges, spawn_pool, Dataset,
    Seed, Threads, Vectors,
};

/// Bytes of the per-call distance tile, enough cells for one kernel call to keep every core busy.
/// The base is walked in tiles of this size, so memory stays fixed however large the base grows.
const DISTANCE_TILE_BYTES: usize = 1 << 30;

/// Exact top-K via NumKong's packed distance kernels plus a ForkUnion-parallel
/// top-K pass (NumKong provides distances but no top-K).
///
/// `base` lists the parts of one base set, numbered on from each other; the base parts and
/// `queries` must agree on width, and `ground_truth` is `[query_count, top_k]`. Every query is
/// scored against each base tile in one call. Ties break toward the lower base index.
pub fn compute_top_k<Metric: PackedDistance>(
    base: &[MatrixView<'_, Metric>],
    queries: MatrixView<'_, Metric>,
    ground_truth: MatrixSpan<'_, u32>,
    threads: Threads,
) -> Result<(), GroundTruthError> {
    let tile_rows = DISTANCE_TILE_BYTES / (queries.shape()[0].max(1) * std::mem::size_of::<Metric::Distance>());
    compute_top_k_tiled(base, queries, ground_truth, tile_rows, threads)
}

/// [`compute_top_k`] with an explicit base tile, which tests shrink to cross tile boundaries.
fn compute_top_k_tiled<Metric: PackedDistance>(
    base: &[MatrixView<'_, Metric>],
    queries: MatrixView<'_, Metric>,
    mut ground_truth: MatrixSpan<'_, u32>,
    tile_rows: usize,
    threads: Threads,
) -> Result<(), GroundTruthError> {
    let dimensions = queries.shape()[1];
    if let Some(part) = base.iter().find(|part| part.shape()[1] != dimensions) {
        return Err(GroundTruthError::DimensionMismatch {
            queries: dimensions,
            base: part.shape()[1],
        });
    }
    let base_count: usize = base.iter().map(|part| part.shape()[0]).sum();
    let query_count = queries.shape()[0];
    let top_k = checked_top_k(base_count, query_count, &ground_truth)?;
    let tile_rows = tile_rows.clamp(top_k, base_count);
    eprintln!(
        "  ground truth ({metric}): base_count={base_count}, query_count={query_count}, \
         dimensions={dimensions}, top_k={top_k}, tile_rows={tile_rows}, threads={threads}",
        metric = Metric::metric_name()
    );

    let mut pool = pool_of(threads)?;
    // One packed tile, one distance tile and one ranking per query, reused tile after tile.
    let mut packed_tile = nk::DotsPackedMatrix::<Metric>::empty_in(nk::Global);
    let mut distances = nk::Matrix::full(&[query_count, tile_rows], Metric::Distance::default())?;
    let mut best = Vec::with_capacity_in(query_count * top_k, System);
    best.resize(query_count * top_k, (Metric::WORST_KEY, u32::MAX));
    let mut rankings = Vec::with_capacity_in(query_count, System);
    rankings.extend(best.chunks_exact_mut(top_k));
    let progress = ground_truth_progress(base_count, Metric::metric_name());

    for (tile_start, part, rows) in tiles(base.iter().map(|part| part.shape()[0]), tile_rows) {
        // The slice-range form keeps sub-byte rows whole, since the last axis stays full.
        let tile = base[part].slice(&[nk::SliceRange::range(rows.start, rows.end), nk::SliceRange::full()])?;
        packed_tile.pack_into(&tile)?;
        distances.resize(&[query_count, rows.len()])?;
        Metric::distances_parallel_into(&queries, &packed_tile, &mut distances.span(), &mut pool)?;
        fold_tile(
            distances.as_slice(),
            rows.len(),
            tile_start,
            &mut rankings,
            Metric::ordering_key,
            &mut pool,
        )?;
        progress.set_position((tile_start + rows.len()) as u64);
    }
    progress.finish_and_clear();
    land_ranking(&best, top_k, &mut ground_truth);
    Ok(())
}

/// The shape metrics exact ground truth ranks by, both between centered clouds over the 8 sign flips of
/// their principal axes, so a ranking matches what a USearch shapes index walks by.
#[derive(Clone, Copy, Debug)]
pub enum ShapeMetric {
    Chamfer,
    Hausdorff,
}

/// Point clouds as a shapes dataset stores them: every cloud's points end to end, and each cloud's
/// point count.
#[derive(Clone, Copy)]
pub struct Clouds<'a> {
    pub points: &'a [[nk::f16; 3]],
    pub counts: &'a [u32],
}

impl<'a> Clouds<'a> {
    /// Each cloud's points in order, borrowed from the tape.
    fn each(self) -> impl Iterator<Item = &'a [[nk::f16; 3]]> {
        self.counts.iter().scan(0usize, move |first, &count| {
            let cloud = &self.points[*first..*first + count as usize];
            *first += count as usize;
            Some(cloud)
        })
    }
}

/// Exact shape top-K: every query cloud packed once, the base packed a tile at a time and scored
/// against all of them, rankings carried across tiles. Ties break toward the lower base index.
pub fn compute_shape_top_k(
    base: &[Clouds<'_>],
    queries: Clouds<'_>,
    metric: ShapeMetric,
    ground_truth: MatrixSpan<'_, u32>,
    threads: Threads,
) -> Result<(), GroundTruthError> {
    let tile_rows = DISTANCE_TILE_BYTES / (queries.counts.len().max(1) * std::mem::size_of::<f32>());
    compute_shape_top_k_tiled(base, queries, metric, ground_truth, tile_rows, threads)
}

/// [`compute_shape_top_k`] with an explicit base tile, which tests shrink to cross tile boundaries.
fn compute_shape_top_k_tiled(
    base: &[Clouds<'_>],
    queries: Clouds<'_>,
    metric: ShapeMetric,
    mut ground_truth: MatrixSpan<'_, u32>,
    tile_rows: usize,
    threads: Threads,
) -> Result<(), GroundTruthError> {
    let base_count: usize = base.iter().map(|part| part.counts.len()).sum();
    let query_count = queries.counts.len();
    let top_k = checked_top_k(base_count, query_count, &ground_truth)?;
    let tile_rows = tile_rows.clamp(top_k, base_count);
    eprintln!(
        "  ground truth ({metric:?}): base_count={base_count}, query_count={query_count}, top_k={top_k}, \
         tile_rows={tile_rows}, threads={threads}"
    );

    let mut pool = pool_of(threads)?;
    let mut query_clouds = Vec::with_capacity_in(query_count, System);
    query_clouds.extend(queries.each());
    let packed_queries = nk::ShapesPackedMatrix::new_parallel_in(&query_clouds, nk::Global, &mut pool)?;
    // One tile of cloud borrows, one score tile and one ranking per query for the whole call.
    let mut tile_clouds = Vec::with_capacity_in(tile_rows, System);
    let mut scores = Vec::with_capacity_in(query_count * tile_rows, System);
    scores.resize(query_count * tile_rows, 0.0f32);
    let mut best = Vec::with_capacity_in(query_count * top_k, System);
    best.resize(query_count * top_k, (u32::MAX, u32::MAX));
    let mut rankings = Vec::with_capacity_in(query_count, System);
    rankings.extend(best.chunks_exact_mut(top_k));
    let progress = ground_truth_progress(base_count, "shapes");

    for (tile_start, part, rows) in tiles(base.iter().map(|part| part.counts.len()), tile_rows) {
        tile_clouds.clear();
        tile_clouds.extend(base[part].each().skip(rows.start).take(rows.len()));
        let packed_tile = nk::ShapesPackedMatrix::new_parallel_in(&tile_clouds, nk::Global, &mut pool)?;
        let tile_scores = &mut scores[..query_count * rows.len()];
        match metric {
            ShapeMetric::Chamfer => packed_queries.chamfers_parallel(&packed_tile, tile_scores, None, &mut pool)?,
            ShapeMetric::Hausdorff => packed_queries.hausdorffs_parallel(&packed_tile, tile_scores, None, &mut pool)?,
        }
        fold_tile(
            tile_scores,
            rows.len(),
            tile_start,
            &mut rankings,
            f32::to_bits,
            &mut pool,
        )?;
        progress.set_position((tile_start + rows.len()) as u64);
    }
    progress.finish_and_clear();
    land_ranking(&best, top_k, &mut ground_truth);
    Ok(())
}

/// Every tile of at most `tile_rows` rows across parts of `part_rows` rows each, as the tile's
/// first row across all parts, its part, and its rows within that part.
fn tiles(
    part_rows: impl Iterator<Item = usize>,
    tile_rows: usize,
) -> impl Iterator<Item = (usize, usize, Range<usize>)> {
    part_rows
        .enumerate()
        .flat_map(move |(part, rows)| row_ranges(0..rows, tile_rows).map(move |rows| (part, rows)))
        .scan(0, |tile_start, (part, rows)| {
            let first = *tile_start;
            *tile_start += rows.len();
            Some((first, part, rows))
        })
}

/// The `top_k` the ground truth's width names, checked against the base, the queries and the
/// 32-bit IDs it holds.
fn checked_top_k(
    base_count: usize,
    query_count: usize,
    ground_truth: &MatrixSpan<'_, u32>,
) -> Result<usize, GroundTruthError> {
    if base_count > 1 << 32 {
        return Err(GroundTruthError::IndexOverflow { base_count });
    }
    if ground_truth.shape()[0] != query_count {
        return Err(GroundTruthError::RowCountMismatch {
            ground_truth_rows: ground_truth.shape()[0],
            query_rows: query_count,
        });
    }
    let top_k = ground_truth.shape()[1];
    if top_k == 0 || top_k > base_count {
        return Err(GroundTruthError::TopKTooLarge { top_k, base_count });
    }
    Ok(top_k)
}

fn pool_of(threads: Threads) -> Result<ThreadPool, GroundTruthError> {
    spawn_pool(threads).map_err(|error| GroundTruthError::ThreadPool(format!("{error}")))
}

fn ground_truth_progress(base_count: usize, metric: &str) -> indicatif::ProgressBar {
    let progress = indicatif::ProgressBar::new(base_count as u64);
    progress.set_style(
        indicatif::ProgressStyle::with_template("  {msg} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, ETA {eta})")
            .unwrap()
            .progress_chars("##-"),
    );
    progress.set_message(format!("ground truth ({metric})"));
    progress
}

/// Folds one `[queries, tile_count]` score tile into the queries' rankings, each sorted ascending
/// by `(key, base index)`, one query per task.
fn fold_tile<Distance: Copy + Sync, Key: Ord + Copy + Send + Sync>(
    scores: &[Distance],
    tile_count: usize,
    tile_start: usize,
    rankings: &mut [&mut [(Key, u32)]],
    key_of: impl Fn(Distance) -> Key + Sync,
    pool: &mut ThreadPool,
) -> Result<(), GroundTruthError> {
    rankings
        .into_par_iter()
        .with_pool(pool)
        .for_each(|ranking, query, _| {
            let last = ranking.len() - 1;
            for (offset, &score) in scores[query * tile_count..][..tile_count].iter().enumerate() {
                let entry = (key_of(score), (tile_start + offset) as u32);
                if entry >= ranking[last] {
                    continue;
                }
                let mut position = last;
                while position > 0 && entry < ranking[position - 1] {
                    ranking[position] = ranking[position - 1];
                    position -= 1;
                }
                ranking[position] = entry;
            }
        })
        .map_err(|error| GroundTruthError::ThreadPool(format!("{error}")))
}

/// Lands every ranking's base indices in `ground_truth`, row by row.
fn land_ranking<Key>(best: &[(Key, u32)], top_k: usize, ground_truth: &mut MatrixSpan<'_, u32>) {
    let rows = ground_truth
        .as_contiguous_slice_mut()
        .expect("ground-truth rows are contiguous");
    for (row, ranking) in rows.chunks_exact_mut(top_k).zip(best.chunks_exact(top_k)) {
        for (slot, &(_, index)) in row.iter_mut().zip(ranking) {
            *slot = index;
        }
    }
}

/// Writes the BigANN-style header every `.fbin`, `.u8bin`, `.i8bin`, `.b1bin` and `.ibin` file
/// opens with: the row count and the width, as little-endian `u32` values.
pub fn write_bin_header(writer: &mut impl Write, rows: usize, width: usize) -> io::Result<()> {
    let field =
        |value: usize| u32::try_from(value).map_err(|_| io::Error::other(format!("{value} overflows a u32 header")));
    writer.write_all(&field(rows)?.to_le_bytes())?;
    writer.write_all(&field(width)?.to_le_bytes())
}

/// Writes a whole BigANN-style file: its header, then `body`.
pub fn write_bin_file(path: &Path, rows: usize, width: usize, body: &[u8]) -> io::Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    write_bin_header(&mut writer, rows, width)?;
    writer.write_all(body)?;
    writer.flush()?;
    eprintln!("Wrote {}", path.display());
    Ok(())
}

/// Samples `query_count` rows of the bit-packed `base` as queries with `seed`, computes their exact
/// Hamming top-`top_k` over every base shard, and writes `query.<count>.b1bin` and
/// `groundtruth.<count>.ibin` into `output`.
pub fn write_sampled_ground_truth(
    base: &Dataset,
    query_count: usize,
    top_k: usize,
    seed: Seed,
    threads: Threads,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let query_count = query_count.min(base.rows());
    let mut random = StdRng::seed_from_u64(seed.into());
    let sampled = sample_without_replacement(&mut random, base.rows(), query_count);
    let mut query_bytes = Vec::with_capacity_in(query_count * base.vector_bytes(), System);
    query_bytes.resize(query_count * base.vector_bytes(), 0u8);
    let Vectors::B1x8(queries) = base.gather(sampled.iter(), base.dimensions(), &mut query_bytes) else {
        return Err("sampled ground truth needs a bit-packed base".into());
    };
    let mut shards = Vec::new_in(System);
    for shard in base.shards() {
        let Vectors::B1x8(shard) = shard else {
            return Err("sampled ground truth needs a bit-packed base".into());
        };
        shards.push(shard);
    }

    eprintln!("Computing brute-force hamming top-{top_k} ground truth (NumKong + ForkUnion)...");
    let mut ground_truth = nk::Matrix::full(&[query_count, top_k], 0u32)?;
    compute_top_k(&shards, queries, ground_truth.span(), threads)?;

    let query_path = output.join(format!("query.{query_count}.b1bin"));
    write_bin_file(&query_path, query_count, base.dimensions(), &query_bytes)?;
    let ground_truth_path = output.join(format!("groundtruth.{query_count}.ibin"));
    // SAFETY: `u32` is POD.
    let ground_truth_bytes = unsafe { pod_slice_as_bytes(ground_truth.as_slice()) };
    write_bin_file(&ground_truth_path, query_count, top_k, ground_truth_bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rand::{rngs::StdRng, RngExt, SeedableRng};

    use super::{compute_shape_top_k_tiled, compute_top_k_tiled, nk, Clouds, ShapeMetric};
    use crate::{binary_view, matrix_view, Threads};

    const BASE_COUNT: usize = 300;
    const QUERY_COUNT: usize = 20;
    const TOP_K: usize = 10;
    const PART_ROWS: usize = 137;

    fn two_threads() -> Threads {
        Threads(NonZeroUsize::new(2).unwrap())
    }

    /// Runs the tiled top-K across several base tiles, the last tile short,
    /// and returns each query's neighbor ids.
    fn top_k<Metric: crate::packed_distance::PackedDistance>(
        base: &[nk::MatrixView<'_, Metric>],
        queries: nk::MatrixView<'_, Metric>,
    ) -> nk::Matrix<u32> {
        let mut ground_truth = nk::Matrix::full(&[QUERY_COUNT, TOP_K], 0u32).unwrap();
        compute_top_k_tiled(base, queries, ground_truth.span(), 64, two_threads()).unwrap();
        ground_truth
    }

    /// Ascending `(distance, index)` order of `distances`, cut to `TOP_K`.
    fn naive_top_k<Distance: PartialOrd + Copy>(distances: &[Distance]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..distances.len()).collect();
        order.sort_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap().then(a.cmp(&b)));
        order.truncate(TOP_K);
        order
    }

    fn found_ids(ground_truth: &nk::Matrix<u32>) -> impl Iterator<Item = Vec<usize>> + '_ {
        let (rows, _) = ground_truth.as_slice().as_chunks::<TOP_K>();
        rows.iter().map(|row| row.iter().map(|&id| id as usize).collect())
    }

    #[test]
    fn f32_top_k_matches_scalar_l2_at_tail_widths() {
        for dimensions in [2, 3, 7, 20, 64, 100] {
            let mut random = StdRng::seed_from_u64(dimensions as u64);
            let base: Vec<f32> = (0..BASE_COUNT * dimensions)
                .map(|_| random.random_range(-1.0..1.0))
                .collect();
            let queries: Vec<f32> = (0..QUERY_COUNT * dimensions)
                .map(|_| random.random_range(-1.0..1.0))
                .collect();
            let ground_truth = top_k(
                &[matrix_view(&base, BASE_COUNT, dimensions)],
                matrix_view(&queries, QUERY_COUNT, dimensions),
            );
            for (query, found) in queries.chunks_exact(dimensions).zip(found_ids(&ground_truth)) {
                let distances: Vec<f64> = base
                    .chunks_exact(dimensions)
                    .map(|row| {
                        row.iter()
                            .zip(query)
                            .map(|(&b, &q)| (b as f64 - q as f64).powi(2))
                            .sum()
                    })
                    .collect();
                for (&id, &want) in found.iter().zip(&naive_top_k(&distances)) {
                    let (got, want) = (distances[id], distances[want]);
                    assert!(
                        (got - want).abs() <= 1e-9 * want.max(1.0),
                        "dims {dimensions}: {got} != {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn b1bin_top_k_matches_scalar_hamming() {
        for bits in [8, 24, 136] {
            let bytes = bits / 8;
            let mut random = StdRng::seed_from_u64(bits as u64);
            let base: Vec<u8> = (0..BASE_COUNT * bytes).map(|_| random.random()).collect();
            let queries: Vec<u8> = (0..QUERY_COUNT * bytes).map(|_| random.random()).collect();
            // Two base parts, split off a tile boundary, so ids carry on across the parts.
            let (head, tail) = base.split_at(PART_ROWS * bytes);
            let ground_truth = top_k(
                &[
                    binary_view(head, PART_ROWS, bits),
                    binary_view(tail, BASE_COUNT - PART_ROWS, bits),
                ],
                binary_view(&queries, QUERY_COUNT, bits),
            );
            for (query, found) in queries.chunks_exact(bytes).zip(found_ids(&ground_truth)) {
                let distances: Vec<u32> = base
                    .chunks_exact(bytes)
                    .map(|row| row.iter().zip(query).map(|(&b, &q)| (b ^ q).count_ones()).sum())
                    .collect();
                assert_eq!(found, naive_top_k(&distances), "bits {bits}");
            }
        }
    }

    #[test]
    fn shape_top_k_matches_unpacked_tiles() {
        let mut random = StdRng::seed_from_u64(7);
        let counts: Vec<u32> = (0..BASE_COUNT + QUERY_COUNT)
            .map(|_| random.random_range(3..40))
            .collect();
        let points: Vec<[nk::f16; 3]> = (0..counts.iter().sum::<u32>())
            .map(|_| std::array::from_fn(|_| nk::f16::from_f32(random.random_range(-4.0f32..4.0))))
            .collect();
        let points_of = |clouds: &[u32]| clouds.iter().sum::<u32>() as usize;
        let (base_points, query_points) = points.split_at(points_of(&counts[..BASE_COUNT]));
        let base = Clouds {
            points: base_points,
            counts: &counts[..BASE_COUNT],
        };
        let queries = Clouds {
            points: query_points,
            counts: &counts[BASE_COUNT..],
        };
        let base_clouds: Vec<&[[nk::f16; 3]]> = base.each().collect();
        let query_clouds: Vec<&[[nk::f16; 3]]> = queries.each().collect();
        let packed_base = nk::ShapesPackedMatrix::new(&base_clouds).unwrap();
        let packed_queries = nk::ShapesPackedMatrix::new(&query_clouds).unwrap();
        let (head_points, tail_points) = base_points.split_at(points_of(&counts[..PART_ROWS]));
        let parts = [
            Clouds {
                points: head_points,
                counts: &counts[..PART_ROWS],
            },
            Clouds {
                points: tail_points,
                counts: &counts[PART_ROWS..BASE_COUNT],
            },
        ];
        for metric in [ShapeMetric::Chamfer, ShapeMetric::Hausdorff] {
            let mut distances = vec![0.0f32; QUERY_COUNT * BASE_COUNT];
            match metric {
                ShapeMetric::Chamfer => packed_queries.chamfers(&packed_base, &mut distances, None).unwrap(),
                ShapeMetric::Hausdorff => packed_queries.hausdorffs(&packed_base, &mut distances, None).unwrap(),
            }
            let mut ground_truth = nk::Matrix::full(&[QUERY_COUNT, TOP_K], 0u32).unwrap();
            compute_shape_top_k_tiled(&parts, queries, metric, ground_truth.span(), 64, two_threads()).unwrap();
            let (rows, _) = distances.as_chunks::<BASE_COUNT>();
            for (row, found) in rows.iter().zip(found_ids(&ground_truth)) {
                assert_eq!(found, naive_top_k(row), "{metric:?}");
            }
        }
    }
}
