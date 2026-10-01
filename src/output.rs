//! Typed benchmark reports, machine information, and JSON serialization.

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    path::Path,
};

use serde::Serialize;
use serde_json::Value;
use sysinfo::System;

/// Machine descriptor.
#[derive(Debug, Serialize)]
pub struct MachineInfo {
    pub cpu_model: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub ram_bytes: u64,
}

/// Dataset descriptor — describes the input files, not the configuration.
#[derive(Debug, Serialize)]
pub struct DatasetInfo {
    pub base_vectors_path: String,
    pub query_vectors_path: String,
    pub query_neighbors_path: String,
    /// Rows in the base file, before any `--max-base-vectors` cap. Divide
    /// `steps[].vectors_indexed` by this for the share of the ground truth a
    /// step could possibly have found.
    pub vectors_count: usize,
    pub queries_count: usize,
    /// The input file's vector width. A `--dims` sweep truncates per
    /// config and reports the effective width as `config.dimensions`.
    pub dimensions: usize,
    pub neighbors_per_query: usize,
}

/// One measurement step: how much of the base is in the index, what it cost to
/// put it there, and how the index scored.
///
/// `add` is absent on the `--index` load path, where nothing was inserted.
/// `self_search` is present on at most one step — the last — because it runs
/// once against the finished index.
#[derive(Debug, Serialize)]
pub struct StepEntry {
    pub vectors_indexed: usize,
    pub memory_bytes: u64,
    pub add: Option<StepAddEntry>,
    pub ground_truth_search: Option<StepSearchEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_search: Option<StepSearchEntry>,
}

/// The insertion half of a step.
#[derive(Debug, Serialize)]
pub struct StepAddEntry {
    pub elapsed: f64,
    pub throughput: u64,
    #[serde(flatten)]
    pub counters: PhaseCounters,
}

/// One pass of a query set against the index, timed and scored. The same shape
/// serves the supplied ground truth and the self-search, which differ only in
/// where their truth comes from.
#[derive(Debug, Serialize)]
pub struct StepSearchEntry {
    pub queries: usize,
    /// Neighbors requested per query — the k every metric below is taken at.
    pub top_k: usize,
    pub elapsed: f64,
    pub throughput: u64,
    /// Rank-1 truth found anywhere in the top-k: FAISS's `OneRecallAtRCriterion`,
    /// USearch's `mean_recall`. Saturates as k grows — `intersection_at_k` is the
    /// discriminating measure at large k.
    pub recall_at_1: f64,
    pub recall_at_k: f64,
    /// Set overlap `|top-k ∩ truth-k| / k`, order-independent — what
    /// ann-benchmarks and cuVS publish as recall. Absent for a self-search,
    /// whose truth set is a single key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intersection_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ndcg_at_k: Option<f64>,
    #[serde(flatten)]
    pub counters: PhaseCounters,
}

/// Hardware counters for one phase, summed across all online CPUs. Populated
/// only with `--features perf-counters` on Linux and `CAP_PERFMON` (or
/// `kernel.perf_event_paranoid <= 1`); every field is serde-skipped otherwise.
#[derive(Debug, Default, Serialize)]
pub struct PhaseCounters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycles: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_references: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_misses: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_misses: Option<u64>,
}

impl PhaseCounters {
    pub fn from_sample(sample: Option<&crate::perf_counters::CounterSample>) -> Self {
        match sample {
            None => Self::default(),
            Some(s) => Self {
                cycles: Some(s.cycles),
                instructions: Some(s.instructions),
                cache_references: Some(s.cache_references),
                cache_misses: Some(s.cache_misses),
                branch_misses: Some(s.branch_misses),
            },
        }
    }
}

/// Complete report for one backend configuration.
#[derive(Debug, Serialize)]
pub struct ConfigReport {
    pub machine: MachineInfo,
    pub dataset: DatasetInfo,
    pub config: HashMap<String, Value>,
    pub steps: Vec<StepEntry>,
}

/// Collects machine info using the `sysinfo` crate.
pub fn collect_machine_info() -> MachineInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_model = sys
        .cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let physical_cores = System::physical_core_count().unwrap_or(0);
    let logical_cores = sys.cpus().len();
    let ram_bytes = sys.total_memory();

    MachineInfo {
        cpu_model,
        physical_cores,
        logical_cores,
        ram_bytes,
    }
}

/// Write a ConfigReport as pretty JSON to a file.
pub fn write_report(path: &Path, report: &ConfigReport) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(report).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Generate a short hash from config metadata for file naming.
pub fn config_hash(config: &HashMap<String, Value>) -> String {
    let mut hasher = DefaultHasher::new();
    let mut sorted = std::collections::BTreeMap::new_in(std::alloc::System);
    sorted.extend(config.iter());
    format!("{sorted:?}").hash(&mut hasher);
    format!("{:06x}", hasher.finish() & CONFIG_HASH_MASK)
}

/// Keep the low 24 bits of the hash — 6 hex digits, ~16M distinct file names.
/// 6 chars is short enough to paste in commit messages without wrapping but
/// wide enough to avoid collisions across the few hundred config permutations
/// a typical benchmark sweep produces.
const CONFIG_HASH_MASK: u64 = 0xFF_FFFF;
