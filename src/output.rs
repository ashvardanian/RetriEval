use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;

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

/// Dataset descriptor.
#[derive(Debug, Serialize)]
pub struct DatasetInfo {
    pub vectors_path: String,
    pub queries_path: String,
    pub neighbors_path: String,
    pub vectors_count: usize,
    pub queries_count: usize,
    pub dimensions: usize,
    pub neighbors_per_query: usize,
}

/// One measurement step combining add + search results.
///
/// Fields under "Optional perf counters" are populated only when the benchmark
/// was run with `--features perf-counters` on Linux AND the caller has
/// `CAP_PERFMON` (or `kernel.perf_event_paranoid ≤ 1`). They are serde-skipped
/// when absent, so historical reports parse against the updated struct
/// without schema changes.
#[derive(Debug, Serialize)]
pub struct StepEntry {
    pub vectors_indexed: usize,
    pub add_elapsed: f64,
    pub add_throughput: u64,
    pub memory_bytes: u64,
    pub search_elapsed: f64,
    pub search_throughput: u64,
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub recall_at_1_normalized: f64,
    pub recall_at_10_normalized: f64,
    pub ndcg_at_10_normalized: f64,

    // Optional perf counters — summed across all online CPUs, system-wide
    // for the duration of the add-loop (…_add) or search-loop (…_search).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycles_add: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions_add: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_misses_add: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_misses_add: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cycles_search: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions_search: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_misses_search: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_misses_search: Option<u64>,
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
    let sorted: std::collections::BTreeMap<&String, &Value> = config.iter().collect();
    format!("{sorted:?}").hash(&mut hasher);
    format!("{:06x}", hasher.finish() & CONFIG_HASH_MASK)
}

/// Keep the low 24 bits of the hash — 6 hex digits, ~16M distinct file names.
/// 6 chars is short enough to paste in commit messages without wrapping but
/// wide enough to avoid collisions across the few hundred config permutations
/// a typical benchmark sweep produces.
const CONFIG_HASH_MASK: u64 = 0xFF_FFFF;
