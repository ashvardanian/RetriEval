use std::collections::HashMap;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

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
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_model = sys
        .cpus()
        .first()
        .map(|cpu| cpu.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let physical_cores = sys.physical_core_count().unwrap_or(0);
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
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, json)
}

/// Generate a short hash from config metadata for file naming.
pub fn config_hash(config: &HashMap<String, Value>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    let sorted: std::collections::BTreeMap<&String, &Value> = config.iter().collect();
    format!("{sorted:?}").hash(&mut hasher);
    format!("{:06x}", hasher.finish() & 0xFFFFFF)
}
