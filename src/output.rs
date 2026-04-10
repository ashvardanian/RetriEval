use serde::Serialize;
use std::io::Write;

/// Machine descriptor emitted once at the start of a benchmark run.
#[derive(Debug, Serialize)]
pub struct MachineInfo {
    pub phase: &'static str,
    pub cpu_model: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub sockets: usize,
    pub numa_nodes: usize,
    pub ram_bytes: u64,
}

/// A single measurement step emitted as one JSON line.
#[derive(Debug, Serialize)]
pub struct StepRecord {
    pub description: String,
    pub phase: String,
    pub vectors_indexed: usize,
    pub vectors_total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vectors_per_second: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queries_per_second: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_10: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ndcg_at_10: Option<f64>,
}

/// Collects machine info using the `sysinfo` crate.
pub fn collect_machine_info() -> MachineInfo {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_model = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let physical_cores = sys.physical_core_count().unwrap_or(0);
    let logical_cores = sys.cpus().len();
    let ram_bytes = sys.total_memory();

    // Socket and NUMA detection — best-effort from /sys on Linux
    let sockets = detect_sockets();
    let numa_nodes = detect_numa_nodes();

    MachineInfo {
        phase: "machine",
        cpu_model,
        physical_cores,
        logical_cores,
        sockets,
        numa_nodes,
        ram_bytes,
    }
}

fn detect_sockets() -> usize {
    // Read from /sys/devices/system/cpu/cpu0/topology/physical_package_id range
    std::fs::read_dir("/sys/devices/system/node/")
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("node"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(1)
}

fn detect_numa_nodes() -> usize {
    std::fs::read_dir("/sys/devices/system/node/")
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("node"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(1)
}

/// Write a single JSON line to the output.
pub fn emit<W: Write, T: Serialize>(writer: &mut W, record: &T) -> std::io::Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writeln!(writer)?;
    writer.flush()
}
