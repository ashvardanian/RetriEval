//! Typed benchmark reports, machine information, and JSON serialization.

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use serde::Serialize;
use serde_json::Value;
use sysinfo::System;

use crate::perf_counters::CounterSample;

/// Machine descriptor.
#[derive(Debug, Serialize)]
pub struct MachineInfo {
    pub cpu_model: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub ram_bytes: u64,
    pub budget: Budget,
}

/// The cores and memory every engine gets: this process's affinity mask and cgroup limit, which containers inherit.
#[derive(Clone, Debug, Serialize)]
pub struct Budget {
    pub cores: usize,
    /// The affinity mask as a cpuset list, like `0-15`, for `docker create --cpuset-cpus`.
    pub cpus: String,
    /// Bytes the engine may use, page cache included.
    pub memory_limit: u64,
}

static BUDGET: OnceLock<Budget> = OnceLock::new();

impl Budget {
    /// The environment's budget, narrowed by `--memory-limit`, which may not exceed the cgroup's limit.
    pub fn resolve(memory_limit: Option<u64>) -> Result<Self, String> {
        let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let cpus = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map_or_else(|| format!("0-{}", cores - 1), |list| list.trim().to_owned());
        let environment = cgroup_memory_limit().unwrap_or_else(|| System::new_all().total_memory());
        let memory_limit = match memory_limit {
            Some(bytes) if bytes > environment => {
                return Err(format!(
                    "--memory-limit {bytes} exceeds the {environment} bytes this process may use"
                ))
            }
            Some(bytes) => bytes,
            None => environment,
        };
        Ok(Self {
            cores,
            cpus,
            memory_limit,
        })
    }
    /// Fixes the run's budget; later calls keep the first.
    pub fn install(self) -> &'static Self {
        BUDGET.get_or_init(|| self)
    }
}

/// The run's budget, or the environment's when no run installed one, as in tests.
pub fn budget() -> &'static Budget {
    BUDGET.get_or_init(|| Budget::resolve(None).expect("the environment's budget needs no flag"))
}

/// The tightest `memory.max` from this process's cgroup up to the root, or `None` when none is set.
fn cgroup_memory_limit() -> Option<u64> {
    let membership = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut path = PathBuf::from("/sys/fs/cgroup");
    path.push(
        membership
            .lines()
            .find_map(|line| line.strip_prefix("0::"))?
            .trim_start_matches('/'),
    );
    path.ancestors()
        .take_while(|directory| directory.starts_with("/sys/fs/cgroup"))
        .filter_map(|directory| {
            std::fs::read_to_string(directory.join("memory.max"))
                .ok()?
                .trim()
                .parse()
                .ok()
        })
        .min()
}

/// Dataset descriptor — describes the input files, not the configuration.
#[derive(Debug, Clone, Serialize)]
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
    /// The backend's own figure: a server container's anonymous bytes, or an in-process engine's
    /// index size or anonymous bytes.
    pub memory_bytes: u64,
    /// This harness process's anonymous resident bytes, which hold any in-process engine whole.
    pub process_anon_bytes: u64,
    pub add: Option<StepAddEntry>,
    /// Refinement passes run after this step's inserts, before its searches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refine: Option<StepRefineEntry>,
    pub ground_truth_search: Option<StepSearchEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_search: Option<StepSearchEntry>,
}

/// The insertion half of a step.
#[derive(Debug, Serialize)]
pub struct StepAddEntry {
    /// Seconds of inserts plus the wait until the backend finished indexing them.
    pub elapsed: f64,
    pub throughput: u64,
    #[serde(flatten)]
    pub counters: Option<CounterSample>,
}

/// The refinement half of a step: passes, their seconds, and the rows each pass re-wired.
#[derive(Debug, Serialize)]
pub struct StepRefineEntry {
    pub passes: usize,
    pub elapsed: f64,
    pub changed: Vec<u64>,
}

/// One pass of a query set against the index, timed and scored. The same shape
/// serves the supplied ground truth and the self-search, which differ only in
/// where their truth comes from.
#[derive(Debug, Serialize)]
pub struct StepSearchEntry {
    pub queries: usize,
    /// Timed passes over the queries, after one untimed warmup pass; `elapsed` and `throughput` span them all.
    pub passes: usize,
    /// Neighbors requested per query — the k every metric below is taken at.
    pub top_k: usize,
    pub elapsed: f64,
    pub throughput: u64,
    /// The first answer within the true nearest distance: 1-recall@1.
    pub recall_at_1: f64,
    /// Any of the top-k within the true nearest distance, 1-recall@k: FAISS's
    /// `OneRecallAtRCriterion`, USearch's `mean_recall`. Saturates as k grows —
    /// `intersection_at_k` is the discriminating measure at large k.
    pub recall_at_k: f64,
    /// The share of the top-k within the k-th true distance, k-recall@k, order-independent — what
    /// ann-benchmarks and cuVS publish as recall. Absent for a self-search,
    /// whose truth set is a single key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intersection_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ndcg_at_k: Option<f64>,
    /// Hardware counters summed across all online CPUs, present only with `--features
    /// perf-counters` on Linux and `CAP_PERFMON` or `kernel.perf_event_paranoid <= 1`.
    #[serde(flatten)]
    pub counters: Option<CounterSample>,
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
    let system = System::new_all();
    let cpu_model = system
        .cpus()
        .first()
        .map_or_else(|| "unknown".into(), |cpu| cpu.brand().to_string());
    let physical_cores = System::physical_core_count().unwrap_or(0);
    let logical_cores = system.cpus().len();
    let ram_bytes = system.total_memory();

    MachineInfo {
        cpu_model,
        physical_cores,
        logical_cores,
        ram_bytes,
        budget: budget().clone(),
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
    let mut sorted = std::collections::BTreeMap::new();
    sorted.extend(config.iter());
    format!("{sorted:?}").hash(&mut hasher);
    format!("{:06x}", hasher.finish() & CONFIG_HASH_MASK)
}

/// Keep the low 24 bits of the hash — 6 hex digits, ~16M distinct file names.
/// 6 chars is short enough to paste in commit messages without wrapping but
/// wide enough to avoid collisions across the few hundred config permutations
/// a typical benchmark sweep produces.
const CONFIG_HASH_MASK: u64 = 0xFF_FFFF;
