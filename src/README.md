# Observability

Every report records wall-clock throughput for each phase and the backend's own memory figure for each step.
This page covers the deeper attribution: hardware counters, an external `perf stat` sidecar, and how each backend measures memory.

## Hardware Counters

Build with `--features perf-counters` to attribute cost inside the measured loops, like how many cycles construction spent in cache misses.
On Linux this pulls [`perf-event2`][perf-event2] and wraps the `index.add` and `index.search` loops of `bench.rs` with system-wide hardware counters.
Five optional fields land on each phase of each step, in `steps[].add`, `steps[].ground_truth_search` and `steps[].self_search`:

```text
cycles / instructions / cache_references / cache_misses / branch_misses
```

They are `Option<u64>` and skipped when absent, so reports from runs without the feature keep the plain schema.

```sh
sudo sysctl -w kernel.perf_event_paranoid=-1   # once per host
ulimit -n 65536                                 # one descriptor per counter per CPU
cargo build --release --features usearch-backend,perf-counters

retri-eval-usearch \
    --base-vectors data/pubchem-maccs/base.115627267.b1bin \
    --query-vectors data/pubchem-maccs/query.10000.b1bin \
    --query-neighbors data/pubchem-maccs/groundtruth.10000.ibin \
    --data-type b1 --metric hamming --output results/pubchem-maccs
```

- __Scope__ is system-wide per CPU, `pid == -1` and `cpu == i`, one counter group per online CPU, summed at read.
  It is the only way to cover every ForkUnion pool thread, since per-process `inherit(true)` would miss workers spawned before the counters were enabled.
  On shared hosts the numbers include other tenants' activity.
- __Permissions__ require `CAP_PERFMON` or `CAP_SYS_ADMIN`, or `kernel.perf_event_paranoid ≤ 0`.
  Without them `PerfCounters::new` returns `EACCES`, the run prints `perf counters: unavailable …; running without` and completes without the counter fields.
- __Descriptors__ run six per CPU, a no-op leader plus five counters, so 192 CPUs need 1,152, above the usual `ulimit -n 1024`.
  Without raising it the run fails with `EMFILE` around the 170th CPU's group.
- __Platforms__ other than Linux never pull `perf-event2`, which is gated behind `[target.'cfg(target_os = "linux")']`.
  There `PerfCounters::new` returns `Unsupported`, and the feature compiles and runs as if disabled.

## Perf Stat Sidecar

To profile an already-built binary, or to see OS-level metrics next to the hardware counters, run `perf stat` and `mpstat` alongside the benchmark instead of rebuilding.

```sh
sudo apt install linux-tools-common linux-tools-generic sysstat
sudo sysctl -w kernel.perf_event_paranoid=-1

mpstat 1 > results/cohere-en/mpstat.txt &        # 1 Hz all-core utilization
perf stat -a -e cycles,instructions,cache-references,cache-misses,\
LLC-load-misses,branch-misses,context-switches,cpu-migrations,page-faults \
    --output results/cohere-en/perf.txt -- \
    retri-eval-usearch \
        --base-vectors data/cohere-en/base.41488110.b1bin \
        --query-vectors data/cohere-en/query.10000.b1bin \
        --query-neighbors data/cohere-en/groundtruth.10000.ibin \
        --data-type b1 --metric hamming \
        --output results/cohere-en
kill %1
```

This covers the whole process lifetime, dataset loading and ground-truth I/O included, which helps spot cost outside the measured loops.

## Memory Accounting

`steps[].memory_bytes` is whatever the backend reports after each step, so the figures differ in what they count:

- __USearch__ reports the index's own `memory_usage`, without the dataset mapping.
- __FAISS__ reports the resident set of the whole process, dataset pages included.
- __cuVS__ computes it: the host copy of the vectors and keys, plus the device vectors and a graph of `--connectivity` edges per vector.
- __Qdrant, Redis, Weaviate and SurrealDB__ sample the Docker stats API, the container's whole resident set; see [`docker/README.md`](../docker/README.md).
- __LanceDB and Turso__ report 0.

The `peak memory` line printed at the end of a run is the largest `memory_bytes` across its steps.

[perf-event2]: https://crates.io/crates/perf-event2
