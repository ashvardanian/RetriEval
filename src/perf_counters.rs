//! Scoped Linux hardware performance counters for region-level attribution.
//!
//! Lets the benchmark split cycles / instructions / cache-misses / branch-misses
//! across construction vs search phases in the same run, populating optional
//! fields on [`crate::output::StepEntry`]. Equivalent to running
//! `perf stat -a -e cycles,instructions,…` alongside the whole process, but
//! scoped to the add-loop and search-loop separately within each step.
//!
//! ## Scope
//!
//! System-wide per-CPU counters (`pid == -1`, `cpu == i`, one
//! [`KernelCounterGroup`] per online CPU). This captures every thread
//! regardless of when it was spawned — necessary because ForkUnion pool
//! workers are created before the bench loop, so `inherit(true)` on the
//! coordinator wouldn't reach them. Downside: on a shared host the numbers
//! include other tenants' activity.
//!
//! ## Permissions and file descriptor limits
//!
//! Needs `CAP_PERFMON` (or `CAP_SYS_ADMIN`) or `sysctl -w
//! kernel.perf_event_paranoid=0` (or `-1`). Without it [`PerfCounters::new`]
//! returns an `io::Error` and the caller falls back to running without
//! counters.
//!
//! The library we build on (`perf-event2`) creates each counter group with a
//! no-op software event as its leader, so each CPU ends up holding **six file
//! descriptors** (one anchor + five hardware counters). At 192 CPUs that's
//! 1,152 fds — above the 1,024 default `ulimit -n` on most distros. Bump
//! it before running (`ulimit -n 65536`) or expect `EMFILE` on the ~170th
//! CPU's group.
//!
//! ## Portability
//!
//! Only the Linux + `perf-counters` feature path has a real impl; every other
//! target compiles a trivial unit-struct whose [`PerfCounters::new`] returns
//! `io::ErrorKind::Unsupported`. Non-Linux JSON output simply omits the
//! counter fields (they're `skip_serializing_if = "Option::is_none"`).

use serde::Serialize;
use std::io;

#[cfg(all(target_os = "linux", feature = "perf-counters"))]
use perf_event::events::Hardware;
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
use perf_event::{Builder, Group};

/// A set of counters the kernel schedules together and reads atomically.
///
/// This is what Linux calls a *counter group* in `perf_event_open(2)`: one
/// file descriptor per member, all members enabled / disabled / read through
/// the leader fd so they cover identical time windows. Without grouping, the
/// kernel may time-multiplex your counters onto fewer physical PMU registers
/// and ratios like IPC drift.
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
type KernelCounterGroup = Group;

/// A readable handle to one counter inside a [`KernelCounterGroup`].
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
type KernelCounter = perf_event::Counter;

/// Counter readings summed across all online CPUs for one scoped region.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CounterSample {
    pub cycles: u64,
    pub instructions: u64,
    pub cache_references: u64,
    pub cache_misses: u64,
    pub branch_misses: u64,
}

impl CounterSample {
    pub fn ipc(&self) -> f64 {
        self.instructions as f64 / self.cycles.max(1) as f64
    }

    pub fn cache_miss_rate(&self) -> f64 {
        self.cache_misses as f64 / self.cache_references.max(1) as f64
    }
}

/// System-wide hardware performance counters spanning every online CPU.
///
/// Construction opens one [`KernelCounterGroup`] per CPU, each holding five
/// counters (cycles, instructions, cache-refs, cache-misses, branch-misses)
/// read atomically via `PERF_FORMAT_GROUP`. The same [`PerfCounters`] is
/// reused across many `reset_and_enable` / `disable_and_read` pairs to avoid
/// per-step fd churn.
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
pub struct PerfCounters {
    per_cpu: Vec<CpuCounters>,
}

#[cfg(all(target_os = "linux", feature = "perf-counters"))]
struct CpuCounters {
    kernel_group: KernelCounterGroup,
    cycles: KernelCounter,
    instructions: KernelCounter,
    cache_references: KernelCounter,
    cache_misses: KernelCounter,
    branch_misses: KernelCounter,
}

#[cfg(all(target_os = "linux", feature = "perf-counters"))]
impl PerfCounters {
    pub fn new() -> io::Result<Self> {
        let logical_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

        let mut per_cpu = Vec::with_capacity(logical_cpus);
        for cpu in 0..logical_cpus {
            match open_cpu_counters(cpu) {
                Ok(entry) => per_cpu.push(entry),
                // Fail fast on the first CPU — almost always a permission
                // problem, not a hot-unplugged core.
                Err(err) if per_cpu.is_empty() => return Err(err),
                // Later failures are tolerated: some cores may be offline.
                Err(_) => continue,
            }
        }

        if per_cpu.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no online CPUs accepted a perf counter group",
            ));
        }
        Ok(Self { per_cpu })
    }

    pub fn reset_and_enable(&mut self) -> io::Result<()> {
        for entry in &mut self.per_cpu {
            entry.kernel_group.reset()?;
            entry.kernel_group.enable()?;
        }
        Ok(())
    }

    pub fn disable_and_read(&mut self) -> io::Result<CounterSample> {
        let mut sample = CounterSample::default();
        for entry in &mut self.per_cpu {
            entry.kernel_group.disable()?;
            let counts = entry.kernel_group.read()?;
            sample.cycles += counts[&entry.cycles];
            sample.instructions += counts[&entry.instructions];
            sample.cache_references += counts[&entry.cache_references];
            sample.cache_misses += counts[&entry.cache_misses];
            sample.branch_misses += counts[&entry.branch_misses];
        }
        Ok(sample)
    }

    pub fn scope<R>(&mut self, f: impl FnOnce() -> R) -> io::Result<(R, CounterSample)> {
        self.reset_and_enable()?;
        let out = f();
        let sample = self.disable_and_read()?;
        Ok((out, sample))
    }
}

/// Open a five-counter group for `perf stat -a` equivalent capture on one CPU.
#[cfg(all(target_os = "linux", feature = "perf-counters"))]
fn open_cpu_counters(cpu: usize) -> io::Result<CpuCounters> {
    // Leader anchor: a no-op software event created by `Group::builder()`
    // whose sole purpose is to own the group's leader fd. The five real
    // counters below share its settings and are read through the group's
    // atomic read path.
    let mut leader = Group::builder();
    leader.any_pid().one_cpu(cpu).include_kernel();
    let mut kernel_group = leader.build_group()?;

    let add_member = |group: &mut Group, event| -> io::Result<KernelCounter> {
        let mut builder = Builder::new(event);
        builder.any_pid().one_cpu(cpu).include_kernel();
        group.add(&builder)
    };

    let cycles = add_member(&mut kernel_group, Hardware::CPU_CYCLES)?;
    let instructions = add_member(&mut kernel_group, Hardware::INSTRUCTIONS)?;
    let cache_references = add_member(&mut kernel_group, Hardware::CACHE_REFERENCES)?;
    let cache_misses = add_member(&mut kernel_group, Hardware::CACHE_MISSES)?;
    let branch_misses = add_member(&mut kernel_group, Hardware::BRANCH_MISSES)?;

    Ok(CpuCounters {
        kernel_group,
        cycles,
        instructions,
        cache_references,
        cache_misses,
        branch_misses,
    })
}

/// Non-Linux / feature-off stub. `new()` always fails so the struct is never
/// held — the other methods are unreachable in practice.
#[cfg(not(all(target_os = "linux", feature = "perf-counters")))]
pub struct PerfCounters;

#[cfg(not(all(target_os = "linux", feature = "perf-counters")))]
impl PerfCounters {
    pub fn new() -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "perf-counters feature disabled or target_os != linux",
        ))
    }

    pub fn reset_and_enable(&mut self) -> io::Result<()> {
        unreachable!("stub PerfCounters is never constructed")
    }

    pub fn disable_and_read(&mut self) -> io::Result<CounterSample> {
        unreachable!("stub PerfCounters is never constructed")
    }

    pub fn scope<R>(&mut self, _f: impl FnOnce() -> R) -> io::Result<(R, CounterSample)> {
        unreachable!("stub PerfCounters is never constructed")
    }
}
