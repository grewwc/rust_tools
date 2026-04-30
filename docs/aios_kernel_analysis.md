# `aios_kernel` Crate Analysis

## Positioning

`aios_kernel` is currently closer to an "agent runtime control-plane microkernel" than a full data-plane OS.

It already centralizes several important abstractions:

- process lifecycle and scheduling
- wait/signal semantics
- per-process resource accounting and quota checks
- LLM usage accounting
- VFS access
- daemon registration
- point-to-point result channels

Key references:

- [`crates/aios_kernel/src/kernel.rs`](../crates/aios_kernel/src/kernel.rs)
- [`crates/aios_kernel/src/local.rs`](../crates/aios_kernel/src/local.rs)
- [`crates/aios_kernel/src/primitives.rs`](../crates/aios_kernel/src/primitives.rs)

Upstream runtime integration already exists in several places:

- LLM accounting is wired through `src/bin/ai/request.rs`
- VFS is used by `src/bin/ai/tools/storage/file_store.rs`
- daemon registration is used by `src/bin/ai/driver/reflection/background.rs`
- async coordination now has a reusable internal event-source layer through `EpollOps`, with runtime integrations for channels, futexes, task completion, tool cancellation, and request interruption

## Missing Capabilities

From a modern OS perspective, especially one intended to support agent execution and coordination well, the crate is still missing several important capabilities.

### 1. True async waiting primitives

This area is now partially addressed.

`FutexOps` exists and has a kernel event id, and `EpollOps` provides a reusable interest-set primitive over:

- raw kernel `EventId`s
- IPC channels
- futex value or wake sequence changes

The upper runtime also uses these sources for task waits, async tool waits, cancellation, and request interruption.

What is still missing is not the event-source model itself, but deeper async integration. Some paths still rely on:

- polling
- mailbox wake-up text as the user-facing wake signal
- re-check loops in user land

The remaining gap is something closer to:

- lower-overhead async sleep/wake integration
- fewer user-land re-check loops after wake-up
- a cleaner separation between kernel wake state and user-facing mailbox guidance

In short: the wait model is no longer merely conceptual, but it is still implemented as a synchronous local kernel primitive rather than a fully async executor-backed wait subsystem.

#### Recently Addressed In Code

This review found several items in the document that were still actionable and small enough to fix directly:

- ready-queue wakeups now use ordered insertion instead of draining and sorting the entire queue on each enqueue
- descendant traversal now builds a parent-to-children index once instead of scanning the whole process table once per tree node
- `completed_events` now has a bounded recent-event retention window and keeps older events only when they are still referenced by active waiters, epoll registrations, channels, or futexes
- `notify_events_completed()` no longer clones the entire completed-event set on every wake scan

These changes do not remove the single global kernel mutex or the synchronous trait boundary, but they reduce the hottest avoidable local data-structure costs.

### 2. Hierarchical resource control

Current quota management is primarily per-process.

What is missing for multi-agent orchestration is cgroup-like accounting across:

- process trees
- process groups
- parent/child budget delegation
- aggregate limits for a task family

Without that, child tasks can each remain "legal" while the overall workflow exceeds the intended total budget.

### 3. Memory and context-budget accounting

The kernel tracks:

- turns
- tool calls
- token usage
- cost
- filesystem bytes

But it does not yet track the memory-like resources that matter most for long-running agents, such as:

- mailbox bytes
- channel queue bytes
- shared-memory bytes
- trace buffer growth
- history/context footprint

For an agent OS, these are often more operationally relevant than raw CPU-like counters.

### 4. Persistence and crash recovery

`LocalOS` is fully in-memory.

There is no durable state for:

- process table
- event completion state
- IPC channels
- daemon registry
- restart intentions

That means a process manager restart effectively drops kernel state. For a modern long-lived agent system, checkpointing or journaling is still absent.

### 5. Stronger supervisor semantics

The daemon layer is intentionally described as a registry plus cancel protocol, not a fully hosted supervisor.

Still missing are higher-level service-management features such as:

- restart backoff
- health checks
- failure classification
- dependency ordering
- isolation of failure domains

This is enough for background bookkeeping, but not yet enough for robust long-lived orchestration.

### 6. Observability as a first-class control surface

There is already a trace ring, which is a strong start. The crate also exposes recent reads, drain-by-sequence, head sequence, daemon snapshots, channel snapshots, and epoll snapshots.

What is still missing is a complete observability control plane:

- richer process/channel/daemon introspection APIs
- durable trace export
- tracing as the authoritative diagnostic backend rather than a mirror target

Right now trace and snapshots exist, but they have not fully become the equivalent of a durable `/proc` plus tracing substrate for the agent OS.

### 7. Finer-grained security isolation

`ProcessCapabilities` is currently boolean capability gating.

That is useful, but still coarse compared with what a modern agent OS would likely need:

- path-scoped permissions
- argument-scoped tool permissions
- secret scope isolation
- handle-scoped IPC permissions
- delegated and revocable capabilities

This is enough for basic safety, but not enough for stronger multi-tenant or semi-trusted agent coordination.

## Performance Bottlenecks

The main bottlenecks are structural, not micro-optimizations.

### 1. Global kernel mutex

The biggest bottleneck is the shared kernel handle:

- `SharedKernel = Arc<Mutex<Box<dyn Kernel + Send>>>`

This serializes a large amount of activity through one lock:

- scheduling
- accounting
- IPC
- VFS access
- daemon registration
- trace emission

The worst part is that some operations perform blocking filesystem I/O while holding the kernel lock. That makes the mutex both a contention point and a latency amplifier.

### 2. Ready-queue maintenance

This review replaced repeated full ready-queue resorting with priority-ordered insertion.

This removes the previous drain-and-sort behavior on paths such as:

- process spawn
- wake-up on termination
- tick wake-up
- futex wake
- signal resume

The remaining limitation is that the ready queue is still a `VecDeque`, so insertion is still linear in the number of ready processes. That is acceptable for the current local runtime, but a heap or indexed queue would be the next step if scheduler scale becomes important.

### 3. Repeated full-table scans

Several important operations scan the entire process table:

- `advance_tick()`
- `notify_events_completed()`
- `signal_process_group()`
- `check_daemon_restart()`

`collect_descendants()` used to be particularly expensive because it scanned the entire process table repeatedly during traversal. This review fixed that specific case by building a parent-to-children index once per traversal.

The remaining scans are still workable at small scale but will not scale well for dense parallel subagent workloads.

### 4. Completed-event accumulation

`completed_events` stores recently completed events and is used to avoid lost wake-ups.

This review changed it from an unbounded set into a bounded retention window that preserves older entries only while they are still actively referenced.

That is safer for long sessions, but it still has trade-offs:

- very delayed waits on old external event IDs can miss events after they fall outside the recent-event window
- the set still represents completed state in memory only
- wake paths still scan waiting processes

For long sessions with many async tasks, this is no longer an unbounded memory issue, but it remains an area where a generation-indexed event registry would be stronger.

### 5. Heavy cloning of process state

`Process` is relatively large and includes:

- strings
- env map
- mailbox
- allowed tool set
- history path
- working directory
- resource state

However:

- `pop_ready()`
- `pop_all_ready()`
- `list_processes()`

all clone process values.

Spawning also clones inherited environment and tool policy state. This increases overhead as process metadata gets richer.

### 6. Whole-file VFS design

Current VFS APIs are whole-file oriented:

- `read_to_string`
- `write_all`

This is simple but not ideal for agent workloads such as:

- large codebase exploration
- incremental reading
- chunked streaming
- partial file processing

It causes:

- large in-memory copies
- long lock hold times
- unnecessary work for partial reads

### 7. Trace and shared-memory hot-path overhead

Trace records allocate strings and field maps frequently.

Shared-memory validation recomputes checksums over full values during reads and health checks.

Each individual operation is small, but in high-frequency runtime paths these become persistent background costs.

### 8. Runtime pattern magnifies kernel costs

The upper-layer driver loop does the following repeatedly:

- advance scheduler tick
- pop multiple ready processes
- spawn async tasks
- re-enter kernel to finish each process

So even when work is parallelized at the task level, the kernel-side coordination still funnels through central shared state. That makes current bottlenecks highly visible under concurrency.

## Priority Suggestions

If this crate is intended to evolve into a stronger agent OS substrate, the most valuable next steps are:

### Highest priority

- replace the single global blocking mutex architecture
- avoid blocking file I/O while holding kernel state locks

### Second priority

- index waiters/events more directly
- remove remaining whole-table scans from hot paths
- consider replacing the linear ready queue with a heap or indexed priority queue if process counts grow

### Third priority

- add hierarchical budget and quota control
- add memory/context-oriented accounting dimensions

### Fourth priority

- continue integrating futex/channel/event readiness into runtime paths that still poll or sleep
- reduce mailbox-only wake guidance in favor of structured wait results where practical

### Fifth priority

- strengthen persistence, supervision, and observability so the system can run as a longer-lived agent kernel rather than a transient local scheduler

## Short Summary

`aios_kernel` already captures several of the right OS abstractions for agent systems, and the direction is strong.

Its main current limitation is that it behaves more like a centralized local coordinator than a scalable modern OS substrate.

The biggest gaps are:

- partially implemented, but still incomplete, async/evented coordination
- missing hierarchical control and persistence
- coarse-grained security
- structural lock and scan bottlenecks

The biggest performance problem is not one bad function. It is the combination of:

- one global mutex
- linear ready-queue insertion
- whole-table scanning
- state cloning
- whole-file I/O

The worst avoidable local costs called out in this review have been reduced, but the single shared mutex and synchronous local-kernel boundary are still the dominant architectural constraints as agent parallelism and session duration increase.
