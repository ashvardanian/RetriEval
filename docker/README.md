# Docker

Qdrant, Redis, Weaviate and SurrealDB run as Docker containers that each `retri-eval-*` binary spawns and tears down itself.
The compose files here mirror the port bindings and environment variables hard-coded in those binaries, for starting a server by hand.

## Lifecycle

`src/docker.rs` wraps `bollard`, the async Docker API client, and drives every container through the same steps:

1. __Pull__ the pinned image if it is not cached locally.
2. __Run__ it with the binary's port bindings and environment variables.
3. __Wait__ until it answers: Qdrant's `/healthz`, Weaviate's `/v1/.well-known/ready`, SurrealDB's `/health`, or a TCP connect followed by a successful PING for Redis, bounded by `--startup-time-limit`.
4. __Benchmark__ against the container.
5. __Stop and remove__ it when the backend owning its handle is dropped.
   Startup failures before ownership transfers, forced termination or aborted processes can leave a container for manual removal.

Per-step memory comes from the Docker stats API: the container's resident set, including the engine process, its heap and the page cache attributed to it.
That overcounts the index alone, but it is what the engine costs to run.

The Docker daemon must be reachable over its Unix socket or TCP, and the images pullable by the current user.
Where the daemon runs as root, add your user to the `docker` group or run the benchmark with `sudo`.

## Server Caveats

Redis 8 is required for `i8`, `u8`, `f16` and `bf16`, as the older `redis/redis-stack` images on Redis 7.4 reject those types at `FT.CREATE`.
Qdrant's server-side `Float16` and `Uint8` accept `f32` upserts and convert on ingest, so the wire payload is unchanged.
Weaviate stores only `f32` internally, so its wire-type sweep is intentionally a single option.
SurrealDB uses in-memory storage, and `retri-eval-surrealdb` checks that `EXPLAIN` names the HNSW index before measuring queries.
Its inserts send native arrays, and each search batch shares one HTTP request.
