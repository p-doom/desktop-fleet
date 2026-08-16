# desktop-fleet

Slurm-native, capacity-aware fleet management for [`verifiers`](https://github.com/willccbb/verifiers)
env-servers whose scarce resource is an externally-leased machine.

desktop-fleet manages machines. It never sees an action, a grammar, or a reward.

## What a worker is here

A worker owns a QEMU desktop VM inside an apptainer container inside a
multi-node Slurm allocation. Startup takes minutes, the VM can die without the
server dying, and the machine is leased from a scheduler that can preempt it.
`verifiers`' own broker binds one `ipc://` socket per worker behind a single
`tcp://127.0.0.1:5000` frontend, so it is single-node by construction. On top of
it desktop-fleet adds:

1. A registry (`desktop_fleet.registry`) — a lock-protected JSON file every node
   upserts into, so a consumer that starts later discovers the whole fleet from
   disk alone.
2. Capacity accounting (`desktop_fleet.readiness`) — worker pools heartbeat one
   status file each; the fleet counts `ready`/`starting`/`leased`, discards stale
   writers, and exposes a blocking readiness gate.
3. A cross-node broker (`desktop_fleet.broker`) — `tcp://` end to end, one DEALER
   per replica on other nodes, and routing keyed on VM-pool capacity read
   from those status files (`available_ready_sessions`,
   `backend_capacity_rank`). Requests queue instead of failing when every leased
   machine is busy; `ipc://` cannot express this.
4. Supervision (`desktop_fleet.supervise`) — restart a replica that lost its
   desktops, reap the orphaned QEMU/apptainer *process groups* it left behind,
   and give up loudly with an `fleet_unrecoverable.json` marker.

## Layout

| module | responsibility |
| --- | --- |
| `desktop_fleet/spec.py` | `EnvServerSpec`, `FleetRunLayout`, path/env helpers, verifiers env-server TOML rendering |
| `desktop_fleet/registry.py` | the durable, `flock`-protected fleet registry |
| `desktop_fleet/slurm.py` | Slurm identity, `NodeAddr` resolution, `squeue`/`scancel` guards |
| `desktop_fleet/readiness.py` | status-file capacity accounting + the readiness gate CLI |
| `desktop_fleet/supervise.py` | `prepare` / `run` / `submit` / `status` / `cancel` |
| `desktop_fleet/broker.py` | cross-node, capacity-aware ZMQ rollout broker |
| `desktop_fleet/adapters/` | the only place a specific consumer may be named |

## The containment rule

`grep -c prime_rl desktop_fleet/*.py` is `0`. Every consumer-specific name, import,
config key, and path lives in `desktop_fleet/adapters/prime_rl.py`. The core reaches
a consumer only through neutral seams:

* `FleetRunLayout.consumer_paths` — an opaque `name -> absolute path` map, fed by
  `--consumer-path NAME=PATH` or `ENV_FLEET_CONSUMER_PATHS`, round-tripped
  through registry `layout` metadata. The adapter names the keys.
* `supervise.main(..., trainer_section_factory=...)` — a passed-in callable that
  renders the consumer's launch block into the submit report.

Import direction is strictly `adapters -> core`.

## Usage

```bash
# submit the fleet service (with the prime-rl launch hints)
uv run --no-sync python -m desktop_fleet.adapters.prime_rl submit --dry-run

# or with no consumer at all
uv run --no-sync python -m desktop_fleet.supervise submit --nodes 2 --servers-per-node 8

# inside the allocation, per node
python -m desktop_fleet.supervise prepare
python -m desktop_fleet.supervise run --config-dir ... --logs-dir ... \
    --registry ... --env-server-bin ... --start-gateway

# block until enough machines are warm
python -m desktop_fleet.readiness --registry "$OSWORLD_ENV_FLEET_REGISTRY"

# inspect / tear down
python -m desktop_fleet.supervise status --run-id <run-id>
python -m desktop_fleet.supervise cancel --run-id <run-id>
```
