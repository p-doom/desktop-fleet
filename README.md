# desktop-fleet

`desktop-fleet` runs capacity-aware fleets of `verifiers` environment servers
across Slurm nodes. It manages service discovery, readiness, routing, and
process lifetime; desktop actions and VM behavior remain in the `desktop`
package.

## Environment contract

Every fleet is prepared from one absolute JSON contract path. Version 2 carries
a full environment-server session configuration and one named consumer source
with its own explicit taskset and harness configurations. The loader
requires exact keys, validates the status-directory injection path, and rejects
legacy flat metadata.

Preparation writes the validated contract and run layout into the locked fleet
registry. Each environment-server config receives its own pool status path.

## Slurm lifecycle

`submit` uses the packaged `desktop_fleet/run_env_fleet.sbatch`. The allocation
runs one `launch` task per node. Each task prepares its node's configs and
supervises its environment-server process groups; rank zero also starts the
cross-node gateway. Node-count, replica-count, memory, and readiness mismatches
fail before consumer launch.

The registry and pool status files are the service interface used by `status`,
the readiness gate, and the gateway. Stale status writers do not contribute
capacity.

## Commands

```bash
contract=/absolute/path/to/environment.json

uv run --no-sync python -m desktop_fleet.supervise submit \
    --environment-contract "$contract" \
    --nodes 2 \
    --servers-per-node 8 \
    --workers-per-server 4

uv run --no-sync python -m desktop_fleet.supervise status --run-id <run-id>
uv run --no-sync python -m desktop_fleet.readiness \
    --registry /absolute/run/env_registry.json
uv run --no-sync python -m desktop_fleet.supervise cancel \
    --run-id <run-id> --yes
```

Consumer-specific launch hints live under `desktop_fleet/adapters/`; the core
fleet commands do not depend on an adapter.

## Checks

```bash
pytest
ruff check .
mypy desktop_fleet
```
