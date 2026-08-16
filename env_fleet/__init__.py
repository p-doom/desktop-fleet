"""env-fleet: Slurm-native, capacity-aware fleet management for verifiers env-servers.

The scarce resource is an externally-leased machine (a QEMU desktop VM behind an
apptainer container behind a Slurm allocation). env-fleet leases them, registers
them, counts how many are actually warm, routes work only to replicas that have
a free one, and restarts whatever loses theirs. It never sees an action, a
grammar, or a reward.

No consumer-specific coupling belongs anywhere but ``env_fleet.adapters``.
"""

from __future__ import annotations

from env_fleet.readiness import (
    ReadinessSummary,
    active_worker_statuses,
    read_statuses,
    readiness_summary,
    resolve_min_ready,
    resolve_status_dir,
    stale_worker_statuses,
    sum_int_field,
)
from env_fleet.registry import (
    EnvFleetRegistry,
    read_registry,
    read_registry_if_ready,
    upsert_registry,
)
from env_fleet.slurm import (
    SlurmJob,
    confirm_cancel,
    parse_squeue,
    query_squeue,
    run_command,
    select_cancel_job,
    slurm_job_id_from_registry,
    slurm_metadata,
    slurm_node_addrs,
)
from env_fleet.spec import (
    EnvServerSpec,
    FleetRunLayout,
    default_public_host,
    load_runtime_env_file,
    make_server_specs,
    parse_consumer_paths,
    parse_node_addr,
    render_consumer_paths,
    require_absolute_path,
    scratch_root,
    to_toml,
    toml_literal,
    write_env_server_config,
)

__all__ = [
    "EnvFleetRegistry",
    "EnvServerSpec",
    "FleetRunLayout",
    "ReadinessSummary",
    "SlurmJob",
    "active_worker_statuses",
    "confirm_cancel",
    "default_public_host",
    "load_runtime_env_file",
    "make_server_specs",
    "parse_consumer_paths",
    "parse_node_addr",
    "parse_squeue",
    "query_squeue",
    "read_registry",
    "read_registry_if_ready",
    "read_statuses",
    "readiness_summary",
    "render_consumer_paths",
    "require_absolute_path",
    "resolve_min_ready",
    "resolve_status_dir",
    "run_command",
    "scratch_root",
    "select_cancel_job",
    "slurm_job_id_from_registry",
    "slurm_metadata",
    "slurm_node_addrs",
    "stale_worker_statuses",
    "sum_int_field",
    "to_toml",
    "toml_literal",
    "upsert_registry",
    "write_env_server_config",
]
