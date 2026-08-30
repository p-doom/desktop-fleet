"""Fleet lifecycle: prepare configs, supervise replicas, submit/inspect/cancel.

* ``prepare``  -- render one verifiers env-server config per replica on this
  node and upsert them into the shared registry.
* ``run``      -- supervise those replicas plus the broker inside the
  allocation, restarting whatever loses its leased machines.
* ``submit`` / ``status`` / ``cancel`` -- operator front end for the Slurm
  service itself.

Nothing here knows which trainer will consume the fleet. The one place a
consumer shows up is the ``trainer_section_factory`` callable threaded through
:func:`main` and :func:`format_submit_report`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from desktop_fleet.environment import (
    ENVIRONMENT_CONTRACT_ENV,
    EnvironmentContract,
    read_environment_contract,
)
from desktop_fleet.local_runtime import (
    LOCAL_RUNTIME_OWNER_FILE,
    LocalRuntimeScope,
    local_runtime_scope,
)
from desktop_fleet.readiness import (
    ReadinessSummary,
    active_worker_statuses,
    read_statuses,
    readiness_summary,
    stale_worker_statuses,
    sum_int_field,
)
from desktop_fleet.registry import (
    EnvFleetRegistry,
    read_registry,
    read_registry_if_ready,
    upsert_registry,
)
from desktop_fleet.slurm import (
    SlurmJob,
    confirm_cancel,
    query_squeue,
    select_cancel_job,
    slurm_job_id_from_registry,
    slurm_memory_gb,
    slurm_metadata,
    slurm_node_addrs,
)
from desktop_fleet.spec import (
    FleetRunLayout,
    default_public_host,
    env_path,
    load_runtime_env_file,
    make_server_specs,
    parse_consumer_paths,
    write_env_server_config,
)

DEFAULT_FLEET_SCRIPT = Path(__file__).with_name("run_env_fleet.sbatch")
DEFAULT_JOB_NAME = "env_fleet"
UV_PYTHON_COMMAND = ("uv", "run", "--no-sync", "python")
FLEET_MODULE = "desktop_fleet.supervise"
READINESS_MODULE = "desktop_fleet.readiness"
BROKER_MODULE = "desktop_fleet.broker"

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

MappingLike = dict[str, Any]
TrainerSectionFactory = Callable[[FleetRunLayout], str]


@dataclass(frozen=True)
class Opt:
    """A CLI flag whose default comes from -- and whose value exports to -- env."""

    flag: str
    kind: Callable[[str], Any]
    default: Any = None
    env: str | None = None
    read: bool = True  # read the default from the environment

    @property
    def attr(self) -> str:
        return self.flag.lstrip("-").replace("-", "_")

    @property
    def env_name(self) -> str:
        return self.env or "ENV_FLEET_" + self.attr.upper()


OPTION_HELP = {
    "fleet_unhealthy_s": "Seconds the whole fleet may stay unhealthy before restart.",
    "gateway_request_timeout_s": "Seconds before an in-flight request is failed.",
    "max_fleet_restarts": "Maximum whole-fleet restarts before giving up.",
    "replica_unhealthy_s": "Seconds a replica may stay unhealthy before restart.",
    "status_stale_after_s": "Seconds before a pool status file is stale.",
}


def add_options(
    parser: argparse.ArgumentParser,
    options: Sequence[Opt],
    env: Mapping[str, str],
) -> None:
    for opt in options:
        default = (
            env_value(env, opt.env_name, opt.kind, opt.default)
            if opt.read
            else opt.default
        )
        parser.add_argument(
            opt.flag,
            type=opt.kind,
            default=default,
            help=OPTION_HELP.get(opt.attr, argparse.SUPPRESS),
        )


def env_value(
    env: Mapping[str, str],
    name: str,
    kind: Callable[[str], Any],
    default: Any,
) -> Any:
    raw = env.get(name)
    if not raw:
        return default
    try:
        return kind(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}={raw!r} is not a valid {kind.__name__}: {exc}"
        ) from exc


def env_int(env: Mapping[str, str], key: str) -> int | None:
    return env_value(env, key, int, None)


def positive_int(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if resolved < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return resolved


def first_env_int(
    env: Mapping[str, str],
    *keys: str,
    default: int | None = None,
) -> int | None:
    for key in keys:
        if (value := env_int(env, key)) is not None:
            return value
    return default


def format_shell_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_json_atomic(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def broker_command(python: str) -> list[str]:
    return [python, "-m", BROKER_MODULE]


PREPARE_OPTIONS: tuple[Opt, ...] = (
    Opt("--bind-host", str, "0.0.0.0", "ENV_FLEET_BIND_HOST"),
    Opt("--base-port", int, 5200, "ENV_FLEET_BASE_PORT"),
    Opt("--node-rank", int, 0, "SLURM_PROCID"),
    Opt("--servers-per-node", positive_int, 1, "ENV_FLEET_SERVERS_PER_NODE"),
    Opt("--workers-per-server", positive_int, 1, "ENV_FLEET_WORKERS_PER_SERVER"),
    Opt("--replica-offset", int, -1, "ENV_FLEET_REPLICA_OFFSET"),
    Opt("--replica-hosts", str, "", "ENV_FLEET_REPLICA_HOSTS"),
    Opt("--gateway-host", str, None),
    Opt("--gateway-bind-host", str, None),
    Opt("--gateway-port", int, 0),
)


def prepare_main(argv: Sequence[str] | None = None) -> int:
    args = parse_prepare_args(argv)
    environment = read_environment_contract(args.environment_contract)
    enforce_vm_memory_budget(
        os.environ,
        servers_per_node=args.servers_per_node,
        workers_per_server=args.workers_per_server,
        min_ready_sessions=environment.session.min_ready_sessions,
    )
    layout = resolve_prepare_layout(args)
    args.run_root = layout.run_root
    args.registry = layout.registry_path
    args.desktop_pool_root = layout.pool_root

    config_dir = layout.node_configs_dir(args.node_rank)
    log_dir = layout.node_logs_dir(args.node_rank)
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    replica_count = args.replica_count
    replica_offset = args.replica_offset
    if replica_offset < 0:
        replica_offset = args.node_rank * args.servers_per_node

    specs = make_server_specs(
        host=args.host,
        bind_host=args.bind_host,
        base_port=args.base_port,
        node_rank=args.node_rank,
        servers_per_node=args.servers_per_node,
        workers_per_server=args.workers_per_server,
        replica_count=replica_count,
        replica_offset=replica_offset,
        name_prefix=environment.source.name,
        config_dir=config_dir,
        log_dir=log_dir,
        pool_status_root=layout.pool_status_dir,
    )

    metadata = registry_metadata(args, layout, replica_count, environment)
    for spec in specs:
        write_env_server_config(spec, environment.session)

    registry = upsert_registry(
        path=layout.registry_path,
        run_id=args.run_id,
        metadata=metadata,
        servers=specs,
    )
    print(json.dumps(registry.as_dict(), indent=2, sort_keys=True))
    return 0


def parse_prepare_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_runtime_env_file()
    env = os.environ
    layout = FleetRunLayout.from_env(env)
    parser = argparse.ArgumentParser(
        description="Prepare verifiers env-server configs for a CPU desktop fleet."
    )
    # Sub-path defaults are the *explicit* environment overrides only, never the
    # values from_env() derived from the environment's run id. Taking the derived
    # ones would pin every sub-path to the environment's run while --run-id named
    # a different one, so `prepare --run-id NEW` wrote NEW's registry, configs and
    # logs into OLD's directories.
    # ``prepare`` is the only command that creates the run's directories and
    # writes its registry, so it is the one that must not accept an unnamed run:
    # two off-scheduler runs would both land on the fallback id, share one
    # registry and one pool status tree, and sum each other's ready counts.
    named_run_id = env.get("ENV_FLEET_RUN_ID") or env.get("SLURM_JOB_ID")
    parser.add_argument("--run-id", default=named_run_id, required=named_run_id is None)
    parser.add_argument("--run-base", type=Path, default=layout.run_base)
    parser.add_argument(
        "--run-root", type=Path, default=env_path(env, "ENV_FLEET_RUN_ROOT")
    )
    parser.add_argument(
        "--registry", type=Path, default=env_path(env, "ENV_FLEET_REGISTRY")
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=env_path(env, "ENV_FLEET_CONFIGS_DIR")
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=env_path(env, "ENV_FLEET_LOGS_DIR")
    )
    parser.add_argument(
        "--pool-status-dir",
        type=Path,
        default=env_path(env, "ENV_FLEET_DESKTOP_POOL_STATUS_DIR"),
    )
    parser.add_argument(
        "--consumer-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Opaque consumer path recorded in registry layout metadata.",
    )
    parser.add_argument(
        "--host", default=env.get("ENV_FLEET_HOST") or default_public_host()
    )
    contract_path = env.get(ENVIRONMENT_CONTRACT_ENV)
    parser.add_argument(
        "--environment-contract",
        type=Path,
        default=Path(contract_path) if contract_path else None,
        required=contract_path is None,
    )
    add_options(parser, PREPARE_OPTIONS, env)
    raw_replica_count = env.get("ENV_FLEET_REPLICA_COUNT")
    replica_count = (
        positive_int(raw_replica_count) if raw_replica_count is not None else None
    )
    parser.add_argument(
        "--replica-count",
        type=positive_int,
        default=replica_count,
        required=replica_count is None,
    )
    parser.add_argument(
        "--desktop-pool-root",
        type=Path,
        default=env_path(env, "ENV_FLEET_DESKTOP_POOL_ROOT"),
    )
    args = parser.parse_args(argv)
    args.consumer_paths = {
        **layout.consumer_paths,
        **parse_consumer_paths(",".join(args.consumer_path)),
    }
    return args


def resolve_prepare_layout(args: argparse.Namespace) -> FleetRunLayout:
    """Build the effective layout after CLI overrides have been parsed."""
    return FleetRunLayout.for_run(
        run_id=args.run_id,
        run_base=args.run_base,
        run_root=args.run_root,
        registry_path=args.registry,
        pool_root=args.desktop_pool_root,
        pool_status_dir=args.pool_status_dir,
        logs_dir=args.logs_dir,
        configs_dir=args.configs_dir,
        consumer_paths=args.consumer_paths,
    )


def registry_metadata(
    args: argparse.Namespace,
    layout: FleetRunLayout,
    replica_count: int,
    environment: EnvironmentContract,
) -> dict[str, Any]:
    """Build the static service contract written to the fleet registry."""
    return {
        "run_id": args.run_id,
        "environment": environment.as_metadata(),
        "gateway": gateway_config(args, replica_count),
        "layout": layout.as_metadata(),
        "expected_env_servers": replica_count,
        "expected_env_workers": replica_count * args.workers_per_server,
        "expected_ready_sessions": expected_ready_sessions(
            workers_per_server=args.workers_per_server,
            replica_count=replica_count,
            min_ready_sessions=environment.session.min_ready_sessions,
        ),
        "slurm": slurm_metadata(os.environ),
    }


def gateway_config(args: argparse.Namespace, replica_count: int) -> dict[str, Any]:
    gateway_port = args.gateway_port or args.base_port + replica_count
    replica_hosts = resolve_replica_hosts(args)
    gateway_host = args.gateway_host or (
        replica_hosts[0] if replica_hosts else args.host
    )
    bind_host = args.gateway_bind_host or args.bind_host
    return {
        "bind_address": f"tcp://{bind_host}:{gateway_port}",
        "public_address": f"tcp://{gateway_host}:{gateway_port}",
        "backend_addresses": backend_addresses(args, replica_count, replica_hosts),
    }


def backend_addresses(
    args: argparse.Namespace,
    replica_count: int,
    replica_hosts: list[str],
) -> list[str]:
    """Return env-server public addresses in replica-index order."""
    addresses: list[str] = []
    for replica_index in range(replica_count):
        node_rank = replica_index // args.servers_per_node
        local_index = replica_index % args.servers_per_node
        host = replica_hosts[node_rank] if node_rank < len(replica_hosts) else args.host
        port = args.base_port + node_rank * args.servers_per_node + local_index
        addresses.append(f"tcp://{host}:{port}")
    return addresses


def resolve_replica_hosts(args: argparse.Namespace) -> list[str]:
    """Resolve one routable host per fleet node."""
    if explicit_hosts := split_csv(args.replica_hosts):
        return explicit_hosts
    if slurm_hosts := slurm_node_addrs(os.environ):
        return slurm_hosts
    return [args.host]


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def enforce_vm_memory_budget(
    env: Mapping[str, str],
    *,
    servers_per_node: int,
    workers_per_server: int,
    min_ready_sessions: int,
) -> None:
    """Refuse a node whose Slurm memory cannot hold the desktops it plans.

    Every warm session is a whole QEMU guest, so the requirement is the product
    of the three counts, not any one of them. Unchecked, the node starts, boots
    desktops until the cgroup OOM-kills one, and reports it as a flaky pool.
    """
    available_gb = slurm_memory_gb(env)
    if available_gb is None:
        return
    vm_mem_gb = int(env.get("ENV_FLEET_DESKTOP_VM_HOST_MEM_GB") or "25")
    planned_vms = servers_per_node * workers_per_server * min_ready_sessions
    required_gb = planned_vms * vm_mem_gb
    if required_gb > available_gb:
        raise ValueError(
            f"this fleet node plans {planned_vms} desktop VMs "
            f"({servers_per_node} servers x {workers_per_server} workers x "
            f"{min_ready_sessions} ready sessions) needing {required_gb} GB "
            f"at {vm_mem_gb} GB per VM, but its Slurm allocation is "
            f"{available_gb} GB; raise ENV_FLEET_SLURM_MEM_PER_NODE / the "
            f"fleet sbatch --mem, lower ENV_FLEET_WORKERS_PER_SERVER or the "
            "contract's min_ready_sessions, or set "
            f"ENV_FLEET_DESKTOP_VM_HOST_MEM_GB for genuinely smaller VMs"
        )


def expected_ready_sessions(
    *,
    workers_per_server: int,
    replica_count: int,
    min_ready_sessions: int,
) -> int:
    """Compute the mandatory warm sessions required before consumer launch."""
    return replica_count * workers_per_server * min_ready_sessions


def launch_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and supervise one node of a Slurm env fleet."
    )
    parser.parse_args(argv)
    env = os.environ
    allocated_nodes = _required_positive_env(env, "ENV_FLEET_ALLOCATED_NODES")
    slurm_nodes = _required_positive_env(
        env,
        "SLURM_JOB_NUM_NODES" if env.get("SLURM_JOB_NUM_NODES") else "SLURM_NNODES",
    )
    if allocated_nodes != slurm_nodes:
        raise ValueError(
            f"fleet allocation has {slurm_nodes} nodes; expected {allocated_nodes}"
        )
    servers_per_node = _required_positive_env(env, "ENV_FLEET_SERVERS_PER_NODE")
    replica_count = _required_positive_env(env, "ENV_FLEET_REPLICA_COUNT")
    expected_replicas = allocated_nodes * servers_per_node
    if replica_count != expected_replicas:
        raise ValueError(
            f"fleet topology requires {expected_replicas} replicas, got {replica_count}"
        )
    node_rank = int(env.get("SLURM_PROCID", "-1"))
    if node_rank < 0 or node_rank >= allocated_nodes:
        raise ValueError(f"SLURM_PROCID is outside the fleet allocation: {node_rank}")

    prepare_main([])

    layout = FleetRunLayout.from_env(env)
    env_server_bin = Path(sys.executable).with_name("env-server")
    if not env_server_bin.is_file():
        raise RuntimeError(f"env-server is missing beside the fleet Python: {env_server_bin}")
    supervisor_args = [
        "--config-dir",
        str(layout.node_configs_dir(node_rank)),
        "--logs-dir",
        str(layout.node_logs_dir(node_rank)),
        "--registry",
        str(layout.registry_path),
        "--env-server-bin",
        str(env_server_bin),
        "--python",
        sys.executable,
        "--run-root",
        str(layout.run_root),
    ]
    if node_rank == 0:
        supervisor_args.extend(
            [
                "--start-gateway",
                "--gateway-log",
                str(layout.logs_dir / "rollout-gateway.stdout.log"),
            ]
        )
    for opt in SUBMIT_OPTIONS:
        supervisor_args.extend(
            [opt.flag, str(env_value(env, opt.env_name, opt.kind, opt.default))]
        )
    return supervise_main(supervisor_args)


def _required_positive_env(env: Mapping[str, str], name: str) -> int:
    value = env.get(name)
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        return positive_int(value)
    except argparse.ArgumentTypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error


@dataclass(frozen=True)
class PoolHealth:
    status_files: int
    active_status_files: int
    stale_status_files: int
    ready: int
    starting: int
    fresh_starting: int
    stale_starting: int
    oldest_starting_age_s: float | None
    leased: int
    total_failed: int
    last_errors: list[str]

    @property
    def usable_capacity(self) -> int:
        return self.ready + self.leased

    @property
    def startup_capacity(self) -> int:
        return self.ready + self.fresh_starting + self.leased


@dataclass(frozen=True)
class SupervisorPolicy:
    poll_s: float
    startup_grace_s: float
    replica_unhealthy_s: float
    failure_window_s: float
    max_failures_per_window: int
    restart_backoff_s: float
    fleet_unhealthy_s: float
    max_fleet_restarts: int
    terminate_timeout_s: float
    status_stale_after_s: float


@dataclass
class ReplicaRuntime:
    name: str
    config_path: Path
    log_path: Path
    status_dir: Path | None
    command: list[str]
    process: subprocess.Popen[Any] | None = None
    started_at: float = 0.0
    restart_count: int = 0
    next_restart_at: float = 0.0
    unhealthy_since: float | None = None
    last_total_failed: int = 0
    failure_window_started_at: float = 0.0
    failures_in_window: int = 0
    last_restart_reason: str | None = None


class StartingSessionSummary(TypedDict):
    fresh: int
    stale: int
    oldest_age_s: float | None


SUPERVISE_POLICY_FLAGS: tuple[tuple[str, Any, Any], ...] = (
    ("--poll-s", float, 5.0),
    ("--startup-grace-s", float, 900.0),
    ("--replica-unhealthy-s", float, 120.0),
    ("--failure-window-s", float, 300.0),
    ("--max-failures-per-window", int, 8),
    ("--restart-backoff-s", float, 10.0),
    ("--fleet-unhealthy-s", float, 300.0),
    ("--max-fleet-restarts", int, 3),
    ("--terminate-timeout-s", float, 30.0),
    ("--status-stale-after-s", float, 120.0),
    ("--gateway-request-timeout-s", float, 900.0),
    ("--gateway-backend-quarantine-s", float, 30.0),
    ("--gateway-capacity-check-interval", float, 5.0),
    ("--gateway-health-check-interval", float, 2.0),
    ("--gateway-health-check-timeout", float, 5.0),
)


def supervise_main(argv: Sequence[str] | None = None) -> int:
    args = parse_supervise_args(argv)
    logging.basicConfig(
        level=LOG_LEVELS[args.log_level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    policy = SupervisorPolicy(
        poll_s=args.poll_s,
        startup_grace_s=args.startup_grace_s,
        replica_unhealthy_s=args.replica_unhealthy_s,
        failure_window_s=args.failure_window_s,
        max_failures_per_window=args.max_failures_per_window,
        restart_backoff_s=args.restart_backoff_s,
        fleet_unhealthy_s=args.fleet_unhealthy_s,
        max_fleet_restarts=args.max_fleet_restarts,
        terminate_timeout_s=args.terminate_timeout_s,
        status_stale_after_s=args.status_stale_after_s,
    )
    return FleetSupervisor(args=args, policy=policy).run()


def parse_supervise_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervise env-server replicas inside one Slurm allocation."
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--env-server-bin", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--start-gateway", action="store_true")
    parser.add_argument("--gateway-log", type=Path)
    for flag, kind, default in SUPERVISE_POLICY_FLAGS:
        parser.add_argument(flag, type=kind, default=default)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def _validate_local_runtime_root(path: Path) -> Path:
    """Accept only the derived job- and node-owned root, and only unsymlinked.

    Every destructive step below goes through here first: a root reached via a
    symlink at any level, or one that is not the path this task derives for
    itself, is somebody else's directory.
    """
    scope = local_runtime_scope(os.environ)
    candidate = Path(os.path.abspath(path))
    if candidate != scope.runtime_root:
        raise RuntimeError(
            "local runtime root must exactly match the derived job- and "
            f"node-owned root ({scope.runtime_root}), got {path}"
        )
    _refuse_symlinked_runtime_scope(scope)
    return candidate


def _refuse_symlinked_runtime_scope(scope: LocalRuntimeScope) -> None:
    for candidate in (scope.job_root, scope.node_root, scope.runtime_root):
        if candidate.is_symlink():
            raise RuntimeError(
                f"Refusing symlink in local runtime ownership path: {candidate}"
            )


def _runtime_owner_path(path: Path) -> Path:
    return path / LOCAL_RUNTIME_OWNER_FILE


def _write_local_runtime_owner(path: Path) -> None:
    validated = _validate_local_runtime_root(path)
    write_json_atomic(
        _runtime_owner_path(validated),
        local_runtime_scope(os.environ).owner_payload,
    )


def _require_local_runtime_owner(path: Path) -> None:
    marker = _runtime_owner_path(path)
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"Refusing to remove unowned local runtime root: {path}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Invalid local runtime ownership marker: {marker}"
        ) from error
    expected = local_runtime_scope(os.environ).owner_payload
    if payload != expected:
        raise RuntimeError(
            f"Local runtime ownership marker does not match this task: {marker}"
        )


def _remove_local_runtime_root(path: Path) -> None:
    validated = _validate_local_runtime_root(path)
    if not validated.exists():
        return
    if not validated.is_dir():
        raise RuntimeError(f"Refusing to remove non-directory runtime root: {validated}")
    _require_local_runtime_owner(validated)
    shutil.rmtree(validated)


class FleetSupervisor:
    def __init__(self, *, args: argparse.Namespace, policy: SupervisorPolicy):
        self.args = args
        self.policy = policy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stop_requested = False
        self.fleet_unhealthy_since: float | None = None
        self.fleet_restart_count = 0
        self.gateway_process: subprocess.Popen[Any] | None = None
        self.gateway_started_at = 0.0
        self.run_root = args.run_root or args.registry.parent
        self.status_path = args.logs_dir / "supervisor_status.json"
        self.unrecoverable_path = self.run_root / "fleet_unrecoverable.json"
        self.local_runtime_root = local_runtime_scope(os.environ).runtime_root
        self.replicas = self.load_replicas()

    def run(self) -> int:
        self.args.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.install_signal_handlers()
        self.prepare_local_runtime_root()

        try:
            self.start_replicas(self.replicas, reason="initial start")
            if self.args.start_gateway:
                self.start_gateway(reason="initial start")

            while not self.stop_requested:
                now = time.monotonic()
                self.monitor_replicas(now)
                self.monitor_gateway()
                if self.maybe_restart_fleet(now):
                    return 42
                self.write_status(now)
                time.sleep(self.policy.poll_s)
        finally:
            self.stop_gateway()
            self.stop_replicas(self.replicas)
            self.write_status(time.monotonic())
            self.cleanup_local_runtime_root()
        return 0

    def prepare_local_runtime_root(self) -> None:
        """Remove stale allocation state before starting any replica."""
        try:
            _remove_local_runtime_root(self.local_runtime_root)
            self.local_runtime_root.mkdir(parents=True)
            _write_local_runtime_owner(self.local_runtime_root)
        except Exception as exc:
            raise RuntimeError(
                "Failed to clean local runtime root before starting replicas: "
                f"{self.local_runtime_root}"
            ) from exc

    def cleanup_local_runtime_root(self) -> None:
        """Best-effort removal after every owned process group has stopped."""
        try:
            _remove_local_runtime_root(self.local_runtime_root)
        except Exception:
            self.logger.exception(
                "Failed to remove local runtime root during shutdown: %s",
                self.local_runtime_root,
            )

    def install_signal_handlers(self) -> None:
        def request_stop(_signum: int, _frame: Any) -> None:
            self.stop_requested = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

    def load_replicas(self) -> list[ReplicaRuntime]:
        registry = read_registry(self.args.registry)
        by_name = {server.name: server for server in registry.servers}
        replicas: list[ReplicaRuntime] = []
        for config_path in sorted(self.args.config_dir.glob("*.toml")):
            name = config_path.stem
            server = by_name.get(name)
            status_dir = (
                Path(server.pool_status_dir)
                if server is not None and server.pool_status_dir
                else None
            )
            replicas.append(
                ReplicaRuntime(
                    name=name,
                    config_path=config_path,
                    log_path=self.args.logs_dir / f"{name}.stdout.log",
                    status_dir=status_dir,
                    command=[self.args.env_server_bin, "@", str(config_path)],
                )
            )
        if not replicas:
            raise RuntimeError(f"no env-server configs found in {self.args.config_dir}")
        return replicas

    def start_replicas(self, replicas: list[ReplicaRuntime], *, reason: str) -> None:
        for replica in replicas:
            self.start_replica(replica, reason=reason)

    def start_replica(self, replica: ReplicaRuntime, *, reason: str) -> None:
        replica.log_path.parent.mkdir(parents=True, exist_ok=True)
        replica.process = self.spawn(replica.command, replica.log_path)
        now = time.monotonic()
        replica.started_at = now
        replica.next_restart_at = now + self.policy.restart_backoff_s
        replica.unhealthy_since = None
        replica.failure_window_started_at = now
        replica.failures_in_window = 0
        replica.last_total_failed = 0
        replica.last_restart_reason = reason
        self.logger.info(
            "Started env-server %s pid=%s reason=%s",
            replica.name,
            replica.process.pid if replica.process else "?",
            reason,
        )

    def spawn(self, command: list[str], log_path: Path) -> subprocess.Popen[Any]:
        """Start a child in its own session so its process group can be reaped."""
        stdout = log_path.open("ab")
        try:
            return subprocess.Popen(
                command,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
                env=os.environ.copy(),
                start_new_session=True,
            )
        finally:
            stdout.close()

    def restart_replica(self, replica: ReplicaRuntime, *, reason: str) -> None:
        if time.monotonic() < replica.next_restart_at:
            return
        self.logger.warning("Restarting env-server %s: %s", replica.name, reason)
        self.stop_replica(replica)
        archive_status_files(replica.status_dir, replica.name)
        replica.restart_count += 1
        self.start_replica(replica, reason=reason)

    def stop_replicas(self, replicas: list[ReplicaRuntime]) -> None:
        for replica in replicas:
            self.stop_replica(replica)

    def stop_replica(self, replica: ReplicaRuntime) -> None:
        terminate_replica_process_group(
            replica.process, timeout_s=self.policy.terminate_timeout_s
        )
        replica.process = None

    def monitor_replicas(self, now: float) -> None:
        for replica in self.replicas:
            health = read_pool_health(
                replica.status_dir,
                status_stale_after_s=self.policy.status_stale_after_s,
            )
            observe_failure_window(replica, health, now=now, policy=self.policy)
            reason = restart_reason(replica, health, now=now, policy=self.policy)
            if reason is not None:
                self.restart_replica(replica, reason=reason)

    def monitor_gateway(self) -> None:
        if not self.args.start_gateway:
            return
        if self.gateway_process is None:
            self.start_gateway(reason="missing process")
            return
        return_code = self.gateway_process.poll()
        if return_code is None:
            return
        self.logger.warning("Rollout gateway exited with code %s", return_code)
        self.start_gateway(reason=f"process exited with code {return_code}")

    def start_gateway(self, *, reason: str) -> None:
        self.stop_gateway()
        log_path = (
            self.args.gateway_log or self.args.logs_dir / "rollout-gateway.stdout.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *broker_command(self.args.python),
            "--registry",
            str(self.args.registry),
            "--request-timeout-s",
            str(self.args.gateway_request_timeout_s),
            "--backend-quarantine-s",
            str(self.args.gateway_backend_quarantine_s),
            "--capacity-check-interval",
            str(self.args.gateway_capacity_check_interval),
            "--status-stale-after-s",
            str(self.args.status_stale_after_s),
            "--health-check-interval",
            str(self.args.gateway_health_check_interval),
            "--health-check-timeout",
            str(self.args.gateway_health_check_timeout),
        ]
        self.gateway_process = self.spawn(command, log_path)
        self.gateway_started_at = time.monotonic()
        self.logger.info(
            "Started rollout gateway pid=%s reason=%s",
            self.gateway_process.pid if self.gateway_process else "?",
            reason,
        )

    def stop_gateway(self) -> None:
        terminate_replica_process_group(
            self.gateway_process, timeout_s=self.policy.terminate_timeout_s
        )
        self.gateway_process = None

    def maybe_restart_fleet(self, now: float) -> bool:
        if any(
            replica_healthy(replica, now=now, policy=self.policy)
            for replica in self.replicas
        ):
            self.fleet_unhealthy_since = None
            return False
        if self.fleet_unhealthy_since is None:
            self.fleet_unhealthy_since = now
            return False
        if now - self.fleet_unhealthy_since < self.policy.fleet_unhealthy_s:
            return False

        self.fleet_restart_count += 1
        if (
            self.policy.max_fleet_restarts >= 0
            and self.fleet_restart_count > self.policy.max_fleet_restarts
        ):
            self.write_unrecoverable_marker(now)
            return True

        self.logger.warning(
            "Restarting whole env fleet after %.1fs without a healthy replica",
            now - self.fleet_unhealthy_since,
        )
        self.stop_gateway()
        self.stop_replicas(self.replicas)
        for replica in self.replicas:
            archive_status_files(replica.status_dir, replica.name)
        self.start_replicas(self.replicas, reason="fleet unhealthy")
        if self.args.start_gateway:
            self.start_gateway(reason="fleet unhealthy")
        self.fleet_unhealthy_since = None
        return False

    def write_unrecoverable_marker(self, now: float) -> None:
        write_json_atomic(
            self.unrecoverable_path,
            {
                "status": "unrecoverable",
                "updated_at": time.time(),
                "fleet_restart_count": self.fleet_restart_count,
                "max_fleet_restarts": self.policy.max_fleet_restarts,
                "unhealthy_for_s": (
                    0.0
                    if self.fleet_unhealthy_since is None
                    else now - self.fleet_unhealthy_since
                ),
                "replicas": self.replica_statuses(now),
            },
        )
        self.logger.error("Fleet marked unrecoverable: %s", self.unrecoverable_path)

    def write_status(self, now: float) -> None:
        write_json_atomic(
            self.status_path,
            {
                "updated_at": time.time(),
                "registry": str(self.args.registry),
                "fleet_restart_count": self.fleet_restart_count,
                "fleet_unhealthy_since": self.fleet_unhealthy_since,
                "gateway": {
                    "enabled": self.args.start_gateway,
                    "pid": self.gateway_process.pid if self.gateway_process else None,
                    "return_code": (
                        self.gateway_process.poll() if self.gateway_process else None
                    ),
                },
                "replicas": self.replica_statuses(now),
            },
        )

    def replica_statuses(self, now: float) -> list[dict[str, Any]]:
        return [
            replica_status(replica, now=now, policy=self.policy)
            for replica in self.replicas
        ]


def read_pool_health(
    status_dir: Path | None,
    *,
    status_stale_after_s: float | None = None,
) -> PoolHealth:
    if status_dir is None:
        return PoolHealth(
            status_files=0,
            active_status_files=0,
            stale_status_files=0,
            ready=0,
            starting=0,
            fresh_starting=0,
            stale_starting=0,
            oldest_starting_age_s=None,
            leased=0,
            total_failed=0,
            last_errors=[],
        )
    statuses = read_statuses(status_dir, recursive=False)
    now = time.time()
    active_statuses = active_worker_statuses(
        statuses,
        now=now,
        stale_after_s=status_stale_after_s,
    )
    stale_statuses = stale_worker_statuses(
        statuses,
        now=now,
        stale_after_s=status_stale_after_s,
    )
    last_errors = [
        str(status["last_error"])
        for status in active_statuses
        if status.get("last_error")
    ]
    starting_details = summarize_starting_sessions(active_statuses, now=now)
    return PoolHealth(
        status_files=len(statuses),
        active_status_files=len(active_statuses),
        stale_status_files=len(stale_statuses),
        ready=sum_int_field(active_statuses, "ready"),
        starting=sum_int_field(active_statuses, "starting"),
        fresh_starting=starting_details["fresh"],
        stale_starting=starting_details["stale"],
        oldest_starting_age_s=starting_details["oldest_age_s"],
        leased=sum_int_field(active_statuses, "leased"),
        total_failed=sum_int_field(active_statuses, "total_failed"),
        last_errors=last_errors[-5:],
    )


def summarize_starting_sessions(
    statuses: Sequence[Mapping[str, Any]],
    *,
    now: float,
) -> StartingSessionSummary:
    """Split ``starting`` desktops into ones still within their startup budget
    and ones that have blown it -- only the latter justify a restart.

    Reads ``startup_timeout_s`` and ``starting_sessions`` without a default: a
    status file missing either cannot be scored, and treating it as within
    budget is the one answer that is never safe -- it is what stops a replica
    whose desktops are permanently stuck starting from ever being restarted.
    """
    fresh = 0
    stale = 0
    oldest_age_s: float | None = None
    for status in statuses:
        startup_timeout_s = float(status["startup_timeout_s"])
        for session in status["starting_sessions"]:
            age_s = max(0.0, now - float(session["created_at"]))
            oldest_age_s = age_s if oldest_age_s is None else max(oldest_age_s, age_s)
            if age_s >= startup_timeout_s:
                stale += 1
            else:
                fresh += 1
    return {"fresh": fresh, "stale": stale, "oldest_age_s": oldest_age_s}


def observe_failure_window(
    replica: ReplicaRuntime,
    health: PoolHealth,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> None:
    if now - replica.failure_window_started_at > policy.failure_window_s:
        replica.failure_window_started_at = now
        replica.failures_in_window = 0
    delta = max(0, health.total_failed - replica.last_total_failed)
    replica.last_total_failed = health.total_failed
    replica.failures_in_window += delta


def restart_reason(
    replica: ReplicaRuntime,
    health: PoolHealth,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> str | None:
    process = replica.process
    if process is None:
        return "process missing"
    return_code = process.poll()
    if return_code is not None:
        return f"process exited with code {return_code}"

    if (
        policy.max_failures_per_window > 0
        and replica.failures_in_window >= policy.max_failures_per_window
    ):
        return (
            f"{replica.failures_in_window} desktop failures within "
            f"{policy.failure_window_s:.1f}s"
        )

    uptime = now - replica.started_at
    if uptime < policy.startup_grace_s:
        replica.unhealthy_since = None
        return None

    if health.active_status_files <= 0:
        reason = "no active desktop-pool worker status files"
    elif health.usable_capacity <= 0:
        if health.stale_starting > 0:
            reason = f"{health.stale_starting} desktop sessions stuck starting" + (
                ""
                if health.oldest_starting_age_s is None
                else f" for up to {health.oldest_starting_age_s:.1f}s"
            )
        elif health.fresh_starting > 0:
            replica.unhealthy_since = None
            return None
        else:
            reason = "no ready or leased desktop sessions"
    else:
        replica.unhealthy_since = None
        return None

    if replica.unhealthy_since is None:
        replica.unhealthy_since = now
        return None
    if now - replica.unhealthy_since >= policy.replica_unhealthy_s:
        return reason
    return None


def replica_healthy(
    replica: ReplicaRuntime,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> bool:
    process = replica.process
    if process is None or process.poll() is not None:
        return False
    if now - replica.started_at < policy.startup_grace_s:
        return True
    health = read_pool_health(
        replica.status_dir,
        status_stale_after_s=policy.status_stale_after_s,
    )
    return health.usable_capacity > 0 or health.fresh_starting > 0


def replica_status(
    replica: ReplicaRuntime,
    *,
    now: float,
    policy: SupervisorPolicy,
) -> dict[str, Any]:
    health = read_pool_health(
        replica.status_dir,
        status_stale_after_s=policy.status_stale_after_s,
    )
    process = replica.process
    return {
        "name": replica.name,
        "pid": process.pid if process else None,
        "return_code": process.poll() if process else None,
        "healthy": replica_healthy(replica, now=now, policy=policy),
        "restart_count": replica.restart_count,
        "last_restart_reason": replica.last_restart_reason,
        "config_path": str(replica.config_path),
        "log_path": str(replica.log_path),
        "status_dir": str(replica.status_dir) if replica.status_dir else None,
        "pool": {
            "status_files": health.status_files,
            "active_status_files": health.active_status_files,
            "stale_status_files": health.stale_status_files,
            "ready": health.ready,
            "starting": health.starting,
            "fresh_starting": health.fresh_starting,
            "stale_starting": health.stale_starting,
            "oldest_starting_age_s": health.oldest_starting_age_s,
            "leased": health.leased,
            "total_failed": health.total_failed,
            "last_errors": health.last_errors,
        },
    }


def terminate_replica_process_group(
    process: subprocess.Popen[Any] | None,
    *,
    timeout_s: float,
) -> None:
    """Kill a replica's whole process group, whether or not the replica is alive.

    ``spawn`` starts every replica with ``start_new_session=True``, so the group
    id *is* the child's pid.  Reading it back with ``getpgid`` instead would fail
    on precisely the case that leaks -- a worker that already died and left its
    VMs behind, which is why this must not return early on an exited child.

    Signalling the pid of a child we have already reaped cannot reach a
    stranger's group: Linux does not recycle a pid that is still in use as the id
    of a process group with live members, and once the group is empty the signal
    is a no-op.  The VMs are in this group because ``desktop`` deliberately does
    not give QEMU a session of its own.
    """
    if process is None:
        return
    group_id = process.pid
    assert group_id != os.getpgrp(), (
        "refusing to signal the supervisor's own process group; "
        "spawn() must keep start_new_session=True"
    )
    with suppress(ProcessLookupError):
        os.killpg(group_id, signal.SIGTERM)
    if not wait_for_process_group_exit(group_id, timeout_s=timeout_s):
        with suppress(ProcessLookupError):
            os.killpg(group_id, signal.SIGKILL)
        wait_for_process_group_exit(group_id, timeout_s=min(timeout_s, 5.0))
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=timeout_s)


def wait_for_process_group_exit(process_group_id: int, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if not process_group_alive(process_group_id):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return False
        time.sleep(min(0.05, remaining_s))


def process_group_alive(process_group_id: int) -> bool:
    return linux_process_group_has_live_members(process_group_id)


def linux_process_group_has_live_members(process_group_id: int) -> bool:
    """Zombie-aware liveness check; ``killpg(0)`` alone reports zombies as alive."""
    proc_dir = Path("/proc")
    assert proc_dir.is_dir(), "this supervisor only ever runs on Slurm/Linux nodes"
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        stat = _linux_process_state_and_group(entry)
        if stat is None:
            continue
        state, member_process_group_id = stat
        if member_process_group_id == process_group_id and state != "Z":
            return True
    return False


def _linux_process_state_and_group(path: Path) -> tuple[str, int] | None:
    try:
        stat_text = (path / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields_start = stat_text.rfind(")")
    fields = stat_text[fields_start + 2 :].split() if fields_start >= 0 else []
    if len(fields) < 3:
        return None
    try:
        return fields[0], int(fields[2])
    except ValueError:
        return None


def archive_status_files(status_dir: Path | None, replica_name: str) -> Path | None:
    if status_dir is None or not status_dir.exists():
        return None
    paths = sorted(status_dir.glob("*.json"))
    if not paths:
        return None
    archive_dir = status_dir / "archive" / time.strftime("%Y%m%d-%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = archive_dir / path.name
        if target.exists():
            target = archive_dir / f"{path.stem}-{replica_name}{path.suffix}"
        path.replace(target)
    return archive_dir


SUBMIT_OPTIONS: tuple[Opt, ...] = (
    Opt("--replica-unhealthy-s", float, 120.0, "ENV_FLEET_SUPERVISOR_REPLICA_UNHEALTHY_S"),
    Opt("--fleet-unhealthy-s", float, 300.0, "ENV_FLEET_SUPERVISOR_FLEET_UNHEALTHY_S"),
    Opt("--max-fleet-restarts", int, 3, "ENV_FLEET_SUPERVISOR_MAX_FLEET_RESTARTS"),
    Opt("--gateway-request-timeout-s", float, 900.0),
    Opt("--status-stale-after-s", float, 120.0),
)

# Fleet-shape exports whose CLI names and env names diverge.
SUBMIT_SHAPE_EXPORTS: tuple[tuple[str, str], ...] = (
    ("run_id", "ENV_FLEET_RUN_ID"),
    ("servers_per_node", "ENV_FLEET_SERVERS_PER_NODE"),
    ("workers_per_server", "ENV_FLEET_WORKERS_PER_SERVER"),
    ("base_port", "ENV_FLEET_BASE_PORT"),
    ("environment_contract", ENVIRONMENT_CONTRACT_ENV),
)


def parse_fleet_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    load_runtime_env_file()
    env = os.environ
    defaults = FleetRunLayout.from_env(env)
    parser = argparse.ArgumentParser(
        description="Manage a long-lived env fleet Slurm service."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit the fleet Slurm job.")
    submit_parser.add_argument("--script", type=Path, default=DEFAULT_FLEET_SCRIPT)
    submit_parser.add_argument("--run-id", default=env.get("ENV_FLEET_RUN_ID"))
    submit_parser.add_argument("--run-base", type=Path, default=defaults.run_base)
    contract_path = env.get(ENVIRONMENT_CONTRACT_ENV)
    submit_parser.add_argument(
        "--environment-contract",
        type=Path,
        default=Path(contract_path) if contract_path else None,
        required=contract_path is None,
    )
    submit_parser.add_argument("--account", default=env.get("SBATCH_ACCOUNT"))
    submit_parser.add_argument(
        "--partition",
        default=env.get("ENV_FLEET_SLURM_PARTITION") or env.get("SBATCH_PARTITION"),
    )
    submit_parser.add_argument(
        "--time",
        default=env.get("ENV_FLEET_SLURM_TIME")
        or env.get("SBATCH_TIMELIMIT")
        or "24:00:00",
    )
    default_nodes = first_env_int(env, "ENV_FLEET_SLURM_NODES", "SBATCH_NODES")
    if default_nodes is not None:
        default_nodes = positive_int(str(default_nodes))
    submit_parser.add_argument(
        "--nodes",
        type=positive_int,
        default=default_nodes,
        required=default_nodes is None,
    )
    submit_parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=first_env_int(
            env,
            "ENV_FLEET_SLURM_CPUS_PER_TASK",
            "SBATCH_CPUS_PER_TASK",
            default=48,
        ),
    )
    submit_parser.add_argument(
        "--mem",
        default=env.get("ENV_FLEET_SLURM_MEM_PER_NODE")
        or env.get("SBATCH_MEM_PER_NODE")
        or "128G",
    )
    submit_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    submit_parser.add_argument("--servers-per-node", type=positive_int, default=8)
    submit_parser.add_argument("--workers-per-server", type=positive_int, default=4)
    submit_parser.add_argument("--base-port", type=int)
    add_options(
        submit_parser,
        SUBMIT_OPTIONS,
        env,
    )
    submit_parser.add_argument("--dry-run", action="store_true")

    status_parser = subparsers.add_parser("status", help="Summarize fleet health.")
    add_layout_args(status_parser, defaults)
    status_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    status_parser.add_argument("--expected-servers", type=int, default=0)
    status_parser.add_argument(
        "--status-stale-after-s",
        type=float,
        default=env_value(env, "ENV_FLEET_STATUS_STALE_AFTER_S", float, 120.0),
    )

    cancel_parser = subparsers.add_parser(
        "cancel", help="Cancel this user's fleet job."
    )
    add_layout_args(cancel_parser, defaults)
    cancel_parser.add_argument("--job-id")
    cancel_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    cancel_parser.add_argument("--yes", action="store_true")

    return parser.parse_args(argv)


def add_layout_args(parser: argparse.ArgumentParser, defaults: FleetRunLayout) -> None:
    parser.add_argument("--run-id", default=defaults.run_id)
    parser.add_argument("--run-base", type=Path, default=defaults.run_base)
    parser.add_argument("--registry", type=Path)


def resolve_trainer_section(
    trainer_section_factory: TrainerSectionFactory | None,
    layout: FleetRunLayout,
) -> str | None:
    """Render the consumer-specific section, or None with no consumer registered.

    The core CLI runs with no ``trainer_section_factory``; an adapter supplies
    one.
    """
    if trainer_section_factory is None:
        return None
    return trainer_section_factory(layout)


def submit(
    args: argparse.Namespace,
    *,
    trainer_section_factory: TrainerSectionFactory | None = None,
) -> int:
    read_environment_contract(args.environment_contract)
    command = build_sbatch_command(args)
    if args.dry_run:
        print(format_shell_command(command))
        trainer_section = resolve_trainer_section(
            trainer_section_factory, dry_run_layout(args)
        )
        if trainer_section is not None:
            print()
            print(
                "Consumer command preview "
                "(replace <run-id> with the submitted fleet job id if unset):"
            )
            print(trainer_section)
        return 0

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    job_id = parse_sbatch_job_id(result.stdout)
    layout = FleetRunLayout.for_run(
        run_id=args.run_id or job_id, run_base=args.run_base
    )
    trainer_section = resolve_trainer_section(trainer_section_factory, layout)
    print(format_submit_report(job_id, layout, args, trainer_section=trainer_section))
    return 0


def build_sbatch_command(args: argparse.Namespace) -> list[str]:
    command = ["sbatch", "--parsable"]
    for option, value in (
        ("--account", args.account),
        ("--partition", args.partition),
        ("--time", args.time),
        ("--nodes", args.nodes),
        ("--cpus-per-task", args.cpus_per_task),
        ("--mem", args.mem),
        ("--job-name", args.job_name),
    ):
        if value is not None:
            command.extend([option, str(value)])

    arg_values = vars(args)
    exports = {
        "ENV_FLEET_RUN_BASE": str(args.run_base),
        "ENV_FLEET_ALLOCATED_NODES": str(args.nodes),
        "ENV_FLEET_REPLICA_COUNT": str(args.nodes * args.servers_per_node),
    }
    for attr, env_name in SUBMIT_SHAPE_EXPORTS:
        if (value := arg_values.get(attr)) is not None:
            exports[env_name] = str(value)
    for opt in SUBMIT_OPTIONS:
        if (value := arg_values.get(opt.attr)) is not None:
            exports[opt.env_name] = str(value)
    rendered_exports = ",".join(f"{key}={value}" for key, value in exports.items())
    command.append(f"--export=ALL,{rendered_exports}")
    command.append(str(args.script))
    return command
def parse_sbatch_job_id(output: str) -> str:
    first = output.strip().splitlines()[0]
    return first.split(";", 1)[0]


def format_submit_report(
    job_id: str,
    layout: FleetRunLayout,
    args: argparse.Namespace,
    *,
    trainer_section: str | None = None,
) -> str:
    def command(*parts: str) -> str:
        return format_shell_command([*UV_PYTHON_COMMAND, *parts])

    lines = [
        f"Submitted env fleet job {job_id}",
        f"Run id: {layout.run_id}",
        f"Registry: {layout.registry_path}",
        f"Status dir: {layout.pool_status_dir}",
        f"Logs: {layout.logs_dir}",
        "",
        "Commands:",
        "",
        "Status:",
        f"  {command('-m', FLEET_MODULE, 'status', '--run-id', layout.run_id)}",
        "",
        "Readiness:",
        f"  {command('-m', READINESS_MODULE, '--registry', str(layout.registry_path))}",
    ]
    if trainer_section is not None:
        lines.extend(["", "Render config and launch the consumer:", trainer_section])
    lines.extend(
        [
            "",
            "Cancel:",
            f"  {command('-m', FLEET_MODULE, 'cancel', '--run-id', layout.run_id)}",
        ]
    )
    return "\n".join(lines)


def dry_run_layout(args: argparse.Namespace) -> FleetRunLayout:
    return FleetRunLayout.for_run(
        run_id=args.run_id or "<run-id>",
        run_base=args.run_base,
    )


def status(args: argparse.Namespace) -> int:
    registry_path = registry_path_for_args(args)
    registry, registry_error = read_registry_for_status(registry_path)
    layout = layout_for_args(args, registry, registry_path=registry_path)
    summary = readiness_summary(
        argparse.Namespace(
            registry=layout.registry_path,
            status_dir=None,
            pool_status_dir=layout.pool_status_dir,
            expected_servers=args.expected_servers,
            status_stale_after_s=args.status_stale_after_s,
        )
    )
    job_id = args.run_id
    if registry is not None:
        job_id = slurm_job_id_from_registry(registry) or job_id
    jobs = query_squeue(job_id=job_id, job_name=args.job_name)
    print(format_status_report(layout, registry, registry_error, summary, jobs))
    return 0


def registry_path_for_args(args: argparse.Namespace) -> Path:
    registry_path = args.registry
    if registry_path is None:
        registry_path = FleetRunLayout.for_run(
            run_id=args.run_id,
            run_base=args.run_base,
        ).registry_path
    return Path(registry_path)


def read_registry_for_status(
    registry_path: Path,
) -> tuple[EnvFleetRegistry | None, str | None]:
    """Keep a missing registry (fleet not up yet) distinct from a corrupt one."""
    if not registry_path.exists():
        return None, f"missing: {registry_path}"
    registry, error = read_registry_if_ready(registry_path)
    if error is None:
        return registry, None
    return None, f"{registry_path}: {error}"


def layout_for_args(
    args: argparse.Namespace,
    registry: EnvFleetRegistry | None,
    *,
    registry_path: Path | None = None,
) -> FleetRunLayout:
    run_id = registry.run_id if registry is not None else args.run_id
    fallback = FleetRunLayout.for_run(
        run_id=run_id,
        run_base=args.run_base,
        registry_path=registry_path or args.registry,
    )
    if registry is None:
        return fallback
    return FleetRunLayout.from_metadata(registry.metadata, fallback=fallback) or fallback


def format_status_report(
    layout: FleetRunLayout,
    registry: EnvFleetRegistry | None,
    registry_error: str | None,
    summary: ReadinessSummary,
    jobs: Sequence[SlurmJob],
) -> str:
    lines = [
        f"Run id: {layout.run_id}",
        f"Registry: {layout.registry_path}",
        f"Status dir: {summary['status_dir']}",
        f"Logs: {layout.logs_dir}",
    ]
    if jobs:
        job = jobs[0]
        lines.append(
            "Slurm: "
            f"{job.job_id} {job.state} nodes={job.nodes or '?'} "
            f"cpus={job.cpus or '?'} reason={job.reason or '-'}"
        )
    else:
        lines.append("Slurm: no matching running or pending job found")

    lines.extend(
        [
            "Registry ready: "
            f"{summary['registry_ready']} "
            f"({summary['registered_servers']}/{summary['expected_servers']} servers)",
            "Pool: "
            f"ready={summary['ready']}/{summary['min_ready']} "
            f"starting={summary['starting']} "
            f"leased={summary['leased']} "
            f"stale_status_files={summary['stale_status_files']} "
            f"total_failed={summary['total_failed']} "
            f"stale_leases_retired={summary['stale_leases_retired']}",
        ]
    )
    if unhealthy_servers := summary["unhealthy_servers"]:
        lines.append(f"Unhealthy replicas: {unhealthy_servers}")
    retry_scheduled_workers = summary["retry_scheduled_workers"]
    cooling_down_workers = summary["cooling_down_workers"]
    consecutive_start_failures = summary["consecutive_start_failures"]
    startup_cooldown_remaining_s = summary["startup_cooldown_remaining_s"]
    if (
        retry_scheduled_workers
        or cooling_down_workers
        or consecutive_start_failures
        or startup_cooldown_remaining_s > 0.0
    ):
        lines.append(
            "Startup retry: "
            f"scheduled_workers={retry_scheduled_workers} "
            f"cooling_down_workers={cooling_down_workers} "
            f"consecutive_failures={consecutive_start_failures} "
            f"cooldown_remaining_s={startup_cooldown_remaining_s:.1f}"
        )
    if registry_error:
        lines.append(f"Registry error: {registry_error}")
    unrecoverable_path = layout.run_root / "fleet_unrecoverable.json"
    if unrecoverable_path.exists():
        lines.append(f"Fleet unrecoverable marker: {unrecoverable_path}")

    if registry is not None and registry.servers:
        server_summaries = {item["name"]: item for item in summary["server_summaries"]}
        lines.append("Env servers:")
        for server in registry.servers:
            server_summary = server_summaries.get(server.name)
            pool_status = ""
            if server_summary is not None:
                pool_status = (
                    f" ready={server_summary['ready']}"
                    f" starting={server_summary['starting']}"
                    f" leased={server_summary['leased']}"
                    f" stale_status={server_summary['stale_status_files']}"
                    f" failed={server_summary['total_failed']}"
                )
            lines.append(
                f"  {server.name} {server.public_address} "
                f"workers={server.num_workers}{pool_status} log={server.log_path}"
            )
    if errors := summary["last_errors"]:
        lines.append(f"Last errors: {errors}")
    return "\n".join(lines)


def cancel(args: argparse.Namespace) -> int:
    registry = read_registry(registry_path_for_args(args), if_missing="none")
    job_id = args.job_id
    if job_id is None and registry is not None:
        job_id = slurm_job_id_from_registry(registry)
    if job_id is None:
        print("Could not determine a Slurm job id for this fleet.", file=sys.stderr)
        return 2

    jobs = query_squeue(job_id=job_id, job_name=args.job_name)
    try:
        job = select_cancel_job(
            jobs, user=os.environ.get("USER", ""), job_name=args.job_name
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if job is None:
        print(f"No matching running or pending fleet job found for {job_id}.")
        return 0

    if not confirm_cancel(job, yes=args.yes):
        print("Cancel aborted.")
        return 0

    result = subprocess.run(["scancel", job.job_id], check=False)
    if result.returncode == 0:
        print(f"Cancelled fleet job {job.job_id}.")
    return result.returncode


def main(
    argv: Sequence[str] | None = None,
    *,
    trainer_section_factory: TrainerSectionFactory | None = None,
) -> int:
    """Dispatch fleet lifecycle commands."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else ""
    if command == "prepare":
        return prepare_main(arguments[1:])
    if command == "run":
        return supervise_main(arguments[1:])
    if command == "launch":
        return launch_main(arguments[1:])
    args = parse_fleet_args(arguments)
    if args.command == "submit":
        return submit(args, trainer_section_factory=trainer_section_factory)
    if args.command == "status":
        return status(args)
    assert args.command == "cancel", args.command
    return cancel(args)


if __name__ == "__main__":
    raise SystemExit(main())
