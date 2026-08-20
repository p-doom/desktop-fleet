from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

import desktop_fleet.supervise as supervise_module
from desktop_fleet.registry import upsert_registry
from desktop_fleet.slurm import SlurmJob
from desktop_fleet.spec import (
    FleetRunLayout,
    make_server_specs,
    project_root,
    scratch_root,
)
from desktop_fleet.supervise import (
    FleetSupervisor,
    PoolHealth,
    ReplicaRuntime,
    SupervisorPolicy,
    build_prefetch_command,
    build_sbatch_command,
    default_asset_cache_dir,
    default_task_base_path,
    env_value,
    expected_ready_sessions,
    format_status_report,
    format_submit_report,
    harness_config,
    osworld_asset_cache_dir,
    osworld_deployment_root,
    osworld_qcow_path,
    osworld_root,
    parse_fleet_args,
    parse_prepare_args,
    parse_sbatch_job_id,
    process_group_alive,
    read_pool_health,
    registry_metadata,
    resolve_prepare_layout,
    restart_reason,
    terminate_replica_process_group,
)


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_prepare_harness_config_includes_desktop_pool(tmp_path):
    args = SimpleNamespace(
        max_steps=4,
        screen_width=1920,
        screen_height=1080,
        screenshot_timeout=60.0,
        artifact_output_dir=tmp_path / "custom-artifacts",
        cache_dir=tmp_path / "cache",
        run_root=tmp_path / "run",
        osworld_root=tmp_path / "OSWorld",
        qcow_path=tmp_path / "Ubuntu.qcow2",
        desktop_pool_min_ready_sessions=1,
        desktop_pool_max_sessions=1,
        desktop_pool_max_rollouts_per_session=1,
        desktop_pool_checkout_timeout=900.0,
        desktop_pool_lease_timeout=300.0,
        desktop_pool_startup_timeout=1200.0,
        desktop_pool_startup_retry_backoff=30.0,
        desktop_pool_startup_retry_backoff_max=300.0,
        desktop_pool_status_heartbeat_interval=10.0,
        desktop_pool_root=tmp_path / "run" / "desktop_pool",
        desktop_pool_runtime_dir=tmp_path / "runtime" / "pool",
        desktop_pool_log_runtime_dir=tmp_path / "log",
        workers_per_server=2,
    )

    config = harness_config(args)

    assert config["max_steps"] == 4
    desktop = config["desktop"]
    assert desktop["output_dir"] == str(tmp_path / "custom-artifacts")
    assert desktop["desktop_pool_config"] == {
        "min_ready_sessions": 1,
        "max_sessions": 1,
        "max_rollouts_per_session": 1,
        "checkout_timeout_s": 900.0,
        "lease_timeout_s": 300.0,
        "startup_timeout_s": 1200.0,
        "startup_retry_backoff_s": 30.0,
        "startup_retry_backoff_max_s": 300.0,
        "status_heartbeat_interval_s": 10.0,
        "root_dir": str(tmp_path / "run" / "desktop_pool"),
        "runtime_dir": str(tmp_path / "runtime" / "pool"),
        "log_runtime_dir": str(tmp_path / "log"),
    }
    assert expected_ready_sessions(args, replica_count=3) == 6


def test_prepare_uses_runtime_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["desktop-fleet", "prepare"])
    monkeypatch.setenv("PROJECT", str(tmp_path / "project"))
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("USER", "lossin1")
    monkeypatch.setenv("OSWORLD_FLEET_RUN_ID", "run-a")
    monkeypatch.setenv(
        "OSWORLD_DESKTOP_POOL_RUNTIME_DIR",
        str(tmp_path / "runtime" / "pool"),
    )
    monkeypatch.setenv(
        "OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR",
        str(tmp_path / "log"),
    )
    monkeypatch.delenv("OSWORLD_QCOW_PATH", raising=False)

    args = parse_prepare_args([])

    assert args.qcow_path == project_root().parent / "osworld_deployment" / "Ubuntu.qcow2"
    assert args.desktop_pool_runtime_dir == tmp_path / "runtime" / "pool"
    assert args.desktop_pool_startup_timeout == 1200.0
    assert args.desktop_pool_log_runtime_dir == tmp_path / "log"


def test_prepare_run_id_override_moves_every_sub_path(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["desktop-fleet", "prepare"])
    monkeypatch.setenv("OSWORLD_RUN_BASE", str(tmp_path))
    monkeypatch.setenv("OSWORLD_FLEET_RUN_ID", "run-a")
    for name in (
        "OSWORLD_FLEET_RUN_ROOT",
        "OSWORLD_ENV_FLEET_REGISTRY",
        "OSWORLD_DESKTOP_POOL_ROOT",
        "OSWORLD_DESKTOP_POOL_STATUS_DIR",
        "OSWORLD_FLEET_LOGS_DIR",
        "OSWORLD_FLEET_CONFIGS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    layout = resolve_prepare_layout(parse_prepare_args(["--run-id", "run-b"]))

    assert layout.run_id == "run-b"
    assert layout.run_root == tmp_path / "run-b" / "env_fleet"
    assert layout.registry_path == layout.run_root / "env_registry.json"
    assert layout.logs_dir == layout.run_root / "logs"
    assert layout.configs_dir == layout.run_root / "configs"
    assert layout.pool_root == tmp_path / "run-b" / "pool"
    assert layout.pool_status_dir == layout.pool_root / "status"


def test_prepare_honors_explicit_pool_status_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["desktop-fleet", "prepare"])
    monkeypatch.setenv("OSWORLD_RUN_BASE", str(tmp_path))
    monkeypatch.setenv("OSWORLD_FLEET_RUN_ID", "run-a")
    monkeypatch.setenv("OSWORLD_DESKTOP_POOL_STATUS_DIR", str(tmp_path / "shared"))

    layout = resolve_prepare_layout(parse_prepare_args([]))

    assert layout.pool_status_dir == tmp_path / "shared"


def test_prepare_registry_metadata_includes_layout(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="run",
        run_base=tmp_path / "osworld_rl",
    )
    args = SimpleNamespace(
        run_id="run",
        env_id="rl",
        env_name_prefix="osworld",
        task_base_path=tmp_path / "tasks",
        max_tasks=5,
        shuffle_seed=11,
        max_steps=4,
        screen_width=1920,
        screen_height=1080,
        screenshot_timeout=60.0,
        artifact_output_dir=None,
        cache_dir=tmp_path / "cache",
        run_root=layout.run_root,
        osworld_root=tmp_path / "OSWorld",
        qcow_path=tmp_path / "Ubuntu.qcow2",
        desktop_pool_min_ready_sessions=2,
        desktop_pool_max_sessions=2,
        desktop_pool_max_rollouts_per_session=1,
        desktop_pool_checkout_timeout=900.0,
        desktop_pool_lease_timeout=300.0,
        desktop_pool_startup_timeout=1200.0,
        desktop_pool_startup_retry_backoff=30.0,
        desktop_pool_startup_retry_backoff_max=300.0,
        desktop_pool_status_heartbeat_interval=10.0,
        desktop_pool_root=layout.pool_root,
        desktop_pool_runtime_dir=tmp_path / "runtime" / "pool",
        desktop_pool_log_runtime_dir=tmp_path / "log",
        workers_per_server=3,
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        servers_per_node=2,
        replica_hosts="node001,node002",
        gateway_host=None,
        gateway_bind_host=None,
        gateway_port=0,
    )

    metadata = registry_metadata(args, layout, replica_count=4)

    assert metadata["layout"] == layout.as_metadata()
    assert metadata["env_name_prefix"] == "osworld"
    assert "pool_root" not in metadata
    assert "pool_status_dir" not in metadata
    assert metadata["max_tasks"] == 5
    desktop = metadata["harness"]["desktop"]
    assert desktop["output_dir"] == str(layout.run_root / "artifacts")
    assert desktop["desktop_pool_config"]["runtime_dir"] == str(
        tmp_path / "runtime" / "pool"
    )
    assert desktop["desktop_pool_config"]["log_runtime_dir"] == str(tmp_path / "log")
    assert metadata["gateway"] == {
        "bind_address": "tcp://0.0.0.0:5204",
        "public_address": "tcp://node001:5204",
        "backend_addresses": [
            "tcp://node001:5200",
            "tcp://node001:5201",
            "tcp://node002:5202",
            "tcp://node002:5203",
        ],
    }
    assert metadata["expected_env_servers"] == 4
    assert metadata["expected_env_workers"] == 12
    assert "expected_server_count" not in metadata
    assert "expected_worker_count" not in metadata
    assert metadata["expected_ready_sessions"] == 24


def _replica(tmp_path) -> ReplicaRuntime:
    return ReplicaRuntime(
        name="osworld-0000",
        config_path=tmp_path / "config.toml",
        log_path=tmp_path / "server.log",
        status_dir=tmp_path / "status",
        command=[],
        process=SimpleNamespace(poll=lambda: None, pid=123),
        started_at=0.0,
        unhealthy_since=0.0,
    )


def _policy() -> SupervisorPolicy:
    return SupervisorPolicy(
        poll_s=5.0,
        startup_grace_s=10.0,
        replica_unhealthy_s=10.0,
        failure_window_s=300.0,
        max_failures_per_window=0,
        restart_backoff_s=10.0,
        fleet_unhealthy_s=300.0,
        max_fleet_restarts=3,
        terminate_timeout_s=30.0,
        status_stale_after_s=120.0,
    )


def test_supervisor_fresh_starting_capacity_is_recovering_after_grace(tmp_path):
    health = PoolHealth(
        status_files=1,
        active_status_files=1,
        stale_status_files=0,
        ready=0,
        starting=4,
        fresh_starting=4,
        stale_starting=0,
        oldest_starting_age_s=30.0,
        leased=0,
        total_failed=0,
        last_errors=[],
    )

    assert restart_reason(_replica(tmp_path), health, now=30.0, policy=_policy()) is None


def test_supervisor_stale_starting_capacity_is_unhealthy_after_grace(tmp_path):
    health = PoolHealth(
        status_files=1,
        active_status_files=1,
        stale_status_files=0,
        ready=0,
        starting=4,
        fresh_starting=0,
        stale_starting=4,
        oldest_starting_age_s=901.0,
        leased=0,
        total_failed=0,
        last_errors=[],
    )

    assert (
        restart_reason(_replica(tmp_path), health, now=30.0, policy=_policy())
        == "4 desktop sessions stuck starting for up to 901.0s"
    )


def test_supervisor_mixed_starting_capacity_is_unhealthy_after_grace(tmp_path):
    health = PoolHealth(
        status_files=1,
        active_status_files=1,
        stale_status_files=0,
        ready=0,
        starting=4,
        fresh_starting=2,
        stale_starting=2,
        oldest_starting_age_s=901.0,
        leased=0,
        total_failed=0,
        last_errors=[],
    )

    assert (
        restart_reason(_replica(tmp_path), health, now=30.0, policy=_policy())
        == "2 desktop sessions stuck starting for up to 901.0s"
    )


def test_read_pool_health_splits_fresh_and_stale_starting_sessions(tmp_path):
    now = time.time()
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    (status_dir / "worker.json").write_text(
        json.dumps(
            {
                "updated_at": now,
                "closed": False,
                "ready": 0,
                "starting": 2,
                "leased": 0,
                "total_failed": 0,
                "startup_timeout_s": 100.0,
                "starting_sessions": [
                    {"session_id": "fresh", "created_at": now - 10.0},
                    {"session_id": "stale", "created_at": now - 101.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    health = read_pool_health(status_dir, status_stale_after_s=120.0)

    assert health.starting == 2
    assert health.fresh_starting == 1
    assert health.stale_starting == 1
    assert health.oldest_starting_age_s is not None
    assert health.oldest_starting_age_s >= 100.0


def test_a_replicas_process_group_is_reaped_after_the_replica_itself_has_died():
    """The actual leak: a worker that crashed and left its desktops behind.

    ``sh -c 'sleep 300 & exit 0'`` has exactly that shape -- the group leader
    exits, a child outlives it in the same process group -- because ``desktop``
    deliberately keeps QEMU in the pool process's group.  Returning early on an
    already-exited child strands that VM for the rest of the allocation, holding
    its 8 GB and its four host ports, and no status file can rescue it: nothing
    ever wrote the VM's process group anywhere.
    """
    process = subprocess.Popen(["sh", "-c", "sleep 300 & exit 0"], start_new_session=True)
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.01)
        assert process.poll() is not None, "the group leader should have exited"
        assert process_group_alive(process.pid), "the child should outlive the leader"

        terminate_replica_process_group(process, timeout_s=10.0)

        assert not process_group_alive(process.pid)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


class _FakeProcess:
    """A minimal stand-in for ``subprocess.Popen`` driven entirely by the test."""

    _next_pid = 900001

    def __init__(self, returncode: int | None = None) -> None:
        self.pid = _FakeProcess._next_pid
        _FakeProcess._next_pid += 1
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


@pytest.fixture
def recorded_signals(monkeypatch):
    """Record every process group the supervisor signals, and signal nothing.

    Teardown goes through ``os.killpg`` on the child's own pid, so a synthetic
    pid is not enough to keep a test off a real process group -- pids that high
    do exist on a busy node.  Recording is also the only way to check the
    contract that matters: the whole GROUP is signalled, not just the child.
    """
    signalled: list[tuple[int, int]] = []

    def fake_killpg(process_group_id: int, sig: int) -> None:
        signalled.append((process_group_id, sig))

    monkeypatch.setattr(supervise_module.os, "killpg", fake_killpg)
    monkeypatch.setattr(supervise_module, "process_group_alive", lambda _: False)
    return signalled


def _make_registry_and_config(tmp_path, *, with_pool_status: bool = False):
    """Write a one-replica registry + matching rendered-config dir on disk."""
    config_dir = tmp_path / "configs"
    logs_dir = tmp_path / "logs"
    config_dir.mkdir()
    pool_status_root = tmp_path / "pool" / "status" if with_pool_status else None
    if pool_status_root is not None:
        pool_status_root.mkdir(parents=True)

    specs = make_server_specs(
        host="localhost",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=1,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=config_dir,
        log_dir=logs_dir,
        pool_status_root=pool_status_root,
    )
    (config_dir / f"{specs[0].name}.toml").write_text("", encoding="utf-8")
    registry_path = tmp_path / "env_registry.json"
    upsert_registry(path=registry_path, run_id="test-run", metadata={}, servers=specs)
    return config_dir, logs_dir, registry_path


def _build_supervisor(
    tmp_path,
    *,
    policy: SupervisorPolicy,
    start_gateway: bool = False,
    with_pool_status: bool = False,
) -> FleetSupervisor:
    config_dir, logs_dir, registry_path = _make_registry_and_config(
        tmp_path, with_pool_status=with_pool_status
    )
    args = SimpleNamespace(
        run_root=tmp_path / "run",
        registry=registry_path,
        logs_dir=logs_dir,
        config_dir=config_dir,
        env_server_bin="/fake/env-server",
        start_gateway=start_gateway,
        gateway_log=None,
        python=sys.executable,
        gateway_request_timeout_s=900.0,
        gateway_backend_quarantine_s=30.0,
        gateway_capacity_check_interval=5.0,
        status_stale_after_s=policy.status_stale_after_s,
        gateway_health_check_interval=2.0,
        gateway_health_check_timeout=5.0,
    )
    return FleetSupervisor(args=args, policy=policy)


def test_supervisor_replica_unhealthy_threshold_gates_restart_until_elapsed(tmp_path):
    """restart_reason must wait out replica_unhealthy_s, not fire on first sight."""
    policy = _policy()  # replica_unhealthy_s=10.0, startup_grace_s=10.0
    replica = _replica(tmp_path)
    replica.unhealthy_since = None
    no_status_files = PoolHealth(
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

    # first observation only records the unhealthy timestamp -- no restart yet
    assert restart_reason(replica, no_status_files, now=100.0, policy=policy) is None
    assert replica.unhealthy_since == 100.0

    # 8s elapsed, under the 10s threshold: still no restart
    assert restart_reason(replica, no_status_files, now=108.0, policy=policy) is None

    # 11s elapsed, past the 10s threshold: the reason now fires
    assert (
        restart_reason(replica, no_status_files, now=111.0, policy=policy)
        == "no active desktop-pool worker status files"
    )


def test_supervisor_maybe_restart_fleet_waits_for_fleet_unhealthy_threshold(
    tmp_path, monkeypatch
):
    """maybe_restart_fleet must wait out fleet_unhealthy_s before acting."""
    policy = SupervisorPolicy(
        poll_s=5.0,
        startup_grace_s=0.0,
        replica_unhealthy_s=0.0,
        failure_window_s=300.0,
        max_failures_per_window=0,
        restart_backoff_s=10.0,
        fleet_unhealthy_s=10.0,
        max_fleet_restarts=3,
        terminate_timeout_s=5.0,
        status_stale_after_s=120.0,
    )
    supervisor = _build_supervisor(tmp_path, policy=policy)
    monkeypatch.setattr(
        supervisor, "spawn", lambda command, log_path: _FakeProcess(returncode=1)
    )
    supervisor.start_replicas(supervisor.replicas, reason="initial start")

    fleet_restarts = []
    monkeypatch.setattr(
        supervisor,
        "start_replicas",
        lambda replicas, *, reason: fleet_restarts.append(reason),
    )

    assert supervisor.maybe_restart_fleet(200.0) is False
    assert supervisor.fleet_unhealthy_since == 200.0
    assert fleet_restarts == []  # first detection only, no action yet

    assert supervisor.maybe_restart_fleet(205.0) is False  # 5s < 10s threshold
    assert fleet_restarts == []

    assert supervisor.maybe_restart_fleet(211.0) is False  # 11s >= 10s threshold
    assert fleet_restarts == ["fleet unhealthy"]
    assert supervisor.fleet_unhealthy_since is None


def test_supervisor_restarts_crashed_replica_respecting_backoff(tmp_path, monkeypatch):
    policy = SupervisorPolicy(
        poll_s=5.0,
        startup_grace_s=0.0,
        replica_unhealthy_s=0.0,
        failure_window_s=300.0,
        max_failures_per_window=0,
        restart_backoff_s=10.0,
        fleet_unhealthy_s=300.0,
        max_fleet_restarts=3,
        terminate_timeout_s=5.0,
        status_stale_after_s=120.0,
    )
    supervisor = _build_supervisor(tmp_path, policy=policy)
    spawned: list[_FakeProcess] = []

    def fake_spawn(command, log_path):
        process = _FakeProcess()
        spawned.append(process)
        return process

    monkeypatch.setattr(supervisor, "spawn", fake_spawn)
    supervisor.start_replicas(supervisor.replicas, reason="initial start")
    replica = supervisor.replicas[0]
    assert len(spawned) == 1
    assert replica.restart_count == 0

    # the replica process crashes
    spawned[0].returncode = 7
    # still inside the backoff window set by start_replica -- must not restart yet
    supervisor.monitor_replicas(time.monotonic())
    assert len(spawned) == 1, "must not restart before the backoff window elapses"
    assert replica.restart_count == 0

    # simulate the backoff window having elapsed
    replica.next_restart_at = time.monotonic() - 1.0
    supervisor.monitor_replicas(time.monotonic())
    assert len(spawned) == 2, "must restart once backoff has elapsed"
    assert replica.restart_count == 1
    assert replica.last_restart_reason == "process exited with code 7"
    assert replica.process is spawned[1]


def test_supervisor_gives_up_after_repeated_fleet_restarts(tmp_path, monkeypatch):
    """A replica that keeps crashing must eventually be given up on, not looped
    forever -- run() should return 42 and write the unrecoverable marker."""
    monkeypatch.setattr(signal, "signal", lambda *args, **kwargs: None)
    policy = SupervisorPolicy(
        poll_s=0.0,
        startup_grace_s=0.0,
        replica_unhealthy_s=0.0,
        failure_window_s=1.0,
        max_failures_per_window=0,
        restart_backoff_s=0.0,
        fleet_unhealthy_s=0.0,
        max_fleet_restarts=1,
        terminate_timeout_s=0.1,
        status_stale_after_s=120.0,
    )
    supervisor = _build_supervisor(tmp_path, policy=policy)
    monkeypatch.setattr(
        supervisor, "spawn", lambda command, log_path: _FakeProcess(returncode=1)
    )

    iterations = {"count": 0}

    def guarded_sleep(seconds: float) -> None:
        # safety net only: the give-up path below is expected to return long
        # before this trips, so tripping it is itself a test failure signal.
        iterations["count"] += 1
        if iterations["count"] > 500:
            supervisor.stop_requested = True

    monkeypatch.setattr(time, "sleep", guarded_sleep)

    exit_code = supervisor.run()

    assert iterations["count"] < 500, "hit the safety net instead of giving up"
    assert exit_code == 42
    assert supervisor.fleet_restart_count > policy.max_fleet_restarts
    assert supervisor.unrecoverable_path.exists()
    marker = json.loads(supervisor.unrecoverable_path.read_text())
    assert marker["status"] == "unrecoverable"
    assert marker["fleet_restart_count"] == supervisor.fleet_restart_count


def test_supervisor_reaps_gateway_process_on_teardown(
    tmp_path, monkeypatch, recorded_signals
):
    monkeypatch.setattr(signal, "signal", lambda *args, **kwargs: None)
    policy = SupervisorPolicy(
        poll_s=0.0,
        startup_grace_s=1000.0,
        replica_unhealthy_s=1000.0,
        failure_window_s=300.0,
        max_failures_per_window=0,
        restart_backoff_s=10.0,
        fleet_unhealthy_s=1000.0,
        max_fleet_restarts=3,
        terminate_timeout_s=1.0,
        status_stale_after_s=120.0,
    )
    supervisor = _build_supervisor(tmp_path, policy=policy, start_gateway=True)

    replica_processes: list[_FakeProcess] = []
    gateway_processes: list[_FakeProcess] = []

    def fake_spawn(command, log_path):
        process = _FakeProcess(returncode=None)  # stays "alive" the whole time
        if "desktop_fleet.broker" in command:
            gateway_processes.append(process)
        else:
            replica_processes.append(process)
        return process

    monkeypatch.setattr(supervisor, "spawn", fake_spawn)

    def stop_after_one_iteration(seconds: float) -> None:
        supervisor.stop_requested = True

    monkeypatch.setattr(time, "sleep", stop_after_one_iteration)

    exit_code = supervisor.run()

    assert exit_code == 0
    assert len(gateway_processes) == 1
    assert supervisor.gateway_process is None
    # the replica was healthy throughout (long startup grace), so it was never
    # restarted -- only reaped once, at teardown, like the gateway.
    assert supervisor.replicas[0].restart_count == 0
    assert len(replica_processes) == 1
    # Each child's own pid IS its group id (spawn uses start_new_session=True), so
    # a desktop the child left behind is inside the group that gets SIGTERM.
    assert set(recorded_signals) == {
        (replica_processes[0].pid, signal.SIGTERM),
        (gateway_processes[0].pid, signal.SIGTERM),
    }


def test_submit_command_uses_slurm_options_and_exports(tmp_path):
    args = SimpleNamespace(
        script=Path("sbatch/run_osworld_env_fleet.sbatch"),
        account="research",
        partition="booster",
        time="00:30:00",
        nodes=2,
        cpus_per_task=16,
        mem=None,
        job_name="osworld_env_fleet",
        run_id="fleet-a",
        run_base=tmp_path / "runs",
        task_base_path=tmp_path / "tasks",
        asset_cache_dir=tmp_path / "asset-cache",
        servers_per_node=2,
        workers_per_server=4,
        base_port=5300,
        max_tasks=7,
        artifact_output_dir=tmp_path / "artifacts",
        desktop_pool_min_ready_sessions=2,
        desktop_pool_max_sessions=3,
        desktop_pool_max_rollouts_per_session=25,
        desktop_pool_checkout_timeout=120.0,
        desktop_pool_lease_timeout=300.0,
        desktop_pool_startup_timeout=1200.0,
        desktop_pool_startup_retry_backoff=5.0,
        desktop_pool_startup_retry_backoff_max=60.0,
        desktop_pool_status_heartbeat_interval=7.0,
        desktop_pool_root=tmp_path / "desktop-pool",
        desktop_pool_runtime_dir=tmp_path / "desktop-runtime",
        desktop_pool_log_runtime_dir=tmp_path / "desktop-log",
        rollout_timeout=900.0,
        env_max_retries=2,
        replica_unhealthy_s=120.0,
        fleet_unhealthy_s=300.0,
        max_fleet_restarts=3,
        gateway_request_timeout_s=900.0,
        status_stale_after_s=120.0,
    )

    command = build_sbatch_command(args)

    assert command[:2] == ["sbatch", "--parsable"]
    assert "--partition" in command
    assert "booster" in command
    assert "--nodes" in command
    assert "2" in command
    assert any("OSWORLD_FLEET_RUN_ID=fleet-a" in item for item in command)
    assert not any("OSWORLD_TASK_BASE_PATH=" in item for item in command)
    assert not any(str(tmp_path / "asset-cache") in item for item in command)
    assert any("OSWORLD_ENV_WORKERS_PER_SERVER=4" in item for item in command)
    assert any("OSWORLD_MAX_TASKS=7" in item for item in command)
    assert any(
        f"OSWORLD_ARTIFACT_DIR={tmp_path / 'artifacts'}" in item for item in command
    )
    assert any("OSWORLD_DESKTOP_POOL_MIN_READY_SESSIONS=2" in item for item in command)
    assert any("OSWORLD_DESKTOP_POOL_MAX_SESSIONS=3" in item for item in command)
    assert any(
        "OSWORLD_DESKTOP_POOL_MAX_ROLLOUTS_PER_SESSION=25" in item for item in command
    )
    assert any(
        "OSWORLD_DESKTOP_POOL_CHECKOUT_TIMEOUT=120.0" in item for item in command
    )
    assert any("OSWORLD_DESKTOP_POOL_LEASE_TIMEOUT=300.0" in item for item in command)
    assert any("OSWORLD_DESKTOP_POOL_STARTUP_TIMEOUT=1200.0" in item for item in command)
    assert any(
        "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF=5.0" in item for item in command
    )
    assert any(
        "OSWORLD_DESKTOP_POOL_STARTUP_RETRY_BACKOFF_MAX=60.0" in item
        for item in command
    )
    assert any(
        "OSWORLD_DESKTOP_POOL_STATUS_HEARTBEAT_INTERVAL=7.0" in item for item in command
    )
    assert any(
        f"OSWORLD_DESKTOP_POOL_ROOT={tmp_path / 'desktop-pool'}" in item
        for item in command
    )
    assert any(
        f"OSWORLD_DESKTOP_POOL_RUNTIME_DIR={tmp_path / 'desktop-runtime'}" in item
        for item in command
    )
    assert any(
        f"OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR={tmp_path / 'desktop-log'}" in item
        for item in command
    )
    assert any("OSWORLD_ROLLOUT_TIMEOUT=900.0" in item for item in command)
    assert any("OSWORLD_ENV_MAX_RETRIES=2" in item for item in command)
    assert any("OSWORLD_SUPERVISOR_MAX_FLEET_RESTARTS=3" in item for item in command)
    assert any("OSWORLD_GATEWAY_REQUEST_TIMEOUT_S=900.0" in item for item in command)
    assert any("OSWORLD_STATUS_STALE_AFTER_S=120.0" in item for item in command)
    assert parse_sbatch_job_id("12345;juwels\n") == "12345"


def test_prefetch_command_uses_asset_cache(tmp_path):
    args = SimpleNamespace(
        task_base_path=tmp_path / "tasks",
        asset_cache_dir=tmp_path / "asset-cache",
        asset_source_root=None,
    )

    command = build_prefetch_command(args)

    assert command[:4] == ["uv", "run", "--no-sync", "python"]
    assert command[4].endswith("prefetch_osworld_assets.py")
    assert command[-4:] == [
        "--tasks",
        str(tmp_path / "tasks"),
        "--cache-dir",
        str(tmp_path / "asset-cache"),
    ]


def test_default_task_base_uses_osworld_root(tmp_path):
    env = {
        "OSWORLD_ROOT": str(tmp_path / "OSWorld"),
        "OSWORLD_TASK_BASE_PATH": str(tmp_path / "ignored"),
    }

    assert default_task_base_path(env) == (
        tmp_path
        / "OSWorld"
        / "evaluation_examples"
        / "examples"
        / "target_box_empty_desktop"
    )


def test_fleet_parse_args_loads_runtime_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"export SCRATCH={tmp_path / 'scratch'}",
                f"export OSWORLD_ROOT={tmp_path / 'OSWorld'}",
                "export OSWORLD_DESKTOP_POOL_RUNTIME_DIR=/tmp/osworld-runtime",
                "export OSWORLD_DESKTOP_POOL_LOG_RUNTIME_DIR=/tmp/osworld-log",
                "export OSWORLD_FLEET_SLURM_PARTITION=cpu-partition",
                "export OSWORLD_FLEET_SLURM_TIME=08:00:00",
                "export OSWORLD_FLEET_SLURM_NODES=2",
                "export OSWORLD_FLEET_SLURM_CPUS_PER_TASK=24",
                "export OSWORLD_FLEET_SLURM_MEM_PER_NODE=96G",
                "export SBATCH_ACCOUNT=fallback-project",
                "export SBATCH_PARTITION=fallback-partition",
                "export SBATCH_TIMELIMIT=01:00:00",
                "export SBATCH_NODES=3",
                "export SBATCH_CPUS_PER_TASK=12",
                "export SBATCH_MEM_PER_NODE=48G",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", str(env_file))
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.delenv("OSWORLD_ROOT", raising=False)

    args = parse_fleet_args(["submit", "--dry-run"])

    assert args.run_base == tmp_path / "scratch"
    assert args.asset_cache_dir == tmp_path / "scratch" / "osworld_asset_cache"
    assert args.task_base_path == (
        tmp_path
        / "OSWorld"
        / "evaluation_examples"
        / "examples"
        / "target_box_empty_desktop"
    )
    assert args.desktop_pool_runtime_dir == Path("/tmp/osworld-runtime")
    assert args.desktop_pool_log_runtime_dir == Path("/tmp/osworld-log")
    assert args.account == "fallback-project"
    assert args.partition == "cpu-partition"
    assert args.time == "08:00:00"
    assert args.nodes == 2
    assert args.cpus_per_task == 24
    assert args.mem == "96G"


def test_default_asset_cache_uses_scratch(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
    }

    assert default_asset_cache_dir(env) == tmp_path / "scratch" / "osworld_asset_cache"


def test_submit_report_prints_next_steps_without_a_consumer(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="fleet-a",
        run_base=tmp_path / "runs",
    )

    with pytest.raises(SystemExit):
        parse_fleet_args(["render"])

    report = format_submit_report("12345", layout, SimpleNamespace())

    assert "Readiness:" in report
    assert "uv run --no-sync python -m desktop_fleet.supervise status" in report
    assert "uv run --no-sync python -m desktop_fleet.readiness" in report
    assert "Cancel:" in report
    assert "prime_rl" not in report
    assert ".venv/bin/rl" not in report


def test_status_format_and_registry_rendering(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="12345",
        run_base=tmp_path / "osworld_rl",
    )
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=2,
        replica_count=1,
        replica_offset=0,
        name_prefix="osworld",
        config_dir=layout.configs_dir,
        log_dir=layout.logs_dir,
    )[0]
    registry = upsert_registry(
        path=layout.registry_path,
        run_id="12345",
        metadata={"layout": layout.as_metadata(), "expected_env_servers": 1},
        servers=[server],
    )
    summary = {
        "status_dir": str(layout.pool_status_dir),
        "registry_ready": True,
        "registered_servers": 1,
        "expected_servers": 1,
        "ready": 2,
        "min_ready": 2,
        "starting": 0,
        "leased": 0,
        "stale_status_files": 0,
        "total_failed": 1,
        "stale_leases_retired": 3,
        "retry_scheduled_workers": 1,
        "cooling_down_workers": 1,
        "consecutive_start_failures": 2,
        "startup_cooldown_remaining_s": 12.5,
        "last_errors": [],
        "unhealthy_servers": 0,
        "server_summaries": [
            {
                "name": "osworld-0000",
                "ready": 2,
                "starting": 0,
                "leased": 0,
                "stale_status_files": 0,
                "total_failed": 1,
            }
        ],
    }
    job = SlurmJob(
        job_id="12345",
        user="user-a",
        name="osworld_env_fleet",
        state="RUNNING",
        reason="None",
        elapsed="1:00",
        nodes="1",
        cpus="16",
    )

    report = format_status_report(layout, registry, None, summary, [job])

    assert "Slurm: 12345 RUNNING" in report
    assert "ready=2/2" in report
    assert "ready=2 starting=0 leased=0 stale_status=0 failed=1" in report
    assert "Startup retry: scheduled_workers=1" in report
    assert "cooldown_remaining_s=12.5" in report
    assert "tcp://node001:5200" in report
    assert "stale_leases_retired=3" in report

    # A defaulted read would print stale_leases_retired=0 and suppress the
    # unhealthy-replica line entirely, so an incomplete summary reads as a
    # healthy fleet.
    for key in ("stale_leases_retired", "unhealthy_servers", "server_summaries"):
        with pytest.raises(KeyError, match=key):
            format_status_report(
                layout,
                registry,
                None,
                {k: v for k, v in summary.items() if k != key},
                [job],
            )


def test_path_helpers_default_to_project_scratch_and_sibling_osworld(monkeypatch):
    monkeypatch.delenv("SCRATCH", raising=False)
    monkeypatch.delenv("OSWORLD_ROOT", raising=False)
    monkeypatch.delenv("OSWORLD_DEPLOYMENT_ROOT", raising=False)
    monkeypatch.delenv("OSWORLD_QCOW_PATH", raising=False)

    assert osworld_root({}) == project_root({}).parent / "OSWorldRL"
    assert osworld_deployment_root({}) == project_root({}).parent / "osworld_deployment"
    assert osworld_qcow_path({}) == (
        project_root({}).parent / "osworld_deployment" / "Ubuntu.qcow2"
    )
    assert osworld_asset_cache_dir({}) == (
        project_root({}) / ".scratch" / "osworld_asset_cache"
    )


def test_path_helpers_honor_runtime_overrides(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_ROOT": str(tmp_path / "checkout" / "OSWorld"),
        "OSWORLD_DEPLOYMENT_ROOT": str(tmp_path / "deploy"),
        "OSWORLD_QCOW_PATH": str(tmp_path / "images" / "custom.qcow2"),
    }

    assert osworld_root(env) == tmp_path / "checkout" / "OSWorld"
    assert osworld_deployment_root(env) == tmp_path / "deploy"
    assert osworld_qcow_path(env) == tmp_path / "images" / "custom.qcow2"
    assert osworld_asset_cache_dir(env) == tmp_path / "scratch" / "osworld_asset_cache"


@pytest.mark.parametrize(
    ("name", "helper"),
    [
        ("SCRATCH", scratch_root),
        ("OSWORLD_ROOT", osworld_root),
        ("OSWORLD_DEPLOYMENT_ROOT", osworld_deployment_root),
        ("OSWORLD_QCOW_PATH", osworld_qcow_path),
    ],
)
def test_path_helpers_reject_non_absolute_runtime_overrides(name, helper):
    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: "~/not-expanded"})

    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: "relative/path"})

    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: f"${name}/path"})

    with pytest.raises(ValueError, match=f"{name} must be an absolute path"):
        helper({name: f"${{{name}}}/path"})


def test_env_value_rejects_a_malformed_environment_override():
    assert env_value({}, "OSWORLD_ENV_SERVERS_PER_NODE", int, 1) == 1
    with pytest.raises(ValueError, match="OSWORLD_ENV_SERVERS_PER_NODE"):
        env_value({"OSWORLD_ENV_SERVERS_PER_NODE": "8x"}, "OSWORLD_ENV_SERVERS_PER_NODE", int, 1)
