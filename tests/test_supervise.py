from __future__ import annotations

import json
import logging
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
from desktop_fleet.local_runtime import LOCAL_RUNTIME_OWNER_FILE, local_runtime_scope
from desktop_fleet.registry import upsert_registry
from desktop_fleet.slurm import SlurmJob
from desktop_fleet.spec import (
    FleetRunLayout,
    make_server_specs,
)
from desktop_fleet.supervise import (
    DEFAULT_FLEET_SCRIPT,
    FleetSupervisor,
    PoolHealth,
    ReplicaRuntime,
    SupervisorPolicy,
    _remove_local_runtime_root,
    _validate_local_runtime_root,
    _write_local_runtime_owner,
    build_sbatch_command,
    enforce_vm_memory_budget,
    env_value,
    expected_ready_sessions,
    format_status_report,
    format_submit_report,
    launch_main,
    parse_fleet_args,
    parse_prepare_args,
    parse_sbatch_job_id,
    prepare_main,
    process_group_alive,
    read_pool_health,
    registry_metadata,
    resolve_prepare_layout,
    restart_reason,
    terminate_replica_process_group,
)


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("ENV_FLEET_RUNTIME_ENV_FILE", "")


@pytest.fixture(autouse=True)
def local_runtime_env(monkeypatch, tmp_path):
    """The supervisor derives its node-local runtime root from the allocation."""
    monkeypatch.setenv("TMPDIR", str(tmp_path.resolve()))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("ENV_FLEET_REPLICA_COUNT", "1")


def test_expected_ready_sessions_uses_the_harness_pool_requirement(
    environment_contract,
):
    assert expected_ready_sessions(
        workers_per_server=2,
        replica_count=3,
        min_ready_sessions=environment_contract.session.min_ready_sessions,
    ) == 6


def test_prepare_requires_the_environment_contract(monkeypatch):
    monkeypatch.delenv("ENV_FLEET_ENVIRONMENT_CONTRACT", raising=False)

    with pytest.raises(SystemExit):
        parse_prepare_args([])


def test_prepare_requires_the_global_replica_count(
    monkeypatch,
    environment_contract_path,
):
    monkeypatch.delenv("ENV_FLEET_REPLICA_COUNT")

    with pytest.raises(SystemExit):
        parse_prepare_args(
            ["--environment-contract", str(environment_contract_path)]
        )


def test_prepare_run_id_override_moves_every_sub_path(
    monkeypatch,
    tmp_path,
    environment_contract_path,
):
    monkeypatch.setattr(sys, "argv", ["desktop-fleet", "prepare"])
    monkeypatch.setenv("ENV_FLEET_RUN_BASE", str(tmp_path))
    monkeypatch.setenv("ENV_FLEET_RUN_ID", "run-a")
    for name in (
        "ENV_FLEET_RUN_ROOT",
        "ENV_FLEET_REGISTRY",
        "ENV_FLEET_DESKTOP_POOL_ROOT",
        "ENV_FLEET_DESKTOP_POOL_STATUS_DIR",
        "ENV_FLEET_LOGS_DIR",
        "ENV_FLEET_CONFIGS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    layout = resolve_prepare_layout(
        parse_prepare_args(
            [
                "--run-id",
                "run-b",
                "--environment-contract",
                str(environment_contract_path),
            ]
        )
    )

    assert layout.run_id == "run-b"
    assert layout.run_root == tmp_path / "run-b" / "env_fleet"
    assert layout.registry_path == layout.run_root / "env_registry.json"
    assert layout.logs_dir == layout.run_root / "logs"
    assert layout.configs_dir == layout.run_root / "configs"
    assert layout.pool_root == tmp_path / "run-b" / "pool"
    assert layout.pool_status_dir == layout.pool_root / "status"


def test_prepare_honors_explicit_pool_status_dir(
    monkeypatch,
    tmp_path,
    environment_contract_path,
):
    monkeypatch.setattr(sys, "argv", ["desktop-fleet", "prepare"])
    monkeypatch.setenv("ENV_FLEET_RUN_BASE", str(tmp_path))
    monkeypatch.setenv("ENV_FLEET_RUN_ID", "run-a")
    monkeypatch.setenv("ENV_FLEET_DESKTOP_POOL_STATUS_DIR", str(tmp_path / "shared"))

    layout = resolve_prepare_layout(
        parse_prepare_args(
            ["--environment-contract", str(environment_contract_path)]
        )
    )

    assert layout.pool_status_dir == tmp_path / "shared"


def test_prepare_registry_metadata_uses_only_the_environment_contract(
    tmp_path,
    environment_contract,
):
    layout = FleetRunLayout.for_run(
        run_id="run",
        run_base=tmp_path / "runs",
    )
    args = SimpleNamespace(
        run_id="run",
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

    metadata = registry_metadata(
        args,
        layout,
        replica_count=4,
        environment=environment_contract,
    )

    assert metadata["layout"] == layout.as_metadata()
    assert metadata["environment"] == environment_contract.as_metadata()
    assert "pool_root" not in metadata
    assert "pool_status_dir" not in metadata
    for removed in (
        "env_id",
        "env_name_prefix",
        "task_base_path",
        "max_tasks",
        "shuffle_seed",
        "harness",
    ):
        assert removed not in metadata
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
    assert metadata["expected_ready_sessions"] == 12
    assert metadata["expected_ready_sessions"] == (
        metadata["expected_env_workers"]
        * environment_contract.session.min_ready_sessions
    )


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


def _make_registry_and_config(
    tmp_path,
    environment_metadata,
    *,
    with_pool_status: bool = False,
):
    """Write a one-replica registry + matching rendered-config dir on disk."""
    config_dir = tmp_path / "configs"
    logs_dir = tmp_path / "logs"
    config_dir.mkdir()
    pool_status_root = tmp_path / "pool" / "status"
    if with_pool_status:
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
        name_prefix="fixture",
        config_dir=config_dir,
        log_dir=logs_dir,
        pool_status_root=pool_status_root,
    )
    (config_dir / f"{specs[0].name}.toml").write_text("", encoding="utf-8")
    registry_path = tmp_path / "env_registry.json"
    upsert_registry(
        path=registry_path,
        run_id="test-run",
        metadata=environment_metadata,
        servers=specs,
    )
    return config_dir, logs_dir, registry_path


def _build_supervisor(
    tmp_path,
    environment_metadata,
    *,
    policy: SupervisorPolicy,
    start_gateway: bool = False,
    with_pool_status: bool = False,
) -> FleetSupervisor:
    config_dir, logs_dir, registry_path = _make_registry_and_config(
        tmp_path,
        environment_metadata,
        with_pool_status=with_pool_status,
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
    tmp_path,
    monkeypatch,
    environment_metadata,
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
    supervisor = _build_supervisor(
        tmp_path,
        environment_metadata,
        policy=policy,
    )
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


def test_supervisor_restarts_crashed_replica_respecting_backoff(
    tmp_path,
    monkeypatch,
    environment_metadata,
):
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
    supervisor = _build_supervisor(
        tmp_path,
        environment_metadata,
        policy=policy,
    )
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


def test_supervisor_gives_up_after_repeated_fleet_restarts(
    tmp_path,
    monkeypatch,
    environment_metadata,
):
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
    supervisor = _build_supervisor(
        tmp_path,
        environment_metadata,
        policy=policy,
    )
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
    tmp_path,
    monkeypatch,
    recorded_signals,
    environment_metadata,
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
    supervisor = _build_supervisor(
        tmp_path,
        environment_metadata,
        policy=policy,
        start_gateway=True,
    )

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


def test_submit_command_uses_slurm_options_and_contract_export(
    tmp_path,
    environment_contract_path,
):
    args = SimpleNamespace(
        script=Path("sbatch/run_env_fleet.sbatch"),
        account="research",
        partition="booster",
        time="00:30:00",
        nodes=2,
        cpus_per_task=16,
        mem=None,
        job_name="env_fleet",
        run_id="fleet-a",
        run_base=tmp_path / "runs",
        environment_contract=environment_contract_path,
        servers_per_node=2,
        workers_per_server=4,
        base_port=5300,
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
    assert any("ENV_FLEET_RUN_ID=fleet-a" in item for item in command)
    assert any("ENV_FLEET_WORKERS_PER_SERVER=4" in item for item in command)
    assert any("ENV_FLEET_ALLOCATED_NODES=2" in item for item in command)
    assert any("ENV_FLEET_REPLICA_COUNT=4" in item for item in command)
    assert any(
        f"ENV_FLEET_ENVIRONMENT_CONTRACT={environment_contract_path}" in item
        for item in command
    )
    assert any("ENV_FLEET_SUPERVISOR_MAX_FLEET_RESTARTS=3" in item for item in command)
    assert any("ENV_FLEET_GATEWAY_REQUEST_TIMEOUT_S=900.0" in item for item in command)
    assert any("ENV_FLEET_STATUS_STALE_AFTER_S=120.0" in item for item in command)
    assert parse_sbatch_job_id("12345;juwels\n") == "12345"


def test_default_launcher_is_checked_in_and_fans_out_one_task_per_node():
    assert DEFAULT_FLEET_SCRIPT.is_file()
    subprocess.run(["bash", "-n", str(DEFAULT_FLEET_SCRIPT)], check=True)
    source = DEFAULT_FLEET_SCRIPT.read_text(encoding="utf-8")
    assert '--nodes="$allocated_nodes"' in source
    assert '--ntasks="$allocated_nodes"' in source
    assert "--ntasks-per-node=1" in source
    assert "desktop_fleet.supervise launch" in source


def test_launch_prepares_then_supervises_one_node_with_global_topology(
    monkeypatch,
    tmp_path,
):
    python = tmp_path / "venv" / "bin" / "python"
    env_server = python.with_name("env-server")
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    env_server.write_text("", encoding="utf-8")
    monkeypatch.setattr(supervise_module.sys, "executable", str(python))
    monkeypatch.setenv("ENV_FLEET_RUN_ID", "fleet-a")
    monkeypatch.setenv("ENV_FLEET_RUN_BASE", str(tmp_path / "runs"))
    monkeypatch.setenv("ENV_FLEET_ALLOCATED_NODES", "2")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("ENV_FLEET_SERVERS_PER_NODE", "3")
    monkeypatch.setenv("ENV_FLEET_REPLICA_COUNT", "6")
    calls = []
    monkeypatch.setattr(
        supervise_module,
        "prepare_main",
        lambda argv: calls.append(("prepare", argv)) or 0,
    )
    monkeypatch.setattr(
        supervise_module,
        "supervise_main",
        lambda argv: calls.append(("run", argv)) or 0,
    )

    assert launch_main([]) == 0

    assert calls[0] == ("prepare", [])
    run_args = calls[1][1]
    layout = FleetRunLayout.from_env(os.environ)
    assert run_args[run_args.index("--config-dir") + 1] == str(
        layout.node_configs_dir(0)
    )
    assert run_args[run_args.index("--env-server-bin") + 1] == str(env_server)
    assert "--start-gateway" in run_args


def test_launch_rejects_disagreement_between_slurm_and_submitted_topology(monkeypatch):
    monkeypatch.setenv("ENV_FLEET_ALLOCATED_NODES", "2")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")

    with pytest.raises(ValueError, match="fleet allocation has 3 nodes; expected 2"):
        launch_main([])


def test_fleet_parse_args_loads_runtime_env_file(
    monkeypatch,
    tmp_path,
    environment_contract_path,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"export SCRATCH={tmp_path / 'scratch'}",
                (
                    "export ENV_FLEET_ENVIRONMENT_CONTRACT="
                    f"{environment_contract_path}"
                ),
                "export ENV_FLEET_SLURM_PARTITION=cpu-partition",
                "export ENV_FLEET_SLURM_TIME=08:00:00",
                "export ENV_FLEET_SLURM_NODES=2",
                "export ENV_FLEET_SLURM_CPUS_PER_TASK=24",
                "export ENV_FLEET_SLURM_MEM_PER_NODE=96G",
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
    monkeypatch.setenv("ENV_FLEET_RUNTIME_ENV_FILE", str(env_file))
    monkeypatch.delenv("SCRATCH", raising=False)

    args = parse_fleet_args(["submit", "--dry-run"])

    assert args.run_base == tmp_path / "scratch"
    assert args.environment_contract == environment_contract_path
    assert args.account == "fallback-project"
    assert args.partition == "cpu-partition"
    assert args.time == "08:00:00"
    assert args.nodes == 2
    assert args.cpus_per_task == 24
    assert args.mem == "96G"


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


def test_status_format_and_registry_rendering(tmp_path, environment_metadata):
    layout = FleetRunLayout.for_run(
        run_id="12345",
        run_base=tmp_path / "runs",
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
        name_prefix="fixture",
        config_dir=layout.configs_dir,
        log_dir=layout.logs_dir,
        pool_status_root=layout.pool_status_dir,
    )[0]
    registry = upsert_registry(
        path=layout.registry_path,
        run_id="12345",
        metadata={
            **environment_metadata,
            "layout": layout.as_metadata(),
            "expected_env_servers": 1,
        },
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
                "name": "fixture-0000",
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
        name="env_fleet",
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


def test_env_value_rejects_a_malformed_environment_override():
    assert env_value({}, "ENV_FLEET_SERVERS_PER_NODE", int, 1) == 1
    with pytest.raises(ValueError, match="ENV_FLEET_SERVERS_PER_NODE"):
        env_value({"ENV_FLEET_SERVERS_PER_NODE": "8x"}, "ENV_FLEET_SERVERS_PER_NODE", int, 1)


def _runtime_root(tmp_path):
    return tmp_path.resolve() / "desktop-fleet-12345" / "node-0" / "desktop-runtime"


def test_supervisor_runtime_root_startup_and_shutdown_cleanup(tmp_path):
    runtime_root = _validate_local_runtime_root(_runtime_root(tmp_path))
    supervisor = object.__new__(FleetSupervisor)
    supervisor.local_runtime_root = runtime_root
    supervisor.logger = logging.getLogger("test-supervisor-runtime-cleanup")

    supervisor.prepare_local_runtime_root()
    stale = runtime_root / "runtime" / "stale.qcow2"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    supervisor.prepare_local_runtime_root()

    assert runtime_root.is_dir()
    assert not stale.exists()
    assert {path.name for path in runtime_root.iterdir()} == {LOCAL_RUNTIME_OWNER_FILE}
    (runtime_root / "new-runtime").mkdir()

    supervisor.cleanup_local_runtime_root()

    assert not runtime_root.exists()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"TMPDIR": None}, "TMPDIR must be set"),
        ({"SLURM_JOB_ID": None}, "SLURM_JOB_ID must be set"),
        ({"TMPDIR": "relative/base"}, "must be absolute"),
        ({"SLURM_JOB_ID": "../escape"}, "Invalid SLURM_JOB_ID"),
        ({"SLURM_PROCID": "node-one"}, "Invalid SLURM_PROCID"),
        ({"SLURM_PROCID": "-1"}, "Invalid SLURM_PROCID"),
    ],
)
def test_local_runtime_scope_rejects_a_base_it_cannot_own(overrides, match):
    """Nothing derived from a bad base may reach rmtree, so reject it here."""
    env = {"TMPDIR": "/scratch/local", "SLURM_JOB_ID": "12345", "SLURM_PROCID": "0"}
    env.update(overrides)

    with pytest.raises(RuntimeError, match=match):
        local_runtime_scope({k: v for k, v in env.items() if v is not None})


def test_supervisor_run_owns_the_runtime_root_across_the_whole_lifecycle(
    tmp_path,
    monkeypatch,
    recorded_signals,
    environment_metadata,
):
    """The root must exist before the first replica and be gone after the last."""
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
    supervisor = _build_supervisor(
        tmp_path,
        environment_metadata,
        policy=policy,
    )
    runtime_root = supervisor.local_runtime_root
    # what a requeued incarnation of this same job finds on the node
    runtime_root.mkdir(parents=True)
    _write_local_runtime_owner(runtime_root)
    stale = runtime_root / "stale.qcow2"
    stale.write_text("stale", encoding="utf-8")

    owned_at_spawn: list[bool] = []

    def fake_spawn(command, log_path):
        owned_at_spawn.append(
            (runtime_root / LOCAL_RUNTIME_OWNER_FILE).is_file() and not stale.exists()
        )
        return _FakeProcess(returncode=None)

    monkeypatch.setattr(supervisor, "spawn", fake_spawn)
    monkeypatch.setattr(
        time, "sleep", lambda seconds: setattr(supervisor, "stop_requested", True)
    )

    assert supervisor.run() == 0
    assert owned_at_spawn == [True]
    assert not runtime_root.exists()


def test_supervisor_runtime_root_cleanup_refuses_unowned_directory(tmp_path):
    runtime_root = _runtime_root(tmp_path)
    runtime_root.mkdir(parents=True)
    marker = runtime_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned local runtime root"):
        _remove_local_runtime_root(runtime_root)

    assert marker.exists()


def test_supervisor_runtime_root_cleanup_refuses_mismatched_owner(tmp_path):
    runtime_root = _runtime_root(tmp_path)
    runtime_root.mkdir(parents=True)
    owner = runtime_root / LOCAL_RUNTIME_OWNER_FILE
    owner.write_text(
        json.dumps(
            {
                "version": 1,
                "job_id": "another-job",
                "node_rank": 0,
                "runtime_root": str(runtime_root),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match this task"):
        _remove_local_runtime_root(runtime_root)

    assert runtime_root.exists()


def test_supervisor_runtime_root_cleanup_refuses_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    link = _runtime_root(tmp_path)
    link.parent.mkdir(parents=True)
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink in local runtime ownership path"):
        _remove_local_runtime_root(link)

    assert link.is_symlink()
    assert marker.exists()


def test_supervisor_runtime_root_cleanup_refuses_symlinked_job_root(tmp_path):
    target = tmp_path / "target"
    runtime_root = target / "node-0" / "desktop-runtime"
    runtime_root.mkdir(parents=True)
    marker = runtime_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    job_root = tmp_path.resolve() / "desktop-fleet-12345"
    job_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink in local runtime ownership path"):
        _remove_local_runtime_root(job_root / "node-0" / "desktop-runtime")

    assert marker.exists()


def test_supervisor_shutdown_cleanup_failure_is_nonfatal(tmp_path, monkeypatch):
    supervisor = object.__new__(FleetSupervisor)
    supervisor.local_runtime_root = tmp_path / "runtime"
    supervisor.logger = logging.getLogger("test-supervisor-runtime-cleanup-failure")

    def fail_cleanup(_path):
        raise OSError("cleanup failed")

    monkeypatch.setattr(supervise_module, "_remove_local_runtime_root", fail_cleanup)

    supervisor.cleanup_local_runtime_root()


def test_prepare_refuses_a_node_whose_allocation_cannot_hold_its_desktops(
    monkeypatch,
    tmp_path,
    environment_contract_path,
):
    """The budget must fire before prepare creates the run's directories."""
    monkeypatch.setattr(sys, "argv", ["desktop-fleet", "prepare"])
    monkeypatch.setenv("ENV_FLEET_RUN_BASE", str(tmp_path))
    monkeypatch.setenv("ENV_FLEET_RUN_ID", "run-oversubscribed")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "131072")
    monkeypatch.delenv("ENV_FLEET_DESKTOP_VM_HOST_MEM_GB", raising=False)
    for name in (
        "ENV_FLEET_RUN_ROOT",
        "ENV_FLEET_REGISTRY",
        "ENV_FLEET_DESKTOP_POOL_ROOT",
        "ENV_FLEET_DESKTOP_POOL_STATUS_DIR",
        "ENV_FLEET_LOGS_DIR",
        "ENV_FLEET_CONFIGS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="8 desktop VMs"):
        prepare_main(
            [
                "--servers-per-node",
                "1",
                "--workers-per-server",
                "8",
                "--environment-contract",
                str(environment_contract_path),
            ]
        )

    assert not (tmp_path / "run-oversubscribed").exists()


def test_vm_memory_budget_accepts_an_allocation_that_fits():
    enforce_vm_memory_budget(
        {"SLURM_MEM_PER_NODE": "262144"},
        servers_per_node=1,
        workers_per_server=8,
        min_ready_sessions=1,
    )


def test_vm_memory_budget_rejects_an_allocation_too_small_for_the_planned_vms():
    with pytest.raises(ValueError) as excinfo:
        enforce_vm_memory_budget(
            {"SLURM_MEM_PER_NODE": "131072"},
            servers_per_node=1,
            workers_per_server=8,
            min_ready_sessions=1,
        )

    message = str(excinfo.value)
    assert "8 desktop VMs" in message
    assert "1 servers x 8 workers x 1 ready sessions" in message
    assert "200 GB" in message
    assert "25 GB per VM" in message
    assert "128 GB" in message
    assert "ENV_FLEET_SLURM_MEM_PER_NODE" in message
    assert "ENV_FLEET_WORKERS_PER_SERVER" in message
    assert "contract's min_ready_sessions" in message
    assert "ENV_FLEET_DESKTOP_VM_HOST_MEM_GB" in message


def test_vm_memory_budget_derives_the_allocation_from_memory_per_cpu():
    with pytest.raises(ValueError, match="96 GB"):
        enforce_vm_memory_budget(
            {"SLURM_MEM_PER_CPU": "3072", "SLURM_CPUS_ON_NODE": "32"},
            servers_per_node=1,
            workers_per_server=8,
            min_ready_sessions=1,
        )


def test_vm_memory_budget_skips_without_a_slurm_memory_allocation():
    enforce_vm_memory_budget(
        {},
        servers_per_node=4,
        workers_per_server=8,
        min_ready_sessions=4,
    )


def test_vm_memory_budget_honors_the_per_vm_estimate_override():
    enforce_vm_memory_budget(
        {"SLURM_MEM_PER_NODE": "131072", "ENV_FLEET_DESKTOP_VM_HOST_MEM_GB": "16"},
        servers_per_node=1,
        workers_per_server=8,
        min_ready_sessions=1,
    )
