from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_fleet.spec import (
    FleetRunLayout,
    default_public_host,
    load_runtime_env_file,
    make_server_specs,
    parse_consumer_paths,
    project_root,
    render_consumer_paths,
    scratch_root,
    scratch_subdir,
    toml_literal,
    write_env_server_config,
)


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_fleet_run_layout_uses_readable_default_paths(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="12345",
        run_base=tmp_path / "osworld_rl",
    )

    assert layout.run_root == tmp_path / "osworld_rl" / "12345" / "env_fleet"
    assert layout.registry_path == layout.run_root / "env_registry.json"
    assert layout.pool_root == tmp_path / "osworld_rl" / "12345" / "pool"
    assert layout.pool_status_dir == layout.pool_root / "status"
    assert layout.logs_dir == layout.run_root / "logs"
    assert layout.configs_dir == layout.run_root / "configs"
    assert layout.run_dir == tmp_path / "osworld_rl" / "12345"
    assert layout.consumer_paths == {}


def test_fleet_run_layout_rejects_relative_constructor_paths(tmp_path):
    with pytest.raises(ValueError, match="run_base must be an absolute path"):
        FleetRunLayout.for_run(run_id="run-a", run_base="relative/base")

    with pytest.raises(ValueError, match="registry_path must be an absolute path"):
        FleetRunLayout.for_run(
            run_id="run-a",
            run_base=tmp_path / "base",
            registry_path="relative/registry.json",
        )


def test_fleet_run_layout_honors_environment_overrides(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
        "OSWORLD_RUN_BASE": str(tmp_path / "base"),
        "OSWORLD_FLEET_RUN_ROOT": str(tmp_path / "custom-desktop-fleet"),
        "OSWORLD_DESKTOP_POOL_ROOT": str(tmp_path / "custom-desktop-pool"),
        "OSWORLD_ENV_FLEET_REGISTRY": str(tmp_path / "registry.json"),
    }

    layout = FleetRunLayout.from_env(env)

    assert layout.run_id == "run-a"
    assert layout.run_base == tmp_path / "base"
    assert layout.run_root == tmp_path / "custom-desktop-fleet"
    assert layout.pool_root == tmp_path / "custom-desktop-pool"
    assert layout.pool_status_dir == tmp_path / "custom-desktop-pool" / "status"
    assert layout.registry_path == tmp_path / "registry.json"


def test_fleet_run_layout_defaults_to_short_scratch_run_base(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
    }

    layout = FleetRunLayout.from_env(env)

    assert layout.run_base == tmp_path / "scratch"
    assert layout.pool_root == tmp_path / "scratch" / "run-a" / "pool"


def test_fleet_run_layout_keeps_explicit_paths_literal(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "13969570",
        "OSWORLD_RUN_BASE": str(tmp_path / "shared" / "osworld_rl"),
        "OSWORLD_FLEET_RUN_ROOT": str(tmp_path / "custom" / "env_fleet"),
        "OSWORLD_ENV_FLEET_REGISTRY": str(tmp_path / "custom" / "registry.json"),
        "OSWORLD_DESKTOP_POOL_ROOT": str(tmp_path / "custom" / "desktop_pool"),
        "OSWORLD_DESKTOP_POOL_STATUS_DIR": str(tmp_path / "custom" / "status"),
    }

    layout = FleetRunLayout.from_env(env)

    assert layout.run_base == tmp_path / "shared" / "osworld_rl"
    assert layout.run_root == tmp_path / "custom" / "env_fleet"
    assert layout.registry_path == tmp_path / "custom" / "registry.json"
    assert layout.pool_root == tmp_path / "custom" / "desktop_pool"
    assert layout.pool_status_dir == tmp_path / "custom" / "status"


def test_fleet_run_layout_defaults_without_scratch_or_project(monkeypatch):
    monkeypatch.delenv("SCRATCH", raising=False)

    layout = FleetRunLayout.from_env({}, run_id="manual")

    assert layout.run_base == project_root({}) / ".scratch"
    assert layout.registry_path == (
        project_root({}) / ".scratch" / "manual" / "env_fleet" / "env_registry.json"
    )


def test_fleet_run_layout_carries_opaque_consumer_paths(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
        "ENV_FLEET_CONSUMER_PATHS": (
            f"trainer_config_path={tmp_path / 'trainer.toml'},"
            f"trainer_output_dir={tmp_path / 'out'}"
        ),
    }

    layout = FleetRunLayout.from_env(env)

    assert layout.consumer_paths == {
        "trainer_config_path": tmp_path / "trainer.toml",
        "trainer_output_dir": tmp_path / "out",
    }
    assert layout.consumer_path("trainer_config_path") == tmp_path / "trainer.toml"
    assert layout.as_metadata()["trainer_output_dir"] == str(tmp_path / "out")

    round_tripped = FleetRunLayout.from_metadata({"layout": layout.as_metadata()})
    assert round_tripped is not None
    assert round_tripped.consumer_paths == layout.consumer_paths

    rendered = render_consumer_paths(layout.consumer_paths)
    assert parse_consumer_paths(rendered) == layout.consumer_paths
    with pytest.raises(KeyError):
        layout.consumer_path("missing_path")


def test_consumer_paths_must_be_absolute():
    with pytest.raises(ValueError, match="must be an absolute path"):
        parse_consumer_paths("trainer_config_path=relative/trainer.toml")

    with pytest.raises(ValueError, match="entries must be name=path"):
        parse_consumer_paths("trainer_config_path")


def test_make_server_specs_assigns_ports_and_replicas(tmp_path):
    specs = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=2,
        servers_per_node=2,
        workers_per_server=3,
        replica_count=8,
        replica_offset=4,
        name_prefix="osworld",
        config_dir=tmp_path / "configs",
        log_dir=tmp_path / "logs",
        pool_status_root=tmp_path / "status",
    )

    assert [spec.name for spec in specs] == ["osworld-0004", "osworld-0005"]
    assert [spec.port for spec in specs] == [5204, 5205]
    assert [spec.replica_index for spec in specs] == [4, 5]
    assert [spec.replica_count for spec in specs] == [8, 8]
    assert specs[0].public_address == "tcp://node001:5204"
    assert specs[0].num_workers == 3
    assert specs[0].pool_status_dir == str(tmp_path / "status" / "osworld-0004")


def test_default_public_host_prefers_explicit_env():
    host = default_public_host(
        {"OSWORLD_FLEET_HOST": "custom-host"},
        run_command=lambda _command: "NodeAddr=slurm-host",
    )

    assert host == "custom-host"


def test_default_public_host_prefers_slurm_node_addr():
    commands: list[list[str]] = []

    def run_command(command: list[str]) -> str:
        commands.append(command)
        return "NodeName=jwb0127 Arch=x86_64 NodeAddr=jwb0127i NodeHostName=jwb0127"

    host = default_public_host(
        {"SLURMD_NODENAME": "jwb0127"},
        run_command=run_command,
        fqdn_func=lambda: "jwb0127.juwels",
        hostname_func=lambda: "jwb0127",
    )

    assert host == "jwb0127i"
    assert commands == [["scontrol", "show", "node", "jwb0127"]]


def test_default_public_host_falls_back_to_fqdn():
    host = default_public_host(
        {},
        run_command=lambda _command: "",
        fqdn_func=lambda: "jwb0127.juwels",
        hostname_func=lambda: "jwb0127",
    )

    assert host == "jwb0127.juwels"


def test_env_server_config_writes_taskset_without_partition_keys(tmp_path):
    server = make_server_specs(
        host="node001",
        bind_host="0.0.0.0",
        base_port=5200,
        node_rank=0,
        servers_per_node=1,
        workers_per_server=2,
        replica_count=4,
        replica_offset=2,
        name_prefix="osworld",
        config_dir=tmp_path,
        log_dir=tmp_path,
        pool_status_root=tmp_path / "status",
    )[0]
    args = SimpleNamespace(
        task_base_path=tmp_path / "tasks",
        max_tasks=5,
        shuffle_seed=11,
        max_steps=4,
        run_root=tmp_path / "run",
        env_id="rl",
        rollout_timeout=3600.0,
        env_max_retries=2,
    )

    write_env_server_config(
        server,
        args,
        {
            "harness": {
                "max_steps": 4,
                "desktop": {
                    "desktop_pool_config": {
                        "runtime_dir": "/tmp/osworld-runtime",
                    },
                },
            }
        },
    )

    with Path(server.config_path).open("rb") as file:
        config = tomllib.load(file)
    env = config["env"]
    assert env["taskset"] == {
        "id": "rl",
        "base_path": str(tmp_path / "tasks"),
        "max_tasks": 5,
        "shuffle_seed": 11,
    }
    assert env["pool"] == {"type": "static", "num_workers": 2}
    assert env["timeout"] == {"rollout": 3600.0}
    assert env["retries"] == {"rollout": {"max_retries": 2}}
    assert env["max_turns"] == 4
    pool = env["harness"]["desktop"]["desktop_pool_config"]
    assert pool["status_dir"] == server.pool_status_dir
    assert pool["runtime_dir"] == "/tmp/osworld-runtime"


def test_toml_literal_renders_nested_inline_tables():
    rendered = toml_literal({"config": {"taskset": {"base_path": "/tasks"}}})

    assert rendered == '{ config = { taskset = { base_path = "/tasks" } } }'


def test_scratch_helpers_default_to_project_scratch(monkeypatch):
    monkeypatch.delenv("SCRATCH", raising=False)

    assert scratch_root({}) == project_root({}) / ".scratch"
    assert (
        scratch_subdir("osworld_rl", env={})
        == project_root({}) / ".scratch" / "osworld_rl"
    )


def test_scratch_helpers_honor_runtime_overrides(tmp_path):
    env = {"SCRATCH": str(tmp_path / "scratch")}

    assert scratch_root(env) == tmp_path / "scratch"
    assert scratch_subdir("osworld_rl", env=env) == tmp_path / "scratch" / "osworld_rl"


def test_fleet_run_layout_rejects_non_absolute_environment_paths(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
        "OSWORLD_RUN_BASE": "relative/run-base",
    }

    with pytest.raises(ValueError, match="OSWORLD_RUN_BASE must be an absolute path"):
        FleetRunLayout.from_env(env)

    env["OSWORLD_RUN_BASE"] = str(tmp_path / "run-base")
    env["OSWORLD_ENV_FLEET_REGISTRY"] = "${SCRATCH}/registry.json"

    with pytest.raises(
        ValueError,
        match="OSWORLD_ENV_FLEET_REGISTRY must be an absolute path",
    ):
        FleetRunLayout.from_env(env)


def test_runtime_env_file_rejects_non_absolute_path_overrides():
    with pytest.raises(ValueError, match="runtime env file path must be an absolute"):
        load_runtime_env_file(path="runtime.env")

    with pytest.raises(ValueError, match="RL_RUNTIME_ENV_FILE must be an absolute"):
        load_runtime_env_file(env={"RL_RUNTIME_ENV_FILE": "${SCRATCH}/runtime.env"})


def test_runtime_env_file_loads_shell_style_assignments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local runtime settings",
                "PROJECT_ROOT=/tmp/project",
                "export SCRATCH=/tmp/scratch",
                'HF_HOME="${SCRATCH}/huggingface"',
                "OSWORLD_ROOT=${PROJECT_ROOT}/OSWorldRL",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    loaded = load_runtime_env_file(env=env, path=env_file)

    assert loaded == env_file
    assert env == {
        "PROJECT_ROOT": "/tmp/project",
        "SCRATCH": "/tmp/scratch",
        "HF_HOME": "/tmp/scratch/huggingface",
        "OSWORLD_ROOT": "/tmp/project/OSWorldRL",
    }


def test_runtime_env_file_uses_python_dotenv_interpolation(monkeypatch, tmp_path):
    monkeypatch.delenv("SCRATCH", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('HF_HOME="${SCRATCH}/huggingface"\n', encoding="utf-8")
    env: dict[str, str] = {}

    assert load_runtime_env_file(env=env, path=env_file) == env_file
    assert env == {"HF_HOME": "/huggingface"}


def test_an_already_exported_variable_beats_the_runtime_env_file(tmp_path):
    """The file supplies defaults; the caller's environment is the authority.

    This loader runs before argparse, so it is what every environment-backed
    ``Opt`` reads.  Overwriting here silently discards
    ``OSWORLD_FLEET_BASE_PORT=5300 python -m desktop_fleet.supervise ...``.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OSWORLD_FLEET_BASE_PORT=5400\nOSWORLD_FLEET_HOST=from-file\n", encoding="utf-8"
    )
    env = {"OSWORLD_FLEET_BASE_PORT": "5300"}

    assert load_runtime_env_file(env=env, path=env_file) == env_file
    assert env == {
        "OSWORLD_FLEET_BASE_PORT": "5300",
        "OSWORLD_FLEET_HOST": "from-file",
    }


def test_runtime_env_file_can_be_disabled():
    env = {"RL_RUNTIME_ENV_FILE": ""}

    assert load_runtime_env_file(env=env) is None


def test_runtime_env_file_rejects_invalid_lines(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("not-an-assignment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected KEY=value"):
        load_runtime_env_file(path=env_file)
