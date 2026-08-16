from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_fleet.adapters import prime_rl
from desktop_fleet.spec import FleetRunLayout
from desktop_fleet.supervise import format_submit_report


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("RL_RUNTIME_ENV_FILE", "")


def test_prime_rl_paths_default_under_the_run_dir(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="12345",
        run_base=tmp_path / "osworld_rl",
    )

    assert prime_rl.config_path(layout) == (
        tmp_path / "osworld_rl" / "12345" / "prime_rl_fleet.toml"
    )
    assert prime_rl.output_dir(layout) == tmp_path / "osworld_rl" / "12345" / "prime_rl"


def test_prime_rl_paths_honor_environment_overrides(tmp_path):
    env = {
        "SCRATCH": str(tmp_path / "scratch"),
        "OSWORLD_FLEET_RUN_ID": "run-a",
        "OSWORLD_RUN_BASE": str(tmp_path / "base"),
        "OSWORLD_PRIME_RL_CONFIG_PATH": str(tmp_path / "prime_rl.toml"),
        "OSWORLD_PRIME_RL_OUTPUT_DIR": str(tmp_path / "prime_rl"),
    }

    layout = prime_rl.with_prime_rl_paths(FleetRunLayout.from_env(env), env)

    assert prime_rl.config_path(layout) == tmp_path / "prime_rl.toml"
    assert prime_rl.output_dir(layout) == tmp_path / "prime_rl"
    assert layout.as_metadata()["prime_rl_config_path"] == str(
        tmp_path / "prime_rl.toml"
    )
    assert layout.as_metadata()["prime_rl_output_dir"] == str(tmp_path / "prime_rl")


def test_prime_rl_paths_survive_a_registry_metadata_round_trip(tmp_path):
    env = {
        "OSWORLD_PRIME_RL_CONFIG_PATH": str(tmp_path / "custom" / "prime_rl.toml"),
        "OSWORLD_PRIME_RL_OUTPUT_DIR": str(tmp_path / "custom" / "prime_rl_output"),
    }
    layout = prime_rl.with_prime_rl_paths(
        FleetRunLayout.for_run(run_id="13969570", run_base=tmp_path / "shared"),
        env,
    )

    restored = FleetRunLayout.from_metadata({"layout": layout.as_metadata()})

    assert restored is not None
    assert prime_rl.config_path(restored) == tmp_path / "custom" / "prime_rl.toml"
    assert prime_rl.output_dir(restored) == tmp_path / "custom" / "prime_rl_output"
    assert prime_rl.consumer_paths_env_value(layout, env) == (
        f"prime_rl_config_path={tmp_path / 'custom' / 'prime_rl.toml'},"
        f"prime_rl_output_dir={tmp_path / 'custom' / 'prime_rl_output'}"
    )


def test_render_prime_rl_fleet_config_uses_v1_schema(tmp_path):
    metadata = {
        "env_id": "rl",
        "env_name_prefix": "osworld",
        "task_base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
        "gateway": {"public_address": "tcp://node001:5200"},
        "harness": {
            "max_steps": 4,
            "desktop": {
                "output_dir": str(tmp_path / "worker-output"),
                "cache_dir": "/scratch/user/cache",
                "desktop_pool_config": {
                    "min_ready_sessions": 1,
                    "max_sessions": 3,
                },
            },
        },
    }
    config = {
        "output_dir": "/old",
        "orchestrator": {
            "max_inflight_rollouts": 1,
            "train": {"env": [{"name": "old"}]},
        },
        "inference": {"gpu_memory_utilization": 0.85},
    }

    prime_rl.configure_external_fleet(
        config,
        metadata=metadata,
        output_dir=tmp_path / "trainer-output",
        max_inflight_rollouts=2,
        rollout_timeout=3600,
        max_retries=1,
    )

    env = config["orchestrator"]["train"]["env"][0]
    assert config["output_dir"] == str(tmp_path / "trainer-output")
    assert config["orchestrator"]["max_inflight_rollouts"] == 2
    assert env["address"] == "tcp://node001:5200"
    assert env["taskset"] == {
        "id": "rl",
        "base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
    }
    assert env["harness"]["id"] == "rl"
    assert env["harness"]["desktop"]["desktop_pool_config"] == {
        "min_ready_sessions": 0,
        "max_sessions": 3,
    }
    assert env["timeout"] == {"rollout": 3600}
    assert env["retries"] == {"rollout": {"max_retries": 1}}
    assert env["max_turns"] == 4
    assert config["inference"] == {"gpu_memory_utilization": 0.85}


def test_render_prime_rl_fleet_config_requires_gateway_address(tmp_path):
    metadata = {
        "env_id": "rl",
        "env_name_prefix": "osworld",
        "task_base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
        "harness": {"max_steps": 4},
    }

    with pytest.raises(ValueError, match="gateway.public_address"):
        prime_rl.external_env_config(
            metadata,
            rollout_timeout=3600,
            max_retries=1,
        )


def test_render_prime_rl_fleet_config_uses_gateway_address(tmp_path):
    metadata = {
        "env_id": "rl",
        "env_name_prefix": "osworld",
        "task_base_path": "/tasks",
        "max_tasks": 4,
        "shuffle_seed": 7,
        "harness": {"max_steps": 4},
        "gateway": {
            "bind_address": "tcp://0.0.0.0:5202",
            "public_address": "tcp://node001:5202",
            "backend_addresses": ["tcp://node001:5200", "tcp://node001:5201"],
        },
    }

    rendered = prime_rl.external_env_config(
        metadata,
        rollout_timeout=3600,
        max_retries=1,
    )

    assert rendered["address"] == "tcp://node001:5202"
    assert "tcp://node001:5200" not in json.dumps(rendered)


def test_submit_report_prints_prime_rl_next_steps(tmp_path):
    layout = FleetRunLayout.for_run(
        run_id="fleet-a",
        run_base=tmp_path / "runs",
    )

    report = format_submit_report(
        "12345",
        layout,
        SimpleNamespace(),
        trainer_section=prime_rl.format_trainer_command(layout),
    )

    assert "Render config and launch the consumer:" in report
    assert "-m desktop_fleet.adapters.prime_rl render" in report
    assert "prime_rl_fleet.toml" in report
    assert "Readiness:" in report
    assert "uv run --no-sync python -m desktop_fleet.supervise status" in report
    assert "uv run --no-sync python -m desktop_fleet.readiness" in report
    assert "uv run --no-sync python scripts/prime_rl.py" in report
    assert f"{prime_rl.config_path(layout)} --clean-output-dir" in report
    assert ".venv/bin/rl" not in report
    assert "rl/verifiers/prime-rl" not in report
    assert "Cancel:" in report


def test_prime_rl_dir_prefers_explicit_environment(tmp_path):
    assert prime_rl.prime_rl_dir({"PRIME_RL_DIR": str(tmp_path / "pr")}) == (
        tmp_path / "pr"
    )
    with pytest.raises(ValueError, match="PRIME_RL_DIR must be an absolute path"):
        prime_rl.prime_rl_dir({"PRIME_RL_DIR": "relative/prime-rl"})


def test_absolutize_slurm_template_path_leaves_absolute_paths_alone(tmp_path):
    absolute = tmp_path / "template.sbatch.j2"
    absolute.write_text("", encoding="utf-8")
    config = {"slurm": {"template_path": str(absolute)}}

    prime_rl.absolutize_slurm_template_path(config)

    assert config["slurm"]["template_path"] == str(absolute.resolve())

    config_without_slurm: dict[str, object] = {}
    prime_rl.absolutize_slurm_template_path(config_without_slurm)
    assert config_without_slurm == {}


def test_prime_rl_resolve_layout_prefers_registry_metadata(tmp_path):
    layout = FleetRunLayout.for_run(run_id="run", run_base=tmp_path / "base")
    args = SimpleNamespace(
        run_id="fallback",
        run_base=tmp_path / "other",
        registry=Path(tmp_path / "registry.json"),
    )

    resolved = prime_rl.resolve_layout(args, {"layout": layout.as_metadata()})

    assert resolved.run_id == "run"
    assert resolved.run_root == layout.run_root
