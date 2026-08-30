from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_fleet.adapters import prime_rl
from desktop_fleet.spec import FleetRunLayout
from desktop_fleet.supervise import format_submit_report


@pytest.fixture(autouse=True)
def disable_runtime_env_file(monkeypatch):
    monkeypatch.setenv("ENV_FLEET_RUNTIME_ENV_FILE", "")


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
        "ENV_FLEET_RUN_ID": "run-a",
        "ENV_FLEET_RUN_BASE": str(tmp_path / "base"),
        "ENV_FLEET_PRIME_RL_CONFIG_PATH": str(tmp_path / "prime_rl.toml"),
        "ENV_FLEET_PRIME_RL_OUTPUT_DIR": str(tmp_path / "prime_rl"),
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
        "ENV_FLEET_PRIME_RL_CONFIG_PATH": str(tmp_path / "custom" / "prime_rl.toml"),
        "ENV_FLEET_PRIME_RL_OUTPUT_DIR": str(tmp_path / "custom" / "prime_rl_output"),
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


def test_render_prime_rl_fleet_config_uses_current_source_schema(
    tmp_path,
    environment_metadata,
):
    metadata = {
        **environment_metadata,
        "gateway": {"public_address": "tcp://node001:5200"},
    }
    config = {
        "output_dir": "/old",
        "orchestrator": {
            "max_inflight_episodes": 1,
            "group_size": 4,
            "train": {"source": [{"name": "old"}]},
        },
        "inference": {"gpu_memory_utilization": 0.85},
    }

    prime_rl.configure_external_fleet(
        config,
        metadata=metadata,
        output_dir=tmp_path / "trainer-output",
        max_inflight_episodes=2,
    )

    source = config["orchestrator"]["train"]["source"][0]
    assert config["output_dir"] == str(tmp_path / "trainer-output")
    assert config["orchestrator"]["max_inflight_episodes"] == 4
    assert "max_inflight_rollouts" not in config["orchestrator"]
    assert "env" not in config["orchestrator"]["train"]
    assert source == {
        "name": "consumer-source",
        "env": {
            "taskset": {
                "id": "consumer-taskset",
                "dataset": "/datasets/consumer-tasks",
            },
            "agent": {
                "harness": {"id": "consumer-harness"},
                "timeout": {"rollout": 600.0},
                "retries": {"max_retries": 2},
                "max_turns": 4,
            },
        },
        "serve": {"address": "tcp://node001:5200"},
    }
    assert config["inference"] == {"gpu_memory_utilization": 0.85}


def test_render_prime_rl_fleet_config_requires_group_size(environment_metadata):
    config = {
        "output_dir": "/old",
        "orchestrator": {"train": {"source": []}},
    }

    with pytest.raises(ValueError, match="group_size"):
        prime_rl.configure_external_fleet(
            config,
            metadata={
                **environment_metadata,
                "gateway": {"public_address": "tcp://node001:5200"},
            },
            output_dir=Path("/out"),
            max_inflight_episodes=2,
        )


@pytest.mark.parametrize(
    ("stale_key", "message"),
    [
        ("train.env", "orchestrator.train.env is stale"),
        ("max_inflight_rollouts", "orchestrator.max_inflight_rollouts is stale"),
    ],
)
def test_render_prime_rl_fleet_config_rejects_stale_base_schema(
    environment_metadata,
    stale_key,
    message,
):
    config = {
        "orchestrator": {
            "group_size": 4,
            "train": {"source": []},
        }
    }
    if stale_key == "train.env":
        config["orchestrator"]["train"]["env"] = []
    else:
        config["orchestrator"][stale_key] = 1

    with pytest.raises(ValueError, match=message):
        prime_rl.configure_external_fleet(
            config,
            metadata={
                **environment_metadata,
                "gateway": {"public_address": "tcp://node001:5200"},
            },
            output_dir=Path("/out"),
            max_inflight_episodes=2,
        )


def test_render_prime_rl_fleet_config_requires_gateway_address(environment_metadata):
    with pytest.raises(ValueError, match="gateway.public_address"):
        prime_rl.external_source_config(environment_metadata)


def test_render_prime_rl_fleet_config_uses_gateway_address(environment_metadata):
    metadata = {
        **environment_metadata,
        "gateway": {
            "bind_address": "tcp://0.0.0.0:5202",
            "public_address": "tcp://node001:5202",
            "backend_addresses": ["tcp://node001:5200", "tcp://node001:5201"],
        },
    }

    rendered = prime_rl.external_source_config(metadata)

    assert rendered["serve"]["address"] == "tcp://node001:5202"


def test_render_refuses_a_partial_registry_before_counting_workers(
    tmp_path,
    monkeypatch,
):
    registry_path = tmp_path / "registry.json"
    registry = SimpleNamespace(
        servers=[object()],
        metadata={"expected_env_servers": 2, "expected_env_workers": 2},
    )
    monkeypatch.setattr(
        prime_rl,
        "parse_render_args",
        lambda _argv: SimpleNamespace(registry=registry_path),
    )
    monkeypatch.setattr(prime_rl, "read_registry", lambda _path: registry)

    with pytest.raises(ValueError, match="expected 2 env servers, found 1"):
        prime_rl.render_main([])


def test_prime_rl_validation_uses_the_pinned_checkouts_parser(
    tmp_path,
    monkeypatch,
):
    package_dir = tmp_path / "prime-rl"
    python = package_dir / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    config = tmp_path / "generated.toml"
    config.write_text("output_dir = '/tmp/output'\n", encoding="utf-8")
    invocation = {}

    def run(command, **kwargs):
        invocation.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(prime_rl, "prime_rl_dir", lambda: package_dir)
    monkeypatch.setenv("PYTHONHOME", "/wrong/python-home")
    monkeypatch.setenv("PYTHONPATH", "/wrong/import-path")
    monkeypatch.setattr(prime_rl.subprocess, "run", run)

    prime_rl.validate_prime_rl_config(config)

    assert invocation["command"][0] == str(python)
    assert "from prime_rl.configs.rl import RLConfig" in invocation["command"][2]
    assert "RLConfig.model_validate" in invocation["command"][2]
    assert invocation["command"][3] == str(config)
    assert invocation["cwd"] == config.parent
    assert "PYTHONHOME" not in invocation["env"]
    assert "PYTHONPATH" not in invocation["env"]
    assert invocation["env"]["VIRTUAL_ENV"] == str(package_dir / ".venv")
    assert invocation["env"]["PATH"].split(":", 1)[0] == str(
        package_dir / ".venv" / "bin"
    )


def test_submit_report_prints_prime_rl_next_steps(tmp_path, monkeypatch):
    package_dir = tmp_path / "trainer-runtime"
    launcher = package_dir / ".venv" / "bin" / "rl"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PRIME_RL_DIR", str(package_dir))
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
    assert str(launcher) in report
    assert f"{prime_rl.config_path(layout)} --clean-output-dir" in report
    assert "scripts/prime_rl.py" not in report
    assert "rl/verifiers/prime-rl" not in report
    assert "unset PYTHONHOME PYTHONPATH" in report
    assert f"export VIRTUAL_ENV={package_dir / '.venv'}" in report
    assert f'export PATH={package_dir / ".venv" / "bin"}:"$PATH"' in report
    assert "Cancel:" in report


def test_prime_rl_dir_requires_explicit_environment(tmp_path):
    package_dir = tmp_path / "pr"
    package_dir.mkdir()
    assert prime_rl.prime_rl_dir({"PRIME_RL_DIR": str(package_dir)}) == package_dir
    with pytest.raises(ValueError, match="PRIME_RL_DIR is required"):
        prime_rl.prime_rl_dir({})
    with pytest.raises(ValueError, match="PRIME_RL_DIR must be an absolute path"):
        prime_rl.prime_rl_dir({"PRIME_RL_DIR": "relative/prime-rl"})
    with pytest.raises(ValueError, match="PRIME_RL_DIR is not a directory"):
        prime_rl.prime_rl_dir({"PRIME_RL_DIR": str(tmp_path / "missing")})


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
