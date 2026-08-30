from __future__ import annotations

import json

import pytest

from desktop_fleet.environment import (
    EnvironmentContract,
    EnvironmentSession,
    EnvironmentSource,
    environment_contract_from_registry_metadata,
    read_environment_contract,
)


def test_environment_contract_round_trips_exactly(environment_contract, tmp_path):
    payload = environment_contract.as_metadata()
    restored = EnvironmentContract.from_metadata(payload)

    assert restored.as_metadata() == payload

    path = tmp_path / "environment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_environment_contract(path).as_metadata() == payload


def test_environment_contract_path_must_be_absolute():
    with pytest.raises(ValueError, match="must be absolute"):
        read_environment_contract("environment.json")


def test_status_dir_path_is_required_exact_and_unoccupied(tmp_path):
    values = {
        "taskset": {"id": "tasks"},
        "harness": {"runner": {"pool": {"min_ready_sessions": 1}}},
        "output_dir": tmp_path / "output",
        "rollout_timeout": 60.0,
        "max_retries": 1,
        "max_turns": 2,
    }

    with pytest.raises(ValueError, match="non-empty string keys"):
        EnvironmentSession(**values, status_dir_path=())
    with pytest.raises(ValueError, match="parent does not exist"):
        EnvironmentSession(**values, status_dir_path=("other", "status_dir"))
    with pytest.raises(ValueError, match="target already exists"):
        EnvironmentSession(
            **{
                **values,
                "harness": {
                    "runner": {
                        "pool": {
                            "min_ready_sessions": 1,
                            "status_dir": "/old",
                        }
                    }
                },
            },
            status_dir_path=("runner", "pool", "status_dir"),
        )


def test_readiness_is_read_from_the_exact_status_path_parent(environment_contract):
    assert environment_contract.session.min_ready_sessions == 1
    assert "min_ready_sessions" not in environment_contract.session.as_metadata()


def test_source_configs_are_independent_of_the_server_session(
    environment_contract,
):
    assert environment_contract.source.taskset == {
        "id": "consumer-taskset",
        "dataset": "/datasets/consumer-tasks",
    }
    assert environment_contract.source.harness == {"id": "consumer-harness"}
    assert environment_contract.session.taskset["id"] == "server-taskset"
    assert environment_contract.session.harness["id"] == "server-harness"
    assert environment_contract.session.harness["runner"]["pool"] == {
        "min_ready_sessions": 1,
        "max_sessions": 3,
        "node_local_path": "/runtime/node-local.json",
    }


def test_source_config_inputs_are_copied():
    taskset = {
        "id": "consumer-taskset",
        "options": {"split": "original"},
    }
    harness = {
        "id": "consumer-harness",
        "options": {"mode": "original"},
    }
    source = EnvironmentSource(
        name="consumer-source",
        taskset=taskset,
        harness=harness,
    )

    taskset["options"]["split"] = "mutated"
    harness["options"]["mode"] = "mutated"

    assert source.taskset["options"] == {"split": "original"}
    assert source.harness["options"] == {"mode": "original"}


def test_contract_rejects_the_removed_source_transformation_shape(
    environment_contract,
):
    payload = environment_contract.as_metadata()
    payload["source"] = {
        "name": "consumer-source",
        "harness_overrides": {},
        "harness_omit_paths": [],
    }

    with pytest.raises(ValueError, match="environment source keys do not match"):
        EnvironmentContract.from_metadata(payload)


def test_version_one_metadata_fails_clearly(environment_contract):
    payload = environment_contract.as_metadata()
    payload["version"] = 1

    with pytest.raises(ValueError, match="unsupported environment contract version: 1"):
        EnvironmentContract.from_metadata(payload)


@pytest.mark.parametrize(
    "name",
    ["../escape", "with.dot", "two words", " padded ", "", "x" * 65],
)
def test_source_name_must_be_a_safe_path_identifier(environment_contract, name):
    payload = environment_contract.as_metadata()
    payload["source"]["name"] = name

    with pytest.raises(ValueError, match="source.name must be"):
        EnvironmentContract.from_metadata(payload)


def test_contract_rejects_values_toml_cannot_represent(environment_contract):
    payload = environment_contract.as_metadata()
    payload["session"]["taskset"]["unsupported"] = None

    with pytest.raises(ValueError, match="unsupported by fleet TOML"):
        EnvironmentContract.from_metadata(payload)


@pytest.mark.parametrize("bad_value", [{"bad\ud800": "value"}, {"key": "bad\ud800"}])
def test_contract_rejects_non_utf8_config_strings_and_keys(
    environment_contract,
    bad_value,
):
    payload = environment_contract.as_metadata()
    payload["session"]["taskset"].update(bad_value)

    with pytest.raises(ValueError, match="valid UTF-8"):
        EnvironmentContract.from_metadata(payload)


def test_registry_metadata_rejects_the_removed_flat_environment_shape():
    with pytest.raises(ValueError, match="legacy flat environment metadata"):
        environment_contract_from_registry_metadata(
            {
                "env_id": "removed",
                "task_base_path": "/tasks",
                "harness": {"id": "removed"},
            }
        )


def test_contract_rejects_unrecognized_serialized_fields(environment_contract):
    payload = environment_contract.as_metadata()
    payload["session"]["fallback_status_dir"] = "/status"

    with pytest.raises(ValueError, match=r"unexpected=\['fallback_status_dir'\]"):
        EnvironmentContract.from_metadata(payload)
