from __future__ import annotations

import json

import pytest

from desktop_fleet.environment import (
    EnvironmentContract,
    EnvironmentSession,
    EnvironmentSource,
)


@pytest.fixture
def environment_contract(tmp_path) -> EnvironmentContract:
    return EnvironmentContract(
        session=EnvironmentSession(
            taskset={"id": "fixture-taskset", "dataset": "/datasets/tasks"},
            harness={
                "id": "fixture-harness",
                "runner": {
                    "pool": {
                        "min_ready_sessions": 1,
                        "max_sessions": 3,
                        "node_local_path": "/runtime/node-local.json",
                    }
                },
            },
            output_dir=tmp_path / "server-output",
            rollout_timeout=600.0,
            max_retries=2,
            max_turns=4,
            status_dir_path=("runner", "pool", "status_dir"),
        ),
        source=EnvironmentSource(
            name="fixture",
            harness_overrides={"runner": {"pool": {"min_ready_sessions": 0}}},
            harness_omit_paths=(("runner", "pool", "node_local_path"),),
        ),
    )


@pytest.fixture
def environment_metadata(environment_contract) -> dict[str, object]:
    return {"environment": environment_contract.as_metadata()}


@pytest.fixture
def environment_contract_path(tmp_path, environment_contract):
    path = tmp_path / "environment.json"
    path.write_text(
        json.dumps(environment_contract.as_metadata()),
        encoding="utf-8",
    )
    return path
