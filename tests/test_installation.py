"""Installed-interface checks; no hosted inference or archived benchmark inputs."""

import base64
import io
import json
import sys

import pytest
from PIL import Image

from visual_repair_lab import VisualTrainingEnv, make_env
from visual_repair_lab.cli import FAMILIES, main


@pytest.mark.parametrize("family", FAMILIES)
def test_facade_accepts_json_action(family):
    env = make_env(family=family, split="development")
    try:
        assert isinstance(env, VisualTrainingEnv)
        observation, _ = env.reset(seed=3110026)
        assert env.observation_space.contains(observation)
        _, reward, terminated, truncated, _ = env.step('{"type":"observe"}')
        assert reward == -.025 and not terminated and not truncated
    finally:
        env.close()


def test_complete_offline_installation_smoke(tmp_path, capsys):
    output = tmp_path / "smoke.json"
    assert main(["smoke", "--output", str(output)]) == 0
    receipt = json.loads(output.read_text())
    assert receipt == json.loads(capsys.readouterr().out)
    assert receipt["status"] == "passed" and len(receipt["checks"]) == 18
    assert receipt["runtime"]["release"] == "7.0.0"
    assert receipt["runtime"]["environment"] == "VisualTrainingRepair-v0"
    assert receipt["runtime"]["default_action_budget"] == 48
    assert set(receipt["runtime"]["dependencies"]) == {"numpy", "gymnasium", "Pillow"}
    assert all(receipt["runtime"]["module_paths"].values())


def test_rollout_is_public_observation_only(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"observe"}\n'))
    assert main(["rollout", "--max-steps", "1", "--split", "development", "--seed", "3110026"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 2
    assert rows[-1]["truncated"] and not rows[-1]["terminated"]
    assert rows[-1]["reward"] == -2.025
    for row in rows:
        assert set(row) == {"type", "context", "image", "reward", "terminated", "truncated"}
        assert row["image"]["encoding"] == "png;base64"
        image = Image.open(io.BytesIO(base64.b64decode(row["image"]["data"])))
        assert image.size == (1200, 700)
        assert {"world_seed", "_mask", "reward_ledger", "statement_ledger", "parameters",
                "posterior", "positive_process_rewards"}.isdisjoint(row["context"])


def test_training_rewards_require_explicit_cli_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"type":"observe"}\n'))
    assert main(["rollout", "--max-steps", "1", "--reward-mode", "train",
                 "--split", "development", "--curriculum", "foundation", "--seed", "3110026"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1]["reward"] == pytest.approx(-2.025)
    assert rows[-1]["truncated"]
    assert rows[0]["context"]["curriculum"] == "foundation"
    assert "reward_ledger" not in rows[-1]


def test_cli_default_budget_and_curriculum(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["rollout", "--split", "development", "--seed", "3110026"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["context"]["budget"] == 48
    assert row["context"]["curriculum"] == "composed"


@pytest.mark.parametrize("payload", ["not JSON\n", '{"type":"observe","extra":NaN}\n', '{"type":"observe","extra":Infinity}\n'])
def test_rollout_rejects_malformed_protocol(monkeypatch, capsys, payload):
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    assert main(["rollout", "--max-steps", "1", "--split", "development", "--seed", "3110026"]) == 2
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1] == {"type": "protocol_error", "message": "Action must be a JSON object."}


def test_release_and_gym_registration():
    import gymnasium as gym

    import visual_repair_lab

    assert visual_repair_lab.__version__ == "7.0.0"
    assert "VisualRepairReward-v1" not in gym.registry
    assert "VisualCausalRepair-v0" not in gym.registry
    assert "VisualDiscoveryRepair-v0" not in gym.registry
    env = gym.make("VisualTrainingRepair-v0", split="development")
    try:
        observation, _ = env.reset(seed=3110027, options={"world_seed": 3110027})
        context = json.loads(observation["context"])
        assert context["environment"] == "VisualTrainingRepair-v0"
        assert context["budget"] == 48
    finally:
        env.close()


def test_rollout_starter_only():
    with pytest.raises(SystemExit) as error:
        main(["rollout", "--candidate-mode", "audit_mixture"])
    assert error.value.code == 2
