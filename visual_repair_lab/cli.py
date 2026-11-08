"""Offline installation checks and an observation-only JSONL agent interface."""

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from . import __version__, make_env

FAMILIES = ("drag_hysteresis", "linked_brush", "contact_dynamics")


def _metadata():
    dependencies = {}
    for name in ("numpy", "gymnasium", "Pillow"):
        dependencies[name] = importlib.metadata.version(name)
    modules = {}
    for name in (
        "visual_repair_lab", "vision_sort.visual_training.env",
        "vision_sort.visual_training.tasks",
        "vision_sort.visual_causal.tasks", "vision_sort.visual_training.runtime",
    ):
        spec = importlib.util.find_spec(name)
        modules[name] = str(Path(spec.origin).resolve()) if spec and spec.origin else None
    import vision_sort
    runtime_files = list(Path(vision_sort.__file__).parent.rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    hashes = {str(p.relative_to(p.parent.parent if p.parent.name=='visual_repair_lab' else Path(vision_sort.__file__).parent.parent)):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files}
    return {
        "source_sha256": hashes,
        "release": __version__,
        "environment": "VisualTrainingRepair-v0",
        "default_action_budget": 48,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependencies,
        "module_paths": modules,
    }


def _validate_observation(env, observation):
    assert env.observation_space.contains(observation), "Observation violates its declared space"
    assert observation["image"].shape == (700, 1200, 3)
    assert observation["image"].dtype == np.uint8
    context = json.loads(observation["context"])
    assert context["environment"] == "VisualTrainingRepair-v0"
    private_fields = {
        "world_seed", "_world_seed", "mask", "_mask", "reward_ledger", "statement_ledger",
        "remaining_worlds", "information_bits", "training_pool", "oracle_source",
        "posterior", "parameters", "positive_process_rewards", "progress_cases",
    }
    assert private_fields.isdisjoint(context), "Private evaluator fields entered the public context"


def installation_smoke():
    """Run trusted offline self-tests, never a model policy or training episode.

    Gold source is used only inside this evaluator-side test. The JSONL rollout
    interface below never imports or supplies gold source or private seeds.
    """
    from vision_sort.visual_training.runtime import run_program
    from vision_sort.visual_training.tasks import oracle_source

    started = time.perf_counter()
    receipt = {
        "kind": "offline_installation_self_test",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "runtime": _metadata(),
        "checks": [],
        "notes": [
            "Trusted evaluator smoke checks; no model calls or policy training.",
            "Gold repair fixtures and private development seeds are never agent observations.",
            "The bounded interpreter is not a secure OS sandbox.",
            "Pixel identity across operating systems is not guaranteed because fonts vary.",
        ],
    }

    def check(name, function):
        check_started = time.perf_counter()
        try:
            details = function()
            row = {"name": name, "status": "passed", "details": details}
        except Exception as error:  # noqa: BLE001 - report each self-test failure without a raw traceback.
            row = {"name": name, "status": "failed", "error_type": type(error).__name__}
        row["wall_seconds"] = round(time.perf_counter() - check_started, 4)
        receipt["checks"].append(row)

    def repair(family, index):
        env = make_env(family=family, reward_mode="evaluation", split="development",candidate_mode="starter")
        try:
            observation, _ = env.reset(seed=3110020 + index, options={"world_seed": 3110020 + index})
            _validate_observation(env, observation)
            observation, reward, terminated, truncated, _ = env.step(json.dumps({"type": "observe"}))
            assert reward == -.025 and not terminated and not truncated
            _validate_observation(env, observation)
            source = oracle_source(family, env._mask, env.split)
            observation, _, _, _, _ = env.step({"type": "edit", "source": source})
            assert json.loads(observation["context"])["active_revision"] == 0
            observation, _, _, _, _ = env.step({"type": "run"})
            assert json.loads(observation["context"])["public_result"]["schema_valid"]
            observation, reward, terminated, truncated, _ = env.step({"type": "submit"})
            _validate_observation(env, observation)
            final = json.loads(observation["context"])["final_result"]
            assert terminated and not truncated and reward == 9.975
            assert final["success"] and final["passed"] == final["total"]
            try:
                env.step({"type": "observe"})
            except gym.error.ResetNeeded:
                pass
            else:
                raise AssertionError("Completed environment allowed another step")
            return {"passed": final["passed"], "total": final["total"], "actions": 4}
        finally:
            env.close()

    def deadline(family, mode):
        env = make_env(family=family, max_steps=1, reward_mode=mode, split="development")
        try:
            env.reset(seed=3110024, options={"world_seed": 3110024})
            observation, reward, terminated, truncated, _ = env.step({"type": "observe"})
            _validate_observation(env, observation)
            expected_reward = -2.025
            assert not terminated and truncated and math.isclose(reward, expected_reward)
            return {"terminated": terminated, "truncated": truncated, "reward": reward}
        finally:
            env.close()

    def starter(family):
        env = make_env(family=family, split="development")
        try:
            observation, _ = env.reset(seed=3110025, options={"world_seed": 3110025})
            _validate_observation(env, observation)
            context = json.loads(observation["context"])
            assert context["public_result"]["schema_valid"]
            assert env.max_steps == 48
            return {"candidate_mode": "starter", "action_budget": 48, "curriculum": "composed"}
        finally:
            env.close()

    def foundation(family):
        env = make_env(family=family, split="development", curriculum="foundation")
        try:
            observation, _ = env.reset(seed=3110028, options={"world_seed": 3110028})
            _validate_observation(env, observation)
            context = json.loads(observation["context"])
            assert context["curriculum"] == "foundation"
            assert "older behaviors in the starter are correct" in context["prompt"]
            assert env._mask & 63 == 0
            return {"curriculum": "foundation", "older_factors_fixed": True}
        finally:
            env.close()

    for index, family in enumerate(FAMILIES):
        check(f"{family}: edit/run/submit", lambda f=family, i=index: repair(f, i))
        for mode in ("train", "evaluation"):
            check(f"{family}: {mode} deadline", lambda f=family, m=mode: deadline(f, m))
        check(f"{family}: starter and budget", lambda f=family: starter(f))
        check(f"{family}: foundation curriculum", lambda f=family: foundation(f))

    def valid_program():
        result = run_program("def solve(s):\n    b = {'x': 2, 'y': 7}\n    b['x'], b['y'] = b['y'], b['x']\n    return b\n", {})
        assert result["ok"] and result["result"] == {"x": 7, "y": 2}
        return {"executed": True}

    def invalid_program():
        result = run_program("import os\ndef solve(state):\n    return {}\n", {})
        assert not result["ok"] and result["error"]
        return {"rejected": True}

    def looping_program():
        result = run_program("def solve(state):\n    while True:\n        pass\n", {})
        assert not result["ok"] and result["error_code"] == "candidate_instruction_limit"
        return {"bounded": True, "instruction_events": 250000}

    check("bounded runtime: valid program", valid_program)
    check("bounded runtime: unsupported program", invalid_program)
    check("bounded runtime: loop timeout", looping_program)
    receipt["status"] = "passed" if all(row["status"] == "passed" for row in receipt["checks"]) else "failed"
    receipt["wall_seconds"] = round(time.perf_counter() - started, 4)
    return receipt


def _write_receipt(path, receipt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _public_frame(observation, *, reward=None, terminated=False, truncated=False, observation_mode="pixels"):
    stream = io.BytesIO()
    Image.fromarray(observation["image"]).save(stream, format="PNG")
    frame = {
        "type": "observation",
        "context": json.loads(observation["context"]),
        "image": {"encoding": "png;base64", "data": base64.b64encode(stream.getvalue()).decode("ascii")},
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }
    frame['context']['observation_channel'] = observation_mode
    if observation_mode == 'ocr':
        from .observations import pixel_text
        frame['pixel_derived_text'] = pixel_text(observation['image'])
        frame['image'] = None
        frame['context']['image_access'] = False
    elif observation_mode == 'none':
        frame['image'] = None
    return frame


def _emit(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def _record_private(args, env, observation, action=None):
    if args.audit_dir is None:
        return
    root = args.audit_dir
    if env.steps == 0:
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        _write_receipt(root/'private-reset.json', dict(world_seed=env._world_seed,
            scene_seed=env.episode_seed,family=env.family,split=env.split,
            curriculum=env.curriculum,candidate_mode=env.candidate_mode,
            implementation=env.implementation,appearance=env.appearance,
            reward_mode=env.mode,reward_profile=env.profile,max_steps=env.max_steps,
            observation_mode=args.observation_mode,runtime=_metadata()))
    Image.fromarray(observation['image']).save(root/f'{env.steps:04d}.png')
    _write_receipt(root/f'{env.steps:04d}.json',dict(action=action,
        context=json.loads(observation['context']),report=env.research_report()))


def _deliver(args, env, observation, **kwargs):
    frame = _public_frame(observation,observation_mode=args.observation_mode,**kwargs)
    if args.audit_dir is not None:
        _write_receipt(args.audit_dir/f'{env.steps:04d}-public.json',frame)
    _emit(frame)


def _reject_nonfinite(value):
    raise ValueError("Non-finite JSON is not permitted")


def _rollout(args):
    env = make_env(family=args.family, max_steps=args.max_steps,
                   reward_mode=args.reward_mode, split=args.split, curriculum=args.curriculum,
                   reward_profile=args.reward_profile,candidate_mode=args.candidate_mode,
                   appearance=args.appearance,image_access=args.observation_mode != "none")
    try:
        observation, _ = env.reset(seed=args.seed)
        _record_private(args,env,observation)
        _deliver(args,env,observation)
        while True:
            line = sys.stdin.readline(20002)
            if not line:
                return 0
            if len(line.rstrip("\r\n")) > 20000 or not line.endswith("\n"):
                _emit({"type": "protocol_error", "message": "Send one JSON action per line, at most 20000 characters."})
                return 2
            try:
                action = json.loads(line,parse_constant=_reject_nonfinite)
                if not isinstance(action, dict):
                    raise TypeError
            except (ValueError, TypeError, RecursionError):
                _emit({"type": "protocol_error", "message": "Action must be a JSON object."})
                return 2
            from vision_sort.visual_training.runtime import EvaluationInfrastructureError
            try:
                observation, reward, terminated, truncated, _ = env.step(action)
            except EvaluationInfrastructureError as error:
                if args.audit_dir is not None:
                    _write_receipt(args.audit_dir/'abort.json',dict(code=error.code,
                        action=action,episode_scored=False,report=env.research_report()))
                _emit({"type":"infrastructure_error","code":error.code,"episode_scored":False})
                return 75
            _record_private(args,env,observation,action)
            _deliver(args,env,observation,reward=reward,terminated=terminated,truncated=truncated)
            if terminated or truncated:
                return 0
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visual Repair Lab: offline self-test and public JSONL interaction.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Run offline installation checks; no APIs, GPUs, or model weights.")
    smoke.add_argument("--output", type=Path, help="Save the complete JSON self-test receipt.")
    rollout = commands.add_parser("rollout", help="Read JSON action lines; emit public context and base64 PNG observations.")
    rollout.add_argument("--family", choices=FAMILIES, default=FAMILIES[0])
    rollout.add_argument("--seed", type=int, default=7, help="Public scene seed; the private target is sampled independently.")
    rollout.add_argument("--max-steps", type=int, choices=range(1, 201), metavar="1..200", default=48)
    rollout.add_argument("--candidate-mode", choices=("starter","mixed","audit_correct","audit_incomplete"), default="mixed")
    rollout.add_argument("--split", choices=("train", "development", "validation", "test"), default="train",
                         help="Parameter distribution; use development for local checks and reserve evaluation splits.")
    rollout.add_argument("--curriculum", choices=("foundation", "composed"), default="composed")
    rollout.add_argument("--reward-mode", choices=("evaluation", "train"), default="evaluation",
                         help="Evaluation emits only task rewards; train explicitly enables bounded process rewards.")
    rollout.add_argument("--reward-profile", choices=("terminal","progress","tests","claims","full"),default="progress")
    rollout.add_argument('--appearance',choices=('standard','alternate'),default='standard')
    rollout.add_argument('--observation-mode',choices=('pixels','ocr','none'),default='pixels',
                         help='OCR is optional, pixel-derived and evaluation-only; no truth-to-text shortcut.')
    rollout.add_argument('--audit-dir',type=Path,help='Trusted evaluator-only NEW directory; never expose it to the actor.')
    args = parser.parse_args(argv)
    if args.command=='rollout' and args.observation_mode=='ocr' and args.reward_mode!='evaluation':
        parser.error('OCR baseline is evaluation-only; use pixels for training rewards.')
    if args.command == "rollout":
        from vision_sort.visual_training.runtime import EvaluationInfrastructureError
        from .observations import ObservationAdapterError
        try:
            return _rollout(args)
        except EvaluationInfrastructureError as error:
            _emit({'type':'infrastructure_error','code':error.code,'episode_scored':False})
            return 75
        except OSError:
            _emit({'type':'infrastructure_error','code':'audit_storage_failure','episode_scored':False})
            return 75
        except ObservationAdapterError as error:
            _emit({'type':'infrastructure_error','code':str(error),'episode_scored':False})
            return 75
    receipt = installation_smoke()
    if args.output:
        _write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0 if receipt["status"] == "passed" else 1
