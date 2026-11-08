"""Active visual repair with terminal-only semantic grading.

Private goals affect reference pixels and terminal results only. Public app inputs,
execution/schema feedback, interaction transitions and preterminal rewards do not
consume them. The evaluator process must remain outside the policy workspace.
"""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import string

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from .runtime import run_program
from .tasks import (
    HEIGHT,
    WIDTH,
    evaluate_program,
    interact_state,
    make_goal,
    make_state,
    render_current,
    render_reference,
    starter_code,
    task_catalog,
    validate_result,
    evidence_view,
)

OPERATIONS = ("observe", "crop", "interact", "edit", "run", "submit")


class VisualDiscoveryEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self, family: str = "canvas_layout", max_steps: int = 48, render_mode: str | None = "rgb_array"
    ):
        super().__init__()
        self.catalog = {task["family"]: task for task in task_catalog()}
        if family not in self.catalog:
            raise ValueError("Unknown visual active family")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 200:
            raise ValueError("Episode budget is out of range")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Unsupported render mode")
        self.family, self.max_steps, self.render_mode = family, max_steps, render_mode
        self.action_space = spaces.Text(max_length=20000, min_length=1, charset=string.printable)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8),
                "context": spaces.Text(max_length=100000, min_length=1, charset=string.printable),
            }
        )
        self._ready = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        family = options.get("family", self.family)
        if family not in self.catalog:
            raise ValueError("Unknown visual active family")
        self.family = family
        self.episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31))
        goal_seed = options["goal_seed"] if "goal_seed" in options else secrets.randbits(128)
        self._goal = make_goal(family, goal_seed)
        self.initial_state = make_state(family, self.episode_seed)
        self.state = copy.deepcopy(self.initial_state)
        self.source = self.active_source = starter_code(family)
        self.source_revision = self.active_revision = self.preview_revision = 0
        self.steps = 0
        self.total_reward = self.reward = 0.0
        self.terminated = self.truncated = False
        self.history, self.trajectory = [], []
        self.evidence_views = []
        self.final_result, self.crop_box = None, None
        self._execute()
        self.message = (
            "Choose your own investigation. Repair the candidate to match the approved application."
        )
        self._refresh_images()
        self._ready = True
        observation = self._observation()
        self.trajectory.append(
            {
                "type": "reset",
                "observation": copy.deepcopy(observation),
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
            }
        )
        return observation, self._info()

    def _context(self) -> dict:
        task = self.catalog[self.family]
        return {
            "family": self.family,
            "title": task["title"],
            "description": task["description"],
            "prompt": task["prompt"],
            "interactions": copy.deepcopy(task["interactions"]),
            "source": self.source,
            "source_revision": self.source_revision,
            "active_revision": self.active_revision,
            "preview_revision": self.preview_revision,
            "scene": copy.deepcopy(self.state),
            "reference_view": evidence_view(self.family, self.state),
            "evidence_views": list(self.evidence_views),
            "message": self.message,
            "steps": self.steps,
            "budget": self.max_steps,
            "remaining": max(0, self.max_steps - self.steps),
            "public_result": self.public_result,
            "final_result": self.final_result,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "operations": list(OPERATIONS),
            "crop_box": self.crop_box,
            "program_sha256": self._source_hash(),
        }

    def _source_hash(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8", errors="replace")).hexdigest()

    def _observation(self) -> dict:
        return {"image": self._image.copy(), "context": json.dumps(self._context(), ensure_ascii=True)}

    def _info(self) -> dict:
        return {
            "tool_steps": self.steps,
            "reward": self.reward,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def _execute(self) -> None:
        runtime = run_program(self.active_source, self.state)
        valid = runtime["ok"] and validate_result(self.family, self.state, runtime["result"])
        self.output = runtime["result"] if valid else None
        self.public_result = {
            "executed": bool(runtime["ok"]),
            "schema_valid": bool(valid),
            "error": runtime["error"]
            if not runtime["ok"]
            else ("" if valid else "Output does not match the documented JSON schema."),
        }
        self.preview_revision = self.active_revision

    def _refresh_images(self) -> None:
        self._reference_image = render_reference(self.family, self.state, self._goal)
        self._current_image = render_current(self.family, self.state, self.output)
        self._image = np.concatenate((self._reference_image, self._current_image), axis=1)

    @property
    def reference_image(self) -> np.ndarray:
        return self._reference_image.copy()

    @property
    def current_image(self) -> np.ndarray:
        return self._current_image.copy()

    def _run(self) -> None:
        self.active_source, self.active_revision = self.source, self.source_revision
        self._execute()
        self.crop_box = None
        self._refresh_images()
        self.message = (
            "Program executed with schema-valid output. Inspect the current view."
            if self.public_result["schema_valid"]
            else self.public_result["error"]
        )

    def _interact(self, event: dict) -> None:
        if not isinstance(event, dict):
            raise ValueError("Interaction needs an event object")
        kind = event.get("kind")
        if kind == "reset_scene":
            self.state = copy.deepcopy(self.initial_state)
        else:
            self.state = interact_state(self.family, self.state, event)
        self._execute()
        self.crop_box = None
        self._refresh_images()
        view = evidence_view(self.family, self.state)
        if view not in self.evidence_views:
            self.evidence_views.append(view)
        self.message = (
            "Public interaction applied to the active revision."
            if self.public_result["schema_valid"]
            else self.public_result["error"]
        )

    def _submit(self) -> None:
        grade = evaluate_program(self.family, self.source, self.episode_seed, self._goal)
        self.final_result = {
            **grade,
            "program_sha256": self._source_hash(),
            "submitted_revision": self.source_revision,
            "preview_was_current": self.preview_revision == self.source_revision,
        }
        self.terminated = True
        self.reward += 10.0 if grade["success"] else -2.0
        self.message = "Protected evaluation complete. Episode ended."

    def step(self, action):
        if not self._ready or self.terminated or self.truncated:
            raise gym.error.ResetNeeded("Reset before taking another action")
        self.steps += 1
        self.reward = -0.02
        operation, recorded = "invalid", action
        try:
            if isinstance(action, str):
                if len(action) > 20000:
                    raise ValueError("Action too large")
                action = json.loads(action)
            if not isinstance(action, dict):
                raise ValueError("Action must be an object")
            recorded = copy.deepcopy(action)
            operation = action.get("type", action.get("tool", "invalid"))
            if operation == "observe":
                self.crop_box = None
                self._refresh_images()
                self.message = "Full reference and current application view."
            elif operation == "crop":
                box = action.get("box", [action.get(k) for k in ("x", "y", "width", "height")])
                if (
                    not isinstance(box, list)
                    or len(box) != 4
                    or any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                        for v in box
                    )
                ):
                    raise ValueError("Invalid crop")
                x, y, w, h = (int(v) for v in box)
                if x < 0 or y < 0 or w < 8 or h < 8 or x + w > WIDTH or y + h > HEIGHT:
                    raise ValueError("Crop outside image")
                full = np.concatenate((self._reference_image, self._current_image), axis=1)
                self._image = np.asarray(
                    Image.fromarray(full).crop((x, y, x + w, y + h)).resize((WIDTH, HEIGHT))
                )
                self.crop_box = [x, y, w, h]
                self.message = "Enlarged existing pixels. Observe restores the full view."
            elif operation == "edit":
                source = action.get("source")
                if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > 16000:
                    raise ValueError("Invalid source")
                if source != self.source:
                    self.source, self.source_revision = source, self.source_revision + 1
                self.message = "Source saved. Run activates this revision."
            elif operation == "run":
                self._run()
            elif operation == "interact":
                self._interact(action.get("event"))
            elif operation == "submit":
                self._submit()
            else:
                raise ValueError("Unknown action")
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            self.message = "Invalid action. Check the documented tool arguments."
            self.reward -= 0.08
        if self.steps >= self.max_steps and not self.terminated:
            self.truncated = True
            self.message = "Tool budget exhausted without submission."
        self.total_reward += self.reward
        self.history.append(
            {"step": self.steps, "type": operation, "message": self.message, "reward": self.reward}
        )
        observation = self._observation()
        self.trajectory.append(
            {
                "action": recorded,
                "observation": copy.deepcopy(observation),
                "reward": self.reward,
                "terminated": self.terminated,
                "truncated": self.truncated,
            }
        )
        return observation, self.reward, self.terminated, self.truncated, self._info()

    def render(self):
        if not self._ready:
            raise gym.error.ResetNeeded("Reset before rendering")
        return self._image.copy()
