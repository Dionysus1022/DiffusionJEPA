from __future__ import annotations

from types import MethodType
from typing import Any

import gymnasium as gym
import numpy as np
from ogbench.locomaze.maze import make_maze_env
from ogbench.manipspace.envs.puzzle_env import PuzzleEnv


VISUAL_PUZZLE_3X3_ENV_ID = "lewm/VisualPuzzle3x3-v0"
VISUAL_ANTMAZE_LARGE_ENV_ID = "lewm/VisualAntMazeLarge-v0"


def _button_state_vector(value: Any, *, num_buttons: int, name: str) -> np.ndarray:
    states = np.asarray(value, dtype=np.int64).reshape(-1)
    if states.shape != (int(num_buttons),):
        raise ValueError(
            f"{name} must contain {num_buttons} button states, got shape {states.shape}."
        )
    if not np.isin(states, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary button states.")
    return states.copy()


class DatasetVisualPuzzle3x3Env(PuzzleEnv):
    """OGBench visual Puzzle 3x3 with dataset-conditioned reset helpers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("ob_type", "pixels")
        kwargs.setdefault("width", 64)
        kwargs.setdefault("height", 64)
        super().__init__(env_type="3x3", *args, **kwargs)

    def set_dataset_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        button_states: np.ndarray,
    ) -> None:
        states = _button_state_vector(
            button_states,
            num_buttons=self._num_buttons,
            name="button_states",
        )
        self.set_state(np.asarray(qpos), np.asarray(qvel), states)

    def set_dataset_goal(
        self,
        button_states: np.ndarray,
        pixels: np.ndarray,
    ) -> None:
        self._target_button_states = _button_state_vector(
            button_states,
            num_buttons=self._num_buttons,
            name="goal button_states",
        )
        goal_pixels = np.asarray(pixels)
        if goal_pixels.ndim != 3 or goal_pixels.shape[-1] != 3:
            raise ValueError(
                "goal pixels must be an HWC RGB image, "
                f"got shape {goal_pixels.shape}."
            )
        self._cur_goal_ob = goal_pixels.copy()
        self._success = bool(
            np.array_equal(self._cur_button_states, self._target_button_states)
        )

    def get_step_info(self) -> dict[str, Any]:
        info = super().get_step_info()
        if self._cur_goal_ob is not None:
            info["goal"] = np.asarray(self._cur_goal_ob).copy()
        return info

    def get_reset_info(self) -> dict[str, Any]:
        info = super().get_reset_info()
        if info.get("goal_rendered") is None:
            info.pop("goal_rendered", None)
        return info


def _set_antmaze_dataset_state(
    env: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> None:
    qpos_array = np.asarray(qpos)
    qvel_array = np.asarray(qvel)
    if qpos_array.shape != (int(env.model.nq),):
        raise ValueError(
            f"qpos must have shape {(int(env.model.nq),)}, got {qpos_array.shape}."
        )
    if qvel_array.shape != (int(env.model.nv),):
        raise ValueError(
            f"qvel must have shape {(int(env.model.nv),)}, got {qvel_array.shape}."
        )
    env.set_state(qpos_array.copy(), qvel_array.copy())


def _set_antmaze_dataset_goal(
    env: Any,
    qpos: np.ndarray,
    pixels: np.ndarray,
) -> None:
    qpos_array = np.asarray(qpos).reshape(-1)
    if qpos_array.size < 2 or not np.isfinite(qpos_array[:2]).all():
        raise ValueError(
            "goal qpos must contain at least two finite coordinates for AntMaze XY."
        )
    goal_pixels = np.asarray(pixels)
    if goal_pixels.ndim != 3 or goal_pixels.shape[-1] != 3:
        raise ValueError(
            "goal pixels must be an HWC RGB image, "
            f"got shape {goal_pixels.shape}."
        )
    env.set_goal(goal_xy=qpos_array[:2].copy())
    env._dataset_goal_pixels = goal_pixels.copy()


def _step_antmaze_with_dataset_goal(
    env: Any,
    action: np.ndarray,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    observation, reward, terminated, truncated, raw_info = env._dataset_original_step(action)
    info = dict(raw_info)
    goal_pixels = getattr(env, "_dataset_goal_pixels", None)
    if goal_pixels is not None:
        info["goal"] = np.asarray(goal_pixels).copy()
    return observation, reward, terminated, truncated, info


def make_dataset_visual_antmaze_large_env(*args: Any, **kwargs: Any) -> Any:
    """Create visual AntMaze Large with dataset-conditioned reset helpers."""
    kwargs.setdefault("maze_type", "large")
    kwargs.setdefault("ob_type", "pixels")
    kwargs.setdefault("render_mode", "rgb_array")
    kwargs.setdefault("width", 64)
    kwargs.setdefault("height", 64)
    kwargs.setdefault("camera_name", "back")
    env = make_maze_env("ant", "maze", *args, **kwargs)
    env._dataset_original_step = env.step
    env.set_dataset_state = MethodType(_set_antmaze_dataset_state, env)
    env.set_dataset_goal = MethodType(_set_antmaze_dataset_goal, env)
    env.step = MethodType(_step_antmaze_with_dataset_goal, env)
    return env


def register_visual_ogbench_envs() -> None:
    if VISUAL_PUZZLE_3X3_ENV_ID not in gym.registry:
        gym.register(
            id=VISUAL_PUZZLE_3X3_ENV_ID,
            entry_point="evaluation.visual_ogbench_envs:DatasetVisualPuzzle3x3Env",
        )
    if VISUAL_ANTMAZE_LARGE_ENV_ID not in gym.registry:
        gym.register(
            id=VISUAL_ANTMAZE_LARGE_ENV_ID,
            entry_point=(
                "evaluation.visual_ogbench_envs:"
                "make_dataset_visual_antmaze_large_env"
            ),
        )


register_visual_ogbench_envs()
