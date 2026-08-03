"""Evaluate a served Stage 3 LARA/GR00T N1.5 policy on LIBERO."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lara_full.eval.service import ExternalRobotInferenceClient  # noqa: E402

MAX_STEPS_BY_SUITE = {
    "libero_spatial": 720,
    "libero_object": 720,
    "libero_goal": 720,
    "libero_10": 1000,
    "libero_90": 400,
}
ACTION_KEYS = ("position", "orientation", "gripper")


@dataclass
class EvalConfig:
    host: str = "127.0.0.1"
    port: int = 5555
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 1
    task_id: int | None = None
    num_steps_wait: int = 10
    max_steps: int | None = None
    seed: int = 0
    timeout_ms: int = 120_000
    save_video: bool = False
    output_dir: str = "./eval_outputs/lara_full_libero10"


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    scalar = float(np.clip(quaternion[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - scalar * scalar))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    axis_angle = quaternion[:3] * (2.0 * math.acos(scalar) / denominator)
    return axis_angle.astype(np.float32)


def extract_images(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    agent_view = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist_view = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    return agent_view, wrist_view


def to_libero_action(
    action_chunk: dict[str, np.ndarray],
    action_index: int = 1,
) -> np.ndarray:
    components = [np.atleast_1d(action_chunk[f"action.{key}"][action_index]) for key in ACTION_KEYS]
    action = np.concatenate(components).astype(np.float32, copy=False)
    if action.shape != (7,):
        raise ValueError(f"Expected a 7-D LIBERO action, got shape {action.shape}")
    action[-1] = np.sign(1.0 - 2.0 * action[-1])
    return action


def prepare_observation(
    observation: dict[str, Any],
    instruction: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    agent_view, wrist_view = extract_images(observation)
    policy_observation = {
        "video.image": agent_view[None],
        "video.wrist_image": wrist_view[None],
        "state.position": np.asarray(observation["robot0_eef_pos"], dtype=np.float32)[None],
        "state.orientation": quaternion_to_axis_angle(observation["robot0_eef_quat"])[None],
        "state.gripper": np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)[None],
        "annotation.human.action.task_description": [instruction],
    }
    return policy_observation, agent_view, wrist_view


def create_environment(task: Any, resolution: int, seed: int) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    environment = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    environment.seed(seed)
    return environment


def save_episode_video(
    frames: list[np.ndarray],
    output_dir: Path,
    task_id: int,
    episode_id: int,
    success: bool,
) -> Path:
    video_path = output_dir / (
        f"task_{task_id:02d}_episode_{episode_id:02d}_success_{int(success)}.mp4"
    )
    with imageio.get_writer(video_path, codec="libx264", fps=30) as writer:
        for frame in frames:
            writer.append_data(frame)
    return video_path


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    if config.task_suite_name not in MAX_STEPS_BY_SUITE:
        choices = ", ".join(MAX_STEPS_BY_SUITE)
        raise ValueError(f"Unknown suite {config.task_suite_name!r}; choose one of: {choices}")
    if config.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")

    from libero.libero import benchmark

    benchmark_classes = benchmark.get_benchmark_dict()
    task_suite = benchmark_classes[config.task_suite_name]()
    task_ids = [config.task_id] if config.task_id is not None else list(range(task_suite.n_tasks))
    if any(task_id < 0 or task_id >= task_suite.n_tasks for task_id in task_ids):
        raise ValueError(f"task_id must be in [0, {task_suite.n_tasks - 1}]")

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_limit = config.max_steps or MAX_STEPS_BY_SUITE[config.task_suite_name]
    results: list[dict[str, Any]] = []

    with ExternalRobotInferenceClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
    ) as client:
        if not client.ping():
            raise ConnectionError(f"No LARA inference server at {config.host}:{config.port}")

        for task_id in tqdm.tqdm(task_ids, desc="LIBERO tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if config.num_trials_per_task > len(initial_states):
                raise ValueError(f"Task {task_id} only has {len(initial_states)} initial states")
            environment = create_environment(task, resolution=256, seed=config.seed)
            try:
                for episode_id in tqdm.trange(
                    config.num_trials_per_task,
                    desc=f"task {task_id}",
                    leave=False,
                ):
                    environment.reset()
                    observation = environment.set_init_state(initial_states[episode_id])
                    for _ in range(config.num_steps_wait):
                        observation, _, _, _ = environment.step([0, 0, 0, 0, 0, 0, -1])

                    success = False
                    video_frames: list[np.ndarray] = []
                    for policy_step in range(episode_limit):
                        policy_observation, agent_view, wrist_view = prepare_observation(
                            observation,
                            task.language,
                        )
                        if config.save_video:
                            video_frames.append(np.hstack((agent_view, wrist_view)))
                        action_chunk = client.get_action(policy_observation)
                        action = to_libero_action(action_chunk, action_index=1)
                        observation, _, done, _ = environment.step(action.tolist())
                        if done:
                            success = True
                            break

                    episode_result = {
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "instruction": task.language,
                        "success": success,
                        "policy_steps": policy_step + 1,
                    }
                    results.append(episode_result)
                    if config.save_video:
                        episode_result["video"] = str(
                            save_episode_video(
                                video_frames,
                                output_dir,
                                task_id,
                                episode_id,
                                success,
                            )
                        )
                    completed = len(results)
                    successes = sum(result["success"] for result in results)
                    print(
                        f"task={task_id} episode={episode_id} success={success} "
                        f"total={successes}/{completed}",
                        flush=True,
                    )
            finally:
                environment.close()

    successes = sum(result["success"] for result in results)
    summary = {
        "config": asdict(config),
        "successes": successes,
        "episodes": len(results),
        "success_rate": successes / len(results),
        "results": results,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"LIBERO result: {successes}/{len(results)} "
        f"({100.0 * summary['success_rate']:.1f}%)\nSaved to {result_path}",
        flush=True,
    )
    return summary


def main() -> None:
    import tyro

    run_evaluation(tyro.cli(EvalConfig))


if __name__ == "__main__":
    main()
