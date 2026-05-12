#!/usr/bin/env python3
"""Record real ALOHA episodes directly to a local LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from constants import DT, START_ARM_POSE, TASK_CONFIGS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from lerobot_v21 import (
    build_features,
    default_dataset_root,
    episode_start_index,
    next_episode_index,
    parquet_path,
    push_dataset_to_hub,
    remove_episode_files,
    stats_for_array,
    update_metadata,
    video_path,
    write_parquet,
)
from real_env import get_action, make_real_env
from robot_utils import get_arm_gripper_positions
from robot_utils import move_arms, move_grippers, torque_off, torque_on


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    """Move all 4 robots to a pose where it is easy to start demonstration."""
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")

    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position", "time")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    move_grippers(
        [master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right],
        [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2,
        move_time=0.5,
    )

    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print("Close the gripper to start")
    close_thresh = -0.78
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT / 10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print("Started!")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    freq_mean = 1 / dt_mean
    print(f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}")
    return freq_mean


def open_video_writers(dataset_root, episode_index, camera_names, images, fps):
    writers = {}
    for camera_name in camera_names:
        image = images[camera_name]
        if image is None:
            raise RuntimeError(f"No image received for camera {camera_name}")
        height, width = image.shape[:2]
        path = video_path(dataset_root, episode_index, camera_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(width), int(height)))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        writers[camera_name] = writer
    return writers


def write_video_frame(writers, camera_names, images):
    for camera_name in camera_names:
        image = images[camera_name]
        writers[camera_name].write(image[:, :, [2, 1, 0]])


def close_video_writers(writers):
    for writer in writers.values():
        writer.release()


def capture_one_episode(
    max_timesteps,
    camera_names,
    dataset_root,
    episode_index,
    task,
    fps,
    robot_type,
    include_velocity,
    include_effort,
):
    print(f"LeRobot dataset root: {dataset_root}")
    print(f"Episode index: {episode_index}")

    base_frame_index = episode_start_index(dataset_root, episode_index)
    removed_paths = remove_episode_files(dataset_root, episode_index)
    if removed_paths:
        print(f"Overwriting existing episode {episode_index}: removed {len(removed_paths)} old files.")

    master_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="master_left",
        init_node=True,
    )
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name=None,
        robot_name="master_right",
        init_node=False,
    )
    env = make_real_env(init_node=False, setup_robots=False)

    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

    ts = env.reset(fake=True)
    rows = []
    qpos_history = []
    action_history = []
    qvel_history = []
    effort_history = []
    actual_dt_history = []
    writers = None
    image_shape = None

    try:
        for frame_index in tqdm(range(max_timesteps)):
            t0 = time.time()
            action = get_action(master_bot_left, master_bot_right)
            t1 = time.time()

            obs = ts.observation
            if writers is None:
                writers = open_video_writers(dataset_root, episode_index, camera_names, obs["images"], fps)
                image_shape = obs["images"][camera_names[0]].shape

            qpos = np.asarray(obs["qpos"], dtype=np.float32)
            row = {
                "observation.state": qpos.tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "timestamp": np.float32(frame_index / fps).item(),
                "frame_index": int(frame_index),
                "episode_index": int(episode_index),
                "index": int(base_frame_index + frame_index),
                "task_index": 0,
            }
            if include_velocity:
                qvel = np.asarray(obs["qvel"], dtype=np.float32)
                row["observation.velocity"] = qvel.tolist()
                qvel_history.append(qvel)
            if include_effort:
                effort = np.asarray(obs["effort"], dtype=np.float32)
                row["observation.effort"] = effort.tolist()
                effort_history.append(effort)

            write_video_frame(writers, camera_names, obs["images"])
            rows.append(row)
            qpos_history.append(qpos)
            action_history.append(np.asarray(action, dtype=np.float32))

            ts = env.step(action)
            t2 = time.time()
            actual_dt_history.append([t0, t1, t2])
    finally:
        if writers is not None:
            close_video_writers(writers)

    torque_on(master_bot_left)
    torque_on(master_bot_right)
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    print_dt_diagnosis(actual_dt_history)

    if not rows:
        raise RuntimeError("No frames were recorded.")

    features = build_features(
        camera_names,
        image_shape,
        len(rows[0]["observation.state"]),
        len(rows[0]["action"]),
        fps,
        include_velocity,
        include_effort,
    )

    write_parquet(parquet_path(dataset_root, episode_index), rows)

    episode_stats = {
        "observation.state": stats_for_array(np.stack(qpos_history)),
        "action": stats_for_array(np.stack(action_history)),
    }
    if include_velocity:
        episode_stats["observation.velocity"] = stats_for_array(np.stack(qvel_history))
    if include_effort:
        episode_stats["observation.effort"] = stats_for_array(np.stack(effort_history))

    update_metadata(
        dataset_root,
        episode_index,
        len(rows),
        task,
        fps,
        robot_type,
        features,
        episode_stats,
    )
    print(f"Saved LeRobot v2.1 episode {episode_index} with {len(rows)} frames.")
    return True


def resolve_dataset_root(args, task_config):
    if args.output_root:
        return Path(args.output_root).expanduser()
    if args.repo_id:
        return default_dataset_root(args.repo_id)
    return Path(task_config["dataset_dir"]).expanduser() / "lerobot"


def main(args):
    task_config = TASK_CONFIGS[args.task_name]
    dataset_root = resolve_dataset_root(args, task_config)
    max_timesteps = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    episode_index = args.episode_idx if args.episode_idx is not None else next_episode_index(dataset_root)
    task = args.task if args.task else args.task_name

    if args.push_to_hub and not args.repo_id:
        raise SystemExit("--push-to-hub requires --repo-id.")

    capture_one_episode(
        max_timesteps=max_timesteps,
        camera_names=camera_names,
        dataset_root=dataset_root,
        episode_index=episode_index,
        task=task,
        fps=args.fps,
        robot_type=args.robot_type,
        include_velocity=args.include_velocity,
        include_effort=args.include_effort,
    )

    if args.push_to_hub:
        print(f"Pushing LeRobot dataset to Hugging Face Hub: {args.repo_id}")
        push_dataset_to_hub(dataset_root, args.repo_id, private=args.private)
        print(f"Pushed dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", action="store", type=str, help="Task name from constants.py.", required=True)
    parser.add_argument("--episode_idx", action="store", type=int, help="LeRobot episode index.", default=None)
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id, e.g. <hf_user>/<dataset_name>. Defaults local root to HF_LEROBOT_HOME/<repo-id>, HF_HOME/lerobot/<repo-id>, or ~/.cache/huggingface/lerobot/<repo-id>.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override local LeRobot dataset root. If omitted with --repo-id, uses the LeRobot cache path.",
    )
    parser.add_argument("--task", type=str, default=None, help="Natural-language task stored in LeRobot metadata.")
    parser.add_argument("--fps", type=int, default=50, help="Dataset frame rate.")
    parser.add_argument("--robot-type", type=str, default="aloha", help="Robot type metadata.")
    parser.add_argument("--include-velocity", action="store_true", help="Store observation.velocity.")
    parser.add_argument("--include-effort", action="store_true", help="Store observation.effort.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the local LeRobot dataset folder after recording.")
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face dataset repo as private when pushing.")
    main(parser.parse_args())
