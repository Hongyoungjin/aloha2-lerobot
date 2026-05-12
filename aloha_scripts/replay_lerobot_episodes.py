#!/usr/bin/env python3
"""Replay a LeRobot v2.1 ALOHA episode on the real robot."""

from __future__ import annotations

import argparse
from pathlib import Path

from constants import PUPPET_GRIPPER_JOINT_OPEN
from lerobot_v21 import default_dataset_root, read_episode_actions
from real_env import make_real_env
from robot_utils import move_grippers


def resolve_dataset_root(args):
    if args.dataset_root:
        return Path(args.dataset_root).expanduser()
    if args.repo_id:
        return default_dataset_root(args.repo_id)
    raise SystemExit("Provide either --dataset-root or --repo-id.")


def select_actions(actions, start_frame, num_frames):
    if start_frame < 0:
        raise SystemExit("--start-frame must be >= 0.")
    if start_frame >= len(actions):
        raise SystemExit(f"--start-frame {start_frame} is outside episode length {len(actions)}.")

    end_frame = len(actions) if num_frames is None else start_frame + num_frames
    return actions[start_frame:end_frame]


def main(args):
    dataset_root = resolve_dataset_root(args)
    actions = read_episode_actions(dataset_root, args.episode_idx)
    actions = select_actions(actions, args.start_frame, args.num_frames)

    print(f"Replaying {len(actions)} frames from {dataset_root}, episode {args.episode_idx}.")
    print("Make sure both one_side_teleop.py processes are stopped before continuing.")
    if not args.yes:
        response = input("Type 'yes' to start replay: ")
        if response.strip().lower() != "yes":
            print("Replay cancelled.")
            return

    env = make_real_env(init_node=True)
    if not args.skip_reset:
        env.reset()

    for action in actions:
        env.step(action)

    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=None, help="Local LeRobot v2.1 dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id. Uses HF_LEROBOT_HOME/<repo-id>, HF_HOME/lerobot/<repo-id>, or ~/.cache/huggingface/lerobot/<repo-id>.",
    )
    parser.add_argument("--episode_idx", type=int, required=True, help="LeRobot episode index to replay.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to replay.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to replay.")
    parser.add_argument("--skip-reset", action="store_true", help="Do not run env.reset() before replay.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive safety confirmation.")
    main(parser.parse_args())
