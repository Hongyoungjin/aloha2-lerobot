#!/usr/bin/env python3
"""Create quick-look videos from local LeRobot v2.1 ALOHA datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from constants import DT
from lerobot_v21 import default_dataset_root, video_path


def load_info(dataset_root):
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(
            f"Could not find LeRobot metadata at {info_path}\n"
            "If you set HF_LEROBOT_HOME in ~/.zshrc, run this script from a zsh login shell "
            "or pass --dataset-root explicitly."
        )
    return json.loads(info_path.read_text(encoding="utf-8"))


def camera_names_from_info(info):
    camera_names = []
    prefix = "observation.images."
    for key in info["features"]:
        if key.startswith(prefix):
            camera_names.append(key[len(prefix):])
    return sorted(camera_names)


def open_video_captures(dataset_root, episode_index, camera_names):
    captures = {}
    for camera_name in camera_names:
        path = video_path(dataset_root, episode_index, camera_name)
        if not path.exists():
            raise SystemExit(f"Could not find camera video for {camera_name}: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise SystemExit(f"Could not open camera video for {camera_name}: {path}")
        captures[camera_name] = capture
    return captures


def close_video_captures(captures):
    for capture in captures.values():
        capture.release()


def read_next_frame(captures, camera_names):
    frames = []
    for camera_name in camera_names:
        ok, frame = captures[camera_name].read()
        if not ok:
            return None
        frames.append(frame)
    return np.concatenate(frames, axis=1)


def save_episode_video(dataset_root, episode_index, camera_names, output_path, fps):
    captures = open_video_captures(dataset_root, episode_index, camera_names)
    writer = None
    frame_count = 0
    try:
        while True:
            frame = read_next_frame(captures, camera_names)
            if frame is None:
                break
            if writer is None:
                height, width = frame.shape[:2]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(width), int(height)))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}")
            writer.write(frame)
            frame_count += 1
    finally:
        close_video_captures(captures)
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames were read for episode {episode_index}")
    print(f"Saved {frame_count} frames to: {output_path}")


def resolve_output_path(output, dataset_root, episode_index):
    default_name = f"episode_{episode_index}_video.mp4"
    if output is None:
        return dataset_root / default_name

    output_path = Path(output).expanduser()
    if output_path.exists() and output_path.is_dir():
        return output_path / default_name
    if output_path.suffix.lower() != ".mp4":
        return output_path / default_name
    return output_path


def main(args):
    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser()
    elif args.repo_id:
        dataset_root = default_dataset_root(args.repo_id)
    else:
        raise SystemExit("Provide either --dataset-root or --repo-id.")

    info = load_info(dataset_root)
    fps = args.fps if args.fps is not None else int(info.get("fps", int(1 / DT)))
    camera_names = args.camera_names if args.camera_names else camera_names_from_info(info)
    if not camera_names:
        raise SystemExit("No camera video features found in LeRobot metadata.")

    output_path = resolve_output_path(args.output, dataset_root, args.episode_idx)
    save_episode_video(dataset_root, args.episode_idx, camera_names, output_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=None, help="Local LeRobot v2.1 dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id. Uses HF_LEROBOT_HOME/<repo-id>, HF_HOME/lerobot/<repo-id>, or ~/.cache/huggingface/lerobot/<repo-id>.",
    )
    parser.add_argument("--episode_idx", type=int, required=True, help="LeRobot episode index to visualize.")
    parser.add_argument("--output", type=str, default=None, help="Output combined mp4 path, or a directory to receive episode_<idx>_video.mp4.")
    parser.add_argument("--fps", type=int, default=None, help="Override output fps. Defaults to dataset metadata fps.")
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=None,
        help="Camera names to include. Defaults to all observation.images.* keys in metadata.",
    )
    main(parser.parse_args())
