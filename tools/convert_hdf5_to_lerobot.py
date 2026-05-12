#!/usr/bin/env python3
"""Convert ALOHA HDF5 episodes to a LeRobot dataset.

The default target is a LeRobot v2.1-compatible layout because many VLA and
imitation-learning repos still use that format. The v3.0 target uses the
official current LeRobot API.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def import_lerobot_dataset():
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "Could not import lerobot. Install the official LeRobot package in this environment, "
            "then rerun this converter."
        ) from exc
    return LeRobotDataset


def call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return fn(**supported)


def dataset_root(output_root: Path | None, repo_id: str) -> Path:
    if output_root is not None:
        return output_root
    return Path(repo_id.replace("/", "_"))


def require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("Could not import h5py. Install h5py to read ALOHA HDF5 episodes.") from exc
    return h5py


def discover_schema(episode_paths: list[Path], include_velocity: bool, include_effort: bool) -> dict:
    h5py = require_h5py()

    if not episode_paths:
        raise SystemExit("No episode_*.hdf5 files found.")

    with h5py.File(episode_paths[0], "r") as root:
        qpos_shape = root["/observations/qpos"].shape
        action_shape = root["/action"].shape
        camera_names = sorted(root["/observations/images"].keys())
        if not camera_names:
            raise ValueError(f"{episode_paths[0]} has no cameras under /observations/images")
        image_shape = root[f"/observations/images/{camera_names[0]}"].shape[1:]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (qpos_shape[1],),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_shape[1],),
            "names": ["action"],
        },
    }
    if include_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (qpos_shape[1],),
            "names": ["velocity"],
        }
    if include_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (qpos_shape[1],),
            "names": ["effort"],
        }
    for camera_name in camera_names:
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": int(image_shape[0]),
                "video.width": int(image_shape[1]),
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 50,
                "video.channels": int(image_shape[2]),
                "has_audio": False,
            },
        }
    return {"features": features, "camera_names": camera_names}


def v3_features(features: dict) -> dict:
    converted = {}
    for key, value in features.items():
        feature = dict(value)
        if key.startswith("observation.images."):
            feature["dtype"] = "image"
            feature.pop("info", None)
        converted[key] = feature
    return converted


def feature_names(dim: int) -> list[str]:
    left = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    right = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    names = left + right
    return names if len(names) == dim else [f"dim_{idx}" for idx in range(dim)]


def normalize_feature_schema(features: dict, fps: int) -> dict:
    normalized = {}
    for key, feature in features.items():
        item = dict(feature)
        item["shape"] = list(item["shape"])
        if key in {"observation.state", "action"}:
            item["names"] = feature_names(item["shape"][0])
        if key.startswith("observation.images."):
            item["info"] = dict(item["info"])
            item["info"]["video.fps"] = fps
        normalized[key] = item

    for key, dtype in {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }.items():
        normalized[key] = {"dtype": dtype, "shape": [1], "names": None}
    return normalized


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def encode_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Could not import cv2. Install opencv-python to encode v2.1 videos.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    for frame in frames:
        writer.write(frame[:, :, [2, 1, 0]])
    writer.release()


def write_parquet(path: Path, rows: list[dict]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Could not import pyarrow. Install pyarrow to write LeRobot v2.1 parquet files.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def stats_for_array(array: np.ndarray) -> dict:
    array = np.asarray(array, dtype=np.float32)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def combine_stats(values: list[np.ndarray]) -> dict:
    return stats_for_array(np.concatenate(values, axis=0))


def convert_v21(
    episode_paths: list[Path],
    schema: dict,
    output_dir: Path,
    task: str,
    fps: int,
    robot_type: str,
    include_velocity: bool,
    include_effort: bool,
) -> tuple[int, int]:
    h5py = require_h5py()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    task_records = [{"task_index": 0, "task": task}]
    episode_records = []
    episode_stats = []
    all_state = []
    all_action = []
    all_velocity = []
    all_effort = []
    total_frames = 0

    for episode_index, path in enumerate(episode_paths):
        episode_chunk = episode_index // 1000
        parquet_path = output_dir / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"

        with h5py.File(path, "r") as root:
            qpos = np.asarray(root["/observations/qpos"], dtype=np.float32)
            action = np.asarray(root["/action"], dtype=np.float32)
            qvel = np.asarray(root["/observations/qvel"], dtype=np.float32) if include_velocity else None
            effort = (
                np.asarray(root["/observations/effort"], dtype=np.float32)
                if include_effort and "/observations/effort" in root
                else None
            )
            frame_count = int(action.shape[0])

            rows = []
            for frame_index in range(frame_count):
                row = {
                    "observation.state": qpos[frame_index].tolist(),
                    "action": action[frame_index].tolist(),
                    "timestamp": np.float32(frame_index / fps).item(),
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": total_frames + frame_index,
                    "task_index": 0,
                }
                if qvel is not None:
                    row["observation.velocity"] = qvel[frame_index].tolist()
                if effort is not None:
                    row["observation.effort"] = effort[frame_index].tolist()
                rows.append(row)
            write_parquet(parquet_path, rows)

            for camera_name in schema["camera_names"]:
                video_key = f"observation.images.{camera_name}"
                video_path = (
                    output_dir
                    / "videos"
                    / f"chunk-{episode_chunk:03d}"
                    / video_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                frames = np.asarray(root[f"/observations/images/{camera_name}"])
                encode_video(frames, video_path, fps)

        episode_records.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": frame_count,
            }
        )
        stats_record = {
            "episode_index": episode_index,
            "stats": {
                "observation.state": stats_for_array(qpos),
                "action": stats_for_array(action),
            },
        }
        if qvel is not None:
            stats_record["stats"]["observation.velocity"] = stats_for_array(qvel)
        if effort is not None:
            stats_record["stats"]["observation.effort"] = stats_for_array(effort)
        episode_stats.append(stats_record)

        all_state.append(qpos)
        all_action.append(action)
        if qvel is not None:
            all_velocity.append(qvel)
        if effort is not None:
            all_effort.append(effort)
        total_frames += frame_count
        print(f"Converted {path.name}: {frame_count} frames")

    features = normalize_feature_schema(schema["features"], fps)
    if not include_velocity:
        features.pop("observation.velocity", None)
    if not include_effort:
        features.pop("observation.effort", None)

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episode_paths),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_paths) * len(schema["camera_names"]),
        "total_chunks": max(1, (len(episode_paths) + 999) // 1000),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episode_paths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    stats = {
        "observation.state": combine_stats(all_state),
        "action": combine_stats(all_action),
    }
    if all_velocity:
        stats["observation.velocity"] = combine_stats(all_velocity)
    if all_effort:
        stats["observation.effort"] = combine_stats(all_effort)

    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (output_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "meta" / "tasks.jsonl", task_records)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", episode_records)
    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", episode_stats)
    return len(episode_paths), total_frames


def add_episode_v30(
    dataset,
    path: Path,
    camera_names: list[str],
    task: str,
    include_velocity: bool,
    include_effort: bool,
) -> int:
    h5py = require_h5py()

    with h5py.File(path, "r") as root:
        qpos = root["/observations/qpos"]
        qvel = root["/observations/qvel"] if include_velocity else None
        effort = root["/observations/effort"] if include_effort and "/observations/effort" in root else None
        action = root["/action"]
        frame_count = action.shape[0]

        for frame_idx in range(frame_count):
            frame = {
                "observation.state": np.asarray(qpos[frame_idx], dtype=np.float32),
                "action": np.asarray(action[frame_idx], dtype=np.float32),
            }
            if qvel is not None:
                frame["observation.velocity"] = np.asarray(qvel[frame_idx], dtype=np.float32)
            if effort is not None:
                frame["observation.effort"] = np.asarray(effort[frame_idx], dtype=np.float32)
            for camera_name in camera_names:
                frame[f"observation.images.{camera_name}"] = root[f"/observations/images/{camera_name}"][frame_idx]
            dataset.add_frame(frame)

    save_episode = dataset.save_episode
    signature = inspect.signature(save_episode)
    if "task" in signature.parameters:
        save_episode(task=task)
    else:
        save_episode()
    return frame_count


def convert_v30(
    episode_paths: list[Path],
    schema: dict,
    output_root: Path | None,
    repo_id: str,
    task: str,
    fps: int,
    robot_type: str,
    include_velocity: bool,
    include_effort: bool,
    no_videos: bool,
    image_writer_threads: int,
) -> tuple[int, int]:
    LeRobotDataset = import_lerobot_dataset()

    dataset = call_with_supported_kwargs(
        LeRobotDataset.create,
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        features=v3_features(schema["features"]),
        robot_type=robot_type,
        use_videos=not no_videos,
        image_writer_threads=image_writer_threads,
    )

    total_frames = 0
    for episode_path in episode_paths:
        frames = add_episode_v30(
            dataset,
            episode_path,
            schema["camera_names"],
            task,
            include_velocity,
            include_effort,
        )
        total_frames += frames
        print(f"Converted {episode_path.name}: {frames} frames")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    return len(episode_paths), total_frames


def run_official_v21_to_v30(repo_id: str) -> None:
    command = [
        sys.executable,
        "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing episode_*.hdf5 files.")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id, e.g. hong/aloha_real_stack.")
    parser.add_argument("--output-root", type=Path, default=None, help="Local output directory. Defaults to repo_id with '/' replaced by '_'.")
    parser.add_argument("--format", choices=["v2.1", "v3.0"], default="v2.1", help="Output format. v2.1 is the default for VLA compatibility.")
    parser.add_argument("--task", required=True, help="Natural-language task description stored in metadata.")
    parser.add_argument("--fps", type=int, default=50, help="Dataset frame rate. ALOHA real collection uses 50 Hz.")
    parser.add_argument("--robot-type", default="aloha", help="Robot type metadata.")
    parser.add_argument("--no-videos", action="store_true", help="For v3.0 only: store images instead of encoded video if supported.")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--include-velocity", action="store_true", help="Also export /observations/qvel.")
    parser.add_argument("--include-effort", action="store_true", help="Also export /observations/effort when present.")
    parser.add_argument(
        "--also-migrate-v21-to-v30",
        action="store_true",
        help="After a v2.1 export, call LeRobot's official v2.1 -> v3.0 migration module for this repo id.",
    )
    parser.add_argument(
        "--delete-source-hdf5",
        action="store_true",
        help="Delete source episode_*.hdf5 files after conversion finalizes successfully.",
    )
    args = parser.parse_args()

    episode_paths = sorted(args.input_dir.glob("episode_*.hdf5"))
    schema = discover_schema(episode_paths, args.include_velocity, args.include_effort)

    if args.format == "v2.1":
        output_dir = dataset_root(args.output_root, args.repo_id)
        episode_count, total_frames = convert_v21(
            episode_paths,
            schema,
            output_dir,
            args.task,
            args.fps,
            args.robot_type,
            args.include_velocity,
            args.include_effort,
        )
        print(f"Converted {episode_count} episodes, {total_frames} frames to v2.1 at {output_dir}")
        if args.also_migrate_v21_to_v30:
            run_official_v21_to_v30(args.repo_id)
    else:
        episode_count, total_frames = convert_v30(
            episode_paths,
            schema,
            args.output_root,
            args.repo_id,
            args.task,
            args.fps,
            args.robot_type,
            args.include_velocity,
            args.include_effort,
            args.no_videos,
            args.image_writer_threads,
        )
        print(f"Converted {episode_count} episodes, {total_frames} frames to v3.0 repo {args.repo_id}")

    print("Validate the converted dataset in the target training stack before deleting raw HDF5.")

    if args.delete_source_hdf5:
        for episode_path in episode_paths:
            episode_path.unlink()
        print(f"Deleted {len(episode_paths)} source HDF5 files from {args.input_dir}")


if __name__ == "__main__":
    main()
