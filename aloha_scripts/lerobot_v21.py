#!/usr/bin/env python3
"""Small LeRobot v2.1 writer/reader helpers for ALOHA scripts."""

from __future__ import annotations

import json
import os
import pwd
from pathlib import Path

import numpy as np


def lerobot_home():
    for env_name in ("HF_LEROBOT_HOME", "LEROBOT_HOME"):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot"

    home = Path.home()
    sudo_user = os.environ.get("SUDO_USER")
    if home == Path("/root") and sudo_user:
        try:
            home = Path(pwd.getpwnam(sudo_user).pw_dir)
        except KeyError:
            pass

    relocated_home = relocated_lerobot_home(home)
    if relocated_home is not None:
        return relocated_home

    return home / ".cache" / "huggingface" / "lerobot"


def relocated_lerobot_home(home):
    candidates = []
    username = home.name

    if username:
        candidates.append(Path("/media") / username / "home" / "hf" / "lerobot")

    media_root = Path("/media")
    if media_root.exists():
        candidates.extend(media_root.glob("*/home/hf/lerobot"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_dataset_root(repo_id):
    return lerobot_home() / repo_id


def push_dataset_to_hub(dataset_root, repo_id, private=False):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Could not import huggingface_hub. Install it or install the official LeRobot package.") from exc

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(dataset_root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload LeRobot dataset",
    )


def feature_names(dim):
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


def video_key(camera_name):
    return f"observation.images.{camera_name}"


def episode_chunk(episode_index):
    return episode_index // 1000


def parquet_path(dataset_root, episode_index):
    chunk = episode_chunk(episode_index)
    return Path(dataset_root) / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def video_path(dataset_root, episode_index, camera_name):
    chunk = episode_chunk(episode_index)
    return (
        Path(dataset_root)
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key(camera_name)
        / f"episode_{episode_index:06d}.mp4"
    )


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def write_parquet(path, rows):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Could not import pyarrow. Install it with: pip install pyarrow") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_episode_actions(dataset_root, episode_index):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Could not import pyarrow. Install it with: pip install pyarrow") from exc

    path = parquet_path(dataset_root, episode_index)
    if not path.exists():
        raise SystemExit(f"Could not find LeRobot episode parquet at {path}")

    table = pq.read_table(path, columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def stats_for_array(array):
    array = np.asarray(array, dtype=np.float32)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def combine_stats(stat_list):
    if not stat_list:
        return {}

    count = sum(int(stats["count"][0]) for stats in stat_list)
    mins = np.stack([np.asarray(stats["min"], dtype=np.float64) for stats in stat_list])
    maxs = np.stack([np.asarray(stats["max"], dtype=np.float64) for stats in stat_list])
    means = np.stack([np.asarray(stats["mean"], dtype=np.float64) for stats in stat_list])
    stds = np.stack([np.asarray(stats["std"], dtype=np.float64) for stats in stat_list])
    counts = np.asarray([int(stats["count"][0]) for stats in stat_list], dtype=np.float64)

    mean = np.sum(means * counts[:, None], axis=0) / count
    second_moment = np.sum((stds**2 + means**2) * counts[:, None], axis=0) / count
    variance = np.maximum(second_moment - mean**2, 0.0)

    return {
        "min": mins.min(axis=0).tolist(),
        "max": maxs.max(axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [int(count)],
    }


def build_features(camera_names, image_shape, state_dim, action_dim, fps, include_velocity, include_effort):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": feature_names(state_dim),
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": feature_names(action_dim),
        },
    }
    if include_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": feature_names(state_dim),
        }
    if include_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": feature_names(state_dim),
        }

    height, width, channels = image_shape
    for camera_name in camera_names:
        features[video_key(camera_name)] = {
            "dtype": "video",
            "shape": [int(height), int(width), int(channels)],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": int(height),
                "video.width": int(width),
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": int(fps),
                "video.channels": int(channels),
                "has_audio": False,
            },
        }

    for key, dtype in {
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    }.items():
        features[key] = {"dtype": dtype, "shape": [1], "names": None}

    return features


def existing_episode_records(dataset_root):
    return read_jsonl(Path(dataset_root) / "meta" / "episodes.jsonl")


def next_episode_index(dataset_root):
    records = existing_episode_records(dataset_root)
    if records:
        return max(int(record["episode_index"]) for record in records) + 1

    data_root = Path(dataset_root) / "data"
    if not data_root.exists():
        return 0

    indices = []
    for path in data_root.glob("chunk-*/episode_*.parquet"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def total_existing_frames(dataset_root):
    return sum(int(record["length"]) for record in existing_episode_records(dataset_root))


def episode_start_index(dataset_root, episode_index):
    records = sorted(existing_episode_records(dataset_root), key=lambda record: int(record["episode_index"]))
    return sum(
        int(record["length"])
        for record in records
        if int(record["episode_index"]) < int(episode_index)
    )


def remove_episode_files(dataset_root, episode_index):
    paths = [parquet_path(dataset_root, episode_index)]
    paths.extend(Path(dataset_root).glob(f"videos/chunk-*/observation.images.*/episode_{episode_index:06d}.mp4"))

    removed_paths = []
    for path in paths:
        if path.exists():
            path.unlink()
            removed_paths.append(path)
    return removed_paths


def update_metadata(dataset_root, episode_index, frame_count, task, fps, robot_type, features, episode_stats):
    dataset_root = Path(dataset_root)
    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)

    task_records = read_jsonl(meta_root / "tasks.jsonl")
    if not task_records:
        task_records = [{"task_index": 0, "task": task}]

    episode_records = [
        record
        for record in read_jsonl(meta_root / "episodes.jsonl")
        if int(record["episode_index"]) != int(episode_index)
    ]
    episode_records.append({"episode_index": int(episode_index), "tasks": [task], "length": int(frame_count)})
    episode_records = sorted(episode_records, key=lambda record: int(record["episode_index"]))

    episode_stats_records = [
        record
        for record in read_jsonl(meta_root / "episodes_stats.jsonl")
        if int(record["episode_index"]) != int(episode_index)
    ]
    episode_stats_records.append({"episode_index": int(episode_index), "stats": episode_stats})
    episode_stats_records = sorted(episode_stats_records, key=lambda record: int(record["episode_index"]))

    total_frames = sum(int(record["length"]) for record in episode_records)
    image_keys = [key for key in features if key.startswith("observation.images.")]
    max_episode_index = max(int(record["episode_index"]) for record in episode_records)

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episode_records),
        "total_frames": total_frames,
        "total_tasks": len(task_records),
        "total_videos": len(episode_records) * len(image_keys),
        "total_chunks": max(1, episode_chunk(max_episode_index) + 1),
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{len(episode_records)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    stats_by_key = {}
    for record in episode_stats_records:
        for key, value in record["stats"].items():
            stats_by_key.setdefault(key, []).append(value)
    stats = {key: combine_stats(values) for key, values in stats_by_key.items()}

    write_json(meta_root / "info.json", info)
    write_json(meta_root / "stats.json", stats)
    write_jsonl(meta_root / "tasks.jsonl", task_records)
    write_jsonl(meta_root / "episodes.jsonl", episode_records)
    write_jsonl(meta_root / "episodes_stats.jsonl", episode_stats_records)
