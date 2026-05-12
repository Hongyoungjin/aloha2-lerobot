#!/usr/bin/env python3
"""Inspect ALOHA HDF5 episodes before conversion."""

import argparse
from pathlib import Path


def inspect_episode(path: Path) -> dict:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("Could not import h5py. Install h5py to inspect ALOHA episodes.") from exc

    with h5py.File(path, "r") as root:
        required = [
            "/observations/qpos",
            "/observations/qvel",
            "/action",
            "/observations/images",
        ]
        missing = [key for key in required if key not in root]
        if missing:
            raise ValueError(f"{path}: missing required datasets/groups: {missing}")

        qpos_shape = root["/observations/qpos"].shape
        qvel_shape = root["/observations/qvel"].shape
        action_shape = root["/action"].shape
        effort_shape = root["/observations/effort"].shape if "/observations/effort" in root else None
        camera_names = sorted(root["/observations/images"].keys())
        image_shapes = {
            cam: root[f"/observations/images/{cam}"].shape
            for cam in camera_names
        }

        frame_count = action_shape[0]
        expected_frame_shapes = [qpos_shape[0], qvel_shape[0]]
        if effort_shape is not None:
            expected_frame_shapes.append(effort_shape[0])
        expected_frame_shapes.extend(shape[0] for shape in image_shapes.values())
        if any(length != frame_count for length in expected_frame_shapes):
            raise ValueError(
                f"{path}: frame count mismatch, action has {frame_count}, "
                f"other streams have {expected_frame_shapes}"
            )

        sim = bool(root.attrs.get("sim", False))

    return {
        "path": path,
        "sim": sim,
        "frames": frame_count,
        "qpos_shape": qpos_shape,
        "qvel_shape": qvel_shape,
        "effort_shape": effort_shape,
        "action_shape": action_shape,
        "camera_names": camera_names,
        "image_shapes": image_shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Directory containing episode_*.hdf5 files.")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect the first N episodes.")
    args = parser.parse_args()

    episode_paths = sorted(args.dataset_dir.glob("episode_*.hdf5"))
    if args.limit is not None:
        episode_paths = episode_paths[: args.limit]
    if not episode_paths:
        raise SystemExit(f"No episode_*.hdf5 files found in {args.dataset_dir}")

    total_frames = 0
    for path in episode_paths:
        info = inspect_episode(path)
        total_frames += info["frames"]
        print(f"{path.name}: sim={info['sim']} frames={info['frames']}")
        print(f"  qpos={info['qpos_shape']} qvel={info['qvel_shape']} action={info['action_shape']}")
        if info["effort_shape"] is not None:
            print(f"  effort={info['effort_shape']}")
        for cam, shape in info["image_shapes"].items():
            print(f"  {cam}: {shape}")

    print(f"\nOK: {len(episode_paths)} episodes, {total_frames} total frames")


if __name__ == "__main__":
    main()
