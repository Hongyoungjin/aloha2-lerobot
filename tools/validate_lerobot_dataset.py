#!/usr/bin/env python3
"""Smoke-check a converted LeRobot dataset."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="LeRobot repo id or local dataset id.")
    parser.add_argument("--root", type=Path, default=None, help="Optional local root passed to LeRobotDataset.")
    parser.add_argument("--expected-fps", type=int, default=50)
    args = parser.parse_args()

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise SystemExit("Could not import LeRobotDataset in this environment.") from exc

    kwargs = {"repo_id": args.repo_id}
    if args.root is not None:
        kwargs["root"] = args.root
    dataset = LeRobotDataset(**kwargs)
    if len(dataset) == 0:
        raise SystemExit("Dataset loads but has zero frames.")

    first = dataset[0]
    last = dataset[len(dataset) - 1]
    required = ["observation.state", "action", "timestamp"]
    for key in required:
        if key not in first:
            raise SystemExit(f"Missing required key in first sample: {key}")

    fps = getattr(getattr(dataset, "meta", None), "fps", None)
    if fps is not None and int(fps) != args.expected_fps:
        raise SystemExit(f"Expected fps={args.expected_fps}, got {fps}")

    image_keys = [key for key in first if key.startswith("observation.images.")]
    print(f"OK: {len(dataset)} frames")
    print(f"state shape: {tuple(first['observation.state'].shape)}")
    print(f"action shape: {tuple(first['action'].shape)}")
    print(f"image keys: {image_keys}")
    print(f"first timestamp: {first['timestamp']}")
    print(f"last timestamp: {last['timestamp']}")


if __name__ == "__main__":
    main()
