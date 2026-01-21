from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class FrameData:
    frame_id: str
    left_image_path: Path
    right_image_path: Path
    depth_path: Path
    cloud_path: Path
    intrinsics: np.ndarray
    baseline_m: float
    depth_m: np.ndarray | None = None


def _parse_intrinsics(k_path: Path) -> tuple[np.ndarray, float]:
    lines = k_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Expected 2 lines in {k_path}, got {len(lines)}")
    k_values = [float(v) for v in lines[0].split()]
    if len(k_values) != 9:
        raise ValueError(f"Expected 9 intrinsics in {k_path}, got {len(k_values)}")
    intrinsics = np.array(k_values, dtype=np.float64).reshape(3, 3)
    baseline_m = float(lines[1].strip())
    return intrinsics, baseline_m


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_frame(
    frame_dir: Path,
    *,
    load_depth: bool = True,
) -> FrameData:
    left_path = frame_dir / "rect_left.jpg"
    right_path = frame_dir / "rect_right.jpg"
    depth_path = frame_dir / "depth_meter.npy"
    cloud_path = frame_dir / "cloud.ply"
    k_path = frame_dir / "K.txt"

    for required in (left_path, right_path, depth_path, cloud_path, k_path):
        _require_file(required)

    intrinsics, baseline_m = _parse_intrinsics(k_path)
    depth_m = np.load(depth_path) if load_depth else None

    return FrameData(
        frame_id=frame_dir.name,
        left_image_path=left_path,
        right_image_path=right_path,
        depth_path=depth_path,
        cloud_path=cloud_path,
        intrinsics=intrinsics,
        baseline_m=baseline_m,
        depth_m=depth_m,
    )


def discover_frames(root: Path) -> List[Path]:
    frame_dirs = [p for p in root.iterdir() if p.is_dir()]
    return sorted(frame_dirs, key=lambda p: p.name)


def load_dataset(
    root: Path,
    *,
    load_depth: bool = True,
) -> List[FrameData]:
    frames: List[FrameData] = []
    for frame_dir in discover_frames(root):
        frames.append(load_frame(frame_dir, load_depth=load_depth))
    return frames


def iter_frames(
    roots: Iterable[Path],
    *,
    load_depth: bool = True,
) -> List[FrameData]:
    frames: List[FrameData] = []
    for root in roots:
        frames.extend(load_dataset(root, load_depth=load_depth))
    return frames
