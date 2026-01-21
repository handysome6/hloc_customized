from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .dataset import FrameData
from .geometry import Pose, transform_points


def load_ply_ascii(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header in {path}")
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        if not header[0].startswith("ply"):
            raise ValueError(f"Missing PLY magic in {path}")
        fmt_line = next((line for line in header if line.startswith("format ")), "")
        if "ascii" not in fmt_line:
            raise ValueError(f"Only ASCII PLY supported: {path}")
        count_line = next((line for line in header if line.startswith("element vertex")), "")
        if not count_line:
            raise ValueError(f"Missing vertex count in {path}")
        vertex_count = int(count_line.split()[-1])

        points = []
        for _ in range(vertex_count):
            row = handle.readline()
            if not row:
                break
            values = row.strip().split()
            if len(values) < 3:
                continue
            points.append([float(values[0]), float(values[1]), float(values[2])])

    return np.array(points, dtype=np.float64)


def save_ply_ascii(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x} {y} {z}\n")


def fuse_point_clouds(
    frames: List[FrameData],
    poses: Dict[str, Pose],
    output_path: Path,
) -> None:
    all_points = []
    for frame in frames:
        pose = poses.get(frame.frame_id)
        if pose is None:
            continue
        points = load_ply_ascii(frame.cloud_path)
        points_world = transform_points(points, pose)
        all_points.append(points_world)
    if not all_points:
        raise ValueError("No point clouds available to fuse.")
    fused = np.vstack(all_points)
    save_ply_ascii(output_path, fused)
