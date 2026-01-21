from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d

from .dataset import FrameData
from .geometry import Pose


def _pose_to_matrix(pose: Pose) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = pose.matrix[:3, :3]
    matrix[:3, 3] = pose.matrix[:3, 3]
    return matrix


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Loaded empty point cloud: {path}")
    return cloud


def save_point_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> None:
    if cloud.is_empty():
        raise ValueError("No points available to save.")
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=True)


def fuse_point_clouds(
    frames: List[FrameData],
    poses: Dict[str, Pose],
    output_path: Path,
) -> None:
    fused = o3d.geometry.PointCloud()
    for frame in frames:
        pose = poses.get(frame.frame_id)
        if pose is None:
            continue
        cloud = load_point_cloud(frame.cloud_path)
        cloud.transform(_pose_to_matrix(pose))
        fused += cloud
    save_point_cloud(output_path, fused)
