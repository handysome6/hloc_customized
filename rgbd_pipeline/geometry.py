from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Pose:
    matrix: np.ndarray

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(4, dtype=np.float64))

    def inverse(self) -> "Pose":
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        inv = np.eye(4, dtype=np.float64)
        inv[:3, :3] = rotation.T
        inv[:3, 3] = -rotation.T @ translation
        return Pose(inv)

    def compose(self, other: "Pose") -> "Pose":
        return Pose(self.matrix @ other.matrix)


def pose_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
    rmat, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rmat
    pose[:3, 3] = tvec.reshape(3)
    return Pose(pose)


def transform_points(points: np.ndarray, pose: Pose) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    hom = np.hstack([points, ones])
    transformed = (pose.matrix @ hom.T).T
    return transformed[:, :3]


def project_depth_to_points(
    keypoints: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape[:2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    points_3d = []
    valid_indices = []
    for idx, (u, v) in enumerate(keypoints):
        x = int(round(u))
        y = int(round(v))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        z = float(depth_m[y, x])
        if z <= 0.0 or np.isnan(z):
            continue
        x_cam = (u - cx) / fx * z
        y_cam = (v - cy) / fy * z
        points_3d.append([x_cam, y_cam, z])
        valid_indices.append(idx)

    if not points_3d:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32)

    return np.array(points_3d, dtype=np.float64), np.array(valid_indices, dtype=np.int32)


def rotation_error_deg(pose: Pose) -> float:
    rmat = pose.matrix[:3, :3]
    trace = np.clip((np.trace(rmat) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    return float(np.degrees(angle))


def translation_error_m(pose: Pose) -> float:
    return float(np.linalg.norm(pose.matrix[:3, 3]))


def solve_pnp_ransac(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsics: np.ndarray,
    *,
    reproj_error_px: float = 4.0,
    min_inliers: int = 30,
) -> Tuple[Pose | None, int, np.ndarray]:
    if points_3d.shape[0] < min_inliers:
        return None, 0, np.zeros((0,), dtype=np.int32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        intrinsics,
        dist_coeffs,
        reprojectionError=reproj_error_px,
        iterationsCount=2000,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None, 0, np.zeros((0,), dtype=np.int32)
    pose = pose_from_rvec_tvec(rvec, tvec)
    return pose, int(len(inliers)), inliers.reshape(-1)


try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("OpenCV is required for pose estimation") from exc
