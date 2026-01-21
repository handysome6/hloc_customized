from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .dataset import FrameData
from .features import MatchResult, TorchFeatures
from .matching import PairMatches
from .geometry import (
    Pose,
    project_depth_to_points,
    rotation_error_deg,
    solve_pnp_ransac,
    translation_error_m,
)


@dataclass(frozen=True)
class VerifiedEdge:
    frame_id0: str
    frame_id1: str
    pose_1_0: Pose
    inliers: int


def _build_correspondences(
    frame_src: FrameData,
    frame_tgt: FrameData,
    feats_src: TorchFeatures,
    feats_tgt: TorchFeatures,
    matches: MatchResult,
) -> Tuple[np.ndarray, np.ndarray]:
    keypoints_src = feats_src.keypoints.cpu().numpy()
    keypoints_tgt = feats_tgt.keypoints.cpu().numpy()
    idx_src = matches.pairs[:, 0]
    idx_tgt = matches.pairs[:, 1]
    pts_src = keypoints_src[idx_src]
    pts_tgt = keypoints_tgt[idx_tgt]
    points_3d, valid_idx = project_depth_to_points(
        pts_src, frame_src.depth_m, frame_src.intrinsics
    )
    if valid_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    points_2d = pts_tgt[valid_idx].astype(np.float64)
    return points_3d, points_2d


def verify_pair_bidirectional(
    frame_i: FrameData,
    frame_j: FrameData,
    feats_i: TorchFeatures,
    feats_j: TorchFeatures,
    matches_ij: MatchResult,
    *,
    min_inliers: int = 30,
    reproj_error_px: float = 4.0,
    max_rot_deg: float = 10.0,
    max_trans_m: float = 0.5,
) -> VerifiedEdge | None:
    if frame_i.depth_m is None or frame_j.depth_m is None:
        raise ValueError("Depth maps must be loaded for PnP verification.")

    points_3d, points_2d = _build_correspondences(
        frame_i, frame_j, feats_i, feats_j, matches_ij
    )
    pose_ji, inliers_ij, _ = solve_pnp_ransac(
        points_3d,
        points_2d,
        frame_j.intrinsics,
        reproj_error_px=reproj_error_px,
        min_inliers=min_inliers,
    )
    if pose_ji is None:
        return None

    points_3d_rev, points_2d_rev = _build_correspondences(
        frame_j, frame_i, feats_j, feats_i, matches_ij
    )
    pose_ij, inliers_ji, _ = solve_pnp_ransac(
        points_3d_rev,
        points_2d_rev,
        frame_i.intrinsics,
        reproj_error_px=reproj_error_px,
        min_inliers=min_inliers,
    )
    if pose_ij is None:
        return None

    consistency = pose_ji.compose(pose_ij)
    if rotation_error_deg(consistency) > max_rot_deg:
        return None
    if translation_error_m(consistency) > max_trans_m:
        return None

    return VerifiedEdge(
        frame_id0=frame_i.frame_id,
        frame_id1=frame_j.frame_id,
        pose_1_0=pose_ji,
        inliers=min(inliers_ij, inliers_ji),
    )


def build_verified_edges(
    frames: List[FrameData],
    features: Dict[str, TorchFeatures],
    matches: List[PairMatches],
    *,
    min_inliers: int = 30,
    reproj_error_px: float = 4.0,
    max_rot_deg: float = 10.0,
    max_trans_m: float = 0.5,
) -> List[VerifiedEdge]:
    frames_by_id = {frame.frame_id: frame for frame in frames}
    edges: List[VerifiedEdge] = []
    for pair in matches:
        frame_i = frames_by_id[pair.frame_id0]
        frame_j = frames_by_id[pair.frame_id1]
        edge = verify_pair_bidirectional(
            frame_i,
            frame_j,
            features[pair.frame_id0],
            features[pair.frame_id1],
            pair.matches,
            min_inliers=min_inliers,
            reproj_error_px=reproj_error_px,
            max_rot_deg=max_rot_deg,
            max_trans_m=max_trans_m,
        )
        if edge is not None:
            edges.append(edge)
    return edges


class _UnionFind:
    def __init__(self, items: List[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True


def maximum_spanning_tree(
    frame_ids: List[str],
    edges: List[VerifiedEdge],
) -> List[VerifiedEdge]:
    sorted_edges = sorted(edges, key=lambda e: e.inliers, reverse=True)
    uf = _UnionFind(frame_ids)
    mst: List[VerifiedEdge] = []
    for edge in sorted_edges:
        if uf.union(edge.frame_id0, edge.frame_id1):
            mst.append(edge)
        if len(mst) == len(frame_ids) - 1:
            break
    return mst


def initialize_poses(
    frame_ids: List[str],
    mst_edges: List[VerifiedEdge],
    root_id: str,
) -> Dict[str, Pose]:
    adjacency: Dict[str, List[Tuple[str, Pose]]] = {fid: [] for fid in frame_ids}
    for edge in mst_edges:
        adjacency[edge.frame_id0].append((edge.frame_id1, edge.pose_1_0))
        adjacency[edge.frame_id1].append((edge.frame_id0, edge.pose_1_0.inverse()))

    poses: Dict[str, Pose] = {root_id: Pose.identity()}
    stack = [root_id]
    while stack:
        current = stack.pop()
        for neighbor, rel_pose in adjacency[current]:
            if neighbor in poses:
                continue
            poses[neighbor] = poses[current].compose(rel_pose)
            stack.append(neighbor)
    return poses
