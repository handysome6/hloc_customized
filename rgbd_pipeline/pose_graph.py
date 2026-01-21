from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

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


@dataclass(frozen=True)
class VerificationStats:
    frame_id0: str
    frame_id1: str
    matches: int
    valid_3d_forward: int
    valid_3d_reverse: int
    depth_median_forward: float
    depth_median_reverse: float
    inliers_forward: int
    inliers_reverse: int
    rot_error_deg: float
    trans_error_m: float
    status: str
    reason: str


def _build_correspondences(
    frame_src: FrameData,
    frame_tgt: FrameData,
    feats_src: TorchFeatures,
    feats_tgt: TorchFeatures,
    matches: MatchResult,
    *,
    swap_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    keypoints_src = feats_src.keypoints.cpu().numpy()
    keypoints_tgt = feats_tgt.keypoints.cpu().numpy()
    if swap_indices:
        idx_src = matches.pairs[:, 1]
        idx_tgt = matches.pairs[:, 0]
    else:
        idx_src = matches.pairs[:, 0]
        idx_tgt = matches.pairs[:, 1]
    pts_src = keypoints_src[idx_src]
    pts_tgt = keypoints_tgt[idx_tgt]
    points_3d, valid_idx = project_depth_to_points(
        pts_src, frame_src.depth_m, frame_src.intrinsics
    )
    if valid_idx.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            0,
            0.0,
        )
    points_2d = pts_tgt[valid_idx].astype(np.float64)
    median_depth = float(np.median(points_3d[:, 2])) if points_3d.size else 0.0
    return points_3d, points_2d, int(points_3d.shape[0]), median_depth


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
) -> Tuple[VerifiedEdge | None, VerificationStats]:
    if frame_i.depth_m is None or frame_j.depth_m is None:
        stats = VerificationStats(
            frame_id0=frame_i.frame_id,
            frame_id1=frame_j.frame_id,
            matches=0,
            valid_3d_forward=0,
            valid_3d_reverse=0,
            depth_median_forward=0.0,
            depth_median_reverse=0.0,
            inliers_forward=0,
            inliers_reverse=0,
            rot_error_deg=0.0,
            trans_error_m=0.0,
            status="rejected",
            reason="missing_depth",
        )
        return None, stats

    points_3d, points_2d, valid_3d, depth_median_fwd = _build_correspondences(
        frame_i, frame_j, feats_i, feats_j, matches_ij
    )
    matches_count = int(matches_ij.pairs.shape[0])
    if valid_3d < min_inliers:
        stats = VerificationStats(
            frame_id0=frame_i.frame_id,
            frame_id1=frame_j.frame_id,
            matches=matches_count,
            valid_3d_forward=valid_3d,
            valid_3d_reverse=0,
            depth_median_forward=depth_median_fwd,
            depth_median_reverse=0.0,
            inliers_forward=0,
            inliers_reverse=0,
            rot_error_deg=0.0,
            trans_error_m=0.0,
            status="rejected",
            reason="too_few_3d_points",
        )
        return None, stats
    pose_ji, inliers_ij, _ = solve_pnp_ransac(
        points_3d,
        points_2d,
        frame_j.intrinsics,
        reproj_error_px=reproj_error_px,
        min_inliers=min_inliers,
    )
    if pose_ji is None:
        stats = VerificationStats(
            frame_id0=frame_i.frame_id,
            frame_id1=frame_j.frame_id,
            matches=matches_count,
            valid_3d_forward=valid_3d,
            valid_3d_reverse=0,
            depth_median_forward=depth_median_fwd,
            depth_median_reverse=0.0,
            inliers_forward=inliers_ij,
            inliers_reverse=0,
            rot_error_deg=0.0,
            trans_error_m=0.0,
            status="rejected",
            reason="pnp_failed_forward",
        )
        return None, stats

    points_3d_rev, points_2d_rev, valid_3d_rev, depth_median_rev = _build_correspondences(
        frame_j,
        frame_i,
        feats_j,
        feats_i,
        matches_ij,
        swap_indices=True,
    )
    if valid_3d_rev < min_inliers:
        stats = VerificationStats(
            frame_id0=frame_i.frame_id,
            frame_id1=frame_j.frame_id,
            matches=matches_count,
            valid_3d_forward=valid_3d,
            valid_3d_reverse=valid_3d_rev,
            depth_median_forward=depth_median_fwd,
            depth_median_reverse=depth_median_rev,
            inliers_forward=inliers_ij,
            inliers_reverse=0,
            rot_error_deg=0.0,
            trans_error_m=0.0,
            status="rejected",
            reason="too_few_3d_points_reverse",
        )
        return None, stats
    pose_ij, inliers_ji, _ = solve_pnp_ransac(
        points_3d_rev,
        points_2d_rev,
        frame_i.intrinsics,
        reproj_error_px=reproj_error_px,
        min_inliers=min_inliers,
    )
    if pose_ij is None:
        stats = VerificationStats(
            frame_id0=frame_i.frame_id,
            frame_id1=frame_j.frame_id,
            matches=matches_count,
            valid_3d_forward=valid_3d,
            valid_3d_reverse=valid_3d_rev,
            depth_median_forward=depth_median_fwd,
            depth_median_reverse=depth_median_rev,
            inliers_forward=inliers_ij,
            inliers_reverse=inliers_ji,
            rot_error_deg=0.0,
            trans_error_m=0.0,
            status="rejected",
            reason="pnp_failed_reverse",
        )
        return None, stats

    consistency = pose_ji.compose(pose_ij)
    rot_err = rotation_error_deg(consistency)
    trans_err = translation_error_m(consistency)
    if rot_err > max_rot_deg or trans_err > max_trans_m:
        stats = VerificationStats(
            frame_id0=frame_i.frame_id,
            frame_id1=frame_j.frame_id,
            matches=matches_count,
            valid_3d_forward=valid_3d,
            valid_3d_reverse=valid_3d_rev,
            depth_median_forward=depth_median_fwd,
            depth_median_reverse=depth_median_rev,
            inliers_forward=inliers_ij,
            inliers_reverse=inliers_ji,
            rot_error_deg=rot_err,
            trans_error_m=trans_err,
            status="rejected",
            reason="inconsistent_pose",
        )
        return None, stats

    edge = VerifiedEdge(
        frame_id0=frame_i.frame_id,
        frame_id1=frame_j.frame_id,
        pose_1_0=pose_ji,
        inliers=min(inliers_ij, inliers_ji),
    )
    stats = VerificationStats(
        frame_id0=frame_i.frame_id,
        frame_id1=frame_j.frame_id,
        matches=matches_count,
        valid_3d_forward=valid_3d,
        valid_3d_reverse=valid_3d_rev,
        depth_median_forward=depth_median_fwd,
        depth_median_reverse=depth_median_rev,
        inliers_forward=inliers_ij,
        inliers_reverse=inliers_ji,
        rot_error_deg=rot_err,
        trans_error_m=trans_err,
        status="accepted",
        reason="ok",
    )
    return edge, stats


def build_verified_edges(
    frames: List[FrameData],
    features: Dict[str, TorchFeatures],
    matches: List[PairMatches],
    *,
    min_inliers: int = 30,
    reproj_error_px: float = 4.0,
    max_rot_deg: float = 10.0,
    max_trans_m: float = 0.5,
) -> Tuple[List[VerifiedEdge], List[VerificationStats]]:
    frames_by_id = {frame.frame_id: frame for frame in frames}
    edges: List[VerifiedEdge] = []
    stats: List[VerificationStats] = []
    logger.info("Verifying {} pairs with bidirectional PnP", len(matches))
    for pair in matches:
        frame_i = frames_by_id[pair.frame_id0]
        frame_j = frames_by_id[pair.frame_id1]
        edge, pair_stats = verify_pair_bidirectional(
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
        stats.append(pair_stats)
        if edge is not None:
            edges.append(edge)
    logger.info("Verified {} edges", len(edges))
    return edges, stats


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
    logger.info("MST initialized with {} edges", len(mst))
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
    logger.info("Initialized poses for {} frames", len(poses))
    return poses
