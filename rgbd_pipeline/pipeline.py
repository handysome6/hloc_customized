from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .dataset import FrameData, load_dataset
from .features import LightGlueExtractor, LightGlueMatcher
from .geometry import Pose
from .matching import PairMatches, exhaustive_match, extract_all_features
from .optimization import optimize_pose_graph
from .pointcloud import fuse_point_clouds
from .pose_graph import build_verified_edges, initialize_poses, maximum_spanning_tree


@dataclass(frozen=True)
class PipelineConfig:
    device: str = "cuda"
    max_keypoints: int = 2048
    min_inliers: int = 30
    reproj_error_px: float = 4.0
    max_rot_deg: float = 10.0
    max_trans_m: float = 0.5
    prior_sigma: float = 1e-3
    between_sigma: float = 1e-2
    robust_kernel: str = "Huber"
    robust_param: float = 1.0


def run_pipeline(
    dataset_root: Path,
    output_cloud: Path,
    *,
    config: PipelineConfig | None = None,
) -> Tuple[Dict[str, Pose], List[PairMatches]]:
    if config is None:
        config = PipelineConfig()

    frames = load_dataset(dataset_root, load_depth=True)
    device = torch.device(config.device)
    extractor = LightGlueExtractor(device=device, max_keypoints=config.max_keypoints)
    matcher = LightGlueMatcher(device=device)

    features = extract_all_features(frames, extractor)
    matches = exhaustive_match(frames, features, matcher)

    edges = build_verified_edges(
        frames,
        features,
        matches,
        min_inliers=config.min_inliers,
        reproj_error_px=config.reproj_error_px,
        max_rot_deg=config.max_rot_deg,
        max_trans_m=config.max_trans_m,
    )
    frame_ids = [frame.frame_id for frame in frames]
    mst = maximum_spanning_tree(frame_ids, edges)
    initial = initialize_poses(frame_ids, mst, root_id=frame_ids[0])

    optimized = optimize_pose_graph(
        initial,
        edges,
        prior_sigma=config.prior_sigma,
        between_sigma=config.between_sigma,
        robust_kernel=config.robust_kernel,
        robust_param=config.robust_param,
    )

    fuse_point_clouds(frames, optimized, output_cloud)
    return optimized, matches
