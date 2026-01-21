from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from loguru import logger

from .geometry import Pose
from .pose_graph import VerifiedEdge


def _pose_to_gtsam(pose: Pose) -> "gtsam.Pose3":
    rotation = pose.matrix[:3, :3]
    translation = pose.matrix[:3, 3]
    rot = gtsam.Rot3(rotation)
    trans = gtsam.Point3(*translation.tolist())
    return gtsam.Pose3(rot, trans)


def _pose_from_gtsam(pose: "gtsam.Pose3") -> Pose:
    matrix = pose.matrix()
    return Pose(matrix)


def optimize_pose_graph(
    initial_poses: Dict[str, Pose],
    edges: Iterable[VerifiedEdge],
    *,
    prior_sigma: float = 1e-3,
    between_sigma: float = 1e-2,
    robust_kernel: str = "Huber",
    robust_param: float = 1.0,
) -> Dict[str, Pose]:
    edges_list = list(edges)
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    key_lookup: Dict[str, int] = {}
    for idx, frame_id in enumerate(initial_poses.keys()):
        key_lookup[frame_id] = idx
        values.insert(idx, _pose_to_gtsam(initial_poses[frame_id]))

    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, prior_sigma)
    root_id = next(iter(initial_poses.keys()))
    graph.add(
        gtsam.PriorFactorPose3(
            key_lookup[root_id], _pose_to_gtsam(initial_poses[root_id]), prior_noise
        )
    )

    base_noise = gtsam.noiseModel.Isotropic.Sigma(6, between_sigma)
    if robust_kernel.lower() == "huber":
        kernel = gtsam.noiseModel.mEstimator.Huber(robust_param)
    elif robust_kernel.lower() == "cauchy":
        kernel = gtsam.noiseModel.mEstimator.Cauchy(robust_param)
    else:
        kernel = gtsam.noiseModel.mEstimator.Tukey(robust_param)
    robust = gtsam.noiseModel.Robust(kernel, base_noise)

    for edge in edges_list:
        pose_1_0 = _pose_to_gtsam(edge.pose_1_0)
        graph.add(
            gtsam.BetweenFactorPose3(
                key_lookup[edge.frame_id0],
                key_lookup[edge.frame_id1],
                pose_1_0,
                robust,
            )
        )

    logger.info(
        "Optimizing pose graph with {} poses and {} edges (kernel={})",
        len(initial_poses),
        len(edges_list),
        robust_kernel,
    )
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("ERROR")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    optimized: Dict[str, Pose] = {}
    for frame_id, key in key_lookup.items():
        optimized[frame_id] = _pose_from_gtsam(result.atPose3(key))
    logger.info("Optimization complete")
    return optimized


try:
    import gtsam  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("GTSAM is required for pose graph optimization") from exc
