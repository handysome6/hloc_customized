from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-D pose-graph pipeline")
    parser.add_argument("dataset_root", type=Path, help="Path to dataset root")
    parser.add_argument(
        "--output-cloud",
        type=Path,
        default=Path("fused_cloud.ply"),
        help="Output PLY path",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (cuda/cpu)")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--reproj-error-px", type=float, default=4.0)
    parser.add_argument("--max-rot-deg", type=float, default=10.0)
    parser.add_argument("--max-trans-m", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        device=args.device,
        max_keypoints=args.max_keypoints,
        min_inliers=args.min_inliers,
        reproj_error_px=args.reproj_error_px,
        max_rot_deg=args.max_rot_deg,
        max_trans_m=args.max_trans_m,
    )
    run_pipeline(args.dataset_root, args.output_cloud, config=config)


if __name__ == "__main__":
    main()
