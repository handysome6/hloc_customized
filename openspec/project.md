# Project Context

## Purpose
Estimate camera poses and build a fused point cloud from a stereo RGB-D dataset.
Use existing per-frame depth to recover 3D points from left images, then run
exhaustive pairwise matching with geometric verification and global optimization.

## Tech Stack
- Python 3
- uv for Python package management
- NumPy for array math and depth handling
- OpenCV for feature matching, PnP, and RANSAC
- PyTorch for SuperPoint/SuperGlue inference
- SuperPoint + SuperGlue for feature extraction and matching
- GTSAM for robust global optimization (factor graph / BA)
- PLY/NumPy I/O for point cloud concatenation

## Project Conventions

### Code Style
- Keep modules small and pipeline stages explicit (extract, match, verify, optimize).
- Use clear, descriptive names for pose transforms (e.g., T_w_c, T_j_i).
- Favor deterministic runs when possible (seeded RANSAC, fixed thresholds).

### Architecture Patterns
- Pipeline-oriented design: data loader -> features -> matches -> pose estimation -> global BA -> point cloud merge.
- Treat each image pair as an edge in a dense pose graph with robust weighting.
- Use bidirectional PnP checks to filter incorrect matches before optimization.

### Testing Strategy
- Focus on smoke tests for dataset I/O, depth-to-3D projection, and pose estimation.
- Use small, representative subsets of frames for repeatable checks.

### Git Workflow
- Keep commits focused on a single pipeline stage or fix.
- Avoid rewriting history unless explicitly requested.

## Domain Context
- Dataset format:
  - `project_path/{timestamp}/rect_left.jpg`, `rect_right.jpg`
  - `depth_meter.npy` (H, W) float32 depth in meters
  - `cloud.ply` point cloud for concatenation
  - `K.txt` intrinsics (fx 0 cx 0 fy cy 0 0 1) and baseline on line 2
- Depth provides absolute scale; use it to lift 2D keypoints into 3D for PnP.
- Exhaustive pairwise matching is expected; robust filtering is required to reject false matches.

## Important Constraints
- The stereo depth dataset is the primary source of scale and 3D structure.
- Pairwise matching can include incorrect pairs; must pass strict geometric verification.
- The `Hierarchical-Localization` folder is a separate codebase used only for snippet reuse.

## External Dependencies
- SuperPoint/SuperGlue model weights
- GTSAM library
