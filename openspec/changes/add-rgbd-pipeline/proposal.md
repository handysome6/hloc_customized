# Change: Add RGB-D pose-graph pipeline MVP

## Why
We need a working end-to-end pipeline that uses existing depth to estimate camera poses
and fuse point clouds for a small (<50 frame) stereo dataset.

## What Changes
- Add a pipeline that loads the dataset format and extracts features from left images.
- Perform exhaustive pairwise matching with LightGlue and strict geometric verification.
- Initialize poses via maximum spanning tree and refine with GTSAM robust optimization.
- Concatenate per-frame point clouds into a fused output.
- Use the `lightglue` Python package (PyTorch) rather than importing the HL repo as a module.

## Impact
- Affected specs: `specs/rgbd-pose-graph/spec.md` (new)
- Affected code: new pipeline modules, dataset I/O, optimization, and a CLI entry point
