## 1. Implementation
- [x] 1.1 Implement dataset loader for `{timestamp}` folders, intrinsics, and depth.
- [x] 1.2 Add feature extraction wrapper using `lightglue` + PyTorch for left images.
- [x] 1.3 Implement exhaustive pairwise matching and store candidate matches.
- [x] 1.4 Add bidirectional PnP (RANSAC) verification and consistency checks.
- [x] 1.5 Build MST pose initialization from verified edges.
- [x] 1.6 Build GTSAM robust factor graph and run global BA.
- [x] 1.7 Concatenate point clouds into a fused output.
- [x] 1.8 Provide a CLI/config to run the pipeline on a dataset root.
- [x] 1.9 Add smoke tests on a small subset (<10 frames).
