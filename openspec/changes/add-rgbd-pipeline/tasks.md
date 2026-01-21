## 1. Implementation
- [ ] 1.1 Implement dataset loader for `{timestamp}` folders, intrinsics, and depth.
- [ ] 1.2 Add feature extraction wrapper using `lightglue` + PyTorch for left images.
- [ ] 1.3 Implement exhaustive pairwise matching and store candidate matches.
- [ ] 1.4 Add bidirectional PnP (RANSAC) verification and consistency checks.
- [ ] 1.5 Build MST pose initialization from verified edges.
- [ ] 1.6 Build GTSAM robust factor graph and run global BA.
- [ ] 1.7 Concatenate point clouds into a fused output.
- [ ] 1.8 Provide a CLI/config to run the pipeline on a dataset root.
- [ ] 1.9 Add smoke tests on a small subset (<10 frames).
