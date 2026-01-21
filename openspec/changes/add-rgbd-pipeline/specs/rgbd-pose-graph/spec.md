## ADDED Requirements
### Requirement: Dataset ingestion
The system SHALL load frames from `{timestamp}` directories and parse intrinsics and
baseline from `K.txt`, plus `rect_left.jpg`, `rect_right.jpg`, and `depth_meter.npy`.

#### Scenario: Valid frame loads
- **WHEN** a frame directory contains the expected files
- **THEN** the loader returns intrinsics, baseline, depth, and image paths for that frame

### Requirement: Feature extraction and exhaustive matching
The system SHALL extract features from left images using the `lightglue` package and
perform exhaustive pairwise matching across all frame pairs.

#### Scenario: Exhaustive matching produces candidate pairs
- **WHEN** N frames are loaded
- **THEN** the system generates matches for all pairs `(i, j)` where `0 <= i < j < N`

### Requirement: Geometric verification with bidirectional PnP
The system SHALL validate matches with bidirectional PnP (RANSAC) using depth-derived
3D points and reject pairs that fail consistency or minimum inlier thresholds.

#### Scenario: Invalid match is rejected
- **WHEN** the forward and reverse PnP transforms are inconsistent or inliers are below threshold
- **THEN** the pair is excluded from the pose graph

### Requirement: Robust pose-graph optimization and outputs
The system SHALL initialize poses via a maximum spanning tree and refine them with a
GTSAM robust factor graph, then output optimized poses and a fused point cloud.

#### Scenario: Successful optimization
- **WHEN** verified edges exist for the dataset
- **THEN** optimized poses and a concatenated point cloud are produced
