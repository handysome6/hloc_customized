from __future__ import annotations

from pathlib import Path

import numpy as np

from rgbd_pipeline.dataset import load_dataset


def _write_frame(root: Path, frame_id: str) -> None:
    frame_dir = root / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / "rect_left.jpg").write_bytes(b"")
    (frame_dir / "rect_right.jpg").write_bytes(b"")
    np.save(frame_dir / "depth_meter.npy", np.ones((2, 2), dtype=np.float32))
    (frame_dir / "cloud.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n",
        encoding="utf-8",
    )
    (frame_dir / "K.txt").write_text(
        "100 0 1 0 100 1 0 0 1\n0.1\n", encoding="utf-8"
    )


def test_dataset_loader_smoke(tmp_path: Path) -> None:
    _write_frame(tmp_path, "0001")
    _write_frame(tmp_path, "0002")

    frames = load_dataset(tmp_path, load_depth=True)

    assert len(frames) == 2
    assert frames[0].baseline_m == 0.1
    assert frames[0].intrinsics.shape == (3, 3)
    assert frames[0].depth_m is not None
