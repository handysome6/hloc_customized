from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint


@dataclass(frozen=True)
class TorchFeatures:
    keypoints: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor | None
    image_size: Tuple[int, int]


@dataclass(frozen=True)
class MatchResult:
    pairs: np.ndarray
    scores: np.ndarray


def load_image_gray(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    tensor = torch.from_numpy(image).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def _unbatch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim >= 2:
        return value[0]
    return value


class LightGlueExtractor:
    def __init__(self, device: torch.device, max_keypoints: int = 2048) -> None:
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)

    def extract(self, image_path: Path) -> TorchFeatures:
        image = load_image_gray(image_path, self.device)
        with torch.inference_mode():
            feats = self.extractor.extract(image)
        keypoints = _unbatch(feats["keypoints"])
        descriptors = _unbatch(feats["descriptors"])
        scores = _unbatch(feats.get("scores")) if "scores" in feats else None
        height, width = image.shape[-2:]
        return TorchFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            image_size=(width, height),
        )


class LightGlueMatcher:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    def match(self, feats0: TorchFeatures, feats1: TorchFeatures) -> MatchResult:
        inputs: Dict[str, torch.Tensor] = {
            "keypoints0": feats0.keypoints.unsqueeze(0),
            "keypoints1": feats1.keypoints.unsqueeze(0),
            "descriptors0": feats0.descriptors.unsqueeze(0),
            "descriptors1": feats1.descriptors.unsqueeze(0),
        }
        with torch.inference_mode():
            output = self.matcher(inputs)
        matches0 = _unbatch(output["matches0"]).cpu().numpy()
        scores0 = _unbatch(output["scores0"]).cpu().numpy()
        valid = matches0 > -1
        pairs = np.stack([np.nonzero(valid)[0], matches0[valid]], axis=1)
        return MatchResult(pairs=pairs, scores=scores0[valid])
