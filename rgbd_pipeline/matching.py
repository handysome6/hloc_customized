from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from loguru import logger

from .dataset import FrameData
from .features import LightGlueExtractor, LightGlueMatcher, MatchResult, TorchFeatures


@dataclass(frozen=True)
class PairMatches:
    frame_id0: str
    frame_id1: str
    matches: MatchResult


def extract_all_features(
    frames: List[FrameData],
    extractor: LightGlueExtractor,
) -> Dict[str, TorchFeatures]:
    features: Dict[str, TorchFeatures] = {}
    logger.info("Extracting features for {} frames", len(frames))
    for frame in frames:
        features[frame.frame_id] = extractor.extract(frame.left_image_path)
    logger.info("Feature extraction complete")
    return features


def exhaustive_match(
    frames: List[FrameData],
    features: Dict[str, TorchFeatures],
    matcher: LightGlueMatcher,
) -> List[PairMatches]:
    results: List[PairMatches] = []
    n = len(frames)
    total_pairs = n * (n - 1) // 2
    logger.info("Running exhaustive matching for {} pairs", total_pairs)
    pair_index = 0
    for i in range(n):
        for j in range(i + 1, n):
            frame_i = frames[i]
            frame_j = frames[j]
            matches = matcher.match(features[frame_i.frame_id], features[frame_j.frame_id])
            results.append(
                PairMatches(
                    frame_id0=frame_i.frame_id,
                    frame_id1=frame_j.frame_id,
                    matches=matches,
                )
            )
            pair_index += 1
            if pair_index % 100 == 0 or pair_index == total_pairs:
                logger.info("Matched {}/{} pairs", pair_index, total_pairs)
    logger.info("Exhaustive matching complete: {} pairs", len(results))
    return results
