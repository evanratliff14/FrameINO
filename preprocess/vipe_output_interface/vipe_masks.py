#!/usr/bin/env python3
"""
Instance-mask loader for one ViPE results video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vipe_io import modality_exists, read_instance_masks, read_instance_phrases

logger = logging.getLogger(__name__)


class VipeMasks:
    """In-memory instance masks for a single ViPE results root."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path: Path | None = None
        self._masks: list[np.ndarray | None] = []
        self.phrases: dict[int, str] = {}
        self.height: int = 0
        self.width: int = 0
        if base_path is not None:
            self.set_video(base_path)

    @property
    def masks(self) -> list[np.ndarray | None]:
        return self._masks

    @property
    def num_frames(self) -> int:
        return len(self._masks)

    def set_video(self, base_path: Path) -> None:
        """Load all instance masks (and phrases if present) from ``base_path/mask/``."""
        base_path = Path(base_path)
        self.base_path = base_path
        self._masks = []
        self.phrases = {}
        self.height, self.width = 0, 0

        if not modality_exists(base_path, "mask", "*.zip"):
            logger.warning("No mask zip under %s; VipeMasks is empty.", base_path)
            return

        by_idx: dict[int, np.ndarray] = {}
        for frame_idx, mask in read_instance_masks(base_path):
            arr = np.asarray(mask, dtype=np.uint8)
            by_idx[int(frame_idx)] = arr
            if self.height == 0:
                self.height, self.width = int(arr.shape[0]), int(arr.shape[1])

        if by_idx:
            t = max(by_idx.keys()) + 1
            self._masks = [None] * t
            for i, m in by_idx.items():
                if 0 <= i < t:
                    self._masks[i] = m

        if modality_exists(base_path, "mask", "*.txt"):
            try:
                self.phrases = read_instance_phrases(base_path)
            except Exception as exc:
                logger.warning("Failed to read instance phrases: %s", exc)
                self.phrases = {}

        logger.info(
            "VipeMasks loaded %d/%d frames, %d phrases from %s",
            sum(1 for m in self._masks if m is not None),
            len(self._masks),
            len(self.phrases),
            base_path,
        )

    def get_masks(self, indices: list[int]) -> list[np.ndarray | None]:
        """Return instance masks for ``indices`` (O(1) per index)."""
        out: list[np.ndarray | None] = []
        n = len(self._masks)
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= n:
                out.append(None)
            else:
                out.append(self._masks[i])
        return out
