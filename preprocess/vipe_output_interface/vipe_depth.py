#!/usr/bin/env python3
"""
Depth-map loader for one ViPE results video.

Owns only metric depth maps; poses / RGB / masks live elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vipe_io import read_depth_frames

logger = logging.getLogger(__name__)


class VipeDepth:
    """In-memory depth maps for a single ViPE results root."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path: Path | None = None
        self._depths: list[np.ndarray | None] = []
        self.height: int = 0
        self.width: int = 0
        if base_path is not None:
            self.set_video(base_path)

    @property
    def depths(self) -> list[np.ndarray | None]:
        return self._depths

    @property
    def num_frames(self) -> int:
        return len(self._depths)

    def set_video(self, base_path: Path) -> None:
        """Load all depth frames from ``base_path/depth/``."""
        base_path = Path(base_path)
        self.base_path = base_path
        by_idx: dict[int, np.ndarray] = {}
        self.height, self.width = 0, 0

        for frame_idx, depth in read_depth_frames(base_path):
            arr = np.asarray(depth, dtype=np.float32)
            by_idx[int(frame_idx)] = arr
            if self.height == 0:
                self.height, self.width = int(arr.shape[0]), int(arr.shape[1])

        if not by_idx:
            self._depths = []
            logger.warning("No depth frames under %s", base_path)
            return

        t = max(by_idx.keys()) + 1
        self._depths = [None] * t
        for i, d in by_idx.items():
            if 0 <= i < t:
                self._depths[i] = d

        logger.info(
            "VipeDepth loaded %d/%d frames from %s (H=%d W=%d)",
            sum(1 for d in self._depths if d is not None),
            t,
            base_path,
            self.height,
            self.width,
        )

    def get_depth(self, indices: list[int]) -> list[np.ndarray | None]:
        """Return depth maps for ``indices`` (O(1) per index)."""
        out: list[np.ndarray | None] = []
        n = len(self._depths)
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= n:
                out.append(None)
            else:
                out.append(self._depths[i])
        return out
