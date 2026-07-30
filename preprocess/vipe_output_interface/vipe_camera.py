#!/usr/bin/env python3
"""
Camera poses + intrinsics for one ViPE results video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vipe_io import read_intrinsics, read_pose_c2w

logger = logging.getLogger(__name__)


class Camera:
    """In-memory camera poses and intrinsics for a single ViPE results root."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path: Path | None = None
        self._c2w: np.ndarray = np.zeros((0, 4, 4), dtype=np.float64)
        self._intrinsics: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        self._pose_inds: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._intr_inds: np.ndarray = np.zeros((0,), dtype=np.int64)
        if base_path is not None:
            self.set_video(base_path)

    @property
    def num_frames(self) -> int:
        return int(self._c2w.shape[0])

    @property
    def c2w(self) -> np.ndarray:
        return self._c2w

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics

    def set_video(self, base_path: Path) -> None:
        """Load poses and intrinsics from ``base_path/pose`` and ``base_path/intrinsics``."""
        base_path = Path(base_path)
        self.base_path = base_path
        self._pose_inds, self._c2w = read_pose_c2w(base_path)
        self._intr_inds, self._intrinsics = read_intrinsics(base_path)
        self._c2w = np.asarray(self._c2w, dtype=np.float64)
        self._intrinsics = np.asarray(self._intrinsics, dtype=np.float64)
        if self._c2w.shape[0] != self._intrinsics.shape[0]:
            logger.warning(
                "Pose count (%d) != intrinsics count (%d) under %s",
                self._c2w.shape[0],
                self._intrinsics.shape[0],
                base_path,
            )
        logger.info("Camera loaded %d frames from %s", self.num_frames, base_path)

    def get_poses(self, indices: list[int]) -> list[np.ndarray | None]:
        """Return cam2world ``[4, 4]`` for each index (O(1) per index)."""
        out: list[np.ndarray | None] = []
        n = self.num_frames
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= n:
                out.append(None)
            else:
                out.append(self._c2w[i])
        return out

    def get_intrinsics(self, indices: list[int]) -> list[np.ndarray | None]:
        """Return ``[fx, fy, cx, cy]`` for each index (O(1) per index)."""
        out: list[np.ndarray | None] = []
        n = int(self._intrinsics.shape[0])
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= n:
                out.append(None)
            else:
                out.append(self._intrinsics[i])
        return out
