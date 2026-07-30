#!/usr/bin/env python3
"""
Sparse point-track loader for one ViPE results video.

Disk format (ViPE ``sparse_tracks/{name}.zip``): per-frame ``{frame:05d}.npz``
with ``ids`` (N,) and ``uv`` (N, 2). Correspondences are reconstructed by
intersecting keypoint IDs across frames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vipe_io import modality_exists, peek_rgb_size, read_sparse_tracks
from vipe_optical_flow import FlowObject, FlowResult

logger = logging.getLogger(__name__)


class SparseTracks:
    """In-memory sparse tracks for a single ViPE results root."""

    def __init__(
        self,
        base_path: Path | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        self.base_path: Path | None = None
        # frame_idx -> (ids [N], uv [N, 2])
        self._frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.height: int = int(height) if height is not None else 0
        self.width: int = int(width) if width is not None else 0
        if base_path is not None:
            self.set_video(base_path, height=height, width=width)

    @property
    def frames(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return self._frames

    @property
    def num_frames(self) -> int:
        if not self._frames:
            return 0
        return max(self._frames.keys()) + 1

    def set_video(
        self,
        base_path: Path,
        *,
        height: int | None = None,
        width: int | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        """Load sparse track observations from ``base_path/sparse_tracks/``."""
        base_path = Path(base_path)
        self.base_path = base_path
        self._frames = {}

        if height is not None and width is not None:
            self.height, self.width = int(height), int(width)
        else:
            try:
                self.height, self.width = peek_rgb_size(base_path)
            except Exception as exc:
                logger.warning("Could not infer image size from RGB (%s); set H/W explicitly.", exc)
                self.height = self.height or 0
                self.width = self.width or 0

        if not modality_exists(base_path, "sparse_tracks", "*.zip"):
            logger.warning("No sparse_tracks zip under %s; SparseTracks is empty.", base_path)
            return

        for frame_idx, ids, uv in read_sparse_tracks(base_path, start=start, end=end):
            self._frames[int(frame_idx)] = (
                np.asarray(ids, dtype=np.int32),
                np.asarray(uv, dtype=np.float32),
            )

        logger.info(
            "SparseTracks loaded %d frames from %s (H=%d W=%d)",
            len(self._frames),
            base_path,
            self.height,
            self.width,
        )

    def get_frame(self, indices: list[int]) -> list[tuple[np.ndarray, np.ndarray] | None]:
        """Return ``(ids, uv)`` for each frame index (O(1) per index)."""
        out: list[tuple[np.ndarray, np.ndarray] | None] = []
        for idx in indices:
            out.append(self._frames.get(int(idx)))
        return out

    def get_flow(self, src_indices: list[int]) -> FlowResult:
        """
        Build full-res ``FlowResult`` edges starting from each ``src`` in ``src_indices``.

        For each ``src``, pairs with ``dst = src + 1`` via shared keypoint IDs.
        Tensor is ``[H, W, 3]`` with ``(du, dv, 1.0)`` at integer source pixels.
        """
        if self.height <= 0 or self.width <= 0:
            logger.warning("SparseTracks.get_flow: height/width unset; returning empty.")
            return FlowResult(edges=[])

        edges: list[FlowObject] = []
        for src in src_indices:
            src = int(src)
            dst = src + 1
            src_obs = self._frames.get(src)
            dst_obs = self._frames.get(dst)
            if src_obs is None or dst_obs is None:
                continue
            src_ids, src_uv = src_obs
            dst_ids, dst_uv = dst_obs
            if src_ids.size == 0 or dst_ids.size == 0:
                continue

            dst_map = {int(i): uv for i, uv in zip(dst_ids.tolist(), dst_uv)}
            tensor = np.zeros((self.height, self.width, 3), dtype=np.float32)
            for kp_id, uv_s in zip(src_ids.tolist(), src_uv):
                uv_d = dst_map.get(int(kp_id))
                if uv_d is None:
                    continue
                u = int(round(float(uv_s[0])))
                v = int(round(float(uv_s[1])))
                if u < 0 or u >= self.width or v < 0 or v >= self.height:
                    continue
                du = float(uv_d[0]) - float(uv_s[0])
                dv = float(uv_d[1]) - float(uv_s[1])
                tensor[v, u, 0] = du
                tensor[v, u, 1] = dv
                tensor[v, u, 2] = 1.0

            if np.any(tensor[..., 2] > 0):
                edges.append(FlowObject(src=src, dst=dst, tensor=tensor))

        return FlowResult(edges=edges)
