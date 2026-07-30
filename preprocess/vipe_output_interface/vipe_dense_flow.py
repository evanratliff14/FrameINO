#!/usr/bin/env python3
"""
Dense optical-flow loader for one ViPE results video.

Loads all ``dense_flow/*.zip`` edges into memory. Retrieval via ``get_edges()``
runs ``pick_edges`` (O(n) in the number of stored edges) and returns full-res
``FlowResult`` tensors. Edge selection and 1/8→full-res upsampling live here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vipe_io import iter_flow_edges, modality_exists
from vipe_optical_flow import FlowObject, FlowResult

logger = logging.getLogger(__name__)

# DROID / ViPE dense flow is stored at image_resolution / 8.
FLOW_RES_SCALE = 8


def pick_edges(edges: list[tuple[int, int]], one_shot: bool = True) -> list[tuple[int, int]]:
    """
    Select a subset of ``(src, dst)`` edges for visualization / tensor building.

    ``one_shot=True`` (default): keep edges whose ``src == 0``.
    ``one_shot=False``: keep consecutive unique-src chain edges.
    """
    picked: list[tuple[int, int]] = []
    if not one_shot:
        srcs: list[int] = []
        for src, _dst in edges:
            if not srcs or src > srcs[-1]:
                srcs.append(src)
        l = 0
        for src, dst in edges:
            if l >= len(srcs) - 1:
                break
            if src == srcs[l] and dst == srcs[l + 1]:
                picked.append((src, dst))
                l += 1
    else:
        for src, dst in edges:
            if src == 0:
                picked.append((src, dst))
    return picked


def _upsample_dense_flow_to_fullres(
    flow_hw3: np.ndarray,
    height: int,
    width: int,
    scale: int = FLOW_RES_SCALE,
) -> np.ndarray:
    """
    Scatter a 1/8-res dense edge ``[h, w, 3]`` into full-res ``[H, W, 3]``.

    Low-res cell ``(i, j)`` → pixel ``(scale*i, scale*j)``;
    displacement ``(u, v)`` → ``scale*(u, v)``; certainty ``w`` unchanged.
    """
    flow = np.asarray(flow_hw3, dtype=np.float32)
    assert flow.ndim == 3 and flow.shape[-1] == 3, f"Expected [h,w,3], got {flow.shape}"
    h8, w8 = flow.shape[:2]
    out = np.zeros((int(height), int(width), 3), dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(h8), np.arange(w8), indexing="ij")
    y = ii * int(scale)
    x = jj * int(scale)
    valid = (y < height) & (x < width)
    y, x = y[valid], x[valid]
    ii, jj = ii[valid], jj[valid]
    out[y, x, 0] = flow[ii, jj, 0] * float(scale)
    out[y, x, 1] = flow[ii, jj, 1] * float(scale)
    out[y, x, 2] = flow[ii, jj, 2]
    return out


class DenseFlow:
    """In-memory dense optical flow for a single ViPE results root."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path: Path | None = None
        # Raw 1/8-res edges keyed by (src, dst)
        self._edges: dict[tuple[int, int], np.ndarray] = {}
        self.height: int = 0
        self.width: int = 0
        if base_path is not None:
            self.set_video(base_path)

    @property
    def edges(self) -> dict[tuple[int, int], np.ndarray]:
        """All loaded low-res edges ``(src, dst) -> [h, w, 3]``."""
        return self._edges

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def set_video(self, base_path: Path) -> None:
        """Load every dense-flow EXR edge under ``base_path/dense_flow/``."""
        base_path = Path(base_path)
        self.base_path = base_path
        self._edges = {}

        if not modality_exists(base_path, "dense_flow", "*.zip"):
            logger.warning("No dense_flow zip under %s; DenseFlow is empty.", base_path)
            self.height, self.width = 0, 0
            return

        for src, dst, flow in iter_flow_edges(base_path):
            self._edges[(int(src), int(dst))] = np.asarray(flow, dtype=np.float32)

        # Full-res size from the flow lattice (h8*8, w8*8), not RGB.
        if self._edges:
            sample = next(iter(self._edges.values()))
            h8, w8 = sample.shape[:2]
            self.height = int(h8) * FLOW_RES_SCALE
            self.width = int(w8) * FLOW_RES_SCALE
        else:
            self.height, self.width = 0, 0

        logger.info(
            "DenseFlow loaded %d edges from %s (H=%d W=%d)",
            len(self._edges),
            base_path,
            self.height,
            self.width,
        )

    def get_edges(self, one_shot: bool = True) -> FlowResult:
        """
        Return a ``FlowResult`` for edges selected by ``pick_edges`` (O(n)).

        Each ``FlowObject.tensor`` is full-res ``[H, W, 3]`` — callers need no
        further upsample / raw-edge access for normal use.
        """
        if not self._edges or self.height <= 0 or self.width <= 0:
            return FlowResult(edges=[])

        edge_keys = sorted(self._edges.keys())
        chosen = pick_edges(edge_keys, one_shot=one_shot)
        out: list[FlowObject] = []
        for src, dst in chosen:
            low = self._edges.get((src, dst))
            if low is None:
                continue
            full = _upsample_dense_flow_to_fullres(low, self.height, self.width)
            out.append(FlowObject(src=src, dst=dst, tensor=full))
        return FlowResult(edges=out)
