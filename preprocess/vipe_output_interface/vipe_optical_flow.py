#!/usr/bin/env python3
"""
Shared optical-flow types and tensor ops for dense and sparse ViPE flow.

``FlowObject.tensor`` is always full image resolution ``[H, W, 3]`` = ``(u, v, w)``.
Ops take full-res tensors / ``FlowResult`` only — they never import DenseFlow /
SparseTracks and never deal with edge selection or 1/8-res EXRs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FlowObject:
    """One directed optical-flow edge at full image resolution."""

    src: int
    dst: int
    tensor: np.ndarray  # float32 [H, W, 3] = (u, v, w)


@dataclass
class FlowResult:
    """Collection of flow edges (dense, sparse, or merged)."""

    edges: list[FlowObject] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.edges)

    def by_src(self) -> dict[int, FlowObject]:
        """Map ``src -> FlowObject`` (last wins if duplicates)."""
        return {e.src: e for e in self.edges}

    def by_dst(self) -> dict[int, FlowObject]:
        """Map ``src -> FlowObject`` (last wins if duplicates)."""
        return {e.dst: e for e in self.edges}

    def get_edge_indices(self) ->list[tuple()]:
        return [(edge.src, edge.dst) for edge in self.edges ]

    def get_srcs(self):
        return [edge.src for edge in self.edges]

    def get_dsts(self):
        return [edge.dst for edge in self.edges]

    def is_oneshot(self):
        if len(set(self.get_srcs())) ==1:
            return True
        else:
            return False




def flow_arrows_for_src(
    flow_hw3: np.ndarray,
    certainty_thresh: float,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert one full-res flow field into arrow endpoints on the **src** frame.

    Args:
      flow_hw3: float ``[H, W, 3]`` = ``(u, v, certainty)`` in full-image pixels
      certainty_thresh: keep cells with ``w >= certainty_thresh`` and ``w > 0``
      stride: subsample the grid for efficiency (1 = every cell)

    Returns:
      src_xy: float32 ``[N, 2]`` origins ``(x=u, y=v)``
      dst_xy: float32 ``[N, 2]`` landings
      w:      float32 ``[N]`` certainty values
    """
    flow = np.asarray(flow_hw3, dtype=np.float32)
    assert flow.ndim == 3 and flow.shape[-1] == 3, (
        f"Expected flow [H,W,3], got shape {flow.shape}"
    )
    h, w = flow.shape[:2]
    step = max(1, int(stride))
    ii, jj = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing="ij")
    u = flow[ii, jj, 0]
    v = flow[ii, jj, 1]
    cert = flow[ii, jj, 2]
    keep = (
        np.isfinite(u)
        & np.isfinite(v)
        & (cert >= float(certainty_thresh))
        & (((np.pow(u, 2) + np.pow(v, 2) > 1)))
    )
    if not np.any(keep):
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty.copy(), np.zeros((0,), dtype=np.float32)

    src_x = jj[keep].astype(np.float32)
    src_y = ii[keep].astype(np.float32)
    dst_x = src_x + u[keep]
    dst_y = src_y + v[keep]
    src_xy = np.stack([src_x, src_y], axis=-1)
    dst_xy = np.stack([dst_x, dst_y], axis=-1)
    return src_xy, dst_xy, cert[keep].astype(np.float32)


def merge_flow_results(*results: FlowResult) -> FlowResult:
    """Concatenate edges from multiple ``FlowResult``s (order preserved)."""
    edges: list[FlowObject] = []
    for result in results:
        if result is None:
            continue
        edges.extend(result.edges)
    return FlowResult(edges=edges)

