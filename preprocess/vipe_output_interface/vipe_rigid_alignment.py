
"""
High-level file for helper functions that executes rigid alignment on top of object types
"""
from __future__ import annotations

from vipe_optical_flow import FlowObject, FlowResult, flow_arrows_for_src, merge_flow_results
from vipe_depth import VipeDepth
from vipe_masks import VipeMasks
import numpy as np
from vipe_camera import Camera
import torch


def kabsch_umeyama(
    P: np.ndarray,
    Q: np.ndarray,
    estimate_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find R, t (and optionally scale s) such that Q ≈ s * R @ P + t.

    P, Q: (n, m) corresponding points. Returns R (m,m), t (m,), s (float).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape
    n, m = P.shape

    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    X = P - p_mean
    Y = Q - q_mean

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(m)
    D[-1, -1] = d
    R = Vt.T @ D @ U.T

    if estimate_scale:
        var_X = (X ** 2).sum() / n
        s = float((S * np.diag(D)).sum() / var_X)
    else:
        s = 1.0
    t = q_mean - s * R @ p_mean
    return R, t, s

def unmask_points(masks: VipeMasks,
            flow: FlowResult, 
            depth: VipeDepth, 
            camera: Camera,
            indices: list[int] | None,
            height: int,
            width: int):

    if indices is None:
        indices = list(range(masks.num_frames))

    masks = masks.get_masks(indices)
    flow = flow.edges
    depth = depth.get_depth(indices)
    # gives us per-frame 1 x 4, but we want in matrix form
    
    # list of tuples?
    correspondences = []
    u = torch.arange(width, dtype=torch.int32)   # (W,)
    v = torch.arange(height, dtype=torch.int32)  # (H,)

    # indexing='xy' -> u varies along columns, v varies along rows
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')  # both (H, W)
    for flow_object in FlowResult.edges:
        src = flow_object.src
        dst = flow_object.dst
        mask_src = masks[src]
        mask_dst = masks[dst]
        flow_object.tensor =  flow_object.tensor * mask_src


        points_with_uv = torch.cat([
            flow_object.tensor,
            u_grid.unsqueeze(-1),  # (H, W, 1)
            v_grid.unsqueeze(-1),  # (H, W, 1)
        ], dim=-1)  # (H, W, 5) -> du, dv, w, u, v

        mask = (points_with_uv[:, :, 0] == 0) & (points_with_uv[:, :, 1] == 0)
        #returns points
        points = points_with_uv[mask]
        dst_points = points.copy()
        dst_points[:,:,3] += dst_points[:, :, 0]
        dst_points[:,:,4] += dst_points[:, :, 1]
        points = points[:, :, 3:4]
        dst_points = dst_points[:, :, 3:4]

        return torch.stack(points, dst_points)

    # TODO: use function to do Camera. i2c, then multiply by the depeth, then Camera.c2w

        



