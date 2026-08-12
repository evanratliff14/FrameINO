"""
High-level file for helper functions that executes rigid alignment on top of object types
"""
from __future__ import annotations

from vipe_optical_flow import FlowObject, FlowResult, flow_arrows_for_src, merge_flow_results
from vipe_depth import VipeDepth
from vipe_masks import InstanceMask
import numpy as np
from vipe_camera import Camera


def kabsch_umeyama(
    P: np.ndarray,
    Q: np.ndarray,
    estimate_scale: bool = False,
    p_mean=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find R, t (and optionally scale s) such that Q ≈ s * R @ P + t.

    P, Q: (n, m) corresponding points. Returns R (m,m), t (m,), s (float).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape
    n, m = P.shape

    if p_mean == None:
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
    return R, t, s, p_mean


def get_points(
    mask: InstanceMask,
    flow: FlowResult,
    indices: list[int] | None,
):
    """
    Function to load all points T, H,W, C=(du, dv, w, u,v) per instance based on the relevant indices of optical flow
    """

    # if all sources are the same, we know that oneshot is True
    oneshot = flow.is_oneshot()

    flow_dict = flow.by_dst() if oneshot else flow.by_src()  # build a dict of index: flowobject

    # T, H, W, C — stack flow edges for absolute frame indices
    flow_mat = np.stack([flow_dict[i].tensor for i in indices], axis=0)
    # instance-wise mask. T, H, W, 1  (full-video [N,H,W] InstanceMask, sliced to indices)
    mask_t = mask.as_float()[np.asarray(indices, dtype=np.int64)][..., None]
    flow_mat = flow_mat * mask_t

    _, height, width, _ = flow_mat.shape
    u = np.arange(width)
    v = np.arange(height)
    # indexing='uv' -> u varies along columns, v varies along rows -  we'll make u and v channel
    v_grid, u_grid = np.meshgrid(v, u, indexing="ij")  # both (H, W)

    # we need explicit u, v
    points_with_uv = np.concatenate(
        [
            flow_mat,
            np.broadcast_to(u_grid[None, ..., None], (*flow_mat.shape[:3], 1)).astype(np.float32),
            np.broadcast_to(v_grid[None, ..., None], (*flow_mat.shape[:3], 1)).astype(np.float32),
        ],
        axis=-1,
    )  # (T, H, W, 5) -> du, dv, w, u, v

    # returns points with flow. if oneshot, then all these points will be at the same u,v
    t0_points = None
    dst_points = None
    if oneshot:
        # copy because its a basic view slice
        t0_points = points_with_uv[0, :, :, :].copy()
        # first frame has no flow
        t0_points[:, :, 0:3] = 0

        dst_points = points_with_uv
        dst_points[:, :, :, 3] += dst_points[:, :, :, 0]
        dst_points[:, :, :, 4] += dst_points[:, :, :, 1]
    else:
        raise NotImplementedError()
    # stack along the T axis
    return np.concatenate([t0_points[None, ...], dst_points], axis=0)


def get_points_with_depth(
    depth: VipeDepth,
    # should be shape T, N, C=5 (du, dv, w, u,v)
    points: np.ndarray,
    indices: list[int],
):
    """
    Take T, N, C, where C has the last two dimensions as (u, v) to T, N, C+1, where the last three dimensions will be (u,v,Z)
    """
    # points: (T, N, C=5) -> channels are du, dv, w, u, v
    depths = depth.get_depth(indices)  # list of [H, W]; stack to (T, H, W)
    depths = np.stack([np.asarray(d, dtype=np.float32) for d in depths], axis=0)

    T, N, C = points.shape

    u = points[:, :, -2]  # (T, N)
    v = points[:, :, -1]  # (T, N)

    # pixel coords should be int for indexing, we round to the nearest pixel
    u_idx = np.round(u).astype(np.int64)
    v_idx = np.round(v).astype(np.int64)

    # build a T index array that matches shape (T, N), so each row t only pulls from depths[t]
    t_idx = np.arange(T)[:, None]  # (T, 1) -> broadcasts to (T, N)

    z = depths[t_idx, v_idx, u_idx]  # (T, N), gathered Z values

    # get z to the same shape with this syntax
    points_with_depth = np.concatenate([points, z[:, :, None]], axis=-1)  # (T, N, 6)

    return points_with_depth


def points_i2c(
    # shape T, N, C=3 (U, V, Z)
    points_i: np.ndarray,
    indices: list[int],
    camera: Camera,
):
    # input homogenous coords (we already have them)
    points_c = camera.i2c(x=points_i, indices=indices)
    return points_c


def points_c2w(
    points_c: np.ndarray,
    indices: list[int],
    camera: Camera,
):
    points_w = camera.c2w(x=points_c, indices=indices)
    return points_w
