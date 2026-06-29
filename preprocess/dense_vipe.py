#!/usr/bin/env python3
"""Dense 4D point cloud export using ViPE depth + camera poses (compatible with offline_view_3d_demo.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vis_motion import (
    infer_dense_point_cloud_vipe,
    read_video_to_tensor,
    tensor_to_video_rgb,
    timer
)
def _grid_query_points(
    width: int,
    height: int,
    cols: int,
    rows: int,
    margin_ratio: float,
    max_points: int,
) -> np.ndarray:
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    margin_x = float(max(width - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    margin_y = float(max(height - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    xs = np.linspace(margin_x, float(max(width - 1, 0)) - margin_x, num=cols, dtype=np.float32)
    ys = np.linspace(margin_y, float(max(height - 1, 0)) - margin_y, num=rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if grid.shape[0] > max_points:
        pick = np.linspace(0, grid.shape[0] - 1, num=max_points, dtype=np.int64)
        grid = grid[pick]
    return grid.astype(np.float32)


def _build_uv_grid(width: int, height: int, cols: int, rows: int, max_points: int) -> np.ndarray:
    pts = _grid_query_points(
        width=int(width),
        height=int(height),
        cols=int(cols),
        rows=int(rows),
        margin_ratio=0.02,
        max_points=int(max_points),
    )
    return pts.astype(np.float32)


def _sample_rgb_from_uv_sequence(video_rgb: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    video = np.asarray(video_rgb, dtype=np.uint8)
    uv = np.asarray(uv_px, dtype=np.float32)
    t = int(video.shape[0])
    n = int(uv.shape[1])
    rgb = np.zeros((t, n, 3), dtype=np.uint8)
    for ti in range(t):
        for qi in range(n):
            x = float(uv[ti, qi, 0])
            y = float(uv[ti, qi, 1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xi = int(np.clip(np.rint(x), 0, max(video.shape[2] - 1, 0)))
            yi = int(np.clip(np.rint(y), 0, max(video.shape[1] - 1, 0)))
            rgb[ti, qi] = video[ti, yi, xi]
    return rgb


def _infer_regular_grid_shape(query_uv_px: np.ndarray) -> tuple[int, int] | None:
    pts = np.asarray(query_uv_px, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] <= 0:
        return None
    xs = np.unique(np.round(pts[:, 0], decimals=4))
    ys = np.unique(np.round(pts[:, 1], decimals=4))
    if int(xs.size) * int(ys.size) != int(pts.shape[0]):
        return None
    return int(ys.size), int(xs.size)


def _compute_non_boundary_candidate_mask(
    *,
    query_uv_px: np.ndarray,
    xyz_ref0_frame0: np.ndarray,
    visibility_frame0: np.ndarray,
    rel_thresh: float,
    abs_thresh: float,
    dilate_radius: int,
) -> np.ndarray:
    num_points = int(query_uv_px.shape[0])
    keep = np.ones((num_points,), dtype=bool)
    grid_shape = _infer_regular_grid_shape(query_uv_px)
    if grid_shape is None:
        return keep

    rows, cols = grid_shape
    xyz = np.asarray(xyz_ref0_frame0, dtype=np.float32).reshape(rows, cols, 3)
    vis = np.asarray(visibility_frame0, dtype=bool).reshape(rows, cols)
    finite = np.isfinite(xyz).all(axis=-1)
    valid = vis & finite
    z = xyz[..., 2]

    boundary = np.zeros((rows, cols), dtype=bool)

    def _mark(
        diff: np.ndarray,
        z_ref: np.ndarray,
        vmask: np.ndarray,
        sl_a: tuple[slice, slice],
        sl_b: tuple[slice, slice],
    ) -> None:
        thresh = np.maximum(float(abs_thresh), float(rel_thresh) * np.maximum(np.abs(z_ref), 1e-6))
        edge = vmask & np.isfinite(diff) & (diff > thresh)
        boundary[sl_a] |= edge
        boundary[sl_b] |= edge

    if cols > 1:
        diff_x = np.abs(z[:, 1:] - z[:, :-1])
        vmask_x = valid[:, 1:] & valid[:, :-1]
        z_ref_x = np.minimum(np.abs(z[:, 1:]), np.abs(z[:, :-1]))
        _mark(diff_x, z_ref_x, vmask_x, (slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    if rows > 1:
        diff_y = np.abs(z[1:, :] - z[:-1, :])
        vmask_y = valid[1:, :] & valid[:-1, :]
        z_ref_y = np.minimum(np.abs(z[1:, :]), np.abs(z[:-1, :]))
        _mark(diff_y, z_ref_y, vmask_y, (slice(1, None), slice(None)), (slice(None, -1), slice(None)))

    if int(dilate_radius) > 0:
        k = int(2 * int(dilate_radius) + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1) > 0

    keep = (~boundary).reshape(-1)
    keep &= valid.reshape(-1)
    return keep


def _compute_point_motion_scores(
    *,
    xyz_ref0: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_ref0, dtype=np.float32)
    vis = np.asarray(visibility, dtype=bool)
    conf = np.asarray(confidence, dtype=np.float32)
    num_frames, num_points = xyz.shape[:2]
    finite = np.isfinite(xyz).all(axis=-1)
    valid = vis & finite
    motion_scores = np.zeros((num_points,), dtype=np.float32)
    visible_counts = valid.sum(axis=0).astype(np.int32)

    for qi in range(num_points):
        valid_idx = np.flatnonzero(valid[:, qi])
        if valid_idx.size < 2:
            continue
        pts = xyz[valid_idx, qi]
        ref = pts[0]
        displacement = np.linalg.norm(pts - ref[None, :], axis=-1)
        smooth_motion = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
        motion_scores[qi] = (
            float(np.nanpercentile(displacement, 90))
            + 0.35 * float(np.nanpercentile(smooth_motion, 75))
            + 0.02 * float(np.nanmean(conf[valid_idx, qi]))
        )

    return motion_scores, visible_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense point cloud export via ViPE depth unprojection.")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--track-min-visible-frames", type=int, default=6)
    parser.add_argument(
        "--output_npz",
        type=str,
        default=None,
        help="Output NPZ path (default: tmp/<stem>_point_cloud.npz)",
    )
    parser.add_argument(
        "--vipe_pipeline",
        type=str,
        default="no_vda",
        help="ViPE Hydra pipeline preset (default: no_vda).",
    )
    args = parser.parse_args()

    path = Path(args.video_path)
    video_rgb = read_video_to_tensor(path, sample_step=args.sample_step, max_frames=args.num_frames)
    video_rgb_np = tensor_to_video_rgb(video_rgb)
    num_frames = int(video_rgb_np.shape[0])
    h0, w0 = int(video_rgb_np.shape[1]), int(video_rgb_np.shape[2])

    with timer("Building grid of query points"):
        point_query_uv_px = _build_uv_grid(w0, h0, cols=w0, rows=h0, max_points=16384)
        num_points = int(point_query_uv_px.shape[0])

    with timer("Computing point cloud"):
        points_xyz_ref0, points_vis, points_conf = infer_dense_point_cloud_vipe(
            video_path=path,
            point_query_uv_px=point_query_uv_px,
            num_frames=num_frames,
            sample_step=int(args.sample_step),
            pipeline=str(args.vipe_pipeline),
        )

    num_frames = int(points_xyz_ref0.shape[0])
    video_rgb_np = video_rgb_np[:num_frames]

    suppress_depth_boundary_tracks = True
    depth_boundary_rel_thresh = 0.12
    depth_boundary_abs_thresh = 0.20
    depth_boundary_dilate = 1

    points_uv_px = np.tile(point_query_uv_px[None, :, :], (num_frames, 1, 1)).astype(np.float32)
    points_rgb = _sample_rgb_from_uv_sequence(video_rgb=video_rgb_np, uv_px=points_uv_px)
    allowed_track_mask = np.ones((num_points,), dtype=bool)
    if bool(suppress_depth_boundary_tracks):
        with timer("Computing the motion tracking mask"):
            allowed_track_mask = _compute_non_boundary_candidate_mask(
                query_uv_px=point_query_uv_px,
                xyz_ref0_frame0=points_xyz_ref0[0],
                visibility_frame0=points_vis[0],
                rel_thresh=float(depth_boundary_rel_thresh),
                abs_thresh=float(depth_boundary_abs_thresh),
                dilate_radius=int(depth_boundary_dilate),
            )

    point_motion_scores, point_visible_counts = _compute_point_motion_scores(
        xyz_ref0=points_xyz_ref0,
        visibility=points_vis,
        confidence=points_conf,
    )
    dynamic_threshold = (
        float(np.nanpercentile(point_motion_scores, 80)) if np.any(point_motion_scores > 0) else np.inf
    )
    track_min_visible_frames = int(args.track_min_visible_frames)
    point_is_dynamic = (point_motion_scores >= dynamic_threshold) & (
        point_visible_counts >= max(2, track_min_visible_frames)
    )

    valid_xyz = np.isfinite(points_xyz_ref0).all(axis=-1) & points_vis
    if np.any(valid_xyz):
        flat = points_xyz_ref0[valid_xyz]
        xyz_min = flat.min(axis=0).astype(np.float32)
        xyz_max = flat.max(axis=0).astype(np.float32)
        xyz_center = ((xyz_min + xyz_max) * 0.5).astype(np.float32)
        xyz_radius = float(np.max(xyz_max - xyz_min) * 0.55)
    else:
        raise RuntimeError("No valid points!")

    output_path = (
        Path(args.output_npz)
        if args.output_npz
        else Path.cwd() / Path("tmp") / f"{path.stem}_point_cloud.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points_xyz_ref0=points_xyz_ref0.astype(np.float32),
        points_vis=points_vis.astype(np.bool_),
        points_conf=points_conf.astype(np.float32),
        points_rgb=points_rgb.astype(np.uint8),
        allowed_track_mask=allowed_track_mask.astype(np.bool_),
        point_is_dynamic=point_is_dynamic.astype(np.bool_),
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        xyz_center=xyz_center,
        xyz_radius=np.asarray([xyz_radius], dtype=np.float32),
        coordinate_convention=np.asarray("opencv_ref0"),
    )
    print(f"Saved point cloud to {output_path}")
