#!/usr/bin/env python3
"""Dense 4D point cloud export using ViPE depth + camera poses (compatible with offline_view_3d_demo.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vis_motion import (
    infer_dense_point_cloud_vipe,
    read_video_to_tensor,
    tensor_to_video_rgb,
    timer,
)
from Open_d4rt.vis.build_like_demo import (
    _build_uv_grid,
    _compute_non_boundary_candidate_mask,
    _compute_point_motion_scores,
    _sample_rgb_from_uv_sequence,
)


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
