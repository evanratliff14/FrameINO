#!/usr/bin/env python3
"""
Dense 4D point cloud export using SpaTrackerV2 VGGT4Track (compatible with offline_view_3d_demo.py).

Uses Eulerian per-frame depth unprojection at a fixed UV grid (same semantics as dense_track.py /
OpenD4RT): at each frame t, grid cell q is lifted from depth at pixel (u_q, v_q) in that frame,
then transformed into ref0. RGB is sampled at the same fixed UV each frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vis_motion import (
    _ensure_spatracker_path,
    _sample_depth_bilinear,
    _scale_uv_to_preprocessed,
    _unproject_uv_depth,
    normalize_c2w_to_ref0,
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


def infer_dense_point_cloud_spatrack2(
    *,
    device: torch.device,
    video_tensor: torch.Tensor,
    point_query_uv_px: np.ndarray,
    video_hw: tuple[int, int],
    front_ckpt: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a dense ref0 point cloud by unprojecting VGGT4Track depth at a fixed UV grid each frame.
    """
    _ensure_spatracker_path()
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    height, width = video_hw
    num_frames = int(video_tensor.shape[0])
    num_points = int(point_query_uv_px.shape[0])

    print(f"Loading SpaTrackV2 front-end from {front_ckpt}")
    vggt4track_model = VGGT4Track.from_pretrained(front_ckpt)
    vggt4track_model.eval().to(device)

    video = video_tensor.float().to(device)
    video_proc = preprocess_image(video)[None]
    proc_h = int(video_proc.shape[3])
    proc_w = int(video_proc.shape[4])

    print("Running VGGT4Track front-end (camera, depth)...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predictions = vggt4track_model(video_proc / 255.0)
            extrinsic = predictions["poses_pred"]
            intrinsic = predictions["intrs"]
            depth_map = predictions["points_map"][..., 2]
            depth_conf = predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().detach().cpu().numpy()
    c2w = extrinsic.squeeze().detach().cpu().numpy().astype(np.float64)
    intrs = intrinsic.squeeze().detach().cpu().numpy().astype(np.float64)
    depth_conf_np = depth_conf.squeeze().detach().cpu().numpy().astype(np.float32)

    if depth_tensor.ndim == 2:
        depth_tensor = depth_tensor[None, ...]
    if depth_conf_np.ndim == 2:
        depth_conf_np = depth_conf_np[None, ...]
    if c2w.ndim == 2:
        c2w = np.tile(c2w[None, ...], (num_frames, 1, 1))
    if intrs.ndim == 2:
        intrs = np.tile(intrs[None, ...], (num_frames, 1, 1))

    num_frames = min(num_frames, int(depth_tensor.shape[0]), int(c2w.shape[0]), int(intrs.shape[0]))
    depth_tensor = depth_tensor[:num_frames]
    depth_conf_np = depth_conf_np[:num_frames]
    c2w = c2w[:num_frames]
    intrs = intrs[:num_frames]

    t_ref0_cam = normalize_c2w_to_ref0(c2w)
    scaled_uv = _scale_uv_to_preprocessed(
        point_query_uv_px,
        orig_hw=(height, width),
        proc_hw=(proc_h, proc_w),
    )

    points_xyz_ref0 = np.full((num_frames, num_points, 3), np.nan, dtype=np.float32)
    points_vis = np.zeros((num_frames, num_points), dtype=bool)
    points_conf = np.full((num_frames, num_points), np.nan, dtype=np.float32)

    print(
        f"Unprojecting VGGT4Track depth for {num_points} grid queries "
        f"over {num_frames} frames (proc resolution {proc_w}x{proc_h})..."
    )
    for t in range(num_frames):
        depth_t = np.asarray(depth_tensor[t], dtype=np.float64)
        conf_t = np.asarray(depth_conf_np[t], dtype=np.float32)
        k = np.asarray(intrs[t], dtype=np.float64)

        depth_vals = _sample_depth_bilinear(depth_t, scaled_uv)
        conf_vals = _sample_depth_bilinear(conf_t, scaled_uv)
        xyz_cam = _unproject_uv_depth(scaled_uv, depth_vals, k)

        rot = t_ref0_cam[t, :3, :3].astype(np.float64)
        trans = t_ref0_cam[t, :3, 3].astype(np.float64)
        xyz_ref0 = (xyz_cam.astype(np.float64) @ rot.T) + trans[None, :]

        valid = (
            (conf_vals > 0.5)
            & np.isfinite(depth_vals)
            & (depth_vals > 1e-4)
            & np.isfinite(xyz_ref0).all(axis=-1)
        )
        points_xyz_ref0[t, valid, :] = xyz_ref0[valid].astype(np.float32)
        points_vis[t, valid] = True
        points_conf[t, valid] = conf_vals[valid].astype(np.float32)

    return points_xyz_ref0, points_vis, points_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense point cloud export via SpaTrackerV2 VGGT4Track depth.")
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
        "--spatrack_front_ckpt",
        type=str,
        default="Yuxihenry/SpatialTrackerV2_Front",
        help="SpaTrackV2 VGGT4Track front-end checkpoint (HF id or local path).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.video_path)
    video_rgb = read_video_to_tensor(path, sample_step=args.sample_step, max_frames=args.num_frames)
    video_rgb_np = tensor_to_video_rgb(video_rgb)
    num_frames = int(video_rgb_np.shape[0])
    h0, w0 = int(video_rgb_np.shape[1]), int(video_rgb_np.shape[2])

    with timer("Building grid of query points"):
        point_query_uv_px = _build_uv_grid(w0, h0, cols=h0, rows=w0, max_points=16384)
        num_points = int(point_query_uv_px.shape[0])

    with timer("Computing point cloud"):
        points_xyz_ref0, points_vis, points_conf = infer_dense_point_cloud_spatrack2(
            device=device,
            video_tensor=video_rgb,
            point_query_uv_px=point_query_uv_px,
            video_hw=(h0, w0),
            front_ckpt=args.spatrack_front_ckpt,
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
