#!/usr/bin/env python3
"""Dense 4D point cloud export using SpaTrackerV2 (compatible with offline_view_3d_demo.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vis_motion import (
    _ensure_spatracker_path,
    _scale_uv_to_preprocessed,
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


def _build_spatrack_queries(uv_px: np.ndarray) -> np.ndarray:
    """Pixel UV seeds -> SpaTrack query_xyt [Q, 3] as [frame, u, v]."""
    uv = np.asarray(uv_px, dtype=np.float32)
    frame_idx = np.zeros((uv.shape[0], 1), dtype=np.float32)
    return np.concatenate([frame_idx, uv], axis=1)


def _lift_cam_tracks_to_ref0(
    track3d_pred: np.ndarray,
    t_ref0_cam: np.ndarray,
) -> np.ndarray:
    """Convert [T, N, 3+] camera-space tracks to [T, N, 3] ref0 world coords."""
    xyz_cam = np.asarray(track3d_pred[:, :, :3], dtype=np.float64)
    poses = np.asarray(t_ref0_cam, dtype=np.float64)
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3]
    world = np.einsum("tij,tnj->tni", rot, xyz_cam) + trans[:, None, :]
    return world.astype(np.float32)


def _run_tracker_chunk(
    tracker_model: torch.nn.Module,
    video_for_tracker: torch.Tensor,
    depth_tensor: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    unc_metric: np.ndarray,
    query_xyt: np.ndarray,
    num_frames: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            (
                c2w_traj,
                _intrs_out,
                _point_map,
                _conf_depth,
                track3d_pred,
                _track2d_pred,
                vis_pred,
                conf_pred,
                _video_out,
            ) = tracker_model.forward(
                video_for_tracker,
                depth=depth_tensor,
                intrs=intrs,
                extrs=extrs,
                queries=query_xyt,
                unc_metric=unc_metric,
                fps=1,
                full_point=False,
                iters_track=4,
                query_no_BA=True,
                fixed_cam=False,
                stage=1,
                support_frame=num_frames - 1,
                replace_ratio=0.2,
            )
    return (
        c2w_traj.detach().cpu().numpy(),
        track3d_pred.detach().cpu().numpy(),
        vis_pred.detach().cpu().numpy(),
        conf_pred.detach().cpu().numpy(),
    )


def infer_dense_point_cloud_spatrack2(
    *,
    device: torch.device,
    video_tensor: torch.Tensor,
    point_query_uv_px: np.ndarray,
    video_hw: tuple[int, int],
    front_ckpt: str,
    tracker_ckpt: str,
    query_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_spatracker_path()
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    height, width = video_hw
    num_frames = int(video_tensor.shape[0])
    num_points = int(point_query_uv_px.shape[0])

    print(f"Loading SpaTrackV2 front-end from {front_ckpt}")
    vggt4track_model = VGGT4Track.from_pretrained(front_ckpt)
    vggt4track_model.eval().to(device)

    print(f"Loading SpaTrackV2 tracker from {tracker_ckpt}")
    tracker_model = Predictor.from_pretrained(tracker_ckpt)
    tracker_model.eval().to(device)

    video = video_tensor.float().to(device)
    video_proc = preprocess_image(video)[None]
    video_for_tracker = video_proc.squeeze(0)

    proc_h = int(video_for_tracker.shape[2])
    proc_w = int(video_for_tracker.shape[3])

    print("Running VGGT4Track front-end (camera, depth)...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predictions = vggt4track_model(video_proc / 255.0)
            extrinsic = predictions["poses_pred"]
            intrinsic = predictions["intrs"]
            depth_map = predictions["points_map"][..., 2]
            depth_conf = predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().detach().cpu().numpy()
    extrs = extrinsic.squeeze().detach().cpu().numpy()
    intrs = intrinsic.squeeze().detach().cpu().numpy()
    unc_metric = (depth_conf.squeeze().detach().cpu().numpy() > 0.5).astype(np.float32)

    scaled_uv = _scale_uv_to_preprocessed(
        point_query_uv_px,
        orig_hw=(height, width),
        proc_hw=(proc_h, proc_w),
    )

    chunk_size = max(1, int(query_chunk_size))
    shared_t_ref0_cam: np.ndarray | None = None
    points_xyz_ref0 = np.full((num_frames, num_points, 3), np.nan, dtype=np.float32)
    points_vis = np.zeros((num_frames, num_points), dtype=bool)
    points_conf = np.full((num_frames, num_points), np.nan, dtype=np.float32)

    print(
        f"Running SpaTrackV2 Predictor on {num_points} grid queries "
        f"(chunk_size={chunk_size}, proc resolution {proc_w}x{proc_h})..."
    )

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        query_xyt = _build_spatrack_queries(scaled_uv[start:end])
        c2w, track3d, vis, conf = _run_tracker_chunk(
            tracker_model,
            video_for_tracker,
            depth_tensor,
            intrs,
            extrs,
            unc_metric,
            query_xyt,
            num_frames,
            device,
        )
        if shared_t_ref0_cam is None:
            shared_t_ref0_cam = normalize_c2w_to_ref0(c2w)

        n_chunk = end - start
        points_xyz_ref0[:, start:end, :] = _lift_cam_tracks_to_ref0(
            track3d[:, :n_chunk, :],
            shared_t_ref0_cam,
        )
        vis_arr = np.asarray(vis)
        if vis_arr.ndim == 3:
            vis_arr = vis_arr.squeeze(-1)
        conf_arr = np.asarray(conf)
        if conf_arr.ndim == 3:
            conf_arr = conf_arr.squeeze(-1)
        points_vis[:, start:end] = (vis_arr[:, :n_chunk] > 0.5)
        points_conf[:, start:end] = conf_arr[:, :n_chunk].astype(np.float32)
        print(f"  Tracked queries {start}:{end}")

    if shared_t_ref0_cam is None:
        raise RuntimeError("SpaTrackV2 produced no tracked queries.")

    return points_xyz_ref0, points_vis, points_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense point cloud export via SpaTrackerV2.")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--query_chunk_size", type=int, default=2048)
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
    parser.add_argument(
        "--spatrack_tracker_ckpt",
        type=str,
        default="Yuxihenry/SpatialTrackerV2-Offline",
        help="SpaTrackV2 Predictor checkpoint (HF id or local path).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.video_path)
    video_rgb = read_video_to_tensor(path, sample_step=args.sample_step, max_frames=args.num_frames)
    video_rgb_np = tensor_to_video_rgb(video_rgb)
    num_frames = int(video_rgb_np.shape[0])
    h0, w0 = int(video_rgb_np.shape[1]), int(video_rgb_np.shape[2])

    with timer("Building grid of query points"):
        point_query_uv_px = _build_uv_grid(w0, h0, cols=w0, rows=h0, max_points=16384)
        num_points = int(point_query_uv_px.shape[0])

    with timer("Computing point cloud"):
        points_xyz_ref0, points_vis, points_conf = infer_dense_point_cloud_spatrack2(
            device=device,
            video_tensor=video_rgb,
            point_query_uv_px=point_query_uv_px,
            video_hw=(h0, w0),
            front_ckpt=args.spatrack_front_ckpt,
            tracker_ckpt=args.spatrack_tracker_ckpt,
            query_chunk_size=args.query_chunk_size,
        )

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
