from Open_d4rt.src.core import load_yaml_config
from vis_motion import read_video_to_tensor, tensor_to_video_rgb, load_d4rt_model
import argparse
from pathlib import Path
from vis_motion import timer

from Open_d4rt.vis.build_like_demo import (
    _build_uv_grid,
    _infer_point_cloud_ref0,
    _sample_rgb_from_uv_sequence,
    _compute_non_boundary_candidate_mask,
    _compute_point_motion_scores,
)
from Open_d4rt.infer_track_3d import _resize_video
import torch
import numpy as np

CKPT_PATH = "/home/uft5by/FrameINO/preprocess/Open_d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt"
CONFIG = "preprocess/Open_d4rt/configs/model_effective.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--sample_step", type=int, default=1)
    parser.add_argument("--query_chunk_size", type=int, default=2048)
    parser.add_argument("--camera_grid_size", type=int, default=64)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--umeyama_slide_window", action="store_true")
    parser.add_argument("--track-min-visible-frames", type=int, default=6)
    parser.add_argument("--output_npz", type=str, default=None, help="Output NPZ path (default: tmp/<stem>_point_cloud.npz)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(args.video_path)
    video_rgb = read_video_to_tensor(path, sample_step=args.sample_step, max_frames=args.num_frames)
    video_rgb_np = tensor_to_video_rgb(video_rgb)
    num_frames = int(video_rgb_np.shape[0])
    h0, w0 = int(video_rgb_np.shape[1]), int(video_rgb_np.shape[2])

    model = load_d4rt_model(config_path=CONFIG, ckpt_path=CKPT_PATH, device=device)

    cfg = load_yaml_config(CONFIG)
    image_size = cfg.get_path("model.input.image_size", [h0, w0])
    video_model_rgb = _resize_video(video_rgb_np, image_hw=(int(image_size[0]), int(image_size[1])))

    with timer("Building grid of query points"):
        point_query_uv_px = _build_uv_grid(w0, h0, cols=64, rows=64, max_points=16384)
        num_points = int(point_query_uv_px.shape[0])
        point_query_uv_norm = point_query_uv_px.copy()
        point_query_uv_norm[:, 0] /= float(max(w0 - 1, 1))
        point_query_uv_norm[:, 1] /= float(max(h0 - 1, 1))

    with timer("Computing point cloud"):
        points_xyz_ref0, points_vis, points_conf, _ = _infer_point_cloud_ref0(
            model=model,
            video_model_rgb=video_model_rgb,
            point_query_uv_norm=point_query_uv_norm,
            query_chunk_size=args.query_chunk_size,
            umeyama_slide_window=args.umeyama_slide_window,
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

    output_path = Path(args.output_npz) if args.output_npz else Path.cwd() / Path("tmp") / f"{path.stem}_point_cloud.npz"
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
