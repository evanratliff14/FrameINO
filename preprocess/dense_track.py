from preprocess.Open_d4rt.src.core import load_yaml_config
from vis_motion import (timer, read_video_to_tensor, tensor_to_video_rgb, prepare_video_inputs, 
            _ensure_spatracker_path, _build_spatrack_queries, load_d4rt_model)
import argparse 
import sys
from pathlib import Path
from Open_d4rt.vis.build_like_demo import _export_demo_data

from Open_d4rt.vis.build_like_demo import _build_uv_grid, _export_demo_data, _infer_point_cloud_ref0, _sample_rgb_from_uv_sequence, _compute_non_boundary_candidate_mask, _sample_bool_mask_from_uv_sequence, _compute_point_motion_scores
from Open_d4rt.infer_track_3d import _resize_video
import torch
import numpy as np


# python preprocess/3d_visualize.py   --model opend4rt --video_path preprocess/media/fight.mp4  --output_npz tmp/fight_opend4rt.npz  --ckpt_path /home/uft5by/FrameINO/preprocess/Open_d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt   --config preprocess/Open_d4rt/configs/model_effective.yaml   --num_frames 32   --umeyama_slide_window   --sample_step 10 --query_chunk_size 2048 --camera_grid_size 32
CKPT_PATH = "/home/uft5by/FrameINO/preprocess/Open_d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt"
CONFIG = "preprocess/Open_d4rt/configs/model_effective.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type= int, default = 64)
    parser.add_argument("--sample_step", type = int, default = 1)
    parser.add_argument("--query_chunk_size", type = int, default = 2048)
    parser.add_argument("--camera_grid_size", type = int, default = 64)
    parser.add_argument("--video_path", type= str, required=True)
    parser.add_argument("--umeyama_slide_window", action='store_true')

    args = parser.parse_args()

    device = torch.cuda.device("cuda")

    video_rgb = read_video_to_tensor(args.video_path, sample_step= args.sample_step, max_frames = args.num_frames)

    model = load_d4rt_model(config_path = CONFIG, ckpt_path = CKPT_PATH, device = device)

    #TCHW
    h,w = video_rgb.size(2), video_rgb.size(3)

    cfg= load_yaml_config(CONFIG)

    image_size = cfg.get_path("model.input.image_size", [int(video_rgb.shape[1]), int(video_rgb.shape[2])])
    # uses resized for inference and original for translation to wc
    video_model_rgb = _resize_video(video_rgb, image_hw=(int(image_size[0]), int(image_size[1])))

    point_query_uv_px = _build_uv_grid(w, h, cols=64, rows=64, max_points=4096)
    num_points = int(point_query_uv_px.shape[0])
    points_xyz_ref0, points_vis, points_conf, _  = _infer_point_cloud_ref0(
        model = model,
        video_model_rgb=video_model_rgb,
        point_query_uv_norm=point_query_uv_px,
        query_chunk_size=args.query_chunk_size,
        umeyama_slide_window=args.umeyama_slide_windowm

    )

    suppress_depth_boundary_tracks = True
    depth_boundary_rel_thresh = 0.12
    depth_boundary_abs_thresh = 0.20
    depth_boundary_dilate = 1,
    point_dynamic_mask_thw = None
    # attach rgb to the points
    points_rgb = _sample_rgb_from_uv_sequence(video_rgb=video_rgb, uv_px=points_uv_px)
    allowed_track_mask = np.ones((num_points,), dtype=bool)
    if bool(suppress_depth_boundary_tracks):
        allowed_track_mask = _compute_non_boundary_candidate_mask(
            query_uv_px=point_query_uv_px,
            xyz_ref0_frame0=points_xyz_ref0[0],
            visibility_frame0=points_vis[0],
            rel_thresh=float(depth_boundary_rel_thresh),
            abs_thresh=float(depth_boundary_abs_thresh),
            dilate_radius=int(depth_boundary_dilate),
        )
    if point_dynamic_mask_thw is not None:
        point_is_dynamic = _sample_bool_mask_from_uv_sequence(point_dynamic_mask_thw, points_uv_px)
        point_motion_scores = np.zeros((num_points,), dtype=np.float32)
    else:
        point_motion_scores, point_visible_counts = _compute_point_motion_scores(
            xyz_ref0=points_xyz_ref0,
            visibility=points_vis,
            confidence=points_conf,
        )
        dynamic_threshold = float(np.nanpercentile(point_motion_scores, 80)) if np.any(point_motion_scores > 0) else np.inf
        point_is_dynamic = (point_motion_scores >= dynamic_threshold) & (point_visible_counts >= max(2, int(track_min_visible_frames)))

    valid_xyz = np.isfinite(points_xyz_ref0).all(axis=-1) & points_vis
    if np.any(valid_xyz):
        flat = points_xyz_ref0[valid_xyz]
        xyz_min = flat.min(axis=0).astype(np.float32)
        xyz_max = flat.max(axis=0).astype(np.float32)
        xyz_center = ((xyz_min + xyz_max) * 0.5).astype(np.float32)
        xyz_radius = float(np.max(xyz_max - xyz_min) * 0.55)
    else:
        xyz_min = np.zeros((3,), dtype=np.float32)
        xyz_max = np.zeros((3,), dtype=np.float32)
        xyz_center = np.zeros((3,), dtype=np.float32)
        xyz_radius = 1.0
    