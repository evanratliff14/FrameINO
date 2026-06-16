import torch
import decord
from decord import VideoReader, cpu, gpu

# uses Umeyama and SVD to compute camera intrinsics, extrinics
from Open-d4rt.vis import build_like_demo
from Open-d4rt.src.model.d4rt import D4RTModel
import argparse




def read_video_to_tensor(video_path):
    # if torch.cuda.is_available():
    #     vr = VideoReader(video_path, ctx=gpu(0))
    # else:
    # loading on cpu is safer
    vr = VideoReader(video_path, ctx = cpu(0))

    # we only load the necessary frames to avoid OOM
    tensor_frames = vr.get_batch(range(len(vr))) # Returns an NDArray
    
    tensor = torch.from_numpy(tensor_frames.asnumpy())
    
    # Permute from [T, H, W, C] to [T, C, H, W]
    return tensor.permute(0, 3, 1, 2)

def construct_query(args):


# call the forward pass with a query for t: 0 - t_end on a pt

# get the camera extrinsics and the point coords

# function to segment the image for a wanted class (extract this from pre written code) and get uv 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type="str", default =None)

    model = torch.load_state_dict()

    args = parser.parse_args()

    model = D4RTmodel()

    

def _export_demo_data(
    *,
    model: torch.nn.Module,
    video_rgb: np.ndarray,
    video_model_rgb: np.ndarray,
    point_query_uv_px: np.ndarray,
    point_query_chunk_size: int,
    track_query_chunk_size: int,
    track_selection: str,
    track_max_points: int,
    track_min_visible_frames: int,
    track_query_uv_px: np.ndarray | None = None,
    track_query_t_src: np.ndarray | None = None,
    camera_data: dict[str, np.ndarray] | None = None,
    predicted_camera_data: dict[str, np.ndarray] | None = None,
    point_dynamic_mask_thw: np.ndarray | None = None,
    suppress_depth_boundary_tracks: bool = True,
    depth_boundary_rel_thresh: float = 0.12,
    depth_boundary_abs_thresh: float = 0.20,
    depth_boundary_dilate: int = 1,
    umeyama_slide_window: bool = False,
    umeyama_slide_window_dense: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    num_frames = int(video_model_rgb.shape[0])
    clip_frames = _model_clip_frames(model)
    h0, w0 = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    hm, wm = int(video_model_rgb.shape[1]), int(video_model_rgb.shape[2])

    aspect_value = np.asarray([[float(wm) / float(max(1, hm))]], dtype=np.float32)
    aspect_tensor = torch.from_numpy(aspect_value).to(device=device, dtype=torch.float32)
    video_tensor = torch.from_numpy(video_model_rgb).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0

    point_query_uv_norm = point_query_uv_px.copy()
    point_query_uv_norm[:, 0] /= float(max(w0 - 1, 1))
    point_query_uv_norm[:, 1] /= float(max(h0 - 1, 1))

    num_points = int(point_query_uv_px.shape[0])

    points_xyz_ref0 = np.full((num_frames, num_points, 3), np.nan, dtype=np.float32)
    points_vis = np.zeros((num_frames, num_points), dtype=bool)
    points_uv_px = np.tile(point_query_uv_px[None, :, :], (num_frames, 1, 1)).astype(np.float32)
    points_conf = np.full((num_frames, num_points), np.nan, dtype=np.float32)
    points_rgb = np.zeros((num_frames, num_points, 3), dtype=np.uint8)
    points_xyz_ref0, points_vis, points_conf, _ = _infer_point_cloud_ref0(
        model=model,
        video_model_rgb=video_model_rgb,
        point_query_uv_norm=point_query_uv_norm,
        query_chunk_size=point_query_chunk_size,
        umeyama_slide_window=bool(umeyama_slide_window),
    )

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

    if track_selection == "motion":
        if point_dynamic_mask_thw is not None and int(point_dynamic_mask_thw.shape[0]) > 0:
            track_indices = _select_dynamic_interior_track_queries(
                query_uv_px=point_query_uv_px,
                dynamic_mask_hw=point_dynamic_mask_thw[0],
                max_tracks=int(track_max_points),
                allowed_mask=allowed_track_mask,
            )
            if track_indices.size <= 0:
                track_indices = _select_motion_tracks(
                    query_uv_px=point_query_uv_px,
                    xyz_ref0=points_xyz_ref0,
                    visibility=points_vis,
                    confidence=points_conf,
                    max_tracks=int(track_max_points),
                    min_visible_frames=int(track_min_visible_frames),
                    allowed_mask=allowed_track_mask,
                )
        else:
            track_indices = _select_motion_tracks(
                query_uv_px=point_query_uv_px,
                xyz_ref0=points_xyz_ref0,
                visibility=points_vis,
                confidence=points_conf,
                max_tracks=int(track_max_points),
                min_visible_frames=int(track_min_visible_frames),
                allowed_mask=allowed_track_mask,
            )
        track_query_uv_px = point_query_uv_px[track_indices]
    else:
        if track_query_uv_px is None:
            raise ValueError("track_query_uv_px is required when track_selection='grid'.")
        if track_query_t_src is not None:
            track_query_t_src = np.asarray(track_query_t_src, dtype=np.int64).reshape(-1)
            if track_query_t_src.shape[0] != int(track_query_uv_px.shape[0]):
                raise ValueError(
                    f"track_query_t_src must have shape [{int(track_query_uv_px.shape[0])}], got {track_query_t_src.shape}"
                )

    track_query_uv_norm = track_query_uv_px.copy()
    track_query_uv_norm[:, 0] /= float(max(w0 - 1, 1))
    track_query_uv_norm[:, 1] /= float(max(h0 - 1, 1))
    num_tracks = int(track_query_uv_px.shape[0])
    if track_query_t_src is None:
        track_query_t_src = np.zeros((num_tracks,), dtype=np.int64)
    else:
        track_query_t_src = np.asarray(track_query_t_src, dtype=np.int64).reshape(num_tracks)
    track_payload = _infer_tracks(
        model=model,
        video_model_rgb=video_model_rgb,
        query_uv_norm=track_query_uv_norm.astype(np.float32),
        query_chunk_size=track_query_chunk_size,
        query_src_indices_global=track_query_t_src,
        umeyama_slide_window=bool(umeyama_slide_window),
        umeyama_slide_window_dense=bool(umeyama_slide_window_dense),
    )
    tracks_xyz_ref0 = np.asarray(track_payload["tracks_xyz_ref0"], dtype=np.float32)
    tracks_uv_px = np.asarray(track_payload["tracks_uv_norm"], dtype=np.float32)
    tracks_uv_px[..., 0] *= float(max(w0 - 1, 1))
    tracks_uv_px[..., 1] *= float(max(h0 - 1, 1))
    tracks_vis = np.asarray(track_payload["tracks_visibility"], dtype=bool)
    tracks_conf = np.asarray(track_payload["tracks_confidence"], dtype=np.float32)

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

    if camera_data is not None and "K" in camera_data:
        ref0_k = np.asarray(camera_data["K"][0], dtype=np.float32)
    elif predicted_camera_data is not None and "K" in predicted_camera_data:
        ref0_k = np.asarray(predicted_camera_data["K"][0], dtype=np.float32)
    else:
        ref0_k = _estimate_ref0_intrinsics(
            xyz_ref0_frame0=points_xyz_ref0[0],
            uv_px_frame0=point_query_uv_px,
            visibility_frame0=points_vis[0],
            image_width=int(w0),
            image_height=int(h0),
        )

    return {
        "video_width": int(w0),
        "video_height": int(h0),
        "num_frames": int(num_frames),
        "clip_frames": int(clip_frames),
        "track_query_uv_px": track_query_uv_px.astype(np.float32),
        "track_query_t_src": track_query_t_src.astype(np.int64),
        "track_xyz_ref0": tracks_xyz_ref0.astype(np.float32),
        "track_uv_px": tracks_uv_px.astype(np.float32),
        "track_visibility": tracks_vis.astype(np.bool_),
        "track_confidence": tracks_conf.astype(np.float32),
        "track_stitch_diagnostics": track_payload.get("stitch_diagnostics", {}),
        "point_query_uv_px": point_query_uv_px.astype(np.float32),
        "point_xyz_ref0": points_xyz_ref0.astype(np.float32),
        "point_visibility": points_vis.astype(np.bool_),
        "point_uv_px": points_uv_px.astype(np.float32),
        "point_confidence": points_conf.astype(np.float32),
        "point_motion_score": point_motion_scores.astype(np.float32),
        "point_is_dynamic": np.asarray(point_is_dynamic, dtype=np.bool_),
        "point_rgb": points_rgb.astype(np.uint8),
        "bounds_min": xyz_min,
        "bounds_max": xyz_max,
        "bounds_center": xyz_center,
        "bounds_radius": np.asarray([xyz_radius], dtype=np.float32),
        "ref0_K": ref0_k.astype(np.float32),
        "camera_K_seq": None if camera_data is None else np.asarray(camera_data["K"], dtype=np.float32),
        "camera_T_ref0_cam": None if camera_data is None else np.asarray(camera_data["T_ref0_cam"], dtype=np.float32),
        "pred_camera_K_seq": None if predicted_camera_data is None else np.asarray(predicted_camera_data["K"], dtype=np.float32),
        "pred_camera_T_ref0_cam": None if predicted_camera_data is None else np.asarray(predicted_camera_data["T_ref0_cam"], dtype=np.float32),
        "pred_camera_valid_intrinsics": None if predicted_camera_data is None else np.asarray(predicted_camera_data["valid_intrinsics"], dtype=np.bool_),
        "pred_camera_valid_extrinsics": None if predicted_camera_data is None else np.asarray(predicted_camera_data["valid_extrinsics"], dtype=np.bool_),
    }
