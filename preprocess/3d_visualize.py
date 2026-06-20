#!/usr/bin/env python3
"""
3D Camera + Identity Trajectory Demo.

For a single input video, estimates:
  1. Camera world position (x, y, z) over time in the ref0 frame.
  2. Identity world centroid over time via frame-0 panoptic queries + 3D tracking.

Backends (--model):
  - opend4rt: OpenD4RT camera branches + 3D track head
  - spatrackerv2: SpaTrackV2 VGGT4Track front-end + Predictor joint tracking

Coordinate convention (ref0_opencv_t0_identity):
  - ref0 is the world frame anchored at frame 0.
  - T_ref0_cam[t] is a 4x4 camera-to-world transform; camera position is T_ref0_cam[t, :3, 3].
  - tracks_xyz_ref0[q, t] are 3D points in the same ref0 world frame.

Scale is model-relative (no metric GT); trajectories show relative motion structure.
Export NPZ with --output_npz, then view locally via offline_view_3d_demo.py.

python preprocess/3d_visualize.py --model opend4rt --video_path preprocess/1917.mp4
  --ckpt_path preprocess/Open_d4rt/checkpoints/.../opend4rt.ckpt --output_npz tmp/trajectories.npz

python preprocess/3d_visualize.py --model spatrackerv2 --video_path preprocess/media/1917.mp4
  --output_npz preprocess/tmp/spatrack_trajectories.npz

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from sklearn.cluster import KMeans
import cv2
import time
from contextlib import contextmanager
import json

@contextmanager
def timer(block_name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{block_name}] Execution time: {end - start:.4f} seconds")

# ---------------------------------------------------------------------------
# Path setup: OpenD4RT (nested repo) + FrameINO root (for OneFormer helpers).
# ---------------------------------------------------------------------------
PREPROCESS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PREPROCESS_ROOT.parent
OPEN_D4RT_ROOT = PREPROCESS_ROOT / "Open_d4rt"
SPATRACKER_ROOT = PREPROCESS_ROOT / "SpaTrackerV2"

MOTIONABLE_OBJECT = [
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                        'sports ball', 'kite', 'flower', 
                        # We delete:
                        # Belows are newly added cases
                        'snowboard', 'surfboard', 'skateboard',
                    ]

# NOTE: We want to make it simpler for the object motion CTRL case, so neglect some that may be useful in the Camera CTRL
REFERENCE_OBJECT_CLASS = [
                            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                            'bird', 'cat', 'dog', 
                            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
                            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
                            'pizza', 'donut', 'cake', 'chair', 'dining table', 'laptop', 'mouse', 'remote', 
                            'keyboard', 'cell phone', 'book', 'clock', 
                            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'blanket', 'cardboard', 'counter',
                            'flower', 'fruit', 'pillow', 'towel', 'food-other-merged', 'door-stuff',
                        ]

NON_OBJECT_CLASS = [
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'tv', 'potted plant', 'couch', 'parking meter', 'fire hydrant', 'stop sign',
                        'toilet', 'banner', 'net', 'platform', 'road', 'snow', 'sea', 'railroad', 'roof', 'traffic light', 'bench', 
                        'floor-wood', 'gravel', 'light', 'playingfield', 'mountain-merged', 'water-other', 'wall-brick', 'wall-stone', 
                        'wall-tile', 'rock-merged', 'mirror-stuff', 'sand', 'bed', 'bridge', 'stairs', 'house', 'vase', 'curtain',
                        'grass-merged', 'dirt-merged', 'paper-merged', 'window-blind', 'building-other-merged',  'shelf', 'tent',
                        'wall-other-merged', 'rug-merged', 'river', 'window-other', 'fence-merged', 'ceiling-merged', 'tree-merged', 
                        'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'wall-wood', 
                    ]


for path in (OPEN_D4RT_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from infer_track_3d import (  # noqa: E402
    _infer_tracks,
    _resize_video,
    _resolve_device,
    _unwrap_state_dict,
)
from src.core import load_checkpoint, load_yaml_config, seed_everything  # noqa: E402
from src.model import build_model  # noqa: E402
from vis.build_like_demo import _predict_camera_branches  # noqa: E402


# =============================================================================
# Video I/O
# =============================================================================


def read_video_to_tensor(video_path: str | Path, sample_step: int = 10, max_frames: int | None = None) -> torch.Tensor:
    """
    Load video frames as a float-ready tensor [T, C, H, W] in RGB order.

    Decord reads on CPU for stability; cap frames with max_frames to limit memory.
    Uniformly sample every sample_step frames
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_frames = len(vr)
    if max_frames is not None and int(max_frames) > 0:
        num_frames = min(num_frames, int(max_frames))
    # load frames efficiently
    tensor_frames = vr.get_batch(range(0, num_frames*sample_step, sample_step))
    tensor = torch.from_numpy(tensor_frames.asnumpy())
    return tensor.permute(0, 3, 1, 2)


def tensor_to_video_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert [T, C, H, W] (or [T, H, W, C]) tensor to uint8 RGB [T, H, W, 3]."""
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 + 1e-3:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def prepare_video_inputs(
    video_rgb: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep original-resolution RGB and build model-input resized copy."""
    video_rgb = np.asarray(video_rgb, dtype=np.uint8)
    video_model_rgb = _resize_video(video_rgb, image_hw=image_hw)
    return video_rgb, video_model_rgb


# =============================================================================
# Model loading
# =============================================================================


def _disable_encoder_pretrain(cfg: Any) -> None:
    """Skip VideoMAE weight fetch at inference time (checkpoint supplies weights)."""
    encoder_cfg = cfg.get_path("model.encoder", {})
    if isinstance(encoder_cfg, dict):
        pretrained_cfg = encoder_cfg.setdefault("pretrained", {})
        if isinstance(pretrained_cfg, dict):
            pretrained_cfg["enabled"] = False


def load_d4rt_model(
    config_path: str | Path,
    ckpt_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Build D4RT from yaml config and load checkpoint weights."""
    cfg = load_yaml_config(config_path)
    _disable_encoder_pretrain(cfg)
    seed_everything(int(cfg.get_path("experiment.seed", 42)), deterministic=True)

    model = build_model(cfg["model"]).eval().to(device)
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# =============================================================================
# OneFormer identity UV sampling (frame 0)
# =============================================================================

_ONEFORMER_READY = False
_ONEFORMER_DATASET = "COCO (133 classes)"
_ONEFORMER_BACKBONE = "Swin-L"


def _ensure_oneformer() -> None:
    """Lazy-load OneFormer panoptic model (requires `conda activate oneformer`)."""
    global _ONEFORMER_READY
    if _ONEFORMER_READY:
        return
    from preprocess.filter_panoptic_multi import setup_modules

    setup_modules()
    _ONEFORMER_READY = True


def sample_identity_uv_queries(
    frame0_rgb: np.ndarray,
    *,
    num_queries: int = 48,
    min_mask_area_ratio: float = 0.005,
    max_mask_area_ratio: float = 0.85,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Segment frame 0 with OneFormer, pick the largest instance of `class_name`,
    and return UV query points (pixel coords) via K-means on mask pixels.

    Returns:
        uv_px: [N, 2] float32 pixel coordinates (u=x, v=y).
        meta:  diagnostic info (class name, area ratio, num raw mask pixels).
    """
    from preprocess.filter_panoptic_multi import segment

    _ensure_oneformer()
    frame = np.asarray(frame0_rgb, dtype=np.uint8)
    # array_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # output_path = Path.cwd() / "frame.png"

    # cv2.imwrite(str(output_path), array_bgr)


    height, width = int(frame.shape[0]), int(frame.shape[1])

    with timer("segment_model"):
        panoptic_seg, segments_info, metadata = segment(
            frame, _ONEFORMER_DATASET, _ONEFORMER_BACKBONE, debug=False
        )
    height_pan, width_pan = panoptic_seg.shape

    best_area = 0.0
    best_mask = None
    best_label = None
    for segment_info in segments_info:
        category_id = int(segment_info["category_id"])
        text_name = metadata.stuff_classes[category_id]
        if text_name not in MOTIONABLE_OBJECT:
            continue
        seg_id = int(segment_info["id"])
        mask = (panoptic_seg == seg_id).cpu().numpy()
        area_ratio = float(mask.sum()) / float(max(height_pan * width_pan, 1))
        if area_ratio < min_mask_area_ratio or area_ratio > max_mask_area_ratio:
            continue
        if area_ratio > best_area:
            best_area = area_ratio
            best_mask = mask
            best_label = text_name

    if best_mask is None:
        raise RuntimeError(
            f"No motionable object instance found in frame 0. "
            "Try another video or class, or run with `conda activate oneformer`."
        )

    # Collect mask pixel coordinates (tuples) in panoptic resolution, subsample for K-means.
    ys, xs = np.where(best_mask)
    if ys.size == 0:
        raise RuntimeError(f"Empty mask for class in frame 0.")

    rng = np.random.default_rng(0)
    max_pool = min(ys.size, 5000)
    if ys.size > max_pool:
        pick = rng.choice(ys.size, size=max_pool, replace=False)
        # get same indices of ys, xs, equivalent to choosing multiple points from mask
        ys, xs = ys[pick], xs[pick]

    points_pan = np.stack([ys, xs], axis=1).astype(np.float64)
    n_clusters = max(1, min(int(num_queries), points_pan.shape[0])) # we limit to n_queries and then enforce that it should be >1
    # we cluster to try to get an evenly spread distribution of points without the overhead of law of large num
    centers = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(points_pan).cluster_centers_
    # round raw centroid to nearest pixel
    centers = np.rint(centers).astype(np.int64)

    uv_px: list[list[float]] = []
    for cord_y, cord_x in centers:
        # we enforce that each center must be a trackable point
        if not best_mask[cord_y, cord_x]:
            continue
        # Map panoptic coords back to original frame resolution.
        y_orig = int(np.clip(cord_y * height / max(height_pan, 1), 0, height - 1))
        x_orig = int(np.clip(cord_x * width / max(width_pan, 1), 0, width - 1))
        uv_px.append([float(x_orig), float(y_orig)])

    if len(uv_px) == 0:
        raise RuntimeError(f"K-means produced no in-mask query points for {best_label}.")

    out = np.asarray(uv_px, dtype=np.float32)
    meta = {
        "class_name": best_label,
        "mask_area_ratio": best_area,
        "num_queries": int(out.shape[0]),
    }
    return out, meta


def uv_px_to_norm(uv_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel UV -> normalized [0, 1] per OpenD4RT convention."""
    uv_norm = np.asarray(uv_px, dtype=np.float32).copy()
    uv_norm[:, 0] /= float(max(width - 1, 1))
    uv_norm[:, 1] /= float(max(height - 1, 1))
    return uv_norm


# =============================================================================
# Canonical trajectory bundle (NPZ-compatible for offline_view_3d_demo.py)
# =============================================================================

COORDINATE_CONVENTION = "ref0_opencv_t0_identity"


@dataclass
class TrajectoryResult:
    """Self-contained backend output in ref0 OpenCV convention."""

    camera_xyz_world: np.ndarray
    identity_xyz_world: np.ndarray
    T_ref0_cam: np.ndarray
    K: np.ndarray
    tracks_xyz_ref0: np.ndarray
    tracks_visibility: np.ndarray
    identity_uv_px: np.ndarray
    coordinate_convention: str = COORDINATE_CONVENTION

    def to_npz_kwargs(
        self,
        *,
        video_height: int,
        video_width: int,
    ) -> dict[str, Any]:
        return {
            "camera_xyz_world": self.camera_xyz_world,
            "identity_xyz_world": self.identity_xyz_world,
            "t_ref0_cam": self.T_ref0_cam,
            "k_seq": self.K,
            "tracks_xyz_ref0": self.tracks_xyz_ref0,
            "tracks_visibility": self.tracks_visibility,
            "identity_uv_px": self.identity_uv_px,
            "video_height": video_height,
            "video_width": video_width,
        }


def normalize_c2w_to_ref0(c2w_traj: np.ndarray) -> np.ndarray:
    """Express cumulative c2w poses in ref0 frame with T_ref0_cam[0] = I."""
    c2w = np.asarray(c2w_traj, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w [T,4,4], got {c2w.shape}")
    c0_inv = np.linalg.inv(c2w[0])
    return np.stack([c0_inv @ c2w[t] for t in range(c2w.shape[0])], axis=0).astype(np.float32)


def cam_space_tracks_to_ref0(
    xyz_cam: np.ndarray,
    t_ref0_cam: np.ndarray,
) -> np.ndarray:
    """
    Lift per-frame camera-space 3D points into ref0 world.

    Args:
        xyz_cam: [T, Q, 3] points in each frame's camera coordinates.
        t_ref0_cam: [T, 4, 4] camera-to-world in ref0.

    Returns:
        tracks_xyz_ref0: [Q, T, 3]
    """
    xyz = np.asarray(xyz_cam, dtype=np.float64)
    poses = np.asarray(t_ref0_cam, dtype=np.float64)
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3]
    world = np.einsum("tij,tqj->tqi", rot, xyz) + trans[:, None, :]
    return np.transpose(world, (1, 0, 2)).astype(np.float32)


def compute_identity_centroid(
    tracks_xyz_ref0: np.ndarray,
    tracks_visibility: np.ndarray,
) -> np.ndarray:
    """Per-frame nanmean of visible query tracks -> [T, 3]."""
    tracks = np.asarray(tracks_xyz_ref0, dtype=np.float32)
    vis = np.asarray(tracks_visibility, dtype=bool)
    num_frames = int(tracks.shape[1])
    identity_xyz = np.full((num_frames, 3), np.nan, dtype=np.float32)
    for t in range(num_frames):
        vis_q = vis[:, t] & np.isfinite(tracks[:, t]).all(axis=-1)
        if np.any(vis_q):
            identity_xyz[t] = np.nanmean(tracks[vis_q, t], axis=0)
    return identity_xyz


# =============================================================================
# Trajectory extraction (per-backend building blocks)
# =============================================================================


def extract_camera_trajectory(
    model: torch.nn.Module,
    video_model_rgb: np.ndarray,
    image_hw: tuple[int, int],
    *,
    camera_grid_size: int,
    query_chunk_size: int,
    umeyama_slide_window: bool,
) -> dict[str, np.ndarray]:
    """
    Predict per-frame intrinsics K and camera-to-world poses T_ref0_cam.

    Extrinsics use Umeyama rigid alignment between ref-frame and target-frame
    3D query correspondences, then invert to obtain T_ref0_cam (camera in world).
    """
    camera_data = _predict_camera_branches(
        model=model,
        video_model_rgb=video_model_rgb,
        image_hw=image_hw,
        camera_grid_size=int(camera_grid_size),
        camera_query_chunk_size=int(query_chunk_size),
        predict_intrinsics=True,
        predict_extrinsics=True,
        umeyama_slide_window=bool(umeyama_slide_window),
    )
    if camera_data is None:
        raise RuntimeError("Camera branch prediction returned None.")

    t_ref0_cam = np.asarray(camera_data["T_ref0_cam"], dtype=np.float32)
    camera_xyz_world = t_ref0_cam[:, :3, 3].copy()
    return {
        "K": np.asarray(camera_data["K"], dtype=np.float32),
        "T_ref0_cam": t_ref0_cam,
        "camera_xyz_world": camera_xyz_world,
        "valid_intrinsics": np.asarray(camera_data["valid_intrinsics"], dtype=bool),
        "valid_extrinsics": np.asarray(camera_data["valid_extrinsics"], dtype=bool),
    }


def extract_identity_centroid_trajectory(
    model: torch.nn.Module,
    video_model_rgb: np.ndarray,
    identity_uv_norm: np.ndarray,
    *,
    query_chunk_size: int,
    umeyama_slide_window: bool,
) -> dict[str, np.ndarray]:
    """
    Track frame-0 identity queries through the clip; return per-frame centroid in ref0 world.
    """
    num_queries = int(identity_uv_norm.shape[0])
    track_payload = _infer_tracks(
        model=model,
        video_model_rgb=video_model_rgb,
        query_uv_norm=identity_uv_norm.astype(np.float32),
        query_chunk_size=int(query_chunk_size),
        query_src_indices_global=np.zeros((num_queries,), dtype=np.int64),
        umeyama_slide_window=bool(umeyama_slide_window),
    )
    tracks_xyz_ref0 = np.asarray(track_payload["tracks_xyz_ref0"], dtype=np.float32)  # [Q, T, 3]
    tracks_vis = np.asarray(track_payload["tracks_visibility"], dtype=bool)  # [Q, T]
    identity_xyz_world = compute_identity_centroid(tracks_xyz_ref0, tracks_vis)

    return {
        "tracks_xyz_ref0": tracks_xyz_ref0,
        "tracks_visibility": tracks_vis,
        "tracks_confidence": np.asarray(track_payload.get("tracks_confidence", np.nan), dtype=np.float32),
        "identity_xyz_world": identity_xyz_world,
    }


# =============================================================================
# Backend: OpenD4RT
# =============================================================================


def run_opend4rt_backend(
    *,
    config_path: str | Path,
    ckpt_path: str | Path,
    device: torch.device,
    video_rgb: np.ndarray,
    identity_uv_px: np.ndarray,
    identity_uv_norm: np.ndarray,
    image_hw: tuple[int, int],
    video_hw: tuple[int, int],
    camera_grid_size: int,
    query_chunk_size: int,
    umeyama_slide_window: bool,
) -> TrajectoryResult:
    """Run OpenD4RT camera + identity tracking; return NPZ-ready trajectories."""
    height, width = video_hw
    video_rgb, video_model_rgb = prepare_video_inputs(video_rgb, image_hw=image_hw)

    print(f"Loading D4RT model from {ckpt_path}")
    model = load_d4rt_model(config_path, ckpt_path, device)

    print("Estimating camera trajectory (Umeyama extrinsics + intrinsics)...")
    camera_result = extract_camera_trajectory(
        model=model,
        video_model_rgb=video_model_rgb,
        image_hw=(height, width),
        camera_grid_size=int(camera_grid_size),
        query_chunk_size=int(query_chunk_size),
        umeyama_slide_window=bool(umeyama_slide_window),
    )

    with timer("tracking"):
        print("Tracking identity queries in ref0 world frame...")
        identity_result = extract_identity_centroid_trajectory(
            model=model,
            video_model_rgb=video_model_rgb,
            identity_uv_norm=identity_uv_norm,
            query_chunk_size=int(query_chunk_size),
            umeyama_slide_window=bool(umeyama_slide_window),
        )

    return TrajectoryResult(
        camera_xyz_world=camera_result["camera_xyz_world"],
        identity_xyz_world=identity_result["identity_xyz_world"],
        T_ref0_cam=camera_result["T_ref0_cam"],
        K=camera_result["K"],
        tracks_xyz_ref0=identity_result["tracks_xyz_ref0"],
        tracks_visibility=identity_result["tracks_visibility"],
        identity_uv_px=identity_uv_px,
    )


# =============================================================================
# Backend: SpaTrackV2
# =============================================================================

_SPATRACKER_PATH_INSTALLED = False


def _ensure_spatracker_path() -> None:
    """Add preprocess/SpaTrackerV2 to sys.path for upstream models.SpaTrackV2 imports."""
    global _SPATRACKER_PATH_INSTALLED
    if _SPATRACKER_PATH_INSTALLED:
        return
    root = str(SPATRACKER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _SPATRACKER_PATH_INSTALLED = True


def _scale_uv_to_preprocessed(
    uv_px: np.ndarray,
    orig_hw: tuple[int, int],
    proc_hw: tuple[int, int],
) -> np.ndarray:
    """Map pixel UV from original video resolution to preprocessed tracker resolution."""
    orig_h, orig_w = orig_hw
    proc_h, proc_w = proc_hw
    scaled = np.asarray(uv_px, dtype=np.float32).copy()
    scaled[:, 0] *= float(proc_w) / float(max(orig_w, 1))
    scaled[:, 1] *= float(proc_h) / float(max(orig_h, 1))
    return scaled


def _scale_intrinsics_to_original(
    k_seq: np.ndarray,
    orig_hw: tuple[int, int],
    proc_hw: tuple[int, int],
) -> np.ndarray:
    """Scale K from preprocessed tracker resolution back to original video resolution."""
    orig_h, orig_w = orig_hw
    proc_h, proc_w = proc_hw
    k = np.asarray(k_seq, dtype=np.float32).copy()
    scale_x = float(orig_w) / float(max(proc_w, 1))
    scale_y = float(orig_h) / float(max(proc_h, 1))
    k[:, 0, 0] *= scale_x
    k[:, 0, 2] *= scale_x
    k[:, 1, 1] *= scale_y
    k[:, 1, 2] *= scale_y
    return k


def _build_spatrack_queries(identity_uv_px: np.ndarray) -> np.ndarray:
    """OneFormer UV seeds -> SpaTrack query_xyt [Q, 3] as [frame, u, v]."""
    uv = np.asarray(identity_uv_px, dtype=np.float32)
    frame_idx = np.zeros((uv.shape[0], 1), dtype=np.float32)
    return np.concatenate([frame_idx, uv], axis=1)


def run_spatrackerv2_backend(
    *,
    device: torch.device,
    video_tensor: torch.Tensor,
    identity_uv_px: np.ndarray,
    video_hw: tuple[int, int],
    front_ckpt: str,
    tracker_ckpt: str,
) -> TrajectoryResult:
    """
    Run SpaTrackV2 VGGT4Track front-end + Predictor tracking.
    Converts outputs to ref0 OpenCV NPZ convention internally.
    """
    _ensure_spatracker_path()
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

    height, width = video_hw
    num_frames = int(video_tensor.shape[0])

    print(f"Loading SpaTrackV2 front-end from {front_ckpt}")
    vggt4track_model = VGGT4Track.from_pretrained(front_ckpt)
    vggt4track_model.eval().to(device)

    print(f"Loading SpaTrackV2 tracker from {tracker_ckpt}")
    tracker_model = Predictor.from_pretrained(tracker_ckpt)
    tracker_model.eval().to(device)

    # Keep 0-255 float on GPU (decord convention); preprocess before front-end + tracker.
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
        identity_uv_px,
        orig_hw=(height, width),
        proc_hw=(proc_h, proc_w),
    )
    query_xyt = _build_spatrack_queries(scaled_uv)
    print(f"  SpaTrack queries: {query_xyt.shape[0]} points at frame 0 "
          f"(proc resolution {proc_w}x{proc_h})")

    print("Running SpaTrackV2 Predictor (joint 3D tracking)...")
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            (
                c2w_traj,
                intrs_out,
                _point_map,
                _conf_depth,
                track3d_pred,
                _track2d_pred,
                vis_pred,
                _conf_pred,
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

    c2w = c2w_traj.detach().cpu().numpy()
    t_ref0_cam = normalize_c2w_to_ref0(c2w)
    camera_xyz_world = t_ref0_cam[:, :3, 3].copy()

    xyz_cam = track3d_pred[:, :, :3].detach().cpu().numpy()
    tracks_xyz_ref0 = cam_space_tracks_to_ref0(xyz_cam, t_ref0_cam)

    vis = vis_pred.squeeze(-1).detach().cpu().numpy()
    if vis.ndim == 2:
        tracks_visibility = (vis > 0.5).T.astype(bool)
    else:
        tracks_visibility = (vis > 0.5).astype(bool)

    k_seq = intrs_out.detach().cpu().numpy().astype(np.float32)
    if k_seq.shape[0] != num_frames:
        k_seq = k_seq[:num_frames]
    k_seq = _scale_intrinsics_to_original(
        k_seq,
        orig_hw=(height, width),
        proc_hw=(proc_h, proc_w),
    )

    identity_xyz_world = compute_identity_centroid(tracks_xyz_ref0, tracks_visibility)

    return TrajectoryResult(
        camera_xyz_world=camera_xyz_world,
        identity_xyz_world=identity_xyz_world,
        T_ref0_cam=t_ref0_cam,
        K=k_seq,
        tracks_xyz_ref0=tracks_xyz_ref0,
        tracks_visibility=tracks_visibility,
        identity_uv_px=identity_uv_px,
    )


def _path_length(xyz: np.ndarray) -> float:
    """Sum of Euclidean step lengths along a [T, 3] trajectory (skipping NaN gaps)."""
    pts = np.asarray(xyz, dtype=np.float64)
    total = 0.0
    prev = None
    for row in pts:
        if not np.isfinite(row).all():
            prev = None
            continue
        if prev is not None:
            total += float(np.linalg.norm(row - prev))
        prev = row
    return total


def print_trajectory_summary(
    camera_xyz: np.ndarray,
    identity_xyz: np.ndarray,
    *,
    identity_meta: dict[str, Any],
) -> None:
    """Print a concise summary for stdout output."""
    print("\n=== Trajectory summary (ref0 world frame, model-relative scale) ===")
    print(f"Identity class: {identity_meta.get('class_name', '?')} "
          f"({identity_meta.get('num_queries', '?')} query points)")
    for name, traj in (("Camera", camera_xyz), ("Identity centroid", identity_xyz)):
        valid = np.isfinite(traj).all(axis=-1)
        n_valid = int(np.count_nonzero(valid))
        print(f"\n{name}:")
        print(f"  Valid frames: {n_valid}/{traj.shape[0]}")
        if n_valid > 0:
            first = traj[valid][0]
            last = traj[valid][-1]
            print(f"  Start xyz: [{first[0]:.4f}, {first[1]:.4f}, {first[2]:.4f}]")
            print(f"  End   xyz: [{last[0]:.4f}, {last[1]:.4f}, {last[2]:.4f}]")
            print(f"  Path length: {_path_length(traj):.4f}")
    print()


def save_trajectories_npz(
    output_path: str | Path,
    *,
    camera_xyz_world: np.ndarray,
    identity_xyz_world: np.ndarray,
    t_ref0_cam: np.ndarray,
    k_seq: np.ndarray,
    tracks_xyz_ref0: np.ndarray,
    tracks_visibility: np.ndarray,
    identity_uv_px: np.ndarray,
    video_height: int,
    video_width: int,
) -> None:
    """Persist trajectory arrays and metadata for offline_view_3d_demo.py."""
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = PREPROCESS_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        camera_xyz_world=camera_xyz_world.astype(np.float32),
        identity_xyz_world=identity_xyz_world.astype(np.float32),
        T_ref0_cam=t_ref0_cam.astype(np.float32),
        K=k_seq.astype(np.float32),
        tracks_xyz_ref0=tracks_xyz_ref0.astype(np.float32),
        tracks_visibility=tracks_visibility.astype(bool),
        identity_uv_px=identity_uv_px.astype(np.float32),
        video_height=np.int32(video_height),
        video_width=np.int32(video_width),
        coordinate_convention=np.asarray(COORDINATE_CONVENTION),
    )
    print(f"Saved trajectories to {out_path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    default_config = PREPROCESS_ROOT / "Open_d4rt" / "configs" / "model_effective.yaml"
    parser = argparse.ArgumentParser(
        description="3D camera + identity centroid trajectory demo (OpenD4RT or SpaTrackV2 + OneFormer)."
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video path.")
    parser.add_argument("--segment", action='store_true', help="Whether to create or overwrite existing segment JSON. Evaluates to true if typed, false if not.")
    parser.add_argument(
        "--model",
        type=str,
        default="opend4rt",
        choices=("opend4rt", "spatrackerv2"),
        help="Trajectory backend: opend4rt or spatrackerv2.",
    )
    parser.add_argument("--config", type=str, default=str(default_config), help="D4RT model config yaml.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="D4RT checkpoint path (required for opend4rt).")
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
    parser.add_argument("--num_frames", type=int, default=64, help="Max frames to process.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--num_identity_queries", type=int, default=24, help="UV query points on identity mask.")
    parser.add_argument("--query_chunk_size", type=int, default=1024)
    parser.add_argument("--camera_grid_size", type=int, default=16, help="Coarse grid for camera branch queries.")
    parser.add_argument(
        "--umeyama_slide_window",
        action="store_true",
        help="Stitch long sequences with Umeyama Sim(3) sliding windows (clip > 48 frames).",
    )
    parser.add_argument("--output_npz", type=str, default=None, help="Path to save trajectory NPZ for offline viewing.")
    parser.add_argument(
        "--sample_step",
        type=int,
        default=10,
        help="Subsample every N frames when loading video (default 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)

    if args.model == "opend4rt" and not args.ckpt_path:
        raise ValueError("--ckpt_path is required when --model opend4rt")

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print(f"Loading video: {video_path}")
    video_tensor = read_video_to_tensor(video_path, sample_step=args.sample_step, max_frames=int(args.num_frames))
    video_rgb = tensor_to_video_rgb(video_tensor)
    num_frames, height, width = int(video_rgb.shape[0]), int(video_rgb.shape[1]), int(video_rgb.shape[2])

    # path to temporary npy file
    script_dir = Path(__file__).resolve().parent

    output_path = script_dir / "persistent"
    
    print(f"  Frames: {num_frames}, resolution: {width}x{height}")

    if args.segment:


        print("Segmenting frame-0 with OneFormer...")
        identity_uv_px, identity_meta = sample_identity_uv_queries(
            video_rgb[0],
            num_queries=int(args.num_identity_queries),
        )
        print(
            f"  Sampled {identity_meta['num_queries']} query points "
            f"(mask area ratio {identity_meta['mask_area_ratio']:.3f})"
        )


        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Save your numpy array
        np.save(output_path /  "identity_uv_px.npy", identity_uv_px)
        with open(output_path / "identity_meta.json", "w") as f:
            json.dump(identity_meta, f, indent=4)

        return

    identity_uv_px = np.load(output_path / "identity_uv_px.npy")
    with open(output_path / "identity_meta.json", "r") as f:
        identity_meta = json.load(f)

    if args.model == "opend4rt":
        cfg = load_yaml_config(args.config)
        image_size = cfg.get_path("model.input.image_size", [256, 256])
        image_hw = (int(image_size[0]), int(image_size[1]))
        identity_uv_norm = uv_px_to_norm(identity_uv_px, width=width, height=height)
        with timer(f"opend4rt {num_frames}"):
            result = run_opend4rt_backend(
                config_path=args.config,
                ckpt_path=args.ckpt_path,
                device=device,
                video_rgb=video_rgb,
                identity_uv_px=identity_uv_px,
                identity_uv_norm=identity_uv_norm,
                image_hw=image_hw,
                video_hw=(height, width),
                camera_grid_size=int(args.camera_grid_size),
                query_chunk_size=int(args.query_chunk_size),
                umeyama_slide_window=bool(args.umeyama_slide_window),
            )
    else:
        with timer(f"spatracker {num_frames}"):
            result = run_spatrackerv2_backend(
                device=device,
                video_tensor=video_tensor,
                identity_uv_px=identity_uv_px,
                video_hw=(height, width),
                front_ckpt=args.spatrack_front_ckpt,
                tracker_ckpt=args.spatrack_tracker_ckpt,
            )

    print_trajectory_summary(
        result.camera_xyz_world,
        result.identity_xyz_world,
        identity_meta=identity_meta,
    )

    if args.output_npz:
        save_trajectories_npz(args.output_npz, **result.to_npz_kwargs(video_height=height, video_width=width))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
