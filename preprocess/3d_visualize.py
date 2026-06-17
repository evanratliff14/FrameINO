#!/usr/bin/env python3
"""
3D Camera + Identity Trajectory Demo.

For a single input video, estimates:
  1. Camera world position (x, y, z) over time in the OpenD4RT ref0 frame.
  2. Identity (person) world centroid over time via frame-0 panoptic queries + 3D tracking.

Coordinate convention (OpenD4RT / data_schema.md):
  - ref0 is the world frame anchored at frame 0.
  - T_ref0_cam[t] is a 4x4 camera-to-world transform; camera position is T_ref0_cam[t, :3, 3].
  - tracks_xyz_ref0[q, t] are 3D points in the same ref0 world frame.

Scale is model-relative (no metric GT); trajectories show relative motion structure.
Export NPZ with --output_npz, then view locally via offline_view_3d_demo.py.

python preprocess/3d_visualize.py   --video_path /home/uft5by/FrameINO/preprocess/1917.mp4   --ckpt_path /home/uft5by/FrameINO/preprocess/Open_d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt   --config preprocess/Open_d4rt/configs/model_effective.yaml   --num_frames 10000   --umeyama_slide_window   --output_npz tmp/trajectories.npz

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from sklearn.cluster import KMeans
import cv2
import time
from contextlib import contextmanager

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
# Trajectory extraction
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

    num_frames = int(tracks_xyz_ref0.shape[1])
    identity_xyz_world = np.full((num_frames, 3), np.nan, dtype=np.float32)
    for t in range(num_frames):
        vis_q = tracks_vis[:, t] & np.isfinite(tracks_xyz_ref0[:, t]).all(axis=-1)
        if np.any(vis_q):
            identity_xyz_world[t] = np.nanmean(tracks_xyz_ref0[vis_q, t], axis=0)

    return {
        "tracks_xyz_ref0": tracks_xyz_ref0,
        "tracks_visibility": tracks_vis,
        "tracks_confidence": np.asarray(track_payload.get("tracks_confidence", np.nan), dtype=np.float32),
        "identity_xyz_world": identity_xyz_world,
    }


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


COORDINATE_CONVENTION = "ref0_opencv_t0_identity"


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
        description="3D camera + identity centroid trajectory demo (OpenD4RT + OneFormer)."
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video path.")
    parser.add_argument("--config", type=str, default=str(default_config), help="D4RT model config yaml.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="D4RT checkpoint path.")
    parser.add_argument("--num_frames", type=int, default=64, help="Max frames to process.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--num_identity_queries", type=int, default=48, help="UV query points on identity mask.")
    parser.add_argument("--query_chunk_size", type=int, default=1024)
    parser.add_argument("--camera_grid_size", type=int, default=16, help="Coarse grid for camera branch queries.")
    parser.add_argument(
        "--umeyama_slide_window",
        action="store_true",
        help="Stitch long sequences with Umeyama Sim(3) sliding windows (clip > 48 frames).",
    )
    parser.add_argument("--output_npz", type=str, default=None, help="Path to save trajectory NPZ for offline viewing.")
    parser.add_argument("--sample_step", type=int, default=None, help="Range function third argument to contruct frames for tracking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print(f"Loading video: {video_path}")
    video_tensor = read_video_to_tensor(video_path, sample_step=args.sample_step, max_frames=int(args.num_frames))
    video_rgb = tensor_to_video_rgb(video_tensor)
    num_frames, height, width = int(video_rgb.shape[0]), int(video_rgb.shape[1]), int(video_rgb.shape[2])
    print(f"  Frames: {num_frames}, resolution: {width}x{height}")

    cfg = load_yaml_config(args.config)
    image_size = cfg.get_path("model.input.image_size", [256, 256])
    image_hw = (int(image_size[0]), int(image_size[1]))
    video_rgb, video_model_rgb = prepare_video_inputs(video_rgb, image_hw=image_hw)

    print(f"Loading D4RT model from {args.ckpt_path}")
    model = load_d4rt_model(args.config, args.ckpt_path, device)

    print(f"Segmenting frame-0 with OneFormer...")
    identity_uv_px, identity_meta = sample_identity_uv_queries(
        video_rgb[0],
        num_queries=int(args.num_identity_queries),
    )
    identity_uv_norm = uv_px_to_norm(identity_uv_px, width=width, height=height)
    print(f"  Sampled {identity_meta['num_queries']} query points "
          f"(mask area ratio {identity_meta['mask_area_ratio']:.3f})")

    print("Estimating camera trajectory (Umeyama extrinsics + intrinsics)...")
    camera_result = extract_camera_trajectory(
        model=model,
        video_model_rgb=video_model_rgb,
        image_hw=(height, width),
        camera_grid_size=int(args.camera_grid_size),
        query_chunk_size=int(args.query_chunk_size),
        umeyama_slide_window=bool(args.umeyama_slide_window),
    )
    with timer("tracking"):
        print("Tracking identity queries in ref0 world frame...")
        identity_result = extract_identity_centroid_trajectory(
            model=model,
            video_model_rgb=video_model_rgb,
            identity_uv_norm=identity_uv_norm,
            query_chunk_size=int(args.query_chunk_size),
            umeyama_slide_window=bool(args.umeyama_slide_window),
        )
    camera_xyz_world = camera_result["camera_xyz_world"]
    identity_xyz_world = identity_result["identity_xyz_world"]

    print_trajectory_summary(camera_xyz_world, identity_xyz_world, identity_meta=identity_meta)

    if args.output_npz:
        save_trajectories_npz(
            args.output_npz,
            camera_xyz_world=camera_xyz_world,
            identity_xyz_world=identity_xyz_world,
            t_ref0_cam=camera_result["T_ref0_cam"],
            k_seq=camera_result["K"],
            tracks_xyz_ref0=identity_result["tracks_xyz_ref0"],
            tracks_visibility=identity_result["tracks_visibility"],
            identity_uv_px=identity_uv_px,
            video_height=height,
            video_width=width,
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
