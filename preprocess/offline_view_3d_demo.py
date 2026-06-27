#!/usr/bin/env python3
"""
Offline Viser viewer for trajectory NPZ files exported by 3d_visualize.py.

- 3D scene uses a fixed OpenCV-ref0 -> Y-up viewer rotation so frame-0 camera
  points horizontally (parallel to ground), not sky-facing.
- 2D slider frames show depth-colored reprojections of 3D identity tracks.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import viser

# OpenCV ref0: +x right, +y down, +z forward.
# Y-up viewer: map ref0 +z (forward) -> viewer +x (horizontal look along ground).
R_OPENCV_TO_VIEWER = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


# =============================================================================
# Video I/O
# =============================================================================


def load_video_frames(video_path: Path, sample_step: int = 10, max_frames: int | None = None) -> np.ndarray:
    """Load video frames as uint8 RGB [T, H, W, 3] via OpenCV."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_step ==0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= int(max_frames * sample_step):
            break
        i+=1
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    frames = np.stack(frames, axis=0)
    return frames


# =============================================================================
# Viewer coordinate transform (display only; NPZ stays in ref0 OpenCV)
# =============================================================================


def transform_points_for_viewer(xyz: np.ndarray) -> np.ndarray:
    """Map ref0 OpenCV world points into Y-up viewer coordinates."""
    pts = np.asarray(xyz, dtype=np.float64)
    out = pts.copy()
    valid = np.isfinite(pts).all(axis=-1)
    if np.any(valid):
        out[valid] = (R_OPENCV_TO_VIEWER @ pts[valid].T).T
    return out.astype(np.float32)


def transform_pose_for_viewer(t_ref0_cam: np.ndarray) -> np.ndarray:
    """Map camera-to-world pose from ref0 OpenCV into Y-up viewer coordinates."""
    pose = np.asarray(t_ref0_cam, dtype=np.float64).reshape(4, 4).copy()
    r = pose[:3, :3]
    t = pose[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_OPENCV_TO_VIEWER @ r
    out[:3, 3] = R_OPENCV_TO_VIEWER @ t
    return out.astype(np.float32)


# =============================================================================
# Dense point cloud (load + Viser scene helpers)
# =============================================================================


@dataclass
class PointCloudBundle:
    points_xyz_ref0: np.ndarray
    points_vis: np.ndarray
    points_conf: np.ndarray
    points_rgb: np.ndarray | None
    allowed_track_mask: np.ndarray | None
    point_is_dynamic: np.ndarray | None
    xyz_center: np.ndarray
    xyz_radius: float
    coordinate_convention: str
    num_frames: int


def _subsample_indices(total: int, keep: int) -> np.ndarray:
    keep = max(0, min(int(keep), int(total)))
    if keep <= 0:
        return np.zeros((0,), dtype=np.int64)
    if keep >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=keep, dtype=np.int64)


def _ensure_time_series(arr: np.ndarray, *, name: str) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 2:
        return data[None, ...]
    if data.ndim != 3:
        raise ValueError(f"{name} must have shape [T,N,C] or [N,C], got {data.shape}")
    return data


def _resolve_point_cloud_path(path: Path) -> Path:
    """Resolve legacy np.save outputs that append `.npy` to the requested filename."""
    if path.exists():
        return path
    npy_variant = Path(f"{path}.npy")
    if npy_variant.exists():
        return npy_variant
    return path


def load_point_cloud_npz(path: Path) -> PointCloudBundle:
    """Load a dense point cloud NPZ exported by dense_track.py (or legacy dict saves)."""
    path = _resolve_point_cloud_path(Path(path))
    if not path.exists():
        raise FileNotFoundError(f"Point cloud NPZ not found at: {path}")

    points_xyz_ref0: np.ndarray | None = None
    points_vis: np.ndarray | None = None
    points_conf: np.ndarray | None = None
    points_rgb: np.ndarray | None = None
    allowed_track_mask: np.ndarray | None = None
    point_is_dynamic: np.ndarray | None = None
    xyz_center = np.zeros((3,), dtype=np.float32)
    xyz_radius = 1.0
    coordinate_convention = "opencv_ref0"

    data = np.load(path, allow_pickle=True)
    try:
        if isinstance(data, np.lib.npyio.NpzFile) and "points_xyz_ref0" in data.files:
            points_xyz_ref0 = np.asarray(data["points_xyz_ref0"], dtype=np.float32)
            points_vis = np.asarray(data["points_vis"], dtype=bool)
            points_conf = np.asarray(data["points_conf"], dtype=np.float32)
            points_rgb = (
                np.asarray(data["points_rgb"], dtype=np.uint8) if "points_rgb" in data.files else None
            )
            allowed_track_mask = (
                np.asarray(data["allowed_track_mask"], dtype=bool)
                if "allowed_track_mask" in data.files
                else None
            )
            point_is_dynamic = (
                np.asarray(data["point_is_dynamic"], dtype=bool)
                if "point_is_dynamic" in data.files
                else None
            )
            if "xyz_center" in data.files:
                xyz_center = np.asarray(data["xyz_center"], dtype=np.float32)
            if "xyz_radius" in data.files:
                xyz_radius = float(np.asarray(data["xyz_radius"]).reshape(-1)[0])
            if "coordinate_convention" in data.files:
                coordinate_convention = str(np.asarray(data["coordinate_convention"]).item())
        elif int(getattr(data, "ndim", -1)) == 0:
            payload = data.item()
            if not isinstance(payload, dict):
                raise ValueError(f"Legacy point cloud payload in {path} is not a dict")
            points_xyz_ref0 = np.asarray(payload["points_xyz_ref0"], dtype=np.float32)
            points_vis = np.asarray(payload["points_vis"], dtype=bool)
            points_conf = np.asarray(payload["points_conf"], dtype=np.float32)
            points_rgb = np.asarray(payload["points_rgb"], dtype=np.uint8) if "points_rgb" in payload else None
            allowed_track_mask = (
                np.asarray(payload["allowed_track_mask"], dtype=bool)
                if "allowed_track_mask" in payload
                else None
            )
            point_is_dynamic = (
                np.asarray(payload["point_is_dynamic"], dtype=bool)
                if "point_is_dynamic" in payload
                else None
            )
            xyz_center = np.asarray(payload.get("xyz_center", xyz_center), dtype=np.float32)
            xyz_radius = float(payload.get("xyz_radius", xyz_radius))
            coordinate_convention = str(payload.get("coordinate_convention", coordinate_convention))
        else:
            raise ValueError(f"Unrecognized point cloud format in {path}")
    finally:
        if hasattr(data, "close"):
            data.close()

    if points_xyz_ref0 is None or points_vis is None or points_conf is None:
        raise ValueError(f"Point cloud file {path} is missing required arrays")

    points_xyz_ref0 = _ensure_time_series(points_xyz_ref0, name="points_xyz_ref0")
    if points_vis.ndim == 1:
        points_vis = np.tile(points_vis[None, :], (points_xyz_ref0.shape[0], 1))
    points_vis = np.asarray(points_vis, dtype=bool)
    if points_conf.ndim == 1:
        points_conf = np.tile(points_conf[None, :], (points_xyz_ref0.shape[0], 1))
    if points_rgb is not None:
        points_rgb = _ensure_time_series(points_rgb, name="points_rgb")
    if point_is_dynamic is not None and point_is_dynamic.ndim == 2:
        point_is_dynamic = point_is_dynamic.any(axis=0)

    return PointCloudBundle(
        points_xyz_ref0=points_xyz_ref0,
        points_vis=points_vis,
        points_conf=np.asarray(points_conf, dtype=np.float32),
        points_rgb=points_rgb,
        allowed_track_mask=allowed_track_mask,
        point_is_dynamic=point_is_dynamic,
        xyz_center=np.asarray(xyz_center, dtype=np.float32).reshape(3),
        xyz_radius=float(xyz_radius),
        coordinate_convention=coordinate_convention,
        num_frames=int(points_xyz_ref0.shape[0]),
    )


def _depth_fallback_colors(xyz_ref0: np.ndarray) -> np.ndarray:
    z = xyz_ref0[:, 2]
    valid = np.isfinite(z)
    if not np.any(valid):
        return np.full((xyz_ref0.shape[0], 3), 180, dtype=np.uint8)
    z_valid = z[valid]
    z_min, z_max = float(z_valid.min()), float(z_valid.max())
    denom = max(z_max - z_min, 1e-6)
    colors = np.full((xyz_ref0.shape[0], 3), 180, dtype=np.uint8)
    for i in np.flatnonzero(valid):
        t = float(np.clip((z[i] - z_min) / denom, 0.0, 1.0))
        idx = int(round(t * 255.0))
        bgr = cv2.applyColorMap(np.array([[idx]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        colors[i] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return colors


def _confidence_probability(conf: np.ndarray) -> np.ndarray:
    """
    Map stored confidence to a [0, 1] probability.

    OpenD4RT dense exports store raw logits; SpaTrackV2 stores sigmoid outputs.
    """
    c = np.asarray(conf, dtype=np.float32)
    out = np.zeros_like(c, dtype=np.float32)
    finite = np.isfinite(c)
    if not np.any(finite):
        return out
    cf = c[finite]
    if float(np.max(cf)) > 1.0 or float(np.min(cf)) < 0.0:
        out[finite] = 1.0 / (1.0 + np.exp(-cf))
    else:
        out[finite] = cf
    return out


def prepare_point_cloud_for_frame(
    bundle: PointCloudBundle,
    frame_idx: int,
    *,
    mode: Literal["3d", "4d"] = "4d",
    show_static: bool = True,
    show_dynamic: bool = False,
    point_budget: int = 30000,
    conf_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select dense-cloud points and map them into viewer space.

    3D mode unions visible points from every frame (full-scene reconstruction).
    4D mode uses a single timeline frame.

    Static vs dynamic filtering uses ``point_is_dynamic`` when available:
    - show_static=False masks out non-dynamic points
    - show_dynamic=False masks out dynamic points (static scene only)

    Points below ``conf_threshold`` (on a [0, 1] scale) are excluded.
    """
    if not show_static and not show_dynamic:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    if mode == "3d":
        frame_indices = range(bundle.num_frames)
    else:
        frame_indices = [int(np.clip(frame_idx, 0, max(bundle.num_frames - 1, 0)))]

    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []

    for t in frame_indices:
        xyz_t = bundle.points_xyz_ref0[t]
        vis_t = bundle.points_vis[t] if bundle.points_vis.ndim == 2 else bundle.points_vis
        valid = np.isfinite(xyz_t).all(axis=-1) & vis_t
        if bundle.allowed_track_mask is not None:
            valid = valid & bundle.allowed_track_mask
        if float(conf_threshold) > 0.0:
            conf_prob = _confidence_probability(bundle.points_conf[t])
            valid = valid & (conf_prob >= float(conf_threshold))

        if bundle.point_is_dynamic is not None:
            is_dyn = bundle.point_is_dynamic
            keep = np.zeros_like(valid, dtype=bool)
            if show_static:
                keep |= valid & ~is_dyn
            if show_dynamic:
                keep |= valid & is_dyn
            valid = keep
        elif not show_static:
            valid = np.zeros_like(valid, dtype=bool)

        idx = np.flatnonzero(valid)
        if idx.size <= 0:
            continue
        xyz_chunks.append(xyz_t[idx])
        if bundle.points_rgb is not None:
            rgb_chunks.append(bundle.points_rgb[t, idx].astype(np.uint8))
        else:
            rgb_chunks.append(_depth_fallback_colors(xyz_t[idx]))

    if not xyz_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    xyz_sel = np.concatenate(xyz_chunks, axis=0)
    colors = np.concatenate(rgb_chunks, axis=0)
    if xyz_sel.shape[0] > int(point_budget):
        pick = _subsample_indices(xyz_sel.shape[0], int(point_budget))
        xyz_sel = xyz_sel[pick]
        colors = colors[pick]

    return transform_points_for_viewer(xyz_sel), colors


def add_point_cloud_to_viser_scene(
    scene: Any,
    name: str,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    scene_radius: float = 1.0,
    point_size_scale: float = 1.0,
    point_shape: str = "circle",
) -> Any:
    """Add a colored point cloud to any Viser scene graph node."""
    if int(points.shape[0]) <= 0:
        return None
    return scene.add_point_cloud(
        name,
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
        point_size=max(float(scene_radius) * 0.0035 * float(point_size_scale), 0.003),
        point_shape=point_shape,
        precision="float32",
    )


# =============================================================================
# 3D projection (raw ref0 OpenCV; used for 2D overlay)
# =============================================================================


def project_world_to_image(
    k: np.ndarray,
    t_ref0_cam: np.ndarray,
    p_world: np.ndarray,
) -> tuple[float, float, float] | None:
    """
    Project a ref0-world 3D point into pixel coordinates at the given camera frame.

    Returns (u, v, z_cam) or None if behind the camera / invalid.
    """
    p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    t_cw = np.linalg.inv(np.asarray(t_ref0_cam, dtype=np.float64).reshape(4, 4))
    p_cam = (t_cw @ p_h)[:3]
    z = float(p_cam[2])
    if not np.isfinite(z) or z <= 1e-6:
        return None
    proj = np.asarray(k, dtype=np.float64).reshape(3, 3) @ p_cam
    return float(proj[0] / z), float(proj[1] / z), z


def compute_global_depth_range(
    *,
    tracks_xyz_ref0: np.ndarray,
    tracks_visibility: np.ndarray,
    k_seq: np.ndarray,
    t_ref0_cam: np.ndarray,
    num_frames: int,
) -> tuple[float, float]:
    """Min/max positive camera-space depth over all visible reprojections."""
    depths: list[float] = []
    for t in range(num_frames):
        k_t = k_seq[t] if t < k_seq.shape[0] else k_seq[0]
        pose_t = t_ref0_cam[t] if t < t_ref0_cam.shape[0] else t_ref0_cam[0]
        vis_q = tracks_visibility[:, t] & np.isfinite(tracks_xyz_ref0[:, t]).all(axis=-1)
        for q in np.flatnonzero(vis_q):
            proj = project_world_to_image(k_t, pose_t, tracks_xyz_ref0[q, t])
            if proj is not None:
                depths.append(proj[2])
    if not depths:
        return 0.0, 1.0
    return float(min(depths)), float(max(depths))


def depth_to_bgr(z: float, z_min: float, z_max: float) -> tuple[int, int, int]:
    """Map depth to BGR via TURBO colormap (for cv2 drawing on RGB frame)."""
    denom = max(z_max - z_min, 1e-6)
    t = float(np.clip((z - z_min) / denom, 0.0, 1.0))
    idx = int(round(t * 255.0))
    bgr = cv2.applyColorMap(np.array([[idx]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    # OpenCV is BGR; convert to RGB tuple for drawing on RGB frame.
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def render_depth_colored_reprojection(
    frame_rgb: np.ndarray,
    frame_idx: int,
    *,
    tracks_xyz_ref0: np.ndarray,
    tracks_visibility: np.ndarray,
    k_seq: np.ndarray,
    t_ref0_cam: np.ndarray,
    identity_uv_px: np.ndarray | None,
    z_min: float,
    z_max: float,
    use_global_depth: bool,
    point_radius: int,
) -> np.ndarray:
    """Overlay depth-colored circles for 3D track reprojections onto an RGB frame."""
    out = np.asarray(frame_rgb, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    t = int(frame_idx)
    k_t = k_seq[t] if t < k_seq.shape[0] else k_seq[0]
    pose_t = t_ref0_cam[t] if t < t_ref0_cam.shape[0] else t_ref0_cam[0]

    vis_q = tracks_visibility[:, t] & np.isfinite(tracks_xyz_ref0[:, t]).all(axis=-1)
    frame_depths: list[float] = []
    projections: list[tuple[int, int, float]] = []

    for q in np.flatnonzero(vis_q):
        proj = project_world_to_image(k_t, pose_t, tracks_xyz_ref0[q, t])
        if proj is None:
            continue
        u, v, z = proj
        if not (0.0 <= u < w and 0.0 <= v < h):
            continue
        frame_depths.append(z)
        projections.append((int(round(u)), int(round(v)), z))

    if use_global_depth:
        depth_min, depth_max = z_min, z_max
    elif frame_depths:
        depth_min, depth_max = float(min(frame_depths)), float(max(frame_depths))
    else:
        depth_min, depth_max = z_min, z_max

    for u, v, z in projections:
        color = depth_to_bgr(z, depth_min, depth_max)
        cv2.circle(out, (u, v), int(point_radius), color, thickness=-1, lineType=cv2.LINE_AA)

    # Frame-0 seed queries as white rings (reference for where tracking started).
    if t == 0 and identity_uv_px is not None and identity_uv_px.size > 0:
        for u0, v0 in np.asarray(identity_uv_px, dtype=np.float32):
            if not np.isfinite(u0) or not np.isfinite(v0):
                continue
            ui, vi = int(round(float(u0))), int(round(float(v0)))
            if 0 <= ui < w and 0 <= vi < h:
                cv2.circle(out, (ui, vi), int(point_radius) + 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return out


# =============================================================================
# Viser helpers
# =============================================================================


def _rotmat_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to viser wxyz quaternion."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw, qx = 0.25 * s, (r[2, 1] - r[1, 2]) / s
        qy, qz = (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2.0
        qw, qx = (r[2, 1] - r[1, 2]) / s, 0.25 * s
        qy, qz = (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2.0
        qw, qx = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s
        qy, qz = 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2.0
        qw, qx = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s
        qy, qz = (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return tuple(float(v) for v in q.tolist())


def _fov_from_k(k: np.ndarray, image_h: int) -> float:
    kk = np.asarray(k, dtype=np.float64).reshape(3, 3)
    fy = float(kk[1, 1])
    if not np.isfinite(fy) or abs(fy) < 1e-6:
        return math.radians(50.0)
    return float(2.0 * math.atan2(float(image_h) * 0.5, fy))


def _trajectory_line_segments(xyz: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(xyz, dtype=np.float32)
    segments: list[np.ndarray] = []
    prev = None
    for row in pts:
        if not np.isfinite(row).all():
            prev = None
            continue
        if prev is not None:
            segments.append(np.stack([prev, row], axis=0))
        prev = row
    if not segments:
        return None
    return np.stack(segments, axis=0)


def jump_client_view_to_camera_pose(
    client: Any,
    pose_viewer: np.ndarray,
    *,
    fov: float | None = None,
    fallback_look_distance: float = 1.0,
) -> None:
    """
    Align a Viser client navigation camera with a camera-to-world pose in viewer space.

    Uses the same forward/up convention as add_camera_frustum (optical axis = R[:, 2]).
    """
    cam = getattr(client, "camera", None)
    if cam is None:
        return
    pose = np.asarray(pose_viewer, dtype=np.float64).reshape(4, 4)
    rot = pose[:3, :3]
    pos = pose[:3, 3]
    fwd = rot[:, 2]
    up = -rot[:, 1]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
    up = up / max(np.linalg.norm(up), 1e-12)
    look_distance = float(max(fallback_look_distance, 1e-3))
    try:
        cur_pos = np.asarray(cam.position, dtype=np.float64)
        cur_look = np.asarray(cam.look_at, dtype=np.float64)
        cur_dist = float(np.linalg.norm(cur_look - cur_pos))
        if np.isfinite(cur_dist) and cur_dist > 1e-3:
            look_distance = cur_dist
    except Exception:
        pass
    look_at = pos + fwd * look_distance
    try:
        if fov is not None and np.isfinite(float(fov)) and float(fov) > 1e-6:
            cam.fov = float(fov)
        cam.position = tuple(float(x) for x in pos.tolist())
        cam.look_at = tuple(float(x) for x in look_at.tolist())
        cam.up_direction = tuple(float(x) for x in up.tolist())
    except Exception:
        return


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Viser offline trajectory player.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to trajectories.npz")
    parser.add_argument("--video_path", type=str, required=True, help="Path to local copy of video.")
    parser.add_argument("--port", type=int, default=8080, help="Local port for Viser.")
    parser.add_argument("--sample_step", type=int, default=10, help="Video frame sampling step.")
    parser.add_argument("--point_cloud_npz", type=str, default=None, help="Optional dense point cloud NPZ.")
    parser.add_argument("--point_budget", type=int, default=30000, help="Max dense cloud points per frame.")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Min dense-point confidence probability in [0, 1] (OpenD4RT logits are sigmoid-mapped).",
    )
    args = parser.parse_args()

    print(f"Loading arrays from {args.npz_path}...")
    data = np.load(args.npz_path)

    camera_xyz_world = np.asarray(data["camera_xyz_world"], dtype=np.float32)
    identity_xyz_world = np.asarray(data["identity_xyz_world"], dtype=np.float32)
    t_ref0_cam = np.asarray(data["T_ref0_cam"], dtype=np.float32)
    k_seq = np.asarray(data["K"], dtype=np.float32)
    tracks_xyz_ref0 = np.asarray(data["tracks_xyz_ref0"], dtype=np.float32)
    tracks_visibility = np.asarray(data["tracks_visibility"], dtype=bool)
    identity_uv_px = np.asarray(data["identity_uv_px"], dtype=np.float32) if "identity_uv_px" in data.files else None

    if "coordinate_convention" in data.files:
        conv = str(np.asarray(data["coordinate_convention"]).item())
        print(f"NPZ coordinate convention: {conv}")

    point_cloud_bundle: PointCloudBundle | None = None
    if args.point_cloud_npz:
        print(f"Loading dense point cloud from {args.point_cloud_npz}...")
        point_cloud_bundle = load_point_cloud_npz(Path(args.point_cloud_npz))

    num_frames_data = int(camera_xyz_world.shape[0])
    print(f"Loading local video frames from {args.video_path}...")
    video_rgb = load_video_frames(Path(args.video_path), sample_step=args.sample_step, max_frames=num_frames_data)

    num_frames = min(int(video_rgb.shape[0]), num_frames_data)
    if point_cloud_bundle is not None and point_cloud_bundle.num_frames != num_frames:
        print(
            f"Warning: point cloud has {point_cloud_bundle.num_frames} frames; "
            f"trimming to {num_frames} to match video/trajectory."
        )
        point_cloud_bundle = PointCloudBundle(
            points_xyz_ref0=point_cloud_bundle.points_xyz_ref0[:num_frames],
            points_vis=point_cloud_bundle.points_vis[:num_frames],
            points_conf=point_cloud_bundle.points_conf[:num_frames],
            points_rgb=(
                point_cloud_bundle.points_rgb[:num_frames]
                if point_cloud_bundle.points_rgb is not None
                else None
            ),
            allowed_track_mask=point_cloud_bundle.allowed_track_mask,
            point_is_dynamic=point_cloud_bundle.point_is_dynamic,
            xyz_center=point_cloud_bundle.xyz_center,
            xyz_radius=point_cloud_bundle.xyz_radius,
            coordinate_convention=point_cloud_bundle.coordinate_convention,
            num_frames=num_frames,
        )

    video_rgb = video_rgb[:num_frames]
    camera_xyz_world = camera_xyz_world[:num_frames]
    identity_xyz_world = identity_xyz_world[:num_frames]
    t_ref0_cam = t_ref0_cam[:num_frames]
    k_seq = k_seq[:num_frames]
    tracks_xyz_ref0 = tracks_xyz_ref0[:, :num_frames]
    tracks_visibility = tracks_visibility[:, :num_frames]

    height, width = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    if "video_height" in data.files and "video_width" in data.files:
        npz_h, npz_w = int(data["video_height"]), int(data["video_width"])
        if (npz_h, npz_w) != (height, width):
            print(
                f"Warning: video resolution {width}x{height} differs from NPZ "
                f"({npz_w}x{npz_h}). Reprojection may be misaligned."
            )

    camera_xyz_viewer = transform_points_for_viewer(camera_xyz_world)
    identity_xyz_viewer = transform_points_for_viewer(identity_xyz_world)
    tracks_xyz_viewer = transform_points_for_viewer(
        tracks_xyz_ref0.reshape(-1, 3)
    ).reshape(tracks_xyz_ref0.shape)

    centroid_xyz_viewer: np.ndarray | None = None
    if point_cloud_bundle is not None:
        centroid_xyz_viewer = transform_points_for_viewer(point_cloud_bundle.xyz_center[None, :])[0]

    z_min_global, z_max_global = compute_global_depth_range(
        tracks_xyz_ref0=tracks_xyz_ref0,
        tracks_visibility=tracks_visibility,
        k_seq=k_seq,
        t_ref0_cam=t_ref0_cam,
        num_frames=num_frames,
    )

    all_pts = []
    for arr in (camera_xyz_viewer, identity_xyz_viewer):
        valid = np.isfinite(arr).all(axis=-1)
        if np.any(valid):
            all_pts.append(arr[valid])
    radius = (
        max(float(np.max(np.linalg.norm(all_pts[0] - all_pts[0].mean(axis=0), axis=1))), 0.5)
        if all_pts
        else 1.0
    )
    if point_cloud_bundle is not None:
        radius = max(radius, float(point_cloud_bundle.xyz_radius))

    server = viser.ViserServer(host="127.0.0.1", port=int(args.port))
    print(f"\nLocal server active: http://localhost:{args.port}")

    with server.gui.add_folder("Timeline", expand_by_default=True):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0)
        prev_btn = server.gui.add_button("Previous frame")
        next_btn = server.gui.add_button("Next frame")
        play_box = server.gui.add_checkbox("Play", initial_value=False)
        loop_box = server.gui.add_checkbox("Loop", initial_value=True)
        fps_slider = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=8)

    with server.gui.add_folder("Display", expand_by_default=True):
        show_camera_position = server.gui.add_checkbox("Show camera position", initial_value=True)
        show_centroid = server.gui.add_checkbox("Show object centroid", initial_value=False)
        show_frustum = server.gui.add_checkbox("Show camera frustum", initial_value=True)
        show_tracks = server.gui.add_checkbox("Show identity track points (3D)", initial_value=True)
        show_dense_motion_points = server.gui.add_checkbox("Show dense motion tracking points", initial_value=False)
        show_reprojection = server.gui.add_checkbox("Show 2D reprojection", initial_value=True)
        global_depth_scale = server.gui.add_checkbox("Global depth colormap scale", initial_value=False)
        point_radius_slider = server.gui.add_slider("Reprojection point radius", min=2, max=12, step=1, initial_value=5)

    show_dense_cloud = None
    cloud_mode = None
    show_cloud_background = None
    cloud_size_slider = None
    conf_threshold_slider = None
    if point_cloud_bundle is not None:
        with server.gui.add_folder("Dense point cloud", expand_by_default=True):
            show_dense_cloud = server.gui.add_checkbox("Show dense point cloud", initial_value=True)
            cloud_mode = server.gui.add_dropdown(
                "Point cloud mode",
                options=("3D", "4D"),
                initial_value="4D",
            )
            show_cloud_background = server.gui.add_checkbox("Show background points", initial_value=True)
            conf_threshold_slider = server.gui.add_slider(
                "Confidence threshold",
                min=0.0,
                max=1.0,
                step=0.05,
                initial_value=float(args.conf_threshold),
            )
            cloud_size_slider = server.gui.add_slider("Cloud point size", min=0.2, max=3.0, step=0.1, initial_value=1.0)

    frame_image = server.gui.add_image(video_rgb[0], label="rgb_frame")

    dynamic_handles: list[Any] = []
    static_cloud_handle: Any | None = None
    static_cloud_signature: tuple[Any, ...] | None = None
    render_lock = threading.Lock()

    def clear_dynamic() -> None:
        for h in dynamic_handles:
            try:
                h.remove()
            except Exception:
                pass
        dynamic_handles.clear()

    def clear_static_cloud() -> None:
        nonlocal static_cloud_handle, static_cloud_signature
        if static_cloud_handle is not None:
            try:
                static_cloud_handle.remove()
            except Exception:
                pass
            static_cloud_handle = None
        static_cloud_signature = None

    def add_dynamic(h: Any) -> None:
        if h is not None:
            dynamic_handles.append(h)

    def _add_static_trajectory(name: str, xyz: np.ndarray, color: tuple[int, int, int]) -> None:
        segs = _trajectory_line_segments(xyz[:num_frames])
        if segs is not None:
            seg_colors = np.tile(np.asarray(color, dtype=np.uint8), (segs.shape[0], 2, 1))
            server.scene.add_line_segments(
                f"/trajectories/{name}/path",
                points=segs.astype(np.float32),
                colors=seg_colors,
                line_width=3.0,
            )
        valid = np.isfinite(xyz[:num_frames]).all(axis=-1)
        if np.any(valid):
            server.scene.add_point_cloud(
                f"/trajectories/{name}/head",
                points=xyz[:num_frames][valid][-1][None, :].astype(np.float32),
                colors=np.asarray([color], dtype=np.uint8),
                point_size=max(radius * 0.02, 0.02),
                point_shape="sparkle",
            )

    _add_static_trajectory("camera", camera_xyz_viewer, (255, 64, 64))
    _add_static_trajectory("identity", identity_xyz_viewer, (64, 220, 100))

    def sync_all_clients_to_frame_camera(frame_idx: int) -> None:
        t = int(np.clip(int(frame_idx), 0, max(num_frames - 1, 0)))
        if not np.isfinite(t_ref0_cam[t]).all():
            return
        pose_viewer = transform_pose_for_viewer(t_ref0_cam[t])
        k_t = k_seq[t] if t < k_seq.shape[0] else k_seq[0]
        fov = float(_fov_from_k(k_t, height))
        look_dist = max(radius * 0.8, 0.5)
        for client in server.get_clients().values():
            jump_client_view_to_camera_pose(
                client,
                pose_viewer,
                fov=fov,
                fallback_look_distance=look_dist,
            )

    @server.on_client_connect
    def _on_client_connect(client: viser.ClientHandle) -> None:
        sync_all_clients_to_frame_camera(int(frame_slider.value))

    def _cloud_mode_value() -> Literal["3d", "4d"]:
        if cloud_mode is None:
            return "4d"
        return "3d" if str(cloud_mode.value).upper() == "3D" else "4d"

    def _render_dense_cloud(t: int) -> None:
        nonlocal static_cloud_handle, static_cloud_signature
        if point_cloud_bundle is None or show_dense_cloud is None or not bool(show_dense_cloud.value):
            clear_static_cloud()
            return

        mode = _cloud_mode_value()
        show_static = bool(show_cloud_background.value) if show_cloud_background is not None else True
        show_dynamic = bool(show_dense_motion_points.value)
        conf_threshold = (
            float(conf_threshold_slider.value) if conf_threshold_slider is not None else float(args.conf_threshold)
        )
        pts, cols = prepare_point_cloud_for_frame(
            point_cloud_bundle,
            t,
            mode=mode,
            show_static=show_static,
            show_dynamic=show_dynamic,
            point_budget=int(args.point_budget),
            conf_threshold=conf_threshold,
        )
        if pts.shape[0] <= 0:
            clear_static_cloud()
            return

        size_scale = float(cloud_size_slider.value) if cloud_size_slider is not None else 1.0
        if mode == "3d":
            signature = (
                mode,
                show_static,
                show_dynamic,
                round(conf_threshold, 3),
                int(args.point_budget),
                round(size_scale, 3),
                int(pts.shape[0]),
            )
            if static_cloud_signature != signature:
                clear_static_cloud()
                static_cloud_handle = add_point_cloud_to_viser_scene(
                    server.scene,
                    "/dense_cloud",
                    pts,
                    cols,
                    scene_radius=radius,
                    point_size_scale=size_scale,
                )
                static_cloud_signature = signature
        else:
            clear_static_cloud()
            add_dynamic(
                add_point_cloud_to_viser_scene(
                    server.scene,
                    "/dense_cloud",
                    pts,
                    cols,
                    scene_radius=radius,
                    point_size_scale=size_scale,
                )
            )

    def render() -> None:
        with render_lock:
            clear_dynamic()
            t = int(np.clip(int(frame_slider.value), 0, max(num_frames - 1, 0)))

            if bool(show_reprojection.value):
                display_frame = render_depth_colored_reprojection(
                    video_rgb[t],
                    t,
                    tracks_xyz_ref0=tracks_xyz_ref0,
                    tracks_visibility=tracks_visibility,
                    k_seq=k_seq,
                    t_ref0_cam=t_ref0_cam,
                    identity_uv_px=identity_uv_px,
                    z_min=z_min_global,
                    z_max=z_max_global,
                    use_global_depth=bool(global_depth_scale.value),
                    point_radius=int(point_radius_slider.value),
                )
            else:
                display_frame = video_rgb[t]
            frame_image.image = display_frame

            if bool(show_camera_position.value) and np.isfinite(camera_xyz_viewer[t]).all():
                add_dynamic(
                    server.scene.add_point_cloud(
                        "/current/camera",
                        points=camera_xyz_viewer[t][None, :].astype(np.float32),
                        colors=np.asarray([[255, 80, 80]], dtype=np.uint8),
                        point_size=max(radius * 0.03, 0.03),
                    )
                )
            if np.isfinite(identity_xyz_viewer[t]).all():
                add_dynamic(
                    server.scene.add_point_cloud(
                        "/current/identity",
                        points=identity_xyz_viewer[t][None, :].astype(np.float32),
                        colors=np.asarray([[80, 255, 120]], dtype=np.uint8),
                        point_size=max(radius * 0.03, 0.03),
                    )
                )

            if bool(show_centroid.value) and centroid_xyz_viewer is not None and np.isfinite(centroid_xyz_viewer).all():
                add_dynamic(
                    server.scene.add_point_cloud(
                        "/current/centroid",
                        points=centroid_xyz_viewer[None, :].astype(np.float32),
                        colors=np.asarray([[255, 220, 64]], dtype=np.uint8),
                        point_size=max(radius * 0.05, 0.04),
                        point_shape="sparkle",
                    )
                )

            if bool(show_tracks.value):
                vis_q = tracks_visibility[:, t] & np.isfinite(tracks_xyz_viewer[:, t]).all(axis=-1)
                if np.any(vis_q):
                    pts = tracks_xyz_viewer[vis_q, t]
                    add_dynamic(
                        server.scene.add_point_cloud(
                            "/current/identity_tracks",
                            points=pts.astype(np.float32),
                            colors=np.tile(np.asarray([[120, 255, 160]], dtype=np.uint8), (pts.shape[0], 1)),
                            point_size=max(radius * 0.012, 0.01),
                        )
                    )

            _render_dense_cloud(t)

            if bool(show_frustum.value) and np.isfinite(t_ref0_cam[t]).all():
                pose_viewer = transform_pose_for_viewer(t_ref0_cam[t])
                k_t = k_seq[t] if t < k_seq.shape[0] else k_seq[0]
                add_dynamic(
                    server.scene.add_camera_frustum(
                        "/current/camera_frustum",
                        fov=float(_fov_from_k(k_t, height)),
                        aspect=float(width) / float(max(height, 1)),
                        scale=max(radius * 0.15, 0.1),
                        color=(255, 255, 255),
                        image=video_rgb[t],
                        wxyz=_rotmat_to_wxyz(pose_viewer[:3, :3]),
                        position=tuple(float(x) for x in pose_viewer[:3, 3].tolist()),
                    )
                )

    def step_frame(delta: int) -> None:
        frame_slider.value = int(np.clip(int(frame_slider.value) + delta, 0, max(num_frames - 1, 0)))

    prev_btn.on_click(lambda _: step_frame(-1))
    next_btn.on_click(lambda _: step_frame(1))

    frame_slider.on_update(lambda _: render())
    show_camera_position.on_update(lambda _: render())
    show_centroid.on_update(lambda _: render())
    show_frustum.on_update(lambda _: render())
    show_tracks.on_update(lambda _: render())
    show_dense_motion_points.on_update(lambda _: render())
    show_reprojection.on_update(lambda _: render())
    global_depth_scale.on_update(lambda _: render())
    point_radius_slider.on_update(lambda _: render())
    if show_dense_cloud is not None:
        show_dense_cloud.on_update(lambda _: render())
    if cloud_mode is not None:
        cloud_mode.on_update(lambda _: render())
    if show_cloud_background is not None:
        show_cloud_background.on_update(lambda _: render())
    if cloud_size_slider is not None:
        cloud_size_slider.on_update(lambda _: render())
    if conf_threshold_slider is not None:
        conf_threshold_slider.on_update(lambda _: render())

    render()
    sync_all_clients_to_frame_camera(0)
    try:
        while True:
            if bool(play_box.value) and num_frames > 1:
                step = 1
                t_cur = int(frame_slider.value)
                t_next = t_cur + step
                if t_next >= num_frames:
                    if bool(loop_box.value):
                        t_next = 0
                    else:
                        play_box.value = False
                        t_next = t_cur
                if t_next != t_cur:
                    frame_slider.value = t_next
                time.sleep(max(1.0 / float(fps_slider.value), 1e-3))
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("Closing down viewer...")


if __name__ == "__main__":
    main()
