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
from pathlib import Path
from typing import Any

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= int(max_frames):
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    frames = np.stack(frames, axis=0)
    frames = frames[:: sample_step, :,:,:]
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
    parser.add_argument("--sample_step", type=int, default=10, help="Local port for Viser.")
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

    num_frames_data = int(camera_xyz_world.shape[0])
    print(f"Loading local video frames from {args.video_path}...")
    video_rgb = load_video_frames(Path(args.video_path), sample_step = args.sample_step, max_frames=num_frames_data)

    num_frames = min(int(video_rgb.shape[0]), num_frames_data)
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

    # Viewer-space trajectories (display only).
    camera_xyz_viewer = transform_points_for_viewer(camera_xyz_world)
    identity_xyz_viewer = transform_points_for_viewer(identity_xyz_world)
    tracks_xyz_viewer = transform_points_for_viewer(
        tracks_xyz_ref0.reshape(-1, 3)
    ).reshape(tracks_xyz_ref0.shape)

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

    server = viser.ViserServer(host="127.0.0.1", port=int(args.port))
    print(f"\nLocal server active: http://localhost:{args.port}")

    frame_slider = server.gui.add_slider("Frame", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0)
    show_frustum = server.gui.add_checkbox("Show camera frustum", initial_value=True)
    show_tracks = server.gui.add_checkbox("Show identity track points (3D)", initial_value=True)
    show_reprojection = server.gui.add_checkbox("Show 2D reprojection", initial_value=True)
    global_depth_scale = server.gui.add_checkbox("Global depth colormap scale", initial_value=False)
    point_radius_slider = server.gui.add_slider("Reprojection point radius", min=2, max=12, step=1, initial_value=5)

    dynamic_handles: list[Any] = []
    render_lock = threading.Lock()

    def clear_dynamic() -> None:
        for h in dynamic_handles:
            try:
                h.remove()
            except Exception:
                pass
        dynamic_handles.clear()

    def add_dynamic(h: Any) -> None:
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

    frame_image = server.gui.add_image(video_rgb[0], label="rgb_frame")

    def sync_all_clients_to_frame_camera(frame_idx: int) -> None:
        """Set each connected client's viewport to the estimated camera at frame_idx."""
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

            if np.isfinite(camera_xyz_viewer[t]).all():
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

    frame_slider.on_update(lambda _: render())
    show_frustum.on_update(lambda _: render())
    show_tracks.on_update(lambda _: render())
    show_reprojection.on_update(lambda _: render())
    global_depth_scale.on_update(lambda _: render())
    point_radius_slider.on_update(lambda _: render())

    render()
    sync_all_clients_to_frame_camera(0)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Closing down viewer...")


if __name__ == "__main__":
    main()
