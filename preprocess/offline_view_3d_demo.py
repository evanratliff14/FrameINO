import argparse
import math
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import viser

def load_video_frames(video_path: Path) -> np.ndarray:
    """Load video frames directly using OpenCV on your laptop."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB (Viser format)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return np.stack(frames, axis=0)

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
    segments = []
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

def main():
    parser = argparse.ArgumentParser(description="Local Viser offline trajectory player.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to downloaded trajectories.npz")
    parser.add_argument("--video_path", type=str, required=True, help="Path to local copy of video.")
    parser.add_argument("--port", type=int, default=8080, help="Local port to spin up Viser.")
    args = parser.parse_args()

    # 1. Load data
    print(f"Loading arrays from {args.npz_path}...")
    data = np.load(args.npz_path)
    
    camera_xyz_world = data["camera_xyz_world"]
    identity_xyz_world = data["identity_xyz_world"]
    t_ref0_cam = data["T_ref0_cam"]
    k_seq = data["K"]
    tracks_xyz_ref0 = data["tracks_xyz_ref0"]
    tracks_visibility = data["tracks_visibility"]

    print(f"Loading local video frames from {args.video_path}...")
    video_rgb = load_video_frames(Path(args.video_path))
    
    # Cap video frames to match data count if needed
    num_frames = min(int(video_rgb.shape[0]), int(camera_xyz_world.shape[0]))
    video_rgb = video_rgb[:num_frames]
    height, width = int(video_rgb.shape[1]), int(video_rgb.shape[2])

    # 2. Determine scene scale
    all_pts = []
    for arr in (camera_xyz_world, identity_xyz_world):
        valid = np.isfinite(arr).all(axis=-1)
        if np.any(valid):
            all_pts.append(arr[valid])
    radius = max(float(np.max(np.linalg.norm(all_pts[0] - all_pts[0].mean(axis=0), axis=1))), 0.5) if all_pts else 1.0

    # 3. Spin up local Viser server
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    print(f"\n🚀 Local server active! Open your browser at: http://localhost:{args.port}")

    # UI Widgets
    frame_slider = server.gui.add_slider("Frame", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0)
    show_frustum = server.gui.add_checkbox("Show camera frustum", initial_value=True)
    show_tracks = server.gui.add_checkbox("Show identity track points", initial_value=True)

    dynamic_handles = []
    render_lock = threading.Lock()

    def clear_dynamic():
        for h in dynamic_handles:
            try: h.remove()
            except Exception: pass
        dynamic_handles.clear()

    def _add_static_trajectory(name: str, xyz: np.ndarray, color: tuple[int, int, int]):
        segs = _trajectory_line_segments(xyz[:num_frames])
        if segs is not None:
            seg_colors = np.tile(np.asarray(color, dtype=np.uint8), (segs.shape[0], 2, 1))
            server.scene.add_line_segments(
                f"/trajectories/{name}/path", points=segs, colors=seg_colors, line_width=3.0
            )
        valid = np.isfinite(xyz[:num_frames]).all(axis=-1)
        if np.any(valid):
            server.scene.add_point_cloud(
                f"/trajectories/{name}/head",
                points=xyz[:num_frames][valid][-1][None, :],
                colors=np.asarray([color], dtype=np.uint8),
                point_size=max(radius * 0.02, 0.02),
                point_shape="sparkle",
            )

    _add_static_trajectory("camera", camera_xyz_world, (255, 64, 64))
    _add_static_trajectory("identity", identity_xyz_world, (64, 220, 100))

    frame_image = server.gui.add_image(video_rgb[0], label="rgb_frame")

    def render():
        with render_lock:
            clear_dynamic()
            t = int(frame_slider.value)
            frame_image.image = video_rgb[t]

            if np.isfinite(camera_xyz_world[t]).all():
                dynamic_handles.append(server.scene.add_point_cloud(
                    "/current/camera", points=camera_xyz_world[t][None, :],
                    colors=np.asarray([[255, 80, 80]], dtype=np.uint8), point_size=max(radius * 0.03, 0.03)
                ))
            if np.isfinite(identity_xyz_world[t]).all():
                dynamic_handles.append(server.scene.add_point_cloud(
                    "/current/identity", points=identity_xyz_world[t][None, :],
                    colors=np.asarray([[80, 255, 120]], dtype=np.uint8), point_size=max(radius * 0.03, 0.03)
                ))

            if bool(show_tracks.value):
                vis_q = tracks_visibility[:, t] & np.isfinite(tracks_xyz_ref0[:, t]).all(axis=-1)
                if np.any(vis_q):
                    pts = tracks_xyz_ref0[vis_q, t]
                    dynamic_handles.append(server.scene.add_point_cloud(
                        "/current/identity_tracks", points=pts,
                        colors=np.tile(np.asarray([[120, 255, 160]], dtype=np.uint8), (pts.shape[0], 1)),
                        point_size=max(radius * 0.012, 0.01)
                    ))

            if bool(show_frustum.value) and np.isfinite(t_ref0_cam[t]).all():
                pose = t_ref0_cam[t]
                k_t = k_seq[t] if t < k_seq.shape[0] else k_seq[0]
                dynamic_handles.append(server.scene.add_camera_frustum(
                    "/current/camera_frustum", fov=float(_fov_from_k(k_t, height)),
                    aspect=float(width) / float(max(height, 1)), scale=max(radius * 0.15, 0.1),
                    color=(255, 255, 255), image=video_rgb[t], wxyz=_rotmat_to_wxyz(pose[:3, :3]),
                    position=tuple(pose[:3, 3].tolist())
                ))

    frame_slider.on_update(lambda _: render())
    show_frustum.on_update(lambda _: render())
    show_tracks.on_update(lambda _: render())
    
    render() # Initial draw
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        print("Closing down viewer...")

if __name__ == "__main__":
    main()