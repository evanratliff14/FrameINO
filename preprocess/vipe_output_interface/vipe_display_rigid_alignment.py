#!/usr/bin/env python3
"""
Viser demo for rigid alignment inside a ViPE world reconstruction.

Keyframe-only: world PCD + camera frustums, posed SAM3D meshes, per-instance
mesh XOR masked-PCD toggles, and side-by-side GT vs reprojected 2D.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import viser
import viser.transforms as tf
from PIL import Image

from vipe_view_local import (
    VideoBundle,
    pinhole_rays,
    resolve_device,
    unproject_depth_to_camera_pcd,
)

logger = logging.getLogger(__name__)

# Distinct tints for instance meshes (RGB uint8).
_INSTANCE_PALETTE = np.array(
    [
        [255, 96, 96],
        [96, 200, 255],
        [120, 220, 120],
        [255, 180, 64],
        [200, 120, 255],
        [255, 220, 64],
        [64, 220, 200],
        [220, 120, 160],
    ],
    dtype=np.uint8,
)


def _rt_to_wxyz_position(rt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rt = np.asarray(rt, dtype=np.float64)
    wxyz = tf.SO3.from_matrix(rt[:3, :3]).wxyz
    position = rt[:3, 3].astype(np.float64)
    return wxyz, position


def _instance_color(i: int) -> np.ndarray:
    return _INSTANCE_PALETTE[int(i) % len(_INSTANCE_PALETTE)]


def _tint_mesh_vertices(mesh: Any, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices, faces, vertex_colors) for viser mesh_simple."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    colors = np.tile(np.asarray(rgb, dtype=np.uint8)[None, :], (verts.shape[0], 1))
    return verts, faces, colors


def _cam_to_world(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    return (points_cam.astype(np.float64) @ R.T) + t[None, :]


def _world_to_cam(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    return (points_world.astype(np.float64) - t[None, :]) @ R


def reproject_scene_image(
    *,
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
    points_world: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """Z-buffer project world points into an RGB image (nearest depth wins)."""
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if points_world.size == 0:
        return out
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    cam = _world_to_cam(points_world, c2w)
    z = cam[:, 2]
    valid = z > 1e-4
    u = fx * (cam[:, 0] / np.maximum(z, 1e-8)) + cx
    v = fy * (cam[:, 1] / np.maximum(z, 1e-8)) + cy
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    in_img = valid & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(in_img):
        return out
    ui, vi, z = ui[in_img], vi[in_img], z[in_img]
    cols = np.asarray(colors, dtype=np.uint8)[in_img]
    # Farther first so nearer overwrites.
    order = np.argsort(-z)
    ui, vi, z, cols = ui[order], vi[order], z[order], cols[order]
    for uu, vv, zz, c in zip(ui, vi, z, cols):
        if zz < zbuf[vv, uu]:
            zbuf[vv, uu] = zz
            out[vv, uu] = c
    return out


@dataclass
class KeyframeCache:
    frame_idx: int
    c2w: np.ndarray
    intrinsics: np.ndarray  # fx, fy, cx, cy
    rgb: np.ndarray
    points_world: np.ndarray  # [P, 3]
    colors: np.ndarray  # [P, 3] uint8
    instance_ids: np.ndarray  # [P]
    fov: float


@dataclass
class SceneHandles:
    frame_handle: Any
    frustum_handle: Any
    pcd_handle: Any | None
    cache: KeyframeCache


@dataclass
class MeshHandles:
    instance_id: int
    frame_handle: Any
    mesh_handle: Any | None
    vertices_local: np.ndarray
    vertex_colors: np.ndarray


@dataclass
class ClientState:
    client: viser.ClientHandle
    scene_frames: list[SceneHandles] = field(default_factory=list)
    mesh_handles: list[MeshHandles] = field(default_factory=list)
    mode_by_id: dict[int, str] = field(default_factory=dict)  # "mesh" | "pcd"
    current_k: int = 0
    gui_timestep: Any = None
    gui_frame_label: Any = None
    gui_gt: Any = None
    gui_reproj: Any = None
    gui_point_size: Any = None
    gui_frustum_size: Any = None


def display_rigid_alignment(
    base_path: Path,
    indices: list[int],
    Rt: list[np.ndarray],
    meshes: list,
    instance_ids: list[int],
    *,
    host: str = "127.0.0.1",
    port: int = 20541,
    spatial_subsample: int = 2,
) -> None:
    """
    Launch a paused viser demo of rigid alignment in the ViPE world scene.
    """
    base_path = Path(base_path)
    indices = [int(i) for i in indices]
    k = len(indices)
    if k == 0:
        raise ValueError("indices must be non-empty")
    if not (len(Rt) == len(meshes) == len(instance_ids)):
        raise ValueError("Rt, meshes, and instance_ids must have equal length")
    for i, rt in enumerate(Rt):
        rt = np.asarray(rt, dtype=np.float64)
        if rt.shape != (k, 4, 4):
            raise ValueError(f"Rt[{i}] expected shape ({k}, 4, 4), got {rt.shape}")
        Rt[i] = rt

    device = resolve_device("auto")
    logger.info("Loading ViPE bundle from %s (device=%s)", base_path, device)
    bundle = VideoBundle.load(base_path)

    # Precompute keyframe caches in world coordinates.
    caches: list[KeyframeCache] = []
    rays: np.ndarray | None = None
    first_up: np.ndarray | None = None
    for t in indices:
        c2w = bundle.camera.get_c2w([t])[0]
        intr = bundle.camera.get_intrinsics([t])[0]
        depth = bundle.depth.get_depth([t])[0]
        rgb = bundle.rgb[t] if t < len(bundle.rgb) else None
        id_map = bundle.masks.get_id_map([t])[0]
        if c2w is None or intr is None:
            raise RuntimeError(f"Missing camera for frame {t}")
        if rgb is None:
            rgb = np.full((bundle.height, bundle.width, 3), 180, dtype=np.uint8)
        else:
            rgb = np.asarray(rgb, dtype=np.uint8)
        fx, fy, cx, cy = map(float, intr)
        fov = float(2.0 * np.arctan2(bundle.height / 2.0, fy))
        if first_up is None:
            first_up = c2w[:3, 1].astype(np.float64)
        if rays is None:
            rays = pinhole_rays(
                bundle.height, bundle.width, fx, fy, cx, cy, spatial_subsample, device
            )
        if depth is not None:
            pcd_cam, depth_mask = unproject_depth_to_camera_pcd(
                depth, rays, spatial_subsample, device
            )
            sampled_rgb = rgb[::spatial_subsample, ::spatial_subsample]
            if id_map is not None:
                sampled_ids = id_map[::spatial_subsample, ::spatial_subsample]
            else:
                sampled_ids = np.zeros(sampled_rgb.shape[:2], dtype=np.uint8)
            pcd_flat = pcd_cam.reshape(-1, 3)
            rgb_flat = sampled_rgb.reshape(-1, 3)
            ids_flat = sampled_ids.reshape(-1).astype(np.int64)
            mask_flat = depth_mask.reshape(-1)
            pcd_flat = pcd_flat[mask_flat]
            rgb_flat = rgb_flat[mask_flat]
            ids_flat = ids_flat[mask_flat]
            pts_w = _cam_to_world(pcd_flat, c2w).astype(np.float32)
        else:
            pts_w = np.zeros((0, 3), dtype=np.float32)
            rgb_flat = np.zeros((0, 3), dtype=np.uint8)
            ids_flat = np.zeros((0,), dtype=np.int64)

        caches.append(
            KeyframeCache(
                frame_idx=t,
                c2w=np.asarray(c2w, dtype=np.float64),
                intrinsics=np.asarray(intr, dtype=np.float64),
                rgb=rgb,
                points_world=pts_w,
                colors=rgb_flat.astype(np.uint8),
                instance_ids=ids_flat,
                fov=fov,
            )
        )

    phrases = {iid: bundle.masks.phrase_for(iid) for iid in instance_ids}
    default_modes = {
        int(iid): ("mesh" if meshes[j] is not None else "pcd")
        for j, iid in enumerate(instance_ids)
    }

    server = viser.ViserServer(host=host, port=port, verbose=False)
    logger.info("Rigid-alignment demo: http://%s:%d  (keyframes=%d)", host, port, k)
    clients: dict[int, ClientState] = {}

    def _filtered_pcd(cache: KeyframeCache, modes: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
        hide_ids = {iid for iid, mode in modes.items() if mode == "mesh"}
        if not hide_ids or cache.instance_ids.size == 0:
            return cache.points_world, cache.colors
        keep = ~np.isin(cache.instance_ids, list(hide_ids))
        return cache.points_world[keep], cache.colors[keep]

    def _reproject_for_state(state: ClientState) -> np.ndarray:
        cache = caches[state.current_k]
        fx, fy, cx, cy = cache.intrinsics
        pts, cols = _filtered_pcd(cache, state.mode_by_id)
        extra_pts = []
        extra_cols = []
        for mh in state.mesh_handles:
            if state.mode_by_id.get(mh.instance_id, "pcd") != "mesh":
                continue
            if mh.vertices_local.size == 0:
                continue
            rt = Rt[instance_ids.index(mh.instance_id)][state.current_k]
            R = rt[:3, :3]
            t = rt[:3, 3]
            world = (mh.vertices_local.astype(np.float64) @ R.T) + t[None, :]
            # Subsample mesh verts for speed.
            step = max(1, world.shape[0] // 20000)
            world = world[::step]
            extra_pts.append(world.astype(np.float32))
            extra_cols.append(
                np.tile(mh.vertex_colors[0:1], (world.shape[0], 1)).astype(np.uint8)
            )
        if extra_pts:
            pts = np.concatenate([pts] + extra_pts, axis=0) if pts.size else np.concatenate(extra_pts, axis=0)
            cols = np.concatenate([cols] + extra_cols, axis=0) if cols.size else np.concatenate(extra_cols, axis=0)
        # Downscale target for GUI.
        h, w = cache.rgb.shape[:2]
        scale = min(1.0, 480.0 / max(h, w))
        th, tw = max(1, int(h * scale)), max(1, int(w * scale))
        sfx, sfy, scx, scy = fx * scale, fy * scale, cx * scale, cy * scale
        return reproject_scene_image(
            height=th,
            width=tw,
            fx=sfx,
            fy=sfy,
            cx=scx,
            cy=scy,
            c2w=cache.c2w,
            points_world=pts,
            colors=cols,
        )

    def _gt_thumb(cache: KeyframeCache) -> np.ndarray:
        img = Image.fromarray(cache.rgb)
        img.thumbnail((480, 480), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.uint8)

    def _apply_keyframe(state: ClientState, k_idx: int) -> None:
        k_idx = int(np.clip(k_idx, 0, len(caches) - 1))
        prev = state.current_k
        state.current_k = k_idx
        with state.client.atomic():
            if state.scene_frames:
                state.scene_frames[prev].frame_handle.visible = False
                state.scene_frames[prev].frustum_handle.visible = False
                if state.scene_frames[prev].pcd_handle is not None:
                    state.scene_frames[prev].pcd_handle.visible = False
                state.scene_frames[k_idx].frame_handle.visible = True
                state.scene_frames[k_idx].frustum_handle.visible = True
            _update_pcd_and_meshes(state)
            if state.gui_frame_label is not None:
                state.gui_frame_label.value = f"frame={caches[k_idx].frame_idx}  (k={k_idx}/{len(caches)-1})"
            if state.gui_gt is not None:
                state.gui_gt.image = _gt_thumb(caches[k_idx])
            if state.gui_reproj is not None:
                state.gui_reproj.image = _reproject_for_state(state)

    def _update_pcd_and_meshes(state: ClientState) -> None:
        cache = caches[state.current_k]
        pts, cols = _filtered_pcd(cache, state.mode_by_id)
        sh = state.scene_frames[state.current_k]
        point_size = float(state.gui_point_size.value) if state.gui_point_size is not None else 0.01
        if sh.pcd_handle is not None:
            sh.pcd_handle.remove()
            sh.pcd_handle = None
        if pts.shape[0] > 0:
            sh.pcd_handle = state.client.scene.add_point_cloud(
                name=f"/frames/t{cache.frame_idx}/point_cloud",
                points=pts.astype(np.float32),
                colors=cols.astype(np.uint8),
                point_size=point_size,
                point_shape="rounded",
            )
            sh.pcd_handle.visible = True
        for mh in state.mesh_handles:
            mode = state.mode_by_id.get(mh.instance_id, "pcd")
            show = mode == "mesh" and mh.mesh_handle is not None
            if mh.frame_handle is not None:
                mh.frame_handle.visible = show
            if mh.mesh_handle is not None:
                mh.mesh_handle.visible = show
            if show:
                rt = Rt[instance_ids.index(mh.instance_id)][state.current_k]
                wxyz, pos = _rt_to_wxyz_position(rt)
                mh.frame_handle.wxyz = wxyz
                mh.frame_handle.position = pos

    def _reset_view(state: ClientState) -> None:
        cache = caches[state.current_k]
        c2w = cache.c2w
        # Place viewer slightly behind the camera, looking along +Z_cam.
        eye = c2w[:3, 3] - 0.35 * c2w[:3, 2]
        target = c2w[:3, 3] + 1.5 * c2w[:3, 2]
        up = -c2w[:3, 1]
        state.client.camera.position = tuple(float(x) for x in eye.tolist())
        state.client.camera.look_at = tuple(float(x) for x in target.tolist())
        state.client.camera.up_direction = tuple(float(x) for x in up.tolist())
        state.client.camera.fov = float(cache.fov)

    def _build_client(client: viser.ClientHandle) -> ClientState:
        state = ClientState(client=client, mode_by_id=dict(default_modes))
        if first_up is not None:
            client.scene.set_up_direction(-first_up)

        with client.gui.add_folder("Scene"):
            state.gui_point_size = client.gui.add_slider(
                "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.012
            )
            state.gui_frustum_size = client.gui.add_slider(
                "Frustum scale", min=0.02, max=0.5, step=0.01, initial_value=0.12
            )

            @state.gui_point_size.on_update
            async def _(_) -> None:
                _update_pcd_and_meshes(state)

            @state.gui_frustum_size.on_update
            async def _(_) -> None:
                for sh in state.scene_frames:
                    sh.frustum_handle.scale = float(state.gui_frustum_size.value)

            reset_btn = client.gui.add_button("Reset view to keyframe camera")

            @reset_btn.on_click
            async def _(_) -> None:
                _reset_view(state)

        with client.gui.add_folder("Playback"):
            state.gui_timestep = client.gui.add_slider(
                "Keyframe", min=0, max=max(k - 1, 0), step=1, initial_value=0
            )
            state.gui_frame_label = client.gui.add_text(
                "Frame", f"frame={caches[0].frame_idx}  (k=0/{k-1})"
            )
            controls = client.gui.add_button_group("Control", options=["Prev", "Next"])
            # Explicitly no autoplay / FPS loop — paused on load.

            @controls.on_click
            async def _(_) -> None:
                cur = int(state.gui_timestep.value)
                if controls.value == "Prev":
                    state.gui_timestep.value = (cur - 1) % k
                else:
                    state.gui_timestep.value = (cur + 1) % k

            @state.gui_timestep.on_update
            async def _(_) -> None:
                _apply_keyframe(state, int(state.gui_timestep.value))

        with client.gui.add_folder("Instances"):
            client.gui.add_markdown(
                "Per instance: **mesh** hides that mask's PCD and shows SAM3D; "
                "**pcd** shows masked points and hides the mesh."
            )
            for j, iid in enumerate(instance_ids):
                label = f"{iid}: {phrases.get(iid, 'entity')}"
                options = ("mesh", "pcd") if meshes[j] is not None else ("pcd",)
                initial = state.mode_by_id[int(iid)]
                if initial not in options:
                    initial = options[0]
                    state.mode_by_id[int(iid)] = initial
                dropdown = client.gui.add_dropdown(label, options=options, initial_value=initial)

                @dropdown.on_update
                async def _(event=None, _iid=int(iid), _dd=dropdown) -> None:
                    state.mode_by_id[_iid] = str(_dd.value)
                    _update_pcd_and_meshes(state)
                    if state.gui_reproj is not None:
                        state.gui_reproj.image = _reproject_for_state(state)

        with client.gui.add_folder("2D keyframe"):
            client.gui.add_markdown("Left: GT video &nbsp;&nbsp; Right: reprojected scene")
            state.gui_gt = client.gui.add_image(_gt_thumb(caches[0]), label="GT")
            state.gui_reproj = client.gui.add_image(
                np.zeros((64, 64, 3), dtype=np.uint8), label="Reprojected"
            )

        # Build keyframe scene nodes (all hidden except first).
        for cache in caches:
            wxyz, pos = _rt_to_wxyz_position(cache.c2w)
            frame = client.scene.add_frame(
                f"/frames/t{cache.frame_idx}",
                axes_length=0.08,
                axes_radius=0.006,
                wxyz=wxyz,
                position=pos,
                visible=False,
            )
            thumb = Image.fromarray(cache.rgb)
            thumb.thumbnail((240, 240), Image.Resampling.LANCZOS)
            h, w = cache.rgb.shape[:2]
            frustum = client.scene.add_camera_frustum(
                f"/frames/t{cache.frame_idx}/frustum",
                fov=cache.fov,
                aspect=w / max(h, 1),
                scale=float(state.gui_frustum_size.value),
                image=np.array(thumb),
                visible=False,
            )
            state.scene_frames.append(
                SceneHandles(
                    frame_handle=frame,
                    frustum_handle=frustum,
                    pcd_handle=None,
                    cache=cache,
                )
            )

        # Mesh nodes at identity; poses updated per keyframe.
        for j, iid in enumerate(instance_ids):
            mesh = meshes[j]
            color = _instance_color(j)
            frame = client.scene.add_frame(
                f"/meshes/{iid}",
                axes_length=0.04,
                axes_radius=0.003,
                visible=False,
            )
            mesh_handle = None
            verts = np.zeros((0, 3), dtype=np.float32)
            vcols = color[None, :].copy()
            if mesh is not None:
                verts, faces, vcols = _tint_mesh_vertices(mesh, color)
                mesh_handle = client.scene.add_mesh_simple(
                    f"/meshes/{iid}/geom",
                    vertices=verts,
                    faces=faces,
                    color=(float(color[0]) / 255.0, float(color[1]) / 255.0, float(color[2]) / 255.0),
                    visible=False,
                )
            state.mesh_handles.append(
                MeshHandles(
                    instance_id=int(iid),
                    frame_handle=frame,
                    mesh_handle=mesh_handle,
                    vertices_local=verts,
                    vertex_colors=vcols,
                )
            )

        _apply_keyframe(state, 0)
        _reset_view(state)
        return state

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        clients[client.client_id] = _build_client(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle) -> None:
        clients.pop(client.client_id, None)

    while True:
        time.sleep(10.0)
