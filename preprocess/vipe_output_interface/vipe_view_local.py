#!/usr/bin/env python3
"""
Local Viser viewer for existing ViPE artifacts (pose / rgb / depth / flow / masks).

Standalone: does NOT import the ``vipe`` package (no CUDA extensions).
Uses Torch on Apple MPS when available, otherwise CPU — never CUDA.

Example:
  python preprocess/vipe_output_interface/vipe_view_local.py \\
    --base_path preprocess/vipe/vipe_results_flow \\
    --port 20540

    python preprocess/vipe_output_interface/vipe_view_local.py --base_path preprocess/vipe/vipe_results_flow --port 20540
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import viser
import viser.transforms as tf
from PIL import Image

from vipe_camera import Camera
from vipe_depth import VipeDepth
from vipe_dense_flow import FLOW_RES_SCALE, DenseFlow
from vipe_io import modality_exists, read_rgb_frames
from vipe_masks import InstanceMask, VipeMasks
from vipe_optical_flow import FlowResult, flow_arrows_for_src

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

YELLOW = np.array([255.0, 220.0, 64.0], dtype=np.float32)
YELLOW_BLEND = 0.55

_DEVICE: torch.device | None = None


def resolve_device(prefer: str = "auto") -> torch.device:
    """Resolve Torch device: ``mps`` if available, else ``cpu``. Never returns ``cuda``."""
    prefer = prefer.lower().strip()
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {prefer!r} (use auto|mps|cpu)")


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = resolve_device("auto")
    return _DEVICE


def _jet_rgb(t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    r = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def reliable_depth_mask_range(
    depth: torch.Tensor,
    window_size: int = 5,
    ratio_thresh: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert window_size % 2 == 1, "Window size must be odd."
    depth_unsq = depth.unsqueeze(0).unsqueeze(0)
    local_max = F.max_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_min = -F.max_pool2d(-depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    local_mean = F.avg_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
    ratio = (local_max - local_min) / (local_mean + eps)
    ratio = ratio.squeeze(0).squeeze(0)
    return (ratio < ratio_thresh) & (depth > 0)


def pinhole_rays(
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    spatial_subsample: int,
    device: torch.device,
) -> np.ndarray:
    v = torch.arange(0, height, spatial_subsample, device=device, dtype=torch.float32)
    u = torch.arange(0, width, spatial_subsample, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    x = (uu - float(cx)) / float(fx)
    y = (vv - float(cy)) / float(fy)
    z = torch.ones_like(x)
    rays = torch.stack([x, y, z], dim=-1)
    return rays.detach().cpu().numpy().astype(np.float32)


def unproject_depth_to_camera_pcd(
    depth: np.ndarray,
    rays: np.ndarray,
    spatial_subsample: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    depth_t = torch.from_numpy(np.asarray(depth, dtype=np.float32)).to(device)
    mask_t = reliable_depth_mask_range(depth_t)
    depth_s = depth_t[::spatial_subsample, ::spatial_subsample]
    mask_s = mask_t[::spatial_subsample, ::spatial_subsample]
    depth_np = depth_s.detach().cpu().numpy()
    mask_np = mask_s.detach().cpu().numpy().astype(bool)
    pcd = rays * depth_np[..., None]
    return pcd.astype(np.float32), mask_np


def blend_entity_colors(
    base_rgb: np.ndarray,
    instance_ids: np.ndarray,
    selected_ids: set[int],
) -> np.ndarray:
    colors = np.asarray(base_rgb, dtype=np.uint8).copy()
    if not selected_ids or instance_ids.size == 0:
        return colors
    hit = np.isin(instance_ids, list(selected_ids))
    if not np.any(hit):
        return colors
    blended = (1.0 - YELLOW_BLEND) * colors[hit].astype(np.float32) + YELLOW_BLEND * YELLOW
    colors[hit] = np.clip(blended, 0, 255).astype(np.uint8)
    return colors


def _points_in_selected_instances(
    u: np.ndarray,
    v: np.ndarray,
    instances: list[InstanceMask],
    selected_ids: set[int],
) -> np.ndarray:
    """OR membership across selected ``InstanceMask``s at pixel coords ``(u, v)``."""
    keep = np.zeros((len(u),), dtype=bool)
    for im in instances:
        if im.instance_id in selected_ids:
            keep |= im.contains_uv(u, v)
    return keep


def id_map_from_instances(
    instances: list[InstanceMask] | None,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Compose a packed uint8 id map from per-instance binary masks."""
    if not instances:
        return None
    out = np.zeros((height, width), dtype=np.uint8)
    for im in instances:
        out[im.mask] = np.uint8(im.instance_id)
    return out


def unproject_uv_depth_world(
    uv_xy: np.ndarray,
    depth_hw: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
) -> np.ndarray:
    uv = np.asarray(uv_xy, dtype=np.float64)
    depth = np.asarray(depth_hw, dtype=np.float64)
    h, w = depth.shape
    u = np.clip(np.rint(uv[:, 0]), 0, w - 1).astype(np.int64)
    v = np.clip(np.rint(uv[:, 1]), 0, h - 1).astype(np.int64)
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 1e-4)
    x = (uv[:, 0] - float(cx)) / max(float(fx), 1e-8) * z
    y = (uv[:, 1] - float(cy)) / max(float(fy), 1e-8) * z
    cam = np.stack([x, y, z], axis=-1)
    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    world = (cam @ R.T) + t[None, :]
    world = world.astype(np.float32)
    world[~valid] = np.nan
    return world


def flow_segments_world(
    flow_result: FlowResult,
    camera: Camera,
    depth: VipeDepth,
    masks: VipeMasks,
    *,
    selected_entity_ids: set[int],
    certainty_thresh: float = 0.1,
    max_arrows_per_src: int = 2000,
    arrow_stride: int = 1,
    require_src_mask: bool = True,
) -> dict[int, np.ndarray]:
    """
    Build world-space flow line segments keyed by **src** frame index.

    Returns ``{src: segments}`` where ``segments`` is float32 ``[M, 2, 3]``.
    """
    out: dict[int, np.ndarray] = {}

    for edge in flow_result.edges:
        src, dst = int(edge.src), int(edge.dst)
        src_xy, dst_xy, _w = flow_arrows_for_src(
            edge.tensor,
            certainty_thresh=certainty_thresh,
            stride=max(1, int(arrow_stride)),
        )
        if src_xy.shape[0] == 0:
            continue

        if require_src_mask:
            src_instances = masks.get_masks([src])[0]
            if not src_instances or not selected_entity_ids:
                continue
            keep = _points_in_selected_instances(
                src_xy[:, 0],
                src_xy[:, 1],
                src_instances,
                selected_entity_ids,
            )
            src_xy = src_xy[keep]
            dst_xy = dst_xy[keep]
            if src_xy.shape[0] == 0:
                continue

        src_depth, dst_depth = depth.get_depth([src, dst])
        if src_depth is None or dst_depth is None:
            continue

        pose_s, pose_d = camera.get_c2w([src, dst])
        intr_s, intr_d = camera.get_intrinsics([src, dst])
        if pose_s is None or pose_d is None or intr_s is None or intr_d is None:
            continue

        fx_s, fy_s, cx_s, cy_s = intr_s
        fx_d, fy_d, cx_d, cy_d = intr_d
        p0 = unproject_uv_depth_world(
            src_xy, src_depth, float(fx_s), float(fy_s), float(cx_s), float(cy_s), pose_s
        )
        p1 = unproject_uv_depth_world(
            dst_xy, dst_depth, float(fx_d), float(fy_d), float(cx_d), float(cy_d), pose_d
        )
        both = np.isfinite(p0).all(axis=-1) & np.isfinite(p1).all(axis=-1)
        p0, p1 = p0[both], p1[both]
        if p0.shape[0] == 0:
            continue

        if p0.shape[0] > max_arrows_per_src:
            pick = np.linspace(0, p0.shape[0] - 1, num=int(max_arrows_per_src), dtype=np.int64)
            p0, p1 = p0[pick], p1[pick]

        out[src] = np.stack([p0, p1], axis=1).astype(np.float32)

    return out


# =============================================================================
# Loaded video bundle
# =============================================================================


@dataclass
class VideoBundle:
    """All in-memory loaders for one results root."""

    base_path: Path
    camera: Camera
    depth: VipeDepth
    masks: VipeMasks
    dense_flow: DenseFlow
    rgb: list[np.ndarray | None]
    height: int
    width: int

    @property
    def num_frames(self) -> int:
        return self.camera.num_frames

    @classmethod
    def load(cls, base_path: Path) -> "VideoBundle":
        base_path = Path(base_path)
        camera = Camera(base_path)
        depth = VipeDepth(base_path)
        masks = VipeMasks(base_path)
        dense_flow = DenseFlow(base_path)

        t = camera.num_frames
        rgb: list[np.ndarray | None] = [None] * t
        height = depth.height or masks.height or dense_flow.height
        width = depth.width or masks.width or dense_flow.width

        if modality_exists(base_path, "rgb", "*.mp4"):
            for fi, frame in read_rgb_frames(base_path):
                if fi >= t:
                    break
                rgb[fi] = np.asarray(frame, dtype=np.uint8)
                if height == 0:
                    height, width = int(frame.shape[0]), int(frame.shape[1])

        if height == 0 or width == 0:
            raise RuntimeError(f"Could not infer image size under {base_path}")

        return cls(
            base_path=base_path,
            camera=camera,
            depth=depth,
            masks=masks,
            dense_flow=dense_flow,
            rgb=rgb,
            height=height,
            width=width,
        )


# =============================================================================
# Viser UI
# =============================================================================


@dataclass
class GlobalContext:
    bundle: VideoBundle
    device: torch.device


_global_context: GlobalContext | None = None


@dataclass
class SceneFrameHandle:
    frame_idx: int
    frame_handle: viser.FrameHandle
    frustum_handle: viser.CameraFrustumHandle
    pcd_handle: viser.PointCloudHandle | None = None
    flow_handle: viser.LineSegmentsHandle | None = None
    base_rgb: np.ndarray | None = None
    instance_ids: np.ndarray | None = None

    def __post_init__(self):
        self.visible = False

    @property
    def visible(self) -> bool:
        return self.frame_handle.visible

    @visible.setter
    def visible(self, value: bool):
        self.frame_handle.visible = value
        self.frustum_handle.visible = value
        if self.pcd_handle is not None:
            self.pcd_handle.visible = value
        if self.flow_handle is not None:
            self.flow_handle.visible = value


class ClientClosures:
    """Client-side GUI + scene rebuild."""

    def __init__(self, client: viser.ClientHandle):
        self.client = client

        async def _run():
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            finally:
                self.cleanup()

        self.task = asyncio.create_task(_run())
        self.gui_playback_handle: viser.GuiFolderHandle | None = None
        self.gui_timestep: viser.GuiSliderHandle | None = None
        self.gui_framerate: viser.GuiSliderHandle | None = None
        self.gui_entities_handle: viser.GuiFolderHandle | None = None
        self.gui_entity_checks: dict[int, viser.GuiCheckboxHandle] = {}
        self.instance_phrases: dict[int, str] = {}
        self.scene_frame_handles: list[SceneFrameHandle] = []
        self.current_displayed_timestep: int = 0
        self._cached_flow_segs: dict[int, np.ndarray] = {}
        self._flow_cache_key: tuple | None = None

    async def stop(self):
        self.task.cancel()
        await self.task

    async def run(self):
        logger.info(f"Client {self.client.client_id} connected")
        bundle = self.global_context().bundle

        with self.client.gui.add_folder("Sample"):
            self.gui_name = self.client.gui.add_text("Results", bundle.base_path.name)
            self.gui_t_sub = self.client.gui.add_slider("Temporal subsample", min=1, max=16, step=1, initial_value=1)
            self.gui_s_sub = self.client.gui.add_slider("Spatial subsample", min=1, max=8, step=1, initial_value=2)
            self.gui_t_sub.on_update(self.on_sample_update)
            self.gui_s_sub.on_update(self.on_sample_update)

        with self.client.gui.add_folder("Scene"):
            self.gui_point_size = self.client.gui.add_slider(
                "Point size", min=0.0001, max=0.01, step=0.001, initial_value=0.001
            )

            @self.gui_point_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    if frame_node.pcd_handle is not None:
                        frame_node.pcd_handle.point_size = self.gui_point_size.value

            self.gui_frustum_size = self.client.gui.add_slider(
                "Frustum size", min=0.01, max=0.5, step=0.01, initial_value=0.15
            )

            @self.gui_frustum_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    frame_node.frustum_handle.scale = self.gui_frustum_size.value

            self.gui_colorful_frustum_toggle = self.client.gui.add_checkbox(
                "Colorful Frustum",
                initial_value=False,
            )

            @self.gui_colorful_frustum_toggle.on_update
            async def _(_) -> None:
                self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

            self.gui_fov = self.client.gui.add_slider("FoV", min=30.0, max=120.0, step=1.0, initial_value=60.0)

            @self.gui_fov.on_update
            async def _(_) -> None:
                self.client.camera.fov = np.deg2rad(self.gui_fov.value)

            self.gui_show_flow = self.client.gui.add_checkbox("Show flow", initial_value=True)

            @self.gui_show_flow.on_update
            async def _(_) -> None:
                await self.on_flow_filter_update(None)

            self.gui_flow_thresh = self.client.gui.add_slider(
                "Flow certainty thresh", min=0.0, max=1.0, step=0.05, initial_value=0.1
            )

            @self.gui_flow_thresh.on_update
            async def _(_) -> None:
                await self.on_flow_filter_update(None)

            self.gui_max_flow_arrows = self.client.gui.add_slider(
                "Max flow arrows / src", min=100, max=8000, step=100, initial_value=2000
            )

            @self.gui_max_flow_arrows.on_update
            async def _(_) -> None:
                await self.on_flow_filter_update(None)

            gui_snapshot = self.client.gui.add_button(
                "Snapshot",
                hint="Take a snapshot of the current scene",
            )

            @gui_snapshot.on_click
            def _(_) -> None:
                file_name = f"{bundle.base_path.name}.png"
                snapshot_img = self.client.get_render(height=720, width=1280, transport_format="png")
                self.client.send_file_download(file_name, iio.imwrite("<bytes>", snapshot_img, extension=".png"))

        await self.on_sample_update(None)

        while True:
            if self.gui_framerate is not None and self.gui_framerate.value > 0:
                self._incr_timestep()
                await asyncio.sleep(1.0 / self.gui_framerate.value)
            else:
                await asyncio.sleep(1.0)

    async def on_sample_update(self, _):
        self._rebuild_entities_gui()
        with self.client.atomic():
            self._rebuild_scene()
        self._rebuild_playback_gui()
        self._apply_entity_colors()
        self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

    async def on_flow_filter_update(self, _):
        self._flow_cache_key = None
        with self.client.atomic():
            self._rebuild_scene()
        self._rebuild_playback_gui()
        self._apply_entity_colors()
        self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

    def _selected_entity_ids(self) -> set[int]:
        return {eid for eid, checkbox in self.gui_entity_checks.items() if bool(checkbox.value)}

    def _apply_entity_colors(self) -> None:
        selected = self._selected_entity_ids()
        for frame_node in self.scene_frame_handles:
            if frame_node.pcd_handle is None or frame_node.base_rgb is None:
                continue
            ids = (
                frame_node.instance_ids
                if frame_node.instance_ids is not None
                else np.zeros((frame_node.base_rgb.shape[0],), dtype=np.uint8)
            )
            colors = blend_entity_colors(frame_node.base_rgb, ids, selected)
            frame_node.pcd_handle.colors = colors

    def _rebuild_entities_gui(self) -> None:
        bundle = self.global_context().bundle

        if self.gui_entities_handle is not None:
            self.gui_entities_handle.remove()
            self.gui_entities_handle = None
        self.gui_entity_checks = {}

        self.instance_phrases = {
            eid: bundle.masks.phrase_for(eid) for eid in bundle.masks.instance_ids
        }
        if not self.instance_phrases or bundle.masks.num_frames == 0:
            logger.warning("No instance masks/phrases under %s; skipping Entities GUI.", bundle.base_path)
            return

        entity_ids = list(bundle.masks.instance_ids)
        if not entity_ids:
            return

        self.gui_entities_handle = self.client.gui.add_folder("Entities")
        with self.gui_entities_handle:
            self.client.gui.add_markdown(
                "TrackAnything instance tracks. Toggle to shade matching points yellow "
                "and to gate src-centered flow arrows."
            )
            for eid in entity_ids:
                label = self.instance_phrases.get(eid, "entity")
                default_on = label.strip().lower() != "sky"
                checkbox = self.client.gui.add_checkbox(
                    f"{eid}: {label}",
                    initial_value=default_on,
                )

                @checkbox.on_update
                async def _(_) -> None:
                    self._apply_entity_colors()
                    await self.on_flow_filter_update(None)

                self.gui_entity_checks[eid] = checkbox

    def _set_frustum_color(self, colorful: bool):
        for frame_idx, frame_node in enumerate(self.scene_frame_handles):
            if not colorful:
                frame_node.frustum_handle.color = (0, 0, 0)
            else:
                denom = max(len(self.scene_frame_handles) - 1, 1)
                frame_node.frustum_handle.color = _jet_rgb(1.0 - frame_idx / denom)

    def _get_flow_segments(self, bundle: VideoBundle) -> dict[int, np.ndarray]:
        if not bool(self.gui_show_flow.value):
            return {}
        if bundle.dense_flow.num_edges == 0:
            return {}

        selected = frozenset(self._selected_entity_ids())
        if not selected and self.instance_phrases:
            selected = frozenset(
                eid
                for eid, name in self.instance_phrases.items()
                if eid > 0 and name.strip().lower() != "sky"
            )

        key = (
            float(self.gui_flow_thresh.value),
            int(self.gui_max_flow_arrows.value),
            selected,
        )
        if key == self._flow_cache_key and self._cached_flow_segs is not None:
            return self._cached_flow_segs

        max_arrows = int(self.gui_max_flow_arrows.value)
        arrow_stride = 1 if max_arrows >= 4000 else (2 if max_arrows >= 1500 else 4)

        flow_result = bundle.dense_flow.get_edges()
        segs = flow_segments_world(
            flow_result,
            bundle.camera,
            bundle.depth,
            bundle.masks,
            selected_entity_ids=set(selected),
            certainty_thresh=float(self.gui_flow_thresh.value),
            max_arrows_per_src=max_arrows,
            arrow_stride=arrow_stride,
            require_src_mask=True,
        )
        self._cached_flow_segs = segs
        self._flow_cache_key = key
        n_arrows = sum(int(s.shape[0]) for s in segs.values())
        logger.info(
            "Flow segments: %d src frames, %d arrows total (thresh=%.2f, scale=%d)",
            len(segs),
            n_arrows,
            float(self.gui_flow_thresh.value),
            FLOW_RES_SCALE,
        )
        return segs

    def _rebuild_scene(self) -> None:
        bundle = self.global_context().bundle
        device = self.global_context().device
        spatial_subsample: int = int(self.gui_s_sub.value)
        temporal_subsample: int = int(self.gui_t_sub.value)

        flow_segs = self._get_flow_segments(bundle)

        rays: np.ndarray | None = None
        first_frame_y: np.ndarray | None = None

        self.client.scene.reset()
        self.client.camera.fov = np.deg2rad(self.gui_fov.value)
        self.scene_frame_handles = []

        for frame_idx in range(bundle.num_frames):
            if frame_idx % temporal_subsample != 0:
                continue

            rgb = bundle.rgb[frame_idx] if frame_idx < len(bundle.rgb) else None
            depth = bundle.depth.get_depth([frame_idx])[0]
            instances = bundle.masks.get_masks([frame_idx])[0]
            inst_mask = id_map_from_instances(instances, bundle.height, bundle.width)
            if rgb is None and depth is None:
                continue

            c2w = bundle.camera.get_c2w([frame_idx])[0]
            intr = bundle.camera.get_intrinsics([frame_idx])[0]
            if c2w is None or intr is None:
                continue
            fx, fy, cx, cy = intr
            frame_height = bundle.height
            frame_width = bundle.width
            fov = float(2.0 * np.arctan2(frame_height / 2.0, float(fy)))

            if rgb is not None:
                sampled_rgb = np.asarray(rgb, dtype=np.uint8)[::spatial_subsample, ::spatial_subsample]
            else:
                sampled_rgb = np.full(
                    (
                        (frame_height + spatial_subsample - 1) // spatial_subsample,
                        (frame_width + spatial_subsample - 1) // spatial_subsample,
                        3,
                    ),
                    180,
                    dtype=np.uint8,
                )

            sampled_inst = None
            if inst_mask is not None:
                sampled_inst = np.asarray(inst_mask, dtype=np.uint8)[::spatial_subsample, ::spatial_subsample]

            if first_frame_y is None:
                first_frame_y = c2w[:3, 1].astype(np.float64)
                self.client.scene.set_up_direction(-first_frame_y)

            if rays is None:
                rays = pinhole_rays(
                    frame_height,
                    frame_width,
                    float(fx),
                    float(fy),
                    float(cx),
                    float(cy),
                    spatial_subsample,
                    device,
                )

            if depth is not None:
                pcd, depth_mask = unproject_depth_to_camera_pcd(depth, rays, spatial_subsample, device)
            else:
                pcd, depth_mask = None, None

            segs = flow_segs.get(frame_idx)
            frame_node = self._make_frame_nodes(
                frame_idx,
                c2w,
                sampled_rgb,
                fov,
                pcd,
                depth_mask,
                sampled_inst,
                flow_segments=segs,
            )
            self.scene_frame_handles.append(frame_node)

        logger.info(
            "Built scene for %s: %d frames (t_sub=%d, s_sub=%d, device=%s, flow=%s)",
            bundle.base_path.name,
            len(self.scene_frame_handles),
            temporal_subsample,
            spatial_subsample,
            device,
            "on" if bool(self.gui_show_flow.value) else "off",
        )

    def _make_frame_nodes(
        self,
        frame_idx: int,
        c2w: np.ndarray,
        rgb: np.ndarray,
        fov: float,
        pcd: np.ndarray | None,
        pcd_mask: np.ndarray | None = None,
        instance_mask: np.ndarray | None = None,
        flow_segments: np.ndarray | None = None,
    ) -> SceneFrameHandle:
        handle = self.client.scene.add_frame(
            f"/frames/t{frame_idx}",
            axes_length=0.05,
            axes_radius=0.005,
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
        )
        frame_height, frame_width = rgb.shape[:2]

        frame_thumbnail = Image.fromarray(rgb)
        frame_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
        frustum_handle = self.client.scene.add_camera_frustum(
            f"/frames/t{frame_idx}/frustum",
            fov=fov,
            aspect=frame_width / max(frame_height, 1),
            scale=self.gui_frustum_size.value,
            image=np.array(frame_thumbnail),
        )

        base_rgb: np.ndarray | None = None
        instance_ids: np.ndarray | None = None
        if pcd is not None:
            pcd_flat = pcd.reshape(-1, 3)
            rgb_flat = rgb.reshape(-1, 3)
            if instance_mask is not None:
                inst_flat = instance_mask.reshape(-1).astype(np.uint8)
            else:
                inst_flat = np.zeros((pcd_flat.shape[0],), dtype=np.uint8)
            if pcd_mask is not None:
                mask_flat = pcd_mask.reshape(-1)
                pcd_flat = pcd_flat[mask_flat]
                rgb_flat = rgb_flat[mask_flat]
                inst_flat = inst_flat[mask_flat]
            base_rgb = rgb_flat.astype(np.uint8).copy()
            instance_ids = inst_flat
            colors = blend_entity_colors(base_rgb, instance_ids, self._selected_entity_ids())
            pcd_handle = self.client.scene.add_point_cloud(
                name=f"/frames/t{frame_idx}/point_cloud",
                points=pcd_flat,
                colors=colors,
                point_size=self.gui_point_size.value,
                point_shape="rounded",
            )
        else:
            pcd_handle = None

        flow_handle = None
        if flow_segments is not None and flow_segments.shape[0] > 0:
            n = int(flow_segments.shape[0])
            colors = np.tile(np.array([64, 220, 255], dtype=np.uint8), (n, 2, 1))
            flow_handle = self.client.scene.add_line_segments(
                name=f"/flow/t{frame_idx}",
                points=flow_segments.astype(np.float32),
                colors=colors,
                line_width=2.0,
            )

        return SceneFrameHandle(
            frame_idx=frame_idx,
            frame_handle=handle,
            frustum_handle=frustum_handle,
            pcd_handle=pcd_handle,
            flow_handle=flow_handle,
            base_rgb=base_rgb,
            instance_ids=instance_ids,
        )

    def _incr_timestep(self):
        if self.gui_timestep is not None and self.scene_frame_handles:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.scene_frame_handles)

    def _decr_timestep(self):
        if self.gui_timestep is not None and self.scene_frame_handles:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % len(self.scene_frame_handles)

    def _rebuild_playback_gui(self):
        bundle = self.global_context().bundle
        self.gui_name.value = bundle.base_path.name
        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()
        self.gui_playback_handle = self.client.gui.add_folder("Playback")

        n_frames = max(len(self.scene_frame_handles) - 1, 0)
        with self.gui_playback_handle:
            self.gui_timestep = self.client.gui.add_slider(
                "Timeline", min=0, max=n_frames, step=1, initial_value=0
            )
            gui_timestep = self.gui_timestep
            gui_frame_control = self.client.gui.add_button_group("Control", options=["Prev", "Next"])
            self.gui_framerate = self.client.gui.add_slider("FPS", min=0, max=30, step=1.0, initial_value=15)

            @gui_frame_control.on_click
            async def _(_) -> None:
                if gui_frame_control.value == "Prev":
                    self._decr_timestep()
                else:
                    self._incr_timestep()

            self.current_displayed_timestep = gui_timestep.value
            if self.scene_frame_handles:
                self.scene_frame_handles[0].visible = True

            @gui_timestep.on_update
            async def _(_) -> None:
                if not self.scene_frame_handles:
                    return
                current_timestep = int(gui_timestep.value)
                prev_timestep = self.current_displayed_timestep
                with self.client.atomic():
                    self.scene_frame_handles[current_timestep].visible = True
                    if prev_timestep != current_timestep:
                        self.scene_frame_handles[prev_timestep].visible = False
                self.current_displayed_timestep = current_timestep

    def cleanup(self):
        logger.info(f"Client {self.client.client_id} disconnected")

    @classmethod
    def global_context(cls) -> GlobalContext:
        global _global_context
        assert _global_context is not None, "Global context not initialized"
        return _global_context


def run_viser(
    base_path: Path,
    port: int = 20540,
    host: str = "127.0.0.1",
    device: torch.device | None = None,
) -> None:
    device = device or resolve_device("auto")
    logger.info("Loading video from %s (device=%s)", base_path, device)
    bundle = VideoBundle.load(base_path)

    global _global_context
    _global_context = GlobalContext(bundle=bundle, device=device)
    logger.info(
        "Loaded %s: frames=%d depth=%d masks=%d flow_edges=%d",
        bundle.base_path.name,
        bundle.num_frames,
        bundle.depth.num_frames,
        bundle.masks.num_frames,
        bundle.dense_flow.num_edges,
    )

    server = viser.ViserServer(host=host, port=port, verbose=False)
    logger.info("Viser server: http://%s:%d", host, port)
    client_closures: dict[int, ClientClosures] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle):
        client_closures[client.client_id] = ClientClosures(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle):
        await client_closures[client.client_id].stop()
        del client_closures[client.client_id]

    while True:
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Shutting down server...")
            break
    server.stop()


def main() -> int:
    default_base = Path(__file__).resolve().parent / "vipe" / "vipe_results_flow"
    parser = argparse.ArgumentParser(
        description="Local Viser viewer for ViPE artifacts (no vipe package; Torch MPS/CPU)."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=default_base,
        help=f"ViPE results directory (default: {default_base})",
    )
    parser.add_argument("--port", "-p", type=int, default=20540, help="Viser port.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "mps", "cpu"),
        help="Torch device: auto (MPS then CPU), mps, or cpu. Never cuda.",
    )
    args = parser.parse_args()

    global _DEVICE
    _DEVICE = resolve_device(args.device)
    run_viser(
        base_path=args.base_path,
        port=int(args.port),
        host=str(args.host),
        device=_DEVICE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
