#!/usr/bin/env python3
"""
Instance-mask loader for one ViPE results video.

Packed per-frame id maps are kept in memory; callers receive
``InstanceMask`` objects (id + phrase + boolean ``[N, H, W]``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vipe_io import modality_exists, read_instance_masks, read_instance_phrases

logger = logging.getLogger(__name__)


@dataclass
class InstanceMask:
    """Binary mask for one tracked instance spanning the full video."""

    instance_id: int
    phrase: str
    mask: np.ndarray  # bool [N, H, W]

    def __post_init__(self) -> None:
        self.instance_id = int(self.instance_id)
        self.phrase = str(self.phrase)
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 3:
            raise ValueError(
                f"InstanceMask.mask must be 3D [N, H, W], got shape {self.mask.shape}"
            )

    @property
    def height(self) -> int:
        return int(self.mask.shape[1])

    @property
    def width(self) -> int:
        return int(self.mask.shape[2])

    @property
    def frames(self) -> int:
        return int(self.mask.shape[0])

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def at(self, frame_idx: int) -> np.ndarray:
        """Return the 2D ``[H, W]`` plane for absolute video frame ``frame_idx``."""
        t = int(frame_idx)
        if t < 0 or t >= self.frames:
            raise IndexError(
                f"frame_idx {t} out of range for InstanceMask with N={self.frames}"
            )
        return self.mask[t]

    def as_uint8(self) -> np.ndarray:
        """``0/1`` uint8 copy of the mask (handy for packing / I/O)."""
        return self.mask.astype(np.uint8)

    def as_float(self) -> np.ndarray:
        """``0/1`` float32 copy — suitable for multiplying with flow tensors."""
        return self.mask.astype(np.float32)

    def contains_uv(
        self,
        u: np.ndarray,
        v: np.ndarray,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Boolean membership for pixel coords ``(u=x, v=y)`` on frame ``frame_idx``."""
        plane = self.at(frame_idx)
        uu = np.clip(np.rint(u), 0, self.width - 1).astype(np.int64)
        vv = np.clip(np.rint(v), 0, self.height - 1).astype(np.int64)
        return plane[vv, uu]


class VipeMasks:
    """In-memory instance masks for a single ViPE results root."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path: Path | None = None
        self._id_maps: list[np.ndarray | None] = []
        self.phrases: dict[int, str] = {}
        self.height: int = 0
        self.width: int = 0
        if base_path is not None:
            self.set_video(base_path)

    @property
    def num_frames(self) -> int:
        return len(self._id_maps)

    @property
    def instance_ids(self) -> list[int]:
        """Sorted catalog of known instance ids (phrases ∪ ids present in maps)."""
        ids = {int(i) for i in self.phrases if int(i) > 0}
        for id_map in self._id_maps:
            if id_map is None:
                continue
            ids.update(int(x) for x in np.unique(id_map) if int(x) > 0)
        return sorted(ids)

    def phrase_for(self, instance_id: int, default: str = "entity") -> str:
        return self.phrases.get(int(instance_id), default)

    def set_video(self, base_path: Path) -> None:
        """Load all instance masks (and phrases if present) from ``base_path/mask/``."""
        base_path = Path(base_path)
        self.base_path = base_path
        self._id_maps = []
        self.phrases = {}
        self.height, self.width = 0, 0

        if not modality_exists(base_path, "mask", "*.zip"):
            logger.warning("No mask zip under %s; VipeMasks is empty.", base_path)
            return

        by_idx: dict[int, np.ndarray] = {}
        for frame_idx, mask in read_instance_masks(base_path):
            arr = np.asarray(mask, dtype=np.uint8)
            by_idx[int(frame_idx)] = arr
            if self.height == 0:
                self.height, self.width = int(arr.shape[0]), int(arr.shape[1])

        if by_idx:
            t = max(by_idx.keys()) + 1
            self._id_maps = [None] * t
            for i, m in by_idx.items():
                if 0 <= i < t:
                    self._id_maps[i] = m

        if modality_exists(base_path, "mask", "*.txt"):
            try:
                self.phrases = {
                    int(k): str(v) for k, v in read_instance_phrases(base_path).items()
                }
            except Exception as exc:
                logger.warning("Failed to read instance phrases: %s", exc)
                self.phrases = {}

        # Backfill phrases for ids that appear in maps but not in the txt catalog.
        for iid in self.instance_ids:
            self.phrases.setdefault(iid, "entity")

        logger.info(
            "VipeMasks loaded %d/%d frames, %d instances from %s",
            sum(1 for m in self._id_maps if m is not None),
            len(self._id_maps),
            len(self.instance_ids),
            base_path,
        )

    def get_id_map(self, indices: list[int]) -> list[np.ndarray | None]:
        """Return packed uint8 ``[H, W]`` instance-id maps for ``indices`` (O(1) per index)."""
        out: list[np.ndarray | None] = []
        n = len(self._id_maps)
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= n:
                out.append(None)
            else:
                out.append(self._id_maps[i])
        return out

    def get_masks(self) -> list[InstanceMask]:
        """Return one video-spanning ``InstanceMask`` (``[N, H, W]``) per known id."""
        return [self._build_instance_mask(iid) for iid in self.instance_ids]

    def all_instances(self) -> list[InstanceMask]:
        """Alias for ``get_masks()``."""
        return self.get_masks()

    def get_instance(self, instance_id: int) -> InstanceMask:
        """Return one instance's binary mask over the full video ``[N, H, W]``."""
        return self._build_instance_mask(int(instance_id))

    def ids_at(
        self,
        frame_idx: int,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray | None:
        """Packed instance ids at pixel coords ``(u=x, v=y)``, or ``None`` if no map."""
        maps = self.get_id_map([frame_idx])
        id_map = maps[0]
        if id_map is None:
            return None
        h, w = id_map.shape
        uu = np.clip(np.rint(u), 0, w - 1).astype(np.int64)
        vv = np.clip(np.rint(v), 0, h - 1).astype(np.int64)
        return id_map[vv, uu]

    def _build_instance_mask(self, instance_id: int) -> InstanceMask:
        """Stack ``(id_map == iid)`` over all frames into ``[N, H, W]`` (False if missing)."""
        iid = int(instance_id)
        n = self.num_frames
        h, w = int(self.height), int(self.width)
        if n == 0 or h <= 0 or w <= 0:
            planes = np.zeros((0, max(h, 0), max(w, 0)), dtype=bool)
        else:
            planes = np.zeros((n, h, w), dtype=bool)
            for t, id_map in enumerate(self._id_maps):
                if id_map is None:
                    continue
                planes[t] = id_map == iid
        return InstanceMask(
            instance_id=iid,
            phrase=self.phrase_for(iid),
            mask=planes,
        )
