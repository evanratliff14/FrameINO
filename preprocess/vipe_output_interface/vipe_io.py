#!/usr/bin/env python3
"""
ViPE artifact IO for local (non-CUDA) tooling.

All public readers take only the ViPE results **root** folder. Subfolder names
(``depth/``, ``pose/``, ``dense_flow/``, …) are hardcoded; each modality folder
is expected to contain exactly one matching file.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import OpenEXR
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenEXR is required to read ViPE depth/*.zip and dense_flow/*.zip EXR frames. "
        "Install with: pip install OpenEXR"
    ) from exc

logger = logging.getLogger(__name__)


def sole_file(directory: Path, pattern: str) -> Path:
    """
    Return the unique file matching ``pattern`` under ``directory``.

    Raises ``FileNotFoundError`` if the directory is missing or empty,
    ``RuntimeError`` if more than one match exists.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Expected modality directory: {directory}")
    matches = sorted(directory.glob(pattern))
    # Ignore macOS metadata
    matches = [p for p in matches if p.name != ".DS_Store"]
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one file matching {pattern!r} in {directory}; "
            f"found {len(matches)}: {[p.name for p in matches]}"
        )
    return matches[0]


def modality_exists(base_path: Path, subdir: str, pattern: str) -> bool:
    """True if ``{base_path}/{subdir}`` exists and contains exactly one ``pattern`` match."""
    try:
        sole_file(Path(base_path) / subdir, pattern)
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def _parse_edge_name(file_name: str) -> tuple[int, int]:
    """Parse ``"{src}_{dst}.exr"`` into integer ``(src, dst)`` video-frame indices."""
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected edge name '{{src}}_{{dst}}.exr', got {file_name!r}")
    return int(parts[0]), int(parts[1])


def _read_exr_uvw(exr_path: str) -> np.ndarray:
    """Read float16 channels ``u,v,w`` from an EXR file path into float32 ``[h,w,3]``."""
    exr = OpenEXR.InputFile(exr_path)
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    u, v, w = exr.channels(["u", "v", "w"])
    u_arr = np.frombuffer(u, dtype=np.float16).reshape((height, width)).astype(np.float32)
    v_arr = np.frombuffer(v, dtype=np.float16).reshape((height, width)).astype(np.float32)
    w_arr = np.frombuffer(w, dtype=np.float16).reshape((height, width)).astype(np.float32)
    return np.stack([u_arr, v_arr, w_arr], axis=-1)


# ---------------------------------------------------------------------------
# Pose / intrinsics / RGB
# ---------------------------------------------------------------------------


def read_pose_c2w(base_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load OpenCV cam2world poses from ``{base_path}/pose/*.npz``.

    Returns:
      inds: int array [T] of video-frame indices
      c2w:  float64 [T, 4, 4] camera-to-world matrices
    """
    npz_path = sole_file(Path(base_path) / "pose", "*.npz")
    data = np.load(npz_path)
    return np.asarray(data["inds"]), np.asarray(data["data"], dtype=np.float64)


def read_intrinsics(base_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pinhole intrinsics ``[fx, fy, cx, cy]`` from ``{base_path}/intrinsics/*.npz``.

    Optionally validates ``*_camera.txt`` if present (PINHOLE only).

    Returns:
      inds: int array [T]
      intrinsics: float64 [T, 4]
    """
    base = Path(base_path)
    intr_path = sole_file(base / "intrinsics", "*.npz")
    data = np.load(intr_path)
    inds = np.asarray(data["inds"])
    intrinsics = np.asarray(data["data"], dtype=np.float64)

    camera_files = sorted((base / "intrinsics").glob("*_camera.txt"))
    if camera_files:
        with camera_files[0].open("r") as f:
            camera_types = [line.split(":")[1].strip() for line in f.readlines() if line.strip()]
        if camera_types and any(ct != "PINHOLE" for ct in camera_types):
            raise ValueError(f"Only PINHOLE is supported; got {set(camera_types)}")

    assert intrinsics.ndim == 2 and intrinsics.shape[1] >= 4, (
        f"Unexpected intrinsics shape {intrinsics.shape}; expected [T, 4+] as fx,fy,cx,cy"
    )
    return inds, intrinsics[:, :4]


def read_rgb_frames(base_path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield ``(frame_idx, rgb)`` from the sole ``{base_path}/rgb/*.mp4``.

    ``rgb`` is uint8 ``[H, W, 3]`` in RGB order.
    """
    rgb_path = sole_file(Path(base_path) / "rgb", "*.mp4")
    try:
        import cv2

        cap = cv2.VideoCapture(str(rgb_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {rgb_path}")
        frame_idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                yield frame_idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_idx += 1
        finally:
            cap.release()
        return
    except Exception as cv_exc:
        logger.warning("OpenCV RGB load failed (%s); trying imageio ffmpeg.", cv_exc)

    import imageio

    reader = imageio.get_reader(str(rgb_path), "ffmpeg")
    try:
        for frame_idx, rgb in enumerate(reader):
            yield frame_idx, np.asarray(rgb)
    finally:
        reader.close()


def peek_rgb_size(base_path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` of the first RGB frame without loading the whole video."""
    for _, rgb in read_rgb_frames(base_path):
        return int(rgb.shape[0]), int(rgb.shape[1])
    raise RuntimeError(f"No RGB frames under {base_path}/rgb")


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def read_depth_frames(base_path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield ``(frame_idx, depth)`` from the sole ``{base_path}/depth/*.zip``.

    Each ``depth`` is float32 ``[H, W]`` metric depth.
    """
    zip_file_path = sole_file(Path(base_path) / "depth", "*.zip")
    valid_width, valid_height = 0, 0
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(Path(file_name).stem)
            with z.open(file_name) as f:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".exr") as tmp:
                        tmp.write(f.read())
                        tmp.flush()
                        exr = OpenEXR.InputFile(tmp.name)
                except OSError:
                    logger.warning(
                        "Failed to load EXR %s-%s; returning NaN map.",
                        zip_file_path,
                        file_name,
                    )
                    assert valid_width > 0 and valid_height > 0
                    yield frame_idx, np.full((valid_height, valid_width), np.nan, dtype=np.float32)
                    continue
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = width = dw.max.x - dw.min.x + 1
                valid_height = height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width))
                yield frame_idx, depth_data.astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Instance masks
# ---------------------------------------------------------------------------


def read_instance_masks(base_path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield ``(frame_idx, instance_id)`` uint8 ``[H, W]`` from ``{base_path}/mask/*.zip``.
    """
    import cv2

    zip_file_path = sole_file(Path(base_path) / "mask", "*.zip")
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(Path(file_name).stem)
            with z.open(file_name) as f:
                mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise RuntimeError(f"Failed to decode instance mask {file_name}")
                yield frame_idx, np.asarray(mask, dtype=np.uint8).copy()


def read_instance_phrases(base_path: Path) -> dict[int, str]:
    """
    Parse the sole ``{base_path}/mask/*.txt`` lines ``{id}: {phrase}`` into ``{id: phrase}``.
    """
    phrase_path = sole_file(Path(base_path) / "mask", "*.txt")
    instance_phrases: dict[int, str] = {}
    with phrase_path.open("r") as f:
        for line in f.readlines():
            if not line.strip() or ":" not in line:
                continue
            idx, phrase = line.split(":", 1)
            instance_phrases[int(idx)] = phrase.strip()
    return instance_phrases


# ---------------------------------------------------------------------------
# Dense flow
# ---------------------------------------------------------------------------


def list_flow_edges(base_path: Path) -> list[tuple[int, int]]:
    """
    List all ``(src, dst)`` edges in ``{base_path}/dense_flow/*.zip``.

    Returns a sorted list of video-frame index pairs from EXR member names.
    """
    try:
        zip_path = sole_file(Path(base_path) / "dense_flow", "*.zip")
    except FileNotFoundError:
        logger.warning("Dense flow zip not found under %s/dense_flow", base_path)
        return []
    edges: list[tuple[int, int]] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.endswith(".exr"):
                continue
            edges.append(_parse_edge_name(name))
    return sorted(edges)


def read_flow_edge(base_path: Path, src: int, dst: int) -> np.ndarray:
    """
    Read one edge ``(src -> dst)`` as float32 ``[h, w, 3]`` = ``(u, v, w)`` at 1/8 res.

    Raises ``KeyError`` if the edge is missing from the zip.
    """
    zip_path = sole_file(Path(base_path) / "dense_flow", "*.zip")
    member = f"{src}_{dst}.exr"
    with zipfile.ZipFile(zip_path, "r") as z:
        if member not in z.namelist():
            raise KeyError(f"Edge ({src}, {dst}) not in {zip_path} (looked for {member!r})")
        with z.open(member) as f:
            with tempfile.NamedTemporaryFile(suffix=".exr") as tmp:
                tmp.write(f.read())
                tmp.flush()
                return _read_exr_uvw(tmp.name)


def iter_flow_edges(base_path: Path) -> Iterator[tuple[int, int, np.ndarray]]:
    """
    Yield ``(src, dst, flow_hw3)`` for every EXR edge under ``{base_path}/dense_flow/``.

    Each ``flow_hw3`` is float32 ``[h, w, 3]`` = ``(u, v, certainty)`` at 1/8 res.
    """
    try:
        zip_path = sole_file(Path(base_path) / "dense_flow", "*.zip")
    except FileNotFoundError:
        return
        yield  # pragma: no cover — keep generator type
    with zipfile.ZipFile(zip_path, "r") as z:
        names = sorted(n for n in z.namelist() if n.endswith(".exr"))
        for name in names:
            src, dst = _parse_edge_name(name)
            with z.open(name) as f:
                with tempfile.NamedTemporaryFile(suffix=".exr") as tmp:
                    tmp.write(f.read())
                    tmp.flush()
                    flow = _read_exr_uvw(tmp.name)
            yield src, dst, flow


# ---------------------------------------------------------------------------
# Sparse tracks
# ---------------------------------------------------------------------------


def read_sparse_tracks(
    base_path: Path,
    start: int | None = None,
    end: int | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Yield ``(frame_idx, ids, uv)`` from ``{base_path}/sparse_tracks/*.zip``.

    Each member is ``{frame:05d}.npz`` with ``ids`` ``(N,)`` int32 and ``uv`` ``(N, 2)`` float32.
    Optional ``[start, end)`` filters by frame index.
    """
    zip_path = sole_file(Path(base_path) / "sparse_tracks", "*.zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in sorted(z.namelist()):
            if not file_name.endswith(".npz"):
                continue
            frame_idx = int(Path(file_name).stem)
            if start is not None and frame_idx < start:
                continue
            if end is not None and frame_idx >= end:
                continue
            with z.open(file_name) as f:
                data = np.load(f)
                ids = np.asarray(data["ids"], dtype=np.int32).copy()
                uv = np.asarray(data["uv"], dtype=np.float32).copy()
            yield frame_idx, ids, uv
