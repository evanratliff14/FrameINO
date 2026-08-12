#!/usr/bin/env python3
"""
CLI for visualizing ViPE dense / sparse optical flow.

Pass only the ViPE results root; modality paths are resolved by ``vipe_io``.

Run from the repo root. Prefer an explicit ``--base_path`` (real data is typically
``preprocess/vipe/vipe_results_flow``; the script default is under this file's directory).

Examples::

  # Edge report (list + gap histogram)
  python preprocess/vipe_output_interface/vipe_visualize_flow.py \\
    --base_path preprocess/vipe/vipe_results_flow report --out /tmp/flow_edges.txt

  # OpenCV arrow viewer (src-centered edges)
  python preprocess/vipe_output_interface/vipe_visualize_flow.py \\
    --base_path preprocess/vipe/vipe_results_flow show
  python preprocess/vipe_output_interface/vipe_visualize_flow.py \\
    --base_path preprocess/vipe/vipe_results_flow show --thresh 0.1 --stride 4 --wait_ms 0
  python preprocess/vipe_output_interface/vipe_visualize_flow.py \\
    --base_path preprocess/vipe/vipe_results_flow show --sparse

  # Full-res [T,H,W,2] tensor stats / optional save
  python preprocess/vipe_output_interface/vipe_visualize_flow.py \\
    --base_path preprocess/vipe/vipe_results_flow tensor
  python preprocess/vipe_output_interface/vipe_visualize_flow.py \\
    --base_path preprocess/vipe/vipe_results_flow tensor --thresh 0.1 --out /tmp/flow.pt
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from vipe_dense_flow import FLOW_RES_SCALE, DenseFlow, pick_edges
from vipe_io import list_flow_edges, modality_exists, read_rgb_frames
from vipe_optical_flow import FlowResult, flow_arrows_for_src
from vipe_sparse_tracks import SparseTracks

logger = logging.getLogger(__name__)


def build_fullres_flow_tensor(
    flow_result: FlowResult,
    num_frames: int,
    height: int,
    width: int,
    certainty_thresh: float,
) -> torch.Tensor:
    """
    Build a src-indexed full-res flow tensor ``[T, H, W, 2]`` (u, v only)
    from a ``FlowResult`` of full-res ``FlowObject``s.
    """
    t = int(num_frames)
    h_img, w_img = int(height), int(width)
    out = torch.zeros((t, h_img, w_img, 2), dtype=torch.float32)

    for edge in flow_result.edges:
        src = int(edge.src)
        if src < 0 or src >= t:
            continue
        flow = np.asarray(edge.tensor, dtype=np.float32)
        cert = flow[..., 2]
        keep = np.isfinite(cert) & (cert >= float(certainty_thresh)) & (cert > 0)
        ii, jj = np.nonzero(keep)
        if ii.size == 0:
            continue
        valid = (ii < h_img) & (jj < w_img)
        ii, jj = ii[valid], jj[valid]
        out[src, ii, jj, 0] = torch.from_numpy(flow[ii, jj, 0])
        out[src, ii, jj, 1] = torch.from_numpy(flow[ii, jj, 1])
    return out


def _flow_to_bgr_preview(
    flow_hw3: np.ndarray,
    certainty_thresh: float,
    bg_bgr: np.ndarray | None,
    arrow_stride: int = 2,
) -> np.ndarray:
    """
    Render one full-res ``[H, W, 3]`` flow field as BGR with arrows.

    Canvas size matches the flow tensor; RGB is resized to that canvas.
    """
    import cv2

    flow = np.asarray(flow_hw3, dtype=np.float32)
    assert flow.ndim == 3 and flow.shape[-1] == 3, f"Expected [H,W,3], got {flow.shape}"
    h_img, w_img = flow.shape[:2]
    if bg_bgr is None:
        canvas = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    else:
        canvas = cv2.resize(bg_bgr, (w_img, h_img), interpolation=cv2.INTER_AREA)

    src_xy, dst_xy, _ = flow_arrows_for_src(
        flow,
        certainty_thresh=certainty_thresh,
        stride=max(1, int(arrow_stride)),
    )
    for (x0, y0), (x1, y1) in zip(src_xy, dst_xy):
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        cv2.arrowedLine(canvas, p0, p1, (255, 255, 255), 1, tipLength=0.3)
    return canvas


def visualize_flow_video(
    base_path: Path,
    certainty_thresh: float = 0.0,
    arrow_stride: int = 2,
    wait_ms: int = 50,
    *,
    use_sparse: bool = False,
) -> None:
    """
    OpenCV window stepping through src-centered flow edges from the results root.
    """
    import cv2

    base_path = Path(base_path)
    dense = DenseFlow(base_path)
    result = dense.get_edges()
    if use_sparse and modality_exists(base_path, "sparse_tracks", "*.zip"):
        sparse = SparseTracks(base_path, height=dense.height, width=dense.width)
        srcs = [e.src for e in result.edges] or list(range(max(sparse.num_frames - 1, 0)))
        result = sparse.get_flow(srcs)

    if not result.edges:
        logger.error("No flow edges under %s", base_path)
        return

    bg_by_frame: dict[int, np.ndarray] = {}
    if modality_exists(base_path, "rgb", "*.mp4"):
        for idx, rgb in read_rgb_frames(base_path):
            bg_by_frame[idx] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    win = "vipe flow (src-centered)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for edge in result.edges:
        preview = _flow_to_bgr_preview(
            edge.tensor,
            certainty_thresh=certainty_thresh,
            bg_bgr=bg_by_frame.get(edge.src),
            arrow_stride=arrow_stride,
        )
        label = f"src={edge.src} -> dst={edge.dst}  thresh={certainty_thresh}"
        cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow(win, preview)
        key = cv2.waitKey(int(wait_ms)) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyWindow(win)


def write_flow_edge_report(base_path: Path, out_txt: Path) -> Path:
    """Write edge inventory + gap histogram for ``base_path/dense_flow/``."""
    edges = list_flow_edges(base_path)
    srcs = [s for s, _ in edges]
    dsts = [d for _, d in edges]
    src_counts = Counter(srcs)
    dst_counts = Counter(dsts)
    dup_srcs = sorted(s for s, c in src_counts.items() if c > 1)
    dup_dsts = sorted(d for d, c in dst_counts.items() if c > 1)
    gap_hist = Counter((d - s) for s, d in edges)
    picked = pick_edges(edges)

    lines: list[str] = []
    lines.append(f"# Dense flow edge report for {Path(base_path).resolve()}")
    lines.append(f"# num_edges={len(edges)}  pick_edges={len(picked)}")
    lines.append("")
    lines.append("## Edges (src -> dst)")
    for src, dst in edges:
        lines.append(f"  ({src}, {dst})   gap={dst - src}")
    lines.append("")
    lines.append("## pick_edges selection")
    for src, dst in picked:
        lines.append(f"  ({src}, {dst})")
    lines.append("")
    lines.append("## Duplicate sources / destinations")
    lines.append(
        f"  duplicate_srcs={'yes: ' + ','.join(map(str, dup_srcs)) if dup_srcs else 'none'}"
    )
    lines.append(
        f"  duplicate_dsts={'yes: ' + ','.join(map(str, dup_dsts)) if dup_dsts else 'none'}"
    )
    lines.append("")
    lines.append("## Histogram of (dst - src)")
    for gap in sorted(gap_hist):
        lines.append(f"  gap={gap:4d}  count={gap_hist[gap]}")
    lines.append("")

    out_path = Path(out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote edge report (%d edges) to %s", len(edges), out_path)
    return out_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    default_base = Path(__file__).resolve().parent / "vipe" / "vipe_results_flow"
    parser = argparse.ArgumentParser(
        description="Inspect / visualize ViPE flow from a results root (no vipe package)."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=default_base,
        help=f"ViPE results root (default: {default_base})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_report = sub.add_parser("report", help="Write edge list + gap histogram to a txt file.")
    p_report.add_argument("--out", type=Path, required=True, help="Output .txt path")

    p_show = sub.add_parser("show", help="OpenCV popup of src-centered flow arrows.")
    p_show.add_argument("--thresh", type=float, default=0.0, help="Certainty threshold on channel w")
    p_show.add_argument("--stride", type=int, default=2, help="Arrow grid stride")
    p_show.add_argument("--wait_ms", type=int, default=1000, help="cv2.waitKey delay (0 = keypress)")
    p_show.add_argument("--sparse", action="store_true", help="Prefer sparse tracks if available")

    p_tensor = sub.add_parser("tensor", help="Build [T,H,W,2] src-indexed full-res flow and print stats.")
    p_tensor.add_argument("--thresh", type=float, default=0.0)
    p_tensor.add_argument("--out", type=Path, default=None, help="Optional .pt save path")

    args = parser.parse_args()
    base = Path(args.base_path)

    if args.cmd == "report":
        write_flow_edge_report(base, args.out)
        return 0

    if args.cmd == "show":
        visualize_flow_video(
            base,
            certainty_thresh=float(args.thresh),
            arrow_stride=int(args.stride),
            wait_ms=int(args.wait_ms),
            use_sparse=bool(args.sparse),
        )
        return 0

    if args.cmd == "tensor":
        dense = DenseFlow(base)
        result = dense.get_edges()
        max_src = max((e.src for e in result.edges), default=-1)
        num_frames = max(max_src + 1, 1)
        try:
            from vipe_camera import Camera

            num_frames = max(Camera(base).num_frames, num_frames)
        except Exception:
            pass
        tens = build_fullres_flow_tensor(
            result,
            num_frames=num_frames,
            height=dense.height,
            width=dense.width,
            certainty_thresh=float(args.thresh),
        )
        nonzero = int((tens.abs().sum(dim=-1) > 0).sum().item())
        print(
            f"tensor shape={tuple(tens.shape)} nonzero_pixels={nonzero} "
            f"edges={len(result.edges)} scale={FLOW_RES_SCALE}"
        )
        if args.out is not None:
            torch.save(tens, args.out)
            print(f"saved {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
