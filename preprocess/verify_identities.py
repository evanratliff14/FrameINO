'''
    verify vipe instance identities via qwen-vl collage + bbox matching
'''

import os
import sys
import csv
import gc
import time
import argparse
import json
import math
import tempfile

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# vipe mask loaders
from preprocess.vipe_output_interface.vipe_masks import VipeMasks, InstanceMask

csv.field_size_limit(sys.maxsize)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

TILE_SIZE = 224
MAX_COLS = 4
GRAY = (114, 114, 114)
MIN_AREA = 10000
AREA_FRAC = 0.5
IOU_THRESH = 0.3
SKIP_PHRASES = {"background", "sky"}

INSTRUCTION_PROMPT = (
    "This image is a collage of object crops. "
    "Detect every valid medium-to-large subject (people, animals, rigid objects). "
    "Do NOT detect sky, background, amorphous / non-rigid blobs, or tiny clutter. "
    "Return ONLY a JSON list of objects with boxes in [x_min, y_min, x_max, y_max] format "
    'on a 0 to 1000 normalized scale, like: [{"bbox":[x1,y1,x2,y2]}, ...]. '
    "No markdown, no prose."
)


def prepare_inputs_for_vllm(messages, processor, image_inputs):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs},
    }


def _touches_border(plane: np.ndarray) -> bool:
    return bool(
        plane[0, :].any()
        or plane[-1, :].any()
        or plane[:, 0].any()
        or plane[:, -1].any()
    )


def _bbox_from_mask(plane: np.ndarray):
    ys, xs = np.where(plane)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


def pick_first_frame(im: InstanceMask, n_frames: int):
    """
    return first eligible frame index, or None.
    prefer fully in-canvas; else area-only eligibility.
    """
    areas = im.mask[:n_frames].reshape(n_frames, -1).sum(axis=1).astype(np.int64)
    max_area = int(areas.max()) if n_frames > 0 else 0
    if max_area <= 0:
        return None

    fully_in = []
    area_only = []
    for t in range(n_frames):
        a = int(areas[t])
        if a < MIN_AREA or a < AREA_FRAC * max_area:
            continue
        plane = im.mask[t]
        if not _touches_border(plane):
            fully_in.append(t)
        else:
            area_only.append(t)

    if fully_in:
        return fully_in[0]
    if area_only:
        return area_only[0]
    return None


def _letterbox_tile(crop_rgb: np.ndarray, tile_size: int = TILE_SIZE) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    scale = float(tile_size) / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(crop_rgb).resize((nw, nh), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    tile = np.full((tile_size, tile_size, 3), GRAY, dtype=np.uint8)
    y0 = (tile_size - nh) // 2
    x0 = (tile_size - nw) // 2
    tile[y0 : y0 + nh, x0 : x0 + nw] = resized
    return tile


def build_collage(video_rgb, instance_masks, first_frames):
    """
    build gray canvas of masked letterboxed tiles.
    returns collage uint8 [H,W,3], id_to_tile_rect {iid: (x0,y0,x1,y1)}.
    """
    tiles = []
    order = []
    for im in instance_masks:
        iid = int(im.instance_id)
        if iid not in first_frames:
            continue
        t = int(first_frames[iid])
        plane = im.mask[t]
        x0, y0, x1, y1 = _bbox_from_mask(plane)
        frame = video_rgb[t]
        crop = frame[y0:y1, x0:x1].copy()
        m = plane[y0:y1, x0:x1]
        crop[~m] = GRAY
        tiles.append(_letterbox_tile(crop))
        order.append(iid)

    n = len(tiles)
    if n == 0:
        return None, {}

    cols = min(MAX_COLS, n)
    rows = int(math.ceil(n / cols))
    canvas = np.full((rows * TILE_SIZE, cols * TILE_SIZE, 3), GRAY, dtype=np.uint8)
    id_to_tile_rect = {}
    for i, (iid, tile) in enumerate(zip(order, tiles)):
        r, c = divmod(i, cols)
        y0, x0 = r * TILE_SIZE, c * TILE_SIZE
        canvas[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = tile
        id_to_tile_rect[iid] = (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)
    return canvas, id_to_tile_rect


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _parse_bboxes(text: str):
    text = text.strip()
    # strip optional markdown fences
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    data = json.loads(text[start : end + 1])
    boxes = []
    for item in data:
        bbox = item["bbox"] if isinstance(item, dict) else item
        # Convert relative [0, 1000] scale to canvas pixel scale
        x1 = (float(bbox[0]) / 1000.0) * canvas_w
        y1 = (float(bbox[1]) / 1000.0) * canvas_h
        x2 = (float(bbox[2]) / 1000.0) * canvas_w
        y2 = (float(bbox[3]) / 1000.0) * canvas_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def map_boxes_to_ids(boxes, id_to_tile_rect):
    matched = set()
    for box in boxes:
        best_iid, best_iou = None, 0.0
        for iid, rect in id_to_tile_rect.items():
            iou = _iou(box, rect)
            # also accept if box covers most of the tile
            rx0, ry0, rx1, ry1 = rect
            tile_area = max(1, (rx1 - rx0) * (ry1 - ry0))
            ix0, iy0 = max(box[0], rx0), max(box[1], ry0)
            ix1, iy1 = min(box[2], rx1), min(box[3], ry1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            cover = float(inter) / float(tile_area)
            score = max(iou, cover if cover >= 0.5 else 0.0)
            if score > best_iou:
                best_iou, best_iid = score, iid
        if best_iid is not None and best_iou >= IOU_THRESH:
            matched.add(int(best_iid))
    return matched


def verify_identities(video_path, vipe_output_filepath, llm, processor, sampling_params):
    """
    returns {instance_id: first_frame} for qwen-accepted subjects only.
    """
    masks = VipeMasks(vipe_output_filepath)
    instance_masks = [
        im
        for im in masks.get_masks()
        if int(im.instance_id) > 0 and im.phrase.strip().lower() not in SKIP_PHRASES
    ]

    vr = VideoReader(video_path, ctx=cpu(0))
    n_frames = min(len(vr), masks.num_frames)
    if n_frames <= 0 or not instance_masks:
        return {}

    # load only frames we may need after picking; first pass uses area on masks only
    first_frames = {}
    for im in instance_masks:
        t = pick_first_frame(im, n_frames)
        if t is not None:
            first_frames[int(im.instance_id)] = int(t)

    if not first_frames:
        return {}

    needed = sorted(set(first_frames.values()))
    batch = vr.get_batch(needed).asnumpy()  # [K,H,W,C] rgb
    idx_map = {t: i for i, t in enumerate(needed)}
    # dense list indexed by absolute frame for collage helper
    video_rgb = [None] * n_frames
    for t, i in idx_map.items():
        video_rgb[t] = batch[i]

    collage, id_to_tile_rect = build_collage(video_rgb, instance_masks, first_frames)
    if collage is None or not id_to_tile_rect:
        return {}

    # write collage for process_vision_info path-based image load
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    collage_img = Image.fromarray(collage)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": collage_img},
            {"type": "text", "text": INSTRUCTION_PROMPT},
        ]
    }]
    image_inputs, _video_inputs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
    )
    inputs = [prepare_inputs_for_vllm(messages, processor, image_inputs)]
    out = llm.generate(inputs, sampling_params=sampling_params)
    generated_text = out[0].outputs[0].text

    os.remove(tmp_path)

    boxes = _parse_bboxes(generated_text)
    matched = map_boxes_to_ids(boxes, id_to_tile_rect)
    return {iid: first_frames[iid] for iid in matched if iid in first_frames}


def single_process(input_csv_folder_path, store_csv_folder_path, GPU_offset, llm, processor, sampling_params):
    store_freq = 10

    csv_idx = GPU_offset
    csv_file_path = os.path.join(input_csv_folder_path, "sub" + str(csv_idx) + ".csv")
    print("CSV file we read is ", csv_file_path)

    store_file_path = os.path.join(store_csv_folder_path, "sub" + str(csv_idx) + ".csv")
    if not resume and os.path.exists(store_file_path):
        os.remove(store_file_path)

    find_resume = True
    if resume:
        find_resume = False
        with open(store_file_path, "r") as file:
            reader = csv.reader(file)
            store_rows = list(reader)
            last_store_row = store_rows[-1]
            print("The number of rows we have processed in the store csv is ", len(store_rows))

    start_time = time.time()
    info_lists = []
    with open(csv_file_path) as file_obj:
        reader_obj = csv.reader(file_obj)

        for row_idx, row in enumerate(reader_obj):
            if row_idx == 0:
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                new_addition_content = ["valid_identities"]
                print("The first row is ", row + new_addition_content)

                if not resume:
                    with open(store_file_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([row + new_addition_content])
                continue

            video_path = row[elements["video_path"]]
            vipe_output_filepath = row[elements["vipe_output_filepath"]]

            if resume:
                if video_path == last_store_row[elements["video_path"]]:
                    print("We find resume at", row_idx)
                    find_resume = True
                    continue

            if not find_resume:
                continue

            result = verify_identities(
                video_path, vipe_output_filepath, llm, processor, sampling_params
            )
            # json keys as strings for stable csv cells
            payload = {str(k): int(v) for k, v in sorted(result.items())}
            info_lists.append(row + [json.dumps(payload)])
            print("Finished Instance", str(row_idx), "valid=", payload, "\n")

            if row_idx % store_freq == 0:
                print("We have processed ", float(row_idx / 1000), "K video")
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" % (full_time_spent // 60, full_time_spent % 60))

                with open(store_file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)

                info_lists = []
                gc.collect()
                torch.cuda.empty_cache()

        with open(store_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU_offset", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = "/scratch/uft5by/Qwen3.8-27B-Instruct"
    input_csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_vipe"
    store_csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_verified_identities"
    GPU_offset = args.GPU_offset
    resume = False
    debug = True

    processor = AutoProcessor.from_pretrained(checkpoint_path)
    print("Loaded processor... \n")

    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        tensor_parallel_size=1,
        seed=0,
        max_model_len=2200,
        enable_prefix_caching=True,
        max_num_seqs=8,
        dtype = "bfloat16"
    )
    print("Instantiated VLM \n")

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024
    )

    if not os.path.exists(store_csv_folder_path):
        os.makedirs(store_csv_folder_path)

    single_process(
        input_csv_folder_path,
        store_csv_folder_path,
        GPU_offset,
        llm,
        processor,
        sampling_params,
    )

    print("Finished!")
