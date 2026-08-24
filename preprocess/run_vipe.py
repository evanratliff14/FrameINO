'''
    Run ViPE inference over a sharded csv of videos.

    Loads the annotation pipeline once and reuses cached models across videos.
    Per-row start/end frames are applied via SliceStreamProcessor on RawMp4Stream.

    Launch from preprocess/ with the vipe project env, e.g.:
      uv run --project vipe python run_vipe.py --GPU_offset 0
'''

import argparse
import ast
import csv
import os
import sys
import time
from pathlib import Path

csv.field_size_limit(sys.maxsize)
from preprocess.vipe.vipe.streams.base import ProcessedVideoStream, SliceStreamProcessor
from preprocess.vipe.vipe.streams.raw_mp4_stream import RawMp4Stream
from preprocess.vipe.vipe import make_pipeline
from preprocess.vipe.vipe.config import parse_typed_config

script_dir = os.path.dirname(os.path.abspath(__file__))
import torch


def build_pipeline(pipeline_name: str, output_root: str):

    overrides = [
        f"pipeline={pipeline_name}",
        f"pipeline.output.path={output_root}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
    ]
    args = parse_typed_config("default", hydra_args=overrides)
    return make_pipeline(args.pipeline)


def run_one_video(vipe_pipeline, video_path: str, start_frame: int, end_frame: int, out_path: str):
    

    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # preserve per-video artifact root used by downstream loaders
    vipe_pipeline.out_path = out_dir
    vipe_pipeline.out_cfg.path = str(out_dir)

    slice_processor = SliceStreamProcessor(start_frame=start_frame, end_frame=end_frame)
    video_stream = ProcessedVideoStream(
        RawMp4Stream(Path(video_path)),
        [slice_processor],
    ).cache(desc="Reading video stream")

    print(
        f"Running ViPE on {video_path} frames [{start_frame}, {end_frame}) -> {out_path}",
        flush=True,
    )
    vipe_pipeline.run(video_stream)


def single_process(csv_folder_path, store_folder_path, GPU_offset, basepath, pipeline_name):
    store_freq = 10

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA machine not available!")

    results_root = os.path.join(basepath, "OpenVid-1M", "csv", "vipe_results")
    os.makedirs(results_root, exist_ok=True)

    print("Loading ViPE pipeline (models cached for all videos in this process)...", flush=True)
    vipe_pipeline = build_pipeline(pipeline_name, results_root)
    print("ViPE pipeline ready.", flush=True)

    csv_file_path = os.path.join(csv_folder_path, f"sub{GPU_offset}.csv")
    print("CSV file we read is ", csv_file_path)

    store_file_path = os.path.join(store_folder_path, f"sub{GPU_offset}.csv")
    if os.path.exists(store_file_path):
        os.remove(store_file_path)

    start_time = time.time()
    info_lists = []
    batch_to_write = []

    with open(csv_file_path, mode="r", encoding="utf-8") as file_obj:
        reader_obj = csv.reader(file_obj)

        for idx, row in enumerate(reader_obj):

            if idx == 0:
                elements = {key: element_idx for element_idx, key in enumerate(row)}

                # Update header with added outputs
                header_row = row + ["used_range", "vipe_output_filepath"]
                print("The first row is ", header_row)

                with open(store_file_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header_row)
                continue

            video_path = row[elements["video_path"]]

            # Safely evaluate string representation of lists
            try:
                valid_ranges = ast.literal_eval(row[elements["scene_cut"]])
            except (ValueError, SyntaxError):
                continue

            first_valid_range = None
            for r in valid_ranges:
                # Correct range check: end_frame - start_frame >= 100
                if r[1] - r[0] >= 100:
                    first_valid_range = r
                    break

            if first_valid_range is None:
                continue

            valid_duration_col = row[elements["valid_duration"]]
            valid_duration_val = ast.literal_eval(valid_duration_col) if isinstance(valid_duration_col, str) else valid_duration_col

            base_offset = valid_duration_val[0] if isinstance(valid_duration_val, (list, tuple)) else int(valid_duration_val)

            start_frame = base_offset + first_valid_range[0]
            end_frame = base_offset + first_valid_range[1]

            vipe_output_filepath = os.path.join(
                results_root, os.path.basename(video_path).split(".")[0]
            )

            run_one_video(
                vipe_pipeline,
                video_path,
                start_frame,
                end_frame,
                vipe_output_filepath,
            )

            # Construct row payload (avoid dictionary assignment on plain list)
            processed_row = row + [str(first_valid_range), vipe_output_filepath]

            info_lists.append(processed_row)
            batch_to_write.append(processed_row)

            if len(batch_to_write) >= store_freq:
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
                print(f"Peak Tensor VRAM Used: {peak_memory:.2f} MB")
                print(f"Peak Total VRAM Cached: {peak_reserved:.2f} MB")

                print(f"We have processed {float(len(info_lists) / 1000):.3f}K valid videos (Total rows scanned: {idx})")
                full_time_spent = int(time.time() - start_time)
                print(f"Time spent is {full_time_spent // 60} min {full_time_spent % 60} s", flush=True)

                # Append buffered rows to CSV
                with open(store_file_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(batch_to_write)

                # Clear batch buffer
                batch_to_write.clear()

        # Write remaining unprocessed rows in the buffer after loop completion
        if batch_to_write:
            with open(store_file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(batch_to_write)
            batch_to_write.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU_offset", type=int, default=0)
    parser.add_argument("--pipeline", type=str, default="default")
    args = parser.parse_args()

    basepath = "/scratch/uft5by"
    csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vlm_left"
    store_folder_path = "/scratch/uft5by/OpenVid-1M/objects/general_dataset_vipe"
    GPU_offset = args.GPU_offset

    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path, exist_ok=True)

    start_time = time.time()
    single_process(
        csv_folder_path,
        store_folder_path,
        GPU_offset,
        basepath,
        pipeline_name=args.pipeline,
    )
    full_time_spent = int(time.time() - start_time)
    print(
        f"Total time spent for process {GPU_offset} is {full_time_spent // 60} min {full_time_spent % 60} s",
        flush=True,
    )
