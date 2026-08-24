'''
    Run ViPE inference over a sharded csv of videos.
'''

import argparse
import ast
import csv
import os
import subprocess
import sys
import time
import torch

csv.field_size_limit(sys.maxsize)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
vipe_dir = script_dir


def single_process(csv_folder_path, store_folder_path, GPU_offset, basepath):
    store_freq = 10

    # Ensure subprocesses target the correct GPU assigned to this shard
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_offset)

    csv_file_path = os.path.join(csv_folder_path, f"sub{GPU_offset}.csv")
    print("CSV file we read is ", csv_file_path)

    store_file_path = os.path.join(store_folder_path, f"sub{GPU_offset}.csv")
    if os.path.exists(store_file_path):
        os.remove(store_file_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA machine not available!")

    start_time = time.time()
    
    # Decouple in-memory processing tracking from disk write buffering
    info_lists = []       # Total processed rows
    batch_to_write = []   # Buffer specifically for periodic CSV writes

    with open(csv_file_path, mode="r", encoding="utf-8") as file_obj:
        reader_obj = csv.reader(file_obj)

        for idx, row in enumerate(reader_obj):

            # Handle CSV Header
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

            # Safely evaluate frame offsets
            valid_duration_col = row[elements["valid_duration"]]
            valid_duration_val = ast.literal_eval(valid_duration_col) if isinstance(valid_duration_col, str) else valid_duration_col
            
            # If valid_duration is a list/tuple, grab the first element; otherwise use as int
            base_offset = valid_duration_val[0] if isinstance(valid_duration_val, (list, tuple)) else int(valid_duration_val)

            start_frame = base_offset + first_valid_range[0]
            end_frame = base_offset + first_valid_range[1]

            vipe_output_filepath = os.path.join(
                basepath, "OpenVid-1M", "csv", "vipe_results", os.path.basename(video_path).split(".")[0]
            )
            os.makedirs(os.path.dirname(vipe_output_filepath), exist_ok=True)

            cmd = [
                "uv", "run", "--project", "vipe", "vipe", "infer",
                str(video_path),
                "--start_frame", str(start_frame),
                "--end_frame", str(end_frame),
                "-o", vipe_output_filepath
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=vipe_dir, check=True)

            # Construct row payload (avoid dictionary assignment on plain list)
            processed_row = row + [str(first_valid_range), vipe_output_filepath]
            
            info_lists.append(processed_row)
            batch_to_write.append(processed_row)

            # Periodic batch writing based on actual written buffer length
            if len(batch_to_write) >= store_freq:
                if torch.cuda.is_available():
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
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()

    basepath = "/scratch/uft5by"
    csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vlm_left"
    store_folder_path = "/scratch/uft5by/OpenVid-1M/objects/general_dataset_vipe"
    GPU_offset = args.GPU_offset

    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path, exist_ok=True)

    start_time = time.time()
    single_process(csv_folder_path, store_folder_path, GPU_offset, basepath)
    full_time_spent = int(time.time() - start_time)
    print(f"Total time spent for process {GPU_offset} is {full_time_spent // 60} min {full_time_spent % 60} s", flush=True)