'''
    track 6dof object poses with cotracker over verified vipe identities.
    reads verified-identity csv shards and writes padded Rt to
    /scratch/uft5by/OpenVid-1M/objects/<video_name>/f{id}.npz
'''

import os
import sys
import csv
import gc
import json
import time
import argparse

import numpy as np
import torch

csv.field_size_limit(sys.maxsize)

root_path = os.path.abspath(".")
sys.path.append(root_path)
script_dir = os.path.dirname(os.path.abspath(__file__))
# vipe_object_track / camera / depth use bare sibling imports

from preprocess.vipe_output_interface.vipe_masks import VipeMasks
from preprocess.vipe_output_interface.vipe_camera import Camera
from preprocess.vipe_output_interface.vipe_depth import VipeDepth
from preprocess.vipe_output_interface.vipe_io import read_rgb_frames
from preprocess.vipe_output_interface.vipe_object_track import track_objects_with_cotracker


def pad_rt_to_num_frames(frame_rt, num_frames):
    """
    build [num_frames, 4, 4] with identity fill.
    accepts dict[frame -> Rt] or an array already shaped [T, 4, 4] / [K, 4, 4].
    """
    padded = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))

    if isinstance(frame_rt, dict):
        for t, Rt in frame_rt.items():
            ti = int(t)
            if 0 <= ti < num_frames:
                padded[ti] = np.asarray(Rt, dtype=np.float64)
        return padded

    arr = np.asarray(frame_rt, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (4, 4):
        padded[0] = arr
        return padded

    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
        t = min(arr.shape[0], num_frames)
        padded[:t] = arr[:t]
        return padded

    raise ValueError(f"unsupported Rt structure with shape/type {type(frame_rt)} {getattr(frame_rt, 'shape', None)}")


def load_video_tensor(vipe_output_filepath):
    """load vipe rgb as float tensor [T, C, H, W]."""
    frames = [fr for _, fr in read_rgb_frames(vipe_output_filepath)]
    video_np = np.stack(frames, axis=0)
    return torch.from_numpy(video_np).permute(0, 3, 1, 2).contiguous()


def track_one_video(video_path, vipe_output_filepath, valid_identities, objects_root):
    """
    run cotracker 6dof for verified ids and write f{id}.npz under objects_root/video_name/.
    returns objects_dir path (or None if skipped).
    """
    if not valid_identities:
        return None

    valid_ids = {int(k) for k in valid_identities.keys()}
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    objects_dir = os.path.join(objects_root, video_name)
    os.makedirs(objects_dir, exist_ok=True)

    vipe_masks = VipeMasks(vipe_output_filepath)
    all_masks = vipe_masks.get_masks()
    masks = {iid: all_masks[iid] for iid in valid_ids if iid in all_masks}
    if not masks:
        print("no matching instance masks for", video_path, "valid_ids=", valid_ids)
        return None

    camera = Camera(vipe_output_filepath)
    depths = VipeDepth(vipe_output_filepath)
    video = load_video_tensor(vipe_output_filepath)
    num_frames = int(video.shape[0])
    if num_frames <= 0:
        return None

    results = track_objects_with_cotracker(
        video=video,
        masks=masks,
        camera=camera,
        depths=depths,
        end_frame=num_frames,
        start_frame=0,
    )

    for iid, frame_rt in results.items():
        padded = pad_rt_to_num_frames(frame_rt, num_frames)
        out_path = os.path.join(objects_dir, f"f{int(iid)}.npz")
        np.savez(out_path, Rt=padded)
        print("saved", out_path, "shape=", padded.shape)

    return objects_dir


def single_process(input_csv_folder_path, store_csv_folder_path, objects_root, GPU_offset):
    store_freq = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_offset)

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

                new_addition_content = ["objects_dir"]
                print("The first row is ", row + new_addition_content)

                if not resume:
                    with open(store_file_path, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([row + new_addition_content])
                continue

            video_path = row[elements["video_path"]]
            vipe_output_filepath = row[elements["vipe_output_filepath"]]
            valid_identities = json.loads(row[elements["valid_identities"]])

            if resume:
                if video_path == last_store_row[elements["video_path"]]:
                    print("We find resume at", row_idx)
                    find_resume = True
                    continue

            if not find_resume:
                continue

            print("This is instance", row_idx, "and we are processing", video_path)
            objects_dir = track_one_video(
                video_path, vipe_output_filepath, valid_identities, objects_root
            )
            info_lists.append(row + [objects_dir if objects_dir is not None else ""])
            print("Finished Instance", str(row_idx), "objects_dir=", objects_dir, "\n")

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

    input_csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_verified_identities"
    store_csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_track_6df"
    objects_root = "/scratch/uft5by/OpenVid-1M/objects"
    GPU_offset = args.GPU_offset
    resume = False

    if not os.path.exists(store_csv_folder_path):
        os.makedirs(store_csv_folder_path)
    if not os.path.exists(objects_root):
        os.makedirs(objects_root)

    single_process(
        input_csv_folder_path,
        store_csv_folder_path,
        objects_root,
        GPU_offset,
    )

    print("Finished!")
