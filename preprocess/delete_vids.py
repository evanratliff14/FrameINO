import glob
import os

import pandas as pd


if __name__ == "__main__":

    input_csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vlm"
    video_folder = "/scratch/uft5by/OpenVid-1M/videos"

    pattern = os.path.join(input_csv_folder_path, "sub*.csv")
    filepaths = glob.glob(pattern)
    csv = pd.concat([pd.read_csv(fp) for fp in filepaths], ignore_index=True)

    keep_list = csv[["video_path"]].copy()
    keep_list["sort_key"] = keep_list["video_path"].astype(str).str.strip().str.split("/").str[-1]
    keep_list = keep_list.sort_values("sort_key").reset_index(drop=True)

    scratch_videos = os.listdir(video_folder)
    scratch_videos = sorted(scratch_videos)

    dry_run = True

    l = 0
    count = 0
    i = 0
    keep_keys = keep_list["sort_key"].tolist()
    for file in scratch_videos:
        while i < len(keep_keys) and keep_keys[i] < file:
            i += 1
        if i < len(keep_keys) and keep_keys[i] == file:
            l += 1
            i += 1
        else:
            count += 1
            if not dry_run:
                os.remove(os.path.join(video_folder, file))

    print(f"kept={l}  deleted={count}  dry_run={dry_run}")
