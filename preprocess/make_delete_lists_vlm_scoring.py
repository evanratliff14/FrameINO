"""
Cut all csvs in directory by VLM scoring columns from vlm_score.py.

Independent keep filters:
  - NSFW question score < 0.8
  - real-life question score < 0.5
  - text overlays / watermarks / borders question score < 0.5
  - sum(motion blur, vehicles/subjects, foreground occlusion) < --threshold
"""
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

PROMPT_TEXT = "The video has text overlays, watermarks, artificial borders, or multiple views?"
PROMPT_REAL = "The video is of real-life?"
PROMPT_BLUR = "Does video suffer from motion blur, camera jittering, or sudden viewpoint shift?"
PROMPT_OCC = (
    "Foreground occlusion examples: fog, heavy rain, or a person passing too close to the camera. "
    "Does video contain significant foreground occlusion?"
)
# PROMPT_OBJ = (
#     "Does the scene contain any vehicles, animals, humans, objects, tools, or other objects "
#     "suitable for moving around the scene artificially?"
# )
PROMPT_NSFW = (
    "Does the scene contain sexual, violent/gory, political, or any other 'Not-Safe-For-Work' content?"
)
PROMPT_OBJ_MOTION = "Does the scene contain any vehicles, animals, humans, objects, tools, or other objects suitable for moving around the scene artificially?"

NSFW_THRESH = 0.75
REAL_THRESH = 0.75
TEXT_THRESH = 0.75
OBJ_MOTION_THRESH = 0.21

def process_one(fp, output_filepath, threshold):
    df = pd.read_csv(fp)
    total_length = len(df)

    combo = df[PROMPT_BLUR] + df[PROMPT_OCC]
    df = df[
        (df[PROMPT_NSFW] >= NSFW_THRESH)
        & (df[PROMPT_REAL] >= REAL_THRESH)
        & (df[PROMPT_TEXT] >= TEXT_THRESH)
        & (combo >= threshold)
        & (df[PROMPT_OBJ_MOTION] >= OBJ_MOTION_THRESH)
    ]
    left_length = len(df)

    filename = os.path.basename(fp)
    out_fp = os.path.join(output_filepath, filename)
    df.to_csv(out_fp, index=False)
    return total_length, left_length


def main(input_filepath, output_filepath, num_workers, threshold):

    os.makedirs(output_filepath, exist_ok=True)

    pattern = os.path.join(input_filepath, "sub*.csv")
    filepaths = glob.glob(pattern)

    if not filepaths:
        print(f"No CSV files found matching pattern: {pattern}")
        return

    total_length = 0
    left_length = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_one, fp, output_filepath, threshold)
            for fp in filepaths
        ]
        for fut in as_completed(futures):
            t, l = fut.result()
            total_length += t
            left_length += l

    print(f"Filtering Complete. Saved data from {left_length} / {total_length} rows to '{output_filepath}'.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    input_filepath = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vlm"
    output_filepath = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vlm_left"
    argparser.add_argument("--num_workers", type=int, default=8)
    argparser.add_argument("--threshold", type=float, required=True,
                           help="keep rows where blur+vehicles+occlusion sum is < this value")
    args = argparser.parse_args()

    main(
        input_filepath=input_filepath,
        output_filepath=output_filepath,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )
