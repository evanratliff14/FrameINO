import os, sys, shutil
import pandas as pd
import time
import csv
import ffmpeg
import imageio
import json
import copy
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool
import multiprocessing
import cv2
import argparse
import subprocess
import glob
from tqdm import tqdm
import logging
import torchvision
import torch
import re


logger = logging.getLogger(__name__)



# global variables so I don't have to pass into the worker task function (they are const)
csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_raw/"
store_folder_path = "/scratch/uft5by/OpenVid-1M/metadata/general_dataset_filter_basic"

# Filter Setting
min_num_frames_needed = 100     # ~ 49 * 2
max_num_frames_needed = 500     # If there are too many frames, it is not very ideal then
max_crop_iter_num = 1           # Cannot foreover crop long videos, which repeat one video too much
valid_fps_range = [20, 31]      # Exactly for 24/30 FPS
valid_aspect_ratio = 1.35       # Min valid aspect ratio Here we want to filter 1:1 case 
min_width_threshold = 400       # The height is 0.7 * min_width_threshold
crop_long_frames = True         # Whether we crop video that is too long to speed up

def single_process( csv_folder_path,
                    store_folder_path,
                    process_idx, 
                    min_num_frames_needed, 
                    max_num_frames_needed,
                    valid_aspect_ratio,
                    min_width_threshold,
                    crop_long_frames,
                    valid_fps_range
                    ):

    # Setting
    store_freq = 50


    # Read the csv file
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(process_idx) + ".csv")
    store_file_path = os.path.join(store_folder_path, "sub" + str(process_idx) + ".csv")
    logger.info("We are processing %s", csv_file_path)


    # Prepare the folder
    if os.path.exists(store_file_path):
        os.remove(store_file_path)


    # Read all row in the csv file
    info_lists = []
    start_time = time.time()
    invalid_num_frames_too_small, invalid_num_frames_too_many, invalid_aspect_ratio, invalid_resolution, invalid_fps, invalid_duration = [], [], [], [], [], []

    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for idx, row in enumerate(reader_obj): 
            if idx == 0:    # The first line is the title of content

                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                # Append the first row with all index information
                row.extend(["height", "width", "num_frames", "fps", "total_seconds", "valid_duration"])   # Add all new elements   "num_frames" is already incuded
                info_lists.append(row)
                logging.debug("The first row is %s ", row)
                continue


            elif idx % store_freq == 0:    # Store and update the log
                
                # Log
                logging.info(f"We have processed %d videos; process idx %d", idx, process_idx)
                logging.info("The number of valid videos we found in this iter is %d", len(info_lists))
                full_time_spent = int(time.time() - start_time)
                logging.info("Time spent is %d min %d s", full_time_spent//60, full_time_spent%60)

                # Store the csv
                with open(store_file_path, 'a', newline='', encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)

                # Restart the info_lists
                info_lists = []    


            # Try to read basic information and see if it is still valid
            try:

                video_path = row[elements["video_path"]]
                
                # this is a heavily utilized hack that is common
                # after this is run, ffmpeg itself will give us statistics in stderr that we can parse
                # in essence we do minimal overhead and let ffmpeg count statistics for us
                cmd = [
                    'ffmpeg', 
                    '-i', video_path, 
                    # fetches video
                    '-map', '0:v:0', 
                    # gets stream copy of compressed pixels
                    '-c', 'copy', 
                    # we save nothing in RAM
                    '-f', 'null', '-'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                #see above on why we look at stderr
                stderr_output = result.stderr

                # get frame count
                match_frames = re.search(r'frame=\s*(\d+)', stderr_output)
                num_frames = int(match_frames.group(1)) if match_frames else 0
                
                # get resolution
                match_res = re.search(r"Video:.*,\s+(\d+)x(\d+)", stderr_output)
                if match_res:
                    width = int(match_res.group(1))
                    height = int(match_res.group(2))
                else:
                    width, height = 0, 0
                    
                # getfps
                match_fps = re.search(r"(\d+(?:\.\d+)?)\s+fps", stderr_output)
                fps = float(match_fps.group(1)) if match_fps else 0.0

                valid_duration = [0, num_frames]         # Duration for the range of frame idx we will read, very fixed
                aspect_ratio = width / height
                
                # get total_seconds
                match_time = re.search(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})', stderr_output)
                if match_time:
                    hours = int(match_time.group(1))
                    minutes = int(match_time.group(2))
                    seconds = int(match_time.group(3))
                    centiseconds = int(match_time.group(4))
                    
                    total_seconds = (hours * 3600) + (minutes * 60) + seconds + (centiseconds / 100.0)
                else:
                    # Fallback math if time string parsing fails
                    total_seconds = num_frames / fps if fps != 0 else 0.0
            

                # Set threshold for the invalid fps
                aspect_ratio = width / height
                if aspect_ratio < valid_aspect_ratio:
                    logging.error("The aspect ratio is not ideal: %f", aspect_ratio)
                    invalid_aspect_ratio.append([video_path, "Invalid Aspect Ratio at " + str(aspect_ratio)])
                    continue


                # Filter for those whose resolution is too small
                if width < min_width_threshold or height < 0.7 * min_width_threshold:
                    logging.error("The width is too small: %d", width)
                    invalid_resolution.append([video_path, "Invalid Resolution with width and height " + str(width) + ", " + str(height)])
                    continue
                
                # Check the FPS
                if fps < valid_fps_range[0] or fps > valid_fps_range[1]:
                    logging.error("The fps is not ideal: %d", fps)
                    invalid_fps.append([video_path, "Invalid FPS at " + str(fps)])
                    continue

                # Check threshold for frame num
                if num_frames <= min_num_frames_needed:
                    logging.error("The number of frames is too small: %d from video %s", num_frames, video_path)
                    invalid_num_frames_too_small.append([video_path, "Invalid Number of frame of " + str(num_frames)])
                    continue
                

                # For more frames available, we can choose to crop the video
                if num_frames >= max_num_frames_needed:
                    logging.error("The number of frames is too many: %d", num_frames)

                    # Crop the video
                    if crop_long_frames:
                        
                        # Rewrite the valid duratio range
                        crop_section_num = min(num_frames // max_num_frames_needed, max_crop_iter_num)
                        
                        for crop_idx in range(crop_section_num):
                            # Find the valid duration
                            valid_duration = [crop_idx * max_num_frames_needed, (crop_idx + 1) * max_num_frames_needed]

                            # Extend to a copy of the existing information
                            existing_row = copy.deepcopy(row)
                            existing_row.extend([height, width, num_frames, fps, total_seconds, json.dumps(valid_duration)])
                            info_lists.append(existing_row)

                        continue

                    else:
                        invalid_num_frames_too_many.append([video_path, "Invalid Number of frame of " + str(num_frames)])
                        continue


                # Record the valid one
                row.extend([height, width, num_frames, fps, total_seconds, json.dumps(valid_duration)])
                info_lists.append(row)

            except Exception as Error:
                logging.error("error as %s", str(Error))
                continue
    


    logging.info("invalid_num_frames_too_small %d, invalid_num_frames_too_many %d, invalid_aspect_ratio %d, invalid_resolution %d, invalid_fps %d, and invalid_duration %d", 
          len(invalid_num_frames_too_small), len(invalid_num_frames_too_many), len(invalid_aspect_ratio), len(invalid_resolution), len(invalid_fps), len(invalid_duration))
    # print("Valid video num is ", len(info_lists))

    # Store the csv for the remaining information
    with open(store_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(info_lists)

    return 1



def worker_task(csv_file_path):
    """Wrapper function to process a single CSV file chunk."""
    # Extract the chunk index from the filename
    filename = os.path.basename(csv_file_path)
    process_idx = int(''.join(filter(str.isdigit, filename)))
    
    single_process(
        csv_folder_path=csv_folder_path,
        store_folder_path=store_folder_path,
        process_idx=process_idx,
        min_num_frames_needed=min_num_frames_needed,
        max_num_frames_needed=max_num_frames_needed,
        valid_aspect_ratio=valid_aspect_ratio,
        min_width_threshold=min_width_threshold,
        valid_fps_range = valid_fps_range,
        crop_long_frames = crop_long_frames,  
    )
    return 1

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    all_csv_files = glob.glob(os.path.join(csv_folder_path, "sub*.csv"))

    # Prepare the output directory
    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path, exist_ok=True)

    print(f"Found {len(all_csv_files)} chunks. Processing with {args.num_workers} workers...")

    with Pool(processes=args.num_workers) as pool:
        # Total is exactly the number of CSV files to process
        with tqdm(total=len(all_csv_files), desc="Processing CSV Chunks") as pbar:
            for result_row_count in pool.imap_unordered(worker_task, all_csv_files):
                pbar.update(1) 
