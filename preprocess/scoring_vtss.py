'''
    Get the VTSS Video Training suitability score from Panda 36M. the score consists of several sub metrics inputs that it calculates on its own
    Therefore, we separate this script from scoring_img.py to investigate the usability of this metric
'''

import os, sys, shutil
import pandas as pd
import time
import csv
import collections
from multiprocessing import Process
import multiprocessing
import cv2
import ffmpeg
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import argparse
import torch
import json
from torchvision.io import read_video
import math

import torch
from decord import VideoReader, cpu, gpu

def read_video_to_tensor(video_path, valid_duration):
    # if torch.cuda.is_available():
    #     vr = VideoReader(video_path, ctx=gpu(0))
    # else:
    # loading on cpu is safer
    vr = VideoReader(video_path, ctx = cpu(0))

    # we only load the necessary frames to avoid OOM
    tensor_frames = vr.get_batch(range(valid_duration[0], valid_duration[1])) # Returns an NDArray
    
    tensor = torch.from_numpy(tensor_frames.asnumpy())
    
    # Permute from [T, H, W, C] to [T, C, H, W]
    return tensor.permute(0, 3, 1, 2)


csv.field_size_limit(sys.maxsize)


# Import files from the local folder
# root_path = os.path.abspath('.')
# sys.path.append(root_path)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the script's directory (instead of root_path = '.')
sys.path.append(script_dir)

# Append the nested folder relative to the script's directory
assessment_path = os.path.join(script_dir, "Koala_36M", "training_suitability_assessment")
sys.path.append(assessment_path)

import call_inference


def print_model_size(model):
    # 1. Count different parameter groups
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 2. Calculate actual memory footprint based on data types
    param_mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_mem_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_mem_bytes = param_mem_bytes + buffer_mem_bytes
    
    # Convert bytes to Megabytes
    total_mem_mb = total_mem_bytes / (1024 ** 2)
    
    print("=========================================")
    print(f"       MODEL PARAMETERS & SIZE           ")
    print("=========================================")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Params:      {trainable_params:,}")
    print(f"Non-Trainable Params:  {non_trainable_params:,}")
    print("-----------------------------------------")
    print(f"Estimated GPU VRAM:    {total_mem_mb:.2f} MB")
    print("=========================================")

@torch.no_grad
def single_process( csv_folder_path,
                    store_folder_path,
                    GPU_offset,
                    clip_length,
                    num_clips
                ):

    # Setting
    store_freq = 10


    # Read the csv file
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(GPU_offset) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store file path
    store_file_path = os.path.join(store_folder_path, "sub" + str(GPU_offset) + ".csv")
    if os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)




    # Init model with different Device
    if not torch.cuda.is_available():
        raise Exception("We should have a cuda machine available!")
    device = torch.device("cuda")

    model = call_inference.get_model(device=device)
    model.eval()
    print_model_size(model)

    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        cur_idx = 0
        transform = transforms.Resize((224, 224)) 

        for idx, row in enumerate(reader_obj): 

            # For the first row case (With all title content)
            if idx == 0:    # The first line is the title of content
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                info_lists.append(row + ["vtss"])
                print("The first row is ", info_lists[0])

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)
                continue

            # see inference.py in koala36M
            # Read the video by ffmpeg
            video_path = row[elements["video_path"]]
            valid_duration = json.loads(row[elements["valid_duration"]])
            video_tensor  = read_video_to_tensor(video_path, valid_duration)
            video_tensor = transform(video_tensor)
            
            
            with torch.no_grad():
                # type and normalize from raw format
                video_tensor = video_tensor.to(torch.float32)
                video_tensor = video_tensor / 255.0
                video_tensor = video_tensor[valid_duration[0] : valid_duration[1], ...]
                video_tensor = video_tensor.unsqueeze(0)

                valid_duration_length = valid_duration[1]-valid_duration[0]
                video_tensor = video_tensor.to(device)
                clip_length = clip_length
                
                result = call_inference.call(model, video_tensor, num_clips, clip_length)
                
                result = torch.mean(result).item()
                print("Result:", result, flush=True)


                
            row.append(result)
            info_lists.append(row)

            # Log update
            if idx % store_freq == 0:
                if torch.cuda.is_available():
                    # Returns the maximum memory occupied by tensors in bytes
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) 
                    # Returns the maximum memory managed by the caching allocator
                    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
                    
                    print(f"Peak Tensor VRAM Used: {peak_memory:.2f} MB")
                    print(f"Peak Total VRAM Cached: {peak_reserved:.2f} MB")
                print("We have processed ", float(idx/1000), "K video")
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60), flush=True)


                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists[-1*store_freq:])
                

        # Last append for the rest; the following might raise bugs
        # with open(store_file_path, 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     left_amount = idx % store_freq
        #     writer.writerows(info_lists[-1*left_amount:])



if __name__ == "__main__":

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    parser.add_argument('--num_clips', type=int, default = 5)
    parser.add_argument('--clip_length', type=int, default = 16)
    args = parser.parse_args()


    # Fundamental Setting
    csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_SceneCut_left"       # Input
    store_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vtss"               # Output
    GPU_offset = args.GPU_offset
    num_clips = args.num_clips
    clip_length = args.clip_length



    # Prepare the csv file
    if not os.path.exists(store_folder_path):
        # shutil.rmtree(store_folder_path)
        os.makedirs(store_folder_path)

    # Our sbatch will have 32 of these scripts, one for each GPU
    start_time = time.time()
    single_process(csv_folder_path, store_folder_path, GPU_offset, num_clips=num_clips, clip_length=clip_length)
    full_time_spent = int(time.time() - start_time)
    print("Total time spent for this video is %d min %d s" %(full_time_spent//60, full_time_spent%60), flush=True)


