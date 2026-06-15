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
from Koala_36M.training_suitability_assessment import inference
from torchvision.io import read_video

csv.field_size_limit(sys.maxsize)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)



@torch.no_grad
def single_process( csv_folder_path,
                    store_folder_path,
                    GPU_offset
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

    model = inference.get_model(device=device)

    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        cur_idx = 0
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

            try:
                # see inference.py in koala36M
                # Read the video by ffmpeg
                video_path = row[elements["video_path"]]
                valid_duration = json.loads(row[elements["valid_duration"]])
                video_tensor, audio_tensor, metadata = read_video(video_path, output_format="TCHW")
                video_tensor = video_tensor[valid_duration[0] : valid_duration[1]]

                result = inference.call(model, video_tensor, min(5, video_tensor.size(0)//100))

            except Exception as e:
                print("Exception in inference: ", e, flush=True)
                continue
                
            row.append(result)
            info_lists.append(row)

            # Log update
            if idx % store_freq == 0:
                print(f"Result (freq {store_freq}): {result}")
                print("We have processed ", float(idx/1000), "K video")
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

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
    args = parser.parse_args()


    # Fundamental Setting
    csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_SceneCut_left"       # Input
    store_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vtss"               # Output
    tmp_folder_name = "tmp_img_scoring/"        # temporary folder to store intermediate result
    GPU_offset = args.GPU_offset



    # Prepare the csv file
    if not os.path.exists(store_folder_path):
        # shutil.rmtree(store_folder_path)
        os.makedirs(store_folder_path)

    # Our sbatch will have 32 of these scripts, one for each GPU
    start_time = time.time()
    single_process(csv_folder_path, store_folder_path, GPU_offset)
    full_time_spent = int(time.time() - start_time)
    print("Total time spent for this video is %d min %d s" %(full_time_spent//60, full_time_spent%60), flush=True)


