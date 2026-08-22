'''
    Run ViPE inference over a sharded csv of videos.
'''

import os, sys, shutil
import time
import csv
import argparse
import subprocess
import torch

csv.field_size_limit(sys.maxsize)


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the script's directory (instead of root_path = '.')
sys.path.append(script_dir)

vipe_dir = os.path.join(script_dir)


def single_process(csv_folder_path, store_folder_path, GPU_offset, basepath):

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


    if not torch.cuda.is_available():
        raise Exception("We should have a cuda machine available!")


    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj:

        reader_obj = csv.reader(file_obj)

        for idx, row in enumerate(reader_obj):

            # For the first row case (With all title content)
            if idx == 0:    # The first line is the title of content
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                info_lists.append(row + ["vipe_output_filepath"])
                print("The first row is ", info_lists[0])

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)
                continue

            video_path = row[elements["video_path"]]
            valid_ranges = list(row[elements["SceneCut_Autoshot"]])
            first_valid_range = None
            for range in valid_ranges:
                if range[0] -range[1] >=100:
                    first_valid_range = range
                    break
            # the video will not be included in the output csv
            if first_valid_range == None:
                continue

            start_frame = row[list(elements[valid_duration])[0]] + first_valid_range[0]
            end_frame = row[list(elements[valid_duration])[0]] + first_valid_range[1]
            
            vipe_output_filepath = os.path.join(
                basepath, "OpenVid-1M", "csv", "vipe_results", os.path.basename(video_path)
            )
            os.makedirs(os.path.dirname(vipe_output_filepath), exist_ok=True)

            cmd = ["uv", "run", "--project", "vipe", "vipe", "infer", video_path, "--start_frame", start_frame, "--end_frame", end_frame, "-o", vipe_output_filepath]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=vipe_dir, check=True)

            row.append(vipe_output_filepath)
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
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            left_amount = idx % store_freq
            writer.writerows(info_lists[-1*left_amount:])



if __name__ == "__main__":

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()


    # Fundamental Setting
    basepath = "/scratch/uft5by"
    csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vlm_left"       # Input
    store_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_vipe"                       # Output
    GPU_offset = args.GPU_offset



    # Prepare the csv file
    if not os.path.exists(store_folder_path):
        # shutil.rmtree(store_folder_path)
        os.makedirs(store_folder_path)

    # Our sbatch will have 32 of these scripts, one for each GPU
    start_time = time.time()
    single_process(csv_folder_path, store_folder_path, GPU_offset, basepath)
    full_time_spent = int(time.time() - start_time)
    print("Total time spent for this video is %d min %d s" %(full_time_spent//60, full_time_spent%60), flush=True)
