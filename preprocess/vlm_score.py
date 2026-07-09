'''
    Video Caption by Qwen VL 3.6 35B-A3B
'''

import os, sys, shutil
import csv
import gc
import time
import argparse
import signal
import ffmpeg
import json
import numpy as np
import math
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

csv.field_size_limit(sys.maxsize)       # Default setting is 131072, 10x expand should be enough
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def get_message(video_path, frame_start, frame_end, instruction_prompts):
# Messages containing a local video path and a text query


                                # video_np = video_full_np[valid_duration[0] : valid_duration[1]]

    messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "resized_height": target_height,
                                "resized_width": target_width,
                                "frame_start": frame_start,
                                "frame_end": frame_end,
                                # force it to process 2 fps (uniform subsample) with process_vision_info() util
                                "fps": 2.0,
                            },
                            {
                                "type": "text", 
                                "text": p,
                            },
                        ],
                    } for p in instruction_prompts
                ]
    return messages
        


def single_process(input_csv_folder_path, store_csv_folder_path, GPU_offset, llm, processor,legend, instruction_prompt, sampling_params):
    

    # Setting
    store_freq = 10
    device = 'cuda'


    # Read the csv file
    csv_idx = GPU_offset
    csv_file_path = os.path.join(input_csv_folder_path, "sub" + str(csv_idx) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store file path
    store_file_path = os.path.join(store_csv_folder_path, "sub" + str(csv_idx) + ".csv")
    if not resume and os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)

    # Resume the store csv  
    find_resume = True
    if resume:      # Read the last store row
        find_resume = False
        with open(store_file_path, 'r') as file:
            reader = csv.reader(file)
            store_rows = list(reader)
            last_store_row = store_rows[-1]
            print("The number of rows we have processed in the store csv is ", len(store_rows))
            

    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj:
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for row_idx, row in enumerate(reader_obj): 

            # For the first row case (With all title content)
            if row_idx == 0:    # The first line is the title of content
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                
                new_addition_content = [store_prompt_name]
                print("The first row is ", row + new_addition_content)

                # Store the first row to csv
                if not resume:
                    with open(store_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([row + new_addition_content])
                continue


            # Read important information
            video_path = row[elements["video_path"]]
            valid_duration = json.loads(row[elements["valid_duration"]])
            

            # Resume mode will execute until we have the last store row matched
            if resume:
                if video_path == last_store_row[elements["video_path"]] and valid_duration == json.loads(last_store_row[elements["valid_duration"]]):     # Check Video Path and the valid duration
                    print("We find resume at", row_idx)        # In caption, this row should match len(store_rows).  
                    find_resume = True      # We find the exact row we want
                    continue        # Should continue; else, we repeat the same one again.

            if not find_resume:
                continue

            
            try:

                frame_start =valid_duration[0]
                frame_end =valid_duration[1]
                messages = get_message(video_path, frame_start, frame_end, instruction_prompts=instruction_prompt)
                inputs = prepare_inputs_for_vllm(messages = messages, processor=processor)
                inputs = inputs.to("cuda")

                if debug:
                    for i, input_ in enumerate(inputs):
                        print()
                        print('=' * 40)
                        print(f"Inputs[{i}]: {input_['prompt']=!r}")
                    print('\n' + '>' * 40)

                ## ** python "dereference" unpacks the dictionary into arguments
                generated_ids = llm.generate(**inputs, sampling_params=sampling_params, batch_size = len(inputs))
                 
                if debug:
                    for i, output in enumerate(generated_ids):
                        generated_text = output.outputs[0].text
                        print()
                        print('=' * 40)
                        print(f"Generated text: {generated_text!r}")

                # take the prompt out of the output
                generated_ids_trimmed = [
                                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                        ]
                #decode
                output = processor.batch_decode(
                                                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                                    )[0]
                print("Output text is ", output, " for the video ", video_path, " of the range " + str(frame_start) + "-" + str(frame_end) + "\n")


                #now output is a list of logits
                output = np.array(output)
                output[output ==0]= -1
                # determine the desirability of each output
                output = output * legend
                # acquire scores of good (1) and bad (0) where before these scores represented T/F
                output = output[output==-1]= 0


                # Update the text prompt
                info_lists.append(row.to_list())
                print("Finished Instance", str(row_idx), "\n")

                # Clean cache
                gc.collect()

                # Log update (The update will be quite random for this file, because we may skip earlier on)
                if row_idx % store_freq == 0:
                    
                    print("We have processed ", float(row_idx/1000), "K video")
                    full_time_spent = int(time.time() - start_time)
                    print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

                    # Store the csv
                    with open(store_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(info_lists)

                    # Restart the info_lists
                    info_lists = [] 


            except Exception as error:
                print("There is exception case", error)
                continue   


        # Final Log update
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)




if __name__ == "__main__":

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()


    # Model and inputs outputs Setting       
    checkpoint_path = "Qwen/Qwen3.6-35B-A3B"     # Qwen2.5-VL-7B-Instruct  Qwen2.5-VL-72B-Instruct.  It seems that 32B is newer and competitive compared to 72B version
    input_csv_folder_path = "/PATH/TO/CSV_FOLDER/folder"                  # Input 
    store_csv_folder_path = "/PATH/TO/CSV_FOLDER/folder"          # Output
    GPU_offset = args.GPU_offset
    resume = True


    # Video Processing Setting
    store_prompt_name = "Structured_Text_Prompt"
    target_height = 256
    target_width = 384
    max_frames_consider = 160             # About 81 * 2
    sample_frame_freq = 16                # 原来是1fps，大改就是24个step; 目前更加dense一点的吧，设置16
    debug = True

    # Batch of prompts
    # Engineering: we prove binary 0,1 outcome with few tokens possible by asking there exists questions if possible
    instruction_prompt = [
        "The video has text overlays, watermarks, artificial borders, or multiple views?",
        "The video is of real-life?",
        "Does the scene contain enough stable background features to perform camera estimation?"
        "Is any part of the video heavily unfocused or motion-blurred?",
        # g. Are there major occlusions blocking subject from view? -> we will use this later
        "Is there at least one object and does the object move in at least two out of three camera axes",
        "Does the scene contain sexual, violent/gory, political, or any other 'Not-Safe-For-Work' content?"
        # "Does the camera move dynamically?"
    ]

     # 1 means that if the answer is true to the question[idx], then we give it a score of 1. -1 means that if the question is 
    # false, then we store a 1. Else store 0 
    legend = np.array([-1, 1, 1, -1, 1, -1])

    # Init the model
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        max_model_len=4096, # Caps total context sequence window to save huge VRAM allocations
        enable_prefix_caching=False,   # Disables memory retention between different queries
        max_num_seqs=64               # Allows vLLM to process up to 64 independent chunks in parallel
    )

    # retrieve the encodings of our target outputs
    token_id_0 = processor.tokenizer.encode("0", add_special_tokens=False)[0]
    token_id_1 = processor.tokenizer.encode("1", add_special_tokens=False)[0]

    # Apply a massive positive logit bias to ONLY these two tokens
    # Mathematically impossible for the model to pick anything else
    binary_logit_bias = {
        token_id_0: 100.0,
        token_id_1: 100.0
    }

    sampling_params = SamplingParams(
        temperature=0,          # deterministic
        max_tokens=1,           # only one token is needed now
        logit_bias=binary_logit_bias
    )
    

   


    if not os.path.exists(store_csv_folder_path):
        os.makedirs(store_csv_folder_path)


    # Inferece Process
    single_process(input_csv_folder_path, store_csv_folder_path, GPU_offset, llm, processor,legend, instruction_prompt, sampling_params)


    print("Finished!")


