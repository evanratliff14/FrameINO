'''
    Video filtering by Qwen VL 3.6 27B
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
import traceback
try:
    from vllm.logprobs import Logprob
except ImportError:
    from vllm.sequence import Logprob

csv.field_size_limit(sys.maxsize)       # Default setting is 131072, 10x expand should be enough
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# having troubles flash infreer
# os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"


def prepare_inputs_for_vllm(messages, processor, video_inputs, video_kwargs):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking =False)
    
    return {
        'prompt': text,
        # reuse decoded frames b/c we share the video across the list of questions
        'multi_modal_data': {"video": video_inputs[0]},
        'mm_processor_kwargs': video_kwargs
    }

def get_messages(video_path, frame_start, frame_end, instruction_prompt):
# Messages containing a local video path and a text query


                                # video_np = video_full_np[valid_duration[0] : valid_duration[1]]
    video = {
                "type": "video",
                "video": video_path,
                "resized_height": target_height,
                "resized_width": target_width,
                "frame_start": frame_start,
                "frame_end": frame_end,
                # force it to process 2 fps (uniform subsample) with process_vision_info() util
                "fps": 2.0,
            }
    messages = [
                    {
                        "role": "user",
                        "content": [
                            video,
                            {
                                "type": "text", 
                                "text": p,
                            },
                        ],
                    } for p in instruction_prompt
                ]
    
    return messages
        


def single_process(input_csv_folder_path, store_csv_folder_path, GPU_offset, llm, processor,legend, instruction_prompt, sampling_params):
    

    # Setting
    store_freq = 10


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

                
                new_addition_content = instruction_prompt
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
                messages = get_messages(video_path, frame_start, frame_end, instruction_prompt=instruction_prompt)

                 # qwen_vl_utils 0.0.14+ reqired
                 # only decord decode the video once
                _, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=processor.image_processor.patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=True
                )

                inputs = [prepare_inputs_for_vllm(messages = [system_prompt, m], processor=processor,
                            video_inputs=video_inputs, video_kwargs = video_kwargs) for m in messages
                ]
                if row_idx ==0 and debug:
                    print([m.prompt for m in inputs], flush=True)

                out = llm.generate(
                    inputs, 
                    sampling_params=sampling_params
                    # batch_size = len(inputs)
                )

                if debug and row_idx%10 == 0:
                        
                    for o in out:
                        prompt = o.prompt
                        generated_text = o.outputs[0].text   # <-- the string you want
                        print(f"Prompt: {prompt!r}\nGenerated: {generated_text!r}", flush=True)

                probs = []
                for i, o in enumerate(out):
                    # for each item in the batch, we extract the first token's log probs
                    token_data = o.outputs[0].logprobs[0]

                    log_prob_0 = token_data.get(token_id_0, float('-inf'))
                    log_prob_1 = token_data.get(token_id_1, float('-inf'))

                    if isinstance(log_prob_0, Logprob):
                        log_prob_0 = log_prob_0.logprob
                    if isinstance(log_prob_1, Logprob):
                        log_prob_1 = log_prob_1.logprob
                    
                    # Convert log-probabilities back to standard linear absolute probabilities
                    # e.g., e^(log_p) = p
                    prob_0 = math.exp(log_prob_0) if log_prob_0 != float('-inf') else 0.0
                    prob_1 = math.exp(log_prob_1) if log_prob_1 != float('-inf') else 0.0


                    print(f"Absolute probability of token '0': {prob_0:.6f} ({prob_0 * 100:.2f}%)")
                    print(f"Absolute probability of token '1': {prob_1:.6f} ({prob_1 * 100:.2f}%)")
                    
                    total_prob = prob_1 + prob_0
                    total_prob += 1e-10
                    prob_0, prob_1 = prob_0*(1/total_prob),  prob_1*(1/total_prob)

                    print(f"Normalized probability of token '0': {prob_0:.6f} ({prob_0 * 100:.2f}%)")
                    print(f"Normalized probability of token '1': {prob_1:.6f} ({prob_1 * 100:.2f}%)")

                    probs.append(prob_1)


                #now output is a list of logits
                output = np.array(probs)
                # determine the desirability of each output - 1-(p(true)) if legend has -1, else we take p(true)
                output = output * legend + helper
                # acquire scores between 0-1 representing a bad-good range


                # Update the text prompt
                info_lists.append(row + output.tolist())
                print("Finished Instance", str(row_idx), "\n")

                

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
                    
                    gc.collect()
                    torch.cuda.empty_cache()


            except Exception as error:
                print("There is exception case", error)
                if debug:
                    traceback.print_exc()
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
    checkpoint_path = "/scratch/uft5by/Qwen3.6-27B"     # Qwen2.5-VL-7B-Instruct  Qwen2.5-VL-72B-Instruct.  It seems that 32B is newer and competitive compared to 72B version
    input_csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_scene_cut_left"                  # Input 
    store_csv_folder_path = "/scratch/uft5by/OpenVid-1M/csv/general_dataset_scoring_vlm"          # Output
    GPU_offset = args.GPU_offset
    resume = False


    # Video Processing Setting
    store_prompt_name = "Structured_Text_Prompt"
    target_height = 256
    target_width = 384
    debug = True

    # Batch of prompts
    # Engineering: we prove binary 0,1 outcome with few tokens possible by asking there exists questions if possible
    instruction_prompt = [
        "The video has text overlays, watermarks, artificial borders, or multiple views?",
        "The video is of real-life?",
        "Does video suffer from motion blur, camera jittering, or sudden viewpoint shift?",
        "Foreground occlusion examples: fog, heavy rain, or a person passing too close to the camera. Does video contain significant foreground occlusion?",
        "Does the scene contain any vehicles, animals, humans, objects, tools, or other objects suitable for moving around the scene artificially?",
        "Does the scene contain sexual, violent/gory, political, or any other 'Not-Safe-For-Work' content?",
        "Does the scene contain an object that leaves the frame of view at any time?",
        "Does the scene contain an object that enters the frame of view at any time?"
    ]

     # 1 means that if the answer is true to the question[idx], then we give it a score of 1. -1 means that if the question is 
    # false, then we store a 1. Else store 0 
    legend = np.array([-1, 1, -1, -1, 1, -1, 1, 1])
    # we use this construction for flipping logit scores
    helper = np.array([1 if k==-1 else 0 for k in legend])



    # Init the model
    processor = AutoProcessor.from_pretrained(
        checkpoint_path
    )
    print(f"Loaded processor... \n")

    system_prompt = {"role": "system", "content": "Answer Format: Output exactly 1 for Yes or 0 for No. Never include prose, markdown, or spaces."}
    
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        tensor_parallel_size=1,
        seed=0,
        max_model_len=2200, # Caps total context sequence window. experiencing 2145
        enable_prefix_caching=True,
        max_num_seqs=32,            # Allows vLLM to process up to 64 independent chunks in parallel
        # quantization="fp8"
    )
    print(f"Instantiated VLM \n")

    # # retrieve the encodings of our target outputs
    token_id_0 = processor.tokenizer.encode("0", add_special_tokens=False)[0]
    token_id_1 = processor.tokenizer.encode("1", add_special_tokens=False)[0]
    print(f"Encoded bare logits \n")

    # # Apply a massive positive logit bias to ONLY these two tokens
    # # Mathematically impossible for the model to pick anything else
    # binary_logit_bias = {
    #     token_id_0: 100.0,
    #     token_id_1: 100.0
    # }
    sampling_params = SamplingParams(
        temperature=0,          
        max_tokens=10,           
        logprobs=5  # Tells vLLM to return logprobs for top 5 candidates       
    )

    if not os.path.exists(store_csv_folder_path):
        os.makedirs(store_csv_folder_path)


    # Inferece Process
    single_process(input_csv_folder_path, store_csv_folder_path, GPU_offset, llm, processor,legend, instruction_prompt, sampling_params)


    print("Finished!")


