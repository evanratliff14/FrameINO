import torch
import cv2
from model import DiViDeAddEvaluator
import numpy as np

from time import time
import math

import yaml
import csv

from pathlib import Path

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]

# we rescale to 
def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

def call(model, video, num_clips, clip_length):
    video_dict ={}
    with torch.no_grad():
        b, t,c, h, w = video.size()
        # truncate
        truncated_num = t - (t%clip_length)
        video = video[:,0:truncated_num, ...]

        # collect into batch
        video = video.permute(0,2,1,3,4).reshape(1, c, t//clip_length, clip_length, h, w).permute(0,2,1,3,4,5).reshape(b * (t//clip_length), c, clip_length, h, w) 
        # we can collect a near-uniform sample by dropping every other clip
        while video.size(0) > num_clips:
            video = video[::2, ...]


        video_dict["fragments"] = video
        result = model(video_dict,inference=True, reduce_scores=True, pooled=True)
        # vtss is just the mean of the scores?
    return result


def get_model(device):

    script_dir = Path(__file__).resolve().parent

    yaml_path = script_dir / "test.yml"
    with open(yaml_path, "r") as f:
        opt = yaml.safe_load(f)

    
    model = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)

    state_dict = torch.load(script_dir / opt["test_load_path"], map_location=device)["state_dict"]
    
    if "test_load_path_aux" in opt:
        aux_state_dict = torch.load(script_dir / opt["test_load_path_aux"], map_location=device)["state_dict"]
        
        from collections import OrderedDict
        
        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v
            
        for k, v in aux_state_dict.items():
            if k.startswith("frag"):
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v
        
        state_dict = fusion_state_dict
        
    model.load_state_dict(state_dict, strict=True)

    return model

    