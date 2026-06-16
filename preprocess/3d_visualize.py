import torch
import decord
from decord import VideoReader, cpu, gpu

# uses Umeyama and SVD to compute camera intrinsics, extrinics
from Open_d4rt.vis import build_like_demo
from Open_d4rt.src.model.d4rt import D4RTModel
from Open_d4rt import infer_track_3d
import argparse
import numpy as np



def read_video_to_tensor(video_path):
    # if torch.cuda.is_available():
    #     vr = VideoReader(video_path, ctx=gpu(0))
    # else:
    # loading on cpu is safer
    vr = VideoReader(video_path, ctx = cpu(0))

    # we only load the necessary frames to avoid OOM
    tensor_frames = vr.get_batch(range(len(vr))) # Returns an NDArray
    
    tensor = torch.from_numpy(tensor_frames.asnumpy())
    
    # Permute from [T, H, W, C] to [T, C, H, W]
    return tensor.permute(0, 3, 1, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type="str", default =None)

    model = torch.load_state_dict()

    args = parser.parse_args()

    model = D4RTModel()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        Exception("No cuda device is availalbe!")

    # IF NEEDED: explore how the opend4rt repo gets points on an identity first
    # function to segment the image for a wanted class (extract this from pre written code) and get uv 


    query = infer_track_3d._build_query_for_targets() # add arguments for the whole video!

    # construct the query

    # get the camera extrinsics, specifically the x,y,z relative to the camera origin at t_0 (do math if you need)

    # 



    