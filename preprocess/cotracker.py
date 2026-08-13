import torch

import imageio.v3 as iio
from preprocess.vipe_output_interface.vipe_masks import InstanceMask
import numpy as np


def get_point_tracks(
        video: torch.tensor,
        # B, N, 
        queries: torch.tensor,
        end_frame: int,
        keyframes: list[int],
        start_frame: int= 0,
    ):
    """
    Compute point tracks from start_frame : end_frame
    Returns: # pred_tracks B T N 2,  pred_visibility B T N 1
    """

    device = 'cuda'
    # add batch dim
    video = torch.tensor(video)[None].float().to(device)  # B T C H W

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

    # Initialize online processing
    cotracker(video_chunk=video, is_first_step=True, queries=queries)

    # Process the video
    pred_tracks, pred_visibility = np.ndarray(), np.ndarray()
    indices = [start_frame]
    for ind in range(start_frame, end_frame - cotracker.step, cotracker.step):
        # 1 piece of  sliding window where every window is 2x the length of the jump
        pred_tracks, pred_visibility = cotracker(
            video_chunk=video[:, ind : ind + cotracker.step * 2]
        )  # B T N 2,  B T N 1
        indices.append(ind + cotracker.step * 2)

        
    # cotracker api returns results from 0-n 
    return pred_tracks, pred_visibility, indices
