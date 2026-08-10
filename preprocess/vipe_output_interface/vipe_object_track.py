import numpy as np
from vipe_depth import VipeDepth
from vipe_camera import Camera
from vipe_optical_flow import FlowResult, FlowObject
from vipe_masks import VipeMasks, InstanceMask
from vipe_dense_flow import DenseFlow
from vipe_rigid_alignment import kabsch_umeyama, get_points_and_indices, get_points_with_depth, points_i2c, points_c2w
import argparse

def track_object(base_path, instance_mask: InstanceMask) -> np.ndarray:
    depths = VipeDepth()
    depths.set_video(base_path)

    camera = Camera()
    camera.set_video(base_path)

    flow = DenseFlow()
    flow.set_video(base_path)
    # return us a FlowResult object which is planned compatible with non-oneshot flow and sparse flow
    flow_result = flow.get_edges(one_shot=True)

    # returns 

    oneshot = flow_result.is_oneshot()

    indices = flow_result.get_dsts() if oneshot else flow_result.get_srcs()

    # T, H, W, C
    points = get_points_and_indices(mask = instance_mask, flow = flow_result, indices = indices)
    T, H, W, C = points.shape()
    # flatten the H, W dims into N pts
    points = points.reshape(T, H*W, C)
    # slice to just get u, v
    points = points[:, :, 3:4]

    points = get_points_with_depth(depth = depths, points=points, indices=indices)

    points = points_i2c(points, indices, camera)
    points = points_c2w(points, indices, camera)

    centroid_0 = None
    points_0 = points[0, :, :]
    T, N, C= points.shape()

    Rt_result = np.empty((0,4,4))
    for i in range(1, T):
        # includes logic to not re-estimate the centroid
        R, t, s, centroid_0 = kabsch_umeyama(P=points_0, Q=points[i, :, :], estimate_scale=False, p_mean = centroid_0)
        Rt = np.concatenate([R, t[:, None]], axis = 1)
        bottom_row = np.array([[0., 0., 0., 1.]])   # (1, 4)
        Rt = np.concatenate([Rt, bottom_row], axis=0)
        Rt_result = np.concatenate([Rt_result, Rt], axis = 0)

    return Rt_result




















# USE VLM - > returns valid instance masks
# if dry run, then just select something and print it out?
def select_identities(base_path, dry_run = True):
    masks = VipeMasks()
    masks.set_video(base_path=args.base_path)
    pass 

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("base_path", type= str)
    argparser.add_argument("dry_run", action = "store_true")

    args = argparser.parse_args()


    
    instance_masks = select_identities(base_path = args.base_path, use_sam = args.dry_run)
    
    Rt = []
    for im in instance_masks:
        Rt.append(track_object(base_path = args.base_path, instance_mask=im))


    # multivisualize function
