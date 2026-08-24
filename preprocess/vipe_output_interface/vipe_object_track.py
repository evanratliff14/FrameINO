import numpy as np
from vipe_depth import VipeDepth
from vipe_camera import Camera
from vipe_optical_flow import FlowResult, FlowObject
from vipe_masks import VipeMasks, InstanceMask
from vipe_dense_flow import DenseFlow
from vipe_rigid_alignment import kabsch_umeyama, get_points, get_points_with_depth, points_i2c, points_c2w
from vipe_display_rigid_alignment import display_rigid_alignment
from vipe_io import read_rgb_frames
from preprocess.SAM3D.sam_3d import reconstruct
from preprocess.cotracker import get_point_tracks
import trimesh
import torch
import argparse
from pathlib import Path

def track_objects_with_cotracker(
        video: torch.tensor,
        # we'll assume some masks appear in start_frame and some don't - we'll leave that up to selection
        masks: dict[int: InstanceMask],
        camera: Camera,
        depths: VipeDepth,
        end_frame: int,
        start_frame: int = 0,
        num_tracks = 5000,
        track_keyframes = True

    ):

    # we store a dict of frame: rot
    result = {id: {} for id in masks.keys()}

    masks = masks[start_frame:end_frame, :, :]
    if track_keyframes:
        keyframes = set()

        # we're going to measure num pixels shown, as well as ground samplel distance squared (Z/f)^2
        # this measures the amount of metric infromation being projected on our image plane (units m^2 / pixel = px*py), assuming pinhole cam
        candidates = {}
        for id, mask in enumerate(masks):
            # sum along H, W -> T
            counts = np.sum(mask, axis = [1, 2])
            # boundary_mask = np.zeros_like(mask, dtype=bool)
            # boundary_mask[:, 0, :] = boundary_mask[:, -1, :] = boundary_mask[:, :, 0] = boundary_mask[:, :, -1] = 1
            # boundary_count = np.sum(boundary_mask * masks, axis = [1,2])
            
            not_in_frame = False
            in_frame = False

            candidate = False

            # thresholding algorithm to measure frame - in keyframes
            for i in range(counts.size(0)):
                # we will end up tracking more points on sequences with better resolution  - this seems appropriate, rather than relying on sub-pixel precision
                if counts[i] <50:
                    not_in_frame = True
                elif counts[i] >=200:
                    depth_i = depths.get_depth(i)
                    depths_i_mask = depth_i * mask
                    avg_Z = np.mean(depths_i_mask[depths_i_mask[:, :] > 0.0], axis =0)
                    intrinsics = camera.get_intrinsics(i)
                    f = np.mean(intrinsics[0:2], axis=0)
                    ground_sample_dist_squared = counts[i]*np.pow(Z/f, 2)
                    # heuristic 0.25cm^2 per pixel and less than 40 near-meter units away
                    if ground_sample_dist_squared >2.5e-5 and avg_Z <=40:
                        in_frame = candidate = True
                    else:
                        in_frame = False

                # not_in_frame measures if it has previuosly been out of frame, in_frame measures the current status
                if not_in_frame and in_frame:
                    not_in_frame = False
                    keyframes.add(i)
            

            if not candidate:
                del masks[id]

        keyframes.add(start_frame)
        keyframes = sorted(list(keyframes))
        # lazy iteration does not mess up indexing when progressing from left
        for i, keyframe in enumerate(keyframes):
            if i>0:
                # we want to minimize the number of keyframes while maintaining that keyframes cannot be within 16 frames of one another 
                # in 16 frames, cotracker will do inference on 3 sliding windows of size 8 step size 4
                if keyframes[i] - keyframes[i-1] <16:
                    del keyframes[i]

    else:
        keyframes = [start_frame, end_frame]


    for keyframe in keyframes[:-1]:

        
        # come in as T, H, W
        masks0 = {id: mask.at(keyframe) for id, mask in masks.items() if (mask.at(keyframe)>0).any()}

        # the order that objects will appear - since id order in query tensor is preserved along the N axis
        ids = masks0.keys()

        num_instances0 = len(ids)

        # we evenly distribute points per instance. smaller instances (surface area wise) require more point per pixel for good estimation
        # due to the assumption that error scales as we approach pixel-level or sub-pixel level precision
        points_per_instance = num_tracks//num_instances0

        coords_per_mask = [torch.nonzero(mask).float() for mask in masks0()]

        # Randomly sample N points (e.g., N=500)
        indices = [torch.randperm(c.size(0))[:points_per_instance] for c in coords_per_mask]
        coords = np.cat([c[indices] for c in coords_per_mask], axis = 0) # Contains [y, x]


        # Format to (B, N, 3) with (t, x, y)
        t = torch.ones((1, N, 1), device=device) * keyframe
        coords = coords[:, [1, 0]].unsqueeze(0) # Swap [y, x] -> [x, y] and add batch dim

        queries = torch.cat([t, coords], dim=-1)

        if i+1<len(keyframes):
            stop = keyframes[i+1]
        else:
            stop = end_frame

        point_tracks, point_visibility, indices = get_point_tracks(video=video, queries = queries, start_frame=keyframe, end_frame = stop)
        # (B, T, N, 2), (B, T, N, 2), (B, T, N, 1)
        correspondences = torch.cat([queries, point_tracks, point_visibility], axis = 3)
        # del the batch dim
        correspondences.squeeze(0)
        T, N, X = correspondences.size()

        # since points return from get_point_tracks in original order of id, we can reconstruct which point belong where without an extra DS
        correspondences = torch.reshape(T, points_per_instance, num_instances0, X)
        correspondences = correspondences.numpy().cpu()

        # mask = correspondences[..., 4] >= 0.5
        # zero out entries where false
        # correspondences = correspondences * mask[..., None]

        # we can use threadpoolexecturor across different instances
        for i in range(correspondences.size(2)):
            instance_tracks = correspondences[:, :, i, :]

            src = instance_tracks[0, :, i, 0:2]
            src_frame = src[0, 0]
            confidence = instance_tracks[0, :, i, 4]

            # note that we are calculating rotation from the keyframe of entry or 0, not necessarily where SAM3D does inference
            src_with_depth = get_points_with_depth(depth=depths, points = src, indices = [start_frame])
            src_W = points_c2w(points_i2c(src_with_depth))
            # src_W is T, N, C=3
            centroid = torch.mean(src_W, axis = 2)

            # we must apply this transformation to the Umeyama-Kabsch-acquired T to get the full transformation
            src_Rt = np.eye(4, 4)

            if result[id].empty():
                Rt = [
                    [1, 0, 0, centroid[0]],
                    [0, 1, 0, centroid[1]],
                    [0, 0, 1, centroid[2]],
                    [0, 0, 0, 1          ]
                    ]
                result[id][src_frame] = Rt
            else:
                src_Rt = results[id][max(results[id].keys())]

            dsts= instance_tracks[1:, :, i, :]
            id = ids[i]
            p_mean = None

            for j in range(size(instance_tracks.size(0))):
                # if this is the first instance of it, we'll say its rotation is I and translation is its world coordinates mean
                    
                dst = instance_tracks[j, :, i, :]
                t= indices[j]
                dst_with_depth = get_points_with_depth(depth=depths, points = dst, indices = [t])
                dst_W = points_c2w(points_i2c(dst_with_depth))
                R, t, s, p_mean = kabsch_umeyama(
                    P=src, Q=points[i, :, :], estimate_scale=False, p_mean=p_mean
                )

                Rt = np.concatenate([R, t[:, None]], axis=1)
                bottom_row = np.array([[0.0, 0.0, 0.0, 1.0]])  # (1, 4)
                Rt = np.concatenate([Rt, bottom_row], axis=0)
                Rt_result = np.concatenate([Rt_result, Rt[None, ...]], axis=0)

                # apply the transformation from the last keyframe
                Rt_result = src_Rt @ Rt_result

                result[i][j] = Rt_result
                
    return result



def track_object_with_flow(
    instance_mask: InstanceMask,
    flow: DenseFlow,
    depths: VipeDepth,
    camera: Camera,
) -> np.ndarray:
    # return us a FlowResult object which is planned compatible with non-oneshot flow and sparse flow
    flow_result = flow.get_edges(one_shot=True)

    # returns

    oneshot = flow_result.is_oneshot()

    indices = flow_result.get_dsts() if oneshot else flow_result.get_srcs()

    # T, H, W, C
    points = get_points(
        mask=instance_mask, flow=flow_result, indices=indices
    )
    T, H, W, C = points.shape
    # flatten the H, W dims into N pts
    points = points.reshape(T, H * W, C)
    # slice to just get u, v
    points = points[:, :, 3:4]

    points = get_points_with_depth(depth=depths, points=points, indices=indices)

    points = points_i2c(points, indices, camera)
    points = points_c2w(points, indices, camera)

    centroid_0 = None
    points_0 = points[0, :, :]
    T, N, C = points.shape

    Rt_result = np.empty((0, 4, 4))
    for i in range(1, T):
        # includes logic to not re-estimate the centroid
        R, t, s, centroid_0 = kabsch_umeyama(
            P=points_0, Q=points[i, :, :], estimate_scale=False, p_mean=centroid_0
        )
        Rt = np.concatenate([R, t[:, None]], axis=1)
        bottom_row = np.array([[0.0, 0.0, 0.0, 1.0]])  # (1, 4)
        Rt = np.concatenate([Rt, bottom_row], axis=0)
        Rt_result = np.concatenate([Rt_result, Rt[None, ...]], axis=0)

    return Rt_result


# USE VLM - > returns valid instance masks
# if dry run, then just select something and print it out?
def select_identities(base_path, dry_run=True) -> list[InstanceMask]:
    masks = VipeMasks()
    masks.set_video(base_path=base_path)
    return masks.get_masks()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("base_path", type=str)
    argparser.add_argument("--dry_run", action="store_true")
    argparser.add_argument("--host", type=str, default="127.0.0.1")
    argparser.add_argument("--port", type=int, default=20541)

    args = argparser.parse_args()
    base_path = Path(args.base_path)

    depths = VipeDepth()
    depths.set_video(base_path)

    camera = Camera()
    camera.set_video(base_path)

    flow = DenseFlow()
    flow.set_video(base_path)

    instance_masks = select_identities(base_path=args.base_path, dry_run=args.dry_run)

    Rt = []
    indices = None
    for im in instance_masks:
        Rt.append(
            track_object(
                instance_mask=im, flow=flow, camera=camera, depths=depths
            )
        )

    keyframe = 0
    rgb_by_idx = {fi: fr for fi, fr in read_rgb_frames(base_path)}
    video_keyframe = rgb_by_idx[keyframe]

    path = Path(base_path) / "mesh"
    path.mkdir(parents=True, exist_ok=True)

    meshes = []
    for im in instance_masks:
        glb_path = path / f"{im.instance_id}.glb"
        if glb_path.exists():
            meshes.append(trimesh.load(str(glb_path), force="mesh"))
        elif args.dry_run:
            meshes.append(None)
        else:
            mesh = reconstruct(video_keyframe, im.mask[keyframe])
            mesh.export(str(glb_path))
            meshes.append(mesh)

    # oneshot track_object Rt has no identity row; prepend I so K matches indices
    Rt_for_display = []
    for rt in Rt:
        I = np.eye(4, dtype=np.float64)[None, ...]
        Rt_for_display.append(np.concatenate([I, rt], axis=0) if rt.shape[0] else I)

    instance_ids = [im.instance_id for im in instance_masks]
    display_rigid_alignment(
        base_path,
        indices,
        Rt_for_display,
        meshes,
        instance_ids,
        host=args.host,
        port=args.port,
    )
