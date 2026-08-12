import numpy as np
from vipe_depth import VipeDepth
from vipe_camera import Camera
from vipe_optical_flow import FlowResult, FlowObject
from vipe_masks import VipeMasks, InstanceMask
from vipe_dense_flow import DenseFlow
from vipe_rigid_alignment import kabsch_umeyama, get_points, get_points_with_depth, points_i2c, points_c2w
from vipe_display_rigid_alignment import display_rigid_alignment
from vipe_io import read_rgb_frames
from SAM3D.sam_3d import reconstruct
import trimesh
import argparse
from pathlib import Path


def track_object(
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
