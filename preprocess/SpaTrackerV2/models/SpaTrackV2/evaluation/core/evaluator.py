# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from models.SpaTrackV2.evaluation.core.tapvid3d_metrics import compute_tapvid3d_metrics
from models.SpaTrackV2.datasets.delta_utils import dataclass_to_cuda_, reproject_2d3d
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from tqdm import tqdm
import os
from models.SpaTrackV2.models.tracker3D.spatrack_modules.alignment import align_points_scale_z_shift

class Evaluator:
    """简化版评估器 - 包含原始和多线程评估功能"""

    def __init__(self, exp_dir) -> None:
        self.exp_dir = exp_dir

    def compute_metrics_3d(self, metrics, sample, pred_traj_3d, pred_visibility, dataset_name):
        if pred_visibility.dtype != torch.bool:
            pred_visibility = pred_visibility > 0.95

        if "lsfodyssey" in dataset_name:
            trajs_g = sample.trajectory3d
            vis_g = sample.visibility
            intrs = sample.intrs
            gt_traj_3d = reproject_2d3d(trajs_g, intrs)
            intrinsics_params = torch.stack(
                [intrs[0, 0, 0, 0], intrs[0, 0, 1, 1], intrs[0, 0, 0, 2], intrs[0, 0, 1, 2]], dim=-1
            )
            query_points = sample.query_points.cpu().numpy()
            out_metrics, _ = compute_tapvid3d_metrics(
                gt_occluded=np.logical_not(vis_g.cpu().numpy()),
                gt_tracks=gt_traj_3d.cpu().numpy(),
                pred_occluded=np.logical_not(pred_visibility.cpu().numpy()),
                pred_tracks=pred_traj_3d.cpu().numpy(),
                intrinsics_params=intrinsics_params.cpu().numpy(),
                scaling="median",
                query_points=query_points,
                order="b t n",
                use_fixed_metric_threshold=False,
                return_scaled_pred=True,
            )

        elif "tapvid3d" in dataset_name:
            trajs_g = sample.trajectory3d
            vis_g = sample.visibility
            intrs = sample.intrs
            intrinsics_params = torch.stack(
                [intrs[0, 0, 0, 0], intrs[0, 0, 1, 1], intrs[0, 0, 0, 2], intrs[0, 0, 1, 2]], dim=-1
            )
            query_points = sample.query_points.cpu().numpy()
            out_metrics, scaled_pred_tracks = compute_tapvid3d_metrics(
                gt_occluded=np.logical_not(vis_g.cpu().numpy()),
                gt_tracks=trajs_g.cpu().numpy(),
                pred_occluded=np.logical_not(pred_visibility.cpu().numpy()),
                pred_tracks=pred_traj_3d.cpu().numpy(),
                intrinsics_params=intrinsics_params.cpu().numpy(),
                scaling="median",
                query_points=query_points,
                order="b t n",
                use_fixed_metric_threshold=False,
                return_scaled_pred=True,
            )
        if isinstance(sample.seq_name, list):
            metrics[sample.seq_name[0]] = out_metrics
        else:
            metrics[sample.seq_name] = out_metrics
        return out_metrics

    @torch.no_grad()
    def evaluate_sequence_3d(
        self,
        model,
        test_dataloader,
        dataset_name: str,
        is_vis: bool = False,
        is_sparse: bool = True,
        lift_3d: bool = False,
        verbose: bool = False,
    ):
        """原始的3D评估方法 - 单线程版本"""
        
        print(f"Starting single-thread evaluation on dataset: {dataset_name}")
        metrics = {}
        
        for idx, sample in enumerate(tqdm(test_dataloader, desc="Evaluating", unit="batch")):
            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    continue
            
            dataclass_to_cuda_(sample)
            
            # 准备queries
            if "lsfodyssey" in dataset_name or "tapvid3d" in dataset_name:
                try:
                    queries = sample.query_points.clone().float()
                except:
                    continue
            else:
                queries = torch.cat([
                    torch.zeros_like(sample.trajectory3d[:, 0, :, :1]),
                    sample.trajectory3d[:, 0],
                ], dim=2)
            
            n_queries = queries.shape[1]
            
            # 模型推理
            with torch.amp.autocast(device_type=queries.device.type, dtype=torch.bfloat16):
                traj_e, traj_d_e, vis_e = model(
                    video=sample.video.clone(),
                    videodepth=sample.videodepth.clone(),
                    queries=queries.clone(),
                    depth_init=sample.videodepth[:, 0],
                    intrs=sample.intrs.clone(),
                    return_3d=True,
                    is_sparse=is_sparse,
                    lift_3d=lift_3d,
                    extrs=None if sample.extrs is None else torch.from_numpy(np.array(sample.extrs)),
                )
            
            traj_e = traj_e[:, :sample.video.shape[1], :n_queries]
            traj_d_e = traj_d_e[:, :sample.video.shape[1], :n_queries]
            vis_e = vis_e[:, :sample.video.shape[1], :n_queries]
            
            # tapvid3d双向跟踪
            if "tapvid3d" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_videodepth = sample.videodepth.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1
                inv_intrs = sample.intrs.clone().flip(1)
                with torch.amp.autocast(device_type=queries.device.type, dtype=torch.bfloat16):
                    inv_traj_e, inv_traj_d_e, inv_vis_e = model(
                        video=inv_video.clone(),
                        videodepth=inv_videodepth.clone(),
                        queries=inv_queries.clone(),
                        depth_init=inv_videodepth[:, 0],
                        intrs=inv_intrs.clone(),
                        return_3d=True,
                        is_sparse=is_sparse,
                        lift_3d=lift_3d,
                        extrs=None if sample.extrs is None else torch.from_numpy(np.array(sample.extrs)).flip(1),
                    )
                
                inv_traj_e = inv_traj_e[:, :sample.video.shape[1], :n_queries].flip(1)
                inv_traj_d_e = inv_traj_d_e[:, :sample.video.shape[1], :n_queries].flip(1)
                inv_vis_e = inv_vis_e[:, :sample.video.shape[1], :n_queries].flip(1)
                
                arange = torch.arange(sample.video.shape[1], device=queries.device)[None, :, None]
                mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, inv_traj_e.shape[-1])
                traj_e[mask] = inv_traj_e[mask]
                traj_d_e[mask[:, :, :, 0]] = inv_traj_d_e[mask[:, :, :, 0]]
                vis_e[mask[:, :, :, 0]] = inv_vis_e[mask[:, :, :, 0]]
            
            vis_e = vis_e > 0.5
            traj_uvd = torch.cat([traj_e, traj_d_e], dim=-1)
            traj_3d = reproject_2d3d(traj_uvd, sample.intrs)
            
            #NOTE: Megasam sometimes will be the results worse than unidepth, e.g. basketball_17
            #NOTE: scale and shift
            # gt_3d = sample.trajectory3d
            # scale_gt, shift_gt = align_points_scale_z_shift(
            #     traj_3d.view(150, -1, 3), 
            #     gt_3d.view(150, -1, 3), 
            #     weight=(vis_e.view(150,-1)>0.5).float(),
            # )
            # traj_3d = traj_3d * scale_gt[None,:,None,None] + shift_gt[None,:,None,:]
            # import pdb; pdb.set_trace()
            
            # if sample.seq_name[0] == "basketball_17":
            # visualize the 4d 
            # Prepare data in tapip3d format
            # data_npz = {}
            # data_npz["coords"] = traj_3d.cpu().numpy().squeeze(0)
            # data_npz["extrinsics"] = torch.eye(4).cpu().numpy()[None,:,:].repeat(traj_3d.shape[1], axis=0) if sample.extrs is None else np.array(sample.extrs)[0]
            # data_npz["intrinsics"] = sample.intrs.cpu().numpy().squeeze(0)
            # data_npz["depths"] = (sample.videodepth.clone()).cpu().numpy().squeeze()
            # data_npz["video"] = sample.video.cpu().numpy().squeeze(0)/255
            # data_npz["visibs"] = vis_e.float().cpu().numpy().squeeze(0)    
            # Save results
            # np.savez("debug.npz", **data_npz)
            # print(f"Tapip3d results saved to debug.npz")

            self.compute_metrics_3d(metrics, sample, traj_3d, vis_e, dataset_name)
            print(sample.seq_name[0])
            if verbose:
                print(f"Processed sample {idx+1}/{len(test_dataloader)}")
        
            # 计算平均值
            if metrics:
                if "avg" in metrics:
                    del metrics["avg"]
                
                valid_results = {k: v for k, v in metrics.items() if k != "avg"}
                if valid_results:
                    avg_metrics = {}
                    first_metrics = list(valid_results.values())[0]
                    for metric_name in first_metrics.keys():
                        metric_values = [v[metric_name] for v in valid_results.values() if metric_name in v]
                        if metric_values:
                            avg_metrics[metric_name] = np.mean(metric_values)
                    metrics["avg"] = avg_metrics
            print(metrics["avg"])
        
        if verbose and metrics.get("avg"):
            print(f"Final results: {len([k for k in metrics.keys() if k != 'avg'])} sequences")
            print("Final Avg:", metrics["avg"])
        
        return metrics

    @torch.no_grad()
    def evaluate_sequence_3d_multithread(
        self,
        model,
        test_dataloader,
        dataset_name: str,
        num_threads: int = None,
        is_sparse: bool = True,
        lift_3d: bool = False,
        verbose: bool = False,
    ):
        """多线程3D评估 - 按GPU数量分线程"""
        
        # 自动检测GPU数量
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_threads is None:
            num_threads = num_gpus
        
        print(f"Starting multithread evaluation with {num_threads} threads on {num_gpus} GPUs")
        
        # 获取数据集并分割
        dataset = test_dataloader.dataset
        total_samples = len(dataset)
        chunk_size = total_samples // num_threads
        if chunk_size == 0:
            chunk_size = 1
        
        sample_chunks = []
        for i in range(0, total_samples, chunk_size):
            chunk_end = min(i + chunk_size, total_samples)
            chunk_indices = list(range(i, chunk_end))
            if chunk_indices:
                sample_chunks.append(chunk_indices)
        
        print(f"Split {total_samples} samples into {len(sample_chunks)} chunks")
        
        # 将原始模型移到CPU释放GPU显存，避免第一张卡占用过多
        model = model.cpu()
        
        # 线程工作函数
        def thread_worker(thread_id, sample_indices):
            # 设置GPU
            if torch.cuda.is_available():
                device_id = thread_id % num_gpus
                torch.cuda.set_device(device_id)
                device = f"cuda:{device_id}"
                print(f"Thread {thread_id} using GPU {device_id}")
            else:
                device = "cpu"
            
            # 模型移到对应GPU
            thread_model = copy.deepcopy(model).to(device)
            thread_metrics = {}
            
            # 处理样本
            for idx, sample_idx in enumerate(tqdm(sample_indices, desc=f"Thread {thread_id}", unit="sample", position=thread_id, leave=True)):
                try:
                    sample = dataset[sample_idx]
                    if sample is None:
                        continue

                    dataclass_to_cuda_(sample)
                    
                    # 移动数据到GPU
                    for attr_name in ['video', 'videodepth', 'intrs', 'trajectory3d', 'query_points', 'trajectory', 'visibility', 'valid']:
                        if hasattr(sample, attr_name):
                            attr_value = getattr(sample, attr_name)
                            if attr_value is not None and hasattr(attr_value, 'to'):
                                setattr(sample, attr_name, attr_value.to(device)[None])
                    
                    # 准备queries
                    if "lsfodyssey" in dataset_name or "tapvid3d" in dataset_name:
                        try:
                            queries = sample.query_points.clone().float()
                        except:
                            continue
                    else:
                        queries = torch.cat([
                            torch.zeros_like(sample.trajectory3d[:, 0, :, :1]),
                            sample.trajectory3d[:, 0],
                        ], dim=2)
                    
                    n_queries = queries.shape[1]
                    
                    # 模型推理
                    with torch.amp.autocast(device_type=queries.device.type, dtype=torch.bfloat16):
                        traj_e, traj_d_e, vis_e = thread_model(
                            video=sample.video.clone(),
                            videodepth=sample.videodepth.clone(),
                        queries=queries.clone(),
                        depth_init=sample.videodepth[:, 0],
                        intrs=sample.intrs.clone(),
                        return_3d=True,
                        is_sparse=is_sparse,
                        lift_3d=lift_3d,
                        )
                    traj_e = traj_e[:, :sample.video.shape[1], :n_queries]
                    traj_d_e = traj_d_e[:, :sample.video.shape[1], :n_queries]
                    vis_e = vis_e[:, :sample.video.shape[1], :n_queries]
                    
                    # tapvid3d双向跟踪
                    if "tapvid3d" in dataset_name:
                        inv_video = sample.video.flip(1).clone()
                        inv_videodepth = sample.videodepth.flip(1).clone()
                        inv_queries = queries.clone()
                        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1
                        inv_intrs = sample.intrs.clone().flip(1)
                        with torch.amp.autocast(device_type=queries.device.type, dtype=torch.bfloat16):
                            inv_traj_e, inv_traj_d_e, inv_vis_e = thread_model(
                                video=inv_video.clone(),
                                videodepth=inv_videodepth.clone(),
                            queries=inv_queries.clone(),
                            depth_init=inv_videodepth[:, 0],
                            intrs=inv_intrs.clone(),
                            return_3d=True,
                            is_sparse=is_sparse,
                            lift_3d=lift_3d,
                            )
                        
                        inv_traj_e = inv_traj_e[:, :sample.video.shape[1], :n_queries].flip(1)
                        inv_traj_d_e = inv_traj_d_e[:, :sample.video.shape[1], :n_queries].flip(1)
                        inv_vis_e = inv_vis_e[:, :sample.video.shape[1], :n_queries].flip(1)
                        
                        arange = torch.arange(sample.video.shape[1], device=queries.device)[None, :, None]
                        mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, inv_traj_e.shape[-1])
                        traj_e[mask] = inv_traj_e[mask]
                        traj_d_e[mask[:, :, :, 0]] = inv_traj_d_e[mask[:, :, :, 0]]
                        vis_e[mask[:, :, :, 0]] = inv_vis_e[mask[:, :, :, 0]]
                    
                    vis_e = vis_e > 0.5
                    traj_uvd = torch.cat([traj_e, traj_d_e], dim=-1)
                    traj_3d = reproject_2d3d(traj_uvd, sample.intrs)
                    
                    #NOTE: scale and shift
                    # gt_3d = sample.trajectory3d
                    # scale_gt, shift_gt = align_points_scale_z_shift(
                    #     traj_3d.view(150, -1, 3), 
                    #     gt_3d.view(150, -1, 3), 
                    #     weight=vis_e.view(150,-1),
                    # )
                    # traj_3d = traj_3d * scale_gt[None,:,None,None] + shift_gt[None,:,None,:]

                    # 计算metrics
                    self.compute_metrics_3d(thread_metrics, sample, traj_3d, vis_e, dataset_name)
                    print(thread_metrics.keys())

                    if verbose:
                        print(f"Thread {thread_id}: Processed {idx+1}/{len(sample_indices)} samples")
                    
                except Exception as e:
                    print(f"Thread {thread_id}: Error processing sample {sample_idx}: {e}")
                    continue
            print(f"Thread {thread_id} completed, processed {len(sample_indices)} samples")
            return thread_metrics
        
        # 执行多线程
        all_metrics = {}
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(thread_worker, i, chunk): i 
                for i, chunk in enumerate(sample_chunks)
            }
            
            for future in as_completed(futures):
                thread_id = futures[future]
                try:
                    thread_metrics = future.result()
                    all_metrics.update(thread_metrics)
                    print(f"Thread {thread_id} results merged")
                except Exception as e:
                    print(f"Thread {thread_id} failed: {e}")
        
        # 计算最终平均值
        if all_metrics:
            if "avg" in all_metrics:
                del all_metrics["avg"]
            
            valid_results = {k: v for k, v in all_metrics.items() if k != "avg"}
            if valid_results:
                avg_metrics = {}
                first_metrics = list(valid_results.values())[0]
                for metric_name in first_metrics.keys():
                    metric_values = [v[metric_name] for v in valid_results.values() if metric_name in v]
                    if metric_values:
                        avg_metrics[metric_name] = np.mean(metric_values)
                all_metrics["avg"] = avg_metrics
        
        if verbose and all_metrics.get("avg"):
            print(f"Final results: {len([k for k in all_metrics.keys() if k != 'avg'])} sequences")
            print("Final Avg:", all_metrics["avg"])
        
        return all_metrics 