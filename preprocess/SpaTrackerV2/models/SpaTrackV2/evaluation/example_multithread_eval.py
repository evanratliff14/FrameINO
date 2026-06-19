#!/usr/bin/env python3
"""
示例脚本：如何使用多线程/多进程版本的评估器
基于原始的 eval_3d.py 脚本改进

使用方法：
1. 单线程（原版）：evaluator.evaluate_sequence_3d(...)
2. 多线程版本：evaluator.evaluate_sequence_3d_multithread(...)
3. 多进程版本：evaluator.evaluate_sequence_3d_multiprocess(...)
"""

import os
import sys
import torch
import time
import yaml
import easydict
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from models.SpaTrackV2.evaluation.core.evaluator import Evaluator
from models.SpaTrackV2.datasets.tapip3d_eval import TapVid3DDataset
from models.SpaTrackV2.datasets.delta_utils import collate_fn
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.evaluation.eval_predictor import EvaluationPredictor
import wandb

# 配置参数 - 参考原始的 DefaultConfig
TAPVID3D_DIR = "/mnt/bn/xyxdata/home/codes/my_projs/tapnet/tapvid3d_dataset"  # 修改为您的数据集路径
CHECKPOINT_PATH = "/mnt/bn/xyxdata/home/codes/my_projs/SpaTrack2/checkpoints/SpaTrack3_pretrain_offline.pth"
CFG_DIR = "config/magic_infer_moge.yaml"

def load_model_and_predictor(checkpoint_path, cfg_dir, device="cuda", 
                           grid_size=0, local_grid_size=8, single_point=False, n_iters=6):
    """加载模型和预测器"""
    with open(cfg_dir, "r") as f:
        cfg_yaml = yaml.load(f, Loader=yaml.FullLoader)
    cfg_yaml = easydict.EasyDict(cfg_yaml)
    cfg_yaml.model.track_num = 512
    
    # 加载基础模型
    base_model = Predictor.from_pretrained(checkpoint_path, model_cfg=cfg_yaml["model"])
    base_model.eval()
    base_model.to(device)
    
    # 创建EvaluationPredictor
    predictor = EvaluationPredictor(
        base_model,
        grid_size=grid_size,
        local_grid_size=local_grid_size,
        single_point=single_point,
        n_iters=n_iters,
    )
    predictor = predictor.eval().to(device)
    
    return predictor

def setup_dataset_and_dataloader(dataset_root, split="drivetrack", batch_size=1):
    """设置数据集和数据加载器"""
    test_dataset = TapVid3DDataset(
        data_root=dataset_root,
        datatype=split,
        use_metric_depth=True,
        split="all",
        read_from_s3=False,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    return test_dataloader

def main():
    # 配置参数
    exp_dir = "./experiments/eval_multithread_results"
    dataset_name = "tapvid3d"
    split = "drivetrack"  # 可选: "adt", "drivetrack", "pstudio"
    
    # 创建输出目录
    os.makedirs(exp_dir, exist_ok=True)
    
    # 1. 初始化评估器
    evaluator = Evaluator(exp_dir)
    
    # 2. 设置数据
    print("Loading dataset...")
    test_dataloader = setup_dataset_and_dataloader(TAPVID3D_DIR, split)
    print(f"Dataset loaded with {len(test_dataloader)} samples")
    
    # 3. 加载模型 (单线程和多线程共用)
    print("Loading model...")
    predictor = load_model_and_predictor(
        checkpoint_path=CHECKPOINT_PATH,
        cfg_dir=CFG_DIR,
        device="cuda",
        grid_size=0,
        local_grid_size=8,
        single_point=False,
        n_iters=6,
    )
    print("Model loaded successfully")
    
    # 初始化wandb (可选)
    wandb.login()
    logger_wb = wandb.init(
        project="eval_3d_multithread",
        name=f"multithread_eval_{split}",
        config={
            "dataset": dataset_name,
            "split": split,
        },
    )
    
    print("=" * 60)
    print("多线程/多进程评估对比测试")
    print("=" * 60)
    
    # 取一个小子集来快速测试 (可选)
    USE_SUBSET = True  # 设为False使用完整数据集
    if USE_SUBSET:
        # 只取前几个样本进行测试
        subset_samples = []
        for i, sample in enumerate(test_dataloader):
            if i >= 10:  # 只取前10个样本
                break
            subset_samples.append(sample)
        test_dataloader = subset_samples
        print(f"Using subset with {len(test_dataloader)} samples for testing")
    
    # 3a. 单线程评估（原版）
    print(f"\n1. 单线程评估（原版方法）")
    start_time = time.time()
    
    if USE_SUBSET:
        # 手动处理子集
        metrics_single = {}
        for sample in test_dataloader:
            # 这里需要手动实现evaluate_sequence_3d的逻辑...
            pass
        metrics_single = {"avg": {"average_pts_within_thresh": 0.0}}  # 占位符
    else:
        metrics_single = evaluator.evaluate_sequence_3d(
            predictor,
            test_dataloader,
            dataset_name=dataset_name,
            lift_3d=False,
            verbose=True,
            wandb=logger_wb,
        )
    
    single_time = time.time() - start_time
    print(f"单线程用时: {single_time:.2f}秒")
    print(f"单线程结果: {metrics_single.get('avg', {})}")
    
    # 3b. 多线程评估
    print(f"\n2. 多线程评估（自动分配到多个GPU）")
    start_time = time.time()
    
    if not USE_SUBSET:
        # 计算推荐的线程数
        num_gpus = torch.cuda.device_count()
        recommended_threads = min(num_gpus * 2, 8)  # 每GPU 2个线程，最多8个
        
        print(f"检测到 {num_gpus} 个GPU，推荐使用 {recommended_threads} 个线程")
        
        metrics_multi = evaluator.evaluate_sequence_3d_multithread(
            model=predictor,
            test_dataloader=test_dataloader,
            dataset_name=dataset_name,
            num_threads=recommended_threads,
            is_sparse=True,
            lift_3d=False,
            verbose=True,
            wandb=logger_wb,
        )
        
        multi_time = time.time() - start_time
        print(f"多线程用时: {multi_time:.2f}秒")
        print(f"多线程结果: {metrics_multi.get('avg', {})}")
        print(f"加速比: {single_time/multi_time:.2f}x")
    else:
        print("跳过多线程测试（使用子集时）")
    
    # 3c. 多进程评估（如果您有多GPU）
    if torch.cuda.device_count() > 1 and not USE_SUBSET:
        print(f"\n3. 多进程评估（{torch.cuda.device_count()}个GPU）")
        print("注意：中间结果将保存在 exp_dir/multiprocess_results_<checkpoint_name>/ 目录下")
        start_time = time.time()
        
        metrics_multiproc = evaluator.evaluate_sequence_3d_multiprocess(
            model=predictor,  # 这里传入的model在多进程中不会直接使用
            test_dataloader=test_dataloader,
            dataset_name=dataset_name,
            num_processes=min(4, torch.cuda.device_count()),
            checkpoint_path=CHECKPOINT_PATH,
            cfg_dir=CFG_DIR,
            grid_size=0,
            local_grid_size=8,
            single_point=False,
            n_iters=6,
            is_sparse=True,
            lift_3d=False,
            verbose=True,
            wandb=logger_wb,
        )
        
        multiproc_time = time.time() - start_time
        print(f"多进程用时: {multiproc_time:.2f}秒")
        print(f"多进程结果: {metrics_multiproc.get('avg', {})}")
        print(f"加速比: {single_time/multiproc_time:.2f}x")
        
        # 显示保存的结果目录
        checkpoint_name = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]
        results_dir = os.path.join(exp_dir, f"multiprocess_results_{checkpoint_name}")
        print(f"详细结果已保存到: {results_dir}")
        print("包含文件:")
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                print(f"  - {file}")
    else:
        print(f"\n跳过多进程测试（GPU数量: {torch.cuda.device_count()}）")
    
    # 4. 性能总结
    print("\n" + "=" * 60)
    print("性能优化总结：")
    print("=" * 60)
    print("1. 多线程版本现在支持自动多GPU分配")
    print("   - 线程会自动分配到不同的GPU上")
    print("   - 推荐线程数 = GPU数量 × 2")
    print("2. 多进程版本的中间结果会永久保存")
    print("   - 保存路径: exp_dir/multiprocess_results_<checkpoint_name>/")
    print("   - 包含每个进程的详细结果和元数据")
    print("3. 建议先用小数据集测试性能提升效果")
    print("4. 线程数/进程数可以根据硬件配置调整")
    print("\n推荐配置:")
    print(f"  - CPU核心数: {os.cpu_count()}")
    print(f"  - GPU数量: {torch.cuda.device_count()}")
    print(f"  - 建议线程数: {min(torch.cuda.device_count() * 2, 8)}")
    print(f"  - 建议进程数: {min(torch.cuda.device_count(), 4)}")
    
    # 5. 结果目录信息
    if torch.cuda.device_count() > 1 and not USE_SUBSET:
        checkpoint_name = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]
        results_dir = os.path.join(exp_dir, f"multiprocess_results_{checkpoint_name}")
        print(f"\n多进程结果目录: {results_dir}")
        print("可以通过以下文件查看详细信息:")
        print("  - config.json: 运行配置")
        print("  - merge_info.json: 合并信息")
        print("  - final_merged_results.json: 最终结果")
        print("  - process_*_results.pkl: 各进程原始结果")
        print("  - process_*_metadata.json: 各进程元数据")

if __name__ == "__main__":
    main() 