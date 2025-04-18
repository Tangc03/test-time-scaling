import os
import argparse
from accelerate import Accelerator
import hpsv2
import torch
from generation.generate import Meissonic
from generation.halton_generate import Meissonic_halton
from tqdm import tqdm
import json
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="T2I-CompBench Pipeline")
    # 添加sample模式选项
    parser.add_argument("--mode", 
                        type=str, 
                        choices=["gen", "eval", "sample_inference"],  # 新增sample_inference
                        required=True,
                        help="Operation mode: gen/sample_inference/eval")
    
    # 新增sample数据路径参数
    parser.add_argument("--sample_path",
                        type=str,
                        default="sample_testset/sample_hpsv2.jsonl",
                        help="Path to sampled prompts dataset")
    
    # 通用参数
    parser.add_argument("--output_dir",
                        type=str,
                        default="./hps_outputs",
                        help="Base directory for generation outputs")
    
    # 生成模式参数
    parser.add_argument("--model",
                        type=str,
                        choices=["Meissonic", "Meissonic_halton", "both"],
                        default="Meissonic",
                        help="Model(s) to use for generation")
    parser.add_argument("--skip_existing",
                        action="store_true",
                        help="Skip existing generated images")
    
    # 评估模式参数
    parser.add_argument("--input_dir",
                        type=str,
                        help="Input directory for evaluation")
    parser.add_argument("--hps_version",
                        type=str,
                        choices=["v2.0", "v2.1"],
                        default="v2.1",
                        help="HPS evaluation version")
    
    return parser.parse_args()

def generate_images(accelerator, args):
    accelerator.print(f"Process {accelerator.process_index} on {accelerator.device}")
    
    # 获取基准prompts
    all_prompts = hpsv2.benchmark_prompts('all')
    
    # 处理模型选择
    models = []
    if args.model == "both":
        models = ["Meissonic", "Meissonic_halton"]
    else:
        models = [args.model]

    # 遍历所有模型
    for model_name in models:
        model = Meissonic if model_name == "Meissonic" else Meissonic_halton
        
        # 准备任务列表
        tasks = []
        for style, prompts in all_prompts.items():
            for prompt_idx, prompt in enumerate(prompts):
                tasks.append((style, prompt_idx, prompt))
        
        # 修正任务分配方式
        with accelerator.split_between_processes(tasks, apply_padding=True) as distributed_tasks:
            # 主进程准备
            if accelerator.is_local_main_process:
                os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
                progress_bar = tqdm(
                    total=len(tasks),
                    desc=f"Generating ({model_name})",
                    position=0,
                    leave=True
                )
            
            accelerator.wait_for_everyone()

            # 处理任务
            for task in tqdm(distributed_tasks):
                style, prompt_idx, prompt = task
                save_dir = os.path.join(args.output_dir, model_name, style)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prompt_idx:05d}.jpg")
                
                if args.skip_existing and os.path.exists(save_path):
                    continue
                    
                try:
                    image = model(prompt, accelerator)
                    image.save(save_path)
                except Exception as e:
                    accelerator.print(f"Error generating {save_path}: {str(e)}")
                
                if accelerator.is_main_process:
                    progress_bar.update(1)
            
            if accelerator.is_main_process:
                progress_bar.close()

def evaluate_images(args):
    """分布式评估实现"""
    # 只在主进程执行评估
    print(f"Starting evaluation with HPS {args.hps_version}")
    score = hpsv2.evaluate(args.input_dir, hps_version=args.hps_version)
    print(f"Final HPS-{args.hps_version} Score: {score}")

def sample_inference(accelerator, args):
    """基于采样prompts的生成实现"""
    # 读取采样数据
    sampled_prompts = defaultdict(list)
    with open(args.sample_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sampled_prompts[data['style']].append(data['prompt'])
    
    # 处理模型选择
    models = []
    if args.model == "both":
        models = ["Meissonic", "Meissonic_halton"]
    else:
        models = [args.model]

    # 遍历所有模型
    for model_name in models:
        model = Meissonic if model_name == "Meissonic" else Meissonic_halton
        
        # 准备任务列表
        tasks = []
        for style, prompts in sampled_prompts.items():
            for prompt_idx, prompt in enumerate(prompts):
                tasks.append((style, prompt_idx, prompt))
        
        # 修正任务分配方式
        with accelerator.split_between_processes(tasks, apply_padding=True) as distributed_tasks:
            # 创建输出目录
            output_root = os.path.join("sample_inference", "hpsv2", model_name)
            if accelerator.is_local_main_process:
                os.makedirs(output_root, exist_ok=True)
                progress_bar = tqdm(
                    total=len(tasks),
                    desc=f"Sample Generating ({model_name})",
                    position=0,
                    leave=True
                )
            
            accelerator.wait_for_everyone()

            # 处理任务
            for task in tqdm(distributed_tasks):
                style, prompt_idx, prompt = task
                save_dir = os.path.join(output_root, style, str(prompt_idx))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prompt_idx:05d}.jpg")
                metadata_path = os.path.join(save_dir, f"metadata_{prompt_idx:05d}.jsonl")

                # 跳过已存在文件
                if args.skip_existing and os.path.exists(save_path):
                    continue
                
                # 保存元数据（仅主进程）
                if accelerator.is_main_process:
                    with open(metadata_path, "w") as fp:
                        json.dump({"style": style, "prompt": prompt}, fp)
                
                # 生成图像
                try:
                    image = model(prompt, accelerator)
                    image.save(save_path)
                except Exception as e:
                    accelerator.print(f"Error generating {save_path}: {str(e)}")
                
                # 更新进度条
                if accelerator.is_main_process:
                    progress_bar.update(1)
            
            if accelerator.is_main_process:
                progress_bar.close()

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "gen":
        accelerator = Accelerator()
    
        # 设备信息打印
        accelerator.print(f"Running on {accelerator.device}")
        accelerator.print(f"Number of processes: {accelerator.num_processes}")
        generate_images(accelerator, args)
    elif args.mode == "eval":
        evaluate_images(args)
    elif args.mode == "sample_inference":  # 新增模式处理
        accelerator = Accelerator()
    
        # 设备信息打印
        accelerator.print(f"Running on {accelerator.device}")
        accelerator.print(f"Number of processes: {accelerator.num_processes}")
        sample_inference(accelerator, args)
    
    accelerator.print("Operation completed")

"""
running command:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m evaluation.hps_evaluate \
    --mode gen \
    --model Meissonic \
    --output_dir ./outputs/hpsv2 \
    --skip_existing
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 -m evaluation.hps_evaluate \
    --mode gen \
    --model Meissonic_halton \
    --output_dir ./outputs/hpsv2 \
    --skip_existing

python -m evaluation.hps_evaluate \
    --mode eval \
    --input_dir ./outputs/hpsv2/Meissonic \
    --hps_version v2.0
python -m evaluation.hps_evaluate \
    --mode eval \
    --input_dir ./outputs/hpsv2/Meissonic_halton \
    --hps_version v2.0

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m evaluation.hps_evaluate \
    --mode sample_inference \
    --model Meissonic \
    --sample_path sample_testset/sample_hpsv2.jsonl
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 -m evaluation.hps_evaluate \
    --mode sample_inference \
    --model Meissonic_halton \
    --sample_path sample_testset/sample_hpsv2.jsonl
"""