import os
import json
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from collections import defaultdict

# 导入模型
from generation.generate import Meissonic
from generation.halton_generate import Meissonic_halton

def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation images using Meissonic models")
    # 添加模式选择参数
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "sample_inference"],
        default="default",
        help="Operation mode: default (normal) or sample_inference"
    )
    # 添加sample路径参数
    parser.add_argument(
        "--sample_path",
        type=str,
        default="sample_testset/sample_geneval.jsonl",
        help="Path to sampled prompts dataset"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="prompts/evaluation_metadata.jsonl",
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["Meissonic", "Meissonic_halton", "both"],
        default="both",
        help="Model to use for generation: 'Meissonic', 'Meissonic_halton', or 'both'"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Base directory to save generated images"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples to generate per prompt"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if output file already exists"
    )
    return parser.parse_args()

def generate_images(model_name, prompts, output_dir, n_samples, skip_existing, accelerator):
    """Generate images using the specified model in distributed mode."""
    
    # 设置输出目录
    if model_name == "Meissonic":
        output_path = os.path.join(output_dir, "Meissonic")
    else:
        output_path = os.path.join(output_dir, "Meissonic_halton")
    
    # 所有进程共同创建目录（exist_ok=True保证安全）
    os.makedirs(output_path, exist_ok=True)
    
    # 对每个提示生成图像
    for index, metadata in enumerate(tqdm(prompts, desc=f"Generating images with {model_name}", disable=not accelerator.is_main_process)):
        prompt = metadata['prompt']
        tag = metadata.get('tag', 'unknown')
        
        # 创建目标文件夹结构
        outpath = os.path.join(output_path, f"{index:0>5}")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        # 仅主进程保存元数据（避免重复写入）
        if accelerator.is_main_process:
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp)
        
        accelerator.print(f"[{model_name}] Prompt ({index+1}/{len(prompts)}): '{prompt}' (Tag: {tag})")
        
        # 计算当前进程需要生成的样本索引
        all_samples = list(range(n_samples))
        my_samples = all_samples[accelerator.process_index::accelerator.num_processes]
        
        # 处理本进程分配的样本
        for sample_idx in my_samples:
            output_file = os.path.join(sample_path, f"{sample_idx:05d}.png")
            
            if skip_existing and os.path.exists(output_file):
                accelerator.print(f"  - Sample {sample_idx+1}/{n_samples} already exists, skipping")
                continue
                
            try:
                # 生成图像（假设模型内部已处理设备分配）
                image = Meissonic(prompt, accelerator) if model_name == "Meissonic" else Meissonic_halton(prompt, accelerator)
                image.save(output_file)
                # accelerator.print(f"  - Process {accelerator.process_index} saved sample {sample_idx+1}/{n_samples}")
            except Exception as e:
                accelerator.print(f"Process {accelerator.process_index} error: {str(e)}")

def generate_sample_inference(accelerator, args):
    """基于采样数据的生成实现"""
    # 读取采样数据
    sampled_prompts = defaultdict(list)
    with open(args.sample_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sampled_prompts[data['tag']].append(data['prompt'])

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
        for tag, prompts in sampled_prompts.items():
            for prompt_idx, prompt in enumerate(prompts):
                tasks.append((tag, prompt_idx, prompt))
        
        # 修正任务分配方式
        with accelerator.split_between_processes(tasks, apply_padding=True) as distributed_tasks:
            # 创建输出目录
            output_root = os.path.join("sample_inference", "geneval", model_name)
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
                tag, prompt_idx, prompt = task
                save_dir = os.path.join(output_root, tag, str(prompt_idx))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prompt_idx:05d}.jpg")
                metadata_path = os.path.join(save_dir, f"metadata_{prompt_idx:05d}.jsonl")

                # 跳过已存在文件
                if args.skip_existing and os.path.exists(save_path):
                    continue
                
                # 保存元数据（仅主进程）
                if accelerator.is_main_process:
                    with open(metadata_path, "w") as fp:
                        json.dump({"tag": tag, "prompt": prompt}, fp)
                
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
            
def main():
    args = parse_args()
    accelerator = Accelerator()
    
    # 设置随机种子
    seed = 42 + accelerator.process_index
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    if args.mode == "sample_inference":
        # 加载采样数据
        with open(args.sample_path) as f:
            sample_data = [json.loads(line) for line in f]
        
        # 执行生成任务
        if args.model in ["Meissonic", "both"]:
            generate_sample_inference(accelerator, args)
        if args.model in ["Meissonic_halton", "both"]:
            generate_sample_inference(accelerator, args)
    else:
        # 原有生成逻辑
        with open(args.metadata_file) as fp:
            prompts = [json.loads(line) for line in fp]
        
        if args.model in ["Meissonic", "both"]:
            generate_images("Meissonic", prompts, args.output_dir, 
                           args.n_samples, args.skip_existing, accelerator)
        if args.model in ["Meissonic_halton", "both"]:
            generate_images("Meissonic_halton", prompts, args.output_dir,
                           args.n_samples, args.skip_existing, accelerator)
    
    accelerator.wait_for_everyone()
    accelerator.print("Operation completed.")

if __name__ == "__main__":
    main()

"""
running command:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m evaluation.geneval_evaluate \
    --model Meissonic \
    --n_samples 4 \
    --output_dir ./outputs/geneval \
    --metadata_file ./prompts/evaluation_metadata.jsonl \
    --skip_existing

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 -m evaluation.geneval_evaluate \
    --model Meissonic_halton \
    --n_samples 4 \
    --output_dir ./outputs/geneval \
    --metadata_file ./prompts/evaluation_metadata.jsonl \
    --skip_existing

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m evaluation.geneval_evaluate \
    --mode sample_inference \
    --model Meissonic \
    --skip_existing \
    --sample_path ./sample_testset/sample_geneval.jsonl

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 -m evaluation.geneval_evaluate \
    --mode sample_inference \
    --model Meissonic_halton \
    --skip_existing \
    --sample_path ./sample_testset/sample_geneval.jsonl
"""