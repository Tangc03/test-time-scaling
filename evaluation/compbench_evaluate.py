import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch
import hashlib
import json
from accelerate import Accelerator
from collections import defaultdict

# 假设模型生成函数已正确导入
from generation.generate import Meissonic
from generation.halton_generate import Meissonic_halton

def parse_args():
    parser = argparse.ArgumentParser(description="Generate T2I-CompBench evaluation images")
    # 添加模式选择参数
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "sample_inference"],
        default="default",
        help="Operation mode: default (normal) or sample_inference"
    )
    # 添加采样数据集路径参数
    parser.add_argument(
        "--sample_path",
        type=str,
        default="sample_testset/sample_compbench.jsonl",
        help="Path to sampled prompts dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./T2I-CompBench_dataset",
        help="Path to the T2I-CompBench_dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["Meissonic", "Meissonic_halton", "both"],
        default="both",
        help="Model(s) to use for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./compbench_outputs",
        help="Base directory to save generated images"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples per prompt"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing images"
    )
    return parser.parse_args()

def clean_filename(text, max_length=200):
    """清理文本以生成安全的文件名，替换非法字符并限制长度"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')
    text = text.strip()
    if len(text) > max_length:
        hash_part = hashlib.md5(text.encode()).hexdigest()[:8]
        text = text[:max_length-9] + '_' + hash_part
    return text

def generate_images_for_task(model_name, task_dir, all_prompts, args, accelerator):
    """重构后的生成函数"""
    generate_func = Meissonic if model_name == "Meissonic" else Meissonic_halton
    
    # 准备任务列表
    tasks = []
    for style, prompts in all_prompts.items():
        for prompt_idx, prompt in enumerate(prompts):
            tasks.append((style, prompt_idx, prompt))
    
    # 使用上下文管理器分配任务
    with accelerator.split_between_processes(tasks, apply_padding=True) as distributed_tasks:
        # 主进程准备进度条
        if accelerator.is_local_main_process:
            progress_bar = tqdm(
                total=len(tasks),
                desc=f"Generating {model_name}/{os.path.basename(task_dir)}",
                position=0,
                leave=True
            )
        
        accelerator.wait_for_everyone()
        
        # 处理分配的任务
        for task in tqdm(distributed_tasks):
            style, prompt_idx, prompt = task
            save_dir = os.path.join(task_dir, style)
            os.makedirs(save_dir, exist_ok=True)
            img_name = f"{clean_filename(prompt)}_{prompt_idx:04d}.png"
            img_path = os.path.join(save_dir, img_name)
            
            if args.skip_existing and os.path.exists(img_path):
                continue
                
            try:
                # 生成图像
                image = generate_func(prompt, accelerator)
                image.save(img_path)
            except Exception as e:
                accelerator.print(f"Error generating {img_path}: {e}")
            
            # 更新进度条
            if accelerator.is_local_main_process:
                progress_bar.update(1)
        
        if accelerator.is_local_main_process:
            progress_bar.close()

def generate_sample_inference(accelerator, args):
    """基于采样数据的生成实现"""
    # 读取采样数据
    sampled_prompts = defaultdict(list)
    with open(args.sample_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sampled_prompts[data['task']].append(data['prompt'])

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
        for task, prompts in sampled_prompts.items():
            for prompt_idx, prompt in enumerate(prompts):
                tasks.append((task, prompt_idx, prompt))
        
        # 修正任务分配方式
        with accelerator.split_between_processes(tasks, apply_padding=True) as distributed_tasks:
            # 创建输出目录
            output_root = os.path.join("sample_inference", "compbench", model_name)
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
                task, prompt_idx, prompt = task
                save_dir = os.path.join(output_root, task, str(prompt_idx))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prompt_idx:05d}.jpg")
                metadata_path = os.path.join(save_dir, f"metadata_{prompt_idx:05d}.jsonl")

                # 跳过已存在文件
                if args.skip_existing and os.path.exists(save_path):
                    continue
                
                # 保存元数据（仅主进程）
                if accelerator.is_main_process:
                    with open(metadata_path, "w") as fp:
                        json.dump({"task": task, "prompt": prompt}, fp)
                
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

    if args.mode == "sample_inference":
        # 加载采样数据
        with open(args.sample_path) as f:
            sample_data = [json.loads(line) for line in f]
        
        # 为每个模型生成图像
        models = ["Meissonic", "Meissonic_halton"] if args.model == "both" else [args.model]
        
        for model in models:
            # 创建输出目录
            output_root = os.path.join(args.output_dir, model)
            os.makedirs(output_root, exist_ok=True)
            
            # 执行采样生成
            generate_sample_inference(
                accelerator,
                args
            )
    
    else:
        # 获取所有测试集文件
        test_files = [f for f in os.listdir(args.dataset_path) 
                     if f.endswith(".txt") and "train" not in f and "val" not in f]
        
        models = ["Meissonic", "Meissonic_halton"] if args.model == "both" else [args.model]
        
        for model in models:
            output_root = os.path.join(args.output_dir, model)
            
            # 每个测试文件视为一个独立任务集
            for test_file in test_files:
                task_name = os.path.basename(test_file).replace('.txt', '')
                task_dir = os.path.join(output_root, task_name)
                
                # 加载prompts
                with open(os.path.join(args.dataset_path, test_file)) as f:
                    prompts = [line.strip() for line in f if line.strip()]
                
                # 组织成类hpsv2的格式
                all_prompts = {task_name: prompts}
                
                # 创建输出目录
                if accelerator.is_local_main_process:
                    os.makedirs(task_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                
                # 执行生成
                generate_images_for_task(
                    model_name=model,
                    task_dir=task_dir,
                    all_prompts=all_prompts,
                    args=args,
                    accelerator=accelerator
                )
    
    accelerator.wait_for_everyone()
    accelerator.print("Operation completed.")

if __name__ == "__main__":
    main()

"""
running command:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m evaluation.compbench_evaluate \
    --dataset_path ./T2I-CompBench_dataset \
    --model Meissonic \
    --output_dir ./outputs/compbench \
    --n_samples 1 \
    --skip_existing
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 -m evaluation.compbench_evaluate \
    --dataset_path ./T2I-CompBench_dataset \
    --model Meissonic_halton \
    --output_dir ./outputs/compbench \
    --n_samples 1 \
    --skip_existing

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 -m evaluation.compbench_evaluate \
    --mode sample_inference \
    --sample_path sample_testset/sample_compbench.jsonl \
    --model Meissonic
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 -m evaluation.compbench_evaluate \
    --mode sample_inference \
    --sample_path sample_testset/sample_compbench.jsonl \
    --model Meissonic_halton
"""