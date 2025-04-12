"""
参考
import hpsv2
def generate_images(accelerator, args):
    '''加速生成实现'''
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
        
        # 准备任务列表 (style, prompt_idx, prompt)
        tasks = []
        for style, prompts in all_prompts.items():
            for prompt_idx, prompt in enumerate(prompts):
                tasks.append((style, prompt_idx, prompt, sample_idx))
        
        # 分布式任务分割
        tasks = accelerator.split_between_processes(tasks)
        
        # 主进程创建目录
        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # 带进度条的生成
        with accelerator.main_process_first():
            progress_bar = tqdm(tasks, 
                              desc=f"Generating ({model_name})",
                              disable=not accelerator.is_local_main_process)
            
            for style, prompt_idx, prompt, sample_idx in progress_bar:
                # 构造保存路径
                save_dir = os.path.join(args.output_dir, model_name, style)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{prompt_idx:05d}_{sample_idx:04d}.jpg")
                
                # 跳过已存在文件
                if args.skip_existing and os.path.exists(save_path):
                    continue
                
                # 生成图像
                try:
                    image = model(prompt)
                    image.save(save_path)
                except Exception as e:
                    accelerator.print(f"Error generating {save_path}: {str(e)}")

def generate_images(model_name, prompts, output_dir, n_samples, skip_existing, accelerator):
    '''Generate images using the specified model in distributed mode.'''
    
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
                image = Meissonic(prompt) if model_name == "Meissonic" else Meissonic_halton(prompt)
                image.save(output_file)
                accelerator.print(f"  - Process {accelerator.process_index} saved sample {sample_idx+1}/{n_samples}")
            except Exception as e:
                accelerator.print(f"Process {accelerator.process_index} error: {str(e)}")

def main():
    args = parse_args()
    accelerator = Accelerator()
    
    # 获取所有测试集文件（不包含train或val的txt文件）
    test_files = []
    for filename in os.listdir(args.dataset_path):
        if filename.endswith(".txt") and "train" not in filename and "val" not in filename:
            test_files.append(os.path.join(args.dataset_path, filename))
    
    # 确定需要使用的模型列表
    models = []
    if args.model == "both":
        models = ["Meissonic", "Meissonic_halton"]
    else:
        models = [args.model]
    
    # 处理每个测试集文件
    for test_file in test_files:
        task_name = os.path.basename(test_file).replace('.txt', '')
        with open(test_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        # 为每个模型生成对应的图像
        for model in models:
            output_root = os.path.join(args.output_dir, model)
            task_dir = os.path.join(output_root, task_name)
            
            # 分布式创建目录
            if accelerator.is_local_main_process:
                os.makedirs(task_dir, exist_ok=True)
            accelerator.wait_for_everyone()
            
            # 保存prompts.txt（仅主进程）
            if accelerator.is_local_main_process:
                prompts_file = os.path.join(task_dir, "prompts.txt")
                with open(prompts_file, 'w') as pf:
                    pf.write('\n'.join(prompts))
            accelerator.wait_for_everyone()
            
            # 生成任务列表
            tasks = []
            for prompt in prompts:
                for sample_idx in range(args.n_samples):
                    tasks.append((prompt, sample_idx))
            
            # 分布式任务划分
            tasks = accelerator.split_between_processes(tasks)
            
            # 执行生成任务
            generate_images_for_task(model, tasks, task_dir, args.skip_existing, accelerator)
    
    accelerator.print("All tasks completed.")

这三段代码考虑的是使用hps，geneval，compbench三个testset的完整情况，为了节省时间，我们现在希望完成以下事项：
1. 统计三个测试集的相关数据
hpsv2统计不同style的prompts数量

geneval的metadata路径为prompts/evaluation_metadata.jsonl，结构如下：
{"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "a photo of a bench"}
{"tag": "single_object", "include": [{"class": "cow", "count": 1}], "prompt": "a photo of a cow"}
{"tag": "single_object", "include": [{"class": "bicycle", "count": 1}], "prompt": "a photo of a bicycle"}
其中tag表示的是对应的人物特点，统计各任务的prompts数量

compbench的prompts在T2I-CompBench_dataset中，以txt的方式存储，我们将文件名不包含train和val的txt文件作为测试集，文件名将下划线换成空格，作为任务名称，统计各任务的prompts数量
"""

"""
HPSv2 统计结果：
  anime: 800 prompts
  concept-art: 800 prompts
  paintings: 800 prompts
  photo: 800 prompts

==================================================

Geneval 统计结果：
  single_object: 80 prompts
  two_object: 99 prompts
  counting: 80 prompts
  colors: 94 prompts
  position: 100 prompts
  color_attr: 100 prompts

==================================================

CompBench 统计结果：
  color: 1000 prompts
  numeracy: 1000 prompts
  shape: 1000 prompts
  complex: 1000 prompts
  3d spatial: 1000 prompts
  non-spatial: 1000 prompts
  spatial: 1000 prompts
  texture: 1000 prompts

参照以上的结果，为了让测试结果更快展现，我们需要对于testset进行采样，
对于HPSv2数据集，我们对于四种style各选取100条数据，作为sample后的数据集
对于Geneval数据集，我们对于每种tag各选取50条数据，作为sample后的数据集
对于CompBench数据集，我们对于每种任务各选取100条数据，作为sample后的数据集

数据集sample后保存在sample_testset中
分别保存在sample_hpsv2.jsonl，sample_geneval.jsonl，sample_compbench.jsonl中
sample_hpsv2.jsonl需要包含style和prompt字段，结构如下：
{"style": "anime", "prompt": "a photo of a cat"}
{"style": "concept-art", "prompt": "a photo of a dog"}
{"style": "paintings", "prompt": "a photo of a cow"}
{"style": "photo", "prompt": "a photo of a bench"}
sample_geneval.jsonl结构如下：
{"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "a photo of a bench"}
sample_compbench.jsonl结构如下：
{"task": "color", "prompt": "a photo of a cow"}
"""

import os
import json
import argparse
import random
from collections import defaultdict
import hpsv2

def count_hpsv2():
    """统计HPSv2不同style的prompts数量"""
    all_prompts = hpsv2.benchmark_prompts('all')
    return {style: len(prompts) for style, prompts in all_prompts.items()}

def sample_hpsv2(output_dir, samples_per_style=100):
    """采样HPSv2数据集"""
    os.makedirs(output_dir, exist_ok=True)
    all_prompts = hpsv2.benchmark_prompts('all')
    
    with open(os.path.join(output_dir, "sample_hpsv2.jsonl"), 'w') as f:
        for style, prompts in all_prompts.items():
            sampled = random.sample(prompts, min(samples_per_style, len(prompts)))
            for prompt in sampled:
                json.dump({"style": style, "prompt": prompt}, f)
                f.write('\n')

def count_geneval(metadata_path):
    """统计Geneval不同tag的prompts数量"""
    tag_counts = defaultdict(int)
    with open(metadata_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            tag_counts[data['tag']] += 1
    return dict(tag_counts)

def sample_geneval(metadata_path, output_dir, samples_per_tag=50):
    """采样Geneval数据集"""
    os.makedirs(output_dir, exist_ok=True)
    tag_entries = defaultdict(list)
    
    # 按tag分组
    with open(metadata_path) as f:
        for line in f:
            entry = json.loads(line)
            tag_entries[entry['tag']].append(entry)
    
    # 采样并保存
    with open(os.path.join(output_dir, "sample_geneval.jsonl"), 'w') as f:
        for tag, entries in tag_entries.items():
            sampled = random.sample(entries, min(samples_per_tag, len(entries)))
            for entry in sampled:
                json.dump(entry, f)
                f.write('\n')

def count_compbench(dataset_path):
    """统计CompBench各任务的prompts数量"""
    task_counts = {}
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt") and "train" not in filename and "val" not in filename:
            task_name = filename.replace(".txt", "").replace("_", " ")
            with open(os.path.join(dataset_path, filename)) as f:
                task_counts[task_name] = sum(1 for line in f if line.strip())
    return task_counts

def sample_compbench(dataset_path, output_dir, samples_per_task=100):
    """采样CompBench数据集"""
    os.makedirs(output_dir, exist_ok=True)
    task_prompts = {}
    
    # 读取所有任务文件
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt") and "train" not in filename and "val" not in filename:
            task_name = filename.replace(".txt", "").replace("_", " ")
            with open(os.path.join(dataset_path, filename)) as f:
                prompts = [line.strip() for line in f if line.strip()]
                task_prompts[task_name] = prompts
    
    # 采样并保存
    with open(os.path.join(output_dir, "sample_compbench.jsonl"), 'w') as f:
        for task, prompts in task_prompts.items():
            sampled = random.sample(prompts, min(samples_per_task, len(prompts)))
            for prompt in sampled:
                json.dump({"task": task, "prompt": prompt}, f)
                f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='测试集统计和采样工具')
    parser.add_argument('--mode', choices=['stat', 'sample'], required=True,
                       help='运行模式：stat统计模式，sample采样模式')
    parser.add_argument('--output-dir', default='sample_testset',
                       help='采样数据输出目录（仅sample模式有效）')
    args = parser.parse_args()

    if args.mode == 'stat':
        # HPSv2 统计
        hps_stats = count_hpsv2()
        print("HPSv2 统计结果：")
        [print(f"  {k}: {v} prompts") for k, v in hps_stats.items()]
        
        # Geneval 统计
        geneval_stats = count_geneval("prompts/evaluation_metadata.jsonl")
        print("\nGeneval 统计结果：")
        [print(f"  {k}: {v} prompts") for k, v in geneval_stats.items()]
        
        # CompBench 统计
        compbench_stats = count_compbench("T2I-CompBench_dataset")
        print("\nCompBench 统计结果：")
        [print(f"  {k}: {v} prompts") for k, v in compbench_stats.items()]
    
    elif args.mode == 'sample':
        print("开始采样数据集...")
        # 采样HPSv2
        sample_hpsv2(args.output_dir, samples_per_style=100)
        # 采样Geneval
        sample_geneval("prompts/evaluation_metadata.jsonl", args.output_dir, samples_per_tag=50)
        # 采样CompBench
        sample_compbench("T2I-CompBench_dataset", args.output_dir, samples_per_task=100)
        print(f"采样完成，结果保存在 {args.output_dir} 目录")

if __name__ == "__main__":
    main()

# running command
# python sample_testset.py --mode stat
# python sample_testset.py --mode sample --output-dir sample_testset