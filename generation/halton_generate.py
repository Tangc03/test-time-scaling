# import os
# import sys
# sys.path.append("./")

# import torch
# from torchvision import transforms
# from src.transformer import Transformer2DModel
# from src.halton_pipeline import Pipeline
# from src.halton_scheduler import Scheduler
# from transformers import (
#     CLIPTextModelWithProjection,
#     CLIPTokenizer,
# )
# from diffusers import VQModel

# import os

# device = 'cuda'

# model_path = "checkpoints/Meissonic"
# model = Transformer2DModel.from_pretrained(model_path,subfolder="transformer",)
# vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
# # text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
# text_encoder = CLIPTextModelWithProjection.from_pretrained(   #using original text enc for stable sampling
#             "checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K"
#         )
# tokenizer = CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer",)
# scheduler = Scheduler.from_pretrained(model_path,subfolder="scheduler",)
# pipe=Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=scheduler)

# pipe = pipe.to(device)

# steps = 100
# CFG = 9
# resolution = 1024 
# negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

# batched_generation = False

# def Meissonic_halton(prompt):
#     # Generate image using the pipeline
#     image = pipe(
#         prompt=prompt, 
#         negative_prompt=negative_prompt,
#         height=resolution,
#         width=resolution,
#         guidance_scale=CFG,
#         num_inference_steps=steps
#     ).images[0]

#     return image

import torch
from torchvision import transforms
from src.transformer import Transformer2DModel
from src.halton_pipeline import Pipeline
from src.halton_scheduler import Scheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import VQModel

# 配置参数
model_path = "checkpoints/Meissonic"
steps = 100
CFG = 9
resolution = 1024
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

# 全局管道实例
_pipe = None

def setup_pipeline(accelerator):
    global _pipe
    if _pipe is not None:
        return _pipe
    
    device = accelerator.device
    
    # 初始化模型组件
    transformer = Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer").to(device)
    vq_model = VQModel.from_pretrained(
        model_path, subfolder="vqvae").to(device)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(
        model_path, subfolder="scheduler")
    
    # 构建管道
    _pipe = Pipeline(
        vq_model, 
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler
    ).to(device)
    
    return _pipe

def Meissonic_halton(prompt, accelerator):
    global _pipe
    if _pipe is None:
        _pipe = setup_pipeline(accelerator)
    
    with torch.no_grad():
        image = _pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=resolution,
            width=resolution,
            guidance_scale=CFG,
            num_inference_steps=steps
        ).images[0]
    
    return image