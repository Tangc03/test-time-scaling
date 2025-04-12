import os
import hpsv2

# Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
all_prompts = hpsv2.benchmark_prompts('all') 

# Iterate over the benchmark prompts to generate images
for style, prompts in all_prompts.items():
    for idx, prompt in enumerate(prompts):
        print(f"Generating image for prompt {idx} in style {style}: {prompt}")