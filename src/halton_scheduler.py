import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def gumbel_noise(t, generator=None):
    device = generator.device if generator is not None else t.device
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))

def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking

def build_halton_mask(probs: torch.Tensor, 
                      mask_len: torch.LongTensor,
                      input_size: int = None,
                      nb_point: int = 10_000) -> torch.BoolTensor:
    """
    生成与mask_by_random_topk格式兼容的Halton序列掩码
    
    Args:
        probs:       概率张量 [batch_size, seq_len]
        mask_len:    每个样本需掩码数量 [batch_size, 1]
        input_size:  可选参数，当序列非正方形时自动推断
        nb_point:    Halton序列采样点数
    
    Returns:
        BoolTensor掩码 [batch_size, seq_len]
    """
    batch_size, seq_len = probs.shape
    
    # 自动推断二维结构 (默认按正方形处理)
    grid_size = int(np.sqrt(seq_len)) if input_size is None else input_size
    if grid_size**2 != seq_len:
        raise ValueError(f"Sequence length {seq_len} is not perfect square, specify input_size manually")

    # Halton序列生成函数
    def halton(b, n_sample):
        """Halton序列生成器"""
        n, d = 0, 1
        res = []
        for _ in range(n_sample):
            x = d - n
            if x == 1:
                n = 1
                d *= b
            else:
                y = d // b
                while x <= y:
                    y //= b
                n = (b + 1) * y - x
            res.append(n / d)
        return torch.tensor(res)

    # 生成二维坐标序列
    coords_x = halton(2, nb_point) * grid_size
    coords_y = halton(3, nb_point) * grid_size
    coordinates = torch.stack([coords_x.long(), coords_y.long()], dim=1)

    # 去重并转换为一维索引
    _, unique_idx = torch.unique(coordinates, dim=0, return_inverse=True)
    flat_indices = coordinates[:,0] * grid_size + coordinates[:,1]
    flat_indices = flat_indices[unique_idx]  # 去除重复坐标

    # 创建优先级掩码模板
    priority_mask = torch.zeros((grid_size**2,), dtype=torch.long)
    priority_mask[flat_indices[:grid_size**2]] = torch.arange(len(flat_indices))
    
    # 批量扩展 & 动态掩码
    batch_priorities = priority_mask.unsqueeze(0).expand(batch_size, -1)
    sorted_idx = torch.argsort(batch_priorities, dim=-1, descending=True)
    
    # 生成动态长度掩码（关键修改处）
    mask = torch.zeros_like(probs, dtype=torch.bool)
    for i in range(batch_size):
        k = int(mask_len[i].item())  # 显式转换为Python整数
        mask[i, sorted_idx[i, :k]] = True  # 现在k是整数
    
    return mask


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor = None


class Scheduler(SchedulerMixin, ConfigMixin):
    order = 1

    temperatures: torch.Tensor

    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: str = "cosine",
    ):
        self.temperatures = None
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        device: Union[str, torch.device] = None,
    ):
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)

        if isinstance(temperature, (tuple, list)):
            self.temperatures = torch.linspace(temperature[0], temperature[1], num_inference_steps, device=device)
        else:
            self.temperatures = torch.linspace(temperature, 0.01, num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        starting_mask_ratio: int = 1,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4

        if two_dim_input:
            batch_size, codebook_size, height, width = model_output.shape
            sample = sample.reshape(batch_size, height * width)
            model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)

        unknown_map = sample == self.config.mask_token_id

        probs = model_output.softmax(dim=-1)

        device = probs.device
        probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
        if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
            probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
        probs_ = probs_.reshape(-1, probs.size(-1))
        pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=device)
        pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
        pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)

        if timestep == 0:
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.config.masking_schedule == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

            mask_ratio = starting_mask_ratio * mask_ratio

            mask_len = (seq_len * mask_ratio).floor()
            # do not mask more than amount previously masked
            mask_len = torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            # mask at least one
            mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)

            selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

            # masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)
            masking = build_halton_mask(selected_probs, mask_len, input_size=height, nb_point=seq_len)

            # Masks tokens with lower confidence.
            prev_sample = torch.where(masking, self.config.mask_token_id, pred_original_sample)

        if two_dim_input:
            prev_sample = prev_sample.reshape(batch_size, height, width)
            pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return SchedulerOutput(prev_sample, pred_original_sample)

    def add_noise(self, sample, timesteps, generator=None):
        step_idx = (self.timesteps == timesteps).nonzero()
        ratio = (step_idx + 1) / len(self.timesteps)

        if self.config.masking_schedule == "cosine":
            mask_ratio = torch.cos(ratio * math.pi / 2)
        elif self.config.masking_schedule == "linear":
            mask_ratio = 1 - ratio
        else:
            raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

        mask_indices = (
            torch.rand(
                sample.shape, device=generator.device if generator is not None else sample.device, generator=generator
            ).to(sample.device)
            < mask_ratio
        )

        masked_sample = sample.clone()

        masked_sample[mask_indices] = self.config.mask_token_id

        return masked_sample