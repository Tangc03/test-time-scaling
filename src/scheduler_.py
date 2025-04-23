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

import torch.nn as nn

class TokenPriorityPredictor(nn.Module):
    def __init__(self, input_dim=8192, hidden_dim=256, history_steps=3):
        super().__init__()
        self.history_steps = history_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1 + history_steps, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, probs, mask_status, history=[]):
        mask_status = mask_status.float().unsqueeze(-1)
        features = torch.cat([probs, mask_status], -1)
        
        if len(history) > 0:
            hist = torch.stack(history[-self.history_steps:], -1)
            features = torch.cat([features, hist], -1)
        else:
            features = torch.cat([features, torch.zeros_like(features[..., :self.history_steps])], -1)
            
        return self.net(features).squeeze(-1)

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
    priority_data: dict = None  # 新增数据收集字段


class Scheduler(SchedulerMixin, ConfigMixin):
    order = 1

    temperatures: torch.Tensor

    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: str = "cosine",
        use_predictor: bool = False,  # 新增配置参数
        predictor_config: dict = None        # 预测器配置
    ):
        self.temperatures = None
        self.timesteps = None
        self.use_predictor = use_predictor

        # 初始化predictor
        if use_predictor:
            config = predictor_config or {}
            self.predictor = TokenPriorityPredictor(
                input_dim=config.get("codebook_size", 8192),
                hidden_dim=config.get("hidden_dim", 256),
                history_steps=config.get("history_steps", 3)
            )
            self.history_cache = {}
            self.training_cache = []

    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        device: Union[str, torch.device] = None,
    ):
        # 保持原有逻辑不变，新增predictor初始化
        if self.use_predictor and not hasattr(self, 'predictor'):
            self.predictor = TokenPriorityPredictor().to(device)

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
        # 新增参数
        use_predictor: bool = False,
        collect_data: bool = False,
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

            if self.use_predictor:
                # 使用predictor生成优先级
                with torch.set_grad_enabled(collect_data):
                    priorities = self._compute_priorities(probs, unknown_map, sample.device)
                    
                # 生成masking
                sorted_idx = torch.argsort(priorities, dim=-1, descending=True)
                masking = self._create_masking(sorted_idx, mask_len)
                
                # 收集训练数据
                if collect_data:
                    self._collect_training_data(
                        probs.detach(),
                        unknown_map.detach(),
                        priorities.detach(),
                        timestep.item()
                    )
            
            else:
                # 使用预测器生成优先级
                with torch.no_grad():
                    priorities = self._get_priorities(
                        probs=probs,
                        unknown_map=unknown_map,
                        batch_id=id(sample)  # 用于关联历史特征
                    )
                # 生成masking
                masking = self._mask_by_priority(mask_len, priorities, unknown_map)
                
                # 收集训练数据
                if collect_data:
                    self._collect_training_data(
                        probs=probs,
                        unknown_map=unknown_map,
                        priorities=priorities,
                        batch_id=id(sample)
                    )

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
    
    def _get_priorities(self, probs, unknown_map, batch_id):
        """获取优先级分数"""
        # 获取历史特征
        history = self.feature_history.get(batch_id, [])
        
        # 运行预测器
        priorities = self.predictor(
            current_probs=probs,
            mask_status=unknown_map,
            history_features=history
        )
        
        # 更新历史特征(保留最后N步)
        new_feature = priorities.unsqueeze(-1).detach()
        if batch_id in self.feature_history:
            self.feature_history[batch_id].append(new_feature)
            if len(self.feature_history[batch_id]) > self.predictor.history_steps:
                self.feature_history[batch_id].pop(0)
        else:
            self.feature_history[batch_id] = [new_feature]
            
        return priorities

    def _mask_by_priority(self, mask_len, priorities, unknown_map):
        """根据优先级分数进行masking"""
        # 只考虑masked位置
        valid_priorities = torch.where(unknown_map, priorities, -torch.inf)
        
        # 按优先级排序
        sorted_indices = torch.argsort(valid_priorities, dim=-1, descending=True)
        
        # 创建mask
        batch_size = mask_len.size(0)
        masking = torch.zeros_like(unknown_map, dtype=torch.bool)
        for i in range(batch_size):
            k = mask_len[i].long().item()
            masking[i, sorted_indices[i, :k]] = True
            
        return masking

    def _collect_training_data(self, probs, unknown_map, priorities, batch_id):
        """收集训练数据"""
        # 存储当前状态和预测结果
        self.training_cache.append({
            'probs': probs.detach().cpu(),
            'unknown_map': unknown_map.detach().cpu(),
            'priorities': priorities.detach().cpu(),
            'history': [f.detach().cpu() for f in self.feature_history.get(batch_id, [])]
        })
        
        # 限制缓存大小
        if len(self.training_cache) > 1e5:  # 100k样本
            self.training_cache = self.training_cache[1000:]
