import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from src.priority_predictor import PriorityPredictor

@dataclass
class SchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor = None


class Scheduler(SchedulerMixin, ConfigMixin):
    order = 1

    temperatures : torch.Tensor
    priority_net : Optional[PriorityPredictor] = None

    @register_to_config
    def __init__(
        self,
        mask_token_id         : int,
        vocab_size            : int,
        priority_ckpt_path    : Optional[str] = None,
        masking_schedule      : str = "cosine",
    ):
        self.temperatures = None
        self.timesteps    = None
        # --------- 新增：加载优先级网络 ----------
        self.priority_net = PriorityPredictor(vocab_size)
        if priority_ckpt_path is not None:
            state = torch.load(priority_ckpt_path, map_location="cpu")
            self.priority_net.load_state_dict(state)
        self.priority_net.eval().requires_grad_(False)
        # --------------------------------------

    # set_timesteps 与 add_noise 与原版本相同，略

    def step(
        self,
        model_output : torch.Tensor,      # logits (B,C,H,W)
        timestep     : torch.long,
        sample       : torch.LongTensor,  # current token ids (B,H,W)
        starting_mask_ratio : int = 1,
        generator    : Optional[torch.Generator] = None,
        return_dict  : bool = True,
    ):
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4
        if two_dim_input:
            B, C, H, W  = model_output.shape
            sample      = sample.reshape(B, H * W)
            logits_flat = model_output.reshape(B, C, H * W).permute(0,2,1)  # (B,L,V)
        else:
            logits_flat = model_output         # 已经是 (B,L,V)

        unknown_map = sample == self.config.mask_token_id  # (B,L)
        probs       = logits_flat.softmax(dim=-1)
        # 预测 token (teacher)
        device = probs.device
        probs_ = probs.to(generator.device) if generator else probs
        if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
            probs_ = probs_.float()
        pred_tokens = torch.multinomial(
            probs_.reshape(-1, probs_.size(-1)), 1, generator=generator
        ).to(device=device).view(*probs.shape[:-1])
        pred_tokens = torch.where(unknown_map, pred_tokens, sample)

        if timestep == 0:
            prev_sample = pred_tokens
        else:
            # ------- 计算 mask_ratio（同旧逻辑）-------
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)
            if self.config.masking_schedule == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError
            mask_ratio *= starting_mask_ratio
            mask_len = (seq_len * mask_ratio).floor()
            mask_len = torch.min(unknown_map.sum(-1, keepdim=True) - 1, mask_len)
            mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)
            mask_len = mask_len.squeeze(-1)          # (B,)

            # ----------- 用 priority_net 选 k 个 token ------------
            with torch.no_grad():
                scores = self.priority_net(
                    logits_flat, unknown_map.float(), int(timestep)
                )                                     # (B,L)
                scores = torch.where(unknown_map, scores,
                                     torch.full_like(scores, -1e4))
            masking = torch.zeros_like(unknown_map)
            for i in range(masking.shape[0]):
                k = int(mask_len[i])
                if k > 0:
                    idx = scores[i].topk(k).indices
                    masking[i, idx] = True
            # -----------------------------------------------------

            prev_sample = torch.where(masking,
                                      self.config.mask_token_id,
                                      pred_tokens)

        if two_dim_input:
            B, H, W = sample.shape[0], int(math.sqrt(sample.shape[1])), int(math.sqrt(sample.shape[1]))
            prev_sample          = prev_sample.reshape(B, H, W)
            pred_original_sample = pred_tokens.reshape(B, H, W)
        else:
            pred_original_sample = pred_tokens

        if not return_dict:
            return (prev_sample, pred_original_sample)
        return SchedulerOutput(prev_sample, pred_original_sample)