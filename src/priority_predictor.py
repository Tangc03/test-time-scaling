# priority_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorityPredictor(nn.Module):
    """
    输入维度:
        logits          : (B, L, V)   — transformer 输出的 logits
        mask_flag       : (B, L, 1)   — 1 表示当前位置是 [MASK]
        timestep_embeds : (B, L, D_t)
    输出:
        priority_score  : (B, L)      — 越大表示越该在当前 step 翻转
    """
    def __init__(self, vocab_size: int, d_hidden: int = 128, d_timestep: int = 32):
        super().__init__()
        self.proj_logits   = nn.Linear(vocab_size, d_hidden)
        self.proj_timestep = nn.Linear(d_timestep, d_hidden)
        self.proj_maskflag = nn.Linear(1, d_hidden)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1)  # -> (B, L, 1)
        )

    @staticmethod
    def get_timestep_embed(t, dim: int):
        """
        Sin/Cos positional encoding style时间步嵌入.
        t : (B,) or (B,1)  int/long
        返回 (B, 1, dim)
        """
        device = t.device
        half   = dim // 2
        freqs  = torch.exp(
            -torch.arange(half, device=device) * math.log(10000.0) / (half - 1)
        )  # (half,)
        args   = t.float().unsqueeze(-1) * freqs  # (B, half)
        emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb.unsqueeze(1)  # (B,1,dim)

    def forward(self, logits, mask_flag, timestep):
        """
        logits     : (B, L, V)
        mask_flag  : (B, L)           (0/1)
        timestep   : int or Tensor scalar  当前 time step
        """
        B, L, V = logits.shape
        logits_f = self.proj_logits(logits)                     # (B,L,H)
        mask_f   = self.proj_maskflag(mask_flag.float().unsqueeze(-1))  # (B,L,H)

        if isinstance(timestep, int):
            timestep = torch.full((B,1), timestep,
                                  dtype=torch.long, device=logits.device)
        t_emb = self.get_timestep_embed(timestep, self.proj_timestep.in_features)  # (B,1,D_t)
        t_f   = self.proj_timestep(t_emb).repeat(1, L, 1)                           # (B,L,H)

        h = logits_f + mask_f + t_f
        score = self.mlp(h).squeeze(-1)                    # (B,L)
        return score