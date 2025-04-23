import torch
import torch.nn as nn
import torch.nn.functional as F

class PriorityPredictor(nn.Module):
    def __init__(self, vocab_size=8192, time_embed_dim=32, hidden_dim=256):
        super().__init__()
        self.time_embed = nn.Embedding(1000, time_embed_dim)  # 假设最大步数1000
        self.logit_proj = nn.Linear(vocab_size, 16)  # 压缩logits
        self.feature_proj = nn.Linear(16 + 1 + time_embed_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, logits, mask_status, timestep):
        # logits: [batch, seq, vocab]
        # mask_status: [batch, seq]
        # timestep: [batch]
        
        # 处理logits
        logit_repr = self.logit_proj(logits)  # [batch, seq, 16]
        
        # 时间嵌入
        time_emb = self.time_embed(timestep.long())  # [batch, time_embed_dim]
        time_emb = time_emb.unsqueeze(1).expand(-1, logit_repr.size(1), -1)  # [batch, seq, time_embed]
        
        # 拼接特征
        features = torch.cat([
            logit_repr,
            mask_status.unsqueeze(-1),  # [batch, seq, 1]
            time_emb
        ], dim=-1)  # [batch, seq, 16+1+time_embed]
        
        # 预测分数
        features = self.feature_proj(features)
        scores = self.mlp(features).squeeze(-1)  # [batch, seq]
        return scores