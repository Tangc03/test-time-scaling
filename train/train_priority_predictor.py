from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import argparse
from src.priority_predictor import PriorityPredictor
from accelerate import Accelerator
import torch.nn.functional as F

import math, os, torch, json
from datasets import load_dataset
from src.priority_predictor import PriorityPredictor

from src.load_transformer import load_trained_transformer  # 写成你已有的加载函数

from tqdm import tqdm

MASK_ID = 8191                   # 按你的 codebook_size-1 修改
VOCAB   = 8192
BATCH   = 8
TIMESTEPS = 16                   # 与 inference scheduler 保持一致
DEVICE  = "cuda"

def timestep_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=DEVICE) *
                      math.log(10000.) / (half-1))
    args  = t.float().unsqueeze(-1) * freqs
    emb   = torch.cat([torch.sin(args), torch.cos(args)], -1)
    if dim % 2 == 1: emb = F.pad(emb, (0,1))
    return emb

def collate(batch):
    # batch 里的 each['ids'] =  (L,)  ground-truth token ids
    ids = [torch.tensor(b["ids"]) for b in batch]
    ids = torch.stack(ids)   # (B,L)
    return {"ids": ids}

def build_dataloader():
    ds = load_dataset("your_dataset", split="train")  # **替换**
    return DataLoader(ds, batch_size=BATCH,
                      shuffle=True, collate_fn=collate)

def collect_features(transformer, dataloader):
    features, labels = [], []
    transformer.eval()
    for batch in tqdm(dataloader):
        tgt = batch["ids"].to(DEVICE)                             # (B,L)
        B, L = tgt.shape
        x_t  = torch.full_like(tgt, MASK_ID)                      # 初始化全 MASK
        for step in range(TIMESTEPS, -1, -1):
            with torch.no_grad():
                logits = transformer(x_t, timestep=step)   # (B, L, V)
                logits = logits.permute(0,2,1) if logits.shape[1]==VOCAB else logits
            mask_flag = (x_t == MASK_ID)                          # (B,L)
            feat = torch.stack([logits, mask_flag.float()], dim=2) # 存储原始
            features.append({"logits": logits.cpu(),
                             "mask": mask_flag.cpu(),
                             "timestep": torch.full((B,1), step),
                             "gt": tgt.cpu()})
            # teacher forcing：一步直接喂真值，免得推理误差累积
            x_t = tgt
    return features

def create_dataset(features):
    # 这里简单演示写文件 -> 再用 PyTorch Dataset 加载
    torch.save(features, "priority_feats.pt")

class PriorityDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        logits = item["logits"]              # (B,L,V)
        mask   = item["mask"]                # (B,L)
        gt     = item["gt"]
        label  = (mask & (gt != MASK_ID)).long()  # (B,L) 1=应翻转
        return {"logits": logits, "mask": mask, "label": label,
                "timestep": item["timestep"]}

def train_priority():
    net = PriorityPredictor(VOCAB).to(DEVICE)
    optim = torch.optim.AdamW(net.parameters(), 1e-4)
    ds    = PriorityDataset("priority_feats.pt")
    dl    = DataLoader(ds, batch_size=1, shuffle=True)   # 已含 (B,L,*) 维度
    for epoch in range(3):
        pbar = tqdm(dl)
        for batch in pbar:
            logits = batch["logits"].squeeze(0).to(DEVICE)   # (B,L,V)
            mask   = batch["mask"].squeeze(0).to(DEVICE)
            label  = batch["label"].squeeze(0).to(DEVICE).float()
            t_step = int(batch["timestep"][0,0])
            pred   = net(logits, mask.float(), t_step)        # (B,L)
            loss   = F.binary_cross_entropy_with_logits(
                        pred[mask], label[mask])              # 只计算 MASK 处
            optim.zero_grad(); loss.backward(); optim.step()
            pbar.set_description(f"loss {loss.item():.4f}")
    torch.save(net.state_dict(), "priority_predictor.ckpt")

if __name__ == "__main__":
    transformer = load_trained_transformer(
        ckpt_dir="checkpoints/Meissonic",
        device="cuda"
    )
    feats = collect_features(transformer, build_dataloader())
    create_dataset(feats)
    # train_priority()