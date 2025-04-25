import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import argparse
from src.priority_predictor import PriorityPredictor
from accelerate import Accelerator
import torch.nn.functional as F
import os

class PriorityDataset(Dataset):
    def __init__(self, data_files):
        self.data = []
        for file in data_files:
            data = torch.load(file)
            self.data.append({
                "logits": data["logits"],
                "masks": data["masks"],
                "timesteps": data["timesteps"],
                "labels": data["labels"]
            })
    
    def __len__(self):
        return sum(len(d["logits"]) for d in self.data)
    
    def __getitem__(self, idx):
        # 动态定位到具体样本
        cum_sum = 0
        for d in self.data:
            if idx < cum_sum + len(d["logits"]):
                local_idx = idx - cum_sum
                return {
                    "logits": d["logits"][local_idx],
                    "mask": d["masks"][local_idx],
                    "timestep": d["timesteps"][local_idx],
                    "label": d["labels"][local_idx]
                }
            cum_sum += len(d["logits"])
        raise IndexError

def collate_fn(batch):
    return {
        "logits": torch.stack([b["logits"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "timestep": torch.stack([b["timestep"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch])
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="priority_predictor")
    args = parser.parse_args()

    accelerator = Accelerator()
    
    # 加载数据集
    data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".pt")]
    dataset = PriorityDataset(data_files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    
    # 初始化模型和优化器
    model = PriorityPredictor()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # 训练循环
    for epoch in range(10):
        for batch in dataloader:
            logits = batch["logits"]  # [batch, seq, vocab]
            mask = batch["mask"]  # [batch, seq]
            timestep = batch["timestep"]  # [batch]
            labels = batch["label"]  # [batch, seq]
            
            # 计算目标：预测正确的token是否被mask
            with torch.no_grad():
                pred_tokens = logits.argmax(dim=-1)  # [batch, seq]
                correct = (pred_tokens == labels).float()  # [batch, seq]
                target = correct * mask  # 只有mask的位置有监督信号
            
            # 前向计算
            scores = model(logits, mask, timestep)  # [batch, seq]
            
            # 损失函数：BCEWithLogitsLoss
            loss = F.binary_cross_entropy_with_logits(scores, target)
            
            # 反向传播
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # 日志记录
            if accelerator.is_main_process:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # 保存模型
        accelerator.save_model(model, args.output_dir)

if __name__ == "__main__":
    main()