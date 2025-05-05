import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from src.priority_predictor import PriorityPredictor
from src.load_transformer import load_trained_transformer
from diffusers import VQModel
from src.scheduler_with_priority import Scheduler
from src.pipeline import Pipeline
from train.dataset_utils import HuggingFaceDataset, MyParquetDataset  # 假设这些类在 train/dataset_utils.py 中

# 全局变量
MASK_ID = 8191  # Adjust based on your codebook_size-1
VOCAB = 8192
BATCH = 8
TIMESTEPS = 1000  # Match Meissonic's scheduler convention
DEVICE = "cuda"

# Collate function
def collate(batch):
    images = torch.stack([b["image"] for b in batch])  # (B, C, H, W)
    return {"images": images}

# 调整后的 build_dataloader 函数
def build_dataloader(args, tokenizer=None):
    """
    构建数据加载器，根据 args.instance_dataset 选择数据集类型。
    
    Args:
        args: 包含配置参数的对象，例如 instance_dataset, instance_data_dir, train_batch_size 等
        tokenizer: 用于文本处理的 tokenizer（可选，取决于数据集需求）
    
    Returns:
        DataLoader: 配置好的数据加载器
    """
    if args.instance_dataset == "HuggingFaceDataset":
        hf_dataset = load_dataset(args.instance_data_dir, split="train")
        dataset = HuggingFaceDataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            image_key='image',
            prompt_key='caption',
            prompt_prefix=getattr(args, 'prompt_prefix', None),
            size=getattr(args, 'resolution', 256),
            text_encoder_architecture=getattr(args, 'text_encoder_architecture', 'open_clip')
        )
    elif args.instance_dataset == "MyParquetDataset":
        dataset = MyParquetDataset(
            root_dir=args.instance_data_dir,
            tokenizer=tokenizer,
            size=getattr(args, 'resolution', 256),
            text_encoder_architecture=getattr(args, 'text_encoder_architecture', 'open_clip')
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.instance_dataset}")

    dataloader = DataLoader(
        dataset,
        batch_size=getattr(args, 'train_batch_size', BATCH),
        shuffle=True,
        num_workers=getattr(args, 'dataloader_num_workers', 0),
        collate_fn=collate,
        pin_memory=True
    )
    return dataloader

# 特征收集函数
def collect_features(transformer, vq_model, dataloader):
    features = []
    transformer.eval()
    vq_model.eval()
    
    for batch in tqdm(dataloader, desc="Collecting features"):
        images = batch["images"].to(DEVICE)  # (B, C, H, W)
        B = images.shape[0]
        
        with torch.no_grad():
            latents = vq_model.encode(images).latents
            _, _, [_, _, tgt] = vq_model.quantize(latents)
            tgt = tgt.reshape(B, -1)  # (B, L), ground-truth token ids
        
        L = tgt.shape[1]
        
        for m in torch.linspace(0.1, 0.9, steps=10):
            num_masked = int(m * L)
            mask = torch.zeros_like(tgt, dtype=torch.bool)
            for b in range(B):
                indices = torch.randperm(L)[:num_masked]
                mask[b, indices] = True
            
            x_t = torch.where(mask, MASK_ID, tgt)  # (B, L)
            timestep = m * TIMESTEPS  # Scale timestep proportionally
            
            with torch.no_grad():
                logits = transformer(x_t, timestep=timestep)  # (B, L, V)
                if logits.shape[1] == VOCAB:
                    logits = logits.permute(0, 2, 1)  # (B, L, V) -> (B, V, L)
            
            features.append({
                "logits": logits.cpu(),
                "mask": mask.cpu(),
                "timestep": torch.full((B, 1), timestep, dtype=torch.float32),
                "gt": tgt.cpu()
            })
    
    return features

# PriorityDataset 类
class PriorityDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        logits = item["logits"]  # (B, L, V)
        mask = item["mask"]      # (B, L)
        gt = item["gt"]          # (B, L)
        
        predicted_token = logits.argmax(dim=-1)  # (B, L)
        label = (predicted_token == gt).float() * mask.float()  # (B, L)
        
        return {
            "logits": logits,
            "mask": mask,
            "label": label,
            "timestep": item["timestep"]
        }

# 训练 PriorityPredictor 的函数
def train_priority(transformer, vq_model, scheduler):
    net = PriorityPredictor(VOCAB).to(DEVICE)
    optim = AdamW(net.parameters(), lr=1e-4)
    ds = PriorityDataset("priority_feats.pt")
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    
    pipeline = Pipeline(
        transformer=transformer,
        vqvae=vq_model,
        scheduler=scheduler,
        tokenizer=None,
        text_encoder=None
    )
    
    for epoch in range(3):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            logits = batch["logits"].squeeze(0).to(DEVICE)  # (B, L, V)
            mask = batch["mask"].squeeze(0).to(DEVICE)      # (B, L)
            label = batch["label"].squeeze(0).to(DEVICE)    # (B, L)
            t_step = batch["timestep"][0, 0].item()         # Scalar
            
            B, L = mask.shape
            x_t = torch.where(mask, MASK_ID, batch["gt"].squeeze(0).to(DEVICE))  # (B, L)
            
            with torch.no_grad():
                pipeline.scheduler.set_priority_predictor(net)
                inferred_logits = pipeline(
                    prompt=None,
                    height=int((L ** 0.5) * 8),
                    width=int((L ** 0.5) * 8),
                    num_inference_steps=int(t_step / 1000 * 64),
                    initial_tokens=x_t.unsqueeze(0)
                ).logits
            
            pred = net(inferred_logits, mask.float(), t_step)  # (B, L)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            pbar.set_description(f"loss {loss.item():.4f}")
    
    torch.save(net.state_dict(), "priority_predictor.ckpt")

# 主函数
if __name__ == "__main__":
    # 模拟 args 对象（根据你的需求配置）
    class Args:
        instance_dataset = "HuggingFaceDataset"  # 或 "MyParquetDataset"
        instance_data_dir = "../parquets_father_dir/"  # 数据集路径
        train_batch_size = 8
        dataloader_num_workers = 4
        resolution = 256
        text_encoder_architecture = "open_clip"
        prompt_prefix = None  # 可选

    args = Args()

    # 加载 tokenizer（根据你的 text_encoder_architecture 选择）
    from transformers import CLIPTokenizer
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # 示例
    
    model_path = "checkpoints/Meissonic"
    # 加载预训练模型
    transformer = load_trained_transformer(ckpt_dir=model_path, device=DEVICE)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae").to(DEVICE)
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", vocab_size=VOCAB)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    # 构建 dataloader
    dataloader = build_dataloader(args, tokenizer)

    # 收集特征
    feats = collect_features(transformer, vq_model, dataloader)
    torch.save(feats, "priority_feats.pt")

    # 训练 PriorityPredictor
    # train_priority(transformer, vq_model, scheduler)