from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from src.priority_predictor import PriorityPredictor
from src.load_transformer import load_trained_transformer
from diffusers import VQModel
from src.scheduler_with_priority import Scheduler
from src.pipeline import Pipeline

MASK_ID = 8191  # Adjust based on your codebook_size-1
VOCAB = 8192
BATCH = 8
TIMESTEPS = 1000  # Match Meissonic's scheduler convention
DEVICE = "cuda"

def collate(batch):
    # Assuming batch contains 'image' keys as in Meissonic; adapt if different
    images = torch.stack([b["image"] for b in batch])  # (B, C, H, W)
    return {"images": images}

def build_dataloader():
    # Load dataset similar to Meissonic; replace with your dataset
    ds = load_dataset("your_dataset", split="train")
    return DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate)

def collect_features(transformer, vq_model, dataloader):
    features = []
    transformer.eval()
    vq_model.eval()
    
    for batch in tqdm(dataloader, desc="Collecting features"):
        images = batch["images"].to(DEVICE)  # (B, C, H, W)
        B = images.shape[0]
        
        # Encode images to tokens using VQ-VAE as in Meissonic
        with torch.no_grad():
            latents = vq_model.encode(images).latents
            _, _, [_, _, tgt] = vq_model.quantize(latents)
            tgt = tgt.reshape(B, -1)  # (B, L), ground-truth token ids
        
        L = tgt.shape[1]
        
        # Generate multiple masking rates
        for m in torch.linspace(0.1, 0.9, steps=10):  # Avoid m=0 to ensure some masking
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
        
        # Label: 1 if prediction is correct for masked positions, 0 otherwise
        predicted_token = logits.argmax(dim=-1)  # (B, L)
        label = (predicted_token == gt).float() * mask.float()  # (B, L)
        
        return {
            "logits": logits,
            "mask": mask,
            "label": label,
            "timestep": item["timestep"]
        }

def train_priority(transformer, vq_model, scheduler):
    net = PriorityPredictor(VOCAB).to(DEVICE)
    optim = AdamW(net.parameters(), lr=1e-4)
    ds = PriorityDataset("priority_feats.pt")
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    
    # Initialize pipeline with scheduler containing priority predictor
    pipeline = Pipeline(
        transformer=transformer,
        vqvae=vq_model,
        scheduler=scheduler,
        tokenizer=None,  # Not used here; adjust if needed
        text_encoder=None
    )
    
    for epoch in range(3):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            logits = batch["logits"].squeeze(0).to(DEVICE)  # (B, L, V)
            mask = batch["mask"].squeeze(0).to(DEVICE)      # (B, L)
            label = batch["label"].squeeze(0).to(DEVICE)    # (B, L)
            t_step = batch["timestep"][0, 0].item()         # Scalar
            
            # Simulate inference using pipeline with current priority predictor
            B, L = mask.shape
            x_t = torch.where(mask, MASK_ID, batch["gt"].squeeze(0).to(DEVICE))  # (B, L)
            
            with torch.no_grad():
                # Assume pipeline accepts token ids and timestep; adjust as per actual API
                pipeline.scheduler.set_priority_predictor(net)
                inferred_logits = pipeline(
                    prompt=None,  # No text prompt needed here
                    height=int((L ** 0.5) * 8),  # Assuming square latent space
                    width=int((L ** 0.5) * 8),
                    num_inference_steps=int(t_step / 1000 * 64),  # Scale steps
                    initial_tokens=x_t.unsqueeze(0)
                ).logits  # Hypothetical output; adjust based on actual pipeline
            
            # Use pipeline logits to train priority predictor
            pred = net(inferred_logits, mask.float(), t_step)  # (B, L)
            loss = F.binary_cross_entropy_with_logits(pred, label)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            pbar.set_description(f"loss {loss.item():.4f}")
    
    torch.save(net.state_dict(), "priority_predictor.ckpt")

if __name__ == "__main__":
    # Load pretrained models as in Meissonic
    transformer = load_trained_transformer(ckpt_dir="checkpoints/Meissonic", device=DEVICE)
    vq_model = VQModel.from_pretrained("path/to/meissonic", subfolder="vqvae").to(DEVICE)
    scheduler = Scheduler.from_pretrained("path/to/meissonic", subfolder="scheduler")
    
    dataloader = build_dataloader()
    feats = collect_features(transformer, vq_model, dataloader)
    torch.save(feats, "priority_feats.pt")
    # train_priority(transformer, vq_model, scheduler)