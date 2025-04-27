# utils/load_transformer.py
import math
import torch
from src.transformer import Transformer2DModel


def _prepare_latent_image_ids(batch, h, w, device, dtype):
    """
    复刻你训练脚本里的辅助函数：生成 img_ids，用来给 Meissonic 填充空间位置信息。
    返回 (h*w, 3) 的 tensor（与训练时保持一致即可）。
    """
    latent = torch.zeros(h // 2, w // 2, 3, dtype=dtype, device=device)
    latent[..., 1] += torch.arange(h // 2, device=device)[:, None]
    latent[..., 2] += torch.arange(w // 2, device=device)[None, :]
    latent = latent.reshape(-1, 3)                # (hw/4, 3)
    return latent


class TokenTransformer(torch.nn.Module):
    """
    把 (B, L) token-id 序列 → (B, L, V) logits
    只保留 **必须** 的输入，内部自动补齐 img_ids / txt_ids / micro_conds。
    """
    def __init__(self, base_model: Transformer2DModel):
        super().__init__()
        self.base = base_model
        self.codebook_size = self.base.config.codebook_size

    @torch.inference_mode()
    def forward(self, token_ids: torch.LongTensor, timestep: int):
        """
        token_ids : (B, L) — L 一定是正方形 (H*W)  
        timestep  : int    — 当前 step，和 Scheduler 里的一致
        """
        B, L = token_ids.shape
        H = W = int(math.sqrt(L))
        assert H * W == L, "sequence length 必须是完全平方数"

        token_2d = token_ids.view(B, H, W)

        # dummy inputs —— teacher forcing 时不需要 text cond / micro cond
        img_ids = _prepare_latent_image_ids(B, 2 * H, 2 * W,
                                            token_ids.device, token_ids.dtype)
        txt_ids = torch.zeros(H * W, 3, dtype=token_ids.dtype,
                              device=token_ids.device)
        micro   = torch.zeros(B, 8, dtype=token_ids.dtype,  # 8 -> 你自己模型中 micro_cond 的维度
                              device=token_ids.device)

        logits = self.base(
            hidden_states=token_2d,
            timestep=torch.full((B,), timestep, device=token_ids.device),
            img_ids=img_ids,
            txt_ids=txt_ids,
            encoder_hidden_states=None,
            micro_conds=micro,
            pooled_projections=None,
        )                                 # (B, V, H, W)

        logits = logits.reshape(B, self.codebook_size, -1) \
                       .permute(0, 2, 1)                  # (B, L, V)
        return logits


def load_trained_transformer(ckpt_dir: str,
                             subfolder: str = "transformer",
                             device: str = "cuda",
                             compile_graph: bool = False):
    """
    ckpt_dir  : 训练完保存的根目录（里面有 transformer/ vqvae/ …）
    subfolder : transformer 权重所在子目录
    device    : "cuda" / "cpu"
    """
    base = Transformer2DModel.from_pretrained(
        ckpt_dir, subfolder=subfolder
    ).to(device)

    base.eval().requires_grad_(False)

    if compile_graph:
        base = torch.compile(base)        # PyTorch 2.0+ Compile，可选

    # 返回包装后的 TokenTransformer
    return TokenTransformer(base).to(device)