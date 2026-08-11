"""
Stark-Nano-Java — model architecture.

A LLaMA-style decoder-only transformer: token embeddings + RoPE (rotary
position embeddings, applied inside attention rather than added at the
input), pre-norm transformer blocks (causal self-attention + SwiGLU MLP
with residual connections), a final RMSNorm, and a linear head tied to the
token embedding. Every dimension is read from a config.ModelConfig —
resize the model entirely from config.py, no edits needed here.

Round 3 architecture change from the original GPT-2-style model (learned
position embeddings, LayerNorm, GELU MLP): RoPE replaces the learned
position-embedding table, RMSNorm replaces LayerNorm, and a SwiGLU MLP
replaces the GELU MLP. This is a breaking change — checkpoints trained
under the pre-Round-3 architecture are not loadable here.
"""
import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelConfig, swiglu_hidden_dim


# ----------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ----------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the cos/sin rotation tables RoPE applies to q/k.
    head_dim must be even. Returns two (max_seq_len, head_dim/2) tensors."""
    assert head_dim % 2 == 0, f"RoPE requires an even head_dim, got {head_dim}"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(q, k, freqs_cos, freqs_sin):
    """Rotate q and k by their position's RoPE angle.
    q, k: (batch, n_head, seq_len, head_dim)."""
    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        seq_len = x.shape[-2]
        cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)
    return rotate(q), rotate(k)


# ----------------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Normalizes by root-mean-square rather than LayerNorm's mean+variance;
    no bias term and no mean-centering, by design."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ----------------------------------------------------------------------------
# Attention (RoPE applied to q/k, right after the qkv projection+reshape)
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # RoPE tables: precomputed once per module (not per forward pass),
        # sized to block_size. Non-persistent -- deterministic from
        # head_dim/block_size/theta, so there's no reason to save/load them
        # in checkpoints (keeps checkpoint size unaffected by this change).
        freqs_cos, freqs_sin = precompute_rope_freqs(self.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Use PyTorch's fused/flash attention kernel when available (torch>=2.0).
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            self.register_buffer("bias_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = self.head_dim
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        q, k = apply_rope(q, k, self.freqs_cos, self.freqs_sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
            att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ----------------------------------------------------------------------------
# SwiGLU MLP
# ----------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig, hidden_mult: float = 8 / 3):
        super().__init__()
        # hidden_mult=8/3 (not config-driven, matches config.swiglu_hidden_dim's
        # default so the two never disagree): SwiGLU uses 3 matrices instead
        # of a GELU-MLP's 2, so 3 * (8/3) = 8 keeps MLP params roughly
        # comparable to the old 2-matrix, 4x-wide GELU MLP.
        hidden_dim = swiglu_hidden_dim(cfg.n_embd, hidden_mult)
        self.gate_proj = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden_dim, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=RMSNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Weight tying: input embedding and output projection share weights.
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init for residual-branch output projections -- attention's
        # c_proj and SwiGLU's down_proj (the GELU-MLP's equivalent layer was
        # also named c_proj; SwiGLU's is down_proj, so both names are
        # checked) -- per GPT-2/LLaMA convention, for training stability at depth.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count. `non_embedding` is kept for call-site
        compatibility with the pre-Round-3 architecture (which had a learned
        position-embedding table that could optionally be excluded) -- RoPE
        has no learned position embedding, so there's nothing to subtract
        now; both call styles return the same total."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, t = idx.shape
        assert t <= self.cfg.block_size, (
            f"sequence length {t} exceeds block_size {self.cfg.block_size}"
        )

        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference-time optimization: only project the last position.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with weight decay applied only to 2D+ params (matmul weights),
        not to biases or norm gains."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        extra = {}
        if device_type == "cuda":
            fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
            if fused_ok:
                extra["fused"] = True
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample `max_new_tokens` tokens following `idx`."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Quick self-test: build the active config's model and run a forward pass.
    from config import get_config

    cfg, _ = get_config()
    model = GPT(cfg)
    print(f"{cfg.name}: {model.get_num_params()/1e6:.3f}M params")

    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, y)
    print("logits:", tuple(logits.shape), "loss:", loss.item())
