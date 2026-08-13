"""
Stark-Nano-Java — configuration.

Every architectural and training knob used anywhere in this project lives
here. Nothing is hard-coded in model.py / train.py / tokenizer.py — they
all import their numbers from this file.

TO SCALE UP: change the single line marked below (ACTIVE_CONFIG /
ACTIVE_TRAIN_CONFIG) to one of the other presets, or copy a preset and
tweak individual fields.
"""
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Model architecture
# ============================================================================

def swiglu_hidden_dim(n_embd: int, hidden_mult: float = 8 / 3) -> int:
    """SwiGLU's hidden-dim convention: ~hidden_mult * n_embd (default 8/3 --
    SwiGLU uses 3 matrices (gate/up/down) instead of a GELU-MLP's 2, so
    3 * (8/3) = 8 keeps total MLP params roughly comparable to the old
    2-matrix, 4x-wide GELU MLP's 2 * 4 = 8), rounded up to a multiple of 64
    for GPU efficiency. Shared between model.py's SwiGLU module and
    ModelConfig.approx_params() so the param-count estimate and the actual
    implementation can never drift apart.
    """
    hidden_dim = int(hidden_mult * n_embd)
    return ((hidden_dim + 63) // 64) * 64


@dataclass
class ModelConfig:
    name: str
    vocab_size: int = 4096
    block_size: int = 128          # max context length, in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False             # bias terms in Linear layers (RMSNorm has none by design)
    tie_weights: bool = True       # tie input embedding & output head weights
    rope_theta: float = 10000.0    # RoPE base frequency (10000.0 is the standard default)

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError(
                f"head_dim ({self.n_embd // self.n_head}) must be even for RoPE"
            )

    def approx_params(self) -> int:
        """Rough total parameter count, for sanity-checking a config.

        LLaMA-style architecture: RoPE has no learned position-embedding
        table (unlike the old learned wpe), attention is unchanged
        (~4 * n_embd^2 for qkv+proj), the MLP is SwiGLU (3 matrices at
        swiglu_hidden_dim() instead of a GELU-MLP's 2 matrices at 4x), and
        RMSNorm has only a weight vector (no bias, no mean-centering) --
        two per block (pre-attn, pre-mlp) plus one final norm.
        """
        v, l, e = self.vocab_size, self.n_layer, self.n_embd
        embedding = v * e  # token embedding only -- no learned position embedding under RoPE
        attn_params = 4 * e * e
        hidden_dim = swiglu_hidden_dim(e)
        mlp_params = 3 * e * hidden_dim
        norm_params = 2 * e  # RMSNorm weight only, x2 per block
        per_layer = attn_params + mlp_params + norm_params
        body = l * per_layer
        head = 0 if self.tie_weights else v * e
        final_norm = e
        return embedding + body + head + final_norm


# ============================================================================
# Training
# ============================================================================

@dataclass
class TrainConfig:
    out_dir: str = "checkpoints"
    data_dir: str = "data"
    batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    max_iters: int = 3000
    warmup_iters: int = 100
    lr_decay_iters: int = 3000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10
    sample_interval: int = 500     # print a generation sample this often (0 = never)
    sample_max_new_tokens: int = 200  # long enough for a typical class to actually close its
                                       # final brace -- matches generate.py/evaluate.py's own
                                       # default. Too short and check_java_compiles() sees only
                                       # truncated code, not a real quality signal (see 80-token
                                       # default's original samples: none ever closed their class)
    patience: int = 10             # stop after this many eval checks with no val_loss
                                    # improvement (0 = disabled, always run max_iters)
    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "auto"            # "auto" | "float32" | "bfloat16" | "float16"
    seed: int = 1337
    compile: bool = False


# ----------------------------------------------------------------------------
# Presets. approx_params() next to each — run `python config.py` to print a
# summary table for all of them.
# ----------------------------------------------------------------------------

CONFIG_1M = ModelConfig(
    name="stark-nano-java-1M",
    vocab_size=2048,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
)

CONFIG_10M = ModelConfig(
    name="stark-nano-java-10M",
    vocab_size=8192,
    block_size=256,
    n_layer=6,
    n_head=8,
    n_embd=352,
    dropout=0.1,
)

CONFIG_100M = ModelConfig(
    name="stark-nano-java-100M",
    vocab_size=16384,  # Round 4: corpus grew from ~500M Java-only tokens to ~18GB Java+Kotlin
                        # (~81% Kotlin by volume) -- 8192 split across two languages compresses
                        # each worse than 8192 dedicated to Java alone did. Doubled to match
                        # CONFIG_1B's existing tier; adds only ~6.3M tied-embedding params at
                        # n_embd=768.
    block_size=1024,  # Round 3: RunPod GPU has headroom, and Java classes routinely exceed 512 tokens
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
)

CONFIG_1B = ModelConfig(
    name="stark-nano-java-1B",
    vocab_size=16384,
    block_size=1024,
    n_layer=24,
    n_head=28,
    n_embd=1792,
    dropout=0.1,
)

# Training hyperparameters tuned per model size (smaller models can afford a
# much higher learning rate and bigger batch size than larger ones).
TRAIN_1M = TrainConfig(
    batch_size=64, learning_rate=1e-3, min_lr=1e-4,
    max_iters=10000, warmup_iters=100, lr_decay_iters=10000,
    eval_interval=200, eval_iters=40, sample_interval=200,
)
TRAIN_10M = TrainConfig(
    batch_size=48, learning_rate=6e-4, min_lr=6e-5,
    max_iters=5000, warmup_iters=200, lr_decay_iters=5000,
    eval_interval=250, eval_iters=50, sample_interval=500,
)
TRAIN_100M = TrainConfig(
    # batch_size=32/grad_accum_steps=2 keeps the same effective batch (64) as
    # the earlier 64/1 split. Halved from 64/1 for Round 3: CONFIG_100M's
    # block_size doubled (512 -> 1024), which roughly doubles per-sample
    # activation memory, so batch_size needed to come down to stay safely
    # under 32GB VRAM. Raise batch_size and lower grad_accum_steps back up
    # (keeping their product at 64) if you confirm more headroom than this.
    batch_size=32, grad_accum_steps=2, learning_rate=3e-4, min_lr=3e-5,
    max_iters=40000, warmup_iters=500, lr_decay_iters=40000,
    eval_interval=500, eval_iters=100, sample_interval=1000,
)
TRAIN_1B = TrainConfig(
    batch_size=4, grad_accum_steps=32, learning_rate=1.5e-4, min_lr=1.5e-5,
    max_iters=60000, warmup_iters=1000, lr_decay_iters=60000,
    eval_interval=1000, eval_iters=100, sample_interval=2000,
)

_PRESETS = {
    "1M": (CONFIG_1M, TRAIN_1M),
    "10M": (CONFIG_10M, TRAIN_10M),
    "100M": (CONFIG_100M, TRAIN_100M),
    "1B": (CONFIG_1B, TRAIN_1B),
}

# ============================================================================
# >>> SCALE UP HERE: change this one line to switch the model size used  <<<
# >>> by default in train.py / generate.py.                              <<<
# ============================================================================
ACTIVE_CONFIG: ModelConfig = CONFIG_100M
ACTIVE_TRAIN_CONFIG: TrainConfig = TRAIN_100M
# ============================================================================


def get_config(preset: Optional[str] = None):
    """Return (ModelConfig, TrainConfig).

    `preset` may be "1M" / "10M" / "100M" / "1B" (case-insensitive). If
    omitted, returns whatever ACTIVE_CONFIG / ACTIVE_TRAIN_CONFIG currently
    point at above.
    """
    if preset is None:
        return ACTIVE_CONFIG, ACTIVE_TRAIN_CONFIG
    key = preset.upper()
    if key not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {list(_PRESETS)}")
    return _PRESETS[key]


if __name__ == "__main__":
    print(f"{'name':26s} {'params':>10s} {'vocab':>7s} {'layers':>7s} "
          f"{'heads':>6s} {'embd':>6s} {'block':>6s}")
    for name, (mc, tc) in _PRESETS.items():
        print(f"{mc.name:26s} {mc.approx_params()/1e6:9.2f}M {mc.vocab_size:7d} "
              f"{mc.n_layer:7d} {mc.n_head:6d} {mc.n_embd:6d} {mc.block_size:6d}")
