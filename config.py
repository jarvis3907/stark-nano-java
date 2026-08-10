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

@dataclass
class ModelConfig:
    name: str
    vocab_size: int = 4096
    block_size: int = 128          # max context length, in tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False             # bias terms in Linear/LayerNorm layers
    tie_weights: bool = True       # tie input embedding & output head weights

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    def approx_params(self) -> int:
        """Rough total parameter count, for sanity-checking a config.

        Uses the standard transformer approximation: each block costs
        ~12 * n_embd^2 (4 for attention qkv+proj, 8 for the 4x MLP), plus
        token/position embeddings (and the output head, unless tied).
        """
        v, b, l, e = self.vocab_size, self.block_size, self.n_layer, self.n_embd
        embedding = v * e + b * e
        per_layer = 12 * e * e + 13 * e  # + biases/LayerNorm params (approx)
        body = l * per_layer
        head = 0 if self.tie_weights else v * e
        final_ln = e
        return embedding + body + head + final_ln


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
    vocab_size=4096,
    block_size=256,
    n_layer=6,
    n_head=8,
    n_embd=352,
    dropout=0.1,
)

CONFIG_100M = ModelConfig(
    name="stark-nano-java-100M",
    vocab_size=8192,
    block_size=512,
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
    batch_size=16, grad_accum_steps=4, learning_rate=3e-4, min_lr=3e-5,
    max_iters=20000, warmup_iters=500, lr_decay_iters=20000,
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
