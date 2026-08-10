# Stark-Nano-Java

A configurable, from-scratch, Java-specialized LLM. Decoder-only transformer
(GPT-style), a Java-aware BPE tokenizer, and a training pipeline that all
scale from a **1M-parameter** model you can train on a laptop CPU in minutes
up to a **1B-parameter** model — by changing one line in [`config.py`](config.py).

## Project structure

```
config.py         Every architectural & training knob. 1M / 10M / 100M / 1B presets.
model.py          The transformer itself (GPT-style decoder-only).
tokenizer.py       From-scratch Java-aware BPE tokenizer.
train.py          Data prep + training loop, with loss tracking & checkpointing.
generate.py        Sample Java code from a trained checkpoint.
data/download.py   Downloads a Java source corpus to train on.
requirements.txt
```

Running any of these produces, under `data/`: `corpus.txt` (raw text),
`tokenizer.json` (trained tokenizer), `train.bin` / `val.bin` (tokenized
dataset), `meta.json`; and under `checkpoints/`: `ckpt.pt` (best checkpoint)
and `loss_log.csv` (loss history).

## Quickstart

```bash
pip install -r requirements.txt

# 1. Get some Java source (default: TheAlgorithms/Java + java-design-patterns)
python data/download.py

# 2. Train the 1M-parameter model (config.py's default) — a few minutes on CPU
python train.py

# 3. Sample some Java from it
python generate.py --prompt "public class Calculator {"
```

`train.py` handles the rest of the pipeline automatically the first time you
run it: it trains a tokenizer sized to the active config's `vocab_size` on
`data/corpus.txt`, tokenizes the corpus into binary shards, then trains.
Re-running it reuses the cached tokenizer/shards instead of redoing that work.

## Scaling up: the one line that matters

Open [`config.py`](config.py) and change:

```python
ACTIVE_CONFIG: ModelConfig = CONFIG_1M
ACTIVE_TRAIN_CONFIG: TrainConfig = TRAIN_1M
```

to `CONFIG_10M` / `TRAIN_10M`, `CONFIG_100M` / `TRAIN_100M`, or `CONFIG_1B` /
`TRAIN_1B`. Or skip editing the file and pass `--preset` on the command line:

```bash
python train.py --preset 10M
python train.py --preset 100M
python train.py --preset 1B
```

| Preset | Params (approx) | vocab | layers | heads | d_model | context |
|--------|-----------------|-------|--------|-------|---------|---------|
| 1M     | ~1.1M            | 2,048 | 4      | 4     | 128     | 128     |
| 10M    | ~10.5M           | 4,096 | 6      | 8     | 352     | 256     |
| 100M   | ~92M             | 8,192 | 12     | 12    | 768     | 512     |
| 1B     | ~957M            | 16,384| 24     | 28    | 1,792   | 1,024   |

Run `python config.py` any time to print this table for the exact presets
currently defined. Every field on `ModelConfig` (vocab size, context length,
layer count, head count, embedding width, dropout, bias, weight tying) and
`TrainConfig` (batch size, gradient accumulation, learning rate schedule,
weight decay, eval cadence, device, dtype, ...) is a plain dataclass field —
tweak any of them, or add your own preset, without touching `model.py` or
`train.py`.

Bigger presets need proportionally more data — see [Data](#data) below for
how to widen the repo list as you scale up.

## Architecture (`model.py`)

Standard decoder-only transformer, same family as GPT-2/nanoGPT:

- Token embedding + learned position embedding
- N pre-norm transformer blocks, each: causal self-attention (uses
  `torch.nn.functional.scaled_dot_product_attention`'s fused/flash kernel
  when available) → residual → MLP (4x expansion, GELU) → residual
- Final LayerNorm → linear head, tied to the token embedding by default
- `model.generate()` for autoregressive sampling with temperature / top-k

Everything is sized from a `ModelConfig` — see `model.py`'s `if __name__ ==
"__main__"` block for a quick param-count + forward-pass smoke test:
`python model.py`.

## Tokenizer (`tokenizer.py`)

A byte-level BPE tokenizer (same algorithm family as GPT-2's), implemented
from scratch, with one Java-specific twist: **before** any BPE merging, source
text is split by a Java-aware regex into syntactic chunks — keywords and
identifiers (with camelCase/PascalCase boundaries split, e.g. `getUserName`
→ `get` `User` `Name`), numeric literals, string/char literals, comments,
multi-char operators (`->`, `==`, `<<=`, ...), and punctuation. BPE merges
are learned *within* chunks only, never across them. This means:

- Common identifier prefixes/suffixes (`get`, `set`, `is`, `Impl`, `Exception`,
  ...) are shared subword tokens across every class that uses them, instead
  of being relearned per-identifier.
- Braces, semicolons, and operators never get glued to neighboring code, and
  string/char literals never split mid-literal.
- Anything the regex doesn't special-case still encodes correctly via raw
  UTF-8 bytes — the tokenizer can never fail to encode a string.

Train one standalone:

```bash
python tokenizer.py --input data/corpus.txt --output data/tokenizer.json --vocab-size 4096
```

Note: on a very small corpus, BPE training can run out of unique byte pairs
before reaching the requested `vocab_size` — the tokenizer will just end up
slightly smaller than requested, which is fine (`train.py` handles this
automatically; use `--retrain-tokenizer` to force a retrain, e.g. after
growing your corpus).

## Data (`data/download.py`)

Default mode needs no extra dependencies or auth: it downloads `.tar.gz`
snapshots of a curated list of permissively-licensed, mostly-Java GitHub
repos straight from GitHub's `codeload` endpoint, extracts every `.java`
file, dedupes by content hash, and concatenates them into `data/corpus.txt`
(raw files also kept under `data/raw/`).

```bash
python data/download.py                                   # default repos, good for 1M/10M
python data/download.py --repos owner/name owner2/name2   # your own repo list
python data/download.py --max-files 5000 --max-bytes 200000
```

As you scale the model up, widen the corpus — e.g. add `google/guava`,
`apache/commons-lang`, `spring-projects/spring-framework`,
`eugenp/tutorials` for 100M/1B-scale runs.

Optional: stream from a Hugging Face code dataset instead (requires
`pip install datasets`; some datasets need `huggingface-cli login`):

```bash
python data/download.py --source huggingface \
    --hf-dataset bigcode/the-stack-smol --hf-config data/java
```

Only `.java` files that decode as UTF-8 and fall within `--min-bytes` /
`--max-bytes` are kept; exact-duplicate file contents are dropped.

## Training (`train.py`)

```bash
python train.py                      # config.ACTIVE_CONFIG (default: 1M)
python train.py --preset 10M
python train.py --max-iters 500      # override iteration count
python train.py --resume             # continue from checkpoints/ckpt.pt
```

What it does, all driven by `TrainConfig`:

- AdamW with weight decay applied only to matrix weights (not biases/LayerNorm)
- Cosine LR schedule with linear warmup
- Gradient accumulation and gradient clipping
- Auto device selection (CUDA → MPS → CPU) and auto mixed-precision dtype
  (bf16 on CUDA when supported, else fp16/fp32)
- Periodic validation-loss evaluation; the best checkpoint (by val loss) is
  saved to `checkpoints/ckpt.pt`
- Early stopping: training halts once val_loss hasn't improved for
  `TrainConfig.patience` eval checks in a row (default 10; set to 0 to
  disable and always run the full `max_iters`) — the saved checkpoint is
  always the best one seen, so this just saves compute, it never costs quality
- Every logged step's train loss, and every eval step's train/val loss + LR,
  is appended to `checkpoints/loss_log.csv` — open it in a spreadsheet or
  plot it (`iter, train_loss, val_loss, lr, elapsed_s`) to track progress
- A qualitative sample generation is printed periodically during training
  (`sample_interval` in `TrainConfig`) so you can watch the model's Java
  output improve, not just the loss number

## Generating (`generate.py`)

```bash
python generate.py --ckpt checkpoints/ckpt.pt --prompt "public class Calculator {" \
    --max-new-tokens 200 --temperature 0.7 --top-k 40 --num-samples 3
```

## Hardware notes

- **1M**: seconds-to-minutes per epoch on any modern CPU. Good for
  iterating on the pipeline itself.
- **10M**: minutes on CPU, faster with MPS/CUDA.
- **100M**: a CUDA GPU is strongly recommended (or patience, on CPU/MPS).
- **1B**: needs a GPU with enough memory for the activations at
  `batch_size × block_size = 4 × 1024`; lower `batch_size` and raise
  `grad_accum_steps` in `TrainConfig` if you run out of memory.

## License note

`data/download.py`'s default repo list (`TheAlgorithms/Java`,
`iluwatar/java-design-patterns`) are MIT/Apache-2.0 licensed. If you add
your own `--repos`, check each repo's license before training on or
redistributing derived output.
