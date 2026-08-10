"""
Stark-Nano-Java — training loop.

    python train.py                     # uses config.ACTIVE_CONFIG
    python train.py --preset 10M        # override with a named preset
    python train.py --resume            # continue from the last checkpoint

Pipeline (all automatic on first run):
  1. Make sure data/corpus.txt exists (run data/download.py if not).
  2. Train (or load) the BPE tokenizer for the active config's vocab_size.
  3. Tokenize the corpus once into data/train.bin + data/val.bin.
  4. Train, logging loss to the console and to <out_dir>/loss_log.csv,
     periodically evaluating on the held-out split, saving the
     best-val-loss checkpoint, and printing a sample generation. Stops
     early if val_loss hasn't improved for TrainConfig.patience eval
     checks in a row (0 disables this and always runs to max_iters).
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from config import ModelConfig, TrainConfig, get_config
from model import GPT
from tokenizer import JavaBPETokenizer

# ----------------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------------

def ensure_tokenizer(cfg: ModelConfig, data_dir: str, retrain: bool = False) -> JavaBPETokenizer:
    tok_path = os.path.join(data_dir, "tokenizer.json")
    tok = JavaBPETokenizer()
    if os.path.exists(tok_path) and not retrain:
        tok.load(tok_path)
        # Note: on a small corpus, BPE training can exhaust every unique byte
        # pair before reaching cfg.vocab_size, so tok.vocab_size may undershoot
        # the config's target — that's expected and fine, not a reason to
        # retrain every run (it would just undershoot again). Only a config
        # asking for a *smaller* vocab than what's cached is worth flagging.
        print(f"Loaded tokenizer from {tok_path} ({tok.vocab_size} tokens, "
              f"config wants {cfg.vocab_size})")
        if tok.vocab_size < cfg.vocab_size:
            print("  (undershoot is fine on a small corpus; delete data/tokenizer.json "
                  "or pass --retrain-tokenizer to force a retrain, e.g. after growing the corpus)")
        return tok

    corpus_path = os.path.join(data_dir, "corpus.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"No corpus found at {corpus_path}. Run `python data/download.py` first."
        )
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print(f"Training tokenizer: vocab_size={cfg.vocab_size} on {len(text):,} chars ...")
    tok = JavaBPETokenizer()
    # train_cached: if this exact (corpus, vocab_size) pair was trained before
    # (e.g. switching back to a preset used earlier), reuse data/tok_cache.pkl
    # instead of retraining — useful when tokenizer.json was deleted/retrained
    # for a different preset in between.
    cache_path = os.path.join(data_dir, "tok_cache.pkl")
    tok.train_cached(text, vocab_size=cfg.vocab_size, cache_path=cache_path, force=retrain)
    tok.save(tok_path)
    print(f"Saved tokenizer -> {tok_path}")
    return tok


def ensure_bins(cfg: ModelConfig, tok: JavaBPETokenizer, data_dir: str):
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    meta_path = os.path.join(data_dir, "meta.json")

    # Bucketed by the tokenizer's *actual* vocab size (which, on a small
    # corpus, can undershoot cfg.vocab_size — see ensure_tokenizer).
    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("vocab_size") == tok.vocab_size:
            print(f"Reusing existing {train_path} / {val_path}")
            return dtype

    corpus_path = os.path.join(data_dir, "corpus.txt")
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print("Tokenizing full corpus ...")
    ids = tok.encode(text)
    ids = np.array(ids, dtype=dtype)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    import json
    with open(meta_path, "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "train_tokens": len(train_ids),
                   "val_tokens": len(val_ids)}, f, indent=2)

    print(f"train.bin: {len(train_ids):,} tokens, val.bin: {len(val_ids):,} tokens")
    return dtype


def get_batch(split, data_dir, dtype, block_size, batch_size, device):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=dtype, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ----------------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------------

def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if device == "cuda":
        return "float16"
    return "float32"


def get_lr(it: int, tc: TrainConfig) -> float:
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / tc.warmup_iters
    if it > tc.lr_decay_iters:
        return tc.min_lr
    decay_ratio = (it - tc.warmup_iters) / max(1, tc.lr_decay_iters - tc.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


@torch.no_grad()
def estimate_loss(model, tc: TrainConfig, cfg: ModelConfig, dtype, device, ctx):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            x, y = get_batch(split, tc.data_dir, dtype, cfg.block_size, tc.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_generation(model, tok, cfg, device, prompt="public class ", max_new_tokens=80):
    model.eval()
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    model.train()
    return tok.decode(out[0].tolist())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="1M | 10M | 100M | 1B (default: config.ACTIVE_CONFIG)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--retrain-tokenizer", action="store_true",
                     help="retrain the BPE tokenizer even if data/tokenizer.json exists")
    args = ap.parse_args()

    cfg, tc = get_config(args.preset)
    if args.max_iters is not None:
        tc.max_iters = args.max_iters
        tc.lr_decay_iters = args.max_iters
    if args.device is not None:
        tc.device = args.device

    torch.manual_seed(tc.seed)
    device = pick_device(tc.device)
    dtype_str = pick_dtype(tc.dtype, device)
    print(f"Config: {cfg.name} | device={device} | dtype={dtype_str}")

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    if device == "cuda":
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = _NullCtx()  # no autocast on cpu/mps

    os.makedirs(tc.out_dir, exist_ok=True)

    tok = ensure_tokenizer(cfg, tc.data_dir, retrain=args.retrain_tokenizer)
    bin_dtype = ensure_bins(cfg, tok, tc.data_dir)

    model = GPT(cfg).to(device)
    print(f"Model: {model.get_num_params()/1e6:.3f}M non-embedding params "
          f"({model.get_num_params(non_embedding=False)/1e6:.3f}M total)")
    if tc.compile:
        model = torch.compile(model)

    optimizer = model.configure_optimizers(tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2), device)

    ckpt_path = os.path.join(tc.out_dir, "ckpt.pt")
    start_iter = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(ckpt_path):
        # weights_only=False: checkpoint embeds ModelConfig/TrainConfig dataclasses.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter_num"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from {ckpt_path} at iter {start_iter} (best_val_loss={best_val_loss:.4f})")

    log_path = os.path.join(tc.out_dir, "loss_log.csv")
    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s"])

    print(f"Training for {tc.max_iters} iters (batch_size={tc.batch_size}, "
          f"grad_accum={tc.grad_accum_steps}, block_size={cfg.block_size}) ...")

    t0 = time.time()
    running_loss = None
    evals_since_improvement = 0
    x, y = get_batch("train", tc.data_dir, bin_dtype, cfg.block_size, tc.batch_size, device)

    for it in range(start_iter, tc.max_iters):
        lr = get_lr(it, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if tc.eval_interval and it % tc.eval_interval == 0:
            losses = estimate_loss(model, tc, cfg, bin_dtype, device, ctx)
            elapsed = time.time() - t0
            print(f"[eval] iter {it}: train_loss={losses['train']:.4f} "
                  f"val_loss={losses['val']:.4f} lr={lr:.2e} elapsed={elapsed:.0f}s")
            log_writer.writerow([it, losses["train"], losses["val"], lr, f"{elapsed:.1f}"])
            log_file.flush()

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                evals_since_improvement = 0
                torch.save({
                    "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": cfg,
                    "train_config": tc,
                    "iter_num": it,
                    "best_val_loss": best_val_loss,
                }, ckpt_path)
                print(f"  saved checkpoint -> {ckpt_path} (val_loss={best_val_loss:.4f})")
            else:
                evals_since_improvement += 1
                if tc.patience and evals_since_improvement >= tc.patience:
                    print(f"Early stopping: val_loss hasn't improved in "
                          f"{evals_since_improvement} eval checks "
                          f"({evals_since_improvement * tc.eval_interval} iters), "
                          f"best={best_val_loss:.4f}")
                    break

        if tc.sample_interval and it > 0 and it % tc.sample_interval == 0:
            text = sample_generation(model, tok, cfg, device)
            print(f"[sample] iter {it}:\n{'-'*60}\n{text}\n{'-'*60}")

        optimizer.zero_grad(set_to_none=True)
        for micro in range(tc.grad_accum_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / tc.grad_accum_steps
            x, y = get_batch("train", tc.data_dir, bin_dtype, cfg.block_size, tc.batch_size, device)
            loss.backward()

        if tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

        loss_val = loss.item() * tc.grad_accum_steps
        running_loss = loss_val if running_loss is None else 0.98 * running_loss + 0.02 * loss_val

        if it % tc.log_interval == 0:
            print(f"iter {it}: loss={loss_val:.4f} (running={running_loss:.4f}) lr={lr:.2e}")

    log_file.close()
    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoint: {ckpt_path}")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
