"""
Stark-Nano-Java — training loop.

    python train.py                     # uses config.ACTIVE_CONFIG
    python train.py --preset 10M        # override with a named preset
    python train.py --resume            # continue from the last checkpoint
    python train.py --wandb --round 4   # also log to Weights & Biases as
                                         # "<model-name>-round4" (needs
                                         # `pip install wandb` + `wandb login`)

Pipeline (all automatic on first run):
  1. Make sure data/corpus.txt exists (run data/download.py if not).
  2. Train (or load) the BPE tokenizer for the active config's vocab_size.
  3. Tokenize the corpus once into data/train.bin + data/val.bin.
  4. Train, logging loss to the console and to <out_dir>/loss_log.csv,
     periodically evaluating on the held-out split, saving the
     best-val-loss checkpoint, and printing a sample generation. Stops
     early if val_loss hasn't improved for TrainConfig.patience eval
     checks in a row (0 disables this and always runs to max_iters).

Local/RunPod split workflow: prep (tokenizer training + corpus encoding)
is CPU-bound and doesn't need a GPU; training is GPU-bound and doesn't
need data/corpus.txt. Split them across machines explicitly instead of
re-running prep on every box:

    ── Local (CPU) ──
    python train.py --prepare-only --preset 100M
    # prints a summary; exits before any model/training code runs

    ── Upload just the 4 prepared files (not corpus.txt/data/raw/) ──
    scp data/tokenizer.json data/train.bin data/val.bin data/meta.json \\
        runpod:/workspace/stark-nano-java/data/

    ── RunPod (GPU) ──
    python train.py --skip-prepare --preset 100M
    # hard-fails immediately if the 4 files are missing or their
    # vocab_size doesn't match the active preset, instead of silently
    # regenerating from a corpus.txt that may not even be on this box
"""
import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import torch

from config import ModelConfig, TrainConfig, get_config
from model import GPT
from tokenizer import JavaBPETokenizer

PREPARED_FILES = ("tokenizer.json", "train.bin", "val.bin", "meta.json")

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
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("vocab_size") == tok.vocab_size:
            print(f"Reusing existing {train_path} / {val_path}")
            return dtype

    corpus_path = os.path.join(data_dir, "corpus.txt")
    corpus_size_mb = os.path.getsize(corpus_path) / 1e6
    print(f"Reading {corpus_path} ({corpus_size_mb:,.0f} MB) ...")
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print("Tokenizing full corpus (this is the slow part on a large corpus -- "
          "progress bar below tracks it) ...")
    ids = tok.encode(text, show_progress=True)
    ids = np.array(ids, dtype=dtype)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    # Hash of the corpus text this tokenization run was built from -- not
    # enforced anywhere (RunPod won't have corpus.txt to compare against),
    # just provenance: if the local corpus changes later without re-running
    # --prepare-only, this is at least detectable by inspection.
    corpus_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    with open(meta_path, "w") as f:
        json.dump({"vocab_size": tok.vocab_size, "train_tokens": len(train_ids),
                   "val_tokens": len(val_ids), "corpus_hash": corpus_hash}, f, indent=2)

    print(f"train.bin: {len(train_ids):,} tokens, val.bin: {len(val_ids):,} tokens")
    return dtype


def load_prepared_data_strict(cfg: ModelConfig, data_dir: str):
    """Strict counterpart to ensure_tokenizer()+ensure_bins(), for
    --skip-prepare: every prepared-data file must already exist and match
    the active config's vocab_size exactly. Hard-fails with a clear message
    instead of silently regenerating anything -- which may not even be
    possible on a box with no data/corpus.txt (the whole point of
    --skip-prepare is to guarantee this path never needs one)."""
    paths = {name: os.path.join(data_dir, name) for name in PREPARED_FILES}

    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "--skip-prepare: missing prepared data file(s):\n"
            + "\n".join(f"  {p}" for p in missing)
            + f"\n\nRun `python train.py --prepare-only --preset <name>` locally "
              f"(active config is {cfg.name}), then sync "
              f"data/{{{', '.join(PREPARED_FILES)}}} to this box."
        )

    with open(paths["meta.json"]) as f:
        meta = json.load(f)

    meta_vocab = meta.get("vocab_size")
    if meta_vocab != cfg.vocab_size:
        raise SystemExit(
            f"--skip-prepare: meta.json says vocab_size={meta_vocab}, active config "
            f"({cfg.name}) wants vocab_size={cfg.vocab_size} — re-run "
            f"`python train.py --prepare-only --preset <name>` locally with the right "
            f"preset, then re-sync data/{{{', '.join(PREPARED_FILES)}}}."
        )

    tok = JavaBPETokenizer().load(paths["tokenizer.json"])
    if tok.vocab_size != meta_vocab:
        raise SystemExit(
            f"--skip-prepare: tokenizer.json's actual vocab_size ({tok.vocab_size}) "
            f"doesn't match meta.json's recorded vocab_size ({meta_vocab}) — these files "
            f"look like they're from different prepare runs (partial sync?). Re-sync "
            f"data/{{{', '.join(PREPARED_FILES)}}} together as one set."
        )

    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
    train_tokens = meta.get("train_tokens")
    val_tokens = meta.get("val_tokens")
    tt = f"{train_tokens:,}" if isinstance(train_tokens, int) else "?"
    vt = f"{val_tokens:,}" if isinstance(val_tokens, int) else "?"
    print(f"--skip-prepare: validated prepared data (vocab_size={tok.vocab_size}, "
          f"train_tokens={tt}, val_tokens={vt})")
    return tok, dtype


def print_prepare_summary(data_dir: str, cfg: ModelConfig):
    paths = {name: os.path.join(data_dir, name) for name in PREPARED_FILES}
    with open(paths["meta.json"]) as f:
        meta = json.load(f)

    def size_mb(p):
        return os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0

    print("\n" + "=" * 70)
    print("Prepare complete. Upload exactly these 4 files to your GPU box:")
    print("=" * 70)
    for name in PREPARED_FILES:
        print(f"  {paths[name]:<28s} {size_mb(paths[name]):8.2f} MB")
    print(f"\n  vocab_size (actual): {meta.get('vocab_size')}")
    print(f"  train_tokens:        {meta.get('train_tokens', 0):,}")
    print(f"  val_tokens:          {meta.get('val_tokens', 0):,}")
    if "corpus_hash" in meta:
        print(f"  corpus_hash:         {meta['corpus_hash'][:16]}...")
    print("\n  Do NOT upload data/corpus.txt or data/raw/ — only the 4 files above;")
    print("  the GPU box never needs raw text, only the tokenized binaries.")
    print("=" * 70)

    # This is the one place a stale-cache mismatch would otherwise slip by
    # silently: ensure_tokenizer()/ensure_bins() reuse whatever's already in
    # data/ (by design, cheap for the common case), but if those cached
    # files are actually from an unrelated earlier/smaller prepare run, the
    # only visible sign is a quiet note buried above -- easy to miss right
    # before uploading the wrong data to a GPU box. Make it impossible to miss.
    actual_vocab = meta.get("vocab_size")
    if actual_vocab != cfg.vocab_size:
        print(f"\n{'!' * 70}")
        print(f"! WARNING: this looks like STALE cached data, not a fresh prepare run!")
        print(f"!   Active config ({cfg.name}) wants vocab_size={cfg.vocab_size}")
        print(f"!   but the prepared files above have vocab_size={actual_vocab}.")
        print(f"!")
        print(f"!   If your corpus is genuinely too small to reach {cfg.vocab_size} unique")
        print(f"!   merges, this is expected and fine. Otherwise, these are almost")
        print(f"!   certainly leftovers from an earlier/different prepare run --")
        print(f"!   delete data/{{tokenizer.json,tok_cache.pkl,train.bin,val.bin,meta.json}}")
        print(f"!   and re-run --prepare-only before uploading anything.")
        print(f"{'!' * 70}")


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
def sample_generation(model, tok, cfg, device, prompt="public class ", max_new_tokens=200):
    model.eval()
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    model.train()
    return tok.decode(out[0].tolist())


# ----------------------------------------------------------------------------
# Optional quality/observability helpers (only exercised with --wandb)
# ----------------------------------------------------------------------------

_JAVAC_AVAILABLE = None


def _javac_available() -> bool:
    global _JAVAC_AVAILABLE
    if _JAVAC_AVAILABLE is None:
        try:
            subprocess.run(["javac", "--version"], capture_output=True, check=True)
            _JAVAC_AVAILABLE = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            _JAVAC_AVAILABLE = False
            print("  (javac not found on PATH -- skipping compile-quality checks; "
                  "install a JDK, e.g. `apt install default-jdk`, to enable them)")
    return _JAVAC_AVAILABLE


def extract_java_class(text: str):
    """First brace-balanced class block in generated text, or None if there
    isn't one (e.g. generation got cut off mid-class -- common at low
    max_new_tokens, not itself a compile failure worth logging as one)."""
    match = re.search(r"((?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+\w+[^{]*\{)", text)
    if not match:
        return None
    start = match.start()
    depth = 0
    for i, c in enumerate(text[start:]):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:start + i + 1]
    return None  # never balanced -- truncated generation, not a real sample


def check_java_compiles(generated_text: str):
    """True/False if the first class in generated_text compiles standalone,
    None if javac isn't available (caller should skip logging in that case,
    not count it as a failure). Prints *why* on any False -- a bare boolean
    doesn't distinguish "generation got cut off before the class closed"
    (a max_new_tokens artifact, not a code-quality signal) from "javac
    actually rejected complete code" (missing-import framework annotations
    like @RestController/@Entity, or a genuine mistake)."""
    if not _javac_available():
        return None
    java_class = extract_java_class(generated_text)
    if not java_class:
        print("    javac compile check: FAIL (generation cut off before the class closed -- "
              "no balanced closing brace; likely a max_new_tokens artifact, not bad code)")
        return False
    name_match = re.search(r"class\s+(\w+)", java_class)
    if not name_match:
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        java_file = os.path.join(tmpdir, f"{name_match.group(1)}.java")
        with open(java_file, "w") as f:
            f.write(java_class)
        try:
            result = subprocess.run(["javac", java_file], capture_output=True,
                                     text=True, timeout=10)
        except subprocess.TimeoutExpired:
            print("    javac compile check: FAIL (javac timed out)")
            return False
        if result.returncode != 0:
            first_err = (result.stderr.strip().splitlines() or ["unknown error"])[0]
            print(f"    javac compile check: FAIL ({first_err})")
        return result.returncode == 0


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
    ap.add_argument("--prepare-only", action="store_true",
                     help="train tokenizer + encode corpus into data/{tokenizer.json,train.bin,"
                          "val.bin,meta.json}, print a summary, and exit -- no model init, no "
                          "training, no GPU needed. For CPU-only data prep before syncing to a "
                          "GPU box (see --skip-prepare).")
    ap.add_argument("--skip-prepare", action="store_true",
                     help="skip tokenizer training + corpus encoding entirely and go straight "
                          "to the training loop -- hard-fails if data/{tokenizer.json,train.bin,"
                          "val.bin,meta.json} are missing or their vocab_size doesn't match the "
                          "active preset, instead of silently regenerating them (which may need "
                          "data/corpus.txt -- not expected to exist on this box). Pair with "
                          "--prepare-only run elsewhere.")
    ap.add_argument("--compile", action="store_true",
                     help="override TrainConfig.compile=True -- wrap the model in "
                          "torch.compile() for this run, without changing the preset's "
                          "persistent default. Adds warmup/recompilation overhead on the "
                          "first few iters in exchange for (usually) higher steady-state "
                          "GPU/Tensor-core utilization.")
    ap.add_argument("--round", type=int, default=1,
                     help="training round number, used only to name the wandb run "
                          "(e.g. --round 4 -> '<model-name>-round4')")
    ap.add_argument("--wandb", action="store_true",
                     help="log loss/lr/sample-generations/compile-rate to Weights & Biases "
                          "(requires `pip install wandb` and `wandb login`); GPU/system stats "
                          "are already captured automatically by wandb itself, no code needed")
    ap.add_argument("--eval-interval", type=int, default=None,
                     help="override TrainConfig.eval_interval -- useful for a short --resume "
                          "smoke test, since eval/checkpoint/wandb-loss-log only fire on "
                          "absolute-iter boundaries and a short run may not hit any at the "
                          "preset's default interval")
    ap.add_argument("--sample-interval", type=int, default=None,
                     help="override TrainConfig.sample_interval, same reasoning as "
                          "--eval-interval")
    ap.add_argument("--sample-max-new-tokens", type=int, default=None,
                     help="override TrainConfig.sample_max_new_tokens -- how many tokens the "
                          "periodic in-loop sample generates. Too short and it won't finish a "
                          "class before running out (check_java_compiles() then sees only "
                          "truncated code, not a real quality signal)")
    args = ap.parse_args()

    if args.prepare_only and args.skip_prepare:
        ap.error("--prepare-only and --skip-prepare are mutually exclusive")
    if args.skip_prepare and args.retrain_tokenizer:
        ap.error("--skip-prepare and --retrain-tokenizer are mutually exclusive "
                  "(--skip-prepare never touches the tokenizer)")

    cfg, tc = get_config(args.preset)
    if args.max_iters is not None:
        tc.max_iters = args.max_iters
        tc.lr_decay_iters = args.max_iters
    if args.device is not None:
        tc.device = args.device
    if args.eval_interval is not None:
        tc.eval_interval = args.eval_interval
    if args.sample_interval is not None:
        tc.sample_interval = args.sample_interval
    if args.sample_max_new_tokens is not None:
        tc.sample_max_new_tokens = args.sample_max_new_tokens
    if args.compile:
        tc.compile = True

    if args.prepare_only:
        print(f"Preparing data for {cfg.name} (vocab_size={cfg.vocab_size}) — CPU only, no GPU needed ...")
        tok = ensure_tokenizer(cfg, tc.data_dir, retrain=args.retrain_tokenizer)
        ensure_bins(cfg, tok, tc.data_dir)
        print_prepare_summary(tc.data_dir, cfg)
        return

    torch.manual_seed(tc.seed)
    device = pick_device(tc.device)
    dtype_str = pick_dtype(tc.dtype, device)
    print(f"Config: {cfg.name} | device={device} | dtype={dtype_str}")
    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} | VRAM: {gpu.total_memory/1e9:.1f}GB")
        print(f"CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")

    wandb = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            raise SystemExit("--wandb requires the wandb package: pip install wandb")

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    if device == "cuda":
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = _NullCtx()  # no autocast on cpu/mps

    os.makedirs(tc.out_dir, exist_ok=True)

    if args.skip_prepare:
        tok, bin_dtype = load_prepared_data_strict(cfg, tc.data_dir)
    else:
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

    if args.wandb:
        meta_path = os.path.join(tc.data_dir, "meta.json")
        train_tokens = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                train_tokens = json.load(f).get("train_tokens")
        wandb.init(
            project="stark-nano-java",
            name=f"{cfg.name}-round{args.round}",
            config={
                "model": cfg.name,
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
                "approx_params": cfg.approx_params(),
                "max_iters": tc.max_iters,
                "batch_size": tc.batch_size,
                "learning_rate": tc.learning_rate,
                "corpus_tokens": train_tokens,
            },
        )

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

            if args.wandb:
                wandb.log({
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "train/lr": lr,
                    "train/iter": it,
                    "train/tokens_seen": it * tc.batch_size * tc.grad_accum_steps * cfg.block_size,
                })

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
            text = sample_generation(model, tok, cfg, device, max_new_tokens=tc.sample_max_new_tokens)
            print(f"[sample] iter {it}:\n{'-'*60}\n{text}\n{'-'*60}")
            compiles = check_java_compiles(text)
            if compiles:  # False/None already handled inline (reason printed, or skipped silently)
                print("  javac compile check: PASS")
            if args.wandb:
                log = {"samples/java": wandb.Html(f"<pre>{html.escape(text)}</pre>"), "train/iter": it}
                if compiles is not None:
                    log["quality/java_compiles"] = 1.0 if compiles else 0.0
                wandb.log(log)

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
    if args.wandb:
        wandb.finish()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
