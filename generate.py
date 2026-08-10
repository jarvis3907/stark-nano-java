"""
Stark-Nano-Java — sample Java code from a trained checkpoint.

    python generate.py --prompt "public class Calculator {"
    python generate.py --ckpt checkpoints/ckpt.pt --max-new-tokens 200 --temperature 0.7
"""
import argparse

import torch

from model import GPT
from tokenizer import JavaBPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="public class ")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available() else "cpu")

    # weights_only=False: our checkpoints embed ModelConfig/TrainConfig dataclass
    # instances (not just tensors) — safe since these are checkpoints we wrote.
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {cfg.name} (iter {ckpt['iter_num']}, val_loss={ckpt['best_val_loss']:.4f})")

    tok = JavaBPETokenizer().load(args.tokenizer)

    ids = tok.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_k=args.top_k)
        text = tok.decode(out[0].tolist())
        print(f"\n===== sample {i + 1}/{args.num_samples} =====")
        print(text)


if __name__ == "__main__":
    main()
