"""
Stark-Nano-Java — qualitative eval: sample a fixed prompt set from a checkpoint
and save the results to evaluation_round<N>.txt for round-over-round comparison.

    python evaluate.py                              # auto-numbers the next round
    python evaluate.py --round 3 --ckpt checkpoints/ckpt.pt
"""
import argparse
import glob
import re
import subprocess
import sys

PROMPTS = [
    ("Basic POJO", "public class User {\n    private Long id;"),
    ("REST Controller", "@RestController\npublic class UserController {"),
    ("JPA Entity", "@Entity\npublic class Product {"),
    ("Spring Service", "@Service\npublic class OrderService {"),
]

# Pinned so results stay comparable across rounds even if generate.py's own
# defaults change later.
GEN_ARGS = ["--max-new-tokens", "200", "--temperature", "0.8", "--top-k", "40"]


def next_round() -> int:
    """Next unused evaluation_round<N>.txt number, so re-running never clobbers
    a prior round by accident."""
    nums = []
    for f in glob.glob("evaluation_round*.txt"):
        m = re.match(r"evaluation_round(\d+)\.txt$", f)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None,
                     help="defaults to the next unused round number")
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    args = ap.parse_args()

    round_num = args.round if args.round is not None else next_round()
    out_path = f"evaluation_round{round_num}.txt"

    chunks = [f"\U0001f9be Round {round_num} Evaluation\n", "=" * 60 + "\n\n"]
    failures = 0

    for name, prompt in PROMPTS:
        chunks.append(f"Test: {name}\n{'-' * 40}\n")
        proc = subprocess.run(
            [sys.executable, "generate.py", "--ckpt", args.ckpt,
             "--prompt", prompt, *GEN_ARGS],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            failures += 1
            err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
            chunks.append(f"[FAILED, exit {proc.returncode}] {err}\n")
            print(f"❌ {name}: {err}")
        else:
            chunks.append(proc.stdout + "\n")
            print(f"✅ {name}")
        chunks.append("=" * 60 + "\n\n")

    with open(out_path, "w") as out:
        out.writelines(chunks)

    status = "with failures" if failures else "clean"
    print(f"✅ Saved to {out_path} ({status})")


if __name__ == "__main__":
    main()
