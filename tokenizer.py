"""
Stark-Nano-Java — Java-aware byte-pair-encoding tokenizer.

This is a from-scratch BPE tokenizer (same family of algorithm as GPT-2's),
with one Java-specific twist: before any BPE merging happens, source text is
split into syntactic chunks using a Java-aware regex — keywords/identifiers
(with camelCase/PascalCase boundaries split out), numeric literals, string
and char literals, comments, multi-character operators, and punctuation are
each their own chunk. BPE merges are then learned *within* chunks only, never
across them. Two benefits for a Java model:

  1. Merges generalize across identifiers: `getUserName`, `getOrderId`, and
     `getUserId` all start by sharing the "get" chunk, so the tokenizer
     doesn't need to relearn common naming prefixes/suffixes from scratch
     for every new identifier it sees.
  2. Syntax stays crisp: `{`, `}`, `;`, `->` etc. are never accidentally
     glued to neighboring identifiers, and string/char literals never get
     split mid-literal.

Encoding still falls back to raw UTF-8 bytes for anything the regex chunker
doesn't special-case, so the tokenizer can never fail to encode a string.
"""
import argparse
import json
import os
import re
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------
# Java-aware pre-tokenizer
# ----------------------------------------------------------------------------

JAVA_SPLIT_PATTERN = re.compile(r"""
    //[^\n]*                                            # line comment
  | /\*.*?\*/                                            # block comment
  | \"(?:\\.|[^\"\\])*\"                                 # string literal
  | '(?:\\.|[^'\\])*'                                    # char literal
  | \b0[xX][0-9a-fA-F_]+[lL]?\b                           # hex literal
  | \b\d[\d_]*\.?[\d_]*(?:[eE][+-]?\d+)?[fFdDlL]?\b       # numeric literal
  | [\$_]?[A-Z]+(?![a-z])                                 # ALL_CAPS / acronym run
  | [\$_]?[A-Z][a-z0-9]*                                  # PascalCase word
  | [\$_]?[a-z][a-z0-9]*                                  # camelCase word / lowercase word
  | ==|!=|<=|>=|&&|\|\||\+\+|--|->|::|<<=|>>>=|>>=|<<|>>>|>>|[+\-*/%&|^!~<>=]=?
  | [{}()\[\];,.:@?]                                      # punctuation
  | [ \t]+
  | \n+
  | .                                                     # anything else, 1 char at a time
""", re.VERBOSE | re.DOTALL)


def java_chunks(text: str) -> List[str]:
    return JAVA_SPLIT_PATTERN.findall(text)


# ----------------------------------------------------------------------------
# Byte-pair encoding
# ----------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _count_pairs(ids: List[int], stats: Dict[Tuple[int, int], int]) -> None:
    for a, b in zip(ids, ids[1:]):
        stats[(a, b)] = stats.get((a, b), 0) + 1


def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class JavaBPETokenizer:
    """Byte-level BPE tokenizer with Java-aware pre-tokenization.

    Base vocabulary is the 256 raw byte values (ids 0-255); `train()` learns
    additional merged tokens on top, up to `vocab_size` minus the special
    tokens appended at the end.
    """

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}

    # -- training -------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        n_special = len(SPECIAL_TOKENS)
        if vocab_size < 256 + n_special:
            raise ValueError(f"vocab_size must be >= {256 + n_special}")
        num_merges = vocab_size - 256 - n_special

        chunks = [c for c in java_chunks(text) if c]
        ids_list = [list(c.encode("utf-8")) for c in chunks]

        merges: Dict[Tuple[int, int], int] = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats: Dict[Tuple[int, int], int] = {}
            for ids in ids_list:
                _count_pairs(ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids_list = [_merge(ids, pair, new_id) for ids in ids_list]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}, count={stats[pair]})")

        self.merges = merges
        self.vocab = vocab
        self.special_tokens = {
            tok: 256 + len(merges) + i for i, tok in enumerate(SPECIAL_TOKENS)
        }

    # -- encode / decode --------------------------------------------------

    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats: Dict[Tuple[int, int], int] = {}
            _count_pairs(ids, stats)
            # Apply the merge that was learned earliest (lowest target id).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.special_tokens["<bos>"])
        for chunk in java_chunks(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        if add_eos:
            ids.append(self.special_tokens["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        inv_special = {v: k for k, v in self.special_tokens.items()}
        parts: List[bytes] = []
        for i in ids:
            if i in inv_special:
                continue  # control tokens don't contribute text
            parts.append(self.vocab.get(i, b"?"))
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence -------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def save(self, path: str) -> None:
        data = {
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> "JavaBPETokenizer":
        with open(path) as f:
            data = json.load(f)
        self.merges = {tuple(pair): new_id for pair, new_id in data["merges"]}
        self.special_tokens = data["special_tokens"]
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[new_id] = vocab[a] + vocab[b]
        self.vocab = vocab
        return self

    @classmethod
    def from_file(cls, path: str) -> "JavaBPETokenizer":
        return cls().load(path)


# ----------------------------------------------------------------------------
# CLI: train a tokenizer from a text corpus
# ----------------------------------------------------------------------------

def _main():
    ap = argparse.ArgumentParser(description="Train the Stark-Nano-Java tokenizer.")
    ap.add_argument("--input", default="data/corpus.txt", help="training corpus (plain text)")
    ap.add_argument("--output", default="data/tokenizer.json", help="where to save the tokenizer")
    ap.add_argument("--vocab-size", type=int, default=None,
                     help="defaults to the active config's vocab_size")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    vocab_size = args.vocab_size
    if vocab_size is None:
        from config import get_config
        cfg, _ = get_config()
        vocab_size = cfg.vocab_size

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print(f"Training BPE tokenizer: vocab_size={vocab_size}, corpus={len(text):,} chars")
    tok = JavaBPETokenizer()
    tok.train(text, vocab_size=vocab_size, verbose=args.verbose)
    tok.save(args.output)
    print(f"Saved tokenizer ({tok.vocab_size} tokens) -> {args.output}")

    sample = text[:200]
    ids = tok.encode(sample)
    print(f"\nRound-trip check on first 200 chars:")
    print(f"  {len(sample)} chars -> {len(ids)} tokens "
          f"(compression ratio {len(sample)/max(len(ids),1):.2f}x)")
    assert tok.decode(ids) == sample, "round-trip mismatch!"
    print("  OK: decode(encode(text)) == text")


if __name__ == "__main__":
    _main()
