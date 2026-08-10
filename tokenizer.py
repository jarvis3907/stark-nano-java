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

Training is cached: `train_cached()` (used by the CLI and by train.py) hashes
the corpus text + vocab_size and, on a match, loads the previously-trained
tokenizer from `data/tok_cache.pkl` instead of retraining from scratch.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import time
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

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

# BPE merges don't need to see an entire multi-hundred-MB corpus to be
# representative — cap tokenizer *training* text by default (the tokenizer
# is then applied to the full corpus for actual encoding). 40M chars is
# comfortably enough Java source for a stable vocab at any of this project's
# preset sizes. Pass max_train_chars=0 to train() / train_cached() to disable.
DEFAULT_MAX_TRAIN_CHARS = 40_000_000


def _sample_text(text: str, max_chars: int, n_windows: int = 20, seed: int = 1337) -> str:
    """Deterministically sample ~max_chars from `text` as `n_windows` random
    contiguous slices spread across it (more representative than a single
    prefix slice, without the cost of shuffling the whole corpus)."""
    if len(text) <= max_chars:
        return text
    import random
    rng = random.Random(seed)
    window = max(1, max_chars // n_windows)
    return "".join(
        text[start:start + window]
        for start in (rng.randint(0, len(text) - window) for _ in range(n_windows))
    )


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


class _FallbackProgress:
    """Minimal progress reporter used when tqdm isn't installed. Shows merge
    number/total, percent complete, ETA, and a `vocab=` postfix — same info
    tqdm would show, just printed on a refreshed line instead of a bar."""

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.start = time.time()
        self.n = 0

    def update(self, vocab_size: int):
        self.n += 1
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.n) / rate if rate > 0 else 0
        pct = 100 * self.n / self.total if self.total else 100
        print(f"\r{self.desc} {self.n}/{self.total} ({pct:5.1f}%) "
              f"ETA {remaining:5.0f}s vocab={vocab_size}", end="", flush=True)

    def close(self):
        print()  # newline after the last \r-updated line


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

    def train(self, text: str, vocab_size: int, verbose: bool = False,
              show_progress: bool = True, max_train_chars: int = DEFAULT_MAX_TRAIN_CHARS) -> None:
        """Train BPE merges on `text`.

        Two optimizations over textbook/naive BPE (recount every pair, over
        every chunk, on every merge — O(num_merges * corpus_size), which for
        vocab_size=8192 (~7900 merges) over a 300MB corpus means hundreds of
        billions of operations in pure Python, i.e. effectively never):

          1. Deduplicate identical chunks and weight by frequency. Source
             code repeats the same tokens constantly (`public`, `;`, common
             identifiers) — a chunk like "public" might occur 500,000 times
             as 500,000 separate list objects. Processing each occurrence
             separately means a merge touching "public" does 500,000x the
             work of a merge touching some rare identifier that appears
             once, even though it's structurally the same computation
             repeated. Deduplicating turns a corpus with millions of raw
             occurrences into typically a few hundred thousand *unique*
             chunk shapes, each processed once and weighted by its count —
             this is the dominant cost and by far the bigger win.
          2. An incremental pair-count index (pair -> count, pair -> {unique
             chunk indices containing it}) so each merge only revisits
             unique chunks that actually contained the merged pair, not
             every unique chunk in the corpus.

        `max_train_chars`: BPE merges don't need to see the entire corpus to
        be representative — pass None/0 to disable and use the full text.
        """
        n_special = len(SPECIAL_TOKENS)
        if vocab_size < 256 + n_special:
            raise ValueError(f"vocab_size must be >= {256 + n_special}")
        num_merges = vocab_size - 256 - n_special

        if max_train_chars and len(text) > max_train_chars:
            print(f"Corpus is {len(text):,} chars; sampling {max_train_chars:,} chars "
                  f"(20 windows spread across the corpus) to train the tokenizer — BPE "
                  f"merges generalize fine from a representative sample, and this keeps "
                  f"training tractable. Pass max_train_chars=0 to force the full corpus.")
            text = _sample_text(text, max_train_chars)

        t0 = time.time()
        chunks = [c for c in java_chunks(text) if c]

        chunk_counts: Dict[str, int] = {}
        for c in chunks:
            chunk_counts[c] = chunk_counts.get(c, 0) + 1
        unique_chunks = list(chunk_counts.keys())
        mult = [chunk_counts[c] for c in unique_chunks]  # occurrence count per unique chunk
        ids_list = [list(c.encode("utf-8")) for c in unique_chunks]

        total_tokens = sum(len(ids) * m for ids, m in zip(ids_list, mult))
        print(f"Pre-tokenized {len(text):,} chars -> {len(chunks):,} chunks, "
              f"{len(unique_chunks):,} unique ({total_tokens:,} byte-tokens) "
              f"in {time.time() - t0:.1f}s")

        merges: Dict[Tuple[int, int], int] = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        train_start = time.time()

        # Global pair -> weighted count, and pair -> {unique-chunk indices
        # containing it}. The latter is what lets each merge skip untouched
        # (unique) chunks.
        pair_counts: Dict[Tuple[int, int], int] = {}
        where: Dict[Tuple[int, int], set] = {}
        for idx, ids in enumerate(ids_list):
            local: Dict[Tuple[int, int], int] = {}
            _count_pairs(ids, local)
            m = mult[idx]
            for p, c in local.items():
                pair_counts[p] = pair_counts.get(p, 0) + c * m
                where.setdefault(p, set()).add(idx)

        # Progress bar shows: merge number/total, % complete, ETA (tqdm's
        # bar format includes all three natively) and current vocab size
        # (via postfix). Falls back to a plain-text reporter if tqdm isn't
        # installed, showing the same four numbers.
        if show_progress and _HAVE_TQDM:
            bar = tqdm(total=num_merges, desc="Training BPE", unit="merge")
        elif show_progress:
            bar = _FallbackProgress(num_merges, desc="Training BPE")
        else:
            bar = None

        for i in range(num_merges):
            if not pair_counts:
                if verbose:
                    print(f"\nRan out of unique pairs after {i} merges "
                          f"(corpus too small for vocab_size={vocab_size}); stopping early.")
                break
            pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i
            affected = where.pop(pair, set())

            for idx in affected:
                ids = ids_list[idx]
                m = mult[idx]

                old_local: Dict[Tuple[int, int], int] = {}
                _count_pairs(ids, old_local)
                for p, c in old_local.items():
                    pair_counts[p] = pair_counts.get(p, 0) - c * m
                    if pair_counts[p] <= 0:
                        del pair_counts[p]

                new_ids = _merge(ids, pair, new_id)
                ids_list[idx] = new_ids

                new_local: Dict[Tuple[int, int], int] = {}
                _count_pairs(new_ids, new_local)
                for p, c in new_local.items():
                    pair_counts[p] = pair_counts.get(p, 0) + c * m
                    where.setdefault(p, set()).add(idx)

                for p in old_local:
                    if p not in new_local:
                        s = where.get(p)
                        if s is not None:
                            s.discard(idx)
                            if not s:
                                del where[p]

            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            current_vocab_size = 256 + len(merges) + n_special

            if bar is not None:
                if _HAVE_TQDM:
                    bar.set_postfix(vocab=current_vocab_size)
                    bar.update(1)
                else:
                    bar.update(current_vocab_size)
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}, chunks_touched={len(affected)})")

        if bar is not None:
            bar.close()
            mins, secs = divmod(time.time() - train_start, 60)
            print(f"✅ Done! {len(merges)} merges in {int(mins)}m {secs:04.1f}s "
                  f"(vocab_size={256 + len(merges) + n_special})")

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

    # -- training with caching --------------------------------------------

    def train_cached(self, text: str, vocab_size: int,
                      cache_path: str = "data/tok_cache.pkl",
                      verbose: bool = False, force: bool = False,
                      max_train_chars: int = DEFAULT_MAX_TRAIN_CHARS) -> "JavaBPETokenizer":
        """Like train(), but skips training entirely if a cache at
        `cache_path` was built from this exact (corpus text, vocab_size)
        pair, and writes one after training otherwise.

        The cache is keyed by a hash of the corpus text + vocab_size, not
        just cache_path's existence, so a changed corpus or vocab_size is
        detected and correctly triggers a retrain instead of silently
        reusing a stale tokenizer.
        """
        fingerprint = hashlib.sha1(text.encode("utf-8")).hexdigest() + f":{vocab_size}"

        if not force and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("fingerprint") == fingerprint:
                    self.merges = cached["merges"]
                    self.vocab = cached["vocab"]
                    self.special_tokens = cached["special_tokens"]
                    print(f"✅ Loaded cached tokenizer from {cache_path} "
                          f"({self.vocab_size} tokens) — skipping training")
                    return self
                print(f"Cache at {cache_path} doesn't match this corpus/vocab_size "
                      f"— retraining.")
            except (pickle.UnpicklingError, EOFError, KeyError, AttributeError):
                print(f"Cache at {cache_path} is unreadable — retraining.")

        self.train(text, vocab_size=vocab_size, verbose=verbose, max_train_chars=max_train_chars)

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "fingerprint": fingerprint,
                "merges": self.merges,
                "vocab": self.vocab,
                "special_tokens": self.special_tokens,
            }, f)
        print(f"✅ Tokenizer cached -> {cache_path}")
        return self

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
    ap.add_argument("--cache", default="data/tok_cache.pkl",
                     help="pickle cache path; reused on a matching corpus+vocab_size")
    ap.add_argument("--no-cache", action="store_true", help="ignore/overwrite the cache")
    ap.add_argument("--max-train-chars", type=int, default=DEFAULT_MAX_TRAIN_CHARS,
                     help="cap on corpus chars used for training (0 = use full corpus); "
                          f"default {DEFAULT_MAX_TRAIN_CHARS:,}")
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
    tok.train_cached(text, vocab_size=vocab_size, cache_path=args.cache,
                      verbose=args.verbose, force=args.no_cache,
                      max_train_chars=args.max_train_chars)
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
