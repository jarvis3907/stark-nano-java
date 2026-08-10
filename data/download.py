"""
Stark-Nano-Java — Java dataset downloader.

Default mode (no extra dependencies, no auth): downloads the `.tar.gz`
snapshot of a curated list of permissively-licensed, mostly-Java GitHub
repositories directly from GitHub's codeload endpoint, extracts every
`.java` file, and concatenates them into a single training corpus.

    python data/download.py                       # default repo list
    python data/download.py --repos owner/name owner2/name2 --branch main
    python data/download.py --max-files 5000 --min-bytes 200 --max-bytes 200000

Optional mode: stream a slice of a Hugging Face code dataset instead
(requires `pip install datasets`, and for gated datasets, `huggingface-cli
login`):

    python data/download.py --source huggingface --hf-dataset bigcode/the-stack-smol \
        --hf-config data/java --max-files 20000

Output:
    data/raw/<owner>__<repo>/**/*.java   -- extracted source files
    data/corpus.txt                      -- all kept files concatenated
    data/corpus_stats.json               -- file/byte counts for the run
"""
import argparse
import hashlib
import io
import json
import os
import tarfile
import urllib.error
import urllib.request

# A small, deliberately diverse set of permissively-licensed, mostly-Java
# repositories. Good enough to bootstrap the 1M/10M configs; add more (see
# README) for 100M/1B-scale runs.
DEFAULT_REPOS = [
    "TheAlgorithms/Java",          # MIT — algorithms & data structures
    "iluwatar/java-design-patterns",  # MIT/Apache-2.0 — idiomatic design patterns
]

CODELOAD_URL = "https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
DEFAULT_BRANCHES = ["main", "master"]


def _download_tarball(repo: str, branches, timeout: int = 60) -> bytes:
    last_err = None
    for branch in branches:
        url = CODELOAD_URL.format(repo=repo, branch=branch)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stark-nano-java"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not download {repo} on branches {branches}: {last_err}")


def _iter_java_members(tar_bytes: bytes):
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".java"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                yield member.name, f.read()


def _hash_existing_raw_files(raw_dir):
    """Hash every .java file already under data/raw/ so --append can dedupe
    new downloads against what's already in the corpus, without re-reading
    corpus.txt itself."""
    hashes = set()
    if not os.path.isdir(raw_dir):
        return hashes
    for dirpath, _, filenames in os.walk(raw_dir):
        for fn in filenames:
            if not fn.endswith(".java"):
                continue
            with open(os.path.join(dirpath, fn), "rb") as f:
                hashes.add(hashlib.sha1(f.read()).hexdigest())
    return hashes


def download_github(repos, branches, out_dir, min_bytes, max_bytes, max_files,
                     append=False, verbose=True):
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, "corpus.txt")
    stats_path = os.path.join(out_dir, "corpus_stats.json")

    prior_repos, prior_files, prior_bytes = [], 0, 0
    if append:
        seen_hashes = _hash_existing_raw_files(raw_dir)
        if verbose:
            print(f"--append: found {len(seen_hashes)} existing files under {raw_dir}, "
                  f"will skip exact duplicates")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                prior = json.load(f)
            prior_repos = prior.get("repos", [])
            prior_files = prior.get("files", 0)
            prior_bytes = prior.get("bytes", 0)
    else:
        seen_hashes = set()

    kept_files = 0
    kept_bytes = 0
    mode = "a" if append and os.path.exists(corpus_path) else "w"

    with open(corpus_path, mode, encoding="utf-8") as corpus:
        for repo in repos:
            if kept_files >= max_files:
                break
            if verbose:
                print(f"Downloading {repo} ...")
            try:
                tar_bytes = _download_tarball(repo, branches)
            except RuntimeError as e:
                print(f"  SKIP: {e}")
                continue

            repo_dir = os.path.join(raw_dir, repo.replace("/", "__"))
            n_this_repo = 0
            for name, content in _iter_java_members(tar_bytes):
                if kept_files >= max_files:
                    break
                size = len(content)
                if size < min_bytes or size > max_bytes:
                    continue

                digest = hashlib.sha1(content).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                # Strip the tarball's top-level "<repo>-<branch>/" directory.
                rel_path = name.split("/", 1)[1] if "/" in name else name
                dest_path = os.path.join(repo_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(text)

                corpus.write(text)
                corpus.write("\n")

                kept_files += 1
                kept_bytes += size
                n_this_repo += 1

            if verbose:
                print(f"  kept {n_this_repo} files")

    all_repos = prior_repos + [r for r in repos if r not in prior_repos]
    stats = {
        "source": "github",
        "repos": all_repos,
        "files": prior_files + kept_files,
        "bytes": prior_bytes + kept_bytes,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        action = "added" if append else "kept"
        print(f"\nDone: {action} {kept_files} files, {kept_bytes / 1e6:.2f} MB this run "
              f"-> {corpus_path} (total: {stats['files']} files, {stats['bytes'] / 1e6:.2f} MB)")
    return stats


def download_huggingface(dataset, config, split, out_dir, max_files, min_bytes, max_bytes,
                          append=False, verbose=True):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' package is required for --source huggingface.\n"
            "Install it with: pip install datasets"
        ) from e

    os.makedirs(out_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, "corpus.txt")
    stats_path = os.path.join(out_dir, "corpus_stats.json")

    # Note: unlike the github source, individual files aren't kept on disk here,
    # so --append only avoids duplicates *within* this run, not against
    # content pulled by a previous run.
    seen_hashes = set()
    prior_files, prior_bytes = 0, 0
    if append and os.path.exists(stats_path):
        with open(stats_path) as f:
            prior = json.load(f)
        prior_files = prior.get("files", 0)
        prior_bytes = prior.get("bytes", 0)

    kept_files = 0
    kept_bytes = 0
    mode = "a" if append and os.path.exists(corpus_path) else "w"

    if verbose:
        print(f"Streaming {dataset} ({config or 'default config'}, split={split}) ...")
    ds = load_dataset(dataset, config, split=split, streaming=True)

    with open(corpus_path, mode, encoding="utf-8") as corpus:
        for ex in ds:
            if kept_files >= max_files:
                break
            text = ex.get("content") or ex.get("text")
            if not text:
                continue
            size = len(text.encode("utf-8"))
            if size < min_bytes or size > max_bytes:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            corpus.write(text)
            corpus.write("\n")
            kept_files += 1
            kept_bytes += size

            if verbose and kept_files % 1000 == 0:
                print(f"  ... {kept_files} files ({kept_bytes / 1e6:.1f} MB)")

    stats = {"source": "huggingface", "dataset": dataset, "config": config,
              "files": prior_files + kept_files, "bytes": prior_bytes + kept_bytes}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        action = "added" if append else "kept"
        print(f"\nDone: {action} {kept_files} files, {kept_bytes / 1e6:.2f} MB this run "
              f"-> {corpus_path} (total: {stats['files']} files, {stats['bytes'] / 1e6:.2f} MB)")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["github", "huggingface"], default="github")
    ap.add_argument("--out-dir", default=os.path.dirname(__file__) or ".")

    # github source
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS,
                     help="owner/name GitHub repos to pull .java files from")
    ap.add_argument("--branch", nargs="*", default=DEFAULT_BRANCHES,
                     help="branch names to try, in order")

    # huggingface source
    ap.add_argument("--hf-dataset", default="bigcode/the-stack-smol")
    ap.add_argument("--hf-config", default="data/java")
    ap.add_argument("--hf-split", default="train")

    # shared filters
    ap.add_argument("--max-files", type=int, default=20000)
    ap.add_argument("--min-bytes", type=int, default=200, help="skip trivially small files")
    ap.add_argument("--max-bytes", type=int, default=200_000, help="skip huge/generated files")
    ap.add_argument("--append", action="store_true",
                     help="add to the existing data/corpus.txt instead of overwriting it "
                          "(default without this flag is to OVERWRITE corpus.txt)")

    args = ap.parse_args()

    corpus_path = os.path.join(args.out_dir, "corpus.txt")
    if os.path.exists(corpus_path) and not args.append:
        existing_mb = os.path.getsize(corpus_path) / 1e6
        print(f"WARNING: {corpus_path} already exists ({existing_mb:.1f} MB) and will be "
              f"OVERWRITTEN. Re-run with --append to add to it instead. Aborting.")
        return

    if args.source == "github":
        download_github(args.repos, args.branch, args.out_dir,
                         args.min_bytes, args.max_bytes, args.max_files, append=args.append)
    else:
        download_huggingface(args.hf_dataset, args.hf_config, args.hf_split, args.out_dir,
                              args.max_files, args.min_bytes, args.max_bytes, append=args.append)


if __name__ == "__main__":
    main()
