"""
Stark-Nano-Java — round 2 extraction/merge (companion to round2_curl.sh).

round2_curl.sh downloads each repo's zip individually into
data/round2/zips/ (with live progress, so a stuck/large repo like
openjdk/jdk is visible instead of silent). This script picks up from
there: extracts every .java/.kt file from each zip, keeps files in a
sane size range, and merges them into a single training corpus.

Two things make this fast:
  1. Zips are processed in parallel (one worker process per zip) --
     they're fully independent, and this machine likely has many idle
     cores while a single Python process burns just one.
  2. Before decompressing a member, its *uncompressed* size is checked
     against the size filter using the zip's own index (zero decompression
     cost). This skips full decompress+decode work entirely for files that
     would just get thrown away anyway -- matters a lot for huge repos
     like openjdk/jdk, which has plenty of oversized generated/data files.

Run:
    python data/round2_extract.py

Output:
    data/round2/raw/*.java, *.kt   -- extracted source files (flattened)
    data/round2/corpus_round2.txt  -- all kept files concatenated
"""
import os
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

# Anchored to this script's own location, not the caller's working
# directory -- same reasoning as download_round2.py/round2_curl.sh, so
# this works whether you run it from the repo root or from inside data/.
ROUND2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2")
ZIPS_DIR = os.path.join(ROUND2_DIR, "zips")
RAW_DIR = os.path.join(ROUND2_DIR, "raw")
CORPUS_PATH = os.path.join(ROUND2_DIR, "corpus_round2.txt")

MIN_CHARS = 100
MAX_CHARS = 100_000


def _extract_one_zip(zip_path: str, raw_dir: str):
    """Runs in a worker process: extract kept .java/.kt files from one zip.
    Returns (zip_name, kept_count, error_message_or_None)."""
    zip_name = os.path.basename(zip_path)
    kept = 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                name = info.filename
                if not (name.endswith('.java') or name.endswith('.kt')):
                    continue
                # Pre-filter on uncompressed size (free -- it's just the
                # zip's index) before paying for decompression+decode.
                # file_size (bytes) is always >= decoded char count for
                # UTF-8, so file_size <= MIN_CHARS safely guarantees the
                # char-count check would fail too. The upper bound is a
                # heuristic (a file could rarely have file_size >= MAX_CHARS
                # bytes but < MAX_CHARS decoded chars if it's heavy with
                # multi-byte UTF-8) -- an acceptable approximation for
                # corpus-building at this scale.
                if info.file_size <= MIN_CHARS or info.file_size >= MAX_CHARS:
                    continue
                content = z.read(info).decode('utf-8', errors='ignore')
                if MIN_CHARS < len(content) < MAX_CHARS:
                    safe = name.replace('/', '_')
                    with open(os.path.join(raw_dir, safe), 'w') as out:
                        out.write(content)
                    kept += 1
        return zip_name, kept, None
    except Exception as e:
        return zip_name, kept, str(e)


def main():
    if not os.path.isdir(ZIPS_DIR):
        raise SystemExit(
            f"No zips found at {ZIPS_DIR}. Run data/round2_curl.sh first."
        )
    os.makedirs(RAW_DIR, exist_ok=True)

    zip_paths = sorted(
        (os.path.join(ZIPS_DIR, f) for f in os.listdir(ZIPS_DIR) if f.endswith('.zip')),
        key=os.path.getsize,  # smallest first -- fast visible progress before the big ones
    )
    if not zip_paths:
        raise SystemExit(f"No .zip files found in {ZIPS_DIR}.")

    # This work is I/O-bound (reading zips, writing many small files to one
    # directory), not CPU-bound -- unlike compute-bound work, throwing more
    # parallel workers at it past a point doesn't help and can actively hurt
    # by overwhelming the storage backend's I/O queue (seen live: most
    # worker processes sitting in 'D' state -- uninterruptible sleep on I/O
    # -- at ~0.3-3.5% CPU each, not doing real work, just contending).
    n_workers = min(len(zip_paths), 4)
    print(f"Extracting {len(zip_paths)} zips using {n_workers} parallel workers ...")

    total = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_extract_one_zip, p, RAW_DIR): p for p in zip_paths}
        pending = set(futures)
        bar = tqdm(total=len(futures), desc="Extracting", unit="zip") if _HAVE_TQDM else None
        HEARTBEAT_S = 15
        while pending:
            done, pending = wait(pending, timeout=HEARTBEAT_S, return_when=FIRST_COMPLETED)
            if not done:
                # Nothing finished in the last HEARTBEAT_S seconds -- say so
                # explicitly instead of leaving a silent gap that's
                # indistinguishable from a genuine hang.
                elapsed = time.time() - t0
                n_done = len(futures) - len(pending)
                hb = f"  ... still working ({n_done}/{len(futures)} done, {elapsed:.0f}s elapsed, no completions in {HEARTBEAT_S}s)"
                bar.write(hb) if bar is not None else print(hb)
                continue
            for fut in done:
                zip_name, kept, err = fut.result()
                total += kept
                msg = f"  {'❌' if err else '✅'} {zip_name}: " + (err if err else f"{kept} files")
                if bar is not None:
                    bar.write(msg)
                    bar.update(1)
                else:
                    print(msg)
        if bar is not None:
            bar.close()

    print(f"\n✅ Total: {total} Java/Kotlin files in {time.time() - t0:.1f}s")

    # Merge into corpus
    print("📝 Merging into corpus_round2.txt...")
    raw_files = os.listdir(RAW_DIR)
    t0 = time.time()
    merge_iter = tqdm(raw_files, desc="Merging", unit="file") if _HAVE_TQDM else raw_files
    with open(CORPUS_PATH, "w") as out:
        for f in merge_iter:
            try:
                with open(os.path.join(RAW_DIR, f)) as inp:
                    out.write(inp.read())
                    out.write("\n\n")
            except Exception:
                pass

    size = os.path.getsize(CORPUS_PATH)
    print(f"✅ Round 2 corpus: {size / 1e6:.1f}MB ({time.time() - t0:.1f}s to merge)")


if __name__ == "__main__":
    main()
