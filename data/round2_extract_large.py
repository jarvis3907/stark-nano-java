"""
Stark-Nano-Java — extraction for the large OpenJDK-family zips.

round2_extract.py's parallel extraction (one worker per zip) causes I/O
contention when many workers hit the storage backend at once -- fine for
the ~17 small repo zips, but the four ~188MB OpenJDK-family zips
(jdk/loom/amber/babylon, each a near-full fork of the JDK source tree,
tens of thousands of .java files each) are exactly the case most exposed
to it. This script processes the four zips *sequentially* (one at a time,
never competing with each other for I/O), but parallelizes *within* each
zip using a small worker pool (WORKERS_PER_ZIP, default 4) -- the same
concurrency level already proven safe for whole-zip parallelism in
round2_extract.py, just applied at file-level granularity here since a
single zip is a single unit of I/O-contention risk when nothing else is
running concurrently with it.

Run after round2_curl.sh has downloaded the zips (round2_extract.py can
have already processed the rest -- this only touches the four listed
below, and writes to the same data/round2/raw/ output directory, so it's
safe to run either before or after round2_extract.py).

    python data/round2_extract_large.py
"""
import multiprocessing as mp
import os
import zipfile

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

# Anchored to this script's own location, not the caller's working
# directory -- same reasoning as the other data/round2_*.py scripts.
ROUND2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2")
ZIPS_DIR = os.path.join(ROUND2_DIR, "zips")
RAW_DIR = os.path.join(ROUND2_DIR, "raw")

LARGE_ZIPS = ["jdk.zip", "loom.zip", "amber.zip", "babylon.zip"]

MIN_CHARS = 100
MAX_CHARS = 100_000
WORKERS_PER_ZIP = 4

# Set once per worker process by _init_worker, so each worker opens its own
# ZipFile handle a single time instead of reopening it per file (multiple
# processes independently reading the same zip file concurrently is safe).
_worker_zip = None


def _init_worker(zip_path: str):
    global _worker_zip
    _worker_zip = zipfile.ZipFile(zip_path)


def _extract_one_file(filename: str) -> int:
    try:
        content = _worker_zip.read(filename).decode('utf-8', errors='ignore')
        if MIN_CHARS < len(content) < MAX_CHARS:
            safe = filename.replace('/', '_')
            with open(os.path.join(RAW_DIR, safe), 'w') as out:
                out.write(content)
            return 1
    except Exception:
        pass
    return 0


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    total = 0

    for name in LARGE_ZIPS:
        zip_path = os.path.join(ZIPS_DIR, name)
        print(f"📦 Extracting {name}...")
        if not os.path.exists(zip_path):
            print(f"  ❌ {name}: not found at {zip_path} (did round2_curl.sh download it?)")
            continue
        try:
            with zipfile.ZipFile(zip_path) as z:
                # infolist() (not namelist()) so file_size is available --
                # lets us skip decompressing entries that can't possibly
                # pass the size filter, instead of always decompressing
                # then throwing the result away. Matters a lot here: these
                # repos have plenty of oversized generated/data files.
                filenames = [
                    info.filename for info in z.infolist()
                    if (info.filename.endswith('.java') or info.filename.endswith('.kt'))
                    and MIN_CHARS < info.file_size < MAX_CHARS
                ]

            if not filenames:
                print(f"  ✅ {name}: 0 files")
                continue

            n_workers = min(WORKERS_PER_ZIP, len(filenames))
            kept = 0
            with mp.Pool(n_workers, initializer=_init_worker, initargs=(zip_path,)) as pool:
                results = pool.imap_unordered(_extract_one_file, filenames, chunksize=50)
                iterator = tqdm(results, total=len(filenames), desc=f"  {name}", unit="file") \
                    if _HAVE_TQDM else results
                for r in iterator:
                    kept += r

            total += kept
            print(f"  ✅ {name}: {kept} files")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"\n✅ Total new files: {total}")
    print("Run data/round2_extract.py's merge step (or re-run round2_extract.py "
          "entirely -- it's safe to re-run, files are overwritten in place) "
          "to fold these into corpus_round2.txt.")


if __name__ == "__main__":
    main()
