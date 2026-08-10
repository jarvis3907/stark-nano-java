"""
Stark-Nano-Java — sequential extraction for the large OpenJDK-family zips.

round2_extract.py's parallel extraction (one worker per zip) causes I/O
contention when many workers hit the storage backend at once -- fine for
the ~17 small repo zips, but the four ~188MB OpenJDK-family zips
(jdk/loom/amber/babylon, each a near-full fork of the JDK source tree)
are exactly the case most exposed to it. Running just these four
sequentially, one at a time, avoids the contention.

Run after round2_curl.sh has downloaded the zips (round2_extract.py can
have already processed the rest -- this only touches the four listed
below, and writes to the same data/round2/raw/ output directory, so it's
safe to run either before or after round2_extract.py).

    python data/round2_extract_large.py
"""
import os
import zipfile

# Anchored to this script's own location, not the caller's working
# directory -- same reasoning as the other data/round2_*.py scripts.
ROUND2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2")
ZIPS_DIR = os.path.join(ROUND2_DIR, "zips")
RAW_DIR = os.path.join(ROUND2_DIR, "raw")

LARGE_ZIPS = ["jdk.zip", "loom.zip", "amber.zip", "babylon.zip"]

MIN_CHARS = 100
MAX_CHARS = 100_000


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
                java_files = [
                    f for f in z.namelist()
                    if f.endswith('.java') or f.endswith('.kt')
                ]
                kept = 0
                for f in java_files:
                    try:
                        content = z.read(f).decode('utf-8', errors='ignore')
                        if MIN_CHARS < len(content) < MAX_CHARS:
                            safe = f.replace('/', '_')
                            with open(os.path.join(RAW_DIR, safe), 'w') as out:
                                out.write(content)
                            kept += 1
                    except Exception:
                        pass
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
