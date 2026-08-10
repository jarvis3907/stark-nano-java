"""
Stark-Nano-Java — round 2 extraction/merge (companion to round2_curl.sh).

round2_curl.sh downloads each repo's zip individually into
data/round2/zips/ (with live progress, so a stuck/large repo like
openjdk/jdk is visible instead of silent). This script picks up from
there: extracts every .java/.kt file from each zip, keeps files in a
sane size range, and merges them into a single training corpus --
the same extraction logic download_round2.py uses, just reading from
already-downloaded zips instead of downloading them itself.

Run:
    python data/round2_extract.py

Output:
    data/round2/raw/*.java, *.kt   -- extracted source files (flattened)
    data/round2/corpus_round2.txt  -- all kept files concatenated
"""
import zipfile
import os

# Anchored to this script's own location, not the caller's working
# directory -- same reasoning as download_round2.py/round2_curl.sh, so
# this works whether you run it from the repo root or from inside data/.
ROUND2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2")
ZIPS_DIR = os.path.join(ROUND2_DIR, "zips")
RAW_DIR = os.path.join(ROUND2_DIR, "raw")
CORPUS_PATH = os.path.join(ROUND2_DIR, "corpus_round2.txt")


def main():
    if not os.path.isdir(ZIPS_DIR):
        raise SystemExit(
            f"No zips found at {ZIPS_DIR}. Run data/round2_curl.sh first."
        )
    os.makedirs(RAW_DIR, exist_ok=True)
    total = 0

    for zip_file in os.listdir(ZIPS_DIR):
        if not zip_file.endswith('.zip'):
            continue
        print(f"📦 Extracting {zip_file}...")
        try:
            with zipfile.ZipFile(os.path.join(ZIPS_DIR, zip_file)) as z:
                java_files = [
                    f for f in z.namelist()
                    if f.endswith('.java') or f.endswith('.kt')
                ]
                kept = 0
                for f in java_files:
                    content = z.read(f).decode('utf-8', errors='ignore')
                    if 100 < len(content) < 100000:
                        safe = f.replace('/', '_')
                        with open(os.path.join(RAW_DIR, safe), 'w') as out:
                            out.write(content)
                        kept += 1
                total += kept
                print(f"  ✅ {kept} files")
        except Exception as e:
            print(f"  ❌ {e}")

    print(f"\n✅ Total: {total} Java/Kotlin files!")

    # Merge into corpus
    print("📝 Merging into corpus_round2.txt...")
    with open(CORPUS_PATH, "w") as out:
        for f in os.listdir(RAW_DIR):
            try:
                with open(os.path.join(RAW_DIR, f)) as inp:
                    out.write(inp.read() + "\n\n")
            except Exception:
                pass

    size = os.path.getsize(CORPUS_PATH)
    print(f"✅ Round 2 corpus: {size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
