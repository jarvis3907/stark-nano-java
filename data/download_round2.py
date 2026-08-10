"""
Stark-Nano-Java — round 2 dataset downloader (extended repo list).

Companion to data/download.py, with a much larger/more specialized repo
list (Spring ecosystem, Hibernate, testing libraries, OpenJDK project
repos, and a couple of Kotlin repos) and its own output location so it
doesn't collide with the main corpus:

    data/round2/raw/*.java, *.kt   -- extracted source files (flattened,
                                       '/' -> '_' in filenames)
    data/round2/corpus_round2.txt  -- all kept files concatenated

Run:
    python data/download_round2.py

Note: this is a standalone corpus, separate from data/corpus.txt. train.py
only reads data/corpus.txt, so pulling this data doesn't feed train.py by
itself -- merge data/round2/corpus_round2.txt into data/corpus.txt (or
point --input at it directly when training the tokenizer) if you want it
included.
"""
import urllib.request
import zipfile
import os

repos = [
    # Spring ecosystem
    ("spring-projects/spring-boot", "4.1.x"),
    ("spring-projects/spring-framework", "6.2.x"),
    ("spring-projects/spring-security", "6.4.x"),
    ("spring-projects/spring-data-jpa", "main"),
    ("spring-projects/spring-data-commons", "main"),
    # JPA/ORM
    ("hibernate/hibernate-orm", "main"),
    ("querydsl/querydsl", "master"),
    # Testing
    ("mockito/mockito", "main"),
    ("assertj/assertj-core", "main"),
    ("junit-team/junit5", "main"),
    # Utilities
    ("google/guava", "master"),
    ("apache/commons-lang", "master"),
    ("FasterXML/jackson-databind", "2.x"),
    # Kotlin + Spring
    ("Kotlin/kotlinx.coroutines", "master"),
    ("JetBrains/Exposed", "main"),
    # ✅ Java 25 specific
    ("openjdk/jdk", "master"),           # JDK source
    ("openjdk/loom", "fibers"),          # Virtual threads Project Loom
    ("openjdk/amber", "master"),         # Pattern matching, records
    ("openjdk/babylon", "master"),       # Code reflection
    # Java 25 examples and demos
    ("eugenp/tutorials", "master"),      # Baeldung tutorials
    ("iluwatar/java-design-patterns", "master"),
    ("TheAlgorithms/Java", "master"),
]


# Anchored to this script's own location (the data/ dir), not the caller's
# working directory -- so `python data/download_round2.py` from the repo
# root and `python download_round2.py` from inside data/ both write to the
# same place instead of the latter creating a stray data/data/round2/.
ROUND2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2")
RAW_DIR = os.path.join(ROUND2_DIR, "raw")
TMP_ZIP = os.path.join(ROUND2_DIR, "tmp.zip")
CORPUS_PATH = os.path.join(ROUND2_DIR, "corpus_round2.txt")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    total = 0

    for repo, branch in repos:
        print(f"📥 {repo}...")
        url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
        try:
            urllib.request.urlretrieve(url, TMP_ZIP)
            with zipfile.ZipFile(TMP_ZIP) as z:
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
            os.remove(TMP_ZIP)
        except Exception as e:
            print(f"  ❌ {e}")

    print(f"\n✅ Total: {total} Java/Kotlin files!")

    # Merge
    print("📝 Merging...")
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
