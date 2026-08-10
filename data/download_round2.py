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
    ("querydsl/querydsl", "main"),
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


def main():
    os.makedirs("data/round2/raw", exist_ok=True)
    total = 0

    for repo, branch in repos:
        print(f"📥 {repo}...")
        url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
        try:
            urllib.request.urlretrieve(url, "data/round2/tmp.zip")
            with zipfile.ZipFile("data/round2/tmp.zip") as z:
                java_files = [
                    f for f in z.namelist()
                    if f.endswith('.java') or f.endswith('.kt')
                ]
                kept = 0
                for f in java_files:
                    content = z.read(f).decode('utf-8', errors='ignore')
                    if 100 < len(content) < 100000:
                        safe = f.replace('/', '_')
                        with open(f"data/round2/raw/{safe}", 'w') as out:
                            out.write(content)
                        kept += 1
                total += kept
                print(f"  ✅ {kept} files")
            os.remove("data/round2/tmp.zip")
        except Exception as e:
            print(f"  ❌ {e}")

    print(f"\n✅ Total: {total} Java/Kotlin files!")

    # Merge
    print("📝 Merging...")
    with open("data/round2/corpus_round2.txt", "w") as out:
        for f in os.listdir("data/round2/raw"):
            try:
                with open(f"data/round2/raw/{f}") as inp:
                    out.write(inp.read() + "\n\n")
            except Exception:
                pass

    size = os.path.getsize("data/round2/corpus_round2.txt")
    print(f"✅ Round 2 corpus: {size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
