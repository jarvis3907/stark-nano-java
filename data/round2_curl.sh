#!/usr/bin/env bash
# Individual curl downloads for data/download_round2.py's repo list, for
# diagnosing which repo (if any) is actually stuck vs. just slow --
# urllib.request.urlretrieve (used by the Python script) shows zero
# progress, so a big/slow repo like openjdk/jdk looks identical to a truly
# hung connection. curl shows a live progress bar and these have a
# connect/max-time cap so a dead connection fails fast instead of hanging.
#
# Run the whole thing:      bash data/round2_curl.sh
# Run just one repo:        copy/paste a single line below and run it directly
# Skip a repo that hangs:   Ctrl+C that one line, move to the next
#
# After downloading, unzip + filter + merge into round2_curl.py's job (see
# the follow-up script) or reuse download_round2.py's extraction logic.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/round2/zips" || exit 1

# --connect-timeout: max seconds to establish connection
# --max-time: max seconds for the whole transfer (raise for openjdk/jdk if
#             it's still transferring but slow -- check the progress bar
#             first before assuming it needs more time)
# -L: follow redirects (codeload can redirect)
# -o: output file
# --fail: nonzero exit on HTTP errors (4xx/5xx) instead of saving an error page

curl -L --fail --connect-timeout 15 --max-time 300 -o spring-boot.zip \
  "https://codeload.github.com/spring-projects/spring-boot/zip/refs/heads/4.1.x"

curl -L --fail --connect-timeout 15 --max-time 300 -o spring-framework.zip \
  "https://codeload.github.com/spring-projects/spring-framework/zip/refs/heads/6.2.x"

curl -L --fail --connect-timeout 15 --max-time 300 -o spring-security.zip \
  "https://codeload.github.com/spring-projects/spring-security/zip/refs/heads/6.4.x"

curl -L --fail --connect-timeout 15 --max-time 300 -o spring-data-jpa.zip \
  "https://codeload.github.com/spring-projects/spring-data-jpa/zip/refs/heads/main"

curl -L --fail --connect-timeout 15 --max-time 300 -o spring-data-commons.zip \
  "https://codeload.github.com/spring-projects/spring-data-commons/zip/refs/heads/main"

curl -L --fail --connect-timeout 15 --max-time 300 -o hibernate-orm.zip \
  "https://codeload.github.com/hibernate/hibernate-orm/zip/refs/heads/main"

curl -L --fail --connect-timeout 15 --max-time 300 -o querydsl.zip \
  "https://codeload.github.com/querydsl/querydsl/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o mockito.zip \
  "https://codeload.github.com/mockito/mockito/zip/refs/heads/main"

curl -L --fail --connect-timeout 15 --max-time 300 -o assertj-core.zip \
  "https://codeload.github.com/assertj/assertj-core/zip/refs/heads/main"

curl -L --fail --connect-timeout 15 --max-time 300 -o junit5.zip \
  "https://codeload.github.com/junit-team/junit5/zip/refs/heads/main"

curl -L --fail --connect-timeout 15 --max-time 300 -o guava.zip \
  "https://codeload.github.com/google/guava/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o commons-lang.zip \
  "https://codeload.github.com/apache/commons-lang/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o jackson-databind.zip \
  "https://codeload.github.com/FasterXML/jackson-databind/zip/refs/heads/2.x"

curl -L --fail --connect-timeout 15 --max-time 300 -o kotlinx-coroutines.zip \
  "https://codeload.github.com/Kotlin/kotlinx.coroutines/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o exposed.zip \
  "https://codeload.github.com/JetBrains/Exposed/zip/refs/heads/main"

# openjdk/jdk is the full JDK source tree -- expect this one to be by far
# the largest/slowest. Given it a longer max-time on purpose; watch the
# progress bar to confirm it's actually transferring, not stalled.
curl -L --fail --connect-timeout 15 --max-time 1200 -o jdk.zip \
  "https://codeload.github.com/openjdk/jdk/zip/refs/heads/master"

# openjdk/loom's "fibers" branch may no longer exist (Loom has largely
# merged into mainline JDK) -- --fail means this will just error cleanly
# instead of hanging or saving a GitHub 404 HTML page as a fake zip.
curl -L --fail --connect-timeout 15 --max-time 300 -o loom.zip \
  "https://codeload.github.com/openjdk/loom/zip/refs/heads/fibers"

curl -L --fail --connect-timeout 15 --max-time 300 -o amber.zip \
  "https://codeload.github.com/openjdk/amber/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o babylon.zip \
  "https://codeload.github.com/openjdk/babylon/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o tutorials.zip \
  "https://codeload.github.com/eugenp/tutorials/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o java-design-patterns.zip \
  "https://codeload.github.com/iluwatar/java-design-patterns/zip/refs/heads/master"

curl -L --fail --connect-timeout 15 --max-time 300 -o algorithms-java.zip \
  "https://codeload.github.com/TheAlgorithms/Java/zip/refs/heads/master"

echo ""
echo "Done. Zips saved under data/round2/zips/. ls -la to see sizes/which succeeded:"
ls -la .
