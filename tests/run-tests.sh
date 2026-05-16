#!/bin/bash
# face-detect non-regression test suite
# Usage: ./helpers/face-detect/tests/run-tests.sh [path/to/face-detect]
#
# Exit codes: 0 = all pass, non-zero = number of failures
set -euo pipefail

BINARY="${1:-./bin/face-detect}"
DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
TOTAL=0

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

assert() {
    local name="$1" condition="$2"
    TOTAL=$((TOTAL + 1))
    if eval "$condition"; then
        green "  PASS: $name"
        PASS=$((PASS + 1))
    else
        red "  FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
}

# Cleanup trap — kill leftover daemons and remove FIFOs on exit
cleanup() {
    set +e
    [[ -n "${DAEMON_PID:-}" ]] && kill "$DAEMON_PID" 2>/dev/null
    rm -f /tmp/fd-test-in /tmp/fd-test-out
    exec 3>&- 2>/dev/null
    exec 4<&- 2>/dev/null
    set -e
}
trap cleanup EXIT

# Verify binary exists
if [[ ! -x "$BINARY" ]]; then
    red "Binary not found or not executable: $BINARY"
    echo "Run 'make face-detect' first."
    exit 1
fi

# ─── Test 1: CLI mode blocked by default ───────────────────────────
bold "Test 1: CLI mode blocked without FACE_DETECT_ALLOW_CLI"
set +e
CLI_OUTPUT=$(unset FACE_DETECT_ALLOW_CLI; "$BINARY" /dev/null 2>&1)
CLI_EXIT=$?
set -e
assert "exits with non-zero code" "[[ $CLI_EXIT -ne 0 ]]"
assert "mentions CLI disabled" "echo '$CLI_OUTPUT' | grep -q 'CLI mode disabled'"

# ─── Test 2: Image without face — no crash, expected output ────────
bold "Test 2: Image without face (solid blue)"
JSON=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" "$DIR/solid-blue-noface.png" 2>/dev/null)
EXIT=$?
assert "exit code 0" "[[ $EXIT -eq 0 ]]"
assert "valid JSON structure" "echo '$JSON' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert all(k in d for k in ('image','width','height','elapsed_ms','engine','engine_dim','faces'))\""
FACES=$(echo "$JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['faces']))")
assert "0 faces detected" "[[ $FACES -eq 0 ]]"
DESC=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('description',''))")
assert "description is 'image' (no face fallback)" "[[ '$DESC' == 'image' ]]"

# ─── Test 3: Image with face — detection + golden values ──────────
bold "Test 3: Image with face (Lenna) — golden values"
JSON=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --lang fr "$DIR/lenna-face.png" 2>/dev/null)
EXIT=$?
assert "exit code 0" "[[ $EXIT -eq 0 ]]"
FACES=$(echo "$JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['faces']))")
assert "exactly 1 face detected" "[[ $FACES -eq 1 ]]"
# Description
DESC=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('description',''))")
assert "description = 'personne'" "[[ '$DESC' == 'personne' ]]"
# Tags — Lenna should have people, adult, hat
assert "tags include 'people'" "echo '$JSON' | python3 -c \"import sys,json; tags=[t['label'] for t in json.load(sys.stdin).get('tags',[])]; assert 'people' in tags, tags\""
assert "tags include 'adult'" "echo '$JSON' | python3 -c \"import sys,json; tags=[t['label'] for t in json.load(sys.stdin).get('tags',[])]; assert 'adult' in tags, tags\""
assert "tags include 'hat'" "echo '$JSON' | python3 -c \"import sys,json; tags=[t['label'] for t in json.load(sys.stdin).get('tags',[])]; assert 'hat' in tags, tags\""
# Embedding
EMBED_LEN=$(echo "$JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['faces'][0]['embedding']))")
assert "embedding is 512d (adaface)" "[[ $EMBED_LEN -eq 512 ]]"
# Confidence
assert "confidence = 1.0" "echo '$JSON' | python3 -c \"import sys,json; c=json.load(sys.stdin)['faces'][0]['confidence']; assert c >= 0.99, c\""
# Quality
assert "quality > 0.5" "echo '$JSON' | python3 -c \"import sys,json; q=json.load(sys.stdin)['faces'][0]['quality']; assert q > 0.5, q\""
# Model field
assert "model = ir18" "echo '$JSON' | python3 -c \"import sys,json; assert json.load(sys.stdin)['model'] == 'ir18'\""
# L2 norm ~1.0
assert "embedding L2-normalized" "echo '$JSON' | python3 -c \"import sys,json,math; e=json.load(sys.stdin)['faces'][0]['embedding']; n=math.sqrt(sum(x*x for x in e)); assert 0.99 < n < 1.01, n\""

# ─── Test 4: --min-quality clamping ─────────────────────────────────
bold "Test 4: --min-quality value clamping"
JSON=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --min-quality 2.0 "$DIR/lenna-face.png" 2>/dev/null)
FACES=$(echo "$JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['faces']))")
assert "min-quality=2.0 clamped to 1.0, filters all faces" "[[ $FACES -eq 0 ]]"

# ─── Test 5: Vision engine fallback ────────────────────────────────
bold "Test 5: Vision engine"
JSON=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --engine vision "$DIR/lenna-face.png" 2>/dev/null)
EXIT=$?
assert "exit code 0 with --engine vision" "[[ $EXIT -eq 0 ]]"
EMBED_LEN=$(echo "$JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['faces'][0]['embedding']) if d['faces'] else 0)")
assert "vision engine 768d embedding" "[[ $EMBED_LEN -eq 768 ]]"
ENGINE=$(echo "$JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['engine'])")
assert "engine field says vision" "[[ '$ENGINE' == 'vision' ]]"

# ─── Test 6: --version flag ───────────────────────────────────────
bold "Test 6: --version"
VOUT=$("$BINARY" --version 2>/dev/null)
assert "--version outputs version string" "echo '$VOUT' | grep -q 'face-detect'"
assert "--version contains engine info" "echo '$VOUT' | grep -q 'engine='"

# ─── Test 7: --lang parameter ─────────────────────────────────────
bold "Test 7: --lang i18n descriptions"
JSON_FR=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --lang fr "$DIR/lenna-face.png" 2>/dev/null)
DESC_FR=$(echo "$JSON_FR" | python3 -c "import sys,json; print(json.load(sys.stdin).get('description',''))")
assert "--lang fr produces French description" "echo '$DESC_FR' | grep -qE 'personne|bébé|enfant|image'"

JSON_EN=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --lang en "$DIR/lenna-face.png" 2>/dev/null)
DESC_EN=$(echo "$JSON_EN" | python3 -c "import sys,json; print(json.load(sys.stdin).get('description',''))")
assert "--lang en produces English description" "echo '$DESC_EN' | grep -qE 'person|baby|child|image'"

# ─── Test 8: IR-50 model (if installed) ────────────────────────────
IR50_MODEL="/opt/homebrew/share/face-detect/AdaFace_IR50.mlpackage"
if [[ -d "$IR50_MODEL" ]]; then
    bold "Test 8: IR-50 model — detection + golden values"
    JSON50=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --model ir50 --lang fr "$DIR/lenna-face.png" 2>/dev/null)
    assert "IR-50 exit code 0" "[[ $? -eq 0 ]]"
    assert "IR-50 model field = ir50" "echo '$JSON50' | python3 -c \"import sys,json; assert json.load(sys.stdin)['model'] == 'ir50'\""
    assert "IR-50 engine = adaface" "echo '$JSON50' | python3 -c \"import sys,json; assert json.load(sys.stdin)['engine'] == 'adaface'\""
    assert "IR-50 dim = 512" "echo '$JSON50' | python3 -c \"import sys,json; assert json.load(sys.stdin)['engine_dim'] == 512\""
    assert "IR-50 detects 1 face" "echo '$JSON50' | python3 -c \"import sys,json; assert len(json.load(sys.stdin)['faces']) == 1\""
    assert "IR-50 description = personne" "echo '$JSON50' | python3 -c \"import sys,json; assert json.load(sys.stdin)['description'] == 'personne'\""
    assert "IR-50 embedding 512d L2-normalized" "echo '$JSON50' | python3 -c \"
import sys,json,math
e=json.load(sys.stdin)['faces'][0]['embedding']
assert len(e)==512, len(e)
n=math.sqrt(sum(x*x for x in e))
assert 0.99 < n < 1.01, n
\""
    # IR-50 embeddings must differ from IR-18 (proves model is actually loaded)
    echo "$JSON" > /tmp/fd-test-ir18.json
    echo "$JSON50" > /tmp/fd-test-ir50.json
    assert "IR-50 embeddings differ from IR-18" "python3 -c \"
import json, math
ir18=json.load(open('/tmp/fd-test-ir18.json'))['faces'][0]['embedding']
ir50=json.load(open('/tmp/fd-test-ir50.json'))['faces'][0]['embedding']
dot=sum(a*b for a,b in zip(ir18,ir50))
n18=math.sqrt(sum(x*x for x in ir18))
n50=math.sqrt(sum(x*x for x in ir50))
cos=dot/(n18*n50)
assert cos < 0.95, f'IR18/IR50 cosine={cos:.4f}, too similar — same model?'
\""
    rm -f /tmp/fd-test-ir18.json /tmp/fd-test-ir50.json
    # IR-50 with --lang en
    DESC50_EN=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --model ir50 --lang en "$DIR/lenna-face.png" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('description',''))")
    assert "IR-50 --lang en description = person" "[[ '$DESC50_EN' == 'person' ]]"
else
    bold "Test 8: IR-50 model — SKIPPED (not installed)"
    green "  SKIP: IR-50 model not found at $IR50_MODEL"
fi

# ─── Test 9: Watch mode — ping + image + shutdown ──────────────────
bold "Test 9: Watch mode — ping + image + shutdown protocol"
rm -f /tmp/fd-test-in /tmp/fd-test-out
mkfifo /tmp/fd-test-in /tmp/fd-test-out

"$BINARY" --idle-timeout 60 --watch --in /tmp/fd-test-in --out /tmp/fd-test-out &
DAEMON_PID=$!
sleep 2

assert "daemon starts" "kill -0 $DAEMON_PID 2>/dev/null"

# Open FIFOs
exec 3>/tmp/fd-test-in
exec 4</tmp/fd-test-out

# Ping
echo '{"ping":true,"id":"test-ping"}' >&3
read -t 10 RESP <&4
assert "ping returns pong" "echo '$RESP' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['pong']==True\""
assert "ping has id field" "echo '$RESP' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['id']=='test-ping'\""
assert "pong has model field" "echo '$RESP' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert 'model' in d\""

# Process image via watch
echo "{\"image\":\"$DIR/lenna-face.png\",\"id\":\"test-face\"}" >&3
read -t 15 RESP <&4
FACES=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('faces',[])))")
assert "watch detects face in Lenna" "[[ $FACES -ge 1 ]]"
assert "watch has id field" "echo '$RESP' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d.get('id')=='test-face'\""

# Shutdown
echo '{"shutdown":true,"id":"test-shutdown"}' >&3
read -t 10 RESP <&4
assert "shutdown response received" "echo '$RESP' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['shutdown']==True\""
assert "shutdown has id field" "echo '$RESP' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['id']=='test-shutdown'\""

exec 3>&-
exec 4<&-
sleep 1

assert "daemon exited after shutdown" "! kill -0 $DAEMON_PID 2>/dev/null"
DAEMON_PID=""

rm -f /tmp/fd-test-in /tmp/fd-test-out

# ─── Test 9: SIGTERM clean exit ────────────────────────────────────
bold "Test 10: SIGTERM clean exit"
rm -f /tmp/fd-test-in /tmp/fd-test-out
mkfifo /tmp/fd-test-in /tmp/fd-test-out

"$BINARY" --idle-timeout 60 --watch --in /tmp/fd-test-in --out /tmp/fd-test-out &
DAEMON_PID=$!
sleep 2

# Open FIFOs to let daemon start
exec 3>/tmp/fd-test-in
exec 4</tmp/fd-test-out

kill -TERM $DAEMON_PID
sleep 1
assert "daemon dies on SIGTERM" "! kill -0 $DAEMON_PID 2>/dev/null"
DAEMON_PID=""

exec 3>&- 2>/dev/null
exec 4<&- 2>/dev/null
rm -f /tmp/fd-test-in /tmp/fd-test-out

# ─── Test 10: No zombies ──────────────────────────────────────────
bold "Test 11: Zero zombies"
set +o pipefail
ZOMBIES=$(pgrep -x face-detect 2>/dev/null | wc -l | tr -d ' ')
set -o pipefail
assert "0 face-detect processes remaining" "[[ ${ZOMBIES:-0} -eq 0 ]]"

# ─── Summary ──────────────────────────────────────────────────────
echo ""
bold "═══════════════════════════════════════════"
if [[ $FAIL -eq 0 ]]; then
    green "ALL $TOTAL TESTS PASSED"
else
    red "$FAIL/$TOTAL TESTS FAILED"
fi
bold "═══════════════════════════════════════════"

exit $FAIL
