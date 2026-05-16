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

# Snapshot pre-existing face-detect PIDs (e.g. archiviste daemon) — exclude from zombie check
PRE_EXISTING_PIDS=$(pgrep -x face-detect 2>/dev/null || true)

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

# ─── Test 11: Predict watchdog flag plumbing ──────────────────────
bold "Test 11: --predict-timeout flag plumbing"
rm -f /tmp/fd-test-in /tmp/fd-test-out
mkfifo /tmp/fd-test-in /tmp/fd-test-out

"$BINARY" --idle-timeout 60 --predict-timeout 30 --watch --in /tmp/fd-test-in --out /tmp/fd-test-out 2>/tmp/fd-test-stderr &
DAEMON_PID=$!
sleep 2

exec 3>/tmp/fd-test-in
exec 4</tmp/fd-test-out

assert "ready log mentions predict_timeout=30s" "grep -q 'predict_timeout=30s' /tmp/fd-test-stderr"

# Process an image — should succeed (predict << 30s)
echo "{\"image\":\"$DIR/lenna-face.png\",\"id\":\"watchdog-ok\"}" >&3
read -t 15 RESP <&4
assert "predict completes well under watchdog" "echo '$RESP' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d.get('id')=='watchdog-ok' and len(d.get('faces',[]))>=1\""

echo '{"shutdown":true,"id":"bye"}' >&3
read -t 10 RESP <&4
exec 3>&-
exec 4<&-
sleep 1
DAEMON_PID=""
rm -f /tmp/fd-test-in /tmp/fd-test-out /tmp/fd-test-stderr

# Clamping check (predict-timeout < 5 should clamp to 5)
CLAMP_OUT=$("$BINARY" --predict-timeout 1 --watch --in /tmp/no-such --out /tmp/no-such 2>&1 &
sleep 0.3; kill $! 2>/dev/null; true)
assert "predict-timeout=1 clamped to 5" "echo \"$CLAMP_OUT\" | grep -q 'clamped from 1 to 5'"

# ─── Test 12: Embedding format options ────────────────────────────
bold "Test 12: --embedding-format float (default) and b64"

# Default float — must have "embedding" array of 512 floats
OUT=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" "$DIR/lenna-face.png" 2>/dev/null)
assert "default format has embedding array of 512 floats" "echo '$OUT' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert isinstance(d['faces'][0]['embedding'], list) and len(d['faces'][0]['embedding'])==512\""
assert "default format has NO embedding_b64 field" "echo '$OUT' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert 'embedding_b64' not in d['faces'][0]\""
# 6-sig-figs roundtrip: re-rounding the JSON value should be idempotent (≤1e-7 diff).
assert "default values pre-rounded to 6 sig figs" "echo '$OUT' | python3 -c \"
import sys, json, math
d = json.load(sys.stdin)
emb = d['faces'][0]['embedding']
def sig6(x):
    if x == 0: return 0.0
    mag = 10 ** (5 - math.floor(math.log10(abs(x))))
    return round(x * mag) / mag
mismatches = [v for v in emb if abs(v - sig6(v)) > 1e-7]
assert len(mismatches) == 0, f'{len(mismatches)}/512 not at 6 sig figs, first: {mismatches[:3]}'
\""

# b64 mode — must have "embedding_b64" string instead of "embedding"
OUT_B64=$(FACE_DETECT_ALLOW_CLI=1 "$BINARY" --embedding-format b64 "$DIR/lenna-face.png" 2>/dev/null)
assert "b64 format has embedding_b64 string" "echo '$OUT_B64' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert isinstance(d['faces'][0]['embedding_b64'], str) and len(d['faces'][0]['embedding_b64'])>2000\""
assert "b64 format has NO embedding array field" "echo '$OUT_B64' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert 'embedding' not in d['faces'][0]\""
assert "b64 decodes to 512 little-endian Float32, L2-normalized" "echo '$OUT_B64' | python3 -c \"
import sys, json, base64, struct, math
d = json.load(sys.stdin)
b = base64.b64decode(d['faces'][0]['embedding_b64'])
assert len(b) == 2048, f'expected 2048 bytes, got {len(b)}'
floats = struct.unpack('<512f', b)
norm = math.sqrt(sum(x*x for x in floats))
assert 0.99 < norm < 1.01, f'L2 norm should be ~1, got {norm}'
\""

# b64 response is meaningfully smaller than float (≈25-30% smaller for 1 face)
B64_LEN=${#OUT_B64}
FLOAT_LEN=${#OUT}
assert "b64 response smaller than float response" "[[ $B64_LEN -lt $FLOAT_LEN ]]"

# Invalid format → exit 2
set +e
FACE_DETECT_ALLOW_CLI=1 "$BINARY" --embedding-format nonsense "$DIR/lenna-face.png" >/dev/null 2>&1
INVALID_EXIT=$?
set -e
assert "invalid --embedding-format rejected (exit 2)" "[[ $INVALID_EXIT -eq 2 ]]"

# ─── Test 13: Unix socket transport ────────────────────────────────
bold "Test 13: Watch mode — Unix domain socket"
SOCK=/tmp/fd-test.sock
rm -f "$SOCK"

"$BINARY" --idle-timeout 60 --watch --socket "$SOCK" 2>/tmp/fd-test-stderr &
DAEMON_PID=$!
sleep 2

assert "socket daemon starts" "kill -0 $DAEMON_PID 2>/dev/null"
assert "socket file created" "[[ -S '$SOCK' ]]"
assert "socket has 0600 permissions" "ls -l '$SOCK' | grep -q '^srw-------'"
assert "ready log mentions transport=unix-socket" "grep -q 'transport=unix-socket' /tmp/fd-test-stderr"
assert "ready log mentions sndbuf=1048576" "grep -q 'sndbuf=1048576' /tmp/fd-test-stderr"

# Run ping + image + shutdown in one session via python (proper duplex socket I/O)
RESPONSES=$(python3 -c "
import socket, json, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect('$SOCK')
f = s.makefile('rwb', buffering=0)
out = []
for req in [
    {'ping': True, 'id': 'sock-ping'},
    {'image': '$DIR/lenna-face.png', 'id': 'sock-img'},
    {'shutdown': True, 'id': 'sock-bye'},
]:
    f.write((json.dumps(req) + '\n').encode())
    line = f.readline()
    out.append(line.decode().strip())
print('\n'.join(out))
")

PONG=$(echo "$RESPONSES" | sed -n 1p)
IMG=$(echo "$RESPONSES" | sed -n 2p)
SHUT=$(echo "$RESPONSES" | sed -n 3p)

assert "socket ping returns pong+id" "echo '$PONG' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['pong']==True and d['id']=='sock-ping'\""
assert "socket image returns 1 face+id" "echo '$IMG' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d['faces'])==1 and d['id']=='sock-img'\""
assert "socket shutdown returns ack" "echo '$SHUT' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['shutdown']==True and d['id']=='sock-bye'\""

wait $DAEMON_PID 2>/dev/null
assert "daemon exited after socket shutdown" "! kill -0 $DAEMON_PID 2>/dev/null"
DAEMON_PID=""
rm -f "$SOCK" /tmp/fd-test-stderr

# Mutually exclusive: --socket cannot combine with --in/--out
set +e
"$BINARY" --watch --socket /tmp/x --in /tmp/y --out /tmp/z 2>/dev/null
MIX_EXIT=$?
set -e
assert "--socket + --in mutually exclusive" "[[ $MIX_EXIT -ne 0 ]]"

# ─── Test 14: No zombies ──────────────────────────────────────────
bold "Test 14: Zero zombies"
set +o pipefail
# Count only NEW face-detect processes (exclude pre-existing ones like archiviste daemon)
CURRENT_PIDS=$(pgrep -x face-detect 2>/dev/null || true)
ZOMBIES=0
for pid in $CURRENT_PIDS; do
    if ! echo "$PRE_EXISTING_PIDS" | grep -qw "$pid"; then
        ZOMBIES=$((ZOMBIES + 1))
    fi
done
set -o pipefail
assert "0 face-detect processes remaining" "[[ $ZOMBIES -eq 0 ]]"

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
