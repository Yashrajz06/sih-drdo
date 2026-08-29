#!/usr/bin/env bash
# One-shot Raspberry Pi 5 setup for the defence noise-suppression demo.
#
# Run this ON THE PI, from the repo root, after copying the project across:
#     bash scripts/setup_pi.sh
#
# It checks the things that actually break Pi deployments -- in the order they
# break -- then installs, benchmarks, and tells you what to do next. Every check
# either passes or explains exactly how to fix it; nothing is assumed.
#
# Deliberately does NOT touch /boot/firmware/config.txt or any system audio
# config. Those changes need a reboot and can leave a board unbootable if wrong,
# so anything in that category is printed as an instruction for you to apply.

set -uo pipefail

BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
FAILED=0
step() { printf '\n%s==> %s%s\n' "$BOLD" "$1" "$OFF"; }
ok()   { printf '  %s[ ok ]%s %s\n' "$GRN" "$OFF" "$1"; }
warn() { printf '  %s[warn]%s %s\n' "$YEL" "$OFF" "$1"; }
bad()  { printf '  %s[FAIL]%s %s\n' "$RED" "$OFF" "$1"; FAILED=1; }
note() { printf '         %s%s%s\n' "$DIM" "$1" "$OFF"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

printf '%s\n' "================================================================"
printf '%s\n' " Raspberry Pi setup — defence speech enhancement"
printf '%s\n' "================================================================"

# ---------------------------------------------------------------- 1. hardware
step "1/7  Board and OS"

MODEL="$(tr -d "\0" < /proc/device-tree/model 2>/dev/null || echo unknown)"
case "$MODEL" in
  *"Pi 5"*) ok "$MODEL" ;;
  *"Pi 4"*) warn "$MODEL — works, but ~2x slower than a Pi 5. Expect a higher RTF." ;;
  *"Pi 3"*) warn "$MODEL — may be too slow for real time. Check the RTF in step 7 carefully." ;;
  unknown)  warn "Not a Raspberry Pi (or model unreadable). Continuing anyway." ;;
  *)        warn "$MODEL" ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64)
    ok "64-bit ARM ($ARCH) — ONNX Runtime wheels available" ;;
  armv7l|armv6l)
    bad "32-bit ARM ($ARCH) — ONNX Runtime has no wheel for this."
    note "Reflash with 64-bit Raspberry Pi OS. Nothing below will work until you do." ;;
  x86_64)
    warn "x86_64 — not a Pi. Fine for rehearsing this script, but the benchmark"
    note "below measures THIS machine, not the target board." ;;
  *)
    warn "Unrecognised architecture ($ARCH) — proceed with caution" ;;
esac

# Undervoltage / throttling is the classic cause of 'random' Pi failures.
if command -v vcgencmd >/dev/null 2>&1; then
  T="$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)"
  if [ "$T" = "0x0" ]; then
    ok "Power and thermals clean (throttled=$T)"
  else
    warn "Throttling flags set: $T"
    note "Usually an underpowered supply. Use the official 27W USB-C PSU."
    note "Bit 0 = under-voltage now, bit 16 = under-voltage since boot."
  fi
fi

# ------------------------------------------------------------- 2. system deps
step "2/7  System packages"
NEEDED=()
for pkg in python3-venv python3-dev libportaudio2 libsndfile1; do
  dpkg -s "$pkg" >/dev/null 2>&1 || NEEDED+=("$pkg")
done
if [ ${#NEEDED[@]} -eq 0 ]; then
  ok "All present"
else
  echo "  installing: ${NEEDED[*]}"
  sudo apt-get update -qq && sudo apt-get install -y -qq "${NEEDED[@]}" \
    && ok "Installed" || bad "apt install failed"
fi

# --------------------------------------------------------------- 3. audio i/o
step "3/7  Audio devices"

if arecord -l 2>/dev/null | grep -q '^card'; then
  ok "Capture device(s) found:"
  arecord -l 2>/dev/null | grep '^card' | sed 's/^/         /'
else
  bad "No capture device."
  note "Plug in a USB sound card or mic, then re-run."
fi

if aplay -l 2>/dev/null | grep -q '^card'; then
  ok "Playback device(s) found:"
  aplay -l 2>/dev/null | grep '^card' | sed 's/^/         /'
else
  bad "No playback device."
  note "The Pi 5 has NO 3.5mm jack — a USB sound card is required for output."
fi

# --------------------------------------------------------------- 4. venv/deps
step "4/7  Python environment"
if [ ! -d .venv ]; then
  python3 -m venv .venv && ok "Created .venv" || bad "venv creation failed"
fi
# Guard the activate: without this, a failed venv creation silently falls through
# and pip installs into system Python, which on Raspberry Pi OS is externally
# managed and will either refuse or quietly break apt-managed packages.
if [ ! -f .venv/bin/activate ]; then
  bad "No .venv/bin/activate — cannot continue safely."
  note "Install the venv module:  sudo apt-get install -y python3-venv"
  note "Then re-run this script."
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Using $(python -c 'import sys;print(sys.executable)')"
python -m pip install --upgrade pip -q 2>/dev/null

# Runtime only. torch/torchaudio are NOT needed on the Pi -- inference runs on
# ONNX Runtime, and pulling torch onto an ARM board is a slow, pointless ~2 GB.
echo "  installing runtime deps (this takes a few minutes on first run)..."
if python -m pip install -q numpy scipy soundfile sounddevice onnxruntime; then
  ok "onnxruntime · sounddevice · numpy · scipy · soundfile"
else
  bad "pip install failed"
  note "If onnxruntime has no wheel, confirm 64-bit OS (step 1) and Python <= 3.12."
fi

# ------------------------------------------------------------------ 5. assets
step "5/7  Model and assets"
MODEL_FILE="models/gtcrn_defence.onnx"
if [ -f "$MODEL_FILE" ]; then
  ok "$MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
else
  bad "$MODEL_FILE missing."
  note "Copy it from your laptop:"
  note "  scp models/gtcrn_defence.onnx <pi>:$REPO/models/"
fi
[ -f demo_noise.wav ] && ok "demo_noise.wav" || warn "demo_noise.wav missing (needed for --inject-noise demos)"

# ------------------------------------------------------- 6. correctness check
step "6/7  Correctness check"
if [ -f "$MODEL_FILE" ]; then
  python scripts/live_demo.py --check --onnx "$MODEL_FILE" 2>&1 | sed 's/^/         /' \
    && ok "Streaming engine runs" || bad "Check failed — see output above"
else
  warn "Skipped (no model file)"
fi

# --------------------------------------------------------------- 7. benchmark
step "7/7  Benchmark on this board"
if [ -f "$MODEL_FILE" ]; then
python - "$MODEL_FILE" <<'PY'
import sys, time, numpy as np
sys.path.insert(0, "scripts")
from streaming_engine import StreamingEnhancer, HOP

path = sys.argv[1]
rng = np.random.default_rng(0)
hops = [(rng.standard_normal(HOP) * 0.05).astype(np.float32) for _ in range(400)]

for threads in (1, 2, 4):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    e = StreamingEnhancer(path)
    e.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    for h in hops[:80]:                      # warm up: first calls allocate
        e.process_hop(h)
    t = []
    for h in hops:
        t0 = time.perf_counter(); e.process_hop(h); t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    med, p95 = float(np.median(t)), float(np.percentile(t, 95))
    rtf = med / 16.0
    verdict = "REAL-TIME" if rtf < 0.5 else ("tight" if rtf < 1.0 else "TOO SLOW")
    print(f"         {threads} thread(s): {med:5.2f} ms/hop  p95 {p95:5.2f} ms  "
          f"RTF {rtf:.3f}  [{verdict}]")

print()
print("         RTF is model time per 16 ms of audio. Below ~0.5 leaves comfortable")
print("         headroom for the audio stack; p95 matters more than the median for")
print("         dropouts, since one slow frame is an audible glitch.")
PY
else
  warn "Skipped (no model file)"
fi

# ----------------------------------------------------------------- next steps
printf '\n%s\n' "================================================================"
if [ "$FAILED" -eq 0 ]; then
  printf ' %sSetup complete.%s\n' "$GRN" "$OFF"
else
  printf ' %sSetup finished with failures — fix those above first.%s\n' "$RED" "$OFF"
fi
printf '%s\n' "================================================================"
cat <<'EOF'

Next:

  source .venv/bin/activate

  # measure real end-to-end latency on this board (speakers, not headphones)
  python scripts/live_demo.py --measure-latency

  # live demo — wired headphones
  python scripts/live_demo.py --onnx models/gtcrn_defence.onnx

  # live demo with injected battlefield noise
  python scripts/live_demo.py --inject-noise demo_noise.wav --noise-gain 0.3 \
      --onnx models/gtcrn_defence.onnx

If you get buffer underruns (clicks/dropouts), raise the block size in
scripts/live_demo.py (blocksize=HOP) to 512 and re-test. That trades a little
latency for stability -- report whichever you actually measure.
EOF
