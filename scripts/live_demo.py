"""
Live GTCRN speech-enhancement demo. Laptop stand-in for the Raspberry Pi target
(no Pi hardware available yet) -- proves the real-time architecture, not the
eventual embedded deployment.

Modes:
  --check                         offline correctness + RTF check, no audio hardware
  --capture-test SEC --out FILE   mic -> file for SEC seconds, no speaker output
  (default, no flags)             mic -> enhance -> speakers, live, 'e' toggles, 'q' quits

Note: this is the pretrained model_trained_on_dns3.tar checkpoint, not yet fine-tuned
for impulsive defence noise (gunshots/shelling) -- see docs/solution-design.md Stage 1.
"""
import argparse
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from streaming_engine import HOP, N_FFT, WINDOW, StreamingEnhancer  # noqa: E402

SAMPLE_RATE = 16000
GTCRN_DIR = REPO_ROOT / "third_party" / "gtcrn"


def best_lag(ref: np.ndarray, test: np.ndarray, max_lag: int = 600) -> tuple[int, float]:
    """Find the sample shift of `test` relative to `ref` that minimizes mean abs
    error. Needed because the causal streaming engine and any non-streaming
    reference computed with centered STFT differ by a fixed frame-alignment
    offset, not just noise -- this both validates correctness and measures the
    real algorithmic delay."""
    best_lag_val, best_err = 0, float("inf")
    for lag in range(max_lag):
        n = min(len(ref), len(test) - lag)
        if n <= 0:
            break
        err = float(np.abs(ref[:n] - test[lag : lag + n]).mean())
        if err < best_err:
            best_err, best_lag_val = err, lag
    return best_lag_val, best_err


def causal_reference(mix: np.ndarray) -> np.ndarray:
    """Non-streaming (whole-utterance) GTCRN forward pass, but framed and
    reconstructed with the exact same manual, non-centered analysis/OLA-synthesis
    convention as StreamingEnhancer (torch's istft(center=False) hits an internal
    NOLA/edge-normalization assertion, and center=True would reflect-pad using
    samples a live stream doesn't have yet -- so we roll our own here to get a
    fair, directly-comparable ground truth). GTCRN's time-axis RNNs are all
    unidirectional (see DPGRNN.inter_rnn, bidirectional=False), so this whole-file
    batch pass is causal and should numerically match the frame-by-frame
    streaming engine almost exactly."""
    sys.path.insert(0, str(GTCRN_DIR))
    from gtcrn import GTCRN

    model = GTCRN().eval()
    ckpt = torch.load(GTCRN_DIR / "checkpoints" / "model_trained_on_dns3.tar", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    n_hops = len(mix) // HOP
    buf = np.zeros(N_FFT, dtype=np.float32)
    frames = []
    for i in range(n_hops):
        buf = np.concatenate([buf[HOP:], mix[i * HOP : (i + 1) * HOP]])
        frames.append(np.fft.rfft(buf * WINDOW, n=N_FFT))
    spec = np.stack(frames, axis=1)  # (257, n_hops) complex

    spec_t = torch.from_numpy(np.stack([spec.real, spec.imag], axis=-1).astype(np.float32))[None]  # (1,257,n_hops,2)
    with torch.no_grad():
        out = model(spec_t)[0].numpy()  # (257, n_hops, 2)

    synth_buf = np.zeros(N_FFT, dtype=np.float32)
    out_samples = np.zeros(n_hops * HOP, dtype=np.float32)
    for i in range(n_hops):
        spec_out = out[:, i, 0] + 1j * out[:, i, 1]
        frame = np.fft.irfft(spec_out, n=N_FFT).astype(np.float32) * WINDOW
        synth_buf = synth_buf + frame
        out_samples[i * HOP : (i + 1) * HOP] = synth_buf[:HOP]
        synth_buf = np.concatenate([synth_buf[HOP:], np.zeros(HOP, dtype=np.float32)])
    return out_samples


def run_streaming(mix: np.ndarray, enabled: bool = True) -> tuple[np.ndarray, float]:
    enhancer = StreamingEnhancer()
    n_hops = len(mix) // HOP
    out = np.zeros(n_hops * HOP, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n_hops):
        hop = mix[i * HOP : (i + 1) * HOP]
        out[i * HOP : (i + 1) * HOP] = enhancer.process_hop(hop, enabled=enabled)
    elapsed = time.perf_counter() - t0
    audio_duration = n_hops * HOP / SAMPLE_RATE
    return out, elapsed / audio_duration


def cmd_check(_args) -> None:
    mix_path = GTCRN_DIR / "stream" / "test_wavs" / "mix.wav"
    mix, fs = sf.read(mix_path, dtype="float32")
    assert fs == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {fs}"

    print(f"input: {mix_path} ({len(mix)/SAMPLE_RATE:.2f}s)")
    print("computing causal batch reference (non-streaming GTCRN, non-centered STFT)...")
    ref = causal_reference(mix)

    print("running streaming ONNX engine hop-by-hop...")
    streaming_out, rtf = run_streaming(mix, enabled=True)

    lag, mean_err = best_lag(ref, streaming_out)
    n = min(len(ref) - lag, len(streaming_out) - lag)
    max_err = float(np.abs(ref[:n] - streaming_out[lag : lag + n]).max())

    out_path = Path("/tmp/streaming_check.wav")
    sf.write(out_path, streaming_out, SAMPLE_RATE)

    print()
    print(f"measured algorithmic delay: {lag} samples (~{lag / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"mean abs error (aligned):   {mean_err:.6f}")
    print(f"max abs error (aligned):    {max_err:.6f}")
    verdict = "real-time capable" if rtf < 1 else "NOT real-time"
    print(f"RTF on this CPU:            {rtf:.4f}  ({verdict})")
    print(f"wrote {out_path}")

    if max_err > 0.05:
        print(
            "\nWARNING: error is larger than expected for a numerically-equivalent "
            "streaming reformulation -- investigate before going live.",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_noise_loop(path: str) -> np.ndarray:
    import torchaudio

    noise, fs = sf.read(path, dtype="float32")
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    if fs != SAMPLE_RATE:
        noise = torchaudio.functional.resample(torch.from_numpy(noise), fs, SAMPLE_RATE).numpy()
    return noise.astype(np.float32)


def _synthetic_pink_noise(seconds: float = 10.0) -> np.ndarray:
    """Cheap pink-ish noise (Paul Kellet's filter) so the demo has a repeatable
    noise bed even with no noise dataset downloaded yet."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(0)
    white = rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white).astype(np.float32)
    return pink / (np.abs(pink).max() + 1e-8)


def cmd_capture_test(args) -> None:
    import sounddevice as sd

    enhancer = StreamingEnhancer()
    collected = []

    def cb(indata, _frames, _time_info, status):
        if status:
            print(status, file=sys.stderr)
        collected.append(enhancer.process_hop(indata[:, 0].copy(), enabled=True))

    print(f"recording {args.capture_test}s from the default input device...")
    with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=HOP, channels=1, dtype="float32", callback=cb):
        sd.sleep(int(args.capture_test * 1000))

    out = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    sf.write(args.out, out, SAMPLE_RATE)
    print(f"wrote {args.out} ({len(out)/SAMPLE_RATE:.1f}s captured)")


def cmd_live(args) -> None:
    import sounddevice as sd

    enhancer = StreamingEnhancer()
    state = {"enabled": True, "running": True, "rtf_samples": [], "last_print": time.time(), "noise_pos": 0}

    noise_loop = None
    if args.inject_noise:
        noise_loop = _load_noise_loop(args.inject_noise)
        print(f"injecting noise from {args.inject_noise} at gain {args.noise_gain}")
    elif args.synthetic_noise:
        noise_loop = _synthetic_pink_noise()
        print(f"injecting synthetic pink noise at gain {args.noise_gain}")

    def callback(indata, outdata, _frames, _time_info, status):
        if status:
            print(status, file=sys.stderr)
        hop = indata[:, 0].copy()
        if noise_loop is not None:
            pos = state["noise_pos"]
            idx = (np.arange(pos, pos + HOP)) % len(noise_loop)
            hop = hop + args.noise_gain * noise_loop[idx]
            state["noise_pos"] = (pos + HOP) % len(noise_loop)

        t0 = time.perf_counter()
        out_hop = enhancer.process_hop(hop, enabled=state["enabled"])
        state["rtf_samples"].append((time.perf_counter() - t0) / (HOP / SAMPLE_RATE))
        outdata[:, 0] = out_hop

        now = time.time()
        if now - state["last_print"] > 2.0:
            recent = state["rtf_samples"][-200:]
            mean_rtf = float(np.mean(recent)) if recent else 0.0
            mode = "ENHANCED" if state["enabled"] else "BYPASS "
            print(f"[{mode}] rolling RTF: {mean_rtf:.3f}")
            state["last_print"] = now

    def key_listener():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while state["running"]:
                ch = sys.stdin.read(1)
                if ch.lower() == "e":
                    state["enabled"] = not state["enabled"]
                    print(f">>> enhancement {'ON' if state['enabled'] else 'OFF'}")
                elif ch.lower() == "q":
                    state["running"] = False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("live demo running. press 'e' to toggle enhancement, 'q' to quit.")
    listener = threading.Thread(target=key_listener, daemon=True)
    listener.start()

    with sd.Stream(samplerate=SAMPLE_RATE, blocksize=HOP, channels=1, dtype="float32", callback=callback):
        try:
            while state["running"] and listener.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            state["running"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="offline correctness + RTF check, no audio hardware")
    parser.add_argument("--capture-test", type=float, metavar="SECONDS", help="mic -> file for SECONDS, no playback")
    parser.add_argument("--out", default="capture_test.wav", help="output wav for --capture-test")
    parser.add_argument("--inject-noise", metavar="FILE", help="loop this wav additively into the mic signal (live mode)")
    parser.add_argument("--synthetic-noise", action="store_true", help="inject synthetic pink noise (live mode)")
    parser.add_argument("--noise-gain", type=float, default=0.3, help="linear gain for injected noise")
    args = parser.parse_args()

    if args.check:
        cmd_check(args)
    elif args.capture_test:
        cmd_capture_test(args)
    else:
        cmd_live(args)


if __name__ == "__main__":
    main()
