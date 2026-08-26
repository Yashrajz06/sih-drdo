"""
Gate 0.1 -- prove which checkpoint the pre-exported streaming ONNX model actually
carries, instead of trusting the upstream export script's hardcoded path.

Why this needs its own test: the demo runs
third_party/gtcrn/stream/onnx_models/gtcrn_simple.onnx, but every metric we report
(Stage-0 benchmark, the Model-A impulsive table) comes from a PyTorch checkpoint. If
those are different models, the numbers don't describe the thing being demoed.

Method: compare the ONNX session against upstream's own StreamGTCRN PyTorch class
loaded with each candidate checkpoint. Same architecture on both sides, so the weights
are the only variable, and the matching checkpoint should agree to float32 precision.

Two weaker approaches were tried first and rejected, recorded here so nobody repeats them:
  - Comparing ONNX audio output against the *batch* GTCRN model: inconclusive. Both
    checkpoints landed within 1.5x of each other (0.0025 vs 0.0038 mean abs error),
    because the streaming reformulation is not bit-exact w.r.t. the batch model, and
    that conversion error swamps the difference between checkpoints.
  - Comparing raw ONNX initializer tensors against state_dict tensors: inconclusive.
    gtcrn_simple.onnx has been through onnxsim, which folds BatchNorm into conv weights
    (214 initializers vs 271 state_dict tensors), so stored values are transformed.

Usage:
    python scripts/verify_onnx_provenance.py
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAM_DIR = REPO_ROOT / "third_party" / "gtcrn" / "stream"
ONNX_PATH = STREAM_DIR / "onnx_models" / "gtcrn_simple.onnx"
CHECKPOINTS = ("model_trained_on_dns3.tar", "model_trained_on_vctk.tar")

# A true weight match should be float32 round-off only; anything above this means the
# ONNX carries different weights.
MATCH_TOL = 1e-5
# The non-matching checkpoint must be dramatically worse, or the test proves nothing.
MIN_SEPARATION = 1000.0

N_FRAMES = 40
CACHE_SHAPES = {
    "conv_cache": (2, 1, 16, 16, 33),
    "tra_cache": (2, 3, 1, 1, 16),
    "inter_cache": (2, 1, 33, 16),
}


def run_onnx(frames: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    caches = {name: np.zeros(shape, np.float32) for name, shape in CACHE_SHAPES.items()}
    outs = []
    for i in range(frames.shape[2]):
        out, caches["conv_cache"], caches["tra_cache"], caches["inter_cache"] = session.run(
            None, {"mix": frames[:, :, i : i + 1, :], **caches}
        )
        outs.append(out)
    return np.concatenate(outs, axis=2)


def run_stream_torch(frames: np.ndarray, checkpoint: str) -> np.ndarray:
    # gtcrn_stream.py does `from modules.convolution import ...`, so the stream dir must
    # be importable. stream/gtcrn.py is byte-identical to the top-level gtcrn.py, so
    # importing GTCRN from here is the same class the rest of the repo uses.
    sys.path.insert(0, str(STREAM_DIR))
    try:
        from gtcrn import GTCRN
        from gtcrn_stream import StreamGTCRN
        from modules.convert import convert_to_stream

        model = GTCRN().eval()
        model.load_state_dict(torch.load(STREAM_DIR / "onnx_models" / checkpoint, map_location="cpu")["model"])
        stream_model = StreamGTCRN().eval()
        convert_to_stream(stream_model, model)

        caches = [torch.zeros(shape) for shape in CACHE_SHAPES.values()]
        outs = []
        with torch.no_grad():
            for i in range(frames.shape[2]):
                out, *caches = stream_model(torch.from_numpy(frames[:, :, i : i + 1, :]), *caches)
                outs.append(out.numpy())
        return np.concatenate(outs, axis=2)
    finally:
        sys.path.remove(str(STREAM_DIR))


def main() -> None:
    rng = np.random.default_rng(0)
    frames = (rng.standard_normal((1, 257, N_FRAMES, 2)) * 0.1).astype(np.float32)

    print(f"comparing {ONNX_PATH.name} against StreamGTCRN with each checkpoint...\n")
    onnx_out = run_onnx(frames)

    errors = {}
    for ckpt in CHECKPOINTS:
        diff = np.abs(onnx_out - run_stream_torch(frames, ckpt))
        errors[ckpt] = float(diff.max())
        print(f"  {ckpt:30s} max abs diff = {errors[ckpt]:.3e}")

    matched = min(errors, key=errors.get)
    other = next(k for k in errors if k != matched)
    separation = errors[other] / max(errors[matched], 1e-12)

    print(f"\nbest match:  {matched}")
    print(f"separation:  {separation:.1f}x vs {other}")

    failures = []
    if errors[matched] > MATCH_TOL:
        failures.append(
            f"best match {matched} differs by {errors[matched]:.3e}, above tolerance "
            f"{MATCH_TOL:.0e} -- the ONNX may carry weights from neither checkpoint"
        )
    if separation < MIN_SEPARATION:
        failures.append(
            f"checkpoints are only {separation:.1f}x apart (need >{MIN_SEPARATION:.0f}x) "
            "-- this test cannot identify which one the ONNX carries"
        )

    if failures:
        print("\nGATE 0.1 FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\nGATE 0.1 PASSED: the streaming ONNX carries {matched}.")
    print("Report metrics from this checkpoint so the demo and the numbers describe the same model.")


if __name__ == "__main__":
    main()
