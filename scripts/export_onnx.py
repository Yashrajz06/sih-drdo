"""
Export a (fine-tuned) GTCRN checkpoint to the streaming ONNX format the live demo runs.

This is the bridge between training and deployment: `finetune.py` produces a PyTorch
checkpoint of the *batch* model, but `streaming_engine.py` runs a frame-by-frame ONNX
model with explicit cache tensors. This script does the conversion and -- crucially --
verifies it.

Why it gets its own script with its own verification: the export path is the most
fragile step in the pipeline. It needs opset 11, onnxsim, and a structural conversion
from the batch model to StreamGTCRN. Upstream's own README documents a bug that had to
be fixed in exactly this simplification step (their issue #3). A fine-tuned model that
cannot export is a model that cannot be demoed, so this is tested against the
unmodified pretrained weights *before* fine-tuning finishes -- if the toolchain is
broken we want to know while there's still time to fix it.

Usage:
    python scripts/export_onnx.py --self-test
    python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out models/gtcrn_finetuned.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
GTCRN_DIR = REPO_ROOT / "third_party" / "gtcrn"
STREAM_DIR = GTCRN_DIR / "stream"

CACHE_SHAPES = {
    "conv_cache": (2, 1, 16, 16, 33),
    "tra_cache": (2, 3, 1, 1, 16),
    "inter_cache": (2, 1, 33, 16),
}
N_FRAMES_CHECK = 40
MATCH_TOL = 1e-4


def _import_stream_modules():
    """gtcrn_stream.py does `from modules.convolution import ...`, so the stream dir
    must be on sys.path. stream/gtcrn.py is byte-identical to the top-level one."""
    sys.path.insert(0, str(STREAM_DIR))
    from gtcrn import GTCRN
    from gtcrn_stream import StreamGTCRN
    from modules.convert import convert_to_stream

    return GTCRN, StreamGTCRN, convert_to_stream


def load_state_dict(checkpoint: Path) -> dict:
    """Accepts either a finetune.py checkpoint or an upstream .tar -- both store the
    weights under a 'model' key, but finetune.py's also carries optimizer state."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError(f"{checkpoint} has no 'model' key (found: {list(ckpt)[:5]})")
    # Only our own checkpoints carry optimizer state; upstream .tar files also have an
    # 'epoch' key, so that alone doesn't identify a fine-tuned checkpoint.
    if "optimizer" in ckpt:
        best = ckpt.get("best_val")
        best_str = f"{best:.4f}" if isinstance(best, float) else "n/a"
        print(f"  fine-tuned checkpoint: epoch {ckpt.get('epoch')}, best val loss {best_str}")
    else:
        print(f"  upstream checkpoint (epoch {ckpt.get('epoch', 'n/a')})")
    return ckpt["model"]


def build_stream_model(state_dict: dict):
    GTCRN, StreamGTCRN, convert_to_stream = _import_stream_modules()
    model = GTCRN().eval()
    model.load_state_dict(state_dict)
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, model)  # remaps weights, incl. the deconv rewrite
    return stream_model


def export(stream_model, out_path: Path, simplify: bool = True) -> Path:
    import onnx

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_spec = torch.randn(1, 257, 1, 2)
    caches = [torch.zeros(shape) for shape in CACHE_SHAPES.values()]

    raw_path = out_path.with_name(out_path.stem + "_raw.onnx")
    export_kwargs = dict(
        input_names=["mix", *CACHE_SHAPES],
        output_names=["enh", *(f"{k}_out" for k in CACHE_SHAPES)],
        opset_version=11,
        verbose=False,
    )
    # opset 11 is not cosmetic -- it is a hard performance requirement. torch>=2.9's
    # dynamo exporter refuses opset 11 and silently upgrades to 18, producing a graph
    # with ~30 extra nodes that benchmarks 3.5x slower per hop (measured: 5.41 ms vs
    # 1.54 ms). That is survivable on a laptop but would push RTF past 1.0 on a
    # Raspberry Pi, i.e. it would break real-time operation on the target hardware.
    # dynamo=False selects the legacy TorchScript exporter, which honours opset 11.
    try:
        torch.onnx.export(stream_model, (dummy_spec, *caches), str(raw_path), dynamo=False, **export_kwargs)
    except TypeError:
        # torch too old to know the flag -- it only has the legacy exporter anyway.
        torch.onnx.export(stream_model, (dummy_spec, *caches), str(raw_path), **export_kwargs)
    onnx.checker.check_model(onnx.load(raw_path))
    print(f"  exported raw ONNX -> {raw_path.name}")

    if not simplify:
        raw_path.rename(out_path)
        return out_path

    from onnxsim import simplify as onnxsim_simplify

    simplified, ok = onnxsim_simplify(onnx.load(raw_path))
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified model")
    onnx.save(simplified, out_path)
    raw_path.unlink()
    print(f"  simplified -> {out_path.name}")
    return out_path


def verify(onnx_path: Path, stream_model) -> tuple[bool, float]:
    """Run the same random frames through ONNX and the PyTorch StreamGTCRN and compare.
    Anything above float32 round-off means the export silently changed the model."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    frames = (rng.standard_normal((1, 257, N_FRAMES_CHECK, 2)) * 0.1).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    caches_np = {k: np.zeros(s, np.float32) for k, s in CACHE_SHAPES.items()}
    onnx_out = []
    for i in range(N_FRAMES_CHECK):
        out, caches_np["conv_cache"], caches_np["tra_cache"], caches_np["inter_cache"] = session.run(
            None, {"mix": frames[:, :, i : i + 1, :], **caches_np}
        )
        onnx_out.append(out)
    onnx_out = np.concatenate(onnx_out, axis=2)

    caches_t = [torch.zeros(s) for s in CACHE_SHAPES.values()]
    torch_out = []
    with torch.no_grad():
        for i in range(N_FRAMES_CHECK):
            out, *caches_t = stream_model(torch.from_numpy(frames[:, :, i : i + 1, :]), *caches_t)
            torch_out.append(out.numpy())
    torch_out = np.concatenate(torch_out, axis=2)

    max_diff = float(np.abs(onnx_out - torch_out).max())
    return max_diff <= MATCH_TOL, max_diff


def self_test() -> None:
    """Gate 1.3: prove the export toolchain works on the unmodified pretrained model.

    Two checks. First the exported model must match its own PyTorch source. Second it
    must match the ONNX the GTCRN authors shipped -- if we can reproduce their artifact
    from their weights, our export procedure is equivalent to theirs.
    """
    print("Gate 1.3 -- ONNX export self-test on the unmodified pretrained checkpoint\n")

    # NOTE: upstream ships TWO different files both named model_trained_on_dns3.tar --
    # checkpoints/ holds epoch 87, stream/onnx_models/ holds epoch 96, and their weights
    # genuinely differ (BatchNorm running_var by up to 138). gtcrn_simple.onnx was
    # exported from the stream/onnx_models copy, so that is the one to use here; using
    # the other makes a working export path look broken.
    src = STREAM_DIR / "onnx_models" / "model_trained_on_dns3.tar"
    print(f"source checkpoint: stream/onnx_models/{src.name}")
    state = load_state_dict(src)
    stream_model = build_stream_model(state)

    out = REPO_ROOT / "models" / "gtcrn_selftest.onnx"
    export(stream_model, out)

    ok, diff = verify(out, stream_model)
    print(f"\n  exported vs PyTorch source: max diff {diff:.3e}  ({'ok' if ok else 'FAIL'})")

    # Cross-check against upstream's shipped ONNX.
    import onnxruntime as ort

    reference = STREAM_DIR / "onnx_models" / "gtcrn_simple.onnx"
    rng = np.random.default_rng(1)
    frames = (rng.standard_normal((1, 257, N_FRAMES_CHECK, 2)) * 0.1).astype(np.float32)

    def run(path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        caches = {k: np.zeros(s, np.float32) for k, s in CACHE_SHAPES.items()}
        outs = []
        for i in range(N_FRAMES_CHECK):
            o, caches["conv_cache"], caches["tra_cache"], caches["inter_cache"] = sess.run(
                None, {"mix": frames[:, :, i : i + 1, :], **caches}
            )
            outs.append(o)
        return np.concatenate(outs, axis=2)

    ref_diff = float(np.abs(run(out) - run(reference)).max())
    ref_ok = ref_diff <= MATCH_TOL
    print(f"  ours vs upstream gtcrn_simple.onnx: max diff {ref_diff:.3e}  ({'ok' if ref_ok else 'FAIL'})")

    out.unlink(missing_ok=True)
    if ok and ref_ok:
        print("\nGATE 1.3 PASSED: the export path reproduces upstream's own artifact.")
        print("A fine-tuned checkpoint can be exported and deployed with confidence.")
    else:
        print("\nGATE 1.3 FAILED: do not rely on this path until fixed.", file=sys.stderr)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="verify the export path on the pretrained model")
    p.add_argument("--checkpoint", type=Path, help="checkpoint to export (finetune.py or upstream .tar)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "gtcrn_finetuned.onnx")
    p.add_argument("--no-simplify", action="store_true", help="skip onnxsim")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.checkpoint:
        p.error("give --checkpoint, or --self-test")

    print(f"exporting {args.checkpoint}")
    stream_model = build_stream_model(load_state_dict(args.checkpoint))
    out = export(stream_model, args.out, simplify=not args.no_simplify)

    ok, diff = verify(out, stream_model)
    print(f"\n  verification: max diff {diff:.3e}  ({'ok' if ok else 'FAIL'})")
    if not ok:
        print("Export does not match its PyTorch source -- do not deploy this.", file=sys.stderr)
        sys.exit(1)

    print(f"\nwrote {out}")
    print(f"Run it live with:  python scripts/live_demo.py --onnx {out}")


if __name__ == "__main__":
    main()
