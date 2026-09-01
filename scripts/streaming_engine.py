"""
Frame-by-frame streaming GTCRN inference.

Uses the pre-exported ONNX model at
third_party/gtcrn/stream/onnx_models/gtcrn_simple.onnx (upstream, MIT licence,
exported by the GTCRN authors from checkpoints/model_trained_on_dns3.tar). This
module only calls that model and handles the STFT/overlap-add framing around it
-- third_party/gtcrn is not modified.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO_ROOT / "third_party" / "gtcrn" / "stream" / "onnx_models" / "gtcrn_simple.onnx"

N_FFT = 512
HOP = 256

# Periodic sqrt-Hann window, matching torch.hann_window(512, periodic=True).pow(0.5)
# (the convention used throughout third_party/gtcrn, including training).
_n = np.arange(N_FFT)
WINDOW = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * _n / N_FFT)).astype(np.float32)


class StreamingEnhancer:
    """Carries GTCRN's recurrent/conv state across process_hop() calls."""

    def __init__(self, onnx_path: Path = ONNX_PATH):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.conv_cache = np.zeros((2, 1, 16, 16, 33), dtype=np.float32)
        self.tra_cache = np.zeros((2, 3, 1, 1, 16), dtype=np.float32)
        self.inter_cache = np.zeros((2, 1, 33, 16), dtype=np.float32)
        self.analysis_buf = np.zeros(N_FFT, dtype=np.float32)
        self.synthesis_buf = np.zeros(N_FFT, dtype=np.float32)
        # Per-hop intermediates, kept for introspection (scripts/make_pipeline_animation.py
        # renders them). These are references to arrays this method already builds, so
        # populating them costs nothing and changes no behaviour.
        self.last: dict = {}

    def process_hop(self, hop: np.ndarray, enabled: bool = True) -> np.ndarray:
        """hop: (HOP,) float32. Returns (HOP,) float32.

        The model always runs (state stays warm even while bypassed) so toggling
        `enabled` mid-stream doesn't cause a stale-hidden-state glitch when
        re-enabled -- only the reconstructed spectrum choice depends on the flag.
        """
        if hop.shape != (HOP,):
            raise ValueError(f"expected hop shape ({HOP},), got {hop.shape}")

        self.analysis_buf = np.concatenate([self.analysis_buf[HOP:], hop])
        windowed = self.analysis_buf * WINDOW
        spec = np.fft.rfft(windowed, n=N_FFT)  # (257,) complex64

        mix = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)[None, :, None, :]  # (1,257,1,2)
        enh, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            None,
            {
                "mix": mix,
                "conv_cache": self.conv_cache,
                "tra_cache": self.tra_cache,
                "inter_cache": self.inter_cache,
            },
        )
        enh = enh[0, :, 0, :]  # (257, 2)
        spec_out = (enh[:, 0] + 1j * enh[:, 1]) if enabled else spec

        frame = np.fft.irfft(spec_out, n=N_FFT).astype(np.float32) * WINDOW
        self.synthesis_buf = self.synthesis_buf + frame
        out_hop = self.synthesis_buf[:HOP].copy()
        self.synthesis_buf = np.concatenate([self.synthesis_buf[HOP:], np.zeros(HOP, dtype=np.float32)])
        self.last = {"windowed": windowed, "spec": spec, "spec_out": spec_out,
                     "frame": frame, "out_hop": out_hop}
        return out_hop
