"""
Classical spectral subtraction -- the pre-neural baseline.

docs/solution-design.md SS2 asks for a "we beat the classical DSP method" comparison,
and asserting it isn't the same as measuring it. This is the standard textbook method
(Boll, "Suppression of acoustic noise in speech using spectral subtraction," IEEE
TASSP 27(2), 1979) with the two refinements that make it a fair rather than a straw-man
opponent: over-subtraction and a spectral floor, both of which reduce the musical-noise
artifact the naive version is notorious for.

It is deliberately causal and uses only a leading noise estimate, matching the
constraints the neural model operates under -- a non-causal baseline that got to see
the whole file first would not be a like-for-like comparison.
"""
import numpy as np

N_FFT = 512
HOP = 256
_n = np.arange(N_FFT)
WINDOW = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * _n / N_FFT)).astype(np.float32)


def spectral_subtraction(
    mix: np.ndarray,
    sample_rate: int = 16000,
    noise_init_s: float = 0.25,
    alpha: float = 2.0,
    beta: float = 0.02,
    smoothing: float = 0.98,
) -> np.ndarray:
    """
    alpha:     over-subtraction factor (>1 removes more noise, risks speech distortion)
    beta:      spectral floor as a fraction of the noise estimate; prevents the
               deep spectral nulls that cause musical noise
    smoothing: how slowly the noise estimate adapts during non-speech frames
    """
    n_hops = len(mix) // HOP
    if n_hops == 0:
        return np.zeros(0, dtype=np.float32)

    analysis = np.zeros(N_FFT, dtype=np.float32)
    frames = []
    for i in range(n_hops):
        analysis = np.concatenate([analysis[HOP:], mix[i * HOP : (i + 1) * HOP]])
        frames.append(np.fft.rfft(analysis * WINDOW, n=N_FFT))
    spec = np.stack(frames, axis=1)  # (bins, frames)

    mag, phase = np.abs(spec), np.angle(spec)

    # Initial noise estimate from the leading frames, then updated on frames that look
    # noise-dominated. Causal: never uses future frames.
    n_init = max(1, int(noise_init_s * sample_rate / HOP))
    noise_mag = mag[:, :n_init].mean(axis=1)

    out_mag = np.zeros_like(mag)
    for t in range(mag.shape[1]):
        frame = mag[:, t]
        subtracted = frame - alpha * noise_mag
        floor = beta * noise_mag
        out_mag[:, t] = np.maximum(subtracted, floor)

        # Update the noise estimate only when this frame is close to the current noise
        # level -- a crude but standard stand-in for a VAD.
        if frame.sum() < 1.5 * noise_mag.sum():
            noise_mag = smoothing * noise_mag + (1 - smoothing) * frame

    out_spec = out_mag * np.exp(1j * phase)  # noisy phase retained (the classic limitation)

    synth = np.zeros(N_FFT, dtype=np.float32)
    out = np.zeros(n_hops * HOP, dtype=np.float32)
    for t in range(out_spec.shape[1]):
        frame = np.fft.irfft(out_spec[:, t], n=N_FFT).astype(np.float32) * WINDOW
        synth = synth + frame
        out[t * HOP : (t + 1) * HOP] = synth[:HOP]
        synth = np.concatenate([synth[HOP:], np.zeros(HOP, dtype=np.float32)])
    return out
