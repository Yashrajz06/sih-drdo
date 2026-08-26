"""
Loss functions -- problem-statement deliverable 3: "a training framework with
optimized hyper-parameters and perceptual loss functions."

The PS names "SI-SNR, L1/L2 loss, and perceptual loss". The GTCRN authors' own
HybridLoss (third_party/gtcrn/loss.py) already covers SI-SNR plus compressed
magnitude and complex spectral terms, so it is reused rather than reimplemented.

Two terms are added here that do not exist upstream, and they are the project's
actual technical contribution:

1. **Multi-resolution STFT loss.** A single 512-point STFT smears a gunshot across
   its 32 ms window. Evaluating the error at several window sizes at once means
   short windows can see the transient's sharp onset while long windows still
   constrain the harmonic structure of speech. (Defossez et al., arXiv 2006.12847)

2. **Asymmetric anti-over-suppression loss.** The failure mode we measured in the
   baseline is the model deleting speech that co-occurs with a loud transient.
   Standard MSE punishes "too loud" and "too quiet" equally, so a model can score
   well by suppressing everything. This term penalises only the direction that
   removes speech energy, which is the one that destroys intelligibility.
   (Braun et al., arXiv 2205.06931)

Both operate on power-law-compressed magnitudes (exponent ~0.3), so a single
high-energy event cannot dominate the gradient -- the reason transients otherwise
either get ignored or trigger over-suppression.
"""
import torch
import torch.nn as nn


def _stft_mag(wav: torch.Tensor, n_fft: int, hop: int, win_length: int) -> torch.Tensor:
    window = torch.hann_window(win_length, device=wav.device)
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop, win_length=win_length,
        window=window, return_complex=True, center=True,
    )
    return torch.abs(spec)


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral-convergence + log-magnitude error, summed over several resolutions."""

    def __init__(
        self,
        fft_sizes=(512, 1024, 2048),
        hop_sizes=(50, 120, 240),
        win_lengths=(240, 600, 1200),
        eps: float = 1e-7,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.params = list(zip(fft_sizes, hop_sizes, win_lengths))
        self.eps = eps

    def forward(self, pred_wav: torch.Tensor, true_wav: torch.Tensor) -> torch.Tensor:
        total = pred_wav.new_zeros(())
        for n_fft, hop, win in self.params:
            pred_mag = _stft_mag(pred_wav, n_fft, hop, win)
            true_mag = _stft_mag(true_wav, n_fft, hop, win)
            # Spectral convergence: relative error, scale-insensitive.
            sc = torch.norm(true_mag - pred_mag, p="fro") / (torch.norm(true_mag, p="fro") + self.eps)
            # Log magnitude: weights quiet bins comparably to loud ones.
            mag = nn.functional.l1_loss(
                torch.log(pred_mag + self.eps), torch.log(true_mag + self.eps)
            )
            total = total + sc + mag
        return total / len(self.params)


class AsymmetricLoss(nn.Module):
    """One-sided penalty on removing speech energy.

    L = mean( max(|S|^c - |S_hat|^c, 0)^2 )

    Only the case where the estimate is *quieter* than the target contributes, so
    the gradient pushes back against over-suppression without also rewarding the
    model for leaving residual noise in place (the other loss terms handle that).
    """

    def __init__(self, n_fft: int = 512, hop: int = 256, compress: float = 0.3):
        super().__init__()
        self.n_fft, self.hop, self.compress = n_fft, hop, compress

    def forward(self, pred_wav: torch.Tensor, true_wav: torch.Tensor) -> torch.Tensor:
        pred_mag = _stft_mag(pred_wav, self.n_fft, self.hop, self.n_fft)
        true_mag = _stft_mag(true_wav, self.n_fft, self.hop, self.n_fft)
        pred_c = (pred_mag + 1e-10) ** self.compress
        true_c = (true_mag + 1e-10) ** self.compress
        deficit = torch.clamp(true_c - pred_c, min=0.0)
        return torch.mean(deficit**2)


class HybridLoss(nn.Module):
    """GTCRN's published loss: compressed magnitude + complex + SI-SNR.

    Reimplemented here (rather than imported from third_party/gtcrn/loss.py) only
    so it accepts waveforms like the other terms and shares one STFT convention.
    Weights are the authors' own.
    """

    def __init__(self, n_fft: int = 512, hop: int = 256):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop

    def forward(self, pred_wav: torch.Tensor, true_wav: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=pred_wav.device).pow(0.5)
        pred = torch.stft(pred_wav, self.n_fft, self.hop, self.n_fft, window, return_complex=True)
        true = torch.stft(true_wav, self.n_fft, self.hop, self.n_fft, window, return_complex=True)

        pred_mag, true_mag = torch.abs(pred) + 1e-10, torch.abs(true) + 1e-10
        # Compressed complex representation: preserves phase, tames dynamic range.
        pred_c = pred / pred_mag ** 0.7
        true_c = true / true_mag ** 0.7
        real_loss = nn.functional.mse_loss(pred_c.real, true_c.real)
        imag_loss = nn.functional.mse_loss(pred_c.imag, true_c.imag)
        mag_loss = nn.functional.mse_loss(pred_mag**0.3, true_mag**0.3)

        # SI-SNR on the waveform.
        pred_z = pred_wav - pred_wav.mean(dim=-1, keepdim=True)
        true_z = true_wav - true_wav.mean(dim=-1, keepdim=True)
        alpha = (pred_z * true_z).sum(-1, keepdim=True) / ((true_z**2).sum(-1, keepdim=True) + 1e-8)
        target = alpha * true_z
        noise = pred_z - target
        si_snr = 10 * torch.log10(
            ((target**2).sum(-1) + 1e-8) / ((noise**2).sum(-1) + 1e-8) + 1e-8
        )
        return 30 * (real_loss + imag_loss) + 70 * mag_loss - si_snr.mean()


class CombinedLoss(nn.Module):
    """L = HybridLoss + w_mrstft * MR-STFT + w_asym * Asymmetric

    gamma (w_asym) is the one knob genuinely worth tuning: too low and the model
    keeps over-suppressing speech under transients, too high and it under-cleans.
    docs/solution-design.md suggests starting near 0.1.
    """

    def __init__(self, w_mrstft: float = 1.0, w_asym: float = 0.1):
        super().__init__()
        self.hybrid = HybridLoss()
        self.mrstft = MultiResolutionSTFTLoss()
        self.asym = AsymmetricLoss()
        self.w_mrstft, self.w_asym = w_mrstft, w_asym

    def forward(self, pred_wav: torch.Tensor, true_wav: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = self.hybrid(pred_wav, true_wav)
        m = self.mrstft(pred_wav, true_wav)
        a = self.asym(pred_wav, true_wav)
        total = h + self.w_mrstft * m + self.w_asym * a
        return total, {
            "total": float(total.detach()),
            "hybrid": float(h.detach()),
            "mrstft": float(m.detach()),
            "asym": float(a.detach()),
        }


if __name__ == "__main__":
    # Gate 1.1 self-test: every term must be finite, and must score a deliberately
    # *better* prediction lower than a worse one. A loss that fails this is worse
    # than useless -- it would train the model in the wrong direction silently.
    torch.manual_seed(0)
    clean = torch.randn(2, 16000) * 0.1
    noise = torch.randn(2, 16000) * 0.1

    good = clean + 0.1 * noise   # close to target
    bad = clean + 1.0 * noise    # far from target
    suppressed = clean * 0.2     # over-suppressed: the failure we care about

    terms = {
        "HybridLoss": HybridLoss(),
        "MultiResolutionSTFT": MultiResolutionSTFTLoss(),
        "Asymmetric": AsymmetricLoss(),
    }

    print(f"{'loss':22s} {'good':>10s} {'bad':>10s} {'ordering':>10s}")
    ok = True
    for name, fn in terms.items():
        g, b = float(fn(good, clean)), float(fn(bad, clean))
        finite = all(map(lambda v: v == v and abs(v) != float("inf"), (g, b)))
        passed = finite and g < b
        ok &= passed
        print(f"{name:22s} {g:10.4f} {b:10.4f} {'ok' if passed else 'FAIL':>10s}")

    # The asymmetric term specifically must punish over-suppression harder than
    # an equally-wrong prediction that is too loud.
    asym = AsymmetricLoss()
    quiet = float(asym(suppressed, clean))
    loud = float(asym(clean * 1.8, clean))
    directional = quiet > loud
    ok &= directional
    print(f"\nasymmetry check: over-suppressed {quiet:.4f} vs too-loud {loud:.4f} -> "
          f"{'ok (penalises suppression more)' if directional else 'FAIL'}")

    combined = CombinedLoss()
    _, parts = combined(good, clean)
    print(f"\ncombined loss parts: {parts}")
    print("\nGATE 1.1 (losses) PASSED" if ok else "\nGATE 1.1 (losses) FAILED")
    raise SystemExit(0 if ok else 1)
