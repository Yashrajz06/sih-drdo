"""
Fine-tuning loop -- the training half of problem-statement deliverables 2 and 3.

Starts from the pretrained model_trained_on_dns3.tar checkpoint rather than random
weights. That checkpoint already reaches PESQ 2.855 on the standard benchmark; the
job here is to close the ~0.2 PESQ gap that remains on real defence noise (measured
baseline: 2.33 at +15 dB against the PS target of 2.5), not to relearn speech
enhancement from scratch. Fine-tuning is also far more realistic on free GPU time.

Built for Colab/Kaggle specifically:
  - checkpoints every epoch, because free sessions disconnect without warning
  - --resume picks up exactly where it stopped, including optimizer state
  - --device auto-detects CUDA, so the same file runs on a laptop CPU for testing

Usage:
    python scripts/finetune.py --smoke-test                  # tiny CPU run, no GPU
    python scripts/finetune.py --epochs 30 --batch-size 16   # real run
    python scripts/finetune.py --resume checkpoints/last.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
GTCRN_DIR = REPO_ROOT / "third_party" / "gtcrn"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(GTCRN_DIR))

from gtcrn import GTCRN  # noqa: E402
from losses import CombinedLoss  # noqa: E402
from train_dataset import NoisySpeechDataset, find_clean_files  # noqa: E402

SAMPLE_RATE = 16000
N_FFT, HOP = 512, 256


def enhance_batch(model: torch.nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    """Waveform -> STFT -> model -> waveform, batched, differentiable."""
    window = torch.hann_window(N_FFT, device=noisy.device).pow(0.5)
    spec = torch.stft(noisy, N_FFT, HOP, N_FFT, window, return_complex=True)
    out = model(torch.view_as_real(spec))
    out_c = torch.view_as_complex(out.contiguous())
    return torch.istft(out_c, N_FFT, HOP, N_FFT, window, length=noisy.shape[-1])


def load_pretrained(device: torch.device, checkpoint: str = "model_trained_on_dns3.tar") -> GTCRN:
    model = GTCRN().to(device)
    ckpt = torch.load(GTCRN_DIR / "checkpoints" / checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def build_loaders(args) -> tuple[DataLoader, DataLoader]:
    clean_files = find_clean_files(args.clean_dir)
    if not clean_files:
        sys.exit(
            f"No clean speech under {args.clean_dir}.\n"
            "Download LibriSpeech dev-clean and extract it there."
        )
    # Speaker-disjoint-ish split: LibriSpeech paths are .../<speaker>/<chapter>/x.flac,
    # so splitting on the sorted file list keeps whole chapters together well enough
    # for a validation signal. The real evaluation is eval_impulsive_noise.py on the
    # untouched MAD test split -- this is only for tracking training progress.
    split = int(len(clean_files) * 0.95)
    train_files, val_files = clean_files[:split], clean_files[split:]
    print(f"clean speech: {len(train_files)} train / {len(val_files)} val files")

    train_ds = NoisySpeechDataset(
        clean_files=train_files,
        split="training",
        segment_seconds=args.segment_seconds,
        snr_range=(args.snr_min, args.snr_max),
        impulsive_weight=args.impulsive_weight,
        epoch_size=args.epoch_size,
        seed=args.seed,
    )
    val_ds = NoisySpeechDataset(
        clean_files=val_files or train_files[-50:],
        split="training",
        segment_seconds=args.segment_seconds,
        snr_range=(args.snr_min, args.snr_max),
        impulsive_weight=args.impulsive_weight,
        epoch_size=max(32, args.epoch_size // 20),
        seed=args.seed + 777,
    )
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False),
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool, max_batches=None, log_every=20):
    model.train(train)
    totals, n = {}, 0
    t0 = time.perf_counter()
    for i, (noisy, clean) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        noisy, clean = noisy.to(device), clean.to(device)

        with torch.set_grad_enabled(train):
            enhanced = enhance_batch(model, noisy)
            loss, parts = criterion(enhanced, clean)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clipping matters here: a loud transient can otherwise produce a
            # gradient spike that undoes a whole epoch of progress.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        if train and log_every and i % log_every == 0:
            print(f"    batch {i:4d}  loss {parts['total']:8.3f}  "
                  f"(hybrid {parts['hybrid']:7.3f}  mrstft {parts['mrstft']:.3f}  asym {parts['asym']:.4f})")

    elapsed = time.perf_counter() - t0
    return ({k: v / max(n, 1) for k, v in totals.items()}, elapsed)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-dir", type=Path, default=REPO_ROOT / "data" / "librispeech")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epoch-size", type=int, default=2000, help="mixtures generated per epoch")
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--impulsive-weight", type=float, default=0.5)
    p.add_argument("--w-mrstft", type=float, default=1.0)
    p.add_argument("--w-asym", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--smoke-test", action="store_true", help="tiny CPU run to verify the loop works")
    args = p.parse_args()

    if args.smoke_test:
        args.epochs, args.batch_size, args.epoch_size = 2, 2, 8
        args.workers, args.device = 0, "cpu"

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_loaders(args)
    model = load_pretrained(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"model: GTCRN, {n_params:,} parameters (from pretrained dns3 checkpoint)")

    criterion = CombinedLoss(w_mrstft=args.w_mrstft, w_asym=args.w_asym).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val, history = 0, float("inf"), []
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_val, history = ck["epoch"] + 1, ck.get("best_val", float("inf")), ck.get("history", [])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_loader.dataset.set_epoch(epoch)  # fresh mixtures every epoch
        print(f"\nepoch {epoch + 1}/{args.epochs}  (lr {scheduler.get_last_lr()[0]:.2e})")

        train_stats, train_time = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_stats, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False, log_every=0)
        scheduler.step()

        print(f"  train {train_stats['total']:8.3f}   val {val_stats['total']:8.3f}   ({train_time:.1f}s)")
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "history": history,
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        # Always write last.pt -- a disconnected Colab session must be resumable.
        torch.save(state, args.out_dir / "last.pt")
        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            state["best_val"] = best_val
            torch.save(state, args.out_dir / "best.pt")
            print(f"  new best (val {best_val:.3f}) -> {args.out_dir / 'best.pt'}")

        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\ndone. best val loss {best_val:.3f}")
    print(f"checkpoints in {args.out_dir}")
    if args.smoke_test:
        # Gate 1.1: the checkpoint must actually load back into a fresh model.
        fresh = GTCRN()
        fresh.load_state_dict(torch.load(args.out_dir / "last.pt", map_location="cpu")["model"])
        print("\nGATE 1.1 (training loop) PASSED: 2 epochs ran and the checkpoint reloads.")
    else:
        print("Next: export to ONNX, then re-run scripts/eval_impulsive_noise.py to fill the Model C row.")


if __name__ == "__main__":
    main()
