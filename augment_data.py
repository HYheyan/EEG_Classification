"""One-time script: generate offline augmented copies of training data.

Expands the training set by applying random combinations of EEG-specific
augmentations to each sample.  The augmented cache can then be used with
``python train.py --augmented``.

Usage:
  python augment_data.py                    # default 3× multiplier
  python augment_data.py --multiplier 5     # 5 copies per sample
  python augment_data.py --seed 42 --multiplier 4

Produces:  data/train_cache_augmented.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
BATCH_PRINT = 100  # print progress every N samples


# ---------------------------------------------------------------------------
# Augmentation functions (all operate on (C, H, W) = (1, 59, T) torch float32)
# ---------------------------------------------------------------------------

def _gaussian_noise(eeg: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Additive white Gaussian noise.  σ drawn uniformly."""
    sigma = 0.01 + rng.random() * 0.11  # [0.01, 0.12]
    noise = torch.randn_like(eeg) * sigma
    return eeg + noise


def _time_shift(eeg: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Circular roll along the time axis."""
    shift = rng.integers(-25, 26)  # [-25, 25]
    if shift != 0:
        eeg = torch.roll(eeg, int(shift), dims=-1)
    return eeg


def _amplitude_scale(eeg: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Scale all amplitudes by a random factor."""
    factor = 0.7 + rng.random() * 0.6  # [0.7, 1.3]
    return eeg * factor


# ---------------------------------------------------------------------------
# NOTE: Only low-risk augmentations are included.
#
#   Safe (keep):
#     - Gaussian noise: EEG always has sensor noise; small σ preserves signal
#     - Time shift: neural activity has natural jitter; circular shift ~±9%
#     - Amplitude scale: amplitude varies across subjects/sessions naturally
#
#   Removed (high risk — can destroy label-relevant information):
#     - Channel dropout: zeros 8-22% electrodes → breaks spatial topology
#     - Time mask: may erase critical ERP components → label mismatch
#     - Time warp: distorts timing of key features + spline artifacts
#     - Freq mask: may zero the discriminant frequency band
#     - Channel noise: correlated neighbour noise alters spatial patterns
# ---------------------------------------------------------------------------

AUG_POOL = [
    _gaussian_noise,
    _time_shift,
    _amplitude_scale,
]


def generate_augmented_copy(eeg: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Apply each safe augmentation independently with ~60% probability.

    This means a copy may get 0–3 augmentations, averaging ~1.8 per copy.
    The randomness ensures diverse combinations across samples.
    """
    result = eeg.clone()
    for aug_fn in AUG_POOL:
        if rng.random() < 0.6:
            result = aug_fn(result, rng)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate offline augmented training cache for EEG classification")
    p.add_argument("--multiplier", "-m", type=int, default=3,
                   help="Number of augmented copies per original sample (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--input", type=str, default=None,
                   help="Input cache path (default: data/train_cache.pt)")
    p.add_argument("--output", type=str, default=None,
                   help="Output cache path (default: data/train_cache_augmented.pt)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    in_path = Path(args.input) if args.input else DATA_ROOT / "train_cache.pt"
    out_path = Path(args.output) if args.output else DATA_ROOT / "train_cache_augmented.pt"

    # ---- Load original cache -----------------------------------------------
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input cache not found: {in_path}\n"
            f"  Run `python convert_data.py` first to generate it."
        )

    archive = torch.load(in_path, map_location="cpu", weights_only=True)
    eeg_orig = archive["eeg"]         # (N, 1, 59, T)
    labels = archive["label"]          # (N,)
    files = archive["eeg_file"]        # list of str
    subjects = archive.get("subject", None)
    subj_to_idx = archive.get("subject_to_idx", {})

    N = len(eeg_orig)
    M = args.multiplier
    print(f"Original samples : {N}")
    print(f"Multiplier       : {M}x")
    print(f"Output samples   : {N} x (1 + {M}) = {N * (1 + M)}")
    print(f"Augmentations    : {len(AUG_POOL)} types, each applied with 60% probability per copy\n")

    # ---- Generate augmented copies -----------------------------------------
    new_eeg_list = [eeg_orig]
    new_labels_list = [labels]
    new_files_list: list[str] = list(files)
    new_subjects_list = [subjects] if subjects is not None else None

    for i in range(N):
        original = eeg_orig[i]
        base_fname = Path(files[i]).stem  # e.g. "sub01_sess3_epoch002"

        for k in range(M):
            aug = generate_augmented_copy(original, rng)
            new_eeg_list.append(aug.unsqueeze(0))
            new_labels_list.append(labels[i:i + 1])
            new_files_list.append(f"{base_fname}_aug{k}.npy")

            if new_subjects_list is not None:
                new_subjects_list.append(subjects[i:i + 1])

        if (i + 1) % BATCH_PRINT == 0:
            print(f"  {i + 1}/{N} samples processed...")

    # ---- Concatenate & save ------------------------------------------------
    print(f"\nConcatenating tensors...")
    data = {
        "eeg": torch.cat(new_eeg_list, dim=0),             # (N*(1+M), 1, 59, T)
        "label": torch.cat(new_labels_list, dim=0),         # (N*(1+M),)
        "eeg_file": new_files_list,                         # list of str
        "multiplier": M,                                     # so downstream knows the expansion factor
    }
    if new_subjects_list is not None:
        data["subject"] = torch.cat(new_subjects_list, dim=0)
    if subj_to_idx:
        data["subject_to_idx"] = subj_to_idx

    torch.save(data, out_path)
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(data['eeg'])} samples → {out_path}  ({mb:.0f} MB)")

    # ---- Quick summary -----------------------------------------------------
    n_bg = (data["label"] == 0).sum().item()
    n_tg = (data["label"] == 1).sum().item()
    print(f"Labels — background: {n_bg}  target: {n_tg}  "
          f"(ratio {n_bg / len(data['label']):.2f})")
    print("Done.")


if __name__ == "__main__":
    main()
