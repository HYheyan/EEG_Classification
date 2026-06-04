"""
EEG binary classification — ensemble inference with TTA.

Discovers all checkpoints in ``models/``, loads each model in turn,
performs 7× test-time augmentation (TTA) per sample, accumulates
softmax probabilities, and produces the final prediction via soft
voting (arithmetic mean of probabilities across all models).

Output: ``res/predictions.csv``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_data import EEGDataset
from model import create_model
from utils import get_device

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RES_DIR = PROJECT_ROOT / "res"

CONFIG = {
    "test_dir": DATA_ROOT / "test",
    "model_dir": MODEL_DIR,
    "output_csv": RES_DIR / "predictions.csv",
    "batch_size": 64,
    "num_workers": 0,
    "use_cpu": False,
    # TTA
    "tta_shifts": (-10, -5, 5, 10),  # time-point circular shifts
    "tta_noise_std": 0.01,            # relative to signal std
    "tta_amp_range": (0.95, 1.05),    # amplitude jitter
}


# =============================================================================
# TTA transforms
# =============================================================================

@torch.no_grad()
def tta_forward(model: torch.nn.Module, eeg: torch.Tensor) -> torch.Tensor:
    """
    Return averaged logits across 7 TTA views:
      1) original
      2-5) 4 circular time shifts
      6) Gaussian noise
      7) amplitude jitter
    """
    logits = model(eeg)  # original

    # (b) circular time shifts
    for shift in CONFIG["tta_shifts"]:
        shifted = torch.roll(eeg, shifts=shift, dims=-1)
        logits = logits + model(shifted)

    # (c) Gaussian noise (independent per call)
    std = eeg.std(dim=(-2, -1), keepdim=True)
    noisy = eeg + CONFIG["tta_noise_std"] * std * torch.randn_like(eeg)
    logits = logits + model(noisy)

    # (d) amplitude jitter
    lo, hi = CONFIG["tta_amp_range"]
    amp = torch.empty(eeg.size(0), 1, 1, 1, device=eeg.device).uniform_(lo, hi)
    logits = logits + model(eeg * amp)

    return logits / 7.0  # average over TTA passes


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = get_device(CONFIG["use_cpu"])
    print(f"Using device: {device}")

    # ---- test dataset --------------------------------------------------
    test_dataset = EEGDataset(data_dir=CONFIG["test_dir"], label_csv=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- discover checkpoints ------------------------------------------
    best_ckpts = sorted(CONFIG["model_dir"].glob("*_fold*_best.pth"))
    ema_ckpts = sorted(CONFIG["model_dir"].glob("*_fold*_ema.pth"))
    all_ckpts = best_ckpts + ema_ckpts

    if not all_ckpts:
        raise FileNotFoundError(
            f"No checkpoints found in {CONFIG['model_dir']}. "
            "Run train.py first."
        )

    print(f"Found {len(best_ckpts)} best + {len(ema_ckpts)} EMA = "
          f"{len(all_ckpts)} checkpoints")

    # ---- ensemble inference --------------------------------------------
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulate weighted probability sums.  Each model's vote is weighted
    # by its best validation accuracy so stronger models influence the
    # final prediction more.
    prob_sum = None   # (N_test, 2) on CPU, weighted sum
    weight_total = 0.0
    eeg_files_all = None
    model_entries: list = []  # per-model record for summary

    for ckpt_path in all_ckpts:
        ckpt_name = ckpt_path.stem  # e.g. "eegnet_fold0_best"
        print(f"  Processing: {ckpt_name}")

        # Parse architecture from filename
        arch = ckpt_name.split("_fold")[0]
        if arch not in ("eegnet", "conformer", "enhanced"):
            print(f"    → unknown architecture '{arch}', skipping")
            continue

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # ---- model weight from validation accuracy ------------------
        val_acc = checkpoint.get("best_val_acc", 0.5)
        weight = val_acc  # higher acc → stronger vote

        # Determine input shape
        input_shape = checkpoint.get("input_shape", (1, 59, 282))

        # Instantiate and load weights
        model = create_model(arch, input_shape=input_shape, num_classes=2).to(device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Bare state_dict (legacy)
            model.load_state_dict(checkpoint)
        model.eval()

        # Inference over test set
        model_probs = []  # list of probability tensors per batch
        batch_files = []

        for eeg, eeg_files in test_loader:
            eeg = eeg.to(device)
            logits = tta_forward(model, eeg)
            probs = F.softmax(logits, dim=1).cpu()
            model_probs.append(probs)
            batch_files.append(eeg_files)

        del model
        torch.cuda.empty_cache()

        # Concatenate this model's probabilities
        model_prob_cat = torch.cat(model_probs, dim=0)  # (N_test, 2)

        if prob_sum is None:
            prob_sum = weight * model_prob_cat
            eeg_files_all = list(batch_files[0])
            for bf in batch_files[1:]:
                eeg_files_all.extend(list(bf))
        else:
            prob_sum = prob_sum + weight * model_prob_cat

        weight_total += weight
        model_entries.append((ckpt_name, arch, val_acc, weight))
        print(f"    → val_acc={val_acc:.4f}  weight={weight:.4f}  "
              f"(total weight: {weight_total:.4f})")

    # ---- final predictions ---------------------------------------------
    avg_probs = prob_sum / weight_total
    pred_indices = avg_probs.argmax(dim=1).tolist()

    predictions = []
    for fname, idx in zip(eeg_files_all, pred_indices):
        label = "background" if idx == 0 else "target"
        predictions.append({"eeg_file": fname, "prediction": label})

    pd.DataFrame(predictions).to_csv(CONFIG["output_csv"], index=False)

    # ---- summary -------------------------------------------------------
    n_bg = sum(1 for p in predictions if p["prediction"] == "background")
    n_tg = sum(1 for p in predictions if p["prediction"] == "target")
    print(f"\n{'=' * 55}")
    print(f"Ensemble complete.  {len(model_entries)} models, "
          f"{len(predictions)} samples.")
    print(f"  background: {n_bg}  ({n_bg / len(predictions):.1%})")
    print(f"  target:     {n_tg}  ({n_tg / len(predictions):.1%})")
    print(f"  Predictions → {CONFIG['output_csv']}")
    print(f"{'=' * 55}")
    # Per-architecture summary
    arch_weights: dict = {}
    for _, a, _, w in model_entries:
        arch_weights[a] = arch_weights.get(a, 0.0) + w
    print("  Vote share by architecture:")
    for a in sorted(arch_weights):
        pct = arch_weights[a] / weight_total * 100
        print(f"    {a:>12s}: {pct:.1f}%")


if __name__ == "__main__":
    main()
