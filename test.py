"""Inference for EEG binary classification with weighted ensemble + TTA.

Usage:
  python test.py                           # ensemble fold_0..fold_4
  python test.py --model path/to/ckpt      # single model
  python test.py --ema                      # use SWA checkpoints
  python test.py --no-tta                   # disable TTA
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_data import EEGDataset
from model import EEGNet
from utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RES_DIR = PROJECT_ROOT / "res"

CONFIG = {
    "test_dir": DATA_ROOT / "test",
    "model_path": MODEL_DIR / "best_model.pth",
    "output_csv": RES_DIR / "predictions.csv",
    "batch_size": 128,
    "num_workers": 0,
}

# Fold weights from 5-fold CV val_acc
FOLD_WEIGHTS = [0.7257, 0.7227, 0.7257, 0.7198, 0.7130]
FOLD_WEIGHTS = np.array(FOLD_WEIGHTS, dtype=np.float32)
FOLD_WEIGHTS = FOLD_WEIGHTS / FOLD_WEIGHTS.sum()

# Subject order (must match training)
SUBJECT_ORDER = ["sub01", "sub02", "sub03", "sub04", "sub05"]

# TTA variants
TTA_SHIFT_PCT = 0.01   # ±1% of time steps
TTA_SCALE = 0.01       # ±1% amplitude


def extract_subject(filename: str) -> int:
    for i, s in enumerate(SUBJECT_ORDER):
        if filename.startswith(s):
            return i
    return 0


def load_model(path: Path, device: torch.device,
               num_subjects: int = 0) -> torch.nn.Module:
    model = EEGNet(
        input_shape=(1, 59, 282), num_subjects=num_subjects,
    ).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def apply_tta_batch(x: torch.Tensor, t_shift: int) -> torch.Tensor:
    """Create 5 TTA variants for a batch.

    Input:  (B, 1, 59, T)
    Output: (5*B, 1, 59, T) — [original, shift+, shift-, scale-, scale+]
    """
    variants = [x]  # original

    # Time shift +1%
    variants.append(torch.roll(x, t_shift, dims=-1))
    # Time shift -1%
    variants.append(torch.roll(x, -t_shift, dims=-1))
    # Scale ×0.99
    variants.append(x * 0.99)
    # Scale ×1.01
    variants.append(x * 1.01)

    return torch.cat(variants, dim=0)  # (5B, 1, 59, T)


def predict_ensemble_weighted(
    models: list[torch.nn.Module],
    weights: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = True,
) -> pd.DataFrame:
    """Weighted ensemble with optional TTA.

    For each sample:
      prob = sum_k( weight_k * mean_tta( softmax(model_k(tta(x))) ) )
    """
    T = 282
    t_shift = max(1, round(T * TTA_SHIFT_PCT))

    rows = []
    with torch.no_grad():
        for batch in loader:
            eeg, files = batch[0], batch[1]
            B = eeg.size(0)

            # Build TTA batch: (5B, 1, 59, T) or just (B, ...) if no TTA
            x_input = apply_tta_batch(eeg, t_shift).to(device) if use_tta else eeg.to(device)
            n_tta = 5 if use_tta else 1

            # Collect weighted probabilities
            weighted_sum = torch.zeros(B, 2, device=device)

            for model, w in zip(models, weights):
                logits = model(x_input)          # (n_tta*B, 2)
                probs = F.softmax(logits, dim=1)  # (n_tta*B, 2)
                probs = probs.view(n_tta, B, 2).mean(dim=0)  # (B, 2)
                weighted_sum += w * probs

            # Final prediction
            confidence, pred = weighted_sum.max(dim=1)
            prob_bg = weighted_sum[:, 0].cpu().numpy()
            prob_tg = weighted_sum[:, 1].cpu().numpy()

            for i, f in enumerate(files):
                rows.append({
                    "eeg_file": f,
                    "prediction": "background" if pred[i].item() == 0 else "target",
                    "confidence": round(confidence[i].item(), 6),
                    "prob_background": round(prob_bg[i].item(), 6),
                    "prob_target": round(prob_tg[i].item(), 6),
                })

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="EEG inference (weighted ensemble + TTA)")
    p.add_argument("--model", type=str, default=None,
                   help="Single model path (overrides ensemble)")
    p.add_argument("--ema", action="store_true",
                   help="Use SWA checkpoints (best_ema.pth)")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time augmentation")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--use-subject", action="store_true",
                   help="Enable subject embedding (must match training)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(args.cpu)
    num_subjects = len(SUBJECT_ORDER) if args.use_subject else 0
    use_tta = not args.no_tta
    print(f"Device: {device} | TTA: {use_tta} | Subject-embed: {args.use_subject}")

    # ---- Load models -------------------------------------------------------
    if args.model:
        paths = [Path(args.model)]
        weights = np.array([1.0], dtype=np.float32)
    else:
        fname = "best_ema.pth" if args.ema else "best_model.pth"
        paths = sorted(MODEL_DIR.glob(f"eegnet_fold*/{fname}"))
        if not paths:
            raise FileNotFoundError(
                f"No models match eegnet_fold*/{fname}\n"
                f"  Run training first: python train.py"
            )
        if len(paths) != len(FOLD_WEIGHTS):
            print(f"  Warning: {len(paths)} models found, "
                  f"{len(FOLD_WEIGHTS)} weights. Using first {min(len(paths), len(FOLD_WEIGHTS))}.")
            weights = FOLD_WEIGHTS[:len(paths)]
        else:
            weights = FOLD_WEIGHTS

    print(f"Models: {len(paths)}")
    for p, w in zip(paths, weights):
        print(f"  {p.name}  (weight={w:.4f})")

    models = [load_model(p, device, num_subjects) for p in paths]

    # ---- Dataset -----------------------------------------------------------
    test_ds = EEGDataset(
        data_dir=CONFIG["test_dir"], label_csv=None,
        return_subject=args.use_subject,
    )
    if args.use_subject:
        test_ds.has_subject = True
        test_ds.subject_to_idx = {s: i for i, s in enumerate(SUBJECT_ORDER)}
        orig_getitem = test_ds.__getitem__
        def new_getitem(idx):
            result = orig_getitem(idx)
            if len(result) == 2:
                eeg, fname = result
                return eeg, fname, extract_subject(fname)
            return result
        test_ds.__getitem__ = new_getitem

    test_loader = DataLoader(
        test_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Test samples: {len(test_ds)}")

    # ---- Predict -----------------------------------------------------------
    df = predict_ensemble_weighted(
        models, weights, test_loader, device, use_tta=use_tta)

    # ---- Save --------------------------------------------------------------
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else CONFIG["output_csv"]
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")
    print(df["prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()
