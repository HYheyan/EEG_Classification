"""Inference for EEG binary classification.

Usage:
  python test.py                           # models/best_model.pth
  python test.py --ensemble                # ensemble all fold models
  python test.py --use-subject             # if trained with --use-subject
  python test.py --model path/to/ckpt      # specific checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    "train_label_csv": DATA_ROOT / "train_labels.csv",
    "model_path": MODEL_DIR / "best_model.pth",
    "output_csv": RES_DIR / "predictions.csv",
    "batch_size": 64,
    "num_workers": 0,
}

# Subject order (must match training)
SUBJECT_ORDER = ["sub01", "sub02", "sub03", "sub04", "sub05"]


def extract_subject(filename: str) -> int:
    """Extract subject index from filename like 'sub02_sess3_epoch004.npy'."""
    for i, s in enumerate(SUBJECT_ORDER):
        if filename.startswith(s):
            return i
    return 0


def load_model(path: Path, device: torch.device,
               num_subjects: int = 0) -> torch.nn.Module:
    model = EEGNet(
        input_shape=(1, 59, 282), num_subjects=num_subjects,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_single(model, loader, device, use_subject=False) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for batch in loader:
            if use_subject:
                eeg, files, subj = batch
                subj = subj.to(device)
            else:
                eeg, files = batch
                subj = None
            logits = model(eeg.to(device), subject_idx=subj)
            preds = logits.argmax(dim=1).cpu().tolist()
            for f, p in zip(files, preds):
                rows.append({"eeg_file": f,
                             "prediction": "background" if p == 0 else "target"})
    return pd.DataFrame(rows)


def predict_ensemble(models, loader, device, use_subject=False) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for batch in loader:
            if use_subject:
                eeg, files, subj = batch
                subj = subj.to(device)
            else:
                eeg, files = batch
                subj = None
            eeg = eeg.to(device)
            probs = torch.stack([
                F.softmax(m(eeg, subject_idx=subj), dim=1) for m in models
            ]).mean(dim=0)
            preds = probs.argmax(dim=1).cpu().tolist()
            for f, p in zip(files, preds):
                rows.append({"eeg_file": f,
                             "prediction": "background" if p == 0 else "target"})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="EEG inference")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--ema", action="store_true", help="Use EMA/SWA models")
    p.add_argument("--pattern", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--use-subject", action="store_true",
                   help="Enable subject embedding (must match training)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(args.cpu)
    num_subjects = len(SUBJECT_ORDER) if args.use_subject else 0
    print(f"Device: {device} | Subject-embed: {args.use_subject}")

    # Build subject mapping from filenames (since test has no label CSV)
    test_ds = EEGDataset(
        data_dir=CONFIG["test_dir"], label_csv=None,
        return_subject=args.use_subject,
    )
    if args.use_subject:
        # Map subjects from filenames for test data
        test_ds.has_subject = True
        test_ds.subject_to_idx = {s: i for i, s in enumerate(SUBJECT_ORDER)}
        # Override __getitem__ behavior: parse subject from filename
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

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else CONFIG["output_csv"]

    if args.ensemble:
        suffix = "_ema.pth" if args.ema else "_best.pth"
        pattern = args.pattern or f"eegnet_fold*{suffix}"
        files = sorted(MODEL_DIR.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No models match '{pattern}'")
        print(f"Ensemble: {len(files)} models")
        for f in files:
            print(f"  {f.name}")
        models = [load_model(f, device, num_subjects) for f in files]
        df = predict_ensemble(models, test_loader, device, args.use_subject)
    else:
        path = Path(args.model) if args.model else CONFIG["model_path"]
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        print(f"Model: {path}")
        model = load_model(path, device, num_subjects)
        df = predict_single(model, test_loader, device, args.use_subject)

    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")
    print(df["prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()
