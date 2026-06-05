"""Inference for EEG binary classification.

Usage:
  python test.py                           # models/best_model.pth
  python test.py --ensemble                # ensemble all fold models
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
    "model_path": MODEL_DIR / "best_model.pth",
    "output_csv": RES_DIR / "predictions.csv",
    "batch_size": 64,
    "num_workers": 0,
}


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = EEGNet(input_shape=(1, 59, 282)).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_single(model, loader, device) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for eeg, files in loader:
            preds = model(eeg.to(device)).argmax(dim=1).cpu().tolist()
            for f, p in zip(files, preds):
                rows.append({"eeg_file": f,
                             "prediction": "background" if p == 0 else "target"})
    return pd.DataFrame(rows)


def predict_ensemble(models, loader, device) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for eeg, files in loader:
            eeg = eeg.to(device)
            probs = torch.stack([F.softmax(m(eeg), dim=1) for m in models]).mean(dim=0)
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
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = get_device(args.cpu)
    print(f"Device: {device}")

    test_ds = EEGDataset(data_dir=CONFIG["test_dir"], label_csv=None)
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
        models = [load_model(f, device) for f in files]
        df = predict_ensemble(models, test_loader, device)
    else:
        path = Path(args.model) if args.model else CONFIG["model_path"]
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        print(f"Model: {path}")
        model = load_model(path, device)
        df = predict_single(model, test_loader, device)

    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")
    print(df["prediction"].value_counts().to_string())


if __name__ == "__main__":
    main()