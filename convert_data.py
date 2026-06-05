"""One-time script: convert all .npy files to preprocessed .pt archives.

Run once:
    python convert_data.py

Produces:
    data/train_cache.pt   — training data + labels + subjects
    data/test_cache.pt    — test data (labels=0 placeholder)

Each .pt file = dict of tensors, ~110 MB for train, already CAR+zscored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

BATCH = 128  # process in batches to keep memory low


def preprocess(raw: np.ndarray) -> np.ndarray:
    """CAR + per-sample z-score. Returns float32."""
    raw = raw.astype(np.float32)
    raw = raw - raw.mean(axis=1, keepdims=True)
    m, s = raw.mean(), raw.std()
    raw = (raw - m) / (s + 1e-6)
    return raw


def convert_train() -> None:
    label_csv = DATA_ROOT / "train_labels.csv"
    train_dir = DATA_ROOT / "train"
    out_path = DATA_ROOT / "train_cache.pt"

    df = pd.read_csv(label_csv)
    n = len(df)
    eeg_list, label_list, subj_list = [], [], []

    has_subject = "subject" in df.columns
    subjects = sorted(df["subject"].unique()) if has_subject else []
    subj_to_idx = {s: i for i, s in enumerate(subjects)}

    for i, (_, row) in enumerate(df.iterrows()):
        eeg = np.load(train_dir / row["eeg_file"])
        eeg = preprocess(eeg)
        eeg_list.append(torch.from_numpy(eeg))
        label_list.append(0 if row["label"] == "background" else 1)
        if has_subject:
            subj_list.append(subj_to_idx[row["subject"]])
        if (i + 1) % BATCH == 0:
            print(f"  train {i + 1}/{n}")

    data = {
        "eeg": torch.stack(eeg_list),           # (N, 1, 59, 282)
        "label": torch.tensor(label_list),       # (N,)
        "eeg_file": df["eeg_file"].tolist(),     # list of str
    }
    if has_subject:
        data["subject"] = torch.tensor(subj_list)
        data["subject_to_idx"] = subj_to_idx

    torch.save(data, out_path)
    print(f"  saved {out_path}  ({out_path.stat().st_size / 1024 / 1024:.0f} MB)")


def convert_test() -> None:
    test_dir = DATA_ROOT / "test"
    out_path = DATA_ROOT / "test_cache.pt"

    files = sorted(p.name for p in test_dir.glob("*.npy"))
    eeg_list = []

    for i, fname in enumerate(files):
        eeg = np.load(test_dir / fname)
        eeg = preprocess(eeg)
        eeg_list.append(torch.from_numpy(eeg))
        if (i + 1) % BATCH == 0:
            print(f"  test {i + 1}/{len(files)}")

    data = {
        "eeg": torch.stack(eeg_list),       # (N, 1, 59, 282)
        "eeg_file": files,                  # list of str
    }
    torch.save(data, out_path)
    print(f"  saved {out_path}  ({out_path.stat().st_size / 1024 / 1024:.0f} MB)")


if __name__ == "__main__":
    print("Converting training data...")
    convert_train()
    print("Converting test data...")
    convert_test()
    print("Done.")
