"""Data loading and preprocessing for EEG classification.

Preprocessing pipeline:
  1. CAR (Common Average Reference) — subtracts mean across electrodes
  2. Per-sample z-score normalization
  3. Data augmentation (noise, time shift, amplitude scale,
     channel dropout, time mask) — training only

Supports label-less mode for test data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


LABEL_TO_INDEX: Dict[str, int] = {"background": 0, "target": 1}
INDEX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INDEX.items()}


class EEGDataset(torch.utils.data.Dataset):
    """EEG dataset with preprocessing and augmentation.

    Parameters
    ----------
    data_dir : Path to .npy files.
    label_csv : Path to labels CSV (None for test mode).
    normalize : Apply CAR + z-score normalization.
    selected_indices : Subset of samples to use (for k-fold CV).
    augment : Apply data augmentation (training only).
    """

    def __init__(
        self,
        data_dir: str | Path,
        label_csv: Optional[str | Path] = None,
        normalize: bool = True,
        selected_indices: Optional[List[int]] = None,
        augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment = augment

        if label_csv is None:
            eeg_files = sorted(p.name for p in self.data_dir.glob("*.npy"))
            self.samples = pd.DataFrame({"eeg_file": eeg_files})
            self.has_label = False
        else:
            self.samples = pd.read_csv(label_csv)
            self.has_label = True

        if selected_indices is not None:
            self.samples = self.samples.iloc[selected_indices].reset_index(drop=True)

        if "eeg_file" not in self.samples.columns:
            raise ValueError("label csv must contain column: eeg_file")

        if self.has_label and "label" not in self.samples.columns:
            raise ValueError("training label csv must contain column: label")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        eeg_path = self.data_dir / row["eeg_file"]
        eeg = np.load(eeg_path).astype(np.float32)

        # Preprocessing
        if self.normalize:
            # CAR: remove common-mode noise
            eeg = eeg - eeg.mean(axis=1, keepdims=True)
            # Per-sample z-score
            mean = eeg.mean()
            std = eeg.std()
            eeg = (eeg - mean) / (std + 1e-6)

        # Augmentation (training only)
        if self.augment:
            eeg = self._apply_augmentation(eeg)

        eeg = np.asarray(eeg, dtype=np.float32)
        eeg_tensor = torch.from_numpy(eeg)

        if self.has_label:
            label = torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long)
            return eeg_tensor, label
        return eeg_tensor, row["eeg_file"]

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_augmentation(eeg: np.ndarray) -> np.ndarray:
        """Random augmentations for EEG. All probabilities can be tuned."""
        rng = np.random.default_rng()

        # Gaussian noise (60% chance, σ up to 0.08)
        if rng.random() < 0.6:
            noise_level = rng.uniform(0.01, 0.08)
            eeg = eeg + rng.normal(0, noise_level, size=eeg.shape).astype(np.float32)

        # Temporal shift (60% chance, ±15 steps)
        if rng.random() < 0.6:
            shift = rng.integers(-15, 16)
            eeg = np.roll(eeg, shift, axis=-1)

        # Amplitude scaling (50% chance, 0.85–1.15×)
        if rng.random() < 0.5:
            eeg = eeg * rng.uniform(0.85, 1.15)

        # Channel dropout (40% chance, drop 8-20% of electrodes)
        if rng.random() < 0.4:
            n_channels = eeg.shape[1]
            n_drop = rng.integers(
                max(1, int(n_channels * 0.08)),
                max(2, int(n_channels * 0.20)) + 1,
            )
            drop_idx = rng.choice(n_channels, size=n_drop, replace=False)
            eeg[:, drop_idx, :] = 0.0

        # Time masking (30% chance, mask a contiguous segment)
        if rng.random() < 0.3:
            t_len = eeg.shape[-1]
            mask_len = rng.integers(t_len // 20, t_len // 8)
            mask_start = rng.integers(0, t_len - mask_len)
            eeg[:, :, mask_start:mask_start + mask_len] = 0.0

        return eeg


# ---------------------------------------------------------------------------
# Train/val splits
# ---------------------------------------------------------------------------

def build_split_indices(
    label_csv: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Stratified train/val split by label."""
    df = pd.read_csv(label_csv)
    rng = np.random.default_rng(seed)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for label_name in sorted(df["label"].unique()):
        label_indices = np.where(df["label"].values == label_name)[0]
        shuffled = rng.permutation(label_indices)
        val_size = max(1, int(len(shuffled) * val_ratio))
        val_indices.extend(shuffled[:val_size].tolist())
        train_indices.extend(shuffled[val_size:].tolist())

    return sorted(train_indices), sorted(val_indices)


def build_kfold_indices(
    label_csv: str | Path,
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Generate k stratified train/val fold index pairs."""
    df = pd.read_csv(label_csv)
    rng = np.random.default_rng(seed)

    bg_indices = np.where(df["label"].values == "background")[0]
    tg_indices = np.where(df["label"].values == "target")[0]
    rng.shuffle(bg_indices)
    rng.shuffle(tg_indices)

    bg_folds = np.array_split(bg_indices, k)
    tg_folds = np.array_split(tg_indices, k)

    folds = []
    for i in range(k):
        val_idx = sorted(np.concatenate([bg_folds[i], tg_folds[i]]).tolist())
        train_idx = sorted(np.concatenate([
            np.concatenate([bg_folds[j] for j in range(k) if j != i]),
            np.concatenate([tg_folds[j] for j in range(k) if j != i]),
        ]).tolist())
        folds.append((train_idx, val_idx))

    return folds