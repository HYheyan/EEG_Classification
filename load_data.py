"""Data loading and preprocessing for EEG classification.

Two loading modes (auto-detected):
  1. .pt cache — single torch.load(), instant (run convert_data.py once)
  2. .npy files — individual np.load() per file, preloaded into RAM

Preprocessing pipeline (applied once, cached):
  CAR (Common Average Reference) → per-sample z-score normalization

Augmentation is applied on-the-fly with torch ops (training only).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


LABEL_TO_INDEX: Dict[str, int] = {"background": 0, "target": 1}
INDEX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INDEX.items()}


class EEGDataset(torch.utils.data.Dataset):
    """EEG dataset with on-the-fly torch augmentation.

    Parameters
    ----------
    data_dir : Path to .npy files (ignored when .pt cache exists).
    label_csv : Path to labels CSV (None for test mode).
    normalize : Apply CAR + z-score (only used when reading raw .npy).
    selected_indices : Subset of samples to use (for k-fold CV).
    augment : Apply data augmentation (training only).
    return_subject : If True, also return the subject index.
    cache_path : Override .pt cache path.
    """

    def __init__(
        self,
        data_dir: str | Path,
        label_csv: Optional[str | Path] = None,
        normalize: bool = True,
        selected_indices: Optional[List[int]] = None,
        augment: bool = False,
        return_subject: bool = False,
        cache_path: Optional[str | Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment = augment
        self.return_subject = return_subject

        # ---- Try .pt cache first --------------------------------------------
        if cache_path is None:
            cache_path = self.data_dir.parent / f"{self.data_dir.name}_cache.pt"
        cache_path = Path(cache_path)

        if cache_path.exists():
            print(f"  Loading from cache: {cache_path}")
            archive = torch.load(cache_path, map_location="cpu", weights_only=True)
            self._eeg = archive["eeg"]                     # (N, 1, 59, T)
            self._files = archive["eeg_file"]               # list of str
            self._labels = archive.get("label", None)        # (N,) or None
            self._subjects = archive.get("subject", None)    # (N,) or None
            self.subject_to_idx: Dict[str, int] = archive.get("subject_to_idx", {})
            self.has_label = self._labels is not None
            self.has_subject = self._subjects is not None

            # Build samples DataFrame for selected_indices support
            rows = {"eeg_file": self._files}
            if self.has_label:
                rows["label"] = self._labels.tolist()
            if self.has_subject:
                rows["subject"] = self._subjects.tolist()
            self.samples = pd.DataFrame(rows)

        else:
            # ---- Fallback: load from .npy files ------------------------------
            warnings.warn(
                f"No cache at {cache_path}. Run `python convert_data.py` once.\n"
                f"  Falling back to loading .npy files (slower)."
            )
            self._load_from_npy(label_csv, selected_indices)
            selected_indices = None  # already applied

        # ---- Apply index selection (k-fold CV) ------------------------------
        if selected_indices is not None:
            self.samples = self.samples.iloc[selected_indices].reset_index(drop=True)
            keep = torch.tensor(selected_indices)
            self._eeg = self._eeg[keep]
            if self._labels is not None:
                self._labels = self._labels[keep]
            if self._subjects is not None:
                self._subjects = self._subjects[keep]

        # Build subject-to-index from data if not from cache
        if self.has_subject and not self.subject_to_idx:
            for s in sorted(self.samples["subject"].unique()):
                self.subject_to_idx[s] = len(self.subject_to_idx)

    def _load_from_npy(
        self,
        label_csv: Optional[str | Path],
        selected_indices: Optional[List[int]],
    ) -> None:
        """Legacy: load individual .npy files into RAM."""
        if label_csv is None:
            eeg_files = sorted(p.name for p in self.data_dir.glob("*.npy"))
            self.samples = pd.DataFrame({"eeg_file": eeg_files})
            self.has_label = False
            self.has_subject = False
        else:
            self.samples = pd.read_csv(label_csv)
            self.has_label = True
            self.has_subject = "subject" in self.samples.columns

        if selected_indices is not None:
            self.samples = self.samples.iloc[selected_indices].reset_index(drop=True)

        # Preload as torch tensors
        tensors, labels, subjects = [], [], []
        for _, row in self.samples.iterrows():
            raw = np.load(self.data_dir / row["eeg_file"]).astype(np.float32)
            if self.normalize:
                raw = raw - raw.mean(axis=1, keepdims=True)
                m, s = raw.mean(), raw.std()
                raw = (raw - m) / (s + 1e-6)
            tensors.append(torch.from_numpy(raw))
            if self.has_label:
                labels.append(LABEL_TO_INDEX[row["label"]])
            if self.has_subject:
                subjects.append(row["subject"])

        self._eeg = torch.stack(tensors)
        self._files = self.samples["eeg_file"].tolist()
        self._labels = torch.tensor(labels) if labels else None
        self._subjects = None
        if subjects:
            if not self.subject_to_idx:
                for s in sorted(set(subjects)):
                    self.subject_to_idx[s] = len(self.subject_to_idx)
            self._subjects = torch.tensor(
                [self.subject_to_idx[s] for s in subjects]
            )
            self.has_subject = True

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._eeg)

    def __getitem__(self, index: int):
        eeg = self._eeg[index]
        if self.augment:
            eeg = eeg.clone()
            eeg = self._augment(eeg)

        if self.has_label:
            label = self._labels[index]
            if self.return_subject and self.has_subject:
                return eeg, label, self._subjects[index]
            return eeg, label

        fname = self._files[index]
        if self.return_subject and self.has_subject:
            return eeg, fname, self._subjects[index]
        return eeg, fname

    # ------------------------------------------------------------------
    # Torch-based augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _augment(eeg: torch.Tensor) -> torch.Tensor:
        """In-place torch augmentations. eeg shape: (1, 59, T)."""

        if torch.rand(1).item() < 0.6:
            sigma = 0.01 + torch.rand(1).item() * 0.07
            eeg.add_(torch.randn_like(eeg).mul_(sigma))

        if torch.rand(1).item() < 0.6:
            shift = torch.randint(-15, 16, (1,)).item()
            if shift != 0:
                eeg.copy_(torch.roll(eeg, shift, dims=-1))

        if torch.rand(1).item() < 0.5:
            eeg.mul_(0.85 + torch.rand(1).item() * 0.3)

        if torch.rand(1).item() < 0.4:
            n_ch = eeg.shape[1]
            n_drop = torch.randint(
                max(1, int(n_ch * 0.08)),
                max(2, int(n_ch * 0.20)) + 1, (1,)
            ).item()
            drop_idx = torch.randperm(n_ch)[:n_drop]
            eeg[:, drop_idx, :] = 0.0

        if torch.rand(1).item() < 0.3:
            t_len = eeg.shape[-1]
            mask_len = torch.randint(t_len // 20, t_len // 8, (1,)).item()
            mask_start = torch.randint(0, t_len - mask_len, (1,)).item()
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
