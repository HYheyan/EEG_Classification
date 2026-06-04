from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


LABEL_TO_INDEX: Dict[str, int] = {"background": 0, "target": 1}
INDEX_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_INDEX.items()}


class EEGDataset(torch.utils.data.Dataset):
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
            eeg_files = sorted(path.name for path in self.data_dir.glob("*.npy"))
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

        if self.normalize:
            # CAR: Common Average Reference — subtract mean across all
            # electrodes at each time point to remove common-mode noise.
            # Shape is (1, 59, 282); electrodes are axis=1.
            eeg = eeg - eeg.mean(axis=1, keepdims=True)

            mean = eeg.mean()
            std = eeg.std()
            eeg = (eeg - mean) / (std + 1e-6)

        if self.augment:
            eeg = self._apply_augmentation(eeg)

        eeg_tensor = torch.from_numpy(eeg)
        if self.has_label:
            label = torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long)
            return eeg_tensor, label
        return eeg_tensor, row["eeg_file"]

    @staticmethod
    def _apply_augmentation(eeg: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single EEG sample.

        Augmentations (each applied with independent probability):
        - Gaussian noise: additive white noise, σ = 0.05 (50 % prob)
        - Temporal shift: roll the signal ±10 time steps (50 % prob)
        - Amplitude scaling: multiply by 0.9–1.1 (50 % prob)
        - Channel dropout: zero out ~10 % of electrodes (30 % prob)
        """
        rng = np.random.default_rng()

        # Gaussian noise
        if rng.random() < 0.5:
            noise = rng.normal(0, 0.05, size=eeg.shape).astype(np.float32)
            eeg = eeg + noise

        # Temporal shift (roll along the last axis = time)
        if rng.random() < 0.5:
            shift = rng.integers(-10, 11)
            eeg = np.roll(eeg, shift, axis=-1)

        # Amplitude scaling
        if rng.random() < 0.5:
            scale = rng.uniform(0.9, 1.1)
            eeg = eeg * scale

        # Channel dropout — randomly zero out ~10 % of electrodes.
        # This prevents the model from relying on any single electrode
        # and mimics real-world electrode variability.
        if rng.random() < 0.3:
            n_channels = eeg.shape[1]
            n_drop = max(1, int(n_channels * 0.10))
            drop_indices = rng.choice(n_channels, size=n_drop, replace=False)
            eeg[:, drop_indices, :] = 0.0

        return eeg


def build_split_indices(
    label_csv: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
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
