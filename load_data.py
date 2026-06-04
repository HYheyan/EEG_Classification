from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

LABEL_TO_INDEX: Dict[str, int] = {"background": 0, "target": 1}
INDEX_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_INDEX.items()}

# Frequency band definitions (low, high) in Hz
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
}


class EEGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        label_csv: Optional[str | Path] = None,
        normalize: Union[bool, str] = True,
        selected_indices: Optional[List[int]] = None,
        bandpass: Optional[str] = None,
        sfreq: float = 250.0,
        augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment = augment

        # ---- bandpass filter design -----------------------------------
        if bandpass is not None:
            if bandpass not in FREQ_BANDS:
                raise ValueError(
                    f"Unknown band: '{bandpass}'. "
                    f"Choose from {list(FREQ_BANDS.keys())}."
                )
            low, high = FREQ_BANDS[bandpass]
            nyq = 0.5 * sfreq
            self.sos = butter(
                4, [low / nyq, high / nyq], btype="band", output="sos"
            )
        else:
            self.sos = None

        # ---- load sample list -----------------------------------------
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
        eeg = np.load(eeg_path).astype(np.float32)  # shape: (1, 59, 282)

        # --- 1. bandpass filtering ------------------------------------
        if self.sos is not None:
            eeg = sosfiltfilt(self.sos, eeg, axis=-1).astype(np.float32)

        # --- 2. normalisation -----------------------------------------
        if self.normalize is True or self.normalize == "global":
            mean = eeg.mean()
            std = eeg.std()
            eeg = (eeg - mean) / (std + 1e-6)
        elif self.normalize == "channel":
            # Per-electrode normalisation: each of the 59 EEG electrodes
            # independently, using statistics computed across time points.
            mean = eeg.mean(axis=-1, keepdims=True)
            std = eeg.std(axis=-1, keepdims=True)
            eeg = (eeg - mean) / (std + 1e-6)

        # --- 3. data augmentation (training only) ---------------------
        if self.augment:
            eeg = self._augment(eeg)

        eeg_tensor = torch.from_numpy(eeg)
        if self.has_label:
            label = torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long)
            return eeg_tensor, label
        return eeg_tensor, row["eeg_file"]

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _augment(eeg: np.ndarray) -> np.ndarray:
        """In-place random augmentations, each applied with p=0.5."""
        rng = np.random.default_rng()

        # (a) Time jitter — additive Gaussian noise
        if rng.random() < 0.5:
            noise_std = 0.05 * float(eeg.std())
            eeg = eeg + rng.normal(0, noise_std, size=eeg.shape).astype(np.float32)

        # (b) Random time shift — circular roll along time axis
        if rng.random() < 0.5:
            max_shift = int(0.1 * eeg.shape[-1])
            shift = int(rng.integers(0, max_shift + 1))
            eeg = np.roll(eeg, shift, axis=-1)

        # (c) Channel dropout — zero entire electrode rows
        if rng.random() < 0.5:
            n_electrodes = eeg.shape[1]  # 59
            n_drop = max(1, int(0.1 * n_electrodes))
            drop_idx = rng.choice(n_electrodes, n_drop, replace=False)
            eeg[:, drop_idx, :] = 0.0

        return eeg
