from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from load_data import EEGDataset
from model import Net
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
    "use_cpu": False,
}


def main() -> None:
    device = get_device(CONFIG["use_cpu"])
    test_dataset = EEGDataset(data_dir=CONFIG["test_dir"], label_csv=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    sample_eeg, _ = test_dataset[0]
    model = Net(input_shape=tuple(sample_eeg.shape)).to(device)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for eeg, eeg_files in test_loader:
            eeg = eeg.to(device)
            logits = model(eeg)
            predicted = logits.argmax(dim=1).cpu().tolist()
            for eeg_file, pred_index in zip(eeg_files, predicted):
                label_name = "background" if pred_index == 0 else "target"
                predictions.append({"eeg_file": eeg_file, "prediction": label_name})

    RES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(CONFIG["output_csv"], index=False)
    print(f"Predictions saved to: {CONFIG['output_csv']}")


if __name__ == "__main__":
    main()
