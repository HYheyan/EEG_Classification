"""Training script for EEG binary classification.

Improvements over the baseline:
- EEGNet model (compact CNN, resistant to overfitting)
- CosineAnnealingWarmRestarts scheduler (cyclical LR)
- Label smoothing (0.1) for softer targets
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=25, restore best weights)
- Data augmentation (Gaussian noise, time shift, amplitude scaling)
- Per-epoch logging of train & val accuracy
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import EEGDataset, build_split_indices
from model import EEGNet
from utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

CONFIG = {
    # Data
    "train_dir": DATA_ROOT / "train",
    "train_label_csv": DATA_ROOT / "train_labels.csv",

    # Training hyperparameters
    "batch_size": 64,
    "epochs": 120,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "val_ratio": 0.2,
    "seed": 42,
    "num_workers": 0,
    "use_cpu": False,

    # Regularisation
    "label_smoothing": 0.1,
    "grad_clip_norm": 1.0,

    # LR schedule (CosineAnnealingWarmRestarts)
    "scheduler_t0": 15,
    "scheduler_t_mult": 2,

    # Early stopping
    "early_stop_patience": 25,

    # Data augmentation (applied to training set only)
    "augment_train": True,

    # EEGNet model hyperparameters (F2 must equal D * F1 for residual)
    "eegnet_F1": 18,
    "eegnet_D": 2,
    "eegnet_F2": 36,
    "eegnet_dropout": 0.3,
}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (average_loss, accuracy) on the given dataloader."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for eeg, labels in dataloader:
            eeg = eeg.to(device)
            labels = labels.to(device)
            logits = model(eeg)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total, correct / total


def main() -> None:
    set_seed(CONFIG["seed"])
    device = get_device(CONFIG["use_cpu"])
    print(f"Using device: {device}")

    # ---- Stratified train / val split ------------------------------------
    train_indices, val_indices = build_split_indices(
        CONFIG["train_label_csv"],
        val_ratio=CONFIG["val_ratio"],
        seed=CONFIG["seed"],
    )

    train_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        selected_indices=train_indices,
        augment=CONFIG["augment_train"],
    )
    val_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        selected_indices=val_indices,
        augment=False,  # never augment validation data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ---- Model ------------------------------------------------------------
    sample_eeg, _ = train_dataset[0]
    model = EEGNet(
        input_shape=tuple(sample_eeg.shape),
        F1=CONFIG["eegnet_F1"],
        D=CONFIG["eegnet_D"],
        F2=CONFIG["eegnet_F2"],
        dropout_rate=CONFIG["eegnet_dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EEGNet | Params: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Class-balanced weights ------------------------------------------
    label_series = pd.read_csv(CONFIG["train_label_csv"])["label"]
    background_count = (label_series == "background").sum()
    target_count = (label_series == "target").sum()
    class_weights = torch.tensor(
        [
            len(label_series) / (2.0 * background_count),
            len(label_series) / (2.0 * target_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    # Label smoothing softens the one-hot targets → reduces overconfidence
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Cosine annealing with warm restarts — cyclical LR helps escape local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG["scheduler_t0"],
        T_mult=CONFIG["scheduler_t_mult"],
        eta_min=1e-6,
    )

    # ---- Training loop ---------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_state: dict | None = None
    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ---- train one epoch ----
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for eeg, labels in train_loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            if CONFIG["grad_clip_norm"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / train_total
        train_acc = train_correct / train_total

        # ---- validate ----
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler (per epoch for WarmRestarts)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- logging ----
        print(
            f"Epoch {epoch:03d} | "
            f"LR={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            + (" *" if val_acc > best_val_acc else "")
        )

        # ---- save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, MODEL_DIR / "best_model.pth")
            print(f"  -> Best model saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1

        # ---- early stopping ----
        if patience_counter >= CONFIG["early_stop_patience"]:
            print(
                f"\nEarly stopping triggered after {epoch} epochs "
                f"(no improvement for {CONFIG['early_stop_patience']} epochs)."
            )
            break

    # ---- Restore best & save final ---------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_DIR / "final_model.pth")

    print(f"\n{'='*60}")
    print(f"Training finished.")
    print(f"  Best val_acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Best model  -> {MODEL_DIR / 'best_model.pth'}")
    print(f"  Final model -> {MODEL_DIR / 'final_model.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
