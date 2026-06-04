"""
EEG binary classification — training script with K-fold CV and advanced training.

Features
--------
- Stratified K-fold cross-validation
- Multiple architectures (eegnet, conformer, enhanced)
- Mixup data augmentation
- AdamW with decoupled weight decay
- Cosine annealing with linear warm-up
- Exponential moving average (EMA) of model weights
- Per-fold checkpointing (best + EMA)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from load_data import EEGDataset
from model import create_model, MODEL_REGISTRY
from utils import get_device, set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # ---- data ----------------------------------------------------------
    "train_dir": DATA_ROOT / "train",
    "train_label_csv": DATA_ROOT / "train_labels.csv",
    # ---- training ------------------------------------------------------
    "architecture": "all",    # "all" | "eegnet" | "conformer" | "enhanced"
    "n_folds": 5,
    "epochs": 120,
    "batch_size": 128,          # doubled for better GPU utilisation
    "num_workers": 2,           # parallel data loading
    "use_cpu": False,
    "use_amp": True,            # mixed-precision (2× speedup on modern GPUs)
    "seed": 42,
    # ---- optimizer -----------------------------------------------------
    "learning_rate": 1e-3,
    "weight_decay": 5e-2,           # AdamW decoupled
    "grad_clip_max_norm": 5.0,
    # ---- scheduler -----------------------------------------------------
    "warmup_epochs": 5,
    "cosine_T_0": 20,               # first restart cycle (epochs)
    "cosine_T_mult": 2,             # ×2 each cycle
    # ---- regularisation ------------------------------------------------
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,             # 0 = disabled
    "mixup_prob": 0.5,              # per-batch probability
    # ---- EMA -----------------------------------------------------------
    "ema_decay": 0.99,
}


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        """Replace model params with EMA shadow (call before eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        """Restore original params (call after eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# =============================================================================
# Mixup
# =============================================================================

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    device: torch.device,
):
    """Return (mixed_x, y_a, y_b, lam).  lam ~ Beta(alpha, alpha)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_loss_fn(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Return (average_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    for eeg, labels in dataloader:
        eeg = eeg.to(device)
        labels = labels.to(device)
        logits = model(eeg)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total, correct / total


# =============================================================================
# Single-fold training
# =============================================================================

def train_one_fold(
    arch: str,
    fold: int,
    train_indices: List[int],
    val_indices: List[int],
    device: torch.device,
) -> dict:
    """Train *arch* on the given fold split.  Returns summary dict."""

    print(f"\n{'#' * 60}")
    print(f"#  Fold {fold + 1}  |  {arch}")
    print(f"{'#' * 60}")

    # ---- datasets ------------------------------------------------------
    train_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        selected_indices=train_indices,
        normalize="channel",
        augment=True,
    )
    val_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        selected_indices=val_indices,
        normalize="channel",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- model ---------------------------------------------------------
    sample_eeg, _ = train_dataset[0]
    input_shape = tuple(sample_eeg.shape)
    model = create_model(arch, input_shape=input_shape, num_classes=2).to(device)

    # Optional: torch.compile (PyTorch ≥ 2.0, ~20% speedup)
    # NOTE: inductor backend requires Triton (Linux only).  On Windows we
    # skip compilation entirely to avoid a hard crash on the first forward pass.
    _can_compile = False
    try:
        import triton  # noqa: F401
        _can_compile = True
    except ImportError:
        pass
    if _can_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile enabled (reduce-overhead)")
        except Exception:
            pass

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,}")

    # ---- class weights ------------------------------------------------
    train_labels = pd.read_csv(CONFIG["train_label_csv"]).iloc[train_indices]["label"]
    bg_count = (train_labels == "background").sum()
    tg_count = (train_labels == "target").sum()
    class_weights = torch.tensor(
        [len(train_labels) / (2.0 * bg_count), len(train_labels) / (2.0 * tg_count)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG["label_smoothing"],
    )

    # ---- optimizer -----------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ---- scheduler (warmup → cosine warm restarts) ---------------------
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=CONFIG["warmup_epochs"],
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG["cosine_T_0"],
        T_mult=CONFIG["cosine_T_mult"],
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[CONFIG["warmup_epochs"]],
    )

    # ---- EMA -----------------------------------------------------------
    ema = EMA(model, decay=CONFIG["ema_decay"])

    # ---- training state ------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history: list = []
    best_val_acc = -1.0
    best_epoch = 0

    # ---- AMP (mixed precision) -----------------------------------------
    use_amp = CONFIG.get("use_amp", False) and (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ---- training loop -------------------------------------------------
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for eeg, labels in train_loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            # ---- optional mixup ----
            y_a, y_b, lam = labels, labels, 1.0
            use_mixup = (
                CONFIG["mixup_alpha"] > 0
                and np.random.random() < CONFIG["mixup_prob"]
            )
            if use_mixup:
                eeg, y_a, y_b, lam = mixup_batch(
                    eeg, labels, CONFIG["mixup_alpha"], device
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(eeg)
                if use_mixup:
                    loss = mixup_loss_fn(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=CONFIG["grad_clip_max_norm"]
            )
            scaler.step(optimizer)
            scaler.update()

            # EMA update after each optimizer step
            ema.update()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples

        # ---- validation (with EMA weights) ----
        ema.apply_shadow()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        ema.restore()

        # Also evaluate raw weights for tracking
        val_loss_raw, val_acc_raw = evaluate(model, val_loader, criterion, device)

        # ---- scheduler ----
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ---- record ----
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_acc_raw": val_acc_raw,
            "lr": current_lr,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:03d} | "
                f"tr_loss={train_loss:.4f} | "
                f"va_acc={val_acc:.4f} | va_acc_raw={val_acc_raw:.4f} | "
                f"lr={current_lr:.2e}"
            )

        # ---- checkpoint ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "ema_shadow": copy.deepcopy({k: v.cpu() for k, v in ema.shadow.items()}),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "arch": arch,
                "fold": fold,
                "history": history,
                "input_shape": input_shape,
            }
            torch.save(best_state, MODEL_DIR / f"{arch}_fold{fold}_best.pth")

            # Also save EMA-only checkpoint (lighter, just weights)
            ema_shadow_cpu = {k: v.cpu().clone() for k, v in ema.shadow.items()}
            ema_state = {
                "model_state_dict": ema_shadow_cpu,
                "arch": arch,
                "fold": fold,
                "best_val_acc": best_val_acc,
                "input_shape": input_shape,
            }
            torch.save(ema_state, MODEL_DIR / f"{arch}_fold{fold}_ema.pth")

    print(
        f"  >> Fold {fold + 1} done.  "
        f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}"
    )

    return {
        "arch": arch,
        "fold": fold,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    set_seed(CONFIG["seed"])
    device = get_device(CONFIG["use_cpu"])
    print(f"Using device: {device}")

    # ---- determine architectures to train ------------------------------
    if CONFIG["architecture"] == "all":
        archs = list(MODEL_REGISTRY.keys())  # ["eegnet", "conformer", "enhanced"]
    else:
        if CONFIG["architecture"] not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown architecture '{CONFIG['architecture']}'. "
                f"Choose from {list(MODEL_REGISTRY.keys())} or 'all'."
            )
        archs = [CONFIG["architecture"]]

    print(f"Architectures: {archs}")
    print(f"K-fold splits: {CONFIG['n_folds']}")
    print(f"Epochs per fold: {CONFIG['epochs']}")

    # ---- load full label dataframe for K-fold --------------------------
    df = pd.read_csv(CONFIG["train_label_csv"])

    # ---- outer loop: architectures → folds ----------------------------
    all_results: list = []

    for arch in archs:
        skf = StratifiedKFold(
            n_splits=CONFIG["n_folds"],
            shuffle=True,
            random_state=CONFIG["seed"],
        )
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(df)), df["label"])
        ):
            result = train_one_fold(
                arch=arch,
                fold=fold,
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                device=device,
            )
            all_results.append(result)

    # ---- summary -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"{'=' * 60}")
    for r in all_results:
        print(
            f"  {r['arch']:>12s}  fold {r['fold']}  "
            f"best_val_acc = {r['best_val_acc']:.4f}  "
            f"(epoch {r['best_epoch']})"
        )

    # Overall average
    avg = np.mean([r["best_val_acc"] for r in all_results])
    print(f"\n  Overall mean best_val_acc = {avg:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
