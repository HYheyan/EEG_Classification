"""Training script for EEG binary classification.

Key techniques for small-data regime (~1500 samples):
- EEGNet — multi-scale depthwise-separable CNN
- Mixup augmentation — smoother decision boundaries
- SWA (Stochastic Weight Averaging) — better generalization
- Warmup + CosineAnnealingWarmRestarts — stable training
- K-fold cross-validation (5-fold) with ensemble save
- Coordinate-wise hyperparameter tuning
- Focal Loss option — focus on hard-to-classify examples
"""

from __future__ import annotations

import argparse
import platform
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from load_data import EEGDataset, build_kfold_indices, build_split_indices
from model import EEGNet
from utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG: Dict = {
    # Paths
    "train_dir": DATA_ROOT / "train",
    "train_label_csv": DATA_ROOT / "train_labels.csv",

    # General
    "epochs": 500,
    "seed": 42,
    "num_workers": 0 if platform.system() == "Windows" else 2,
    "use_cpu": False,
    "early_stop_patience": 50,
    "augment_train": True,

    # Model
    "f1": 16,
    "depth": 2,
    "f2": 32,               # = depth * f1
    "dropout_rate": 0.4,
    # Optimizer
    "learning_rate": 2e-3,
    "weight_decay": 1e-3,
    "batch_size": 128,

    # LR schedule
    "warmup_epochs": 8,
    "scheduler_t0": 25,
    "scheduler_t_mult": 2,

    # Regularization
    "label_smoothing": 0.05,
    "mixup_alpha": 0.2,      # 0 = disable
    "focal_gamma": 0.0,       # 0 = disable; try 2.0 for hard examples
    "grad_clip_norm": 1.0,

    # SWA
    "swa_start": 30,
    "swa_lr": 5e-4,

    # CV
    "cv_mode": "kfold",      # "kfold" | "single" | "mccv"
    "k_folds": 5,
    "val_ratio": 0.2,
    "mccv_rounds": 20,

    # Augmented dataset
    "use_augmented": False,
    "augment_multiplier": 2,
}


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for binary/multi-class classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    With γ=0, reduces to standard CrossEntropyLoss.
    γ=2.0 is a good starting point for imbalanced/hard datasets.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    for batch in dataloader:
        eeg, labels = batch
        eeg, labels = eeg.to(device), labels.to(device)
        logits = model(eeg)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_one_fold(
    fold: int,
    train_indices: List[int],
    val_indices: List[int],
    params: Dict,
    device: torch.device,
) -> Tuple[dict, float, int]:
    """Train one fold. Returns (best_state, best_val_acc, best_epoch)."""

    # ---- Datasets ---------------------------------------------------------
    ds_kwargs = dict(
        data_dir=CONFIG["train_dir"], label_csv=CONFIG["train_label_csv"],
    )
    use_aug = params.get("use_augmented", False)

    if use_aug:
        # Augmented cache layout: [N originals] + [N*M augmented copies]
        #   original i → index i
        #   copy k of original i → index N + i*M + k
        N = len(train_indices) + len(val_indices)
        M = CONFIG["augment_multiplier"]

        # Training: all originals in train split + all their augmented copies
        aug_train_indices = list(train_indices)
        for i in train_indices:
            for k in range(M):
                aug_train_indices.append(N + i * M + k)

        print(f"  Augmented train: {len(aug_train_indices)} samples "
              f"({len(train_indices)} orig + {len(aug_train_indices) - len(train_indices)} aug)")

        train_ds = EEGDataset(
            **ds_kwargs, selected_indices=aug_train_indices,
            augment=CONFIG["augment_train"], augmented=True,
        )
    else:
        train_ds = EEGDataset(
            **ds_kwargs, selected_indices=train_indices,
            augment=CONFIG["augment_train"],
        )

    val_ds = EEGDataset(
        **ds_kwargs, selected_indices=val_indices, augment=False,
        augmented=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"] * 2, shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Model -------------------------------------------------------------
    sample = train_ds[0]
    sample_eeg = sample[0]
    model = EEGNet(
        input_shape=tuple(sample_eeg.shape),
        f1=params["f1"], depth=params["depth"], f2=params["f2"],
        dropout_rate=params["dropout_rate"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # ---- Loss & optimizer --------------------------------------------------
    label_df = pd.read_csv(CONFIG["train_label_csv"])
    train_labels = label_df.iloc[train_indices]["label"]
    bg_n = (train_labels == "background").sum()
    tg_n = (train_labels == "target").sum()
    class_weights = torch.tensor(
        [len(train_labels) / (2.0 * max(bg_n, 1)),
         len(train_labels) / (2.0 * max(tg_n, 1))],
        dtype=torch.float32, device=device,
    )

    if params["focal_gamma"] > 0:
        criterion = FocalLoss(
            gamma=params["focal_gamma"], weight=class_weights,
            label_smoothing=params["label_smoothing"],
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=params["label_smoothing"],
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"], weight_decay=params["weight_decay"],
    )

    # ---- SWA ---------------------------------------------------------------
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=params["swa_lr"])

    # ---- Scheduler ---------------------------------------------------------
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=params["scheduler_t0"],
        T_mult=params["scheduler_t_mult"], eta_min=1e-6,
    )

    # ---- Training loop -----------------------------------------------------
    save_model = params.get("save_model", True)
    fold_dir = MODEL_DIR / f"eegnet_fold{fold}"
    if save_model:
        fold_dir.mkdir(parents=True, exist_ok=True)

    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    patience = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        # Warmup
        if epoch <= params["warmup_epochs"]:
            lr = params["learning_rate"] * epoch / params["warmup_epochs"]
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        elif epoch == params["warmup_epochs"] + 1:
            for pg in optimizer.param_groups:
                pg["lr"] = params["learning_rate"]

        # Train one epoch
        model.train()
        running_loss, train_total = 0.0, 0

        for batch in train_loader:
            eeg, labels = batch
            eeg, labels = eeg.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            if params["mixup_alpha"] > 0:
                mixed, y_a, y_b, lam = mixup_data(eeg, labels, params["mixup_alpha"])
                logits = model(mixed)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(eeg)
                loss = criterion(logits, labels)

            loss.backward()
            if params["grad_clip_norm"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip_norm"])
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            train_total += labels.size(0)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if epoch > params["warmup_epochs"]:
            cos_scheduler.step()

        # SWA update
        if epoch >= params["swa_start"]:
            swa_model.update_parameters(model)
            if epoch > params["swa_start"]:
                swa_scheduler.step()

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1 or (val_acc > best_val_acc and epoch > 1):
            marker = " *" if val_acc > best_val_acc else ""
            print(
                f"    Ep {epoch:03d} | "
                f"loss={running_loss / train_total:.4f} | "
                f"vloss={val_loss:.4f} | vacc={val_acc:.4f}{marker}"
            )

        if patience >= CONFIG["early_stop_patience"]:
            print(f"    Early stop at epoch {epoch}")
            break

    # ---- Finalize -----------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        if save_model:
            torch.save(best_state, fold_dir / "best_model.pth")

    if epoch >= params["swa_start"]:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device)
        if save_model:
            torch.save(deepcopy(swa_model.state_dict()), fold_dir / "best_ema.pth")

    return best_state, best_val_acc, best_epoch


# ---------------------------------------------------------------------------
# K-fold / Single split
# ---------------------------------------------------------------------------

def run_kfold_cv(params: Dict, device: torch.device) -> List[Dict]:
    folds = build_kfold_indices(
        CONFIG["train_label_csv"], k=CONFIG["k_folds"], seed=CONFIG["seed"],
    )
    results = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*55}\nFold {fold+1}/{CONFIG['k_folds']}  "
              f"(train={len(train_idx)}, val={len(val_idx)})\n{'='*55}")
        _, acc, ep = train_one_fold(fold, train_idx, val_idx, params, device)
        results.append({"fold": fold, "val_acc": acc, "best_epoch": ep})
        print(f"  Fold {fold} best: {acc:.4f} (epoch {ep})")
    return results


def run_mccv(params: Dict, device: torch.device) -> List[Dict]:
    """Monte Carlo Cross-Validation.

    Repeated random stratified 80/20 splits with different seeds.
    Does NOT save model checkpoints — only returns validation stats.
    """
    rounds = CONFIG["mccv_rounds"]
    val_ratio = CONFIG["val_ratio"]
    params["save_model"] = False  # MCCV is evaluation-only
    print(f"MCCV: {rounds} rounds, {1 - val_ratio:.0%}/{val_ratio:.0%} split\n")

    results = []
    for r in range(rounds):
        seed = CONFIG["seed"] + r
        train_idx, val_idx = build_split_indices(
            CONFIG["train_label_csv"], val_ratio=val_ratio, seed=seed)

        print(f"\n{'─'*55}")
        print(f"Round {r + 1}/{rounds}  "
              f"(seed={seed})  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'─'*55}")

        _, acc, ep = train_one_fold(r, train_idx, val_idx, params, device)
        results.append({"round": r, "seed": seed, "val_acc": acc, "best_epoch": ep})
        print(f"  Round {r + 1}: best_acc={acc:.4f} (epoch {ep})")

    return results


def run_single_split(params: Dict, device: torch.device):
    train_idx, val_idx = build_split_indices(
        CONFIG["train_label_csv"], val_ratio=CONFIG["val_ratio"],
        seed=CONFIG["seed"],
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    return train_one_fold(0, train_idx, val_idx, params, device)


# ---------------------------------------------------------------------------
# Coordinate-wise line search
# ---------------------------------------------------------------------------

def line_search(device: torch.device, rounds: int = 3) -> None:
    """Coordinate-wise hyperparameter search.

    Each round: fix 4 params, vary the 5th.
    Total: 5 params × ~5 values × 3 rounds ≈ 75 runs.
    """
    search_space = [
        ("f1",              [8, 12, 16, 20, 24]),
        ("dropout_rate",    [0.2, 0.3, 0.4, 0.5]),
        ("learning_rate",   [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]),
        ("weight_decay",    [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        ("mixup_alpha",     [0.0, 0.1, 0.2, 0.3, 0.4]),
    ]

    best_params = _build_params()
    best_acc = 0.0
    best_state = None

    avg_vals = sum(len(vals) for _, vals in search_space) / len(search_space)
    print(f"Line search: {len(search_space)} params × ~{avg_vals:.0f} values × {rounds} rounds "
          f"≈ {int(len(search_space) * avg_vals * rounds)} runs\n")

    for r in range(rounds):
        print(f"{'='*55}")
        print(f"Round {r+1}/{rounds}")
        print(f"Start: { {k: best_params[k] for k in [p[0] for p in search_space]} }")
        print(f"{'='*55}")

        improved = False

        for param_key, values in search_space:
            best_val_for_param = best_params[param_key]
            print(f"\n  --- {param_key}: trying {values} ---")

            for v in values:
                test_params = best_params.copy()
                test_params[param_key] = v
                if param_key == "f1":
                    test_params["f2"] = test_params["depth"] * v

                print(f"    {param_key}={v} ... ", end="", flush=True)
                state, acc, _ = run_single_split(test_params, device)
                print(f"vacc={acc:.4f}", end="")

                if acc > best_acc:
                    best_acc = acc
                    best_state = state
                    best_val_for_param = v
                    improved = True
                    print(" ★")
                else:
                    print()

            best_params[param_key] = best_val_for_param
            if param_key == "f1":
                best_params["f2"] = best_params["depth"] * best_val_for_param

        if not improved:
            print(f"\n  No improvement in round {r+1}, converged.")
            break

    print(f"\n{'='*55}")
    print(f"Best: { {k: best_params[k] for k in [p[0] for p in search_space]} }")
    print(f"Best val_acc: {best_acc:.4f}")
    if best_state:
        torch.save(best_state, MODEL_DIR / "best_model.pth")
        print(f"Saved to {MODEL_DIR / 'best_model.pth'}")

    # Save search log
    _save_search_csv(search_space, best_params, best_acc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_params() -> Dict:
    return {k: CONFIG[k] for k in [
        "f1", "depth", "f2", "dropout_rate",
        "learning_rate", "weight_decay", "batch_size",
        "label_smoothing", "mixup_alpha", "focal_gamma",
        "grad_clip_norm", "scheduler_t0", "scheduler_t_mult",
        "warmup_epochs", "swa_start", "swa_lr",
        "use_augmented",
    ]}


def _save_search_csv(
    search_space: list, best_params: dict, best_acc: float,
) -> None:
    """Append search result to models/search_results.csv."""
    import datetime
    row = {"timestamp": datetime.datetime.now().isoformat(),
           "best_val_acc": round(best_acc, 6)}
    for key, _ in search_space:
        row[key] = best_params[key]
    df = pd.DataFrame([row])
    csv_path = MODEL_DIR / "search_results.csv"
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"Search log: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="EEG classifier training")

    # Model
    p.add_argument("--f1", type=int, default=CONFIG["f1"])
    p.add_argument("--depth", type=int, default=CONFIG["depth"])
    p.add_argument("--f2", type=int, default=None)
    p.add_argument("--dropout", type=float, default=CONFIG["dropout_rate"])
    # Training
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--weight-decay", type=float, default=CONFIG["weight_decay"])
    p.add_argument("--mixup", type=float, default=CONFIG["mixup_alpha"])
    p.add_argument("--focal-gamma", type=float, default=CONFIG["focal_gamma"],
                   help="Focal loss gamma (0=off, try 2.0)")
    p.add_argument("--label-smoothing", type=float, default=CONFIG["label_smoothing"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--cpu", action="store_true")

    # CV
    p.add_argument("--cv", default=CONFIG["cv_mode"],
                   choices=["kfold", "single", "mccv"])
    p.add_argument("--k-folds", type=int, default=CONFIG["k_folds"])
    p.add_argument("--mccv-rounds", type=int, default=CONFIG["mccv_rounds"])

    # Augmented dataset
    p.add_argument("--augmented", action="store_true", default=False,
                   help="Use offline-augmented training cache (train_cache_augmented.pt)")
    p.add_argument("--no-augmented", action="store_false", dest="augmented")
    p.add_argument("--augment-multiplier", type=int, default=CONFIG["augment_multiplier"],
                   help="Multiplier used for augmented cache (default: 2)")

    # Line search
    p.add_argument("--tune", action="store_true",
                   help="Coordinate-wise hyperparameter search (5 params)")
    p.add_argument("--tune-rounds", type=int, default=3,
                   help="Number of rounds for --tune (default: 3)")

    args = p.parse_args()

    CONFIG["f1"] = args.f1
    CONFIG["depth"] = args.depth
    CONFIG["f2"] = args.f2 if args.f2 else args.depth * args.f1
    CONFIG["dropout_rate"] = args.dropout
    CONFIG["epochs"] = args.epochs
    CONFIG["learning_rate"] = args.lr
    CONFIG["batch_size"] = args.batch_size
    CONFIG["weight_decay"] = args.weight_decay
    CONFIG["mixup_alpha"] = args.mixup
    CONFIG["focal_gamma"] = args.focal_gamma
    CONFIG["label_smoothing"] = args.label_smoothing
    CONFIG["seed"] = args.seed
    CONFIG["use_cpu"] = args.cpu
    CONFIG["cv_mode"] = args.cv
    CONFIG["k_folds"] = args.k_folds
    CONFIG["mccv_rounds"] = args.mccv_rounds
    CONFIG["use_augmented"] = args.augmented
    CONFIG["augment_multiplier"] = args.augment_multiplier

    set_seed(CONFIG["seed"])
    device = get_device(CONFIG["use_cpu"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | F1={CONFIG['f1']} D={CONFIG['depth']} "
          f"F2={CONFIG['f2']} | CV={CONFIG['cv_mode']} | "
          f"Epochs={CONFIG['epochs']}")

    params = _build_params()

    if args.tune:
        line_search(device, rounds=args.tune_rounds)
        return

    if CONFIG["cv_mode"] == "single":
        state, acc, ep = run_single_split(params, device)
        torch.save(state, MODEL_DIR / "best_model.pth")
        print(f"\nBest val_acc: {acc:.4f} (epoch {ep})")
    elif CONFIG["cv_mode"] == "mccv":
        results = run_mccv(params, device)
        accs = [r["val_acc"] for r in results]
        epochs = [r["best_epoch"] for r in results]
        print(f"\n{'='*55}")
        print(f"MCCV ({CONFIG['mccv_rounds']} rounds, "
              f"{1 - CONFIG['val_ratio']:.0%}/{CONFIG['val_ratio']:.0%} split)")
        print(f"  Mean val_acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Median: {np.median(accs):.4f}")
        print(f"  Min: {np.min(accs):.4f}  Max: {np.max(accs):.4f}")
        print(f"  Best epoch mean: {np.mean(epochs):.0f}")
        print(f"  (No models saved — MCCV is for evaluation only)")
        print(f"{'='*55}")
    else:
        results = run_kfold_cv(params, device)
        accs = [r["val_acc"] for r in results]
        print(f"\n{'='*55}")
        print(f"K-Fold CV ({CONFIG['k_folds']} folds)")
        for r in results:
            print(f"  Fold {r['fold']}: {r['val_acc']:.4f} @ ep {r['best_epoch']}")
        print(f"  Mean: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Best: {np.max(accs):.4f}  Worst: {np.min(accs):.4f}")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
