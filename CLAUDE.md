# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a university course project (认知科学导论 / Introduction to Cognitive Science) for **EEG binary classification**. Given EEG samples of shape `(1, 59, 282)` — 1 input channel, 59 EEG channels, 282 time points — the task is to classify each sample as `background` or `target`.

The provided code is a **minimal starter skeleton** using PyTorch. Students are expected to improve the model architecture, training strategy, and validation approach.

## Commands

```bash
# Train the model (reads data/train/ and data/train_labels.csv, saves to models/)
python train.py

# Run inference on test set (reads data/test/, loads models/best_model.pth, outputs res/predictions.csv)
python test.py
```

There is no package manager, linting, or test suite — this is a standalone research-style script collection.

## Architecture

### Data flow

```
data/train/*.npy  +  data/train_labels.csv
        │
        ▼
  load_data.py (EEGDataset: load .npy → normalize → tensor)
        │
        ▼
  train.py (train/val split → DataLoader → model → CrossEntropyLoss)
        │
        ▼
  models/best_model.pth, models/final_model.pth
        │
        ▼
  test.py (load model → inference on data/test/*.npy → res/predictions.csv)
```

### File responsibilities

- **`utils.py`** — `set_seed()` for reproducibility (random, numpy, torch, CUDA); `get_device()` returns CUDA if available, else CPU.
- **`load_data.py`** — `EEGDataset` (torch Dataset): loads `.npy` files, applies per-sample z-score normalization (`(x - mean) / (std + 1e-6)`), maps string labels to ints via `LABEL_TO_INDEX`. `build_split_indices()` creates stratified train/val splits by label. Supports label-less mode for test data (returns filename instead of label).
- **`model.py`** — `Net`: baseline MLP that flattens the 3D EEG input and passes it through `Linear → ReLU → Dropout(0.5) → Linear`. Input dimension is computed as product of `input_shape`.
- **`train.py`** — Full training script: stratified split, balanced class weights for `CrossEntropyLoss`, Adam optimizer with weight decay, saves best model (by val accuracy) and final model.
- **`test.py`** — Inference script: loads `best_model.pth`, runs on test set, writes `res/predictions.csv` with columns `eeg_file` and `prediction`.

### Configuration

Both `train.py` and `test.py` define their own `CONFIG` dict at module level with paths relative to `PROJECT_ROOT`. Key hyperparameters in `train.py`: batch size 64, 30 epochs, lr 1e-3, weight decay 1e-4, val ratio 0.2. There is no shared config file.

### Data

- Training data: `.npy` files in `data/train/`, labels in `data/train_labels.csv` (columns: `eeg_file`, `label`)
- Test data: `.npy` files in `data/test/` (no labels provided)
- Sample shape: `(1, 59, 282)` — the leading `1` is a channel dimension
- Labels are balanced with class weights computed from training data

### Expected output format

`res/predictions.csv` with columns:
```csv
eeg_file,prediction
sub01_sess3_epoch002.npy,background
sub01_sess3_epoch004.npy,target
```