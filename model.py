"""
EEG classification models for (1, 59, 282) shaped input.

Provides two architectures:
- EEGNet: Multi-scale depthwise-separable CNN (~21K params with defaults).
  Uses parallel temporal kernels + residual connections for EEG classification.
- EEGClassifier: Deeper CNN + self-attention (~600K params).
  Use when EEGNet still underfits.

Reference:
  Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
  EEG-based Brain-Computer Interfaces", J. Neural Eng., 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EEGNet — multi-scale temporal + residual + 2-layer head
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """Multi-scale depthwise-separable CNN for EEG binary classification.

    Input shape: (batch, 1, n_channels, n_timesteps)  e.g. (B, 1, 59, 282)

    Improvements over vanilla EEGNet
    --------------------------------
    1. Multi-scale temporal kernels (7, 31, 65) in parallel — captures
       high-freq (gamma), mid-freq (alpha/beta), and low-freq (delta/theta)
       components simultaneously.
    2. Residual connection in Block 2 — improves gradient flow and allows
       the network to learn identity mappings where beneficial.
    3. Two-layer classifier head — gives the model more capacity to combine
       extracted features before the final decision.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 59, 282),
        F1: int = 18,
        D: int = 2,
        F2: int = 36,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        _channels, n_electrodes, n_timesteps = input_shape  # (1, 59, 282)

        # Partition F1 across three temporal kernel sizes.
        # Use F1=18 → small=6, medium=6, large=6  (even split)
        f1_small = F1 // 3
        f1_medium = F1 // 3
        f1_large = F1 - f1_small - f1_medium

        # ---- Block 1: multi-scale temporal → spatial ------------------------
        # Three parallel temporal convolutions with different receptive fields.
        # Small kernel (7)  → gamma band (~30+ Hz)
        # Medium kernel (31) → alpha/beta bands (~8-30 Hz)
        # Large kernel (65)  → delta/theta bands (~0.5-8 Hz)
        self.temporal_small = nn.Conv2d(
            1, f1_small, kernel_size=(1, 7), padding=(0, 3), bias=False,
        )
        self.temporal_medium = nn.Conv2d(
            1, f1_medium, kernel_size=(1, 31), padding=(0, 15), bias=False,
        )
        self.temporal_large = nn.Conv2d(
            1, f1_large, kernel_size=(1, 65), padding=(0, 32), bias=False,
        )
        self.bn_temporal = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution — per-feature-map electrode weights.
        self.conv_spatial = nn.Conv2d(
            F1, D * F1, kernel_size=(n_electrodes, 1), groups=F1, bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(D * F1)
        self.elu = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # ---- Block 2: separable temporal + residual -------------------------
        # Depthwise temporal → Pointwise mix, with residual connection.
        # D*F1 and F2 must match for the residual to add directly.
        assert D * F1 == F2, (
            f"Block 2 residual requires D*F1 == F2, got D={D}, F1={F1}, F2={F2}"
        )
        self.conv_sep_depth = nn.Conv2d(
            F2, F2, kernel_size=(1, 17),
            groups=F2, padding=(0, 8), bias=False,
        )
        self.conv_sep_point = nn.Conv2d(
            F2, F2, kernel_size=(1, 1), bias=False,
        )
        self.bn_sep = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # ---- Head: 2-layer classifier ---------------------------------------
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        fc_input = F2 * 8  # e.g. 36 * 8 = 288
        self.fc1 = nn.Linear(fc_input, fc_input // 2)
        self.bn_fc = nn.BatchNorm1d(fc_input // 2)
        self.dropout_fc = nn.Dropout(dropout_rate + 0.1)
        self.fc2 = nn.Linear(fc_input // 2, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, 1, 59, 282)

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
        """
        # ---- Block 1: multi-scale temporal → spatial -----------------------
        # Apply three temporal kernels in parallel, then concatenate.
        out_small = self.temporal_small(x)    # (B, f1_small,  59, 282)
        out_medium = self.temporal_medium(x)  # (B, f1_medium, 59, 282)
        out_large = self.temporal_large(x)    # (B, f1_large,  59, 282)
        x = torch.cat([out_small, out_medium, out_large], dim=1)  # (B, F1, 59, 282)
        x = self.bn_temporal(x)

        # Spatial — collapse electrode dimension
        x = self.conv_spatial(x)              # (B, D*F1, 1, 282)
        x = self.bn_spatial(x)
        x = self.elu(x)
        x = self.pool1(x)                     # (B, D*F1, 1, 70)
        x = self.dropout1(x)

        # ---- Block 2: separable temporal + residual ------------------------
        residual = x                           # (B, F2, 1, 70)
        x = self.conv_sep_depth(x)             # (B, F2, 1, 70)
        x = self.conv_sep_point(x)             # (B, F2, 1, 70)
        x = self.bn_sep(x)
        x = x + residual                       # residual connection
        x = self.elu(x)
        x = self.pool2(x)                      # (B, F2, 1, 8)
        x = self.dropout2(x)

        # ---- Head -----------------------------------------------------------
        x = self.adaptive_pool(x)              # (B, F2, 1, 8)
        x = self.flatten(x)                    # (B, F2*8)
        x = self.fc1(x)                        # (B, F2*4)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        return self.fc2(x)                     # (B, num_classes)


# ---------------------------------------------------------------------------
# EEGClassifier — deeper CNN + self-attention for when more capacity is needed
# ---------------------------------------------------------------------------

class _TemporalBlock(nn.Module):
    """Conv1d block with residual connection."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                                padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.shortcut: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        out = out + identity
        return self.dropout(out)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1)
        return x * scale


class EEGClassifier(nn.Module):
    """Deeper CNN + self-attention classifier for EEG.

    Use when EEGNet underfits.  Includes residual temporal blocks,
    channel attention (SE), and learnable attention pooling.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 59, 282),
        num_classes: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        _ch, n_electrodes, _t = input_shape

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(n_electrodes, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.block1 = _TemporalBlock(64, 64, kernel=7, stride=1, dropout=dropout_rate)
        self.block2 = _TemporalBlock(64, 128, kernel=5, stride=2, dropout=dropout_rate)
        self.block3 = _TemporalBlock(128, 256, kernel=3, stride=2, dropout=dropout_rate)

        self.se_block = _SEBlock(256, reduction=4)

        self.attn_pool = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)                     # (B, 59, 282)
        x = self.spatial_conv(x)             # (B, 64, 282)
        x = self.block1(x)                   # (B, 64, 282)
        x = self.block2(x)                   # (B, 128, 141)
        x = self.block3(x)                   # (B, 256, 71)
        x = self.se_block(x)                 # (B, 256, 71)
        attn_weights = self.attn_pool(x.transpose(1, 2))
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = (x * attn_weights.transpose(1, 2)).sum(dim=2)  # (B, 256)
        return self.classifier(x)


# Backward-compatible alias
Net = EEGNet
