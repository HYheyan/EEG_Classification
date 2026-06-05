"""EEG classification model — EEGNet.

Multi-scale depthwise-separable CNN for EEG binary classification.
Input: (B, 1, 59, 282) — 1 channel, 59 electrodes, 282 time points.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """Multi-scale depthwise-separable CNN for EEG.

    Four parallel temporal kernels (7/31/65/129) capture δ/θ/α/β/γ rhythms
    without explicit bandpass filtering.  Depthwise spatial convolution then
    collapses all electrodes to 1 value per temporal filter.

    Reference:
      Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
      EEG-based Brain-Computer Interfaces", J. Neural Eng., 2018.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 59, 282),
        f1: int = 16,
        depth: int = 2,
        f2: int = 32,
        dropout_rate: float = 0.4,
        use_se: bool = False,
        num_subjects: int = 0,
        subject_embed_dim: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        in_channels, n_electrodes, n_timesteps = input_shape

        # ---- Block 1: multi-scale temporal ---------------------------------
        k = f1 // 4
        sizes = [k, k, k, f1 - 3 * k]
        kernels = [7, 31, 65, 129]

        self.temp_convs = nn.ModuleList([
            nn.Conv2d(in_channels, s, (1, ks), padding=(0, ks // 2), bias=False)
            for s, ks in zip(sizes, kernels)
        ])
        self.bn_temp = nn.BatchNorm2d(f1)

        # Depthwise spatial — collapses all electrodes to 1 value per feature
        self.conv_spatial = nn.Conv2d(
            f1, depth * f1, (n_electrodes, 1), groups=f1, bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(depth * f1)
        self.elu = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ---- Block 2: separable temporal + residual ------------------------
        assert depth * f1 == f2, f"D*F1 != F2 ({depth}*{f1} != {f2})"
        self.conv_depth = nn.Conv2d(
            f2, f2, (1, 17), groups=f2, padding=(0, 8), bias=False,
        )
        self.conv_point = nn.Conv2d(f2, f2, (1, 1), bias=False)
        self.bn_sep = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ---- Head -----------------------------------------------------------
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))
        fc_in = f2 * 8

        self.subject_embed: nn.Embedding | None = None
        if num_subjects > 0:
            self.subject_embed = nn.Embedding(num_subjects, subject_embed_dim)
            fc_in += subject_embed_dim

        hidden = max(fc_in // 3, 32)
        self.fc1 = nn.Linear(fc_in, hidden)
        self.bn_fc = nn.BatchNorm1d(hidden)
        self.drop_fc = nn.Dropout(dropout_rate + 0.1)
        self.fc2 = nn.Linear(hidden, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor,
                subject_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Block 1: multi-scale temporal → spatial collapse
        x = torch.cat([conv(x) for conv in self.temp_convs], dim=1)
        x = self.bn_temp(x)

        x = self.conv_spatial(x)               # (B, F2, 1, 282)
        x = self.bn_spatial(x)
        x = self.elu(x)
        x = self.pool1(x)                      # (B, F2, 1, 70)
        x = self.drop1(x)

        # Block 2: separable temporal + residual
        residual = x
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn_sep(x)
        x = x + residual
        x = self.elu(x)
        x = self.pool2(x)                      # (B, F2, 1, 8)
        x = self.drop2(x)

        # Head
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        if self.subject_embed is not None and subject_idx is not None:
            s = self.subject_embed(subject_idx)
            x = torch.cat([x, s], dim=1)

        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        return self.fc2(x)


Net = EEGNet
