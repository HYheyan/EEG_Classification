"""
EEG classification models.

EEGNet    — original: one-shot spatial collapse (59→1), fixed temporal kernels
EEGNetV2  — progressive spatial reduction (59→15→4), learnable kernel blending
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EEGNet (original)
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """Multi-scale depthwise-separable CNN for EEG.

    Input: (B, 1, 59, 282)
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 59, 282),
        f1: int = 20,
        depth: int = 2,
        f2: int = 40,
        dropout_rate: float = 0.3,
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


# ---------------------------------------------------------------------------
# EEGNetV2 — progressive spatial reduction + learnable kernel mixing
# ---------------------------------------------------------------------------

class EEGNetV2(nn.Module):
    """EEGNet variant with gradual spatial reduction.

    Key differences from EEGNet:
      - Spatial: 59→15→4 region preservation (vs 59→1 collapse)
      - Temporal: 6 learnable kernel branches (vs 4 fixed)
      - Region attention: weighted fusion of spatial regions before FC

    Input: (B, 1, 59, 282)
    Params: ~55K (similar to EEGNet with f1=28)
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 59, 282),
        f1: int = 24,
        depth: int = 2,
        f2: int = 48,
        dropout_rate: float = 0.3,
        use_se: bool = False,
        num_subjects: int = 0,
        subject_embed_dim: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        in_channels, n_electrodes, n_timesteps = input_shape

        # ---- Block 1: multi-scale temporal (6 branches) ---------------------
        # Smaller kernels for fine-grained time scales
        kernels = [3, 7, 15, 31, 65, 129]
        k_per_branch = f1 // len(kernels)
        sizes = [k_per_branch] * (len(kernels) - 1) + [f1 - k_per_branch * (len(kernels) - 1)]

        self.temp_convs = nn.ModuleList([
            nn.Conv2d(in_channels, s, (1, ks), padding=(0, ks // 2), bias=False)
            for s, ks in zip(sizes, kernels)
        ])
        self.bn_temp = nn.BatchNorm2d(f1)
        self.elu = nn.ELU(inplace=True)
        self.drop_t = nn.Dropout(dropout_rate * 0.5)

        # ---- Block 2: progressive spatial reduction -------------------------
        # Stage 1: 59 → 30 (conv stride 2)
        self.sp_conv1 = nn.Conv2d(f1, f1 * 2, (5, 1), padding=(2, 0),
                                   stride=(2, 1), bias=False)
        self.sp_bn1 = nn.BatchNorm2d(f1 * 2)

        # Stage 2: 30 → 15 (conv stride 2)
        self.sp_conv2 = nn.Conv2d(f1 * 2, f2, (5, 1), padding=(2, 0),
                                   stride=(2, 1), bias=False)
        self.sp_bn2 = nn.BatchNorm2d(f2)

        # Stage 3: 15 → 4 (adaptive pooling over electrodes)
        # Output: (B, F2, 4, T)

        # ---- Block 3: temporal processing per region ------------------------
        self.temp_depth = nn.Conv2d(f2, f2, (1, 17), groups=f2,
                                     padding=(0, 8), bias=False)
        self.temp_point = nn.Conv2d(f2, f2, (1, 1), bias=False)
        self.temp_bn = nn.BatchNorm2d(f2)
        self.pool_t = nn.AvgPool2d((1, 4))
        self.drop_t2 = nn.Dropout(dropout_rate)

        # ---- Block 4: region attention + merge ------------------------------
        # Learn per-region importance weights
        self.region_pool = nn.AdaptiveAvgPool2d((4, 1))   # (B, F2, 4, 1)
        self.region_fc = nn.Linear(f2, 1, bias=False)      # per-region scoring
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))

        # ---- Head -----------------------------------------------------------
        fc_in = f2 * 4  # 4 temporal features × F2 channels (regions merged)

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
        # Block 1: multi-scale temporal
        x = torch.cat([conv(x) for conv in self.temp_convs], dim=1)
        x = self.bn_temp(x)
        x = self.elu(x)
        x = self.drop_t(x)                     # (B, F1, 59, 282)

        # Block 2: progressive spatial reduction
        x = self.sp_conv1(x)                   # (B, F1*2, 30, 282)
        x = self.sp_bn1(x)
        x = self.elu(x)

        x = self.sp_conv2(x)                   # (B, F2, 15, 282)
        x = self.sp_bn2(x)
        x = self.elu(x)

        # Adaptive spatial pooling: 15 → 4 regions
        x = F.adaptive_avg_pool2d(x, (4, x.shape[-1]))  # (B, F2, 4, 282)

        # Block 3: temporal processing per region
        residual = x
        x = self.temp_depth(x)
        x = self.temp_point(x)
        x = self.temp_bn(x)
        x = x + residual
        x = self.elu(x)
        x = self.pool_t(x)                     # (B, F2, 4, 70)
        x = self.drop_t2(x)

        # Block 4: region attention weighted fusion
        attn = self.region_pool(x)             # (B, F2, 4, 1)
        attn = attn.squeeze(-1).transpose(1, 2)  # (B, 4, F2)
        attn = self.region_fc(attn).squeeze(-1)   # (B, 4)
        attn = F.softmax(attn, dim=-1)            # (B, 4)
        x = x * attn.unsqueeze(1).unsqueeze(-1)   # (B, F2, 4, 70)
        x = x.sum(dim=2)                          # (B, F2, 70)
        x = x.unsqueeze(2)                        # (B, F2, 1, 70)

        x = self.adaptive_pool(x)              # (B, F2, 1, 4)
        x = torch.flatten(x, 1)                # (B, F2*4)

        if self.subject_embed is not None and subject_idx is not None:
            s = self.subject_embed(subject_idx)
            x = torch.cat([x, s], dim=1)

        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

Net = EEGNet
