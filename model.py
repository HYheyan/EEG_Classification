"""
EEG classification architectures for binary (background / target) decoding.

Architectures
-------------
- EEGNet          — depthwise-separable Conv2d (Lawhern et al. 2018), ~5K params
- EEGConformer    — CNN temporal stem + Conformer blocks over time, ~180K params
- EnhancedHybridNet — upgraded 3-branch (Temporal + Spatial + Transformer), ~3M params
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utilities
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Two-layer MLP with GeLU, used inside Conformer blocks."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Architecture 1: EEGNet
# =============================================================================

class EEGNet(nn.Module):
    """
    EEGNet from Lawhern et al. (2018), adapted for binary classification.

    Input shape: (1, n_electrodes, n_times) — e.g. (1, 59, 282)

    Parameters
    ----------
    n_electrodes : int
        Number of EEG channels (59).
    n_times : int
        Number of time points (282).
    n_classes : int
        Number of output classes (2).
    F1 : int
        Temporal filter count (block 1).
    D : int
        Depth multiplier for spatial filters.
    F2 : int
        Pointwise filter count (block 2).
    kernel_time : int
        Temporal kernel length for block-1 conv.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_electrodes: int = 59,
        n_times: int = 282,
        n_classes: int = 2,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,
        kernel_time: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        # ---- Block 1 --------------------------------------------------
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_time), padding=(0, kernel_time // 2))
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial conv — collapses electrode dimension
        self.depthwise = nn.Conv2d(
            F1, D * F1, (n_electrodes, 1), groups=F1, bias=False
        )
        self.bn_depth = nn.BatchNorm2d(D * F1)

        self.pool1 = nn.AvgPool2d((1, 4))

        # ---- Block 2 (separable) --------------------------------------
        # The depthwise step of separable conv
        self.sep_depth = nn.Conv2d(
            D * F1, D * F1, (1, 16), groups=D * F1, padding=(0, 8), bias=False
        )
        # The pointwise step of separable conv
        self.sep_point = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F2)

        self.pool2 = nn.AvgPool2d((1, 8))

        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        # ---- Head -----------------------------------------------------
        # Compute the flattened feature size after both blocks.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_electrodes, n_times)
            dummy = self._forward_features(dummy)
            self.feat_dim = dummy.numel()

        self.classifier = nn.Linear(self.feat_dim, n_classes)

        self._init_weights()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Block 1 + Block 2 without dropout or the final classifier."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn_depth(x)
        x = self.elu(x)
        x = self.pool1(x)

        # Block 2
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 59, 282)  → ensure 4-D
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            x = x.squeeze(1)  # (batch, 1, 59, 282) if extra dim

        x = self._forward_features(x)
        x = self.dropout(x)
        x = x.flatten(1)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# =============================================================================
# Architecture 2: EEGConformer
# =============================================================================

class ConformerBlock(nn.Module):
    """
    Single Conformer block (macaron-FeedForward, MHSA, convolution, FFN).

    Operates on (batch, seq_len, dim).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        conv_kernel: int = 31,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(dim)
        self.ffn1 = FeedForward(dim, ffn_expansion, dropout)

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout),
        )

        self.ffn2_norm = nn.LayerNorm(dim)
        self.ffn2 = FeedForward(dim, ffn_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- first half-FFN (macaron) ---
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        # --- multi-head self-attention ---
        residual = x
        x = self.attn_norm(x)
        x = residual + self.attn(x, x, x)[0]

        # --- depthwise convolution ---
        residual = x
        x = self.conv_norm(x)
        x = residual + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # --- second half-FFN (macaron) ---
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))

        return x


class EEGConformer(nn.Module):
    """
    CNN temporal stem + Conformer blocks over time dimension.

    The stem extracts local temporal features, then a stack of Conformer
    blocks models long-range dependencies via self-attention over electrode
    tokens while using depthwise conv for local refinement.

    Input shape: (1, 59, 282)  →  internally reshaped to 4-D.

    Parameters
    ----------
    n_electrodes : int
        Number of EEG channels (59).
    n_times : int
        Number of time points (282).
    n_classes : int
        Number of output classes.
    stem_channels : int
        Output channels of the conv stem.
    d_model : int
        Transformer hidden dimension.
    heads : int
        Multi-head attention heads.
    depth : int
        Number of Conformer blocks.
    ffn_expansion : int
        Expansion factor for feed-forward layers inside Conformer blocks.
    conv_kernel : int
        Kernel size for depthwise conv inside Conformer blocks.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_electrodes: int = 59,
        n_times: int = 282,
        n_classes: int = 2,
        stem_channels: int = 24,
        d_model: int = 64,
        heads: int = 4,
        depth: int = 2,
        ffn_expansion: int = 2,
        conv_kernel: int = 31,
        dropout: float = 0.15,
    ):
        super().__init__()

        # ---- Conv stem ------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(
                1, stem_channels, (1, 7), stride=(1, 2), padding=(0, 3)
            ),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(
                stem_channels, stem_channels, (1, 15), stride=(1, 2), padding=(0, 7)
            ),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(
                stem_channels, stem_channels, (1, 15), stride=(1, 2), padding=(0, 7)
            ),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )

        # Compute actual stem output dimensions with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_electrodes, n_times)
            dummy = self.stem(dummy)  # (1, stem_channels, n_electrodes, T_out)
            _, _, _, t_out = dummy.shape
            self._stem_time = t_out  # e.g. 36

        # Per-electrode feature dim after flattening
        feat_per_electrode = stem_channels * self._stem_time
        self.input_proj = nn.Linear(feat_per_electrode, d_model)

        # ---- Positional encoding + Conformer -------------------------
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=n_electrodes + 10)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, heads, conv_kernel,
                           ffn_expansion=ffn_expansion, dropout=dropout)
            for _ in range(depth)
        ])
        self.post_norm = nn.LayerNorm(d_model)

        # ---- Classifier ----------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 5:
            x = x.squeeze(1)

        batch = x.size(0)

        # (batch, 1, 59, 282) → (batch, stem_channels, 59, ~35)
        x = self.stem(x)

        # Reshape: (batch, stem_channels, 59, t) → (batch, 59, stem_channels * t)
        x = x.permute(0, 2, 1, 3).flatten(2)  # (batch, 59, feat_per_electrode)
        x = self.input_proj(x)  # (batch, 59, d_model)

        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.post_norm(x)

        # Pool over electrode axis
        x = x.mean(dim=1)  # (batch, d_model)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# =============================================================================
# Architecture 3: EnhancedHybridNet (balanced 3-branch, ~150K params)
# =============================================================================
# Design principles (fixing the previous ~3M-param version):
#   1. Convolutions WITH padding — no silent edge truncation.
#   2. Per-branch bottleneck projections — each branch contributes equally
#      to the fused representation (instead of temporal dominating 77 %).
#   3. Dummy forward to compute branch output dims — no hard-coded sizes.
#   4. Lightweight transformer (d_model=56, 2 layers).
#   5. Total ≈ 150 K parameters — comparable to EEGConformer (~180 K).
# =============================================================================


class EnhancedTemporalBranch(nn.Module):
    """3-layer 1D-CNN along time, per-electrode, with padding & bottleneck."""

    def __init__(
        self,
        n_electrodes: int = 59,
        n_times: int = 282,
        channels: tuple = (16, 28, 40),
        kernels: tuple = (11, 7, 5),
        pool_size: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_electrodes = n_electrodes
        # Padding = kernel//2  →  same-length output (before stride-1 conv)
        self.conv1 = nn.Conv1d(1, channels[0], kernels[0],
                               padding=kernels[0] // 2)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernels[1],
                               padding=kernels[1] // 2)
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], kernels[2],
                               padding=kernels[2] // 2)
        self.bn3 = nn.BatchNorm1d(channels[2])
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = x.squeeze(1)                               # (B, 59, 282)
        x = x.reshape(batch * self.n_electrodes, 1, -1)  # (B*59, 1, 282)

        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(x)                                # (B*59, C_out, pool_size)

        _, c, t = x.shape
        x = x.reshape(batch, self.n_electrodes, c, t)   # (B, 59, C_out, pool_size)
        x = x.flatten(2)                                 # (B, 59, C_out * pool_size)
        x_mean = x.mean(dim=1)                           # (B, feat)
        x_max = x.max(dim=1)[0]                          # (B, feat)
        return torch.cat([x_mean, x_max], dim=1)


class EnhancedSpatialBranch(nn.Module):
    """3-layer 1D-CNN along electrode axis, per-time-point, with padding."""

    def __init__(
        self,
        n_electrodes: int = 59,
        n_times: int = 282,
        channels: tuple = (12, 20, 28),
        kernels: tuple = (5, 5, 3),
        pool_size: int = 6,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_times = n_times
        self.conv1 = nn.Conv1d(1, channels[0], kernels[0],
                               padding=kernels[0] // 2)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernels[1],
                               padding=kernels[1] // 2)
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], kernels[2],
                               padding=kernels[2] // 2)
        self.bn3 = nn.BatchNorm1d(channels[2])
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = x.squeeze(1)                    # (B, 59, 282)
        x = x.permute(0, 2, 1)              # (B, 282, 59)
        x = x.reshape(-1, 1, 59)            # (B*282, 1, 59)

        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(x)                     # (B*282, C_out, pool_size)

        _, c, psize = x.shape
        x = x.reshape(batch, self.n_times, c * psize)  # (B, 282, feat_per_tp)
        x_mean = x.mean(dim=1)                          # (B, feat)
        x_max = x.max(dim=1)[0]                         # (B, feat)
        return torch.cat([x_mean, x_max], dim=1)


class EnhancedTransformerBranch(nn.Module):
    """Lightweight transformer over electrodes with CLS-token readout."""

    def __init__(
        self,
        n_electrodes: int = 59,
        d_raw: int = 282,
        d_model: int = 56,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.token_proj = nn.Linear(d_raw, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=n_electrodes + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        x = x.squeeze(1)                                # (B, 59, 282)
        x = self.token_proj(x)                           # (B, 59, d_model)

        cls = self.cls_token.expand(batch, -1, -1)       # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                   # (B, 60, d_model)

        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0, :]                                 # CLS readout


class EnhancedHybridNet(nn.Module):
    """
    Balanced 3-branch EEG classifier (~150 K parameters).

    Branches
    --------
    1. **Temporal**  — 3-layer padded Conv1d per electrode → mean+max pool
    2. **Spatial**   — 3-layer padded Conv1d per time point → mean+max pool
    3. **Transformer** — lightweight self-attention over electrodes + CLS

    Each branch is projected to the same dimensionality before fusion so
    that no single branch dominates the classifier input.
    """

    def __init__(
        self,
        input_shape: tuple = (1, 59, 282),
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        _, n_electrodes, n_times = input_shape

        # ---- branches ----------------------------------------------------
        self.temporal = EnhancedTemporalBranch(
            n_electrodes=n_electrodes, n_times=n_times, dropout=dropout,
        )
        self.spatial = EnhancedSpatialBranch(
            n_electrodes=n_electrodes, n_times=n_times, dropout=dropout,
        )
        self.transformer = EnhancedTransformerBranch(
            n_electrodes=n_electrodes, d_raw=n_times, dropout=dropout,
        )

        # ---- compute actual output dims (no more hard-coding!) -----------
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            t_dim = self.temporal(dummy).size(1)
            s_dim = self.spatial(dummy).size(1)
            tr_dim = self.transformer(dummy).size(1)

        # ---- per-branch bottleneck → equal contribution ------------------
        proj_dim = 64
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.s_proj = nn.Sequential(
            nn.Linear(s_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.tr_proj = nn.Sequential(
            nn.Linear(tr_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
        )

        fused_dim = proj_dim * 3  # 192

        # ---- lightweight classifier --------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.6),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

        # report dimensions for transparency
        self._branch_dims = {
            "temporal_raw": t_dim, "spatial_raw": s_dim,
            "transformer_raw": tr_dim, "proj_dim": proj_dim,
            "fused_dim": fused_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.t_proj(self.temporal(x))
        s = self.s_proj(self.spatial(x))
        tr = self.tr_proj(self.transformer(x))
        return self.classifier(torch.cat([t, s, tr], dim=1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")


# =============================================================================
# Model registry
# =============================================================================

MODEL_REGISTRY: Dict[str, type] = {
    "eegnet": EEGNet,
    "conformer": EEGConformer,
    "enhanced": EnhancedHybridNet,
}

# Backward-compatible alias
Net = EnhancedHybridNet


def create_model(
    name: str,
    input_shape: tuple = (1, 59, 282),
    num_classes: int = 2,
) -> nn.Module:
    """
    Factory: instantiate a model by name.

    Parameters
    ----------
    name : str
        One of ``"eegnet"``, ``"conformer"``, ``"enhanced"``.
    input_shape : tuple
        Shape of a single EEG sample ``(C, electrodes, time)``.
    num_classes : int
        Number of output classes.

    Returns
    -------
    model : nn.Module
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}."
        )
    cls = MODEL_REGISTRY[name]

    # EEGNet and Conformer need explicit electrode / time counts
    if name in ("eegnet", "conformer"):
        return cls(
            n_electrodes=input_shape[1],
            n_times=input_shape[2],
            n_classes=num_classes,
        )
    return cls(input_shape=input_shape, num_classes=num_classes)
