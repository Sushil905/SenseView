from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16

from ..fusion.frontal_view_selector import select_lowest_yaw


class LegacyQualityAwareMultiViewFER(nn.Module):
    """Shared visual encoder, learned view-quality fusion, temporal Transformer."""
    def __init__(self, n_classes: int = 7, pretrained: bool = True, hidden_dim: int = 256, heads: int = 8, tiny: bool = False):
        super().__init__()
        self.tiny = tiny
        if tiny:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            encoder_dim = 64
        else:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            encoder_dim = backbone.fc.in_features
        self.projection = nn.Linear(encoder_dim, hidden_dim)
        self.quality = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        if tiny:
            self.temporal = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=hidden_dim * 4, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, frames: torch.Tensor, view_mask: torch.Tensor | None = None):
        # frames: B, T, V, C, H, W. view_mask True where an actual view exists.
        b, t, v, c, h, w = frames.shape
        embedded = self.encoder(frames.reshape(b * t * v, c, h, w)).flatten(1)
        embedded = self.projection(embedded).reshape(b, t, v, -1)
        quality_logits = self.quality(embedded).squeeze(-1)
        if view_mask is not None:
            quality_logits = quality_logits.masked_fill(~view_mask.bool(), float("-inf"))
        view_weights = quality_logits.softmax(dim=-1)
        fused = (embedded * view_weights.unsqueeze(-1)).sum(dim=2)
        context = self.temporal(fused)
        return {"logits": self.classifier(context), "view_weights": view_weights}


class ViTBaseFER(nn.Module):
    """ImageNet ViT-Base/16 backbone with a seven-expression output head."""

    def __init__(self, n_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        feature_dim = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Identity()
        self.classifier = nn.Linear(feature_dim, n_classes)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))


class ViTBaseLSTM(nn.Module):
    """ViT-Base frame embeddings, lowest-yaw view selection, and temporal LSTM."""

    architecture = "vit_base_lstm_lowest_yaw"

    def __init__(self, n_classes: int = 7, pretrained: bool = True,
                 lstm_hidden_dim: int = 256, lstm_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.fer = ViTBaseFER(n_classes=n_classes, pretrained=pretrained)
        self.temporal = nn.LSTM(
            input_size=768, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden_dim * 2, n_classes))

    def forward(self, frames: torch.Tensor, yaw_degrees: torch.Tensor | None = None,
                valid_mask: torch.Tensor | None = None):
        # Accept [B,T,C,H,W] single-view clips and [B,T,V,C,H,W] multi-view clips.
        if frames.ndim == 5:
            frames = frames.unsqueeze(2)
        if frames.ndim != 6:
            raise ValueError("frames must have shape [B,T,C,H,W] or [B,T,V,C,H,W]")
        batch, timesteps, views, channels, height, width = frames.shape
        if yaw_degrees is None:
            if views > 1:
                raise ValueError("yaw_degrees is required to select among multiple camera views")
            yaw_degrees = torch.zeros((batch, timesteps, views), device=frames.device)
        if tuple(yaw_degrees.shape) != (batch, timesteps, views):
            raise ValueError("yaw_degrees must have shape [B,T,V]")
        has_valid_view = torch.isfinite(yaw_degrees)
        if valid_mask is not None:
            has_valid_view &= valid_mask.bool()
        if not has_valid_view.any(dim=-1).all():
            raise ValueError("each timestep needs at least one detected view with a yaw estimate")
        selected_indices, view_weights = select_lowest_yaw(yaw_degrees, valid_mask)
        batch_indices = torch.arange(batch, device=frames.device)[:, None]
        time_indices = torch.arange(timesteps, device=frames.device)[None, :]
        selected = frames[batch_indices, time_indices, selected_indices]
        embeddings = self.fer.encode(selected.reshape(batch * timesteps, channels, height, width))
        embeddings = embeddings.reshape(batch, timesteps, -1)
        temporal_features, _ = self.temporal(embeddings)
        return {
            "logits": self.classifier(temporal_features),
            "view_weights": view_weights,
            "selected_view_indices": selected_indices,
        }


# Existing local demo checkpoints use the earlier ResNet/quality-fusion model.
QualityAwareMultiViewFER = LegacyQualityAwareMultiViewFER
