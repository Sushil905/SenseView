"""Select the camera view with the smallest absolute yaw estimate."""

from __future__ import annotations

import torch


def select_lowest_yaw(yaw_degrees: torch.Tensor, valid_mask: torch.Tensor | None = None):
    """Return indices and one-hot weights for the most frontal available view.

    ``yaw_degrees`` has shape ``[..., views]``. Missing yaw values (NaN) and
    masked views are ignored. Ties resolve to the first view. If all views are
    invalid, the first view is selected so downstream tensors remain defined.
    """
    if yaw_degrees.ndim < 1 or yaw_degrees.shape[-1] == 0:
        raise ValueError("yaw_degrees must have at least one view on its last axis")
    score = yaw_degrees.abs()
    valid = torch.isfinite(score)
    if valid_mask is not None:
        if valid_mask.shape != yaw_degrees.shape:
            raise ValueError("valid_mask must have the same shape as yaw_degrees")
        valid &= valid_mask.bool()
    score = score.masked_fill(~valid, float("inf"))
    indices = score.argmin(dim=-1)
    no_valid = ~valid.any(dim=-1)
    indices = torch.where(no_valid, torch.zeros_like(indices), indices)
    weights = torch.nn.functional.one_hot(indices, num_classes=yaw_degrees.shape[-1]).to(yaw_degrees.dtype)
    return indices, weights


def select_frontal_view(yaw_degrees, valid_mask=None):
    """Compatibility name for lowest-absolute-yaw selection."""
    return select_lowest_yaw(yaw_degrees, valid_mask)
