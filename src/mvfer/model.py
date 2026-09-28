"""Backward-compatible import for the multi-view FER model."""

from .fer.vit_fer import QualityAwareMultiViewFER

__all__ = ["QualityAwareMultiViewFER"]
