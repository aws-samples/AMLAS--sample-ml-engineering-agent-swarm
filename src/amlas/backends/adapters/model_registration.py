"""Managed-only fixed model acquisition registration; no SDK at construction."""

from ..ports import ModelWeightAcquisitionBackend
from ..registry import register_adapter
from .model_weight_acquisition import ProcessingModelWeightAcquisitionBackend

register_adapter(
    ModelWeightAcquisitionBackend,
    "processing",
    ProcessingModelWeightAcquisitionBackend,
    is_default=True,
)
