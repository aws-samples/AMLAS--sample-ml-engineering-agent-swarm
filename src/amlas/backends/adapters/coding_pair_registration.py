"""Managed-only paired registration; bootstrap never imports partial conversation state."""

from ..ports import PairedCodingBackend
from ..registry import register_adapter


def _build_paired_adapter():
    from .coding_pair import SageMakerPairedCodingBackend

    return SageMakerPairedCodingBackend()


register_adapter(PairedCodingBackend, "sagemaker", _build_paired_adapter, is_default=True)
