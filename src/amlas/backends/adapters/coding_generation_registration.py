"""Managed-only generation registration; defer private-state imports until resolution."""

from ..ports import CodingGenerationBackend
from ..registry import register_adapter


def _build_generation_adapter():
    from .coding_generation import SageMakerCodingGenerationBackend

    return SageMakerCodingGenerationBackend()


register_adapter(CodingGenerationBackend, "sagemaker", _build_generation_adapter, is_default=True)
