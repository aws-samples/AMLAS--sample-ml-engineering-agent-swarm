"""Managed-only preparation registration; constructors never create SDK clients."""

from ..ports import CodingPreparationBackend
from ..registry import register_adapter
from .coding_preparation import ProcessingCodingPreparationBackend

register_adapter(CodingPreparationBackend, "processing", ProcessingCodingPreparationBackend, is_default=True)
