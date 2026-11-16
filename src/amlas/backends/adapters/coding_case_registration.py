"""Fixed managed case registration; constructor is SDK-free and unqualified."""

from ..ports import CodingCaseFreezeBackend
from ..registry import register_adapter
from .coding_case_freeze import ProcessingCodingCaseFreezeBackend

register_adapter(CodingCaseFreezeBackend, "processing", ProcessingCodingCaseFreezeBackend, is_default=True)
