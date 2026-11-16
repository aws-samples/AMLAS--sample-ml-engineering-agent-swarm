"""Fixed managed SFT registry; no SDK or admission at construction."""

from ..ports import QwenSftBackend
from ..registry import register_adapter
from .qwen_sft import SageMakerQwenSftBackend

register_adapter(QwenSftBackend, "sagemaker", SageMakerQwenSftBackend, is_default=True)
