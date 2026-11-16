"""Fresh normal import path only; no explicit registration imports or native client."""

from amlas.tools.orchestration_tools import run_ml_capability

# The normal orchestration import must complete before other startup modules.
assert callable(run_ml_capability)
from amlas.config.settings import Settings  # noqa: E402
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend  # noqa: E402

backend = ToolBindingBackend(Settings())
assert type(backend._generation_store._adapter).__name__ == "SageMakerCodingGenerationBackend"
assert type(backend._pair_store._adapter).__name__ == "SageMakerPairedCodingBackend"
assert backend._generation_store._adapter._sm is None
assert backend._generation_store._adapter._s3 is None
assert backend._pair_store._adapter._sm is None
assert backend._pair_store._adapter._s3 is None
state = backend._invocation_state()
assert state["_amlas_generation_store"] is backend._generation_store
assert state["_amlas_pair_store"] is backend._pair_store
print("COLD_NORMAL_CODING_BOOTSTRAP_PASS")
