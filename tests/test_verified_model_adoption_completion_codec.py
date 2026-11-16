"""Nested adopted SFT completion pin contract; neutral DATA, no native effects."""

from dataclasses import replace

from test_verified_model_adoption_sft_parents import CURRENT, _adopted, _reference, catalog

from amlas.backends.types import JobHandle
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


def test_nested_adopted_sft_source_has_explicit_completion_pin_codec():
    adopted = _adopted()
    handle = JobHandle(
        "owned-training",
        "sft",
        "arn:aws:sagemaker:us-west-2:" + catalog().account_id + ":training-job/owned-training",
        {
            "runtime_session_id": CURRENT,
            "catalog": catalog(),
            "model_source": adopted,
            "prepared_source": None,
            "operation": "train",
            "recipe_id": "full_v1",
            "request": {"RoleArn": catalog().execution_role_arn},
            "run_ref": _reference(CURRENT, "RUN.json"),
            "training_record_count": 1000,
            "staged_refs": dict(adopted.file_refs),
        },
    )
    try:
        pin = ToolBindingBackend._producer_handle_pin("sft", handle)
    except ValueError:
        raise AssertionError("nested adopted SFT model source lacks an explicit completion codec") from None
    assert pin == ToolBindingBackend._producer_handle_pin("sft", handle)
    original = adopted.original
    handle.extra["model_source"] = replace(
        adopted, adoption_ref=replace(adopted.adoption_ref, version_id="changed")
    )
    assert ToolBindingBackend._producer_handle_pin("sft", handle) != pin
    assert adopted.original == original
