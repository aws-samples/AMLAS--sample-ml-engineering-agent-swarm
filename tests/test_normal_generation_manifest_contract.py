"""Fresh cold fixture uses the actual generation binding validator unchanged."""

import asyncio
import hashlib

import pytest
from model_acquisition_fixtures import BUCKET
from normal_sft_cold_fixture import close, setup
from test_runtime_prompt_entry import inert_runtime_setup

from amlas.backends.adapters import coding_generation
from amlas.training.qwen_sft import validate_generation_binding


def test_fresh_cold_fixture_generation_manifest_passes_actual_binding_validator(tmp_path):
    async def scenario():
        patch = pytest.MonkeyPatch()
        state = None
        try:
            inert_runtime_setup.__wrapped__(patch)
            state = setup(patch, tmp_path)
            value = {
                "schema_version": 1,
                "operation": "generate",
                "catalog_id": "qwen3_5_9b",
                "catalog_version": "1",
                "recipe_id": "humaneval_greedy_v1",
                "recipe_version": "1",
                "source_commit": "a" * 40,
                "evaluation_id": "declared-source-generation",
                "arm": "base",
                "spec_sha256": "b" * 64,
                "contract_sha256": coding_generation.CONTRACT_SHA256,
                "model_files": coding_generation.expected_model_weight_manifest("qwen3_5_9b", "1")["files"],
                "prompts_ref": {
                    "uri": "s3://" + BUCKET + "/sessions/declared-source/cases/PROMPTS.jsonl",
                    "version_id": "declared-source-version",
                    "sha256": hashlib.sha256(b"declared").hexdigest(),
                    "byte_count": len(b"declared"),
                    "format": "jsonl",
                },
            }
            try:
                observed = validate_generation_binding(value, source_commit="a" * 40)
            except ValueError:
                raise AssertionError("source_fixture_manifest_binding_refused") from None
            assert observed == value and observed is not value
        finally:
            if state is not None:
                await close(state)
            patch.undo()

    asyncio.run(scenario())
