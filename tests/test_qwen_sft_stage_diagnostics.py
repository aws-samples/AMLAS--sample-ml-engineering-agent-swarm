"""Bounded source-owned failure stages expose no native exception content."""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

from amlas.sagemaker_entrypoints import qwen_sft_train as worker
from amlas.training import qwen_sft as core

root = Path(__file__).resolve().parents[1] / "src" / "amlas"


class StageDiagnostics(unittest.TestCase):
    def test_stage_success_preserves_value(self):
        self.assertEqual(core.run_fixed_stage("tokenizer_load", lambda: {"value": 1}), {"value": 1})

    def test_stage_refusal_keeps_only_enum_and_exception_class(self):
        def fail():
            raise RuntimeError("private-value-that-must-never-be-logged")

        with self.assertRaises(core.FixedStageError) as found:
            core.run_fixed_stage("model_load", fail)
        result = json.loads(worker._format_refusal(found.exception))
        self.assertEqual(result["stage"], "model_load")
        self.assertEqual(result["cause_type"], "RuntimeError")
        self.assertNotIn("private-value", json.dumps(result))
        self.assertNotIn("private-value", str(found.exception))

    def test_unadmitted_stage_refuses_before_function(self):
        calls = []
        with self.assertRaises(ValueError):
            core.run_fixed_stage("caller-code-or-path", lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_runtime_fingerprint_keeps_only_version_values(self):
        module = types.SimpleNamespace(__version__="2.14.0+cu130", version=types.SimpleNamespace(cuda="13.0"))
        value = core.fixed_runtime_fingerprint(module)
        self.assertEqual(value["torch_version"], "2.14.0+cu130")
        self.assertEqual(value["cuda_version"], "13.0")
        self.assertEqual(set(value), {"torch_version", "cuda_version"})

    def test_pinned_torch_string_subclass_retains_numeric_version(self):
        class Version(str):
            def __str__(self):
                return "private-value"

        module = types.SimpleNamespace(
            __version__=Version("2.14.0+cu130"), version=types.SimpleNamespace(cuda="13.0")
        )
        value = core.fixed_runtime_fingerprint(module)
        self.assertEqual(value["torch_version"], "2.14.0+cu130")
        self.assertNotIn("private-value", json.dumps(value))

    def test_runtime_fingerprint_never_echoes_unbounded_values(self):
        module = types.SimpleNamespace(
            __version__="private-value", version=types.SimpleNamespace(cuda="private-value")
        )
        value = core.fixed_runtime_fingerprint(module)
        self.assertNotIn("private-value", json.dumps(value))
        self.assertEqual(value["torch_version"], "unknown")

    def test_actual_managed_gpu_failure_reports_only_source_stage(self):
        import os
        from unittest.mock import patch

        from amlas.data_eng import inspection_schema

        class Held:
            alias = "/proc/self/fd/123"

            def __init__(self, *_args):
                pass

            def verify(self):
                pass

            def close(self):
                pass

        def fail_gpu():
            raise RuntimeError("private-driver-error-content")

        fake_torch = types.SimpleNamespace(
            __version__="2.14.0",
            version=types.SimpleNamespace(cuda="13.0"),
            cuda=types.SimpleNamespace(is_available=fail_gpu),
        )
        binding = {
            "recipe_id": "fit_v1",
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "model_files": [],
        }
        with (
            patch.dict(os.environ, {"AMLAS_SOURCE_COMMIT": "a" * 40}),
            patch.object(inspection_schema, "validate_native_privilege_proof", return_value=None),
            patch.object(core, "validate_staged_inputs", return_value={}),
            patch.object(core, "_HeldNativeInputs", Held),
            patch.object(core, "_fixed_output_fd", return_value=3),
            patch.dict(
                sys.modules,
                {"torch": fake_torch, "transformers": types.SimpleNamespace(AutoTokenizer=object)},
            ),
        ):
            with self.assertRaises(core.FixedStageError) as found:
                core.managed_operation(
                    "train",
                    "fit_v1",
                    source_commit="a" * 40,
                    stage_binding=binding,
                    privilege_proof={},
                    stage_manifest_sha256="b" * 64,
                )
        result = json.loads(worker._format_refusal(found.exception))
        self.assertEqual(result["stage"], "gpu_probe")
        self.assertEqual(result["cause_type"], "RuntimeError")
        self.assertNotIn("private-driver", json.dumps(result))

    def test_managed_operation_uses_fixed_stage_hooks(self):
        import ast

        tree = ast.parse((root / "training" / "qwen_sft.py").read_text())
        operation = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "managed_operation"
        )
        stages = {
            node.args[0].value
            for node in ast.walk(operation)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_fixed_stage"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        self.assertTrue(
            {
                "input_verify",
                "base_input_verify",
                "ml_import",
                "gpu_probe",
                "tokenizer_load",
                "train_tokenize",
                "validation_tokenize",
                "model_load",
                "adapter_attach",
                "optimizer_train",
                "greedy_probe",
                "checkpoint_save",
                "reload_load",
                "reload_generate",
            }.issubset(stages)
        )

    def test_structured_template_compatibility_is_preserved(self):
        original = core.ConversationCompatibilityError({"developer_role": 2})

        def fail():
            raise original

        with self.assertRaises(core.ConversationCompatibilityError) as found:
            core.run_fixed_stage("train_tokenize", fail)
        self.assertIs(found.exception, original)
        result = json.loads(worker._format_refusal(original))
        self.assertEqual(result["status"], "model_template_compatibility_refused")
        self.assertEqual(result["reason_counts"], {"developer_role": 2})

    def test_ordinary_exception_remains_generic_and_content_free(self):
        result = json.loads(worker._format_refusal(RuntimeError("private-value")))
        self.assertNotIn("stage", result)
        self.assertNotIn("private-value", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
