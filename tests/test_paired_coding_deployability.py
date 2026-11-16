"""Neutral source/deployment fixtures; no image construction or AWS mutation."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shlex
import tomllib
from pathlib import Path

from test_runtime_deployment_advisory import run_runtime
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

ROOT = Path(__file__).resolve().parents[1]


def test_paired_image_packs_fixed_cpu_worker_unprivileged_with_locked_closure():
    recipe = ROOT / "Dockerfile.coding-pair"
    assert recipe.exists(), "fixed AMD64 paired-worker recipe missing"
    text = recipe.read_text()
    assert "--platform=linux/amd64" in text and "@sha256:" in text
    assert "--require-hashes" in text and "--only-binary=:all:" in text
    assert "USER 0:0" in text
    assert 'ENTRYPOINT ["/usr/local/bin/amlas-processing-init", "coding-pair"]' in text
    assert "AMLAS_CODING_PAIR_SOURCE_COMMIT" in text
    assert (
        "COPY --chmod=0555 src/" in text and "COPY --chmod=0555 scripts/coding_pair_worker_native.py" in text
    )
    assert "git clone" not in text and "aws s3" not in text
    wrapper = ROOT / "scripts/coding_pair_worker_native.py"
    assert wrapper.exists()
    assert "amlas.sagemaker_entrypoints.coding_pair" in wrapper.read_text()


def test_final_pair_image_has_no_persistent_python_loader_hint():
    text = (ROOT / "Dockerfile.coding-pair").read_text()
    final = text.rsplit("\nFROM ", 1)[1]
    environment = {}
    for line in final.splitlines():
        if line.startswith("ENV "):
            for item in shlex.split(line[4:]):
                key, value = item.split("=", 1)
                environment[key] = value
    assert "PYTHONPATH" not in environment, "image-owned loader hint refuses before native drop"
    assert {key for key in environment if key.startswith("PYTHON")} == {
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
    }
    assert environment["PYTHONUNBUFFERED"] == environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_pair_build_import_uses_only_fixed_path_under_isolated_python():
    text = (ROOT / "Dockerfile.coding-pair").read_text()
    final = text.rsplit("\nFROM ", 1)[1]
    run = next(line for line in final.splitlines() if line.startswith("RUN /opt/venv/bin/python"))
    assert "RUN /opt/venv/bin/python -I -B -c " in run, (
        "isolated Python ignores the bytecode environment flag"
    )
    assert 'import sys; sys.path.insert(0, "/app/src"); import amlas.sagemaker_entrypoints.coding_pair' in run


def test_pair_build_helper_guards_account_and_native_amd64_before_docker_mutation():
    helper = ROOT / "scripts/lib/build_coding_pair_image.sh"
    assert helper.exists(), "account-guarded paired image build helper missing"
    text = helper.read_text()
    guard = text.index('require_account_build_host "$ACCT" "$REGION"')
    assert guard < text.index("buildx build")
    assert "--platform linux/amd64" in text
    assert "Dockerfile.coding-pair" in text and "AMLAS_SOURCE_COMMIT" in text
    assert "--push" not in text, "reviewable local artifact build must precede separate publish"


def test_selected_pair_and_existing_lifecycle_bindings_reach_runtime_environment(runtime_workspace):
    selected = {
        "AMLAS_RUNTIME_APPROVED_ACTIONS": '["coding_pair_flow"]',
        "CODING_PAIR_BACKEND": "sagemaker",
        "AMLAS_CODING_PAIR_IMAGE_URI": "111122223333.dkr.ecr.us-east-1.amazonaws.com/worker@sha256:"
        + "a" * 64,
        "AMLAS_CODING_PAIR_SOURCE_COMMIT": "b" * 40,
        "AMLAS_CODING_PAIR_ADMISSION_DIGEST": "a" * 64,
        "AMLAS_CODING_PAIR_LAUNCHER_SHA256": "6" * 64,
        "CODING_GENERATION_BACKEND": "sagemaker",
        "AMLAS_CODING_GENERATION_IMAGE_URI": "generation-image",
        "AMLAS_CODING_GENERATION_SOURCE_COMMIT": "c" * 40,
        "AMLAS_CODING_GENERATION_ADMISSION_DIGEST": "d" * 64,
        "CODING_CASE_FREEZE_BACKEND": "processing",
        "AMLAS_CODING_CASE_FREEZE_IMAGE_URI": "case-image",
        "AMLAS_CODING_CASE_FREEZE_SOURCE_COMMIT": "e" * 40,
        "AMLAS_CODING_CASE_FREEZE_ADMISSION_DIGEST": "f" * 64,
        "QWEN_SFT_BACKEND": "sagemaker",
        "AMLAS_QWEN_SFT_IMAGE_URI": "sft-image",
        "AMLAS_QWEN_SFT_SOURCE_COMMIT": "1" * 40,
        "AMLAS_QWEN_SFT_ADMISSION_DIGEST": "2" * 64,
        "MODEL_WEIGHT_ACQUISITION_BACKEND": "processing",
        "AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI": "111122223333.dkr.ecr.us-east-1.amazonaws.com/model@sha256:"
        + "4" * 64,
        "AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT": "3" * 40,
        "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST": "4" * 64,
    }
    runtime_workspace[1].update(selected)
    vendor = os.environ.get("PYTHONPATH", "")
    runtime_workspace[1]["PYTHONPATH"] = str(ROOT / "src") + (":" + vendor if vendor else "")
    runtime_workspace[1]["AMLAS_UNRECOGNIZED_PAYLOAD_POLICY"] = "never-forward"
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    environment = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())["environmentVariables"]
    assert all(environment.get(key) == value for key, value in selected.items())
    assert "AMLAS_UNRECOGNIZED_PAYLOAD_POLICY" not in environment
    assert {
        "AMLAS_AWS_ACCOUNT_ID",
        "AMLAS_AWS_REGION",
        "AMLAS_SESSION_BUCKET",
        "SAGEMAKER_EXECUTION_ROLE_ARN",
        "AMLAS_DEPLOYMENT_PREFIX",
    } <= set(environment)


def test_sagemaker_broker_role_has_exact_fixed_interpreter_lifecycle_not_create_or_delete():
    tree = ast.parse((ROOT / "infra/cdk_stacks/async_training_stack.py").read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_sagemaker_role"
    ]
    assert len(functions) == 1
    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "PolicyStatement"
    ]
    pairs = [{keyword.arg: keyword.value for keyword in call.keywords} for call in calls]
    broker = [
        value
        for value in pairs
        if isinstance(value.get("sid"), ast.Constant)
        and value["sid"].value == "PairedCodeInterpreterLifecycle"
    ]
    assert len(broker) == 1, "Processing broker lacks fixed CodeInterpreter lifecycle policy"
    policy = broker[0]
    actions = ast.literal_eval(policy["actions"])
    assert set(actions) == {
        "bedrock-agentcore:GetCodeInterpreter",
        "bedrock-agentcore:StartCodeInterpreterSession",
        "bedrock-agentcore:InvokeCodeInterpreter",
        "bedrock-agentcore:StopCodeInterpreterSession",
        "bedrock-agentcore:GetCodeInterpreterSession",
    }
    resource = ast.get_source_segment(
        (ROOT / "infra/cdk_stacks/async_training_stack.py").read_text(), policy["resources"]
    )
    assert (
        "aws.codeinterpreter.v1" in resource
        and "self.region_name" in resource
        and ":aws:code-interpreter/" in resource
    )
    conditions = ast.get_source_segment(
        (ROOT / "infra/cdk_stacks/async_training_stack.py").read_text(), policy["conditions"]
    )
    assert "aws:RequestedRegion" in conditions and "self.region_name" in conditions
    assert "*" not in resource


def test_locked_worker_closure_is_in_memory_binary_hashed_and_excludes_gpu():
    target = ROOT / "scripts/paired_worker_requirements.py"
    spec = importlib.util.spec_from_file_location("paired_requirements", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    closure = module.requirements(tomllib.loads((ROOT / "uv.lock").read_text()))
    assert "boto3==" in closure and "pydantic==" in closure and "scikit-learn==" in closure
    assert all("--hash=sha256:" in row and "==" in row for row in closure.splitlines())
    assert "torch==" not in closure and "nvidia" not in closure
