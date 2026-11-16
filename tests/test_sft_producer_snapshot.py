"""Closed producer snapshots and true cold normal-API observation; neutral fakes."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parent / "fixtures/sft_producer_cold_graph.py"
SPEC = importlib.util.spec_from_file_location("cold_sft_producer_fixture", PATH)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def product():
    return importlib.import_module("amlas.control_plane.producer_snapshot")


@pytest.fixture
def original(monkeypatch, tmp_path):
    return fixture.seed(monkeypatch, tmp_path)


def test_closed_snapshot_publishes_once_and_restores_original_scoped_refs(original):
    module = product()
    repo = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    first = repo.publish_from_store(original.sft, original.original["request_ref"])
    second = repo.publish_from_store(original.sft, original.original["request_ref"])
    assert first == second and first["published"] is True
    restored = repo.load_original(original.original["job_name"])
    assert restored.original_sft_ref == original.original["request_ref"]
    assert (
        restored.original_model_ref == fixture.MODEL_REF
        and restored.original_preparation_ref == fixture.PREP_REF
    )
    encoded = original.table.items[first["record_key"]]["snapshot"]["B"]
    assert not any(
        value in encoded
        for value in (
            b"_byte_proofs",
            b"approval_policy",
            b"credential",
            b"client",
            b"ready",
            b"training_ready",
        )
    )


def test_true_cold_graph_normal_api_observes_original_with_fresh_adapters_and_zero_mutations(
    original, tmp_path
):
    module = product()
    repo = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    repo.publish_from_store(original.sft, original.original["request_ref"])
    state = tmp_path / "neutral-state.json"
    state.write_text(json.dumps(fixture.export_state(original)))
    result = subprocess.run(
        [sys.executable, str(PATH), str(state)],
        capture_output=True,
        text=True,
        timeout=45,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["result"]["job_name"] == original.original["job_name"]
    assert observed["mutations"] == 0 and observed["new_adapter_count"] == 3
    assert observed["model_reads"] > 0 and observed["prep_reads"] > 0 and observed["sft_reads"] > 0


@pytest.mark.parametrize(
    "case",
    ["unknown_schema", "missing_model", "changed_request", "foreign_session", "foreign_role", "extra_policy"],
)
def test_unknown_missing_changed_or_foreign_snapshot_refuses_without_replacement(original, case):
    module = product()
    repo = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    selected = repo.publish_from_store(original.sft, original.original["request_ref"])
    row = original.table.items[selected["record_key"]]
    if case == "foreign_role":
        original.campaign._sts = type(
            "Foreign",
            (),
            {
                "get_caller_identity": lambda self: {
                    "Account": "0" * 12,
                    "UserId": "foreign",
                    "Arn": "foreign",
                }
            },
        )()
    else:
        raw = json.loads(row["snapshot"]["B"])
        if case == "unknown_schema":
            raw["schema_version"] = 99
        elif case == "missing_model":
            raw.pop("model")
        elif case == "changed_request":
            raw["model"]["request_json"] = "{}"
        elif case == "foreign_session":
            raw["binding"]["runtime_session_id"] = "foreign-session"
        else:
            raw["policy"] = {"allowlist": ["submit_qwen_sft"]}
        row["snapshot"]["B"] = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    before = len(original.native.calls)
    with pytest.raises((ValueError, TypeError)):
        repo.load_original(original.original["job_name"])
    assert len(original.native.calls) == before


def test_no_context_stays_invocation_local_and_public_context_dict_does_not_attach(original):
    module = product()
    from amlas.config.settings import Settings
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    first = ToolBindingBackend(Settings())
    second = ToolBindingBackend(Settings())
    assert first._acquisition_store.runtime_session_id != second._acquisition_store.runtime_session_id
    with pytest.raises((ValueError, TypeError)):
        with module.activate_sft_campaign({"job_name": original.original["job_name"]}):
            ToolBindingBackend(Settings())


@pytest.mark.parametrize("kind", ["missing", "ask", "revoked"])
def test_hydration_requires_fresh_policy_before_any_upstream_read(original, monkeypatch, kind):
    module = product()
    from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
    from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend
    from amlas.backends.adapters.qwen_sft import SageMakerQwenSftBackend
    from amlas.backends.ports import CodingPreparationBackend, ModelWeightAcquisitionBackend, QwenSftBackend
    from amlas.backends.registry import REGISTRY
    from amlas.config.settings import Settings
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    repository = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    repository.publish_from_store(original.sft, original.original["request_ref"])
    context = repository.load_original(original.original["job_name"])
    old = REGISTRY.resolve

    def resolve(port, *args, **kwargs):
        if port is ModelWeightAcquisitionBackend:
            adapter = ProcessingModelWeightAcquisitionBackend(
                sagemaker_client=original.model_native, s3_client=original.model_s3
            )
        elif port is CodingPreparationBackend:
            adapter = ProcessingCodingPreparationBackend(
                sagemaker_client=original.prep_native, s3_client=original.prep_s3
            )
        elif port is QwenSftBackend:
            adapter = SageMakerQwenSftBackend(sagemaker_client=original.native, s3_client=original.sft_s3)
        else:
            return old(port, *args, **kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(adapter=adapter)

    monkeypatch.setattr(REGISTRY, "resolve", resolve)
    policy = (
        None
        if kind == "missing"
        else ApprovalPolicy(
            mode=ApprovalMode.ASK if kind == "ask" else ApprovalMode.ALLOWLIST,
            allowlist=[],
            interactive=False,
        )
    )
    before = (
        len(original.model_s3.reads),
        len(original.prep_s3.calls),
        original.native.reads,
        len(original.table.calls),
    )
    with module.activate_sft_campaign(context), pytest.raises(ValueError, match="policy"):
        ToolBindingBackend(Settings(), approval_policy=policy)
    assert (
        len(original.model_s3.reads),
        len(original.prep_s3.calls),
        original.native.reads,
        len(original.table.calls),
    ) == before


@pytest.mark.parametrize("field", ["integer_bool", "boolean_number", "integer_float"])
def test_closed_request_types_refuse_coherent_rehash_before_artifact_reads(original, field):
    import hashlib

    module = product()
    repo = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    selected = repo.publish_from_store(original.sft, original.original["request_ref"])
    row = original.table.items[selected["record_key"]]
    body = json.loads(row["snapshot"]["B"])
    request = json.loads(body["model"]["request_json"])
    if field == "integer_bool":
        request["ProcessingResources"]["ClusterConfig"]["InstanceCount"] = True
    elif field == "boolean_number":
        request["NetworkConfig"]["EnableNetworkIsolation"] = 0
    else:
        request["StoppingCondition"]["MaxRuntimeInSeconds"] = 7200.0
    body["model"]["request_json"] = json.dumps(request, sort_keys=True, separators=(",", ":"))
    row["snapshot"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    row["snapshot_sha256"]["S"] = hashlib.sha256(row["snapshot"]["B"]).hexdigest()
    before = (len(original.model_s3.reads), len(original.prep_s3.calls), original.native.reads)
    with pytest.raises(ValueError):
        repo.load_original(original.original["job_name"])
    assert (len(original.model_s3.reads), len(original.prep_s3.calls), original.native.reads) == before


def test_changed_alias_map_under_original_operation_never_publishes_new_row(original):
    module = product()
    repo = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    selected = repo.publish_from_store(original.sft, original.original["request_ref"])
    encoded = original.table.items[selected["record_key"]]["snapshot"]["B"]
    alternative = "model_acq_" + "9" * 32
    original.sft._models._records[alternative] = original.sft._models._records[fixture.MODEL_REF]
    original.sft._source_refs[original.original["request_ref"]] = (alternative, fixture.PREP_REF)
    before = len([op for op, _ in original.table.calls if op == "PutItem"])
    with pytest.raises(ValueError, match="aliases"):
        repo.publish_from_store(original.sft, original.original["request_ref"])
    assert len([op for op, _ in original.table.calls if op == "PutItem"]) == before
    assert original.table.items[selected["record_key"]]["snapshot"]["B"] == encoded


@pytest.mark.parametrize("case", ["model_catalog_digest", "preparation_profile", "commitpack_counts"])
def test_closed_producer_semantics_refuse_coherent_changes_without_artifact_reads(original, case):
    import hashlib

    module = product()
    repo = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    selected = repo.publish_from_store(original.sft, original.original["request_ref"])
    row = original.table.items[selected["record_key"]]
    body = json.loads(row["snapshot"]["B"])
    if case == "model_catalog_digest":
        body["model"]["expected_manifest_sha256"] = "f" * 64
    elif case == "commitpack_counts":
        body["preparation"]["source_row_counts"] = {}
    else:
        body["preparation"]["header"]["profile_id"] = "unsupported-profile"
        request = json.loads(body["preparation"]["request_json"])
        args = request["AppSpecification"]["ContainerArguments"]
        args[args.index("--profile-id") + 1] = "unsupported-profile"
        body["preparation"]["request_json"] = json.dumps(request, sort_keys=True, separators=(",", ":"))
    row["snapshot"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    row["snapshot_sha256"]["S"] = hashlib.sha256(row["snapshot"]["B"]).hexdigest()
    before = (len(original.model_s3.reads), len(original.prep_s3.calls), original.native.reads)
    with pytest.raises(ValueError):
        repo.load_original(original.original["job_name"])
    assert (len(original.model_s3.reads), len(original.prep_s3.calls), original.native.reads) == before


@pytest.mark.parametrize("after", [False, True])
def test_one_publication_attempt_reconciles_lost_response_only_by_exact_readback(
    original, monkeypatch, after
):
    module = product()
    repo = module.ProducerSnapshotRepository(
        original.campaign, original.binding, dynamodb_client=original.table
    )
    actual = original.table.put_item
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        if after:
            actual(**kwargs)
        raise TimeoutError("neutral lost publication acknowledgement")

    monkeypatch.setattr(original.table, "put_item", fail)
    if after:
        outcome = repo.publish_from_store(original.sft, original.original["request_ref"])
        assert (
            outcome["published"]
            and repo.load_original(original.original["job_name"]).original_sft_ref
            == original.original["request_ref"]
        )
    else:
        with pytest.raises(ValueError, match="unresolved"):
            repo.publish_from_store(original.sft, original.original["request_ref"])
        with pytest.raises(ValueError, match="missing"):
            repo.load_original(original.original["job_name"])
    assert len(calls) == 1 and len(original.native.calls) == 1


def test_concurrent_publishers_retain_one_original_immutable_snapshot(original):
    from concurrent.futures import ThreadPoolExecutor

    module = product()

    def publish():
        return module.ProducerSnapshotRepository(
            original.campaign, original.binding, dynamodb_client=original.table
        ).publish_from_store(original.sft, original.original["request_ref"])

    before = len(original.table.items)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    assert results[0] == results[1] and len(original.table.items) == before + 1
    assert len(original.native.calls) == 1


def test_each_hydration_read_rechecks_current_policy_and_clears_temporary_guards(original, monkeypatch):
    from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
    from amlas.control_plane.producer_snapshot import _ApprovedReads

    calls = []

    class Client:
        def head_object(self, **kwargs):
            calls.append(kwargs)
            return {"ContentLength": 1}

    policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"], interactive=False)
    guarded = _ApprovedReads(Client(), policy)
    guarded.head_object(Bucket="neutral-bucket", Key="neutral-key")
    policy.allowlist.clear()
    with pytest.raises(ValueError, match="policy"):
        guarded.head_object(Bucket="neutral-bucket", Key="neutral-key")
    assert len(calls) == 1
    with pytest.raises(ValueError, match="read-only"):
        guarded.put_object()
