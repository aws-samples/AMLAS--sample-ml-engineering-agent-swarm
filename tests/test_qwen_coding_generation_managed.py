"""Fixed managed generation binding and startup dispatch; source-only fixtures."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest
from amlas.training.coding_generation import CONTRACT_SHA256

SOURCE = "a" * 40


def binding(arm="base"):
    value = {
        "schema_version": 1,
        "operation": "generate",
        "catalog_id": "qwen3_5_9b",
        "catalog_version": "1",
        "recipe_id": "humaneval_greedy_v1",
        "recipe_version": "1",
        "source_commit": SOURCE,
        "evaluation_id": "evaluation_fixture_1",
        "arm": arm,
        "spec_sha256": "b" * 64,
        "contract_sha256": CONTRACT_SHA256,
        "model_files": expected_model_weight_manifest("qwen3_5_9b", "1")["files"],
        "prompts_ref": {
            "uri": "s3://owned-bucket/sessions/session_fixture/case-freeze/owned-case-fixture/output/PROMPTS.jsonl",
            "version_id": "original-prompts-v1",
            "sha256": "c" * 64,
            "byte_count": 1000,
            "format": "jsonl",
        },
    }
    if arm == "tuned":
        value["checkpoint_ref"] = {
            "uri": "s3://owned-bucket/sessions/session_fixture/sft/owned-sft-fixture/output/CHECKPOINT.json",
            "version_id": "original-checkpoint-v1",
            "sha256": "d" * 64,
            "byte_count": 1000,
            "format": "json",
        }
        value["checkpoint_source_commit"] = "e" * 40
    return value


def validator():
    module = importlib.import_module("amlas.training.qwen_sft")
    function = getattr(module, "validate_generation_binding", None)
    assert callable(function), "missing-symbol: validate_generation_binding"
    return function


@pytest.mark.parametrize("arm", ["base", "tuned"])
def test_generation_binding_preserves_image_and_original_checkpoint_source(arm):
    value = binding(arm)
    before = copy.deepcopy(value)
    result = validator()(value, source_commit=SOURCE)
    assert result == before and value == before
    if arm == "tuned":
        assert result["checkpoint_source_commit"] != result["source_commit"]


@pytest.mark.parametrize("field", ["oracle_ref", "policy", "image_uri", "model_path", "approved"])
def test_generation_binding_refuses_extra_authority_or_oracle_fields(field):
    value = binding()
    value[field] = "caller"
    with pytest.raises(ValueError):
        validator()(value, source_commit=SOURCE)


@pytest.mark.parametrize("change", ["contract", "model", "version", "count", "arm", "source"])
def test_generation_binding_refuses_changed_pins_and_malformed_fields(change):
    value = binding()
    if change == "contract":
        value["contract_sha256"] = "f" * 64
    elif change == "model":
        value["model_files"][0]["sha256"] = "f" * 64
    elif change == "version":
        value["prompts_ref"]["version_id"] = "null"
    elif change == "count":
        value["prompts_ref"]["byte_count"] = True
    elif change == "arm":
        value["arm"] = "caller"
    else:
        value["source_commit"] = "f" * 40
    with pytest.raises(ValueError):
        validator()(value, source_commit=SOURCE)


def test_base_arm_cannot_receive_adapter_checkpoint():
    value = binding()
    value["checkpoint_ref"] = binding("tuned")["checkpoint_ref"]
    with pytest.raises(ValueError):
        validator()(value, source_commit=SOURCE)


def _native_proof():
    return {
        "res_uid": [10001] * 3,
        "res_gid": [10001] * 3,
        "groups": [],
        "cap_eff": 0,
        "cap_prm": 0,
        "cap_inh": 0,
        "cap_amb": 0,
        "cap_bnd": 0,
        "no_new_privs": 1,
        "fs_uid": 10001,
        "fs_gid": 10001,
        "keep_caps": 0,
        "seccomp": 2,
        "securebits": 0,
    }


def _worker_fixture(monkeypatch, tmp_path, operation="generate", recipe="humaneval_greedy_v1"):
    worker = importlib.import_module("amlas.sagemaker_entrypoints.qwen_sft_train")
    # Same neutral startup fixture as test_qwen_sft_worker: measured-child guards
    # are observed as fixed fakes, not claims of actual native provider absence.
    for name in worker.FORBIDDEN:
        monkeypatch.delenv(name, raising=False)
    value = binding()
    value["recipe_id"] = recipe
    raw = json.dumps(value, indent=2).encode()
    path = tmp_path / "RUN.json"
    path.write_bytes(raw)
    monkeypatch.setattr(worker, "BINDING", path)
    events, calls = [], []
    proof = _native_proof()
    monkeypatch.setattr(worker, "_guard", lambda: events.append("guard") or proof)
    monkeypatch.setattr(worker, "_guard_output", lambda: events.append("output"))
    monkeypatch.setattr(
        worker, "_provider_observation", lambda stage: events.append(stage) or {"fixture": "same"}
    )

    def load_core():
        events.append("core_import")
        return SimpleNamespace(managed_operation=lambda *args, **kwargs: calls.append((args, kwargs)))

    monkeypatch.setattr(worker, "_load_core", load_core)
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", SOURCE)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen_sft_train.py",
            "--operation",
            operation,
            "--recipe-id",
            recipe,
            "--recipe-version",
            "1",
            "--source-commit",
            SOURCE,
            "--manifest-sha256",
            hashlib.sha256(raw).hexdigest(),
            "--manifest-byte-count",
            str(len(raw)),
        ],
    )
    return worker, path, value, raw, proof, events, calls


def test_worker_generate_routes_exact_raw_binding_after_existing_startup_guards(monkeypatch, tmp_path):
    worker, path, value, raw, proof, events, calls = _worker_fixture(monkeypatch, tmp_path)
    try:
        worker.main()
    except (ValueError, SystemExit) as error:
        pytest.fail("fixed generate startup refused before core: " + type(error).__name__)
    assert events == ["guard", "before_core_import", "output", "core_import", "after_core_import"]
    assert len(calls) == 1 and calls[0][0] == ("generate", "humaneval_greedy_v1")
    assert calls[0][1] == {
        "source_commit": SOURCE,
        "stage_binding": value,
        "privilege_proof": proof,
        "stage_manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        worker.main()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "operation,recipe",
    [
        ("train", "humaneval_greedy_v1"),
        ("reload", "humaneval_greedy_v1"),
        ("generate", "fit_v1"),
    ],
)
def test_worker_cross_operation_recipe_refuses_before_core(monkeypatch, tmp_path, operation, recipe):
    worker, _path, _value, _raw, _proof, events, calls = _worker_fixture(
        monkeypatch, tmp_path, operation, recipe
    )
    with pytest.raises((ValueError, SystemExit)):
        worker.main()
    assert "core_import" not in events and calls == []


def _generation_core():
    module = importlib.import_module("amlas.training.qwen_sft")
    assert callable(getattr(module, "managed_generation", None)), "missing-symbol: managed_generation"
    return module


def test_managed_operation_generate_routes_fixed_binding_without_ml_load(monkeypatch):
    module = _generation_core()
    calls = []
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", SOURCE)
    monkeypatch.setattr(
        module,
        "load_fixed_gpu_model",
        lambda *_a, **_kw: pytest.fail("ML loaded before fixed generation route"),
    )

    def generate(value, **kwargs):
        calls.append((value, kwargs))
        return {"routed": True}

    monkeypatch.setattr(module, "managed_generation", generate)
    value, proof = binding(), _native_proof()
    result = module.managed_operation(
        "generate",
        "humaneval_greedy_v1",
        source_commit=SOURCE,
        stage_binding=value,
        privilege_proof=proof,
        stage_manifest_sha256="f" * 64,
    )
    assert result == {"routed": True}
    assert calls == [
        (value, {"source_commit": SOURCE, "privilege_proof": proof, "stage_manifest_sha256": "f" * 64})
    ]


@pytest.mark.parametrize("gate", ["current_image", "privilege", "raw_manifest"])
def test_generation_shared_admission_refuses_before_generation_or_ml(monkeypatch, gate):
    module = _generation_core()
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", SOURCE)
    monkeypatch.setattr(
        module,
        "managed_generation",
        lambda *_a, **_kw: pytest.fail("generation dispatched before shared gate"),
    )
    monkeypatch.setattr(
        module, "load_fixed_gpu_model", lambda *_a, **_kw: pytest.fail("ML loaded before shared gate")
    )
    source, proof, sha = SOURCE, _native_proof(), "f" * 64
    if gate == "current_image":
        source = "e" * 40
    elif gate == "privilege":
        proof = {"root": True}
    else:
        sha = "invalid"
    with pytest.raises(ValueError):
        module.managed_operation(
            "generate",
            "humaneval_greedy_v1",
            source_commit=source,
            stage_binding=binding(),
            privilege_proof=proof,
            stage_manifest_sha256=sha,
        )


def _generation_publisher():
    module = importlib.import_module("amlas.training.qwen_sft")
    function = getattr(module, "_publish_generation_artifacts", None)
    assert callable(function), "missing-symbol: _publish_generation_artifacts"
    return module, function


def _generation_metadata():
    return {
        "schema_version": 1,
        "operation": "generate",
        "source_commit": SOURCE,
        "case_count": 164,
        "quality_success": False,
        "native_admission": False,
    }


@pytest.mark.parametrize("byte_count", [262145, 12 * 1024**2])
def test_generation_publisher_retains_large_candidates_and_commits_last(monkeypatch, tmp_path, byte_count):
    module, publish = _generation_publisher()
    monkeypatch.setattr(module, "FIXED_OUTPUT", str(tmp_path))
    raw = b'{"source":"' + b"a" * (byte_count - len(b'{"source":""}\n')) + b'"}\n'
    assert len(raw) == byte_count
    order = []
    original = os.fsync

    def sync(fd):
        original(fd)
        name = os.readlink("/proc/self/fd/" + str(fd)).rsplit("/", 1)[-1]
        order.append(name)
        if name == "COMMITTED.json":
            assert (tmp_path / "CANDIDATES.jsonl").exists() and (tmp_path / "GENERATION.json").exists()

    monkeypatch.setattr(os, "fsync", sync)
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        publish(directory, candidates_raw=raw, generation=_generation_metadata())
    finally:
        os.close(directory)
    assert set(path.name for path in tmp_path.iterdir()) == {
        "CANDIDATES.jsonl",
        "GENERATION.json",
        "COMMITTED.json",
    }
    assert (tmp_path / "CANDIDATES.jsonl").read_bytes() == raw
    metadata = json.loads((tmp_path / "GENERATION.json").read_bytes())
    assert metadata["quality_success"] is metadata["native_admission"] is False
    commit = json.loads((tmp_path / "COMMITTED.json").read_bytes())
    assert set(commit["files"]) == {"CANDIDATES.jsonl", "GENERATION.json"}
    assert commit["files"]["CANDIDATES.jsonl"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert commit["files"]["CANDIDATES.jsonl"]["byte_count"] == byte_count
    assert order.index("COMMITTED.json") > order.index("CANDIDATES.jsonl")
    assert order.index("COMMITTED.json") > order.index("GENERATION.json")


@pytest.mark.parametrize("oversize", ["candidates", "metadata"])
def test_generation_publisher_keeps_closed_caps_before_output(monkeypatch, tmp_path, oversize):
    module, publish = _generation_publisher()
    monkeypatch.setattr(module, "FIXED_OUTPUT", str(tmp_path))
    raw, metadata = b"{}\n", _generation_metadata()
    if oversize == "candidates":
        raw = b"a" * (12 * 1024**2 + 1)
    else:
        metadata["oversize"] = "a" * 262144
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            publish(directory, candidates_raw=raw, generation=metadata)
    finally:
        os.close(directory)
    assert list(tmp_path.iterdir()) == []


def test_generation_commit_rechecks_retained_candidate_bytes(monkeypatch, tmp_path):
    module, publish = _generation_publisher()
    monkeypatch.setattr(module, "FIXED_OUTPUT", str(tmp_path))
    original = os.fsync

    def sync(fd):
        original(fd)
        if os.readlink("/proc/self/fd/" + str(fd)).endswith("/COMMITTED.json"):
            with (tmp_path / "CANDIDATES.jsonl").open("r+b", buffering=0) as stream:
                stream.write(b"X")

    monkeypatch.setattr(os, "fsync", sync)
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            publish(directory, candidates_raw=b"{}\n", generation=_generation_metadata())
    finally:
        os.close(directory)
    assert (tmp_path / "COMMITTED.json").read_bytes() == b""


def test_legacy_fixed_json_publication_retains_262kib_limit(monkeypatch, tmp_path):
    module = importlib.import_module("amlas.training.qwen_sft")
    monkeypatch.setattr(module, "FIXED_OUTPUT", str(tmp_path))
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            module._publish_fixed_json(directory, "TRAINING.json", {"text": "a" * 262144})
    finally:
        os.close(directory)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    [ValueError, KeyboardInterrupt, SystemExit],
    ids=["value_error", "keyboard_interrupt", "system_exit"],
)
def test_generation_input_recheck_failure_invalidates_original_commit(monkeypatch, tmp_path, failure):
    module, publish = _generation_publisher()
    monkeypatch.setattr(module, "FIXED_OUTPUT", str(tmp_path))
    observations = []

    def verify_inputs():
        committed = (tmp_path / "COMMITTED.json").exists()
        observations.append(committed)
        if committed:
            raise failure("synthetic fixed source recheck failure")

    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(failure):
            publish(
                directory,
                candidates_raw=b"{}\n",
                generation=_generation_metadata(),
                verify_inputs=verify_inputs,
            )
    finally:
        os.close(directory)
    assert False in observations and True in observations
    assert (tmp_path / "COMMITTED.json").read_bytes() == b""


def _high_level_fixture(monkeypatch, tmp_path, arm):
    from transformers import AutoTokenizer

    from amlas.training import coding_generation as generation_module

    module = _generation_core()
    rows = tuple(
        {
            "case_id": f"HumanEval/{i}",
            "prompt": f'def fn_{i}(x):\n    """Return x."""\n',
            "entry_point": f"fn_{i}",
        }
        for i in range(164)
    )
    raw = b"".join(generation_module.canonical_bytes(row) + b"\n" for row in rows)
    prompts, base, adapter, output = (tmp_path / name for name in ("prompts", "base", "adapter", "output"))
    for directory in (prompts, base, output):
        directory.mkdir()
    (prompts / "PROMPTS.jsonl").write_bytes(raw)
    value = binding(arm)
    value["prompts_ref"].update(sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw))
    checkpoint = None
    if arm == "tuned":
        adapter.mkdir()
        checkpoint = {
            "source_commit": value["checkpoint_source_commit"],
            "files": {
                "adapter_model.safetensors": {"sha256": "1" * 64, "byte_count": 1},
                "adapter_config.json": {"sha256": "2" * 64, "byte_count": 2},
            },
        }
        checkpoint_raw = module._checkpoint_json(checkpoint)
        (adapter / "CHECKPOINT.json").write_bytes(checkpoint_raw)
        value["checkpoint_ref"].update(
            sha256=hashlib.sha256(checkpoint_raw).hexdigest(), byte_count=len(checkpoint_raw)
        )
    for name, target in (
        ("FIXED_PROMPTS", prompts),
        ("FIXED_BASE", base),
        ("FIXED_ADAPTER", adapter),
        ("FIXED_OUTPUT", output),
    ):
        monkeypatch.setattr(module, name, str(target))
    monkeypatch.setenv("AMLAS_SOURCE_COMMIT", SOURCE)
    # Only native ownership/framework boundaries are mocked; actual public-row
    # parser, held prompt FD hashing/name checks, binding and publisher execute.
    monkeypatch.setattr(module, "_require_native_readonly_directory", lambda target: None)
    monkeypatch.setattr(module, "_require_native_readonly_file", lambda descriptor: None)
    held = []

    class Held:
        def __init__(self, target, refs):
            self.target, self.refs, self.alias = str(target), refs, str(target)
            self.verifications, self.closed = 0, False
            held.append(self)

        def verify(self):
            self.verifications += 1

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "_HeldNativeInputs", Held)
    monkeypatch.setattr(module, "_checkpoint_metadata", lambda metadata: None)
    loaded, reloads, tokenizer_calls, generated = [], [], [], []
    model, tokenizer = object(), object()
    monkeypatch.setattr(module, "load_fixed_gpu_model", lambda inputs: loaded.append(inputs) or model)

    def reload(original_model, target, *, expected_ref):
        reloads.append((original_model, str(target), expected_ref))
        return original_model

    monkeypatch.setattr(module, "reload_adapter_checkpoint", reload)

    def load_tokenizer(target, **kwargs):
        tokenizer_calls.append((str(target), kwargs))
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)

    def generate(**kwargs):
        generated.append(kwargs)
        cases = tuple(
            generation_module.CodingGenerationCase(
                row["case_id"],
                "3" * 64,
                "4" * 64,
                1,
                1,
                "generated",
                "def fn(x):\n    return x\n",
                "5" * 64,
                24,
            )
            for row in kwargs["prompts"]
        )
        manifest = module._checkpoint_json({"schema_version": 1, "case_count": 164, "arm": arm})
        return generation_module.CodingGenerationResult(
            kwargs["binding"], CONTRACT_SHA256, cases, manifest, hashlib.sha256(manifest).hexdigest()
        )

    monkeypatch.setattr(generation_module, "generate_humaneval_arm", generate)
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(module, "_fixed_output_fd", lambda: descriptor)
    return (
        module,
        value,
        rows,
        raw,
        output,
        descriptor,
        held,
        loaded,
        reloads,
        tokenizer_calls,
        generated,
        checkpoint,
    )


@pytest.mark.parametrize("arm", ["base", "tuned"])
def test_high_level_generation_consumes_164_public_rows_and_publishes_fixed_files(monkeypatch, tmp_path, arm):
    (
        module,
        value,
        rows,
        _raw,
        output,
        descriptor,
        held,
        loaded,
        reloads,
        tokenizer_calls,
        generated,
        _checkpoint,
    ) = _high_level_fixture(monkeypatch, tmp_path, arm)
    try:
        result = module.managed_generation(
            value, source_commit=SOURCE, privilege_proof=_native_proof(), stage_manifest_sha256="f" * 64
        )
    finally:
        os.close(descriptor)
    assert result["case_count"] == 164 and result["arm"] == arm
    assert result["quality_success"] is result["native_admission"] is False
    assert len(generated) == len(loaded) == len(tokenizer_calls) == 1
    assert (
        generated[0]["prompts"] == rows
        and generated[0]["binding"].prompts_sha256 == value["prompts_ref"]["sha256"]
    )
    assert all(inputs.closed and inputs.verifications >= 2 for inputs in held)
    assert len(held) == (2 if arm == "tuned" else 1)
    assert len(held[0].refs) == 9
    assert (
        tokenizer_calls[0][1]["local_files_only"] is True
        and tokenizer_calls[0][1]["trust_remote_code"] is False
    )
    assert len((output / "CANDIDATES.jsonl").read_bytes().splitlines()) == 164
    assert set(result["files"]) == {"CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json"}
    if arm == "tuned":
        assert value["checkpoint_source_commit"] != SOURCE
        assert len(reloads) == 1 and reloads[0][2] == value["checkpoint_ref"] and len(held[1].refs) == 3
    else:
        assert reloads == []
    commit = json.loads((output / "COMMITTED.json").read_bytes())
    assert commit["quality_success"] is commit["native_admission"] is False


def test_high_level_generation_prompt_pin_refuses_before_framework_load(monkeypatch, tmp_path):
    (
        module,
        value,
        _rows,
        _raw,
        output,
        descriptor,
        _held,
        loaded,
        _reloads,
        tokenizer_calls,
        generated,
        _checkpoint,
    ) = _high_level_fixture(monkeypatch, tmp_path, "base")
    value["prompts_ref"]["sha256"] = "f" * 64
    try:
        with pytest.raises(ValueError, match="public prompt bytes"):
            module.managed_generation(
                value, source_commit=SOURCE, privilege_proof=_native_proof(), stage_manifest_sha256="f" * 64
            )
    finally:
        os.close(descriptor)
    assert loaded == tokenizer_calls == generated == [] and list(output.iterdir()) == []


def test_high_level_generation_original_checkpoint_source_refuses_before_framework_load(
    monkeypatch, tmp_path
):
    (
        module,
        value,
        _rows,
        _raw,
        output,
        descriptor,
        _held,
        loaded,
        _reloads,
        tokenizer_calls,
        generated,
        _checkpoint,
    ) = _high_level_fixture(monkeypatch, tmp_path, "tuned")
    value["checkpoint_source_commit"] = "f" * 40
    try:
        with pytest.raises(ValueError, match="original checkpoint producer source"):
            module.managed_generation(
                value, source_commit=SOURCE, privilege_proof=_native_proof(), stage_manifest_sha256="f" * 64
            )
    finally:
        os.close(descriptor)
    assert loaded == tokenizer_calls == generated == [] and list(output.iterdir()) == []


def test_generation_publication_syncs_directory_before_commit_and_after_final_verification(
    monkeypatch, tmp_path
):
    import stat

    module, publish = _generation_publisher()
    monkeypatch.setattr(module, "FIXED_OUTPUT", str(tmp_path))
    events = []
    original = os.fsync

    def sync(fd):
        original(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append(("directory", frozenset(os.listdir(fd))))
        else:
            events.append(("file", os.readlink("/proc/self/fd/" + str(fd)).rsplit("/", 1)[-1]))

    def verify_inputs():
        events.append(("verify", (tmp_path / "COMMITTED.json").exists()))

    monkeypatch.setattr(os, "fsync", sync)
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        publish(
            directory, candidates_raw=b"{}\n", generation=_generation_metadata(), verify_inputs=verify_inputs
        )
    finally:
        os.close(directory)
    first_commit = next(index for index, event in enumerate(events) if event == ("file", "COMMITTED.json"))
    final_verify = max(index for index, event in enumerate(events) if event == ("verify", True))
    assert any(
        index < first_commit and event == ("directory", frozenset({"CANDIDATES.jsonl", "GENERATION.json"}))
        for index, event in enumerate(events)
    ), "payload namespace not synced before COMMITTED"
    assert any(
        index > final_verify
        and event == ("directory", frozenset({"CANDIDATES.jsonl", "GENERATION.json", "COMMITTED.json"}))
        for index, event in enumerate(events)
    ), "committed namespace not synced after final verification"
