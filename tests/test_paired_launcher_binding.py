"""Trusted pair-only launcher selector fixtures; no binary, model or AWS execution."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import test_managed_generation_product as generation_fixtures
from test_managed_generation_product import CASE_REF
from test_paired_coding_backend import packet as packet
from test_paired_coding_backend import submit
from test_paired_coding_product import PairAdapter
from test_qwen_sft_backend import catalog

from amlas.backends.adapters import coding_pair as backend_module
from amlas.control_plane import paired_binding, paired_state
from amlas.control_plane.sft_state import SftCatalog

LAUNCHER = "d" * 64


@pytest.fixture
def generation_state(monkeypatch):
    return generation_fixtures.state.__wrapped__(monkeypatch)


def pair_catalog(execution=None, launcher_sha256=LAUNCHER):
    kind = getattr(paired_state, "PairedCodingCatalog", None)
    assert kind is not None, "pair-only frozen launcher catalog missing"
    return kind(execution=execution or catalog(), launcher_sha256=launcher_sha256)


def startup_env(launcher_sha256=LAUNCHER):
    execution = catalog()
    return {
        "AMLAS_AWS_ACCOUNT_ID": execution.account_id,
        "AMLAS_AWS_REGION": execution.region,
        "AMLAS_SESSION_BUCKET": execution.bucket,
        "SAGEMAKER_EXECUTION_ROLE_ARN": execution.execution_role_arn,
        "AMLAS_CODING_PAIR_IMAGE_URI": execution.image_uri,
        "AMLAS_CODING_PAIR_SOURCE_COMMIT": execution.source_commit,
        "AMLAS_CODING_PAIR_ADMISSION_DIGEST": execution.admission_digest,
        "AMLAS_DEPLOYMENT_ID": execution.deployment_id,
        "AMLAS_DEPLOYMENT_PREFIX": execution.deployment_prefix,
        "AMLAS_CODING_PAIR_LAUNCHER_SHA256": launcher_sha256,
    }


def test_pair_catalog_composes_frozen_sft_without_adding_generic_fields():
    execution = catalog()
    pair = pair_catalog(execution)
    assert pair.execution is execution and pair.launcher_sha256 == LAUNCHER
    assert type(pair.execution) is SftCatalog
    assert "launcher_sha256" not in {field.name for field in dataclasses.fields(SftCatalog)}
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.launcher_sha256 = "e" * 64


def test_pair_startup_keeps_trusted_launcher_digest_private_and_reads_no_sources(
    generation_state, monkeypatch
):
    generation, _adapter, reads, *_ = generation_state
    monkeypatch.setattr(
        paired_binding.REGISTRY, "resolve", lambda port: SimpleNamespace(adapter=PairAdapter())
    )
    store = paired_binding.build_pair_store(generation, generation._cases, env=startup_env())
    assert type(store._catalog) is getattr(paired_state, "PairedCodingCatalog", None)
    assert store._catalog.launcher_sha256 == LAUNCHER and store._catalog.execution == catalog()
    assert reads == []


@pytest.mark.parametrize("missing", ["", "   "])
def test_missing_launcher_is_deferred_until_submit_without_upstream_reads(
    generation_state, monkeypatch, missing
):
    generation, _adapter, reads, *_ = generation_state
    adapter = PairAdapter()
    monkeypatch.setattr(paired_binding.REGISTRY, "resolve", lambda port: SimpleNamespace(adapter=adapter))
    store = paired_binding.build_pair_store(generation, generation._cases, env=startup_env(missing))
    with pytest.raises(paired_state.PairConfigurationMissingError) as caught:
        store.submit(CASE_REF, "gen_" + "a" * 32, "gen_" + "b" * 32, "spec_" + "c" * 32)
    assert "AMLAS_CODING_PAIR_LAUNCHER_SHA256" in caught.value.missing_selectors
    assert reads == [] and adapter.calls == []


@pytest.mark.parametrize("bad", [None, [], True, "0", "e" * 63, "e" * 65, "G" * 64])
def test_malformed_launcher_startup_refuses_before_registry_or_sources(generation_state, monkeypatch, bad):
    generation, _adapter, reads, *_ = generation_state
    resolved = []
    monkeypatch.setattr(
        paired_binding.REGISTRY,
        "resolve",
        lambda port: resolved.append(port) or SimpleNamespace(adapter=PairAdapter()),
    )
    with pytest.raises(ValueError):
        paired_binding.build_pair_store(generation, generation._cases, env=startup_env(bad))
    assert resolved == [] and reads == []


def test_launcher_binding_is_required_even_when_image_source_admission_are_valid(packet):
    backend, inputs, execution, sm, s3, _validated = packet
    with pytest.raises(paired_state.PairConfigurationMissingError):
        backend.submit(
            inputs=inputs,
            catalog=pair_catalog(execution, ""),
            runtime_session_id=inputs.base.runtime_session_id,
        )
    assert sm.calls == [] and s3.calls == []


def test_backend_fingerprints_trusted_expected_digest_without_opening_installed_binary(packet, monkeypatch):
    backend, inputs, execution, _sm, _s3, _validated = packet
    fingerprints = []

    def fingerprint(*, oracle_launcher_sha256):
        fingerprints.append(oracle_launcher_sha256)
        return "d" * 64

    monkeypatch.setattr(backend_module, "paired_runner_sha256", fingerprint)
    handle = backend.submit(
        inputs=inputs, catalog=pair_catalog(execution), runtime_session_id=inputs.base.runtime_session_id
    )
    assert fingerprints == [LAUNCHER]
    assert handle.extra["run_binding"]["oracle_launcher_sha256"] == LAUNCHER
    assert type(handle.extra["catalog"]) is SftCatalog
    assert type(handle.extra["pair_catalog"]) is getattr(paired_state, "PairedCodingCatalog", None)
    assert handle.extra["pair_catalog"].launcher_sha256 == LAUNCHER
    assert handle.extra["request"]["AppSpecification"]["ImageUri"] == execution.image_uri
    assert handle.extra["request"]["Environment"] == {"AMLAS_CODING_PAIR_LAUNCHER_SHA256": LAUNCHER}


@pytest.mark.parametrize("drift", ["catalog_digest", "run_digest", "execution_image"])
def test_retained_expected_launcher_or_execution_drift_refuses_before_describe(packet, drift):
    handle = submit(packet)
    if drift == "catalog_digest":
        handle.extra["pair_catalog"] = dataclasses.replace(
            handle.extra["pair_catalog"], launcher_sha256="e" * 64
        )
    elif drift == "run_digest":
        handle.extra["run_binding"]["oracle_launcher_sha256"] = "e" * 64
    else:
        pair = handle.extra["pair_catalog"]
        handle.extra["pair_catalog"] = dataclasses.replace(
            pair,
            execution=dataclasses.replace(
                pair.execution, image_uri=pair.execution.image_uri.replace("@sha256:", "@sha256:e")
            ),
        )
    before = len(packet[3].calls)
    assert packet[0].status(handle).state.value == "UNKNOWN"
    assert packet[3].calls[before:] == []


def test_trusted_launcher_selector_is_absent_from_public_pair_tool_schemas():
    from amlas.tools.coding_pair import resolve_coding_evaluation_spec, submit_coding_pair

    for tool in (resolve_coding_evaluation_spec, submit_coding_pair):
        properties = tool.tool_spec["inputSchema"]["json"]["properties"]
        assert not {
            "launcher_sha256",
            "oracle_launcher_sha256",
            "AMLAS_CODING_PAIR_LAUNCHER_SHA256",
            "catalog",
            "pair_catalog",
            "tool_context",
        } & set(properties)


@pytest.mark.parametrize("drift", ["absent", "empty", "wrong_digest", "extra_provider", "extra_selector"])
def test_native_launcher_environment_must_equal_the_one_trusted_field(packet, drift):
    handle = submit(packet)

    def change(description):
        if drift == "absent":
            description.pop("Environment", None)
        elif drift == "empty":
            description["Environment"] = {}
        elif drift == "wrong_digest":
            description["Environment"] = {"AMLAS_CODING_PAIR_LAUNCHER_SHA256": "e" * 64}
        else:
            description["Environment"] = {
                "AMLAS_CODING_PAIR_LAUNCHER_SHA256": LAUNCHER,
                "AWS_PROFILE" if drift == "extra_provider" else "AMLAS_OTHER_SELECTOR": "neutral",
            }

    packet[3].drift = change
    before = len(packet[4].calls)
    assert packet[0].status(handle).state.value == "UNKNOWN"
    assert len(packet[4].calls) == before and packet[3].calls.count("create") == 1
