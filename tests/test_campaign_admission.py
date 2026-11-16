"""Fixed co-owner admission contracts, neutral metadata only; no live IAM proof."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib

import pytest

ACCOUNT = "111122223333"
REGION = "us-west-2"
DEPLOYMENT = "12345678-1234-4234-9234-00000000bb01"
OWNER = "12345678-1234-4234-9234-00000000bb02"
RUNTIME = f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/amlasRuntime-ABCDEFGHIJ"
ENDPOINT = RUNTIME + "/runtime-endpoint/CAMPAIGN"
POLICY = b'{"Version":"2012-10-17","Statement":[]}'
TRUST = b'{"Version":"2012-10-17","Statement":[]}'


def admission_module():
    try:
        return importlib.import_module("amlas.control_plane.campaign_admission")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.campaign_admission":
            pytest.fail("missing-symbol: campaign_admission.AdmissionGuard", pytrace=False)
        raise


class Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value


class NativeReader:
    """Private injected observation port; no policy interpretation or approval."""

    def __init__(self, observation):
        self.observation = observation
        self.calls = 0

    def read_current(self):
        self.calls += 1
        return copy.deepcopy(self.observation)


def native_fields(ca):
    roles = (
        ca.RoleIncarnation(
            role_arn=f"arn:aws:iam::{ACCOUNT}:role/fixture-owner",
            role_id="AROA" + "X" * 17,
            trust_policy=TRUST,
            trust_sha256=hashlib.sha256(TRUST).hexdigest(),
        ),
    )
    return {
        "account_id": ACCOUNT,
        "region": REGION,
        "deployment_id": DEPLOYMENT,
        "deployment_prefix": "amlas-fixture",
        "runtime_arn": RUNTIME,
        "runtime_execution_role": ca.RoleIncarnation(
            role_arn=f"arn:aws:iam::{ACCOUNT}:role/fixture-runtime",
            role_id="AROA" + "Y" * 17,
            trust_policy=TRUST,
            trust_sha256=hashlib.sha256(TRUST).hexdigest(),
        ),
        "endpoint_arn": ENDPOINT,
        "endpoint_name": "CAMPAIGN",
        "runtime_version": "1",
        "endpoint_live_version": "1",
        "endpoint_target_version": None,
        "runtime_policy": POLICY,
        "runtime_policy_sha256": hashlib.sha256(POLICY).hexdigest(),
        "endpoint_policy": POLICY,
        "endpoint_policy_sha256": hashlib.sha256(POLICY).hexdigest(),
        "co_owner_roles": roles,
    }


def fixture_guard():
    ca = admission_module()
    fields = native_fields(ca)
    manifest = ca.AdmissionManifest(
        **fields, owner_namespace=OWNER, change_control_reference="fixture-freeze-1"
    )
    # This is explicit trusted-code fixture approval, never a native baseline.
    approved = ca.approve_manifest(manifest, approval_reference="fixture-owner-review-1")
    reader = NativeReader(ca.AdmissionObservation(**fields))
    clock = Clock()
    guard = ca.AdmissionGuard(approved, reader, clock=clock, max_age_seconds=30)
    return ca, guard, reader, clock, approved


def test_matching_explicit_baseline_mints_only_ephemeral_coowner_evidence():
    ca, guard, reader, _, approved = fixture_guard()
    witness = guard.verify()
    assert type(witness) is ca.AdmissionWitness
    assert witness.owner_namespace == OWNER
    assert witness.baseline_sha256 == approved.manifest_sha256
    assert reader.calls == 1
    assert not hasattr(witness, "caller_arn")
    assert not hasattr(witness, "approval_policy")
    assert not hasattr(witness, "native_admission")


@pytest.mark.parametrize("approved", [None, {}, {"approved": True}, {"owner_namespace": OWNER}])
def test_missing_or_public_claim_cannot_approve_admission(approved):
    ca = admission_module()
    reader = NativeReader({})
    guard = ca.AdmissionGuard(approved, reader)
    with pytest.raises(ca.AdmissionError):
        guard.verify()
    assert reader.calls == 0


def test_manifest_and_native_projection_do_not_mint_server_wrappers():
    ca, _, reader, _, approved = fixture_guard()
    with pytest.raises(ca.AdmissionError):
        ca.ApprovedAdmissionBaseline(approved.manifest, "fixture-owner-review-1")
    with pytest.raises(ca.AdmissionError):
        ca.AdmissionWitness(owner_namespace=OWNER)
    with pytest.raises(ca.AdmissionError):
        ca.AdmissionGuard(reader.observation, reader).verify()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("account_id", "000000000000"),
        ("region", "us-east-1"),
        ("deployment_id", "12345678-1234-4234-9234-00000000bb03"),
        ("endpoint_arn", RUNTIME + "/runtime-endpoint/DEFAULT"),
        ("runtime_version", "2"),
        ("endpoint_live_version", "2"),
        ("endpoint_target_version", "2"),
        ("runtime_policy", POLICY + b" "),
        ("endpoint_policy", POLICY + b" "),
        ("co_owner_roles", ()),
    ],
)
def test_native_drift_refuses_instead_of_reapproving(field, replacement):
    ca, guard, reader, _, _ = fixture_guard()
    reader.observation = dataclasses.replace(reader.observation, **{field: replacement})
    with pytest.raises(ca.AdmissionError):
        guard.verify()


def test_role_reincarnation_and_coherent_trust_rehash_remain_drift():
    ca, guard, reader, _, _ = fixture_guard()
    role = reader.observation.co_owner_roles[0]
    changed = dataclasses.replace(role, role_id="AROA" + "Z" * 17)
    reader.observation = dataclasses.replace(reader.observation, co_owner_roles=(changed,))
    with pytest.raises(ca.AdmissionError):
        guard.verify()
    trust = TRUST + b" "
    changed = dataclasses.replace(role, trust_policy=trust, trust_sha256=hashlib.sha256(trust).hexdigest())
    reader.observation = dataclasses.replace(reader.observation, co_owner_roles=(changed,))
    with pytest.raises(ca.AdmissionError):
        guard.verify()


def test_witness_requires_fresh_native_comparison_and_bounded_age(monkeypatch):
    ca, guard, reader, clock, _ = fixture_guard()
    witness = guard.verify()
    guard.require(witness)
    assert reader.calls == 2
    reader.observation = dataclasses.replace(reader.observation, endpoint_live_version="2")
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)
    reader.observation = dataclasses.replace(reader.observation, endpoint_live_version="1")
    clock.value += 30
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)
    clock.value = 1000
    monkeypatch.setattr(ca.os, "getpid", lambda: witness.process_id + 1)
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)


@pytest.mark.parametrize("age", [0, -1, 61, True, float("nan"), float("inf")])
def test_witness_age_configuration_is_closed(age):
    ca, _, reader, _, approved = fixture_guard()
    with pytest.raises(ca.AdmissionError):
        ca.AdmissionGuard(approved, reader, max_age_seconds=age)


@pytest.mark.parametrize(
    ("field", "value"), [("issued_at", False), ("issued_at", 990.0), ("expires_at", 1031.0)]
)
def test_changed_witness_time_fields_cannot_extend_the_sealed_interval(field, value):
    ca, guard, _, _, _ = fixture_guard()
    witness = guard.verify()
    object.__setattr__(witness, field, value)
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)


def test_native_check_cannot_outlive_witness_window_or_hide_clock_rollback():
    ca, guard, reader, clock, _ = fixture_guard()
    witness = guard.verify()
    clock.value -= 1
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)
    clock.value = 1000
    original = reader.read_current

    def delayed_native():
        reply = original()
        clock.value += 30
        return reply

    reader.read_current = delayed_native
    with pytest.raises(ca.AdmissionError):
        guard.require(witness)


@pytest.mark.parametrize("missing", [None, b"", True])
def test_blank_or_missing_native_policy_is_unknown_not_an_approved_absence(missing):
    ca, guard, reader, _, _ = fixture_guard()
    reader.observation = dataclasses.replace(reader.observation, runtime_policy=missing)
    with pytest.raises(ca.AdmissionError):
        guard.verify()


def test_wrong_types_and_coherent_scope_changes_are_not_manifest_approval():
    ca = admission_module()
    fields = native_fields(ca)
    with pytest.raises(ca.AdmissionError):
        ca.AdmissionManifest(
            **{**fields, "runtime_version": True},
            owner_namespace=OWNER,
            change_control_reference="fixture-freeze",
        )
    with pytest.raises(ca.AdmissionError):
        ca.approve_manifest({"schema_version": 1, **fields}, approval_reference="fixture-owner-review")
    ca, guard, reader, _, _ = fixture_guard()

    class OtherObservation(ca.AdmissionObservation):
        pass

    reader.observation = OtherObservation(**native_fields(ca))
    with pytest.raises(ca.AdmissionError):
        guard.verify()
