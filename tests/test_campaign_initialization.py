"""Neutral atomic initialization protocol; source fakes prove no native protection."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters
from test_campaign_admission import ACCOUNT, DEPLOYMENT, REGION, fixture_guard
from test_campaign_coordinator import PROMPT, TRANSPORT

TABLE = "fixture-campaign-jobs"
MODEL = Session().get_service_model("dynamodb")


def initialization_module():
    try:
        return importlib.import_module("amlas.control_plane.campaign_initialization")
    except ModuleNotFoundError as error:
        if error.name == "amlas.control_plane.campaign_initialization":
            pytest.fail("missing-symbol: campaign_initialization.DynamoCampaignInitialization", pytrace=False)
        raise


class Transactions:
    """Actual input models + one atomic fake-server lock; no credential isolation claim."""

    def __init__(self):
        self.meta = SimpleNamespace(config=SimpleNamespace(retries={"total_max_attempts": 1}))
        self.items = {}
        self.calls = []
        self.lock = threading.Lock()
        self.lose_write = False
        self.fail_reads = False
        self.before_write = None
        self.read_after_write = None

    def transact_get_items(self, **kwargs):
        validate_parameters(kwargs, MODEL.operation_model("TransactGetItems").input_shape)
        assert set(kwargs) == {"TransactItems"}
        assert len(kwargs["TransactItems"]) == 2
        self.calls.append(("TransactGetItems", copy.deepcopy(kwargs)))
        if self.fail_reads:
            raise TimeoutError("neutral transaction read unavailable")
        with self.lock:
            responses = []
            for value in kwargs["TransactItems"]:
                assert set(value) == {"Get"}
                assert value["Get"]["TableName"] == TABLE
                item = self.items.get(value["Get"]["Key"]["job_name"]["S"])
                responses.append({"Item": copy.deepcopy(item)} if item else {})
            return {"Responses": responses, "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0}}

    def transact_write_items(self, **kwargs):
        validate_parameters(kwargs, MODEL.operation_model("TransactWriteItems").input_shape)
        assert set(kwargs) == {"TransactItems", "ClientRequestToken"}
        assert len(kwargs["TransactItems"]) == 2
        assert 1 <= len(kwargs["ClientRequestToken"]) <= 36
        self.calls.append(("TransactWriteItems", copy.deepcopy(kwargs)))
        if self.before_write is not None:
            hook, self.before_write = self.before_write, None
            hook()
        puts = [value["Put"] for value in kwargs["TransactItems"]]
        for put in puts:
            assert put["TableName"] == TABLE
            assert put["ConditionExpression"] == "attribute_not_exists(job_name)"
        with self.lock:
            if any(put["Item"]["job_name"]["S"] in self.items for put in puts):
                raise ClientError({"Error": {"Code": "TransactionCanceledException"}}, "TransactWriteItems")
            for put in puts:
                self.items[put["Item"]["job_name"]["S"]] = copy.deepcopy(put["Item"])
        if self.read_after_write is not None:
            self.read_after_write()
        if self.lose_write:
            self.lose_write = False
            raise TimeoutError("neutral committed reply unavailable")
        return {"ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0}}

    def put_item(self, **kwargs):
        pytest.fail("initialization cannot use nontransactional creation")

    def update_item(self, **kwargs):
        pytest.fail("initialization cannot use an activation update")


class ProtectionReader:
    def __init__(self, ci, configuration, clock):
        self.ci, self.configuration, self.clock = ci, configuration, clock
        self.calls = 0
        self.fail = False
        self.expired = False

    def read_current(self):
        self.calls += 1
        if self.fail:
            raise TimeoutError("neutral protection evidence unavailable")
        now = self.clock()
        return self.ci.InitializationProtectionEvidence(
            configuration=self.configuration,
            observed_at=now,
            expires_at=now if self.expired else now + 30,
            process_id=os.getpid(),
            reader_binding=self,
        )


def config(ci):
    return ci.InitializationProtectionConfig(
        account_id=ACCOUNT,
        region=REGION,
        deployment_id=DEPLOYMENT,
        table_name=TABLE,
        table_arn=f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{TABLE}",
        table_id="12345678-1234-4234-9234-00000000ee01",
        publisher_role_arn=f"arn:aws:iam::{ACCOUNT}:role/fixture-initializer",
        publisher_role_id="AROA" + "I" * 17,
        protection_evidence_sha256=hashlib.sha256(
            b"fixture independently reviewed namespace boundary"
        ).hexdigest(),
        retention_evidence_sha256=hashlib.sha256(b"fixture independently reviewed retention").hexdigest(),
        initializer_code_sha256=hashlib.sha256(b"fixture fixed transaction publisher").hexdigest(),
    )


def fixture_initializer(tx=None):
    ci = initialization_module()
    ca, guard, reader, clock, approved = fixture_guard()
    tx = tx or Transactions()
    # Explicit trusted-code fixture approval is not native IAM qualification.
    boundary = ci.approve_initialization_boundary(config(ci), approval_reference="fixture-protection-review")
    protection = ProtectionReader(ci, boundary.configuration, clock)
    initializer = ci.DynamoCampaignInitialization(
        guard, boundary, private_transaction_client=tx, boundary_reader=protection, clock=clock
    )
    return ci, ca, initializer, guard, reader, clock, tx, approved, boundary


def binding_of(item):
    return json.loads(item["private_record"]["B"])["binding"]


def split_items(tx):
    campaign = next(value for key, value in tx.items.items() if key.startswith("CAMPAIGN#"))
    handoff = next(value for key, value in tx.items.items() if key.startswith("CAMPAIGN_HANDOFF#"))
    return campaign, handoff


def test_atomic_initialization_commits_original_campaign_and_protected_handoff():
    _, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    observed = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    c, h = split_items(tx)
    assert len(tx.items) == 2
    assert binding_of(c) == binding_of(h)
    assert observed.campaign_id == binding_of(h)["campaign_id"]
    assert observed.private_store_correlation == binding_of(h)["private_store_correlation"]
    assert observed.literal_prompt == PROMPT
    assert observed.native_admission is False and observed.action_approved is False
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1


def test_new_initializer_object_recovers_exact_committed_original_without_second_write():
    ci, _, initializer, guard, _, _, tx, _, boundary = fixture_initializer()
    first = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    cold = ci.DynamoCampaignInitialization(
        guard,
        boundary,
        private_transaction_client=tx,
        boundary_reader=initializer._boundary_reader,
        clock=initializer._clock,
    )
    again = cold.resolve_original(guard.verify(), TRANSPORT, literal_prompt=PROMPT)
    assert (again.campaign_ref, again.campaign_id, again.private_store_correlation) == (
        first.campaign_ref,
        first.campaign_id,
        first.private_store_correlation,
    )
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1


def test_lost_commit_response_resolves_whole_original_pair():
    _, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    tx.lose_write = True
    result = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    c, h = split_items(tx)
    assert result.campaign_id == binding_of(h)["campaign_id"] == binding_of(c)["campaign_id"]
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1


def test_unknown_commit_read_never_retries_write_and_cold_resolution_recovers():
    ci, _, initializer, guard, _, _, tx, _, boundary = fixture_initializer()
    tx.lose_write = True
    tx.read_after_write = lambda: setattr(tx, "fail_reads", True)
    with pytest.raises(ci.InitializationUnresolved):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1
    tx.fail_reads, tx.read_after_write = False, None
    cold = ci.DynamoCampaignInitialization(
        guard,
        boundary,
        private_transaction_client=tx,
        boundary_reader=initializer._boundary_reader,
        clock=initializer._clock,
    )
    result = cold.resolve_original(guard.verify(), TRANSPORT, literal_prompt=PROMPT)
    assert result.campaign_id == binding_of(split_items(tx)[1])["campaign_id"]
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1


def test_concurrent_identical_intakes_recover_winner_original_ids():
    ci, _, _, guard, _, _, tx, _, boundary = fixture_initializer()
    constructors = [
        ci.DynamoCampaignInitialization(
            guard,
            boundary,
            private_transaction_client=tx,
            boundary_reader=ProtectionReader(ci, boundary.configuration, guard._clock),
            clock=guard._clock,
        )
        for _ in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda one: one.initialize(guard.verify(), TRANSPORT, PROMPT), constructors))
    assert results[0].campaign_ref == results[1].campaign_ref
    assert results[0].campaign_id == results[1].campaign_id
    assert len(tx.items) == 2


def test_changed_prompt_cannot_rebind_committed_original():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    before = copy.deepcopy(tx.items)
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT + " changed")
    assert tx.items == before


def test_deleted_campaign_with_retained_handoff_is_never_recreated():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    c, _ = split_items(tx)
    tx.items.pop(c["job_name"]["S"])
    before = len(tx.calls)
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert all(name != "TransactWriteItems" for name, _ in tx.calls[before:])
    assert len(tx.items) == 1


def test_delayed_creator_cannot_recreate_campaign_after_winner_commit_and_deletion():
    ci, _, delayed, guard, _, _, tx, _, boundary = fixture_initializer()
    winner = ci.DynamoCampaignInitialization(
        guard,
        boundary,
        private_transaction_client=tx,
        boundary_reader=delayed._boundary_reader,
        clock=delayed._clock,
    )

    def commit_and_remove_campaign():
        winner.initialize(guard.verify(), TRANSPORT, PROMPT)
        c, _ = split_items(tx)
        tx.items.pop(c["job_name"]["S"])

    tx.before_write = commit_and_remove_campaign
    with pytest.raises(ci.InitializationError):
        delayed.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert len(tx.items) == 1
    assert next(iter(tx.items)).startswith("CAMPAIGN_HANDOFF#")


def test_campaign_without_independent_handoff_is_not_adopted():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    _, h = split_items(tx)
    tx.items.pop(h["job_name"]["S"])
    with pytest.raises(ci.InitializationError):
        initializer.resolve_original(guard.verify(), TRANSPORT)


def test_coherent_campaign_binding_replacement_cannot_match_protected_handoff():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    c, _ = split_items(tx)
    body = json.loads(c["private_record"]["B"])
    body["binding"]["campaign_id"] = "12345678-1234-4234-9234-00000000ee02"
    canonical = json.dumps(body["binding"], sort_keys=True, separators=(",", ":")).encode()
    body["binding_sha256"] = hashlib.sha256(canonical).hexdigest()
    c["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ci.InitializationError):
        initializer.resolve_original(guard.verify(), TRANSPORT)


def test_missing_unapproved_or_client_claim_boundary_refuses_before_io():
    ci = initialization_module()
    _, guard, _, clock, _ = fixture_guard()
    for boundary in [None, config(ci), {"protected": True}, {"client_role": "initializer"}]:
        tx = Transactions()
        with pytest.raises(ci.InitializationError):
            ci.DynamoCampaignInitialization(
                guard,
                boundary,
                private_transaction_client=tx,
                boundary_reader=ProtectionReader(ci, config(ci), clock),
                clock=clock,
            )
        assert tx.calls == []


def test_public_metadata_cannot_mint_boundary_wrapper():
    ci = initialization_module()
    with pytest.raises(ci.InitializationError):
        ci.ApprovedInitializationBoundary(config(ci), "fixture-protection-review")


def test_current_admission_revocation_refuses_before_transactional_lookup():
    ci, ca, initializer, guard, reader, _, tx, _, _ = fixture_initializer()
    witness = guard.verify()
    reader.observation = dataclasses_replace(reader.observation, endpoint_live_version="2")
    with pytest.raises(ca.AdmissionError):
        initializer.initialize(witness, TRANSPORT, PROMPT)
    assert tx.calls == []
    assert ci is not None


def dataclasses_replace(value, **changes):
    import dataclasses

    return dataclasses.replace(value, **changes)


@pytest.mark.parametrize("state", ["missing", "drift", "expired", "failed"])
def test_current_protection_evidence_refuses_before_transaction_io(state):
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    reader = initializer._boundary_reader
    if state == "missing":
        reader.read_current = lambda: None
    elif state == "drift":
        reader.configuration = dataclasses_replace(reader.configuration, publisher_role_id="AROA" + "Z" * 17)
    elif state == "expired":
        reader.expired = True
    else:
        reader.fail = True
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []


def test_protection_drift_after_transaction_read_refuses_before_write():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    read = tx.transact_get_items

    def read_then_drift(**kwargs):
        reply = read(**kwargs)
        initializer._boundary_reader.fail = True
        return reply

    tx.transact_get_items = read_then_drift
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert all(name != "TransactWriteItems" for name, _ in tx.calls)


def test_protection_drift_after_commit_refuses_resolved_result():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    tx.read_after_write = lambda: setattr(initializer._boundary_reader, "fail", True)
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert len(tx.items) == 2


def test_advanced_campaign_resolves_original_without_initial_body_equality():
    _, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    first = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    c, _ = split_items(tx)
    body = json.loads(c["private_record"]["B"])
    body["revision"] = 1
    c["coordinator_revision"]["N"] = "1"
    c["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    current = initializer.resolve_original(guard.verify(), TRANSPORT)
    assert current.campaign_ref == first.campaign_ref
    assert current.current_revision == 1


def test_unavailable_partial_transaction_response_never_enters_resolved_state():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    tx.transact_get_items = lambda **_: {"Responses": [{}]}
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.items == {}


def test_current_protection_evidence_is_bound_to_process_and_reader_port():
    ci = initialization_module()
    assert {"process_id", "reader_binding"}.issubset(ci.InitializationProtectionEvidence.__dataclass_fields__)


def test_explicit_null_item_is_malformed_not_transactional_absence():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    tx.transact_get_items = lambda **_: {
        "Responses": [{"Item": None}, {}],
        "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
    }
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.items == {}


def test_changed_handoff_initial_body_digest_refuses_without_reconstruction():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    _, h = split_items(tx)
    body = json.loads(h["private_record"]["B"])
    body["initial_campaign_body_sha256"] = hashlib.sha256(b"neutral different initial body").hexdigest()
    h["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    before = copy.deepcopy(tx.items)
    with pytest.raises(ci.InitializationError):
        initializer.resolve_original(guard.verify(), TRANSPORT)
    assert tx.items == before


def test_unknown_handoff_field_does_not_turn_into_protection_authority():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    _, h = split_items(tx)
    body = json.loads(h["private_record"]["B"])
    body["protected"] = True
    h["private_record"]["B"] = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ci.InitializationError):
        initializer.resolve_original(guard.verify(), TRANSPORT)


def test_actual_closed_item_size_includes_names_and_native_limit():
    ci = initialization_module()
    overhead = len("job_name") + len("private_record") + len("fixture")
    good = {"job_name": {"S": "fixture"}, "private_record": {"B": b"a" * (400 * 1024 - overhead)}}
    assert ci._item_size(good) <= 400 * 1024
    bad = {"job_name": {"S": "fixture"}, "private_record": {"B": b"a" * (400 * 1024)}}
    with pytest.raises(ci.InitializationError):
        ci._item_size(bad)


@pytest.mark.parametrize("field", ["process_id", "reader_binding"])
def test_foreign_current_evidence_refuses_before_any_transaction(field):
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    reader = initializer._boundary_reader
    original = reader.read_current

    def foreign():
        changes = {field: os.getpid() + 1 if field == "process_id" else object()}
        return dataclasses_replace(original(), **changes)

    reader.read_current = foreign
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []


def test_coherent_approved_config_and_digest_substitution_refuses_before_io():
    ci, _, initializer, guard, _, _, tx, _, boundary = fixture_initializer()
    from amlas.control_plane.campaign_coordinator import _digest

    changed = dataclasses_replace(boundary.configuration, publisher_role_id="AROA" + "Z" * 17)
    object.__setattr__(boundary, "configuration", changed)
    object.__setattr__(boundary, "configuration_sha256", _digest(ci._config_fields(changed)))
    initializer._boundary_reader.configuration = changed
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []


@pytest.mark.parametrize("value", [True, float("inf"), float("nan")])
def test_initializer_clock_rejects_boolean_nonfinite_before_reader_or_io(value):
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    initializer._clock = lambda: value
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == [] and initializer._boundary_reader.calls == 0


def test_initializer_clock_rollback_inside_evidence_interval_refuses():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    values = iter([1020.0, 1010.0])
    initializer._clock = lambda: next(values)
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []


def test_inherited_initializer_refuses_before_lock_reader_and_io(monkeypatch):
    ci, _, initializer, _guard, _, _, tx, _, _ = fixture_initializer()

    class NoLock:
        def __enter__(self):
            pytest.fail("inherited initializer reached lock")

    initializer._lock = NoLock()
    monkeypatch.setattr(ci.os, "getpid", lambda: initializer._process_id + 1)
    with pytest.raises(ci.InitializationError):
        initializer.initialize(None, TRANSPORT, PROMPT)
    assert tx.calls == [] and initializer._boundary_reader.calls == 0


def test_unresolved_uncommitted_outcome_observes_without_write_replay():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()

    def unknown(**kwargs):
        validate_parameters(kwargs, MODEL.operation_model("TransactWriteItems").input_shape)
        tx.calls.append(("TransactWriteItems", copy.deepcopy(kwargs)))
        raise TimeoutError("neutral mutation response and commit unavailable")

    tx.transact_write_items = unknown
    with pytest.raises(ci.InitializationUnresolved):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    with pytest.raises(ci.InitializationUnresolved):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1
    assert tx.items == {}


def test_admission_revoked_during_final_protection_read_prevents_transaction():
    ci, ca, initializer, guard, native, _, tx, _, _ = fixture_initializer()
    protection = initializer._boundary_reader
    original = protection.read_current

    def revoke_before_lookup():
        result = original()
        # The second protection read is immediately before first TxGet.
        if protection.calls == 2:
            native.observation = dataclasses_replace(native.observation, endpoint_live_version="2")
        return result

    protection.read_current = revoke_before_lookup
    with pytest.raises(ca.AdmissionError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []
    assert ci is not None


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at": True},
        {"expires_at": True},
        {"process_id": True},
        {"observed_at": float("inf")},
        {"expires_at": float("nan")},
        {"observed_at": 999.0},
        {"observed_at": 1001.0},
        {"expires_at": 1000.0},
        {"expires_at": 1061.0},
    ],
)
def test_current_evidence_finite_types_and_exact_interval_refuse(changes):
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    reader = initializer._boundary_reader
    read = reader.read_current
    reader.read_current = lambda: dataclasses_replace(read(), **changes)
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []


def test_expired_evidence_cannot_be_silently_refreshed_from_retained_object():
    ci, _, initializer, guard, _, clock, tx, _, _ = fixture_initializer()
    reader = initializer._boundary_reader
    retained = reader.read_current()
    reader.read_current = lambda: retained
    clock.value = retained.expires_at
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []


@pytest.mark.parametrize("value", [[], "neutral", b"neutral", True, 7])
def test_nonobject_item_refuses_before_mutation(value):
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    tx.transact_get_items = lambda **_: {
        "Responses": [{"Item": value}, {}],
        "ResponseMetadata": {"HTTPStatusCode": 200, "RetryAttempts": 0},
    }
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.items == {}


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"HTTPStatusCode": True, "RetryAttempts": 0},
        {"HTTPStatusCode": 200, "RetryAttempts": False},
        {"HTTPStatusCode": 200, "RetryAttempts": 1},
        {"HTTPStatusCode": 200.0, "RetryAttempts": 0},
    ],
)
def test_unknown_or_retried_transaction_read_metadata_refuses(metadata):
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    tx.transact_get_items = lambda **_: {"Responses": [{}, {}], "ResponseMetadata": metadata}
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.items == {}


def test_lost_ack_uses_one_post_outcome_transaction_read_and_original_token():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    tx.lose_write = True
    original = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert [name for name, _ in tx.calls] == ["TransactGetItems", "TransactWriteItems", "TransactGetItems"]
    assert tx.calls[1][1]["ClientRequestToken"] == original.campaign_id
    assert sum(ci._item_size(value) for value in tx.items.values()) <= 4 * 1024 * 1024


def test_unknown_evidence_object_is_refused_without_property_execution():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()

    class UnknownEvidence:
        @property
        def configuration(self):
            pytest.fail("untyped evidence property was executed")

    initializer._boundary_reader.read_current = lambda: UnknownEvidence()
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert tx.calls == []


@pytest.mark.parametrize("target", [2, 3, 5])
def test_protection_expiry_during_final_admission_read_refuses(target):
    ci, _, initializer, guard, native, clock, tx, _, _ = fixture_initializer()
    protection = initializer._boundary_reader
    read_protection, read_native = protection.read_current, native.read_current
    delay = [False]

    def short_evidence():
        evidence = read_protection()
        if protection.calls == target:
            delay[0] = True
            return dataclasses_replace(evidence, expires_at=evidence.observed_at + 1)
        return evidence

    def delayed_admission():
        if delay[0]:
            clock.value += 2
            delay[0] = False
        return read_native()

    protection.read_current, native.read_current = short_evidence, delayed_admission
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    if target == 2:
        assert tx.calls == []
    elif target == 3:
        assert [name for name, _ in tx.calls] == ["TransactGetItems"]
    else:
        assert len(tx.items) == 2
        assert [name for name, _ in tx.calls] == [
            "TransactGetItems",
            "TransactWriteItems",
            "TransactGetItems",
        ]


def test_changed_prompt_after_unattempted_refusal_never_writes_old_candidate():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    reader = initializer._boundary_reader
    original = reader.read_current

    def refuse_before_write():
        result = original()
        if reader.calls == 3:
            raise TimeoutError("neutral prewrite protection unavailable")
        return result

    reader.read_current = refuse_before_write
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert initializer._pending and not initializer._attempted
    assert tx.items == {}
    reader.read_current = original
    with pytest.raises(ci.InitializationError):
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT + " changed")
    assert tx.items == {}
    assert all(name != "TransactWriteItems" for name, _ in tx.calls)


def test_reported_write_retry_refuses_after_one_bounded_confirmed_outcome_read():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    write = tx.transact_write_items

    def contradictory(**kwargs):
        result = write(**kwargs)
        result["ResponseMetadata"]["RetryAttempts"] = 1
        return result

    tx.transact_write_items = contradictory
    with pytest.raises(ci.InitializationTransportError) as refused:
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert type(refused.value.confirmed_original) is ci.InitializationObservation
    assert refused.value.confirmed_original.native_admission is False
    assert refused.value.confirmed_original.action_approved is False
    assert len(tx.items) == 2
    assert [name for name, _ in tx.calls] == ["TransactGetItems", "TransactWriteItems", "TransactGetItems"]
    # Fresh read-only resolution preserves the committed original without replay.
    original = initializer.resolve_original(guard.verify(), TRANSPORT, literal_prompt=PROMPT)
    assert original.campaign_id == binding_of(split_items(tx)[1])["campaign_id"]
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1


def test_reported_write_retry_and_unavailable_outcome_retains_unknown_no_replay():
    ci, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
    write = tx.transact_write_items

    def contradictory(**kwargs):
        result = write(**kwargs)
        result["ResponseMetadata"]["RetryAttempts"] = 1
        tx.fail_reads = True
        return result

    tx.transact_write_items = contradictory
    with pytest.raises(ci.InitializationTransportError) as refused:
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert refused.value.confirmed_original is None
    assert isinstance(refused.value.__cause__, ci.InitializationUnresolved)
    assert len(tx.items) == 2
    assert [name for name, _ in tx.calls] == ["TransactGetItems", "TransactWriteItems", "TransactGetItems"]
    tx.fail_reads = False
    original = initializer.resolve_original(guard.verify(), TRANSPORT, literal_prompt=PROMPT)
    assert original.campaign_id == binding_of(split_items(tx)[1])["campaign_id"]
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1


def test_two_empty_intakes_force_conditional_conflict_and_recover_winner():
    ci, _, _, guard, _, _, tx, _, boundary = fixture_initializer()
    barrier = threading.Barrier(2)
    seen_empty = []
    read = tx.transact_get_items

    def both_observe_empty(**kwargs):
        result = read(**kwargs)
        if result["Responses"] == [{}, {}]:
            seen_empty.append(True)
            barrier.wait(timeout=5)
        return result

    tx.transact_get_items = both_observe_empty
    constructors = [
        ci.DynamoCampaignInitialization(
            guard,
            boundary,
            private_transaction_client=tx,
            boundary_reader=ProtectionReader(ci, boundary.configuration, guard._clock),
            clock=guard._clock,
        )
        for _ in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda item: item.initialize(guard.verify(), TRANSPORT, PROMPT), constructors)
        )
    assert len(seen_empty) == 2
    writes = [kwargs for name, kwargs in tx.calls if name == "TransactWriteItems"]
    assert len(writes) == 2
    assert len({request["ClientRequestToken"] for request in writes}) == 2
    assert len(tx.items) == 2
    assert results[0].campaign_ref == results[1].campaign_ref
    assert results[0].campaign_id == results[1].campaign_id
    assert results[0].campaign_id == binding_of(split_items(tx)[1])["campaign_id"]


def test_separate_process_resolves_persisted_original_without_write(tmp_path):
    # Harmless source-library processes, fresh fixture approvals and fake storage;
    # this is not AgentCore/native credential/protection or redelivery proof.
    code = r"""
import base64, json, os, sys
from pathlib import Path
from test_campaign_initialization import fixture_initializer, TRANSPORT, PROMPT
stage, filename = sys.argv[1:]
path = Path(filename)
_, _, initializer, guard, _, _, tx, _, _ = fixture_initializer()
if stage == "read":
    tx.items = json.loads(path.read_text())
    for item in tx.items.values():
        item["private_record"]["B"] = base64.b64decode(item["private_record"]["B"], validate=True)
    result = initializer.resolve_original(guard.verify(), TRANSPORT, literal_prompt=PROMPT)
else:
    result = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    persisted = {}
    for key, item in tx.items.items():
        projected = {name: dict(value) for name, value in item.items()}
        projected["private_record"]["B"] = base64.b64encode(item["private_record"]["B"]).decode("ascii")
        persisted[key] = projected
    with path.open("x") as output:
        json.dump(persisted, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
print(json.dumps({"pid":os.getpid(),"campaign_ref":result.campaign_ref,
    "campaign_id":result.campaign_id,"private_store_correlation":result.private_store_correlation,
    "literal_prompt":result.literal_prompt,"calls":[name for name,_ in tx.calls],
    "native_admission":result.native_admission,"action_approved":result.action_approved}))
"""
    state = tmp_path / "neutral-original.json"
    test_directory = Path(__file__).resolve().parent
    child_env = os.environ.copy()
    paths = [str(test_directory.parent / "src"), str(test_directory)]
    existing_pythonpath = child_env.get("PYTHONPATH")
    if existing_pythonpath is not None:
        paths.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(paths)
    results = []
    for stage in ("publish", "read"):
        run = subprocess.run(
            [sys.executable, "-B", "-c", code, stage, str(state)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=child_env,
        )
        assert run.returncode == 0, f"neutral initialization child {stage} failed: {run.stderr}"
        results.append(json.loads(run.stdout))
    (tmp_path / "process-results.json").write_text(json.dumps(results, sort_keys=True))
    original, recovered = results
    assert len({original["pid"], recovered["pid"], os.getpid()}) == 3
    for field in ("campaign_ref", "campaign_id", "private_store_correlation", "literal_prompt"):
        assert recovered[field] == original[field]
    assert original["calls"] == ["TransactGetItems", "TransactWriteItems", "TransactGetItems"]
    assert recovered["calls"] == ["TransactGetItems"]
    assert recovered["native_admission"] is False and recovered["action_approved"] is False


def test_request_validation_delay_precedes_final_protection_qualification(monkeypatch):
    import botocore.validate

    _ci, _, initializer, guard, _, clock, tx, _, _ = fixture_initializer()
    protection = initializer._boundary_reader
    read = protection.read_current
    deadline = [None]

    def short_current():
        evidence = read()
        deadline[0] = evidence.observed_at + 1
        return dataclasses_replace(evidence, expires_at=deadline[0])

    protection.read_current = short_current
    validate = botocore.validate.validate_parameters

    def delayed_validation(*args, **kwargs):
        result = validate(*args, **kwargs)
        clock.value += 2
        return result

    monkeypatch.setattr(botocore.validate, "validate_parameters", delayed_validation)
    for method in ("transact_get_items", "transact_write_items"):
        native = getattr(tx, method)

        def current_at_dispatch(_native=native, **kwargs):
            assert clock.value < deadline[0]
            return _native(**kwargs)

        setattr(tx, method, current_at_dispatch)
    original = initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert original.native_admission is False
    assert len(tx.items) == 2


def test_retry_violation_preserves_historical_original_when_final_protection_expires():
    ci, _, initializer, guard, native, clock, tx, _, _ = fixture_initializer()
    protection = initializer._boundary_reader
    read_protection, read_native, write = (
        protection.read_current,
        native.read_current,
        tx.transact_write_items,
    )
    delay = [False]

    def contradictory(**kwargs):
        reply = write(**kwargs)
        reply["ResponseMetadata"]["RetryAttempts"] = 1
        return reply

    def short_final_evidence():
        evidence = read_protection()
        if protection.calls == 5:
            delay[0] = True
            return dataclasses_replace(evidence, expires_at=evidence.observed_at + 1)
        return evidence

    def final_admission_consumes_deadline():
        if delay[0]:
            clock.value += 2
            delay[0] = False
        return read_native()

    tx.transact_write_items = contradictory
    protection.read_current = short_final_evidence
    native.read_current = final_admission_consumes_deadline
    with pytest.raises(ci.InitializationTransportError) as refused:
        initializer.initialize(guard.verify(), TRANSPORT, PROMPT)
    assert type(refused.value.__cause__) is ci.InitializationError
    assert "expired during final admission" in str(refused.value.__cause__)
    assert refused.value.currently_qualified is False
    assert refused.value.confirmed_original_is_historical is True
    original = refused.value.confirmed_original
    assert type(original) is ci.InitializationObservation
    assert original.native_admission is False and original.action_approved is False
    assert original.campaign_id == binding_of(split_items(tx)[1])["campaign_id"]
    assert original.literal_prompt == PROMPT
    assert [name for name, _ in tx.calls] == ["TransactGetItems", "TransactWriteItems", "TransactGetItems"]
    assert len(tx.items) == 2
    # New current qualification observes the intact original without write replay.
    recovered = initializer.resolve_original(guard.verify(), TRANSPORT, literal_prompt=PROMPT)
    assert recovered.campaign_ref == original.campaign_ref
    assert sum(name == "TransactWriteItems" for name, _ in tx.calls) == 1
