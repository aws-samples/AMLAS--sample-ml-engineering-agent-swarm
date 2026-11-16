"""Exact raw DATA projection retains current native and context boundaries.

Declared inert GetItem DATA and typed read policies exercise only the actual row
loader. No caller permission, native service result, SDK continuation or cold
recovery is established by these unit tests. Production changes follow AWS RED.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib

import pytest
from test_normal_checkpoint_cas_contract import TABLE_ID, _item, _policy
from test_normal_completion_record_codec import DeclaredGetPort, _declared_sft_record

from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.control_plane.producer_snapshot import _approved_read


@pytest.fixture(autouse=True)
def clear_pure_data_caches():
    checkpoint._validate_record_bytes.cache_clear()
    project = getattr(checkpoint, "_record_projection", None)
    if project is not None:
        project.cache_clear()
    yield
    checkpoint._validate_record_bytes.cache_clear()
    project = getattr(checkpoint, "_record_projection", None)
    if project is not None:
        project.cache_clear()


def fixture():
    context, record = _declared_sft_record()
    port = DeclaredGetPort(context, _item(context, record))
    checks = []
    policy = [_policy()]
    current = {"role": True, "table": True, "configuration": True}

    def qualify():
        _approved_read(policy[0])
        checks.append("fresh")
        if not all(current.values()):
            raise ValueError("declared current qualification refused")

    return context, record, port, checks, policy, current, qualify


def load(context, port, qualify, *, projection=True, table_id=TABLE_ID):
    return checkpoint._load(context, port, table_id, qualify, _projection=projection)


def assert_immutable(value):
    if dataclasses.is_dataclass(value):
        assert value.__dataclass_params__.frozen
        for field in dataclasses.fields(value):
            assert_immutable(getattr(value, field.name))
    elif type(value) is tuple or type(value) is frozenset:
        for child in value:
            assert_immutable(child)
    else:
        assert value is None or type(value) in (str, int, float, bool)


def test_fresh_projection_is_immutable_and_keeps_native_read_cadence():
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    item, first = load(context, port, qualify)
    _item_again, second = load(context, port, qualify)
    assert type(first) is checkpoint._RecordProjection
    assert first.raw == checkpoint._canonical(record)
    assert first.sft == checkpoint._canonical(record["sft"])
    assert first.private_store_correlation == record["private_store_correlation"]
    assert first is second
    assert item == _item(context, record)
    assert_immutable(first)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.raw = "caller mutation"
    assert port.calls == 2 and checks == ["fresh"] * 4


def test_warm_projection_does_not_repeat_full_detached_decode(monkeypatch):
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    calls = []
    actual = checkpoint._record_data

    def decoded(raw):
        calls.append(True)
        return actual(raw)

    monkeypatch.setattr(checkpoint, "_record_data", decoded)
    _item_value, first = load(context, port, qualify)
    initial_decodes = len(calls)
    for _ in range(3):
        assert load(context, port, qualify)[1] is first
    assert len(calls) == initial_decodes, "unchanged projected DATA repeated full row decoding"
    assert first.raw == checkpoint._canonical(record)
    assert port.calls == 4 and checks == ["fresh"] * 8


def test_default_loader_keeps_independent_mutable_data_after_projection():
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    first_view = load(context, port, qualify)[1]
    first = checkpoint._load(context, port, TABLE_ID, qualify)[1]
    first["continuation"]["sdk_snapshot"]["data"]["messages"][0]["content"][0]["text"] = "Changed DATA."
    first["sft"]["intent"]["operation"] = "changed"
    second = checkpoint._load(context, port, TABLE_ID, qualify)[1]
    assert second == record and second is not first
    assert load(context, port, qualify)[1] is first_view
    assert first_view.raw == checkpoint._canonical(record)
    assert port.item == _item(context, record)
    assert port.calls == 4 and checks == ["fresh"] * 8


def test_fresh_changed_row_replaces_projection_without_mutating_prior_view():
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    first = load(context, port, qualify)[1]
    changed = copy.deepcopy(record)
    changed["history"] = [{"role": "user", "content": [{"text": "Fresh changed declared DATA."}]}]
    changed["sft"]["intent"]["operation"] = "declared_other_operation"
    changed["private_store_correlation"] = "33333333-3333-3333-3333-a33333333333"
    port.item = _item(context, changed)
    second = load(context, port, qualify)[1]
    assert second is not first
    assert second.raw == checkpoint._canonical(changed)
    assert second.sft == checkpoint._canonical(changed["sft"])
    assert second.private_store_correlation == changed["private_store_correlation"]
    assert first.raw == checkpoint._canonical(record)
    assert port.calls == 2 and checks == ["fresh"] * 4


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "after_valid"])
def test_duplicate_key_bytes_refuse_even_with_current_matching_hash(warm):
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    if warm:
        load(context, port, qualify)
    raw = checkpoint._canonical(record)
    raw = raw.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    port.item["record"] = {"S": raw}
    port.item["record_sha256"] = {"S": hashlib.sha256(raw.encode()).hexdigest()}
    with pytest.raises(ValueError):
        load(context, port, qualify)
    assert port.calls == int(warm) + 1 and checks == ["fresh"] * (2 * (int(warm) + 1))


@pytest.mark.parametrize(
    "fault",
    [
        "extra_field",
        "missing_history",
        "bool_schema",
        "bad_stage",
        "invalid_correlation",
        "invalid_continuation",
    ],
)
def test_rehashed_malformed_base_or_continuation_bytes_refuse_after_warm_projection(fault):
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    load(context, port, qualify)
    changed = copy.deepcopy(record)
    if fault == "extra_field":
        changed["unrecognized"] = "Declared unsupported DATA."
    elif fault == "missing_history":
        changed.pop("history")
    elif fault == "bool_schema":
        changed["schema_version"] = True
    elif fault == "bad_stage":
        changed["stage"] = "unsupported"
    elif fault == "invalid_correlation":
        changed["private_store_correlation"] = "not-a-correlation"
    else:
        changed["continuation"]["epoch"] = True
    port.item = _item(context, changed)
    with pytest.raises(ValueError):
        load(context, port, qualify)
    assert port.calls == 2 and checks == ["fresh"] * 4


def test_fresh_envelope_and_hash_checks_precede_cache_reuse():
    context, _record, port, checks, _policy_value, _current, qualify = fixture()
    load(context, port, qualify)
    hits = checkpoint._record_projection.cache_info().hits
    port.item["record_sha256"] = {"S": "0" * 64}
    with pytest.raises(ValueError):
        load(context, port, qualify)
    assert checkpoint._record_projection.cache_info().hits == hits
    assert port.calls == 2 and checks == ["fresh"] * 4


@pytest.mark.parametrize("field", ["role", "table", "configuration", "policy"])
def test_current_qualification_denial_refuses_before_another_read(field):
    context, _record, port, _checks, policy, current, qualify = fixture()
    load(context, port, qualify)
    if field == "policy":
        policy[0] = _policy(False)
    else:
        current[field] = False
    with pytest.raises(ValueError):
        load(context, port, qualify)
    assert port.calls == 1


@pytest.mark.parametrize("field", ["session", "run", "catalog", "table_id"])
def test_current_context_drift_refuses_identical_warm_record_bytes(field):
    context, _record, port, checks, _policy_value, _current, qualify = fixture()
    load(context, port, qualify)
    changed = context
    table_id = TABLE_ID
    if field == "session":
        changed = dataclasses.replace(context, session_hash="9" * 64)
    elif field == "run":
        changed = dataclasses.replace(context, run_id="8" * 32)
        port.item["job_name"] = {"S": "PROMPT_RUN#" + changed.run_id}
    elif field == "catalog":
        changed = dataclasses.replace(
            context,
            catalog=dataclasses.replace(
                context.catalog, deployment_id="99999999-9999-9999-9999-a99999999999"
            ),
        )
    else:
        table_id = "99999999-9999-9999-9999-a99999999999"
    port.context = changed
    with pytest.raises(ValueError):
        load(changed, port, qualify, table_id=table_id)
    assert port.calls == 2 and checks == ["fresh"] * 4


def test_prepared_reference_keeps_current_bucket_selection_after_projection(monkeypatch):
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    bucket = "declared-projection-bucket"
    monkeypatch.setenv("AMLAS_SESSION_BUCKET", bucket)
    digest = "a" * 64
    record["prepared_checkpoint_ref"] = {
        "uri": "s3://"
        + bucket
        + "/sessions/"
        + record["private_store_correlation"]
        + "/prompt-runs/"
        + context.run_id
        + "/prepared-tasks/"
        + digest
        + ".json",
        "version_id": "declared-projection-version",
        "sha256": digest,
        "byte_count": 64,
        "format": "json",
    }
    port.item = _item(context, record)
    load(context, port, qualify)
    monkeypatch.setenv("AMLAS_SESSION_BUCKET", "declared-projection-changed-bucket")
    with pytest.raises(ValueError):
        load(context, port, qualify)
    assert port.calls == 2 and checks == ["fresh"] * 4


def test_projection_cache_is_exact_raw_four_entry_lru_without_native_reuse():
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    original = copy.deepcopy(port.item)
    for index in range(5):
        changed = copy.deepcopy(record)
        changed["history"] = [
            {"role": "user", "content": [{"text": "Declared projection variant " + str(index)}]}
        ]
        port.item = original if index == 0 else _item(context, changed)
        load(context, port, qualify)
    port.item = original
    assert load(context, port, qualify)[1].raw == checkpoint._canonical(record)
    info = checkpoint._record_projection.cache_info()
    assert info.maxsize == 4 and info.currsize == 4 and info.misses == 6
    assert port.calls == 6 and checks == ["fresh"] * 12


def test_projection_and_default_native_read_and_qualification_counts_match():
    context, record, first_port, first_checks, _policy_value, _current, first_qualify = fixture()
    _context, _record, second_port, second_checks, _policy2, _current2, second_qualify = fixture()
    for _ in range(3):
        assert checkpoint._load(context, first_port, TABLE_ID, first_qualify)[1] == record
        assert load(context, second_port, second_qualify)[1].raw == checkpoint._canonical(record)
    assert first_port.calls == second_port.calls == 3
    assert first_checks == second_checks == ["fresh"] * 6


def test_current_revocation_after_fresh_get_refuses_before_cached_projection():
    context, _record, port, checks, _policy_value, current, qualify = fixture()
    load(context, port, qualify)
    actual_get = port.get_item
    hits = checkpoint._record_projection.cache_info().hits

    def revoked_get(**kwargs):
        answer = actual_get(**kwargs)
        current["role"] = False
        return answer

    port.get_item = revoked_get
    with pytest.raises(ValueError):
        load(context, port, qualify)
    assert port.calls == 2 and checks == ["fresh"] * 4
    assert checkpoint._record_projection.cache_info().hits == hits


def test_projection_raw_equality_preserves_canonical_boolean_integer_difference():
    context, record, port, checks, _policy_value, _current, qualify = fixture()
    record["history"] = [{"declared": 1}]
    port.item = _item(context, record)
    first = load(context, port, qualify)[1]
    changed = copy.deepcopy(record)
    changed["history"] = [{"declared": True}]
    assert changed == record
    port.item = _item(context, changed)
    second = load(context, port, qualify)[1]
    assert first.raw != second.raw and first is not second
    assert first.sft == second.sft
    assert first.private_store_correlation == second.private_store_correlation
    assert port.calls == 2 and checks == ["fresh"] * 4
