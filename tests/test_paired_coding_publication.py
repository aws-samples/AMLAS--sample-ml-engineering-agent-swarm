"""Pure complete output replay and neutral packet guards; no code/subprocess execution."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_coding_arbiter import binding
from test_paired_coding_runner import Native, Oracle, population

from amlas.arbiter.coding import CodingCasePin, CodingEvaluatorFactory
from amlas.evaluation.coding_oracle_catalog import (
    EXPECTED_CASES,
    ORIGINAL_SOURCE_BYTES,
    ORIGINAL_SOURCE_SHA256,
    verify_pinned_case,
)
from amlas.evaluation.coding_pair import PairedTraceAccumulator, verify_pair_result
from amlas.evaluation.coding_pair_runner import run_paired
from amlas.sagemaker_entrypoints.coding_pair import publish_result
from amlas.training.coding_generation import canonical_bytes


def test_catalog_contains_only_fixed164_metadata_and_refuses_arbitrary_test_text():
    assert len(EXPECTED_CASES) == 164 and ORIGINAL_SOURCE_BYTES == 214438
    assert ORIGINAL_SOURCE_SHA256 == (
        "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"  # pragma: allowlist secret
    )
    assert all(
        set(row)
        == {
            "case_id",
            "entry_point",
            "prompt_sha256",
            "test_sha256",
            "prompt_record_sha256",
            "oracle_record_sha256",
        }
        for row in EXPECTED_CASES
    )
    with pytest.raises(ValueError):
        verify_pinned_case(
            {"case_id": "HumanEval/0", "prompt": "neutral", "entry_point": "fixture"},
            {"case_id": "HumanEval/0", "test": "neutral", "canonical_solution": "neutral"},
        )


def complete_output(tmp_path):
    trace = tmp_path / "INVOCATIONS"
    trace.mkdir()
    native = Native(trace)
    cases, candidates = population()
    # Pure fixture oracle identities use the independently compiled fixed catalog.
    cases = tuple(
        {**case, "oracle_sha256": EXPECTED_CASES[index]["oracle_record_sha256"]}
        for index, case in enumerate(cases)
    )
    result = run_paired(
        binding=binding(),
        cases=cases,
        candidates=candidates,
        client=native,
        cleanup_client=native,
        evidence_root=trace,
        oracle_runner=Oracle(),
        clock=lambda: 1.0,
        sleep=lambda _: None,
    )
    published = publish_result(
        tmp_path, binding=binding(), result=result, cases=cases, source_commit="a" * 40
    )
    bundle = {
        name: (tmp_path / name).read_bytes()
        for name in ("RECEIPTS.jsonl", "EVALUATION.json", "COMMITTED.json")
    }
    trace_raw = (tmp_path / "INVOCATIONS_MANIFEST.jsonl").read_bytes()
    traces = (hashlib.sha256(trace_raw).hexdigest(), len(trace_raw), len(trace_raw.splitlines()))
    accumulator = PairedTraceAccumulator(binding())
    for entry in map(json.loads, trace_raw.splitlines()):
        accumulator.add(entry["path"], json.loads((tmp_path / entry["path"]).read_bytes()))
    return published, bundle, traces, accumulator.finish()


def test_worker_publication_has328_records_and_replays_same_private_report(tmp_path):
    published, bundle, traces, trace = complete_output(tmp_path)
    assert published["evaluated"] is True and published["receipt_count"] == 328
    assert published["quality_success"] is published["native_admission"] is False
    verified = verify_pair_result(
        bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=trace
    )
    assert verified == published
    assert len(bundle["RECEIPTS.jsonl"].splitlines()) == 328


@pytest.mark.parametrize(
    "field,value",
    [
        ("receipt_count", 327),
        ("native_admission", True),
        ("cleanup_complete", 1),
        ("source_commit", "b" * 40),
    ],
)
def test_rehashed_pair_summary_cannot_change_fixed_output_bindings(tmp_path, field, value):
    _, bundle, traces, trace = complete_output(tmp_path)
    evaluation = json.loads(bundle["EVALUATION.json"])
    evaluation[field] = value
    bundle["EVALUATION.json"] = canonical_bytes(evaluation) + b"\n"
    commit = json.loads(bundle["COMMITTED.json"])
    commit["files"]["EVALUATION.json"] = {
        "sha256": hashlib.sha256(bundle["EVALUATION.json"]).hexdigest(),
        "byte_count": len(bundle["EVALUATION.json"]),
        "format": "json",
    }
    bundle["COMMITTED.json"] = canonical_bytes(commit) + b"\n"
    with pytest.raises(ValueError):
        verify_pair_result(bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=trace)


def test_receipt_population_is_complete_before_private_aggregation():
    bound = binding()
    pins = tuple(
        CodingCasePin(case_id=row["case_id"], oracle_sha256=row["oracle_record_sha256"])
        for row in EXPECTED_CASES
    )
    from amlas.arbiter.schema import InvalidSubmissionError

    with pytest.raises(InvalidSubmissionError):
        CodingEvaluatorFactory().build_report(binding=bound, case_pins=pins, receipts=())


def trace_records(directory):
    raw = (directory / "INVOCATIONS_MANIFEST.jsonl").read_bytes()
    return {
        entry["path"]: json.loads((directory / entry["path"]).read_bytes())
        for entry in map(json.loads, raw.splitlines())
    }


def rebound_output(bundle, records):
    manifest = (
        b"".join(
            canonical_bytes(
                {
                    "path": path,
                    "sha256": hashlib.sha256(canonical_bytes(record)).hexdigest(),
                    "byte_count": len(canonical_bytes(record)),
                }
            )
            + b"\n"
            for path, record in sorted(records.items())
        )
        or b"\n"
    )
    trace = {
        "sha256": hashlib.sha256(manifest).hexdigest(),
        "byte_count": len(manifest),
        "format": "jsonl",
        "record_count": len(records),
    }
    evaluation = json.loads(bundle["EVALUATION.json"])
    evaluation["invocations_manifest"] = trace
    result = {**bundle, "EVALUATION.json": canonical_bytes(evaluation) + b"\n"}
    commit = json.loads(result["COMMITTED.json"])
    commit["invocations_manifest"] = trace
    commit["files"]["EVALUATION.json"] = {
        "sha256": hashlib.sha256(result["EVALUATION.json"]).hexdigest(),
        "byte_count": len(result["EVALUATION.json"]),
        "format": "json",
    }
    result["COMMITTED.json"] = canonical_bytes(commit) + b"\n"
    return result, (trace["sha256"], trace["byte_count"], trace["record_count"])


@pytest.mark.parametrize(
    "drift",
    [
        "session_replay",
        "token_replay",
        "ordinal_gap",
        "channel_gap",
        "bootstrap_only",
        "stop_without_get",
        "start_unknown",
        "request_replay",
        "missing_response",
    ],
)
def test_rehashed_trace_requires_original_complete_session_and_value_chain(tmp_path, drift):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = trace_records(tmp_path)
    base = {path: record for path, record in records.items() if path.startswith("INVOCATIONS/base/")}
    tuned = {path: record for path, record in records.items() if path.startswith("INVOCATIONS/tuned/")}
    if drift == "session_replay":
        original = next(record["session_id"] for record in base.values() if "session_id" in record)
        for record in tuned.values():
            if "session_id" in record:
                record["session_id"] = original
    elif drift == "token_replay":
        original = next(
            record["client_token"] for path, record in base.items() if path.endswith("start-attempt.json")
        )
        for path, record in tuned.items():
            if path.endswith("start-attempt.json"):
                record["client_token"] = original
    elif drift == "ordinal_gap":
        for record in base.values():
            if record.get("ordinal") == 1:
                record["ordinal"] = record["binding"]["ordinal"] = 2
    elif drift in ("channel_gap", "missing_response"):
        path = next(path for path in base if "/calls/" in path and path.endswith("response.json"))
        del records[path]
    elif drift == "bootstrap_only":
        records = {
            path: record
            for path, record in records.items()
            if not (path.startswith("INVOCATIONS/base/") and record.get("ordinal", 0) > 0)
        }
    elif drift == "stop_without_get":
        records = {
            path: record
            for path, record in records.items()
            if not (
                path.startswith("INVOCATIONS/base/")
                and (path.endswith("get-attempt.json") or path.endswith("get-observed.json"))
            )
        }
    elif drift == "start_unknown":
        start_path = next(path for path in base if path.endswith("start-attempt.json"))
        record = records[start_path]
        records = {
            path: value
            for path, value in records.items()
            if not path.startswith("INVOCATIONS/base/") or path == start_path
        }
        records["INVOCATIONS/base/000/session/00000001-start-unknown.json"] = {
            "evaluation_id": record["evaluation_id"],
            "case_id": record["case_id"],
            "arm": "base",
            "client_token": record["client_token"],
            "error_type": "TimeoutError",
        }
    else:
        original = {
            record["ordinal"]: record["request_sha256"]
            for record in base.values()
            if record.get("kind") == "attempt"
        }
        for record in tuned.values():
            if "request_sha256" in record and "ordinal" in record:
                record["request_sha256"] = original[record["ordinal"]]
    bundle, traces = rebound_output(bundle, records)
    with pytest.raises(ValueError):
        accumulator = PairedTraceAccumulator(binding())
        for path, record in records.items():
            accumulator.add(path, record)
        verify_pair_result(
            bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=accumulator.finish()
        )


def replace_original_base_receipt(bundle, outcome, *, cleanup_complete):
    from amlas.arbiter.coding import CodingCaseReceipt

    values = list(map(json.loads, bundle["RECEIPTS.jsonl"].splitlines()))
    values[0].update(outcome=outcome, oracle_completed=False, exit_code=None)
    receipts = tuple(CodingCaseReceipt.model_validate(value) for value in values)
    pins = tuple(
        CodingCasePin(case_id=row["case_id"], oracle_sha256=row["oracle_record_sha256"])
        for row in EXPECTED_CASES
    )
    evaluation = json.loads(bundle["EVALUATION.json"])
    evaluation["summary"] = (
        CodingEvaluatorFactory().build_report(binding=binding(), case_pins=pins, receipts=receipts).describe()
    )
    evaluation["cleanup_complete"] = cleanup_complete
    result = {
        **bundle,
        "RECEIPTS.jsonl": b"".join(canonical_bytes(value) + b"\n" for value in values),
        "EVALUATION.json": canonical_bytes(evaluation) + b"\n",
    }
    commit = json.loads(result["COMMITTED.json"])
    for name in ("RECEIPTS.jsonl", "EVALUATION.json"):
        commit["files"][name] = {
            "sha256": hashlib.sha256(result[name]).hexdigest(),
            "byte_count": len(result[name]),
            "format": "jsonl" if name.endswith(".jsonl") else "json",
        }
    result["COMMITTED.json"] = canonical_bytes(commit) + b"\n"
    return result


def test_uncertain_allocation_remains_unresolved_and_keeps_denominator(tmp_path):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = trace_records(tmp_path)
    start = next(
        record
        for path, record in records.items()
        if path.startswith("INVOCATIONS/base/") and path.endswith("start-attempt.json")
    )
    records = {
        path: record
        for path, record in records.items()
        if not path.startswith("INVOCATIONS/base/") or path.endswith("start-attempt.json")
    }
    records["INVOCATIONS/base/000/session/00000001-start-unknown.json"] = {
        "evaluation_id": start["evaluation_id"],
        "case_id": start["case_id"],
        "arm": "base",
        "client_token": start["client_token"],
        "error_type": "TimeoutError",
    }
    bundle = replace_original_base_receipt(bundle, "infrastructure_error", cleanup_complete=False)
    bundle, traces = rebound_output(bundle, records)
    accumulator = PairedTraceAccumulator(binding())
    for path, record in records.items():
        accumulator.add(path, record)
    result = verify_pair_result(
        bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=accumulator.finish()
    )
    assert result["status"] == "cleanup_unresolved" and result["evaluated"] is result["success"] is False
    assert (
        result["receipt_count"] == 328
        and result["summary"]["base_failure_counts"]["infrastructure_error"] == 1
    )
    assert result["quality_success"] is result["native_admission"] is False


def test_unknown_value_call_retains_timeout_without_erasing_terminal_cleanup(tmp_path):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = trace_records(tmp_path)
    path = next(
        path
        for path, record in records.items()
        if path.startswith("INVOCATIONS/base/")
        and record.get("kind") == "completed"
        and record["ordinal"] == 1
    )
    record = records.pop(path)
    records[path.replace("-completed.json", "-unknown.json")] = {
        name: value for name, value in record.items() if name not in {"event_byte_count", "kind"}
    }
    records[path.replace("-completed.json", "-unknown.json")].update(
        kind="unknown", error_type="TimeoutError"
    )
    bundle = replace_original_base_receipt(bundle, "timeout", cleanup_complete=True)
    bundle, traces = rebound_output(bundle, records)
    accumulator = PairedTraceAccumulator(binding())
    for path, record in records.items():
        accumulator.add(path, record)
    result = verify_pair_result(
        bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=accumulator.finish()
    )
    assert result["evaluated"] is result["success"] is True and result["receipt_count"] == 328
    assert result["summary"]["base_failure_counts"]["timeout"] == 1
    assert result["quality_success"] is result["native_admission"] is False


def original_prefix_records(records, *, session_count=None, call_count=None):
    result = dict(records)
    for channel, count in (("session", session_count), ("calls", call_count)):
        if count is None:
            continue
        paths = sorted(path for path in result if path.startswith("INVOCATIONS/base/000/" + channel + "/"))
        for path in paths[count:]:
            del result[path]
    return result


def verify_original_prefix(bundle, records):
    bundle, traces = rebound_output(bundle, records)
    accumulator = PairedTraceAccumulator(binding())
    for path, record in records.items():
        accumulator.add(path, record)
    return verify_pair_result(
        bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=accumulator.finish()
    )


@pytest.mark.parametrize("session_count", [1, 2, 3, 4, 5])
def test_durable_cleanup_prefix_without_post_deadline_error_record_stays_unresolved(tmp_path, session_count):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = original_prefix_records(
        trace_records(tmp_path), session_count=session_count, call_count=0 if session_count == 1 else None
    )
    if session_count == 1:
        # Native allocation never returned an identity; no call record can exist.
        bundle = replace_original_base_receipt(bundle, "infrastructure_error", cleanup_complete=False)
    else:
        # Oracle finished, but Stop/Get interruption prevents operational completion.
        evaluation = json.loads(bundle["EVALUATION.json"])
        evaluation["cleanup_complete"] = False
        bundle = {**bundle, "EVALUATION.json": canonical_bytes(evaluation) + b"\n"}
    result = verify_original_prefix(bundle, records)
    assert result["success"] is result["evaluated"] is result["cleanup_complete"] is False
    assert result["status"] == "cleanup_unresolved" and result["receipt_count"] == 328
    assert result["summary"]["execution_count"] == 328
    assert result["quality_success"] is result["native_admission"] is False


@pytest.mark.parametrize("call_count", [5, 6, 7])
def test_durable_call_prefix_without_final_unknown_write_keeps_timeout_denominator(tmp_path, call_count):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = original_prefix_records(trace_records(tmp_path), session_count=5, call_count=call_count)
    bundle = replace_original_base_receipt(bundle, "timeout", cleanup_complete=False)
    result = verify_original_prefix(bundle, records)
    assert result["success"] is result["evaluated"] is False and result["status"] == "cleanup_unresolved"
    assert result["receipt_count"] == 328 and result["summary"]["base_failure_counts"]["timeout"] == 1
    assert result["quality_success"] is result["native_admission"] is False


@pytest.mark.parametrize("call_count", [5, 6, 7])
def test_incomplete_call_prefix_never_preserves_a_completed_oracle_claim(tmp_path, call_count):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = original_prefix_records(trace_records(tmp_path), session_count=5, call_count=call_count)
    evaluation = json.loads(bundle["EVALUATION.json"])
    evaluation["cleanup_complete"] = False
    bundle = {**bundle, "EVALUATION.json": canonical_bytes(evaluation) + b"\n"}
    with pytest.raises(ValueError, match="completed oracle receipt"):
        verify_original_prefix(bundle, records)


def final_cleanup_unknown(records, *, session_id=None):
    accepted = next(
        record
        for path, record in records.items()
        if path.startswith("INVOCATIONS/base/000/session/") and path.endswith("start-accepted.json")
    )
    result = dict(records)
    result["INVOCATIONS/base/000/session/00000006-cleanup-unknown.json"] = {
        "evaluation_id": accepted["evaluation_id"],
        "case_id": accepted["case_id"],
        "arm": accepted["arm"],
        "session_id": session_id or accepted["session_id"],
        "error_type": "OSError",
    }
    return result


def original_cleanup_unresolved_bundle(bundle):
    evaluation = json.loads(bundle["EVALUATION.json"])
    evaluation["cleanup_complete"] = False
    return {**bundle, "EVALUATION.json": canonical_bytes(evaluation) + b"\n"}


def test_final_same_session_cleanup_unknown_after_terminal_is_unresolved(tmp_path):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = final_cleanup_unknown(trace_records(tmp_path))
    result = verify_original_prefix(original_cleanup_unresolved_bundle(bundle), records)
    assert result["success"] is result["evaluated"] is result["cleanup_complete"] is False
    assert result["status"] == "cleanup_unresolved" and result["receipt_count"] == 328
    assert result["quality_success"] is result["native_admission"] is False


@pytest.mark.parametrize("followon", ["foreign_session", "second_unknown", "get_observed"])
def test_terminal_session_only_allows_one_final_same_session_unknown(tmp_path, followon):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = final_cleanup_unknown(
        trace_records(tmp_path),
        session_id="other-original-session" if followon == "foreign_session" else None,
    )
    last = records["INVOCATIONS/base/000/session/00000006-cleanup-unknown.json"]
    if followon == "second_unknown":
        records["INVOCATIONS/base/000/session/00000007-cleanup-unknown.json"] = dict(last)
    elif followon == "get_observed":
        records["INVOCATIONS/base/000/session/00000007-get-observed.json"] = {
            key: value for key, value in last.items() if key != "error_type"
        }
        records["INVOCATIONS/base/000/session/00000007-get-observed.json"]["status"] = "TERMINATED"
    with pytest.raises(ValueError):
        verify_original_prefix(original_cleanup_unresolved_bundle(bundle), records)


def test_terminal_record_bytes_do_not_upgrade_unconfirmed_producer_cleanup(tmp_path):
    _published, bundle, _traces, _proof = complete_output(tmp_path)
    records = trace_records(tmp_path)
    result = verify_original_prefix(original_cleanup_unresolved_bundle(bundle), records)
    assert result["cleanup_complete"] is result["success"] is result["evaluated"] is False
    assert result["status"] == "cleanup_unresolved" and result["receipt_count"] == 328
    assert result["quality_success"] is result["native_admission"] is False


def test_original_read_failure_reconciliation_trace_requires_every_attempt_and_same_session():
    from amlas.evaluation.coding_pair import _session_chain

    token = "fixture-token"
    session = "original-session"
    records = [
        ("start-attempt", {"client_token": token}),
        ("start-accepted", {"session_id": session}),
        ("stop-attempt", {"session_id": session}),
        ("stop-accepted", {"session_id": session}),
        ("get-attempt", {"session_id": session}),
        (
            "get-failed",
            {
                "session_id": session,
                "error_type": "ClientError",
                "service_error": {"code": "ThrottlingException", "http_status": 429},
            },
        ),
        ("get-attempt", {"session_id": session}),
        ("get-observed", {"session_id": session, "status": "TERMINATED"}),
    ]
    assert _session_chain(records) == (session, True, True)
    with pytest.raises(ValueError):
        _session_chain(records[:6] + records[7:])
    with pytest.raises(ValueError):
        _session_chain([*records[:5], ("get-failed", {"session_id": "foreign"}), *records[6:]])


def _rehash(bundle, evaluation):
    """Rewrite EVALUATION.json and the commit's descriptor of it, so that only the changed field can refuse."""
    bundle["EVALUATION.json"] = canonical_bytes(evaluation) + b"\n"
    commit = json.loads(bundle["COMMITTED.json"])
    commit["files"]["EVALUATION.json"] = {
        "sha256": hashlib.sha256(bundle["EVALUATION.json"]).hexdigest(),
        "byte_count": len(bundle["EVALUATION.json"]),
        "format": "json",
    }
    bundle["COMMITTED.json"] = canonical_bytes(commit) + b"\n"
    return bundle


def test_an_unchanged_evaluation_still_verifies_after_the_rehash_helper(tmp_path):
    published, bundle, traces, trace = complete_output(tmp_path)
    again = _rehash(dict(bundle), json.loads(bundle["EVALUATION.json"]))
    assert again == bundle
    verified = verify_pair_result(
        again, binding=binding(), source_commit="a" * 40, invocations=traces, trace=trace
    )
    assert verified == published


@pytest.mark.parametrize("key", ["base_pass_count_range", "tuned_pass_count_range", "delta_pass_count_range"])
@pytest.mark.parametrize("end", [0, 1])
def test_rehashed_range_fields_are_refused(tmp_path, key, end):
    _, bundle, traces, trace = complete_output(tmp_path)
    evaluation = json.loads(bundle["EVALUATION.json"])
    value = list(evaluation["summary"][key])
    value[end] += 1
    evaluation["summary"][key] = value
    _rehash(bundle, evaluation)
    with pytest.raises(ValueError, match="original complete paired evaluation summary differs"):
        verify_pair_result(bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=trace)


def test_a_dropped_range_field_is_refused(tmp_path):
    _, bundle, traces, trace = complete_output(tmp_path)
    evaluation = json.loads(bundle["EVALUATION.json"])
    assert evaluation["summary"]["delta_pass_count_range"] == [0, 0]
    del evaluation["summary"]["delta_pass_count_range"]
    _rehash(bundle, evaluation)
    with pytest.raises(ValueError, match="original complete paired evaluation summary differs"):
        verify_pair_result(bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=trace)


_FIRST = "Base passed {base} of 164 cases. Tuned passed {tuned} of 164. "
_NONE = "No case ended without a verdict (timeout or infrastructure error). "
_OPEN = (
    "Cases that ended without a verdict (timeout or infrastructure error): base {base_open}, tuned {tuned_open}. "
    "They count as not passed in these totals. "
)
_DIFFERENCE = "The difference in passed cases, tuned minus base, is {difference}. "
_RANGE = (
    "If each case without a verdict is counted either as passed or as not passed, and every other outcome "
    "stays as it is, that difference lies between {low} and {high}. "
)
_LAST = (
    "Each case that ran had 10 seconds for its test and all of its remote calls, counted from the moment "
    "its remote session was accepted. Neither arm ran a case a second time."
)
_BEHIND = "Tuned stays behind base in that whole range. "
_AHEAD = "Tuned stays ahead of base in that whole range. "
_OPEN_SIGN = (
    "In that range tuned can be behind base, level with it or ahead of it, "
    "so this evaluation does not settle which model is ahead. "
)
_LEVEL_OR_AHEAD = "In that range tuned is level with base or ahead of it; it cannot be behind. "
_BEHIND_OR_LEVEL = "In that range tuned is behind base or level with it; it cannot be ahead. "
_NEXT_STEP = "Put result_statement into your final report unchanged."


def _range_summary(base, tuned, base_open, tuned_open):
    return {
        "case_count": 164,
        "base_pass_count": base,
        "tuned_pass_count": tuned,
        "base_pass_count_range": [base, base + base_open],
        "tuned_pass_count_range": [tuned, tuned + tuned_open],
        "delta_pass_count_range": [tuned - (base + base_open), (tuned + tuned_open) - base],
    }


def test_verified_result_carries_the_statement_for_a_run_without_no_verdict_cases(tmp_path):
    published, bundle, traces, trace = complete_output(tmp_path)
    # Fixture population: the one generated answer of each arm passes and 163 cases per arm have no answer.
    assert published["status"] == "evaluated" and published["evaluated"] is True
    assert published["result_statement"] == (
        "Base passed 1 of 164 cases. Tuned passed 1 of 164. "
        "No case ended without a verdict (timeout or infrastructure error). "
        "The difference in passed cases, tuned minus base, is 0. "
        "Each case that ran had 10 seconds for its test and all of its remote calls, "
        "counted from the moment its remote session was accepted. "
        "Neither arm ran a case a second time."
    )
    assert published["next_step"] == "Put result_statement into your final report unchanged."
    verified = verify_pair_result(
        bundle, binding=binding(), source_commit="a" * 40, invocations=traces, trace=trace
    )
    assert verified == published


def test_statement_of_a_run_with_two_tuned_cases_without_a_verdict_reads_word_for_word():
    from amlas.evaluation.coding_pair import pair_result_statement

    assert pair_result_statement(_range_summary(138, 106, 0, 2)) == (
        "Base passed 138 of 164 cases. Tuned passed 106 of 164. "
        "Cases that ended without a verdict (timeout or infrastructure error): base 0, tuned 2. "
        "They count as not passed in these totals. "
        "The difference in passed cases, tuned minus base, is -32. "
        "If each case without a verdict is counted either as passed or as not passed, and every other outcome "
        "stays as it is, that difference lies between -32 and -30. "
        "Tuned stays behind base in that whole range. "
        "Each case that ran had 10 seconds for its test and all of its remote calls, "
        "counted from the moment its remote session was accepted. "
        "Neither arm ran a case a second time."
    )


@pytest.mark.parametrize(
    "base,tuned,base_open,tuned_open,difference,low,high,relation",
    [
        (138, 106, 1, 2, "-32", "-33", "-30", _BEHIND),
        (137, 107, 2, 10, "-30", "-32", "-20", _BEHIND),
        (100, 110, 2, 1, "+10", "+8", "+11", _AHEAD),
        (100, 101, 2, 0, "+1", "-1", "+1", _OPEN_SIGN),
        (101, 100, 0, 2, "-1", "-1", "+1", _OPEN_SIGN),
        # A tie in the counts with one open case in each arm.
        (100, 100, 1, 1, "0", "-1", "+1", _OPEN_SIGN),
        # Ranges that touch zero at one end.
        (100, 100, 0, 1, "0", "0", "+1", _LEVEL_OR_AHEAD),
        (100, 101, 1, 0, "+1", "0", "+1", _LEVEL_OR_AHEAD),
        (100, 100, 1, 0, "0", "-1", "0", _BEHIND_OR_LEVEL),
        (101, 100, 0, 1, "-1", "-1", "0", _BEHIND_OR_LEVEL),
    ],
)
def test_statement_forms_with_no_verdict_cases(
    base, tuned, base_open, tuned_open, difference, low, high, relation
):
    from amlas.evaluation.coding_pair import pair_result_statement

    assert pair_result_statement(_range_summary(base, tuned, base_open, tuned_open)) == (
        _FIRST.format(base=base, tuned=tuned)
        + _OPEN.format(base_open=base_open, tuned_open=tuned_open)
        + _DIFFERENCE.format(difference=difference)
        + _RANGE.format(low=low, high=high)
        + relation
        + _LAST
    )


@pytest.mark.parametrize("base,tuned,difference", [(100, 101, "+1"), (100, 100, "0"), (138, 106, "-32")])
def test_statement_without_no_verdict_cases_states_the_difference_alone(base, tuned, difference):
    from amlas.evaluation.coding_pair import pair_result_statement

    assert pair_result_statement(_range_summary(base, tuned, 0, 0)) == (
        _FIRST.format(base=base, tuned=tuned) + _NONE + _DIFFERENCE.format(difference=difference) + _LAST
    )


def test_statement_takes_the_case_limit_from_the_limits_table(monkeypatch):
    from amlas.evaluation import coding_pair_runner
    from amlas.evaluation.coding_pair import pair_result_statement

    monkeypatch.setattr(coding_pair_runner, "BUDGETS", {**coding_pair_runner.BUDGETS, "case_wall_seconds": 7})
    assert "Each case that ran had 7 seconds for its test" in pair_result_statement(
        _range_summary(100, 101, 0, 0)
    )


@pytest.mark.parametrize("shape", ["uncertain_allocation", "producer_cleanup_not_confirmed"])
def test_an_unresolved_cleanup_carries_no_statement_and_no_next_step(tmp_path, shape):
    published, bundle, _traces, _proof = complete_output(tmp_path)
    assert published["status"] == "evaluated"
    assert published["next_step"] == _NEXT_STEP and published["result_statement"].startswith(
        "Base passed 1 of 164"
    )
    records = trace_records(tmp_path)
    if shape == "uncertain_allocation":
        start = next(
            record
            for path, record in records.items()
            if path.startswith("INVOCATIONS/base/") and path.endswith("start-attempt.json")
        )
        records = {
            path: record
            for path, record in records.items()
            if not path.startswith("INVOCATIONS/base/") or path.endswith("start-attempt.json")
        }
        records["INVOCATIONS/base/000/session/00000001-start-unknown.json"] = {
            "evaluation_id": start["evaluation_id"],
            "case_id": start["case_id"],
            "arm": "base",
            "client_token": start["client_token"],
            "error_type": "TimeoutError",
        }
        bundle = replace_original_base_receipt(bundle, "infrastructure_error", cleanup_complete=False)
        expected_base_range = [0, 1]
    else:
        bundle = original_cleanup_unresolved_bundle(bundle)
        expected_base_range = [1, 1]
    result = verify_original_prefix(bundle, records)
    assert result["status"] == "cleanup_unresolved" and result["evaluated"] is result["success"] is False
    assert result["summary"]["base_pass_count_range"] == expected_base_range
    assert "result_statement" not in result and "next_step" not in result
    assert set(published) - set(result) == {"result_statement", "next_step"}
