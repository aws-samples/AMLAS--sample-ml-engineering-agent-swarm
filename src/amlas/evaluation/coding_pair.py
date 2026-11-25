"""Fixed private artifact replay and paired report arithmetic; no execution grants."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass

from ..arbiter.coding import (
    CodingCasePin,
    CodingCaseReceipt,
    CodingEvaluationBinding,
    CodingEvaluatorFactory,
    coding_spec_bytes,
)
from ..control_plane.paired_state import PairedCodingInputs
from ..data_eng.coding_case_managed import validate_managed_case_bundle
from ..data_eng.coding_case_profile import CASE_IDS, SOURCE_SHA256
from ..data_eng.inspection_schema import decode_object
from ..spec.schema import SpecModel
from ..training.coding_generation import CONTRACT_SHA256, canonical_bytes
from .coding_oracle_catalog import EXPECTED_CASES, verify_pinned_case


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _records(raw, *, count, limit):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError("bounded fixed paired records required")
    lines = raw.splitlines(keepends=True)
    if len(lines) != count or any(not line.endswith(b"\n") or len(line) > 512 * 1024 for line in lines):
        raise ValueError("fixed paired record population differs")
    return tuple(decode_object(line, limit=512 * 1024) for line in lines)


def validate_pair_artifacts(inputs, payloads, *, runner_sha256, runtime_sha256, budgets_sha256):
    """Replay only original case/gen versions and the independently pinned official oracle."""
    if type(inputs) is not PairedCodingInputs or type(payloads) is not dict:
        raise ValueError("typed original paired artifact inputs required")
    groups = {
        "cases": dict(inputs.cases.file_refs),
        "base": dict(inputs.base.file_refs),
        "tuned": dict(inputs.tuned.file_refs),
    }
    expected = {group + "/" + name for group, refs in groups.items() for name in refs}
    if set(payloads) != expected:
        raise ValueError("closed original paired artifact inventory differs")
    for group, refs in groups.items():
        for name, ref in refs.items():
            raw = payloads[group + "/" + name]
            maximum = (
                12 * 1024**2 if name == "CANDIDATES.jsonl" else 2 * 1024**2 if group == "cases" else 262144
            )
            if (
                type(raw) is not bytes
                or not 0 < len(raw) <= maximum
                or len(raw) != ref.byte_count
                or _sha(raw) != ref.sha256
            ):
                raise ValueError("original private paired bytes/hash differs")
    spec = SpecModel.model_validate_json(inputs.spec_bytes)
    if (
        coding_spec_bytes(spec) != inputs.spec_bytes
        or _sha(inputs.spec_bytes) != inputs.spec_sha256
        or spec.task_type != "text_generation"
        or spec.primary_metric != "pass_at_1"
        or spec.generation is None
    ):
        raise ValueError("canonical original resolved spec differs")
    manifest_ref = dict(inputs.cases.file_refs)["CASE_MANIFEST.json"]
    if (
        spec.generation.evaluation.case_manifest_ref is None
        or spec.generation.evaluation.case_manifest_ref.model_dump() != dataclasses.asdict(manifest_ref)
    ):
        raise ValueError("resolved original private case manifest differs")
    bundle = {name: payloads["cases/" + name] for name in groups["cases"]}
    validate_managed_case_bundle(
        bundle, source=inputs.cases.original_source, header=dict(inputs.cases.producer_header)
    )
    prompts = _records(bundle["PROMPTS.jsonl"], count=164, limit=2 * 1024**2)
    oracles = _records(bundle["ORACLES.jsonl"], count=164, limit=2 * 1024**2)
    cases = []
    for index, (prompt, oracle) in enumerate(zip(prompts, oracles, strict=True)):
        expected_case = verify_pinned_case(prompt, oracle)
        if prompt["case_id"] != CASE_IDS[index]:
            raise ValueError("fixed original case order differs")
        cases.append(
            {"prompt": prompt, "oracle": oracle, "oracle_sha256": expected_case["oracle_record_sha256"]}
        )
    candidates, model_hashes, generation_hashes = {}, {}, {}
    for arm, source in (("base", inputs.base), ("tuned", inputs.tuned)):
        if (
            source.arm != arm
            or source.runtime_session_id != inputs.cases.runtime_session_id
            or source.evaluation_id != inputs.base.evaluation_id
        ):
            raise ValueError("original generation arm/session/evaluation differs")
        generation = decode_object(payloads[arm + "/GENERATION.json"], limit=262144)
        commit = decode_object(payloads[arm + "/COMMITTED.json"], limit=262144)
        if (
            set(commit)
            != {
                "schema_version",
                "operation",
                "source_commit",
                "case_count",
                "quality_success",
                "native_admission",
                "files",
            }
            or commit.get("schema_version") != 1
            or type(commit.get("schema_version")) is not int
            or commit.get("operation") != "generate"
            or type(commit.get("source_commit")) is not str
            or not re.fullmatch("[a-f0-9]{40}", commit["source_commit"])
            or commit.get("case_count") != 164
            or type(commit.get("case_count")) is not int
            or commit.get("quality_success") is not False
            or commit.get("native_admission") is not False
            or type(commit.get("files")) is not dict
            or set(commit["files"]) != {"CANDIDATES.jsonl", "GENERATION.json"}
        ):
            raise ValueError("closed original generation commit differs")
        for name in ("CANDIDATES.jsonl", "GENERATION.json"):
            ref = groups[arm][name]
            if commit["files"][name] != {
                "sha256": ref.sha256,
                "byte_count": ref.byte_count,
                "format": ref.format,
            }:
                raise ValueError("original committed generation bytes differ")
        bound = generation.get("binding")
        if (
            type(bound) is not dict
            or set(bound)
            != {"evaluation_id", "arm", "prompts_sha256", "model_sha256", "tokenizer_sha256", "spec_sha256"}
            or bound["evaluation_id"] != source.evaluation_id
            or bound["arm"] != arm
            or bound["spec_sha256"] != inputs.spec_sha256
            or bound["prompts_sha256"] != groups["cases"]["PROMPTS.jsonl"].sha256
            or generation.get("contract_sha256") != CONTRACT_SHA256
            or generation.get("source_commit") != commit["source_commit"]
            or generation.get("case_count") != 164
            or generation.get("quality_success") is not False
            or generation.get("native_admission") is not False
            or any(
                type(bound[field]) is not str or not re.fullmatch("[a-f0-9]{64}", bound[field])
                for field in ("model_sha256", "tokenizer_sha256", "spec_sha256", "prompts_sha256")
            )
        ):
            raise ValueError("original paired generation binding differs")
        rows = _records(payloads[arm + "/CANDIDATES.jsonl"], count=164, limit=12 * 1024**2)
        if type(generation.get("cases")) is not list or len(generation["cases"]) != 164:
            raise ValueError("original complete generation outcomes required")
        from ..backends.adapters.coding_generation import SageMakerCodingGenerationBackend

        for index, row in enumerate(rows):
            if row.get("case_id") != CASE_IDS[index] or canonical_bytes(
                {key: value for key, value in row.items() if key != "source"}
            ) != canonical_bytes(generation["cases"][index]):
                raise ValueError("original candidate/manifest case differs")
            SageMakerCodingGenerationBackend._validate_candidate(row, prompts[index])
        candidates[arm], model_hashes[arm], generation_hashes[arm] = (
            rows,
            bound["model_sha256"],
            groups[arm]["GENERATION.json"].sha256,
        )
    binding = CodingEvaluationBinding(
        evaluation_id=inputs.base.evaluation_id,
        spec_sha256=inputs.spec_sha256,
        case_manifest=dataclasses.asdict(manifest_ref),
        source_sha256=SOURCE_SHA256,
        runner_sha256=runner_sha256,
        runtime_sha256=runtime_sha256,
        budgets_sha256=budgets_sha256,
        generation_sha256=_sha(canonical_bytes(generation_hashes)),
        base_model_sha256=model_hashes["base"],
        tuned_model_sha256=model_hashes["tuned"],
    )
    if binding.base_model_sha256 == binding.tuned_model_sha256:
        raise ValueError("distinct original base/tuned model bindings required")
    return binding, tuple(cases), candidates


_TRACE_KEY = object()
_TRACE_PATH = re.compile(r"INVOCATIONS/(base|tuned)/([0-9]{3})/(session|calls)/([0-9]{8})-([a-z-]+)\.json")


@dataclass(frozen=True, slots=True)
class _PairedTraceProof:
    binding_sha256: str
    rows: tuple
    record_count: int
    key: object


class PairedTraceAccumulator:
    """Bounded hash-only record consistency; never SDK or isolation attestation."""

    def __init__(self, binding):
        if type(binding) is not CodingEvaluationBinding:
            raise ValueError("original typed paired trace binding required")
        self._binding = binding
        self._rows = {}

    def add(self, path, record):
        from .coding_executor import _record_bytes
        from .coding_pair_runner import validate_session_record

        match = _TRACE_PATH.fullmatch(path) if type(path) is str else None
        if (
            not match
            or int(match[2]) > 163
            or int(match[4]) >= 1000000
            or path in self._rows
            or len(self._rows) >= 1000000
        ):
            raise ValueError("closed unique paired trace path required")
        arm, case, channel, number, kind = match.groups()
        case_id = "HumanEval/" + str(int(case))
        if channel == "calls":
            _raw, actual_kind = _record_bytes(record)
            if (
                actual_kind != kind
                or record["evaluation_id"] != self._binding.evaluation_id
                or record["case_id"] != case_id
                or record["arm"] != arm
            ):
                raise ValueError("original paired call trace scope differs")
        else:
            validate_session_record(
                record, kind=kind, case_id=case_id, arm=arm, evaluation_id=self._binding.evaluation_id
            )
        # Retain only closed primitive consistency fields, never raw code/event payloads.
        fields = (
            "session_id",
            "client_token",
            "request_sha256",
            "code_sha256",
            "ordinal",
            "event_count",
            "event_byte_count",
            "index",
            "event_kind",
            "byte_count",
            "status",
            "time",
            "deadline",
        )
        compact = tuple((name, record[name]) for name in fields if name in record)
        self._rows[path] = (arm, case_id, channel, int(number), kind, compact)

    def finish(self):
        return _PairedTraceProof(
            _sha(canonical_bytes(self._binding.model_dump(mode="json"))),
            tuple(self._rows.values()),
            len(self._rows),
            _TRACE_KEY,
        )


def _session_chain(rows):
    if not rows:
        return None, False, False
    if rows[0][0] != "start-attempt":
        raise ValueError("original allocation attempt missing")
    token = rows[0][1]["client_token"]
    if len(rows) == 1:
        return None, False, True
    if rows[1][0] == "start-unknown":
        if rows[1][1]["client_token"] != token or len(rows) != 2:
            raise ValueError("original uncertain allocation chain differs")
        return None, False, True
    if rows[1][0] != "start-accepted":
        raise ValueError("original allocation acceptance missing")
    session = rows[1][1]["session_id"]
    pending, accepted, terminal, expect = False, False, False, "stop-attempt"
    for position, (kind, record) in enumerate(rows[2:], 2):
        if record.get("session_id") != session:
            raise ValueError("original cleanup session/terminal chain differs")
        if kind == "cleanup-unknown":
            if position != len(rows) - 1:
                raise ValueError("uncertain cleanup must retain final original state")
            return session, False, True
        if terminal:
            raise ValueError("original cleanup session/terminal chain differs")
        if expect == "get-observed" and kind == "get-failed":
            expect = "get-attempt"
            continue
        if kind != expect:
            raise ValueError("original cleanup order differs")
        if kind == "stop-attempt":
            pending, expect = True, "stop-accepted"
        elif kind == "stop-accepted":
            accepted, expect = True, "get-attempt"
        elif kind == "get-attempt":
            expect = "get-observed"
        else:
            terminal = record["status"] in ("TERMINATED", "STOPPED")
            expect = "get-attempt"
    return session, bool(pending and accepted and terminal), True


def _call_chain(rows, session):
    if not rows:
        return False
    if session is None:
        raise ValueError("call trace without original accepted session")
    ordinal, state, request, events, event_bytes, complete = -1, "idle", None, 0, 0, False
    deadline, last_time = None, None
    for kind, record in rows:
        if record.get("session_id") != session:
            raise ValueError("cross-session paired invocation replay")
        if deadline is None:
            deadline = record["deadline"]
        if record["deadline"] != deadline or (last_time is not None and record["time"] < last_time):
            raise ValueError("original whole-case deadline/time chain differs")
        last_time = record["time"]
        if kind == "attempt":
            if (
                state not in ("idle", "completed")
                or record["ordinal"] != ordinal + 1
                or record["time"] >= deadline
            ):
                raise ValueError("original invocation attempt/ordinal order differs")
            ordinal, state, request = record["ordinal"], "attempt", record["request_sha256"]
            events, event_bytes, complete = 0, 0, False
            continue
        if record["ordinal"] != ordinal or state in ("idle", "unknown"):
            raise ValueError("original invocation ordinal/state differs")
        if kind in ("response", "completed", "unknown") and record["request_sha256"] != request:
            raise ValueError("original invocation request chain differs")
        if kind == "response":
            if state != "attempt":
                raise ValueError("original invocation response order differs")
            state = "response"
        elif kind == "event":
            if (
                state not in ("response", "event")
                or record["index"] != events
                or record["event_kind"] != "result"
            ):
                raise ValueError("original event framing/order differs")
            events += 1
            event_bytes += record["byte_count"]
            state = "event"
        elif kind == "completed":
            if (
                state != "event"
                or events != 1
                or record["event_count"] != events
                or record["event_byte_count"] != event_bytes
                or record["time"] >= deadline
            ):
                raise ValueError("original complete invocation event proof differs")
            state, complete = "completed", True
        elif kind == "unknown":
            if state not in ("attempt", "response", "event", "completed") or record["event_count"] != events:
                raise ValueError("original uncertain invocation framing differs")
            state, complete = "unknown", False
        else:
            raise ValueError("closed paired invocation stage required")
    return complete and ordinal >= 1


def _verify_trace(proof, binding, receipts, cleanup_complete, count):
    if (
        type(proof) is not _PairedTraceProof
        or proof.key is not _TRACE_KEY
        or proof.binding_sha256 != _sha(canonical_bytes(binding.model_dump(mode="json")))
        or type(count) is not int
        or proof.record_count != count
    ):
        raise ValueError("original private paired trace consistency proof required")
    grouped = {}
    for arm, case_id, channel, index, kind, compact in proof.rows:
        entries = grouped.setdefault((arm, case_id), {}).setdefault(channel, {})
        if index in entries:
            raise ValueError("duplicate original trace channel index")
        entries[index] = (kind, dict(compact))
    all_clean = True
    verified = {}
    original_tokens, original_sessions, original_requests = set(), set(), set()
    for key, channels in grouped.items():
        ordered = {}
        for channel, entries in channels.items():
            if sorted(entries) != list(range(len(entries))):
                raise ValueError("original trace channel index gap")
            ordered[channel] = [entries[index] for index in range(len(entries))]
        session_rows = ordered.get("session", [])
        session, cleaned, attempted = _session_chain(session_rows)
        if session_rows:
            token = session_rows[0][1]["client_token"]
            if token in original_tokens:
                raise ValueError("cross-case original allocation token replay")
            original_tokens.add(token)
        if session is not None:
            if session in original_sessions:
                raise ValueError("cross-case original session replay")
            original_sessions.add(session)
        for kind, record in ordered.get("calls", []):
            if kind == "attempt":
                digest = record["request_sha256"]
                if digest in original_requests:
                    raise ValueError("original invocation request replay")
                original_requests.add(digest)
        calls_complete = _call_chain(ordered.get("calls", []), session)
        all_clean = all_clean and (not attempted or cleaned)
        verified[key] = calls_complete
    if cleanup_complete and not all_clean:
        raise ValueError("unresolved original session cannot claim cleanup completion")
    for receipt in receipts:
        if receipt.oracle_completed and not verified.get((receipt.arm, receipt.case_id), False):
            raise ValueError("completed oracle receipt lacks original complete invocation chain")


PAIR_RESULT_NEXT_STEP = "Put result_statement into your final report unchanged."


def _signed(value):
    return "0" if value == 0 else f"{value:+d}"


def pair_result_statement(summary):
    """Fixed sentences for an evaluated pair; every number comes from the verified summary and the limits table."""
    from .coding_pair_runner import BUDGETS

    cases = summary["case_count"]
    base, tuned = summary["base_pass_count"], summary["tuned_pass_count"]
    base_open = summary["base_pass_count_range"][1] - summary["base_pass_count_range"][0]
    tuned_open = summary["tuned_pass_count_range"][1] - summary["tuned_pass_count_range"][0]
    low, high = summary["delta_pass_count_range"]
    difference = f"The difference in passed cases, tuned minus base, is {_signed(tuned - base)}."
    sentences = [f"Base passed {base} of {cases} cases.", f"Tuned passed {tuned} of {cases}."]
    if base_open + tuned_open == 0:
        sentences.append("No case ended without a verdict (timeout or infrastructure error).")
        sentences.append(difference)
    else:
        sentences.append(
            "Cases that ended without a verdict (timeout or infrastructure error): "
            f"base {base_open}, tuned {tuned_open}."
        )
        sentences.append("They count as not passed in these totals.")
        sentences.append(difference)
        sentences.append(
            "If each case without a verdict is counted either as passed or as not passed, and every other outcome "
            f"stays as it is, that difference lies between {_signed(low)} and {_signed(high)}."
        )
        if high < 0:
            sentences.append("Tuned stays behind base in that whole range.")
        elif low > 0:
            sentences.append("Tuned stays ahead of base in that whole range.")
        elif low < 0 < high:
            sentences.append(
                "In that range tuned can be behind base, level with it or ahead of it, "
                "so this evaluation does not settle which model is ahead."
            )
        elif low == 0:
            sentences.append("In that range tuned is level with base or ahead of it; it cannot be behind.")
        else:
            sentences.append("In that range tuned is behind base or level with it; it cannot be ahead.")
    sentences.append(
        f"Each case that ran had {BUDGETS['case_wall_seconds']} seconds for its test and all of its remote calls, "
        "counted from the moment its remote session was accepted."
    )
    sentences.append("Neither arm ran a case a second time.")
    return " ".join(sentences)


def verify_pair_result(bundle, *, binding, source_commit, invocations, trace):
    if (
        type(binding) is not CodingEvaluationBinding
        or type(bundle) is not dict
        or set(bundle) != {"RECEIPTS.jsonl", "EVALUATION.json", "COMMITTED.json"}
        or type(invocations) is not tuple
        or len(invocations) != 3
    ):
        raise ValueError("closed private paired output proof required")
    commit = decode_object(bundle["COMMITTED.json"], limit=262144)
    expected = {
        "schema_version",
        "operation",
        "source_commit",
        "case_count",
        "quality_success",
        "native_admission",
        "files",
        "invocations_manifest",
    }
    trace_sha, trace_bytes, trace_count = invocations
    trace_descriptor = {
        "sha256": trace_sha,
        "byte_count": trace_bytes,
        "format": "jsonl",
        "record_count": trace_count,
    }
    if (
        set(commit) != expected
        or commit["schema_version"] != 1
        or type(commit["schema_version"]) is not int
        or commit["operation"] != "paired_eval"
        or commit["source_commit"] != source_commit
        or commit["case_count"] != 164
        or type(commit["case_count"]) is not int
        or commit["quality_success"] is not False
        or commit["native_admission"] is not False
        or commit["invocations_manifest"] != trace_descriptor
        or type(commit["files"]) is not dict
        or set(commit["files"]) != {"RECEIPTS.jsonl", "EVALUATION.json"}
    ):
        raise ValueError("original paired output commit differs")
    for name in ("RECEIPTS.jsonl", "EVALUATION.json"):
        raw = bundle[name]
        if (
            type(raw) is not bytes
            or not 0 < len(raw) <= (2 * 1024**2 if name.endswith("jsonl") else 262144)
            or commit["files"][name]
            != {
                "sha256": _sha(raw),
                "byte_count": len(raw),
                "format": "jsonl" if name.endswith("jsonl") else "json",
            }
        ):
            raise ValueError("committed paired output bytes differ")
    values = _records(bundle["RECEIPTS.jsonl"], count=328, limit=2 * 1024**2)
    receipts = tuple(CodingCaseReceipt.model_validate(value) for value in values)
    pins = tuple(
        CodingCasePin(case_id=row["case_id"], oracle_sha256=row["oracle_record_sha256"])
        for row in EXPECTED_CASES
    )
    report = CodingEvaluatorFactory().build_report(binding=binding, case_pins=pins, receipts=receipts)
    summary = report.describe()
    evaluation = decode_object(bundle["EVALUATION.json"], limit=262144)
    if (
        set(evaluation)
        != {
            "schema_version",
            "operation",
            "source_commit",
            "binding",
            "case_count",
            "receipt_count",
            "quality_success",
            "native_admission",
            "cleanup_complete",
            "summary",
            "invocations_manifest",
        }
        or evaluation["schema_version"] != 1
        or type(evaluation["schema_version"]) is not int
        or evaluation["operation"] != "paired_eval"
        or evaluation["source_commit"] != source_commit
        or canonical_bytes(evaluation["binding"]) != canonical_bytes(binding.model_dump(mode="json"))
        or evaluation["case_count"] != 164
        or type(evaluation["case_count"]) is not int
        or evaluation["receipt_count"] != 328
        or type(evaluation["receipt_count"]) is not int
        or evaluation["quality_success"] is not False
        or evaluation["native_admission"] is not False
        or type(evaluation["cleanup_complete"]) is not bool
        or evaluation["invocations_manifest"] != trace_descriptor
        or canonical_bytes(evaluation["summary"]) != canonical_bytes(summary)
    ):
        raise ValueError("original complete paired evaluation summary differs")
    _verify_trace(trace, binding, receipts, evaluation["cleanup_complete"], trace_count)
    result = {
        "success": evaluation["cleanup_complete"],
        "evaluated": evaluation["cleanup_complete"],
        "status": "evaluated" if evaluation["cleanup_complete"] else "cleanup_unresolved",
        "case_count": 164,
        "receipt_count": 328,
        "cleanup_complete": evaluation["cleanup_complete"],
        "summary": summary,
        "quality_success": False,
        "native_admission": False,
    }
    if evaluation["cleanup_complete"] is True:
        # Only an evaluated pair gets the sentences and the order to repeat them. A result whose cleanup was
        # not confirmed is not a completed evaluation and carries neither.
        result["result_statement"] = pair_result_statement(summary)
        result["next_step"] = PAIR_RESULT_NEXT_STEP
    return result


def run_paired(*args, **kwargs):
    from .coding_pair_runner import run_paired as run

    return run(*args, **kwargs)
