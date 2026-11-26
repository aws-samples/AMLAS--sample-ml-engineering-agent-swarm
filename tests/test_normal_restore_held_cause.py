"""Status and resume say why the guardrail's input check held the restore of a run after its worker was replaced.

The first worker of a prompt run can end while its fine-tuning job runs. A resume request then restores the run in
a new worker. Before the stored conversation is loaded, every stored message goes through the deployed
guardrail's input check as the texts derived from it, and the completion result that the restore adds goes
whole. A text that the check blocks, masks or cannot assess holds the restore: the stored history is never
rewritten. The run has not ended, nothing kept is discarded, and every later restore checks again.

These tests pin what a caller reads then. The closed ``resume_state`` events stay as they were. One
``prompt_run_status`` event follows the ``failed`` event, and a status request answers with the same three added
keys: ``run_state`` ``restore_held``, ``message`` and ``restore_hold``. They name the position and kind of the
held message and what the guardrail reported for it, and never a matched value or an identifier. Of the checked
texts they hold only the role name of the held message, which is checked on its own.
The cause is kept in the process that ran the restore, for the session that owns the run, until the next restore
of that run starts. Controls pin what stays as it was: the texts the restore sends and their order, a hold without
a verdict of the check, a restore that fails for another reason, a transient error, another session and a busy
run.

Scripted guardrail answers in the shape of the recorded native answer; no native API, model or network. The
fixtures establish source behaviour, not native guardrail classification, native storage or a deployed run.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest
from bedrock_agentcore.runtime.context import BedrockAgentCoreContext
from botocore.exceptions import ClientError, ReadTimeoutError
from normal_sft_cold_fixture import FIT_FINAL_TEXT, close, cloud_data, original_precommit, setup
from test_input_admission_supplementary_characters import (
    CYCLE,
    GUARD,
    private_checker,
    recorded_none_response,
)
from test_normal_run_checkpoint import PROMPT, collect
from test_normal_run_checkpoint import normal_work as normal_work
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.async_callback import lambda_handler
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat import engine as chat_engine
from amlas.tui.chat import input_admission
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.guardrail_assessment import GuardrailAssessment
from amlas.tui.chat.input_admission import MAX_INPUT_BYTES, InputAdmissionReceipt
from amlas.tui.chat.tool_binding_backend import (
    ToolBindingBackend,
    _history_admission_texts,
    _InputAdmissionHeldError,
    _record_messages,
)

# Values that must never leave the check: a matched value, the guardrail's own output text and a reasoning
# signature. Each is built here, so a search of the answers for it cannot hit the test's own source.
MATCH = " ".join(("Rowan", "Ashford"))
NOTICE = " ".join(("The", "fixture", "guardrail", "withheld", "this", "input."))
SIGNATURE = "QUJD" * 252
SFT_USE = "declared-sft-use"
READ_USE = "declared-read-use"
RECEIPT_KEYS = {
    "schema_version",
    "check_id",
    "opaque_cycle",
    "boundary",
    "source",
    "guardrail_id",
    "guardrail_version",
    "region",
    "byte_count",
    "tool_name",
    "limit_bytes",
    "character_count",
    "byte_measurement",
    "byte_count_lower_bound",
    "keyed_digest",
    "outcome",
    "reason",
    "categories",
    "data_only",
    "native_admission",
    "quality_success",
}
FAILED = {"restore_stage": "snapshot_load", "exception_class": "Exception"}
OPENING = "The last restore of this run in this runtime process was held."
KEPT = (
    "The hold ended nothing and discarded nothing: the fine-tuning job, its results and every stored report"
    " are kept."
)
NEXT = (
    "Each resume_prompt_run request that starts a restore checks the stored conversation again. "
    "The run continues only after the check passes. In this answer, restore_hold carries the same facts as data."
)


def none_answer(text, guard=GUARD):
    """The recorded native answer without a finding, for the selected guardrail."""
    response = recorded_none_response(text)
    details = response["assessments"][0]["appliedGuardrailDetails"]
    details["guardrailId"] = guard
    details["guardrailArn"] = details["guardrailArn"].rsplit("/", 1)[0] + "/" + guard
    return response


def blocked_answer(text, guard=GUARD):
    """One content filter blocked the text at low confidence; the service returns its own notice as output."""
    response = none_answer(text, guard)
    response["action"] = "GUARDRAIL_INTERVENED"
    response["actionReason"] = "Guardrail blocked."
    for entry in response["assessments"][0]["contentPolicy"]["filters"]:
        if entry["type"] == "SEXUAL":
            entry.update(action="BLOCKED", confidence="LOW", detected=True)
    response["outputs"] = [{"text": NOTICE}]
    return response


def masked_answer(text, guard=GUARD):
    """One personal-data entity was masked; the service returns the text with the placeholder."""
    response = none_answer(text, guard)
    response["action"] = "GUARDRAIL_INTERVENED"
    response["actionReason"] = "Guardrail masked."
    for entry in response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]:
        if entry["type"] == "NAME":
            entry.update(action="ANONYMIZED", detected=True, match=MATCH)
    response["outputs"] = [{"text": text.replace(MATCH, "{NAME}")}]
    return response


class ScriptedGuard:
    """Stands in for the guardrail client. ``verdict(text)`` decides each answer; every sent text is recorded."""

    def __init__(self, verdict=None, guard=GUARD):
        self.verdict = verdict if verdict is not None else (lambda _text: None)
        self.guard = guard
        self.texts = []

    def close(self):
        pass

    def apply_guardrail(self, **kwargs):
        text = kwargs["content"][0]["text"]["text"]
        self.texts.append(text)
        answer = self.verdict(text)
        if isinstance(answer, BaseException):
            raise answer
        return none_answer(text, self.guard) if answer is None else answer(text, self.guard)


def as_stored(message):
    """A message as the restore reads it from the stored conversation: every object with its keys sorted."""
    return json.loads(json.dumps(message, sort_keys=True))


def history_texts(message):
    """The texts the restore sends for one message in the form it holds, in the order it sends them."""
    return _history_admission_texts(message)


def held_place(message):
    """The texts of the stored ``message`` up to the one the scripted guardrail answers for.

    That text is the first text block of the message, or its last text when it has no text block. The restore
    sends the texts of a message in this order, so the newest sends equal this list exactly when the check has
    reached that text of that message.
    """
    texts = history_texts(as_stored(message))
    for block in message["content"]:
        if type(block) is dict and type(block.get("text")) is str and block["text"] in texts:
            return texts[: texts.index(block["text"]) + 1]
    return texts


def reaches(message):
    """A test over the sent texts: true when the check has reached the held text of ``message``."""
    place = held_place(message)
    recent = []

    def reached(text):
        recent.append(text)
        del recent[: -len(place)]
        return recent == place

    return reached


def holds(message, answer, *, nth=1):
    """A verdict that answers with ``answer`` for the held text of ``message``, from its ``nth`` send on."""
    reached = reaches(message)
    seen = []

    def verdict(text):
        if not reached(text):
            return None
        seen.append(len(text))
        return answer if len(seen) >= nth else None

    return verdict


def sent(value):
    """The one compact JSON text of a value: the completion result, or a stored message that is sent whole."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def conversation():
    """Four stored messages of a run up to its fine-tuning call; every value is the test's own."""
    return [
        {"role": "user", "content": [{"text": "Fine-tune one licensed coding model."}]},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "", "signature": SIGNATURE}}},
                {"text": "Reading the profile list kept for " + MATCH + "."},
                {
                    "toolUse": {
                        "toolUseId": READ_USE,
                        "name": "list_coding_preparation_profiles",
                        "input": {},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": READ_USE,
                        "status": "success",
                        "content": [{"text": '{"profiles":[]}'}],
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"text": "Submitting the fine-tuning job."},
                {
                    "toolUse": {
                        "toolUseId": SFT_USE,
                        "name": "submit_qwen_sft",
                        "input": {
                            "model_ref": "model_acq_" + "2" * 32,
                            "preparation_ref": "prep_" + "3" * 32,
                            "recipe_id": "full_v1",
                        },
                    }
                },
            ],
        },
    ]


def stored_texts():
    """The texts the restore sends for the four messages of ``conversation``, written out.

    A stored message is read with its keys sorted. So the content comes before the role name, a tool call sends
    the keys and values of its input, then its name, then its id, and a tool result sends the compact JSON of
    its content list, then its status, then its id. An empty reasoning text is not sent.
    """
    return [
        ["Fine-tune one licensed coding model.", "user"],
        [
            SIGNATURE,
            "Reading the profile list kept for " + MATCH + ".",
            "list_coding_preparation_profiles",
            READ_USE,
            "assistant",
        ],
        ['[{"text":"{\\"profiles\\":[]}"}]', "success", READ_USE, "user"],
        [
            "Submitting the fine-tuning job.",
            "model_ref",
            "model_acq_" + "2" * 32,
            "preparation_ref",
            "prep_" + "3" * 32,
            "recipe_id",
            "full_v1",
            "submit_qwen_sft",
            SFT_USE,
            "assistant",
        ],
    ]


def declared_task(messages):
    """The restore's declared DATA input, not an enrolled TASK."""
    job_name = "declared-original-sft"
    return {
        "schema_version": 2,
        "logical_run_id": "0" * 32,
        "private_store_correlation": "declared-component-store",
        "source_refs": [],
        "causal_parent_ref": None,
        "operation_digest": "1" * 64,
        "native_job_name": job_name,
        "native_job_arn": "arn:aws:sagemaker:us-west-2:" + "1" * 12 + ":training-job/" + job_name,
        "pending_tool_use_id": SFT_USE,
        "sdk_snapshot": {
            "scope": "agent",
            "schema_version": "1.0",
            "created_at": "2026-01-01T00:00:00+00:00",
            "data": {
                "messages": copy.deepcopy(messages),
                "state": {},
                "conversation_manager_state": {},
                "interrupt_state": {},
            },
            "app_data": {},
        },
    }


def completion_result(task):
    """The completion result of the fine-tuning job, as the restore builds it."""
    event = {
        "schema_version": 1,
        "event_type": "original_sft_completed",
        "data_only": True,
        "request_ref": "sft_" + task["operation_digest"][:32],
        "tool_use_id": task["pending_tool_use_id"],
        "status": "verified_completed",
    }
    text = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return {"toolUseId": task["pending_tool_use_id"], "status": "success", "content": [{"text": text}]}


def restoring(monkeypatch, guard, *, messages=None, task=None, complete=True):
    """A chat engine at the restore boundary: the real admission and checker, inert agent and context."""
    task = declared_task(conversation() if messages is None else messages) if task is None else task
    selected = private_checker(guard)
    backend = object.__new__(ToolBindingBackend)
    backend.approval_policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["qwen_sft_flow"])
    backend._input_lock = threading.RLock()
    backend._input_hold = False
    backend._input_cycle = CYCLE
    backend._input_receipts = []
    backend._sft_store = object()
    backend._normal_original_result = (
        {
            "request_ref": "sft_" + task["operation_digest"][:32],
            "job_name": task["native_job_name"],
            "job_arn": task["native_job_arn"],
            "trained": True,
            "full_training_complete": True,
        }
        if complete
        else {"status": "PENDING"}
    )
    monkeypatch.setattr(backend, "_selected_input_checker", lambda: selected)
    monkeypatch.setattr(backend, "_activate_research_binding", lambda: None)
    monkeypatch.setattr(backend, "_qualify_research_binding", lambda *, active=True: None)
    effects = []

    class InertAgent:
        messages = None

        def load_snapshot(self, snapshot):
            effects.append("snapshot_load")
            self.messages = copy.deepcopy(snapshot.to_dict()["data"]["messages"])

    def build_agent():
        effects.append("agent_build")
        return InertAgent()

    monkeypatch.setattr(backend, "build_agent", build_agent)

    class InertContext:
        def qualify_original_completion(self, *args, **kwargs):
            effects.append("original_qualification")

        def claim_completion(self, *args, **kwargs):
            effects.append("completion_cas")
            return object()

        def qualify_completion(self, *args, **kwargs):
            effects.append("completion_qualification")

    engine = object.__new__(ChatEngine)
    engine.tool_backend = backend
    prefix = checkpoint._task_causal_prefix(task)
    completed = [*prefix, {"role": "user", "content": [{"toolResult": completion_result(task)}]}]
    return SimpleNamespace(
        engine=engine,
        backend=backend,
        task=task,
        context=InertContext(),
        effects=effects,
        guard=guard,
        prefix=prefix,
        completed=completed,
    )


def restore(work, site):
    """Run one restore at ``site`` and return the error that held it, or None."""
    try:
        if site == "pending":
            work.engine.load_normal_snapshot(work.task)
        else:
            work.engine.claim_normal_completion(work.context, work.task)
    except _InputAdmissionHeldError as error:
        return error
    return None


def cause_of(error):
    """The closed cause DATA that a held restore carries."""
    assert type(error).__name__ == "_RestoreAdmissionHeldError", "the restore hold carries no cause"
    assert isinstance(error, _InputAdmissionHeldError) and error.args == ()
    return error.data


def acting(cause):
    """What the guardrail reported as acting on the text: policy, category, action and confidence."""
    return [
        (record["policy"], record["category"], record["action"], record["confidence"])
        for record in cause["guardrail_assessment"]["records"]
        if record["action"] != "NONE"
    ]


def kinds(message):
    return sorted({next(iter(block)) for block in message["content"]})


def shown(*values):
    return json.dumps(values, ensure_ascii=False)


# ── the check keeps what the guardrail reported ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("answer", "outcome", "reason", "reported"),
    [
        (blocked_answer, "BLOCKED", "NATIVE_BLOCKED", [("CONTENT", "SEXUAL", "BLOCKED", "LOW")]),
        (masked_answer, "ANONYMIZED", "NATIVE_MASKED", [("PII", "NAME", "ANONYMIZED", None)]),
    ],
    ids=["blocked", "masked"],
)
def test_a_check_that_did_not_admit_the_text_unchanged_keeps_what_the_guardrail_reported(
    answer, outcome, reason, reported
):
    text = "The profile list is kept for " + MATCH + "."
    decision = private_checker(ScriptedGuard(lambda _text: answer)).check(
        text, boundary="HISTORY_TEXT", opaque_cycle=CYCLE
    )
    receipt = decision.receipt
    assert (receipt.outcome, receipt.reason) == (outcome, reason)
    assert type(receipt.assessment) is GuardrailAssessment
    assessment = receipt.assessment.to_dict()
    assert assessment["assessment_status"] == "complete" and len(assessment["records"]) == 13
    assert acting({"guardrail_assessment": assessment}) == reported
    blocked = next(record for record in assessment["records"] if record["category"] == "SEXUAL")
    assert blocked["direction"] == "INPUT" and blocked["filter_strength"] == "HIGH"
    # The serialized receipt is what reports carry. It keeps its keys, and nothing of the answer's text.
    serialized = receipt.to_dict()
    assert set(serialized) - {"blocking_categories"} == RECEIPT_KEYS
    visible = shown(serialized, assessment) + repr(receipt)
    assert MATCH not in visible and NOTICE not in visible and text not in visible


def test_an_admitted_check_keeps_no_assessment_and_the_closed_receipt_takes_none():
    decision = private_checker(ScriptedGuard()).check(
        "An ordinary sentence.", boundary="HISTORY_TEXT", opaque_cycle=CYCLE
    )
    assert decision.outcome == "ALLOW" and decision.receipt.assessment is None
    assert set(decision.receipt.to_dict()) == RECEIPT_KEYS
    held = (
        private_checker(ScriptedGuard(lambda _text: blocked_answer))
        .check("An ordinary sentence.", boundary="HISTORY_TEXT", opaque_cycle=CYCLE)
        .receipt
    )
    # The assessment is no field of the closed receipt. The constructor takes none, and a receipt built from
    # the serialized record equals the original and carries none.
    inventory = dataclasses.fields(InputAdmissionReceipt)
    assert "assessment" not in {item.name for item in inventory}
    serialized = held.to_dict()
    listed = ("categories", "blocking_categories")
    arguments = {
        item.name: serialized[item.name]
        for item in inventory
        if item.init and item.name not in (*listed, "transport_diagnostic")
    }
    arguments.update({name: tuple(serialized[name]) for name in listed})
    rebuilt = InputAdmissionReceipt(**arguments)
    assert rebuilt == held and rebuilt.to_dict() == serialized
    assert type(held.assessment) is GuardrailAssessment and rebuilt.assessment is None
    with pytest.raises(TypeError):
        InputAdmissionReceipt(**arguments, assessment=held.assessment)


def test_a_projection_failure_never_changes_the_decision(monkeypatch):
    def broken(_trace):
        raise RuntimeError("declared projection fault")

    monkeypatch.setattr(input_admission, "project_guardrail_trace", broken)
    decision = private_checker(ScriptedGuard(lambda _text: blocked_answer)).check(
        "An ordinary sentence.", boundary="HISTORY_TEXT", opaque_cycle=CYCLE
    )
    assert (decision.outcome, decision.receipt.reason) == ("BLOCKED", "NATIVE_BLOCKED")
    assert decision.forward_text is None and decision.receipt.assessment is None
    assert decision.receipt.blocking_categories == ("CONTENT:SEXUAL",)


def test_an_answer_with_more_assessments_than_the_projection_reads_keeps_its_decision_and_no_record():
    def nine(text, guard=GUARD):
        response = blocked_answer(text, guard)
        response["assessments"] = [copy.deepcopy(response["assessments"][0]) for _ in range(9)]
        return response

    decision = private_checker(ScriptedGuard(lambda _text: nine)).check(
        "An ordinary sentence.", boundary="HISTORY_TEXT", opaque_cycle=CYCLE
    )
    assert (decision.outcome, decision.receipt.reason) == ("BLOCKED", "NATIVE_BLOCKED")
    assert decision.receipt.assessment.to_dict() == {
        "schema_version": 1,
        "assessment_status": "invalid",
        "action": "UNKNOWN",
        "records": [],
    }


# ── the restore carries the cause of its hold ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("site", ["pending", "first_pass", "second_pass"])
def test_a_held_stored_message_is_named_with_its_position_kind_and_what_the_guardrail_reported(
    monkeypatch, site
):
    messages = conversation()
    guard = ScriptedGuard(holds(messages[1], blocked_answer, nth=2 if site == "second_pass" else 1))
    work = restoring(monkeypatch, guard, messages=messages, complete=site != "pending")
    cause = cause_of(restore(work, site))
    assert cause == {
        "schema_version": 1,
        "data_only": True,
        "boundary": "HISTORY_TEXT",
        "message_position": 2,
        "message_count": 4 if site == "pending" else 5,
        "message_role": "assistant",
        "message_blocks": ["reasoningContent", "text", "toolUse"],
        "outcome": "BLOCKED",
        "reason": "NATIVE_BLOCKED",
        "guardrail_assessment": cause["guardrail_assessment"],
    }
    assert acting(cause) == [("CONTENT", "SEXUAL", "BLOCKED", "LOW")]
    assert cause["guardrail_assessment"]["assessment_status"] == "complete"
    # The hold itself is the unchanged one: latched, before the conversation is loaded.
    assert work.backend._input_hold is True
    assert "snapshot_load" not in work.effects and "agent_build" not in work.effects
    assert ("completion_cas" in work.effects) is (site == "second_pass")
    visible = shown(cause)
    assert all(token not in visible for token in (MATCH, NOTICE, SIGNATURE, READ_USE, GUARD))


def test_a_held_completion_result_is_named_as_the_last_message(monkeypatch):
    guard = ScriptedGuard(
        lambda text: blocked_answer if json.loads(text).get("status") == "success" else None
    )
    work = restoring(monkeypatch, guard)
    cause = cause_of(restore(work, "first_pass"))
    assert guard.texts == [sent(completion_result(work.task))]
    assert (cause["boundary"], cause["message_position"], cause["message_count"]) == ("TOOL_RESULT", 5, 5)
    assert (cause["message_role"], cause["message_blocks"]) == ("user", ["toolResult"])
    assert (cause["outcome"], cause["reason"]) == ("BLOCKED", "NATIVE_BLOCKED")
    assert work.effects == [] and SFT_USE not in shown(cause)


def test_a_masked_stored_message_names_the_entity_and_never_the_value(monkeypatch):
    messages = conversation()
    guard = ScriptedGuard(holds(messages[1], masked_answer))
    work = restoring(monkeypatch, guard, messages=messages)
    cause = cause_of(restore(work, "first_pass"))
    assert (cause["outcome"], cause["reason"]) == ("ANONYMIZED", "NATIVE_MASKED")
    assert acting(cause) == [("PII", "NAME", "ANONYMIZED", None)]
    assert cause["message_position"] == 2 and work.effects == []
    # The message was sent with the value; nothing the caller reads holds it.
    assert MATCH in guard.texts[-1] and MATCH not in shown(cause) and "{NAME}" not in shown(cause)


@pytest.mark.parametrize("case", ["transport", "limit"])
def test_a_check_without_a_usable_verdict_names_its_reason(monkeypatch, case):
    messages = conversation()
    if case == "transport":
        fault = ReadTimeoutError(endpoint_url="https://guardrail.invalid")
        guard = ScriptedGuard(holds(messages[1], fault))
    else:
        # One character more than the limit of a single check as one text: the message is sent whole and
        # refused before any call.
        messages[1]["content"][1]["text"] += "a" * (MAX_INPUT_BYTES + 1 - len(sent(messages[1])))
        guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, messages=messages)
    cause = cause_of(restore(work, "first_pass"))
    assert cause["outcome"] == "UNKNOWN"
    assert cause["reason"] == ("TRANSPORT_UNKNOWN" if case == "transport" else "INPUT_LIMIT")
    assert cause["message_position"] == 2 and "guardrail_assessment" not in cause
    # Sent: the completion result and the texts of the first message; in the transport case also the texts
    # of the second message up to its held text. The message over the limit was refused before a call.
    upto = history_texts(as_stored(messages[0]))
    place = held_place(messages[1]) if case == "transport" else []
    assert len(guard.texts) == 1 + len(upto) + len(place)
    assert "guardrail.invalid" not in shown(cause)


def masks(value):
    """An answer that masks ``value`` as a name; ``placeholder`` is what the output holds in its place."""

    def answer(text, guard=GUARD, *, placeholder="{NAME}"):
        response = none_answer(text, guard)
        response["action"] = "GUARDRAIL_INTERVENED"
        response["actionReason"] = "Guardrail masked."
        for entry in response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]:
            if entry["type"] == "NAME":
                entry.update(action="ANONYMIZED", detected=True, match=value)
        response["outputs"] = [{"text": text.replace(value, placeholder)}]
        return response

    return answer


def said(cause_sentence, *reported):
    """The whole message for one cause sentence and what the guardrail reported."""
    return " ".join((OPENING, cause_sentence, *reported, KEPT, NEXT))


def test_a_masked_completion_result_and_a_message_without_content_read_in_their_own_words(monkeypatch):
    build = getattr(runtime, "_restore_held_message", None)
    assert build is not None, "the runtime builds no message for a held restore"
    # The service masks a value in the completion result. The restore builds that result for the finished job
    # and adds it as the last message; it is no stored message.
    guard = ScriptedGuard(
        lambda text: masks(SFT_USE) if json.loads(text).get("status") == "success" else None
    )
    cause = cause_of(restore(restoring(monkeypatch, guard), "first_pass"))
    assert (cause["boundary"], cause["outcome"], cause["reason"]) == (
        "TOOL_RESULT",
        "ANONYMIZED",
        "NATIVE_MASKED",
    )
    assert build(cause) == said(
        "The guardrail's input check masked text in message 5 of 5 of the conversation being restored before"
        " the conversation was loaded (user message with toolResult); the restore was held because the"
        " completion result that the restore adds is never rewritten.",
        "The guardrail reported: personal-data entity NAME, action ANONYMIZED.",
    )
    assert "stored message" not in build(cause) and SFT_USE not in shown(cause, build(cause))
    # A stored message without a content block is named by its role alone.
    messages = conversation()
    messages[1:1] = [{"role": "assistant", "content": []}, {"role": "user", "content": [{"text": "Go on."}]}]
    guard = ScriptedGuard(holds(messages[1], blocked_answer))
    cause = cause_of(restore(restoring(monkeypatch, guard, messages=messages), "first_pass"))
    assert (cause["message_position"], cause["message_count"], cause["message_blocks"]) == (2, 7, [])
    assert build(cause) == said(
        "The guardrail's input check blocked text in message 2 of 7 of the conversation being restored"
        " before the conversation was loaded (assistant message).",
        "The guardrail reported: content filter SEXUAL, confidence LOW, action BLOCKED.",
    )
    unassessed = {key: value for key, value in cause.items() if key != "guardrail_assessment"}
    assert build({**unassessed, "outcome": "UNKNOWN", "reason": "TRANSPORT_UNKNOWN"}) == said(
        "The guardrail's input check gave no usable verdict for text in message 2 of 7 of the conversation"
        " being restored before the conversation was loaded (assistant message; reason TRANSPORT_UNKNOWN)."
    )
    assert build({**unassessed, "outcome": "ANONYMIZED", "reason": "NATIVE_MASKED"}) == said(
        "The guardrail's input check masked text in message 2 of 7 of the conversation being restored before"
        " the conversation was loaded (assistant message); the restore was held because the message that"
        " contains the text is never rewritten."
    )


def test_a_completion_result_masked_in_a_history_pass_reads_as_a_message_of_the_conversation(monkeypatch):
    """The history pass also checks the completion result that the restore adds, as the last message."""
    build = getattr(runtime, "_restore_held_message", None)
    assert build is not None, "the runtime builds no message for a held restore"
    task = declared_task(conversation())
    # The history pass sends the content list of the completion result as one text. The result's own check
    # sends the whole result, which is another text and passes here.
    listed = sent(completion_result(task)["content"])
    guard = ScriptedGuard(lambda text: masks("verified_completed") if text == listed else None)
    work = restoring(monkeypatch, guard, task=task)
    cause = cause_of(restore(work, "first_pass"))
    assert guard.texts[0] == sent(completion_result(task)) and guard.texts[-1] == listed
    assert (cause["boundary"], cause["outcome"], cause["reason"]) == (
        "HISTORY_TEXT",
        "ANONYMIZED",
        "NATIVE_MASKED",
    )
    assert (cause["message_position"], cause["message_count"]) == (5, 5)
    assert (cause["message_role"], cause["message_blocks"]) == ("user", ["toolResult"])
    # Message 5 is the completion result, which is no stored message.
    assert build(cause) == said(
        "The guardrail's input check masked text in message 5 of 5 of the conversation being restored before"
        " the conversation was loaded (user message with toolResult); the restore was held because the"
        " message that contains the text is never rewritten.",
        "The guardrail reported: personal-data entity NAME, action ANONYMIZED.",
    )
    assert "stored message" not in build(cause) and work.effects == []


def unread(text, guard=GUARD):
    """An answer with a policy key that the check does not read."""
    response = blocked_answer(text, guard)
    response["assessments"][0]["declaredPolicy"] = {}
    return response


def detected_without_action(text, guard=GUARD):
    """An answer that marks a filter as detected and names no action for it."""
    response = none_answer(text, guard)
    response["assessments"][0]["contentPolicy"]["filters"][0].update(detected=True)
    return response


def intervened_without_record(text, guard=GUARD):
    """An answer that says the guardrail acted and names no record that did."""
    response = none_answer(text, guard)
    response["action"] = "GUARDRAIL_INTERVENED"
    return response


@pytest.mark.parametrize(
    ("answer", "reason", "assessed", "reported"),
    [
        (unread, "NATIVE_INVALID", False, []),
        (detected_without_action, "NATIVE_INVALID", True, []),
        (intervened_without_record, "NATIVE_UNKNOWN", True, []),
        # The output holds the placeholder of a category that no record reports, so the check refuses it.
        (
            lambda text, guard=GUARD: masks(MATCH)(text, guard, placeholder="{EMAIL}"),
            "MASK_INVALID",
            True,
            ["The guardrail reported: personal-data entity NAME, action ANONYMIZED."],
        ),
    ],
    ids=["unread", "detected_without_action", "intervened_without_record", "mask_refused"],
)
def test_an_answer_that_the_check_refuses_names_what_the_guardrail_reported_only_if_it_was_read(
    monkeypatch, answer, reason, assessed, reported
):
    """The check keeps the assessment of an answer it could read, also when a later step refuses the answer."""
    messages = conversation()
    messages[1]["content"][1]["text"] += " Kept again for " + MATCH + "."
    guard = ScriptedGuard(holds(messages[1], answer))
    work = restoring(monkeypatch, guard, messages=messages)
    cause = cause_of(restore(work, "first_pass"))
    assert (cause["outcome"], cause["reason"]) == ("UNKNOWN", reason)
    assert ("guardrail_assessment" in cause) is assessed
    if assessed:
        assert cause["guardrail_assessment"]["assessment_status"] == "complete"
        assert len(cause["guardrail_assessment"]["records"]) == 13
    assert runtime._restore_held_message(cause) == said(
        "The guardrail's input check gave no usable verdict for text in message 2 of 5 of the conversation"
        " being restored before the conversation was loaded (assistant message with reasoningContent, text,"
        f" toolUse; reason {reason}).",
        *reported,
    )
    assert work.effects == [] and MATCH not in shown(cause) and "{NAME}" not in shown(cause)


@pytest.mark.parametrize("case", ["latched", "selection", "foreign_receipt"])
def test_a_hold_without_a_verdict_of_this_check_stays_the_plain_hold(monkeypatch, case):
    """Only a receipt that this check wrote for this text names a cause."""
    guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, complete=case != "latched")
    if case == "latched":
        # An earlier check left a blocked receipt for the boundary of the restore's first admission, and the
        # hold is already set: no check runs now, and the earlier receipt is no verdict on this text.
        stale = private_checker(ScriptedGuard(lambda _text: blocked_answer)).check(
            "An earlier text.", boundary="HISTORY_TEXT", opaque_cycle=CYCLE
        )
        work.backend._input_receipts.append(stale.receipt)
        work.backend._input_hold = True
    elif case == "selection":

        def changed_selection():
            raise _InputAdmissionHeldError()

        monkeypatch.setattr(work.backend, "_selected_input_checker", changed_selection)
    else:

        class DeclaredChecker:
            def check(self, text, *, boundary, opaque_cycle, tool_name):
                return SimpleNamespace(outcome="BLOCKED", forward_text=None, receipt={"boundary": boundary})

        declared = DeclaredChecker()
        monkeypatch.setattr(work.backend, "_selected_input_checker", lambda: declared)
    error = restore(work, "pending" if case == "latched" else "first_pass")
    assert type(error) is _InputAdmissionHeldError and not hasattr(error, "data")
    assert work.backend._input_hold is True and guard.texts == [] and work.effects == []


def test_the_cause_is_built_only_from_closed_values():
    build = getattr(chat_engine, "_restore_hold_data", None)
    assert build is not None, "the restore builds no cause"
    checker = private_checker(ScriptedGuard(lambda text: blocked_answer if "held" in text else None))
    held = checker.check("A held text.", boundary="HISTORY_TEXT", opaque_cycle=CYCLE).receipt
    admitted = checker.check("An ordinary text.", boundary="HISTORY_TEXT", opaque_cycle=CYCLE).receipt
    message = {"role": "assistant", "content": [{"text": "A held text."}, {"declaredBlock": {}}, "bare"]}
    cause = build(held, "HISTORY_TEXT", 1, 3, message)
    assert cause["message_blocks"] == ["other", "text"] and cause["message_role"] == "assistant"
    assert cause["guardrail_assessment"] == held.assessment.to_dict()
    # An assessment of another type beside the receipt is left out.
    object.__setattr__(held, "assessment", held.assessment.to_dict())
    assert "guardrail_assessment" not in build(held, "HISTORY_TEXT", 1, 3, message)
    for receipt, boundary, position, count, value in (
        (admitted, "HISTORY_TEXT", 1, 3, message),
        (held, "TOOL_RESULT", 1, 3, message),
        (held.to_dict(), "HISTORY_TEXT", 1, 3, message),
        (held, "HISTORY_TEXT", 0, 3, message),
        (held, "HISTORY_TEXT", 4, 3, message),
        (held, "HISTORY_TEXT", True, 3, message),
        (held, "HISTORY_TEXT", 1, 3, {"role": "system", "content": []}),
        (held, "HISTORY_TEXT", 1, 3, {"role": "user", "content": "text"}),
        (held, "HISTORY_TEXT", 1, 3, [message]),
    ):
        assert build(receipt, boundary, position, count, value) is None


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_the_restore_sends_the_same_texts_in_the_same_order(monkeypatch, site):
    """Control: every stored message goes to the check as the texts derived from it, in the stored order."""
    guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, complete=site == "completed")
    assert restore(work, site if site == "pending" else "first_pass") is None
    # The helpers of this file derive the texts of a stored message as they are written out here.
    assert [history_texts(message) for message in work.prefix] == stored_texts()
    history = [text for texts in stored_texts() for text in texts]
    if site == "pending":
        expected = history
        boundaries = ["HISTORY_TEXT"] * 21
    else:
        # The restore adds the completion result as the last message, in the key order it builds it in.
        result = completion_result(work.task)
        history += ["user", SFT_USE, "success", sent(result["content"])]
        expected = [sent(result), *history, *history]
        boundaries = ["TOOL_RESULT", *["HISTORY_TEXT"] * 50]
    assert guard.texts == expected
    assert [receipt.boundary for receipt in work.backend._input_receipts] == boundaries
    assert [receipt.tool_name for receipt in work.backend._input_receipts] == [
        "submit_qwen_sft" if boundary == "TOOL_RESULT" else "UNKNOWN" for boundary in boundaries
    ]
    assert all(getattr(receipt, "assessment", None) is None for receipt in work.backend._input_receipts)
    assert work.backend._input_hold is False and "snapshot_load" in work.effects
    assert _record_messages(work.engine._normal_agent.messages)[: len(work.prefix)] == work.prefix


# ── what a caller of the runtime reads ───────────────────────────────────────────────────────────────────────


async def requested(payload):
    return [event async for event in runtime.agent_invocation(payload) if event.get("event") != "heartbeat"]


def unavailable(run_id):
    return {"event": "prompt_run_status", "status": "unavailable", "run_id": run_id, "found": False}


def restoring_state(run_id):
    return {"event": "resume_state", "state": "restoring", "run_id": run_id}


def failed_state(run_id, diagnostic=None):
    return {
        "event": "resume_state",
        "state": "failed",
        "run_id": run_id,
        "diagnostic": FAILED if diagnostic is None else diagnostic,
    }


def held_answer(base, message, cause):
    return {**base, "run_state": "restore_held", "message": message, "restore_hold": cause}


def blocked_message(position, count, role, blocks):
    return " ".join(
        (
            OPENING,
            f"The guardrail's input check blocked text in message {position} of {count} of the conversation"
            " being restored before the conversation was loaded"
            f" ({role} message with {', '.join(blocks)}).",
            "The guardrail reported: content filter SEXUAL, confidence LOW, action BLOCKED.",
            KEPT,
            NEXT,
        )
    )


def later_clock(monkeypatch):
    """A resume clock the test can move past the pause that follows a failed restore."""
    ahead = SimpleNamespace(seconds=0.0)
    monkeypatch.setattr(runtime, "_resume_clock", lambda: time.monotonic() + ahead.seconds)
    return ahead


def selected_run(normal_work):
    """One enrolled prompt run of the normal fixture; returns its id and the stored state."""
    _work, ports, model = normal_work
    events = collect({"prompt": PROMPT})
    assert events[-1]["status"] == "response_received" and model.calls == 1
    run_id = next(event["run_id"] for event in events if event["event"] == "run_started")
    return run_id, copy.deepcopy(ports.items), copy.deepcopy(ports.objects)


def inject_restore(monkeypatch, guard, *, messages=None, complete=False, prepare=None):
    """Hand the runtime's restore a declared TASK and a fresh engine with the real admission for every attempt.

    The selectors reach only the restore boundary of the app. They are no evidence of a qualified TASK and grant
    nothing: the engine's own admission, checker and hold are the real ones.
    """
    task = declared_task(conversation() if messages is None else messages)
    built = []
    monkeypatch.setattr(checkpoint.NormalRunContext, "has_task_checkpoint", lambda self, **_: True)
    monkeypatch.setattr(
        checkpoint.NormalRunContext, "load_qualified_task", lambda self, **_: (task, (), None, None)
    )

    def engine_for(*, approval_policy, normal_restore):
        assert approval_policy is not None and normal_restore[1] is task
        built.append(restoring(monkeypatch, guard, task=task, complete=complete))
        if prepare is not None:
            prepare(built[-1])
        return built[-1].engine

    monkeypatch.setattr(chat_engine, "ChatEngine", engine_for)
    return SimpleNamespace(task=task, built=built, guard=guard)


def status_reads(monkeypatch, answer):
    """Let the durable status read answer ``answer``; the app's own part of the status is what is under test."""
    monkeypatch.setattr(
        checkpoint,
        "read_normal_run_status",
        lambda run_id, session_hash, policy=None, resume=False: dict(answer),
    )


@pytest.mark.parametrize("complete", [False, True], ids=["pending", "completed"])
def test_a_held_restore_names_its_cause_after_the_closed_failure_and_at_status(
    normal_work, monkeypatch, complete
):
    work, ports, model = normal_work
    run_id, items, objects = selected_run(normal_work)
    messages = conversation()
    guard = ScriptedGuard(holds(messages[1], blocked_answer))
    selected = inject_restore(monkeypatch, guard, messages=messages, complete=complete)
    ahead = later_clock(monkeypatch)

    first = collect({"action": "resume_prompt_run", "run_id": run_id})
    count = 5 if complete else 4
    message = blocked_message(2, count, "assistant", ["reasoningContent", "text", "toolUse"])
    # The two closed events are the ones a held restore always sent; one status event follows them.
    assert first[:2] == [restoring_state(run_id), failed_state(run_id)]
    assert len(first) == 3, "the resume request names no cause"
    cause = first[2]["restore_hold"]
    assert first[2] == held_answer(unavailable(run_id), message, cause)
    assert (cause["boundary"], cause["message_position"], cause["message_count"]) == (
        "HISTORY_TEXT",
        2,
        count,
    )
    assert (cause["message_role"], cause["outcome"], cause["reason"]) == (
        "assistant",
        "BLOCKED",
        "NATIVE_BLOCKED",
    )
    assert acting(cause) == [("CONTENT", "SEXUAL", "BLOCKED", "LOW")]
    # A second request inside the pause starts no restore and names the same cause.
    second = collect({"action": "resume_prompt_run", "run_id": run_id})
    assert second == [failed_state(run_id), first[2]] and len(selected.built) == 1
    # The status keeps what it answered and adds the same three keys.
    status_reads(monkeypatch, unavailable(run_id))
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [first[2]]
    running = {
        "event": "prompt_run_status",
        "status": "available",
        "run_id": run_id,
        "found": True,
        "progress": {"stage": "sft", "pending": True},
    }
    status_reads(monkeypatch, running)
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [
        held_answer(running, message, cause)
    ]
    # An answer of another kind is left as it is.
    enrolled = {"event": "prompt_run_status", "status": "enrolled", "run_id": run_id, "found": True}
    status_reads(monkeypatch, enrolled)
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [enrolled]
    # After the pause the next request restores again, with a fresh engine and fresh checks.
    ahead.seconds = runtime._RESUME_FAILED_COOLDOWN_SECONDS + 1
    assert collect({"action": "resume_prompt_run", "run_id": run_id}) == first and len(selected.built) == 2
    # No answer holds a checked text, a matched value, the guardrail's output or an identifier of the check.
    visible = shown(first, second)
    assert all(token not in visible for token in (MATCH, NOTICE, SIGNATURE, READ_USE, SFT_USE, GUARD))
    assert all(text not in visible for text in guard.texts)
    # Nothing was written or discarded, no model was called and no job was submitted.
    assert ports.items == items and ports.objects == objects
    assert model.calls == 1 and work.native.requests == []


def test_a_masked_message_and_a_check_without_a_verdict_read_in_their_own_words(normal_work, monkeypatch):
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    script = SimpleNamespace(answer=masked_answer)
    reached = reaches(messages[1])
    guard = ScriptedGuard(lambda text: script.answer if reached(text) else None)
    inject_restore(monkeypatch, guard, messages=messages)
    ahead = later_clock(monkeypatch)
    masked = collect({"action": "resume_prompt_run", "run_id": run_id})[-1]
    kind = "(assistant message with reasoningContent, text, toolUse)"
    assert masked["message"] == " ".join(
        (
            OPENING,
            "The guardrail's input check masked text in message 2 of 4 of the conversation being restored before"
            " the conversation was loaded "
            + kind
            + "; the restore was held because the message that contains the"
            " text is never rewritten.",
            "The guardrail reported: personal-data entity NAME, action ANONYMIZED.",
            KEPT,
            NEXT,
        )
    )
    assert MATCH not in shown(masked)
    script.answer = ReadTimeoutError(endpoint_url="https://guardrail.invalid")
    ahead.seconds = runtime._RESUME_FAILED_COOLDOWN_SECONDS + 1
    unknown = collect({"action": "resume_prompt_run", "run_id": run_id})[-1]
    assert unknown["message"] == " ".join(
        (
            OPENING,
            "The guardrail's input check gave no usable verdict for text in message 2 of 4 of the"
            " conversation being restored before the conversation was loaded (assistant message with"
            " reasoningContent, text, toolUse; reason TRANSPORT_UNKNOWN).",
            KEPT,
            NEXT,
        )
    )
    assert unknown["restore_hold"]["outcome"] == "UNKNOWN" and unknown["run_state"] == "restore_held"


def test_the_message_lists_at_most_four_reported_records():
    build = getattr(runtime, "_restore_held_message", None)
    assert build is not None, "the runtime builds no message for a held restore"
    records = [
        {"policy": "CONTENT", "category": category, "action": "BLOCKED", "confidence": "HIGH"}
        for category in ("HATE", "INSULTS", "VIOLENCE")
    ]
    records += [
        {"policy": "PII", "category": category, "action": "ANONYMIZED", "confidence": None}
        for category in ("EMAIL", "PHONE", "NAME")
    ]
    records.append({"policy": "CONTENT", "category": "SEXUAL", "action": "NONE", "confidence": "NONE"})
    text = build(
        {
            "boundary": "HISTORY_TEXT",
            "message_position": 1,
            "message_count": 2,
            "message_role": "user",
            "message_blocks": ["toolResult"],
            "outcome": "BLOCKED",
            "reason": "NATIVE_BLOCKED",
            "guardrail_assessment": {"records": records},
        }
    )
    assert (
        "The guardrail reported: content filter HATE, confidence HIGH, action BLOCKED; content filter INSULTS,"
        " confidence HIGH, action BLOCKED; content filter VIOLENCE, confidence HIGH, action BLOCKED;"
        " personal-data entity EMAIL, action ANONYMIZED and 2 more." in text
    )
    assert "SEXUAL" not in text and "(user message with toolResult)" in text


def test_a_restore_that_passes_after_a_held_one_drops_the_cause(normal_work, monkeypatch):
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    script = SimpleNamespace(answer=blocked_answer)
    reached = reaches(messages[1])
    guard = ScriptedGuard(lambda text: script.answer if reached(text) else None)
    selected = inject_restore(monkeypatch, guard, messages=messages)
    ahead = later_clock(monkeypatch)
    held = collect({"action": "resume_prompt_run", "run_id": run_id})
    assert held[-1].get("run_state") == "restore_held"
    # The guardrail's verdict changes; the next restore passes its checks and loads the conversation.
    script.answer = None
    ahead.seconds = runtime._RESUME_FAILED_COOLDOWN_SECONDS + 1
    passed = collect({"action": "resume_prompt_run", "run_id": run_id})
    assert passed[0] == restoring_state(run_id) and len(passed) == 2
    assert passed[1]["event"] == "prompt_run_status" and "restore_hold" not in passed[1]
    assert selected.built[-1].effects == ["agent_build", "snapshot_load"]
    status_reads(monkeypatch, unavailable(run_id))
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [unavailable(run_id)]
    assert "restore_held" not in shown(passed)


def test_the_stream_function_reads_the_first_event_of_a_held_resume_as_before(normal_work, monkeypatch):
    """The wake delivery reads only the first event; it stays the closed state, and the cause is never first."""
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    inject_restore(monkeypatch, ScriptedGuard(holds(messages[1], blocked_answer)), messages=messages)
    first = collect({"action": "resume_prompt_run", "run_id": run_id})
    second = collect({"action": "resume_prompt_run", "run_id": run_id})
    assert lambda_handler._resume_event_state(json.dumps(first[0]), run_id) == "restoring"
    assert lambda_handler._resume_event_state(json.dumps(first[1]), run_id) == "failed"
    assert lambda_handler._resume_event_state(json.dumps(second[0]), run_id) == "failed"
    assert second[1] == first[2] and second[1]["event"] == "prompt_run_status"
    with pytest.raises(ValueError):
        lambda_handler._resume_event_state(json.dumps(second[1]), run_id)


def test_a_cause_that_cannot_be_rendered_leaves_the_closed_answers(normal_work, monkeypatch):
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    inject_restore(monkeypatch, ScriptedGuard(holds(messages[1], blocked_answer)), messages=messages)
    assert collect({"action": "resume_prompt_run", "run_id": run_id})[-1].get("run_state") == "restore_held"

    def broken(_hold):
        raise KeyError("declared rendering fault")

    monkeypatch.setattr(runtime, "_restore_held_message", broken)
    assert collect({"action": "resume_prompt_run", "run_id": run_id}) == [failed_state(run_id)]
    status_reads(monkeypatch, unavailable(run_id))
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [unavailable(run_id)]


def test_the_cause_event_belongs_to_the_failed_answer_it_follows(normal_work, monkeypatch):
    """A request that answered inside the pause sends the cause of that answer, whatever replaced the record."""
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    inject_restore(monkeypatch, ScriptedGuard(holds(messages[1], blocked_answer)), messages=messages)
    first = collect({"action": "resume_prompt_run", "run_id": run_id})

    async def interleaved():
        stream = runtime.agent_invocation({"action": "resume_prompt_run", "run_id": run_id})
        answered = await stream.__anext__()
        # Another restore of the run ends before this stream goes on: it was held at another message.
        with runtime._PROMPT_REPORT_LOCK:
            record = runtime._RESUME_WORKERS[run_id]
            owner, cause = record["restore_hold"]
            runtime._RESUME_WORKERS[run_id] = {
                **record,
                "restore_hold": (owner, {**cause, "message_position": 4, "message_role": "user"}),
            }
        return [answered] + [event async for event in stream]

    assert asyncio.run(interleaved()) == [failed_state(run_id), first[2]]
    # The next request reads the record as it is now.
    later = collect({"action": "resume_prompt_run", "run_id": run_id})
    assert later[1]["restore_hold"]["message_position"] == 4 and "message 4 of 4" in later[1]["message"]


def test_a_stream_that_ended_without_the_failed_state_sends_no_cause_after_it(normal_work, monkeypatch):
    """The cause event follows the closed ``failed`` state and no other event.

    An operator's run limit ends the stream with its own last event while the worker still restores. A caller
    that asks that stream for more after the restore was held gets nothing more. The status answer and the
    next request carry the cause.
    """
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    answered = threading.Event()

    reached = reaches(messages[1])

    def verdict(text):
        if not reached(text):
            return None
        # The check of this message answers only after the stream has ended at the limit.
        answered.wait(30)
        return blocked_answer

    inject_restore(monkeypatch, ScriptedGuard(verdict), messages=messages)
    monkeypatch.setenv("AMLAS_MAX_RUN_SECONDS", "0.2")

    async def limited():
        events = []
        async for event in runtime.agent_invocation({"action": "resume_prompt_run", "run_id": run_id}):
            if event.get("event") == "heartbeat":
                continue
            events.append(event)
            if event.get("event") == "run_completed":
                # Let the worker go on to its held restore before this caller asks the stream for more.
                answered.set()
                for _ in range(600):
                    with runtime._PROMPT_REPORT_LOCK:
                        record = runtime._RESUME_WORKERS.get(run_id)
                    if record is not None and record["state"] == "failed":
                        break
                    await asyncio.sleep(0.05)
        return events

    events = asyncio.run(limited())
    monkeypatch.delenv("AMLAS_MAX_RUN_SECONDS")
    with runtime._PROMPT_REPORT_LOCK:
        record = dict(runtime._RESUME_WORKERS[run_id])
    assert record["state"] == "failed" and record["diagnostic"] == FAILED and "restore_hold" in record
    assert events[0] == restoring_state(run_id)
    assert (events[1]["event"], events[1]["status"], events[1]["reason"], events[1]["worker_finished"]) == (
        "run_completed",
        "outcome_unknown",
        "wall_clock_exceeded",
        False,
    )
    assert len(events) == 2, "a cause event followed an event that is not the failed state"
    # The cause was kept: the status answer and the next request inside the pause carry it.
    status_reads(monkeypatch, unavailable(run_id))
    status = collect({"action": "get_prompt_run_status", "run_id": run_id})
    message = blocked_message(2, 4, "assistant", ["reasoningContent", "text", "toolUse"])
    assert status == [held_answer(unavailable(run_id), message, status[0].get("restore_hold"))]
    assert collect({"action": "resume_prompt_run", "run_id": run_id}) == [failed_state(run_id), status[0]]


def test_another_session_gets_no_cause(normal_work, monkeypatch):
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    selected = inject_restore(
        monkeypatch, ScriptedGuard(holds(messages[1], blocked_answer)), messages=messages
    )
    owner = collect({"action": "resume_prompt_run", "run_id": run_id})
    assert owner[-1].get("run_state") == "restore_held"
    BedrockAgentCoreContext.set_request_context("another-request", "another-session-of-the-same-process")
    # The other session reads the closed answers of a run it does not own, inside the pause and at status.
    assert collect({"action": "resume_prompt_run", "run_id": run_id}) == [failed_state(run_id)]
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [unavailable(run_id)]
    status_reads(monkeypatch, unavailable(run_id))
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [unavailable(run_id)]
    assert len(selected.built) == 1


def test_a_busy_run_answers_its_closed_state_alone(normal_work, monkeypatch):
    run_id, _items, _objects = selected_run(normal_work)
    messages = conversation()
    inject_restore(monkeypatch, ScriptedGuard(holds(messages[1], blocked_answer)), messages=messages)
    assert collect({"action": "resume_prompt_run", "run_id": run_id})[-1].get("run_state") == "restore_held"
    # A worker that no resume started holds the run: the request answers that state and nothing else.
    with runtime._PROMPT_REPORT_LOCK:
        runtime._ACTIVE_PROMPT_REPORTS.add(run_id)
    try:
        assert collect({"action": "resume_prompt_run", "run_id": run_id}) == [
            {"event": "resume_state", "state": "original_active", "run_id": run_id}
        ]
    finally:
        with runtime._PROMPT_REPORT_LOCK:
            runtime._ACTIVE_PROMPT_REPORTS.discard(run_id)


def throttled(*_args, **_kwargs):
    raise ClientError(
        {"Error": {"Code": "ThrottlingException"}, "ResponseMetadata": {"HTTPStatusCode": 400}}, "GetItem"
    )


@pytest.mark.parametrize("case", ["prefix_differs", "checkpoint_missing", "no_verdict", "transient"])
def test_a_restore_that_ends_for_another_reason_reads_as_before(normal_work, monkeypatch, case):
    """Control: only a verdict of the input check is reported as a held restore."""
    _work, ports, model = normal_work
    run_id, items, objects = selected_run(normal_work)
    guard = ScriptedGuard()

    def latch(built):
        built.backend._input_hold = True

    selected = inject_restore(monkeypatch, guard, prepare=latch if case == "no_verdict" else None)
    monkeypatch.setattr(runtime, "_restore_pause", lambda _seconds: None)
    if case == "prefix_differs":
        # The stored snapshot no longer equals the prefix that the fine-tuning call was bound to.
        monkeypatch.setattr(checkpoint, "_task_causal_prefix", lambda task: conversation()[:3])
        expected = {"restore_stage": "snapshot_load", "exception_class": "ValueError"}
    elif case == "checkpoint_missing":

        def missing(self, **_):
            raise FileNotFoundError("declared missing checkpoint object")

        monkeypatch.setattr(checkpoint.NormalRunContext, "load_qualified_task", missing)
        expected = {"restore_stage": "checkpoint_load", "exception_class": "FileNotFoundError"}
    elif case == "no_verdict":
        # The hold was set before the restore's first check, so the check ran for no text and left no verdict.
        expected = FAILED
    else:
        monkeypatch.setattr(checkpoint, "prepare_normal_run", throttled)
        expected = {"restore_stage": "run_prepare", "exception_class": "Exception"}
    events = collect({"action": "resume_prompt_run", "run_id": run_id})
    state = "failed_transient" if case == "transient" else "failed"
    assert events == [
        restoring_state(run_id),
        {"event": "resume_state", "state": state, "run_id": run_id, "diagnostic": expected},
    ]
    status_reads(monkeypatch, unavailable(run_id))
    assert collect({"action": "get_prompt_run_status", "run_id": run_id}) == [unavailable(run_id)]
    assert "restore_hold" not in runtime._RESUME_WORKERS.get(run_id, {})
    assert guard.texts == [] and "restore_held" not in shown(events)
    assert ports.items == items and ports.objects == objects and model.calls == 1
    assert len(selected.built) == (1 if case in ("prefix_differs", "no_verdict") else 0)


# ── a whole run: held, read, and continued after the verdict changed ─────────────────────────────────────────


def record_of(state):
    return json.loads(state.ports.items["PROMPT_RUN#" + state.run_id]["record"]["S"])


def stored(state):
    return (
        copy.deepcopy(state.ports.items),
        {key: hashlib.sha256(raw).hexdigest() for key, raw in state.ports.objects.items()},
        len(state.native.calls),
        len(state.native.processing),
        state.model.calls,
    )


def script_guard(monkeypatch, state, script):
    """Give the fixture's guardrail client recorded-shape answers; ``script.held`` names the held message.

    ``script.sends`` counts the sends of the held text of that message. The text is held from its
    ``script.from_send``-th send on, which is the first send unless the test sets another. A test ends the
    hold by setting ``script.held`` to None.
    """
    guard = state.work.guard
    selected = "ordinaryselected"
    script.texts = []
    script.sends = 0
    reached = reaches(script.held)

    def apply_guardrail(**kwargs):
        guard.calls.append(kwargs)
        text = kwargs["content"][0]["text"]["text"]
        script.texts.append(text)
        named = reached(text) and script.held is not None
        if named:
            script.sends += 1
        held = named and script.sends >= getattr(script, "from_send", 1)
        return blocked_answer(text, selected) if held else none_answer(text, selected)

    monkeypatch.setattr(guard, "apply_guardrail", apply_guardrail)


def leaves(value):
    """Every string that stands inside the content of a message."""
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from leaves(item)
    elif type(value) is list:
        for item in value:
            yield from leaves(item)


def test_a_run_held_at_its_restore_reads_its_cause_keeps_everything_and_continues_after_the_verdict_changes(
    monkeypatch, tmp_path
):
    """One real run of the scripted fixture: first worker, replaced worker, held restore, changed verdict."""
    ahead = later_clock(monkeypatch)
    script = SimpleNamespace(held=None)

    async def scenario():
        state = setup(monkeypatch, tmp_path, variant="fit-only")
        try:
            await original_precommit(state, monkeypatch)
            await close_original_worker(state.work, state.reader)
            run_id = state.run_id
            prefix = _record_messages(state.prefix)
            position = next(index for index, message in enumerate(prefix) if message["role"] == "assistant")
            script.held = prefix[position]
            script_guard(monkeypatch, state, script)
            status_request = {"action": "get_prompt_run_status", "run_id": run_id}
            resume_request = {"action": "resume_prompt_run", "run_id": run_id}
            report_request = {"action": "get_prompt_report", "run_id": run_id}

            # The job still runs: the resume request restores the pending run and the input check holds it.
            report = await requested(report_request)
            before = stored(state)
            pending = await requested(resume_request)
            assert pending[:2] == [restoring_state(run_id), failed_state(run_id)] and len(pending) == 3
            message = blocked_message(position + 1, len(prefix), "assistant", kinds(script.held))
            cause = pending[2]["restore_hold"]
            assert pending[2] == held_answer(unavailable(run_id), message, cause)
            assert acting(cause) == [("CONTENT", "SEXUAL", "BLOCKED", "LOW")]
            # The status of the running job keeps its answer and adds the three keys.
            running = (await requested(status_request))[0]
            assert running["status"] == "available" and running["progress"]["pending"] is True
            assert running == held_answer(
                {key: running[key] for key in ("event", "status", "run_id", "found", "progress")},
                message,
                cause,
            )
            assert stored(state) == before and await requested(report_request) == report

            # The job completes. The restore of the completed run is held at the same stored message.
            cloud_data(state, completed=True)
            state.native.states[state.original.id] = "Completed"
            before = stored(state)
            ahead.seconds += runtime._RESUME_FAILED_COOLDOWN_SECONDS + 1
            first = await requested(resume_request)
            message = blocked_message(position + 1, len(prefix) + 1, "assistant", kinds(script.held))
            cause = first[2]["restore_hold"]
            assert first == [
                restoring_state(run_id),
                failed_state(run_id),
                held_answer(unavailable(run_id), message, cause),
            ]
            assert (cause["message_position"], cause["message_count"]) == (position + 1, len(prefix) + 1)
            # The second request inside the pause and the status request name the same cause.
            assert await requested(resume_request) == first[1:]
            assert await requested(status_request) == [first[2]]
            # Nothing was written or discarded: the run record, every stored object, the report, no job, no
            # model call. The run is still in its fine-tuning stage and has no report of a resumed worker.
            assert stored(state) == before
            assert await requested(report_request) == report
            assert record_of(state)["stage"] == "sft" and "resumed_report_checkpoint_ref" not in record_of(
                state
            )
            # No answer holds a string of the held message, the guardrail's output or the checked texts.
            # A stored value is now a checked text of its own. The cause gives the role name of the held
            # message, and a value shorter than eight characters can stand in an answer by chance.
            visible = shown(pending, running, first)
            assert NOTICE not in visible and "ordinaryselected" not in visible
            assert all(text not in visible for text in leaves(script.held["content"]) if len(text) >= 8)
            role = cause["message_role"]
            assert role == "assistant" and role in script.texts
            assert all(text not in visible for text in script.texts if len(text) >= 8 and text != role)

            # The verdict changes. The next restore passes its checks and the run continues to its report.
            script.held = None
            ahead.seconds += runtime._RESUME_FAILED_COOLDOWN_SECONDS + 1
            resumed = await asyncio.wait_for(requested(resume_request), 180)
            assert [(event["event"], event.get("state")) for event in resumed] == [
                ("resume_state", "restoring"),
                ("run_started", None),
                ("run_completed", None),
            ]
            terminal = resumed[-1]
            assert (
                terminal["status"] == "response_received"
                and FIT_FINAL_TEXT in terminal["response"]["message"]
            )
            assert "resumed_report_checkpoint_ref" in record_of(state)
            assert len(state.native.calls) == before[2] and len(state.native.processing) == before[3]
            # The cause is gone: the status answers as for every run in the fine-tuning stage.
            assert await requested(status_request) == [unavailable(run_id)]
            assert await requested(resume_request) == [
                {"event": "resume_state", "state": "completed", "run_id": run_id}
            ]
            assert "restore_held" not in shown(resumed)
        finally:
            await close(state)

    asyncio.run(scenario())


def test_a_restore_held_in_its_second_pass_keeps_the_completion_claim_and_the_next_restore_reuses_it(
    monkeypatch, tmp_path
):
    """The restore of a completed run checks the stored conversation twice, and claims the completion between.

    A message that the check admits in the first pass and holds in the second holds the restore after the
    claim was written. The claim is the one stored change. Nothing kept is discarded, and the next restore
    takes the same claim.
    """
    ahead = later_clock(monkeypatch)
    script = SimpleNamespace(held=None, from_send=2)

    async def scenario():
        state = setup(monkeypatch, tmp_path, variant="fit-only")
        try:
            await original_precommit(state, monkeypatch)
            await close_original_worker(state.work, state.reader)
            run_id = state.run_id
            prefix = _record_messages(state.prefix)
            position = next(index for index, message in enumerate(prefix) if message["role"] == "assistant")
            script.held = prefix[position]
            script_guard(monkeypatch, state, script)
            status_request = {"action": "get_prompt_run_status", "run_id": run_id}
            resume_request = {"action": "resume_prompt_run", "run_id": run_id}
            report_request = {"action": "get_prompt_report", "run_id": run_id}
            cloud_data(state, completed=True)
            state.native.states[state.original.id] = "Completed"
            report = await requested(report_request)
            before, record = stored(state), record_of(state)
            assert "continuation" not in record

            first = await requested(resume_request)
            count = len(prefix) + 1
            message = blocked_message(position + 1, count, "assistant", kinds(script.held))
            assert len(first) == 3, "the resume request names no cause"
            cause = first[2]["restore_hold"]
            assert first == [
                restoring_state(run_id),
                failed_state(run_id),
                held_answer(unavailable(run_id), message, cause),
            ]
            assert (cause["message_position"], cause["message_count"]) == (position + 1, count)
            # The held check was the second send of that text: after the completion result and the whole
            # first pass, the second pass went up to the held text of the held message.
            result = json.loads(script.texts[0])
            completed = [*map(as_stored, prefix), {"role": "user", "content": [{"toolResult": result}]}]
            whole_pass = [text for message in completed for text in history_texts(message)]
            upto = [text for message in completed[:position] for text in history_texts(message)]
            assert script.sends == 2
            assert script.texts == [script.texts[0], *whole_pass, *upto, *held_place(script.held)]
            # The completion claim was written between the two passes, and the hold left it in the run record.
            # It is the one stored change: it has started nothing, no other key of the record changed, and
            # every other row, every stored object, the report, the jobs and the model calls are as before.
            after, claimed = stored(state), record_of(state)
            claim = claimed["continuation"]
            assert claimed == {**record, "continuation": claim}
            assert (claim["status"], claim["invocation_state"], claim["invocation_token"]) == (
                "consumed-ready",
                "ready",
                None,
            )
            row = "PROMPT_RUN#" + run_id
            assert {key: item for key, item in after[0].items() if key != row} == {
                key: item for key, item in before[0].items() if key != row
            }
            assert after[1:] == before[1:]
            assert await requested(report_request) == report
            assert claimed["stage"] == "sft" and "resumed_report_checkpoint_ref" not in claimed
            # A request inside the pause and a status request name the same cause and write nothing.
            assert await requested(resume_request) == first[1:]
            assert await requested(status_request) == [first[2]]
            assert stored(state) == after

            # The verdict changes. The next restore takes the claim that the held one left, passes both
            # passes and continues the run to its report.
            script.held = None
            ahead.seconds += runtime._RESUME_FAILED_COOLDOWN_SECONDS + 1
            resumed = await asyncio.wait_for(requested(resume_request), 180)
            assert [(event["event"], event.get("state")) for event in resumed] == [
                ("resume_state", "restoring"),
                ("run_started", None),
                ("run_completed", None),
            ]
            terminal = resumed[-1]
            assert (
                terminal["status"] == "response_received"
                and FIT_FINAL_TEXT in terminal["response"]["message"]
            )
            ended = record_of(state)
            assert ended["continuation"]["token"] == claim["token"]
            assert "resumed_report_checkpoint_ref" in ended
            assert len(state.native.calls) == before[2] and len(state.native.processing) == before[3]
            assert await requested(status_request) == [unavailable(run_id)]
            assert "restore_held" not in shown(resumed)
        finally:
            await close(state)

    asyncio.run(scenario())
