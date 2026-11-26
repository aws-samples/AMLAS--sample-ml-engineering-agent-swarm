"""History admission sends each stored value in the form of its first check or on its own.

The worker of a prompt run can be replaced while its fine-tuning job runs. Before the stored conversation is
loaded again, every stored message goes through the deployed guardrail's input check, and so does the earlier
conversation of a chat turn. A run whose worker is not replaced checks the owner prompt, each tool result and
each model text on their own. The history check uses the same units:

* content in the form of its first check: a text block as its text, a tool result as the compact JSON of its
  content list, byte-equal to the text the capture path sends for a fresh result in the same key order;
* machine-written values each on their own: a tool result's call id and status, a tool call's id, name and
  every key and scalar leaf of its input, a reasoning text, each signature, a redacted value, and every scalar
  leaf outside the content list;
* a block of another shape whole as compact JSON, and a message whole as compact JSON when it cannot be
  walked, when it holds a field other than its role, its content and its metadata, when it yields no text, or
  when that one text is over the size limit or cannot be encoded.

Every string, number and boolean of the stored message is sent in exactly one of these texts. Each text is
admitted as unchanged history: a blocked, masked or unassessed text holds the restore before the conversation
is loaded. Empty strings and nulls are not sent.

Scripted guardrail answers in the shape of the recorded native answer; no native API, model or network. The
fixtures establish source behaviour, not native guardrail classification or a deployed run.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import random
import threading
from types import SimpleNamespace

import pytest
import test_input_admission_wiring as wiring
from strands import tool
from strands.handlers.callback_handler import null_callback_handler
from test_input_admission_supplementary_characters import (
    CYCLE,
    private_checker,
    recorded_none_response,
)
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.control_plane import normal_run_checkpoint as checkpoint
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.tui.chat import engine as chat_engine
from amlas.tui.chat import research_runtime_profile
from amlas.tui.chat import tool_binding_backend as binding
from amlas.tui.chat.engine import ChatEngine
from amlas.tui.chat.input_admission import MAX_INPUT_BYTES
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend, _InputAdmissionHeldError, _record_messages

# Values built here, so that no fixture holds a real signature, identifier or page text.
SIGNATURE_ONE = "QUJD" * 240
SIGNATURE_TWO = "REVG" * 300
RESULT_ONE = "-".join(("first", "stored", "result"))
RESULT_TWO = "-".join(("second", "stored", "result"))
NOTICE = " ".join(("The", "fixture", "guardrail", "withheld", "this", "input."))
SFT_USE = "declared-sft-use"
MODEL_REF = "model_acq_" + "2" * 32
PREPARATION_REF = "prep_" + "3" * 32
HOLD_LINE = "AMLAS_INPUT_HOLD phase=input_admission reason=admission_refused"
NOT_SERIALIZABLE_LINE = "AMLAS_INPUT_HOLD phase=history_qualification reason=history_not_serializable"


def compact(value):
    """The compact JSON text of the specification, written out here and not taken from the source."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def texts_of(message):
    return binding._history_admission_texts(message)


def plan_of(message):
    return binding._history_admission_plan(message)


# ── the helper: one stored message, an ordered list of texts ─────────────────────────────────────────────────


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_a_user_text_and_an_assistant_text_are_sent_as_themselves(role):
    text = ' Leading space, an accent é, a quote ", a backslash \\ and\na second line.'
    message = {"role": role, "content": [{"text": text}]}
    # The characters of the text, never its JSON spelling.
    assert texts_of(message) == [role, text]
    assert compact(text) not in texts_of(message)


def test_a_tool_result_is_sent_as_the_compact_json_of_its_content_list():
    content = [{"text": 'café "quoted"\n'}, {"json": {"b": [None, "", 0, False], "a": {"n": 1.5}}}]
    message = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "declared-read", "status": "success", "content": content}}],
    }
    expected = '[{"text":"café \\"quoted\\"\\n"},{"json":{"b":[null,"",0,false],"a":{"n":1.5}}}]'
    assert binding._tool_result_admission_text(content) == expected == compact(content)
    # One text holds the whole list; nothing inside it is sent again, and nothing inside it is counted.
    assert plan_of(message) == (["user", "declared-read", "success", expected], 0, 0)
    assert plan_of(
        {"role": "user", "content": [{"toolResult": {**message["content"][0]["toolResult"], "content": []}}]}
    ) == (
        ["user", "declared-read", "success", "[]"],
        0,
        0,
    )


def test_machine_written_values_are_each_sent_alone_in_stored_order():
    message = {
        "role": "assistant",
        "content": [
            {"reasoningContent": {"reasoningText": {"text": "", "signature": SIGNATURE_ONE}}},
            {
                "reasoningContent": {
                    "reasoningText": {"text": "A stored thought.", "signature": SIGNATURE_TWO}
                }
            },
            {"reasoningContent": {"redactedContent": "UkVEQUNURUQ="}},
            {
                "toolUse": {
                    "toolUseId": "declared-call",
                    "name": "get_declared_profile",
                    "input": {
                        "profile_id": "declared-profile",
                        "plan_only": False,
                        "limits": {"rows": 0, "share": 1.5, "": "under an empty key"},
                        "tags": ["declared", "", None],
                        "nothing": None,
                    },
                }
            },
        ],
        "metadata": {"usage": {"inputTokens": 1200, "cacheDetails": [{"ttl": "5m", "inputTokens": 0}]}},
    }
    before = copy.deepcopy(message)
    texts, empty, null = plan_of(message)
    assert texts == [
        "assistant",
        SIGNATURE_ONE,
        "A stored thought.",
        SIGNATURE_TWO,
        "UkVEQUNURUQ=",
        "declared-call",
        "get_declared_profile",
        # Every key of a tool call's input is the model's own text and is sent like a value.
        "profile_id",
        "declared-profile",
        "plan_only",
        "false",
        "limits",
        "rows",
        "0",
        "share",
        "1.5",
        "under an empty key",
        "tags",
        "declared",
        "nothing",
        # The names of usage fields are written by the agent SDK; only their values are sent.
        "1200",
        "5m",
        "0",
    ]
    # Not sent and counted: the empty reasoning text, the empty key, the empty tag; two nulls.
    assert (empty, null) == (3, 2)
    assert message == before


def test_the_order_of_the_texts_follows_the_stored_order_of_the_fields():
    result = {"toolUseId": "declared-read", "status": "success", "content": [{"text": "stored"}]}
    use = {"toolUseId": "declared-call", "name": "declared_tool", "input": {"b": "second", "a": "first"}}
    first = {"role": "user", "content": [{"toolResult": result}, {"toolUse": use}], "metadata": {"n": 7}}
    assert texts_of(first) == [
        "user",
        "declared-read",
        "success",
        '[{"text":"stored"}]',
        "declared-call",
        "declared_tool",
        "b",
        "second",
        "a",
        "first",
        "7",
    ]
    # The same message as a checkpoint stores it: every object with its keys sorted.
    stored = json.loads(json.dumps(first, sort_keys=True))
    assert texts_of(stored) == [
        '[{"text":"stored"}]',
        "success",
        "declared-read",
        "a",
        "first",
        "b",
        "second",
        "declared_tool",
        "declared-call",
        "7",
        "user",
    ]
    assert texts_of(stored) == texts_of(copy.deepcopy(stored)), "the order of the sends is stable"


def test_equal_texts_are_sent_again_each_time():
    message = {"role": "user", "content": [{"text": "same"}, {"text": "same"}, {"text": "user"}]}
    assert texts_of(message) == ["user", "same", "same", "user"]


# A name inside a redacted value that is an object or a list would stand in no single text.
REDACTED_NAME = "declared_part"
UNKNOWN_BLOCKS = {
    "another_kind": {"image": {"format": "png", "source": {"location": "declared-location"}}},
    "a_marker_block": {"cachePoint": {"type": "default"}},
    "text_with_another_field": {"text": "seen", "extra": "must not disappear"},
    "text_that_is_no_string": {"text": None},
    "two_kinds_in_one_block": {
        "text": "seen",
        "toolUse": {"toolUseId": "declared-call", "name": "declared_tool", "input": {}},
    },
    "tool_call_with_another_field": {
        "toolUse": {"toolUseId": "declared-call", "name": "declared_tool", "input": {}, "type": "declared"}
    },
    "tool_call_without_input": {"toolUse": {"toolUseId": "declared-call", "name": "declared_tool"}},
    "tool_call_with_a_value_json_does_not_load": {
        "toolUse": {
            "toolUseId": "declared-call",
            "name": "declared_tool",
            "input": {"a": "first", "b": ("x", "y")},
        }
    },
    "tool_call_with_a_key_that_is_no_string": {
        "toolUse": {
            "toolUseId": "declared-call",
            "name": "declared_tool",
            "input": {"a": "first", 1: "second"},
        }
    },
    "tool_result_with_another_field": {
        "toolResult": {"toolUseId": "declared-read", "status": "success", "content": [], "extra": "kept"}
    },
    "tool_result_whose_content_is_no_list": {
        "toolResult": {"toolUseId": "declared-read", "status": "success", "content": "stored"}
    },
    "tool_result_whose_status_is_no_string": {
        "toolResult": {"toolUseId": "declared-read", "status": 200, "content": []}
    },
    "reasoning_with_another_field": {
        "reasoningContent": {"reasoningText": {"text": "", "signature": SIGNATURE_ONE}, "extra": "kept"}
    },
    "reasoning_text_with_another_field": {
        "reasoningContent": {"reasoningText": {"text": "", "signature": SIGNATURE_ONE, "extra": "kept"}}
    },
    "reasoning_signature_that_is_no_string": {
        "reasoningContent": {"reasoningText": {"text": "", "signature": ["kept"]}}
    },
    "empty_reasoning": {"reasoningContent": {}},
    "redacted_value_that_is_an_object": {"reasoningContent": {"redactedContent": {REDACTED_NAME: 7}}},
    "redacted_value_that_is_a_list_of_objects": {
        "reasoningContent": {"redactedContent": [{REDACTED_NAME: "kept"}]}
    },
    "redacted_value_that_is_a_list_of_values": {"reasoningContent": {"redactedContent": ["kept", 7]}},
    "redacted_value_that_is_an_empty_object": {"reasoningContent": {"redactedContent": {}}},
    "redacted_value_that_is_an_empty_list": {"reasoningContent": {"redactedContent": []}},
    "redacted_value_that_is_an_object_beside_a_reasoning_text": {
        "reasoningContent": {
            "reasoningText": {"text": "A stored thought.", "signature": SIGNATURE_ONE},
            "redactedContent": {"outer": [{REDACTED_NAME: None}]},
        }
    },
    "a_bare_string": "seen",
    "a_number": 17,
    "a_list": ["seen"],
    "an_empty_block": {},
}


@pytest.mark.parametrize("name", sorted(UNKNOWN_BLOCKS))
def test_a_block_of_another_shape_is_sent_whole(name):
    block = UNKNOWN_BLOCKS[name]
    message = {"role": "user", "content": [{"text": "before"}, block, {"text": "after"}]}
    before = copy.deepcopy(message)
    # One compact JSON text for the block: nothing of it is sent twice and nothing of it is left out.
    assert plan_of(message) == (["user", "before", compact(block), "after"], 0, 0)
    assert message == before


@pytest.mark.parametrize(
    "value",
    [{REDACTED_NAME: 7}, [{REDACTED_NAME: "kept"}], {"outer": [{REDACTED_NAME: None}]}, {}, []],
    ids=["an_object", "a_list_of_objects", "a_nested_object", "an_empty_object", "an_empty_list"],
)
def test_a_redacted_value_that_is_an_object_or_a_list_sends_its_block_whole(value):
    """A redacted value is sent alone as a single value; the names inside an object or a list are not."""
    reasoning = {"text": "A stored thought.", "signature": SIGNATURE_ONE}
    single = {"reasoningContent": {"reasoningText": reasoning, "redactedContent": "UkVEQUNURUQ="}}
    # With a single redacted value the block is sent value by value.
    assert plan_of({"role": "assistant", "content": [{"text": "before"}, single]}) == (
        ["assistant", "before", "A stored thought.", SIGNATURE_ONE, "UkVEQUNURUQ="],
        0,
        0,
    )
    block = {"reasoningContent": {"reasoningText": reasoning, "redactedContent": value}}
    message = {"role": "assistant", "content": [{"text": "before"}, block, {"text": "after"}]}
    before = copy.deepcopy(message)
    # One compact JSON text for the block, and no value of the block beside it.
    assert plan_of(message) == (["assistant", "before", compact(block), "after"], 0, 0)
    assert message == before


UNWALKED_MESSAGES = {
    "no_object": ["seen"],
    "a_string": "seen",
    "without_a_role": {"content": [{"text": "seen"}]},
    "role_that_is_no_string": {"role": 5, "content": [{"text": "seen"}]},
    "without_content": {"role": "user"},
    "content_that_is_no_list": {"role": "user", "content": "seen"},
    "a_key_that_is_no_string": {"role": "user", "content": [{"text": "seen"}], 1: "kept"},
    "a_value_json_does_not_load_after_the_content": {
        "role": "assistant",
        "content": [{"text": "seen"}],
        "metadata": {"pair": ("later", "values")},
    },
    "a_nested_key_that_is_no_string": {"role": "assistant", "content": [], "metadata": {1: "kept"}},
    "nothing_to_send": {"role": "", "content": []},
    "only_empty_values": {"role": "", "content": [{"text": ""}], "metadata": {"note": None, "names": []}},
}


@pytest.mark.parametrize("name", sorted(UNWALKED_MESSAGES))
def test_a_message_that_cannot_be_walked_or_yields_no_text_is_sent_whole(name):
    message = UNWALKED_MESSAGES[name]
    before = copy.deepcopy(message)
    # The one text of the whole message, and no text of a part beside it.
    assert plan_of(message) == ([compact(message)], 0, 0)
    assert message == before


# The form reads the role, the content list and the agent SDK's metadata of a stored message.
UNKNOWN_FIELD = "declared_note"
UNKNOWN_FIELDS = {
    "a_value": {UNKNOWN_FIELD: "kept"},
    "an_object": {UNKNOWN_FIELD: {"inner": "kept", "count": 7}},
    "nothing": {UNKNOWN_FIELD: None},
    "an_empty_object": {UNKNOWN_FIELD: {}},
    "the_tracking_id_of_a_live_message": {"tracking_id": "declared-tracking-id"},
    "a_known_name_in_other_letters": {"Metadata": {"count": 7}},
    "a_known_name_with_a_space_after_it": {"metadata ": {"count": 7}},
    "two_fields": {UNKNOWN_FIELD: "kept", "declared_other": 7},
}


@pytest.mark.parametrize("place", ["after_the_known_fields", "before_the_known_fields"])
@pytest.mark.parametrize("name", sorted(UNKNOWN_FIELDS))
def test_a_message_with_a_field_that_the_form_does_not_know_is_sent_whole(name, place):
    """A name outside the content is structure and is sent in no single text, so an unknown one goes whole."""
    known = {"role": "assistant", "content": [{"text": "seen"}], "metadata": {"usage": {"inputTokens": 3}}}
    # With only the fields that the form reads, the message is sent text by text.
    assert plan_of(known) == (["assistant", "seen", "3"], 0, 0)
    extra = UNKNOWN_FIELDS[name]
    message = {**known, **extra} if place == "after_the_known_fields" else {**extra, **known}
    before = copy.deepcopy(message)
    # The one text of the whole message, with the name of the field in it, and no text of a part beside it.
    assert plan_of(message) == ([compact(message)], 0, 0)
    assert all(json.dumps(field) in compact(message) for field in extra)
    assert message == before


def test_a_deeply_nested_value_is_sent_whole_or_raises_as_the_whole_message_does():
    nested = "seen"
    for _ in range(5000):
        nested = [nested]
    message = {"role": "assistant", "content": [{"text": "kept"}], "metadata": {"deep": nested}}
    try:
        expected = [compact(message)]
    except RecursionError:
        expected = None
    if expected is None:
        with pytest.raises(RecursionError):
            texts_of(message)
    else:
        assert texts_of(message) == expected


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "i", "status": "s", "content": [{"x": b"y"}]}}],
            },
            TypeError,
        ),
        ({"role": "user", "content": [{"image": {"source": {"bytes": b"y"}}}]}, TypeError),
        ({"role": "assistant", "content": [{"reasoningContent": {"redactedContent": b"y"}}]}, TypeError),
        ({"role": "assistant", "content": [], "metadata": {"value": b"y"}}, TypeError),
        ({"role": "assistant", "content": [], "metadata": {"value": float("nan")}}, ValueError),
        (
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "i", "name": "n", "input": float("inf")}}],
            },
            ValueError,
        ),
        (
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "i", "status": "s", "content": [float("nan")]}}],
            },
            ValueError,
        ),
    ],
    ids=[
        "bytes_in_a_tool_result",
        "bytes_in_a_block_of_another_kind",
        "bytes_as_a_redacted_value",
        "bytes_outside_the_content",
        "not_a_number_outside_the_content",
        "infinity_in_a_tool_call",
        "not_a_number_in_a_tool_result",
    ],
)
def test_a_value_that_json_cannot_carry_raises_as_the_whole_message_did(message, error):
    with pytest.raises(error):
        compact(message)
    with pytest.raises(error):
        texts_of(message)


# ── a message that the checker refuses as one text for its size or its encoding ─────────────────────────────


def two_results(first, second):
    """A stored user message with two tool results, in the key order that a checkpoint keeps."""
    return {
        "content": [
            {
                "toolResult": {
                    "content": [{"text": first}],
                    "status": "success",
                    "toolUseId": "declared-read-one",
                }
            },
            {
                "toolResult": {
                    "content": [{"text": second}],
                    "status": "success",
                    "toolUseId": "declared-read-two",
                }
            },
        ],
        "role": "user",
    }


def sized(size):
    """Two tool results that each fit the limit alone, in a message of exactly ``size`` bytes of compact JSON."""
    second = "c" * 30000
    message = two_results("b" * (size - len(compact(two_results("", second)).encode())), second)
    assert len(compact(message).encode()) == size
    return message


OVER_THE_LIMIT = {
    "two_tool_results_that_each_fit": lambda: sized(MAX_INPUT_BYTES + 1),
    "two_larger_tool_results_that_each_fit": lambda: two_results("b" * 40000, "c" * 40000),
    "a_text_of_exactly_the_limit": lambda: {"role": "user", "content": [{"text": "a" * MAX_INPUT_BYTES}]},
    "many_empty_text_blocks": lambda: {"role": "user", "content": [{"text": ""} for _ in range(6000)]},
    "many_short_values_in_one_tool_call": lambda: {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "declared-call",
                    "name": "declared_tool",
                    "input": {f"k{index}": "v" for index in range(9000)},
                }
            }
        ],
    },
    "more_bytes_than_characters": lambda: two_results("\u00e9" * 17000, "\u00fc" * 17000),
}


@pytest.mark.parametrize("name", sorted(OVER_THE_LIMIT))
def test_a_message_over_the_size_limit_as_one_text_is_sent_whole(name):
    message = OVER_THE_LIMIT[name]()
    before = copy.deepcopy(message)
    whole = compact(message)
    assert len(whole.encode()) > MAX_INPUT_BYTES
    # One text, the whole message: the checker measures that text and refuses it, as it always did.
    assert plan_of(message) == ([whole], 0, 0)
    assert message == before


def test_the_size_of_a_message_is_measured_in_the_bytes_of_its_one_text():
    message = OVER_THE_LIMIT["more_bytes_than_characters"]()
    whole = compact(message)
    assert len(whole) <= MAX_INPUT_BYTES < len(whole.encode())
    assert texts_of(message) == [whole]
    assert binding._HISTORY_MESSAGE_MAX_BYTES == MAX_INPUT_BYTES


def test_a_message_of_exactly_the_size_limit_is_sent_text_by_text():
    message = sized(MAX_INPUT_BYTES)
    first, second = (compact(block["toolResult"]["content"]) for block in message["content"])
    assert plan_of(message) == (
        [first, "success", "declared-read-one", second, "success", "declared-read-two", "user"],
        0,
        0,
    )
    over = sized(MAX_INPUT_BYTES + 1)
    assert plan_of(over) == ([compact(over)], 0, 0)


LONE_SURROGATE = "\ud800"
UNENCODABLE = {
    "a_field_name_outside_the_content": {
        "role": "assistant",
        "content": [{"text": "seen"}],
        "metadata": {"usage": {"declared" + LONE_SURROGATE + "name": 5}},
    },
    "a_top_level_field_name": {
        "role": "assistant",
        "content": [{"text": "seen"}],
        "note" + LONE_SURROGATE: "kept",
    },
    "a_value_outside_the_content": {
        "role": "assistant",
        "content": [{"text": "seen"}],
        "metadata": {"note": "kept " + LONE_SURROGATE},
    },
    "a_text": {"role": "user", "content": [{"text": "before"}, {"text": "seen " + LONE_SURROGATE}]},
    "a_tool_result": {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "declared-read",
                    "status": "success",
                    "content": [{"text": "stored " + LONE_SURROGATE}],
                }
            }
        ],
    },
    "a_tool_call_key": {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "declared-call",
                    "name": "declared_tool",
                    "input": {"key" + LONE_SURROGATE: 1},
                }
            }
        ],
    },
}


@pytest.mark.parametrize("name", sorted(UNENCODABLE))
def test_a_message_that_strict_utf8_cannot_encode_is_sent_whole(name):
    message = UNENCODABLE[name]
    before = copy.deepcopy(message)
    whole = compact(message)
    with pytest.raises(UnicodeEncodeError):
        whole.encode("utf-8")
    # One text, the whole message: the checker refuses that text without a native call, as it always did.
    assert plan_of(message) == ([whole], 0, 0)
    assert message == before


# ── keys that are no plain strings, and a tool result that is sent whole ─────────────────────────────────────


class EqualToText:
    """Not a string, but equal to the block kind ``text`` and with its hash."""

    def __eq__(self, other):
        return other == "text"

    def __hash__(self):
        return hash("text")


def test_a_block_kind_that_is_no_string_is_not_walked_as_a_text_block():
    message = {"role": "user", "content": [{EqualToText(): "seen"}]}
    with pytest.raises(TypeError):
        compact(message)
    with pytest.raises(TypeError):
        texts_of(message)


class RaisingKey(str):
    """A string subclass whose comparison raises; JSON carries it as its string."""

    __hash__ = str.__hash__

    def __eq__(self, other):
        raise RuntimeError("a key was compared before its type was checked")


def raising_key_blocks():
    return {
        "the_block_kind": {RaisingKey("text"): "seen"},
        "a_field_of_a_tool_call": {
            "toolUse": {RaisingKey("toolUseId"): "declared-call", "name": "declared_tool", "input": {}}
        },
        "a_field_of_a_tool_result": {
            "toolResult": {"toolUseId": "declared-read", RaisingKey("status"): "success", "content": []}
        },
        "a_field_of_a_reasoning_block": {
            "reasoningContent": {RaisingKey("reasoningText"): {"text": "", "signature": SIGNATURE_ONE}}
        },
        "a_field_of_a_reasoning_text": {
            "reasoningContent": {"reasoningText": {"text": "", RaisingKey("signature"): SIGNATURE_ONE}}
        },
        "a_key_of_a_tool_call_input": {
            "toolUse": {"toolUseId": "declared-call", "name": "declared_tool", "input": {RaisingKey("a"): 1}}
        },
    }


@pytest.mark.parametrize("name", sorted(raising_key_blocks()))
def test_a_key_that_is_no_plain_string_sends_its_block_whole_without_a_comparison(name):
    block = raising_key_blocks()[name]
    message = {"role": "assistant", "content": [{"text": "before"}, block, {"text": "after"}]}
    assert plan_of(message) == (["assistant", "before", compact(block), "after"], 0, 0)


def test_a_message_field_name_that_is_no_plain_string_sends_the_message_whole_without_a_comparison():
    message = {RaisingKey("role"): "user", "content": [{"text": "seen"}]}
    assert plan_of(message) == (['{"role":"user","content":[{"text":"seen"}]}'], 0, 0)


class RaisingType(type):
    """A type whose comparison raises; the helper tells a type only by identity."""

    def __eq__(cls, other):
        raise RuntimeError("a type was compared")

    __hash__ = type.__hash__


class Opaque(metaclass=RaisingType):
    """A value that is no plain JSON data and whose type cannot be compared."""


def test_a_redacted_value_of_a_type_whose_comparison_raises_is_not_compared():
    message = {"role": "assistant", "content": [{"reasoningContent": {"redactedContent": Opaque()}}]}
    # JSON cannot carry the value, so the helper raises as the whole message does, without a comparison.
    with pytest.raises(TypeError):
        compact(message)
    with pytest.raises(TypeError):
        texts_of(message)


class ReadOnce(dict):
    """A mapping whose items can be read once; a second read finds none."""

    def items(self):
        if getattr(self, "read", False):
            return []
        self.read = True
        return list(super().items())


def test_a_tool_result_that_is_sent_whole_serializes_its_content_once():
    def message():
        result = {"content": [ReadOnce({"leaf": "kept-word"})], "status": " ", "toolUseId": "declared-read"}
        return {"role": "user", "content": [{"toolResult": result}]}

    block = '{"toolResult":{"content":[{"leaf":"kept-word"}],"status":" ","toolUseId":"declared-read"}}'
    assert compact(message()["content"][0]) == block
    # The blank status sends the block whole; that one text holds what the first read of the content gave.
    assert texts_of(message()) == ["user", block]


def test_empty_strings_and_nulls_are_not_sent_and_are_counted():
    message = {
        "role": "assistant",
        "content": [
            {"text": ""},
            {"reasoningContent": {"reasoningText": {"text": "", "signature": ""}}},
            {
                "toolUse": {
                    "toolUseId": "declared-call",
                    "name": "",
                    "input": {"a": "", "b": None, "": [None, ""]},
                }
            },
            {"toolResult": {"toolUseId": "", "status": "", "content": [{"text": ""}, {"json": {"n": None}}]}},
        ],
        "metadata": {"note": None, "label": ""},
    }
    texts, empty, null = plan_of(message)
    assert "" not in texts and None not in texts
    # The empty strings and nulls inside the tool result stay inside its one text.
    assert texts == ["assistant", "declared-call", "a", "b", '[{"text":""},{"json":{"n":null}}]']
    assert (empty, null) == (10, 3)
    # Zero and false are values and are sent.
    kept = {"role": "user", "content": [{"text": "kept"}], "metadata": {"zero": 0, "no": False, "real": 0.0}}
    assert plan_of(kept) == (["user", "kept", "0", "false", "0.0"], 0, 0)


BLANK_IN_A_BLOCK = {
    "text": {"text": " \n"},
    "tool_call_input_value": {
        "toolUse": {
            "toolUseId": "declared-call",
            "name": "declared_tool",
            "input": {"a": "first", "b": ["\t"]},
        }
    },
    "tool_call_input_key": {
        "toolUse": {
            "toolUseId": "declared-call",
            "name": "declared_tool",
            "input": {"a": "first", " ": "second"},
        }
    },
    "tool_name": {"toolUse": {"toolUseId": "declared-call", "name": " ", "input": {"a": "first"}}},
    "tool_call_id": {"toolUse": {"toolUseId": "\n", "name": "declared_tool", "input": {"a": "first"}}},
    "tool_result_id": {"toolResult": {"toolUseId": " ", "status": "success", "content": [{"text": "kept"}]}},
    "tool_result_status": {"toolResult": {"toolUseId": "declared-read", "status": " ", "content": []}},
    "reasoning_text": {"reasoningContent": {"reasoningText": {"text": "\n\n", "signature": SIGNATURE_ONE}}},
    "signature": {"reasoningContent": {"reasoningText": {"text": "A stored thought.", "signature": " "}}},
    "redacted_value": {"reasoningContent": {"redactedContent": "\u00a0"}},
}


@pytest.mark.parametrize("name", sorted(BLANK_IN_A_BLOCK))
def test_a_value_of_only_white_space_sends_its_block_whole(name):
    """The service gives no verdict on white space alone, so such a value is checked inside its block."""
    block = BLANK_IN_A_BLOCK[name]
    message = {"role": "assistant", "content": [{"text": "before"}, block, {"text": "after"}]}
    before = copy.deepcopy(message)
    texts, empty, null = plan_of(message)
    assert texts == ["assistant", "before", compact(block), "after"] and (empty, null) == (0, 0)
    assert all(text.strip() for text in texts), "a text of only white space was sent"
    assert message == before


@pytest.mark.parametrize(
    "message",
    [
        {"role": " ", "content": [{"text": "seen"}]},
        {
            "role": "assistant",
            "content": [{"text": "seen"}],
            "metadata": {"note": "\n", "usage": {"inputTokens": 3}},
        },
        {"role": "assistant", "content": [{"text": "seen"}], "metadata": {"notes": ["kept", " \t"]}},
    ],
    ids=["role", "value_outside_the_content", "nested_value_outside_the_content"],
)
def test_a_value_of_only_white_space_outside_the_content_sends_the_message_whole(message):
    assert plan_of(message) == ([compact(message)], 0, 0)


def test_white_space_inside_a_tool_result_stays_inside_its_one_text():
    content = [{"text": " "}, {"json": {"note": "\n", " ": [" "]}}]
    message = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "declared-read", "status": "success", "content": content}}],
    }
    assert texts_of(message) == ["user", "declared-read", "success", compact(content)]


class Maker:
    """Builds stored messages together with the texts the specification derives from them.

    Every word and number it places is unique, so a search of the derived texts counts how often each one
    was sent. The expected texts come from how each part is built, never from the source under test.
    """

    def __init__(self, seed):
        self.rng = random.Random(seed)  # nosec B311 - seeded test data, not a secret
        self.count = 0
        self.cores, self.unsent = [], []
        self.empty = self.null = self.blank = 0

    def word(self, sent=True):
        self.count += 1
        number, letters = self.count, ""
        for _ in range(5):
            letters += chr(ord("a") + number % 26)
            number //= 26
        word = "q" + letters + "z"
        (self.cores if sent else self.unsent).append(word)
        return word

    def number(self):
        self.count += 1
        self.cores.append(str(20000000 + self.count))
        return 20000000 + self.count

    def string(self, counted=True):
        """One string and the text it is sent as on its own (None when it is not sent)."""
        pick = self.rng.randrange(8)
        if pick == 0:
            self.empty += counted
            return "", None
        if pick == 1:
            # Only white space: not sent on its own, and it sends its block or message whole.
            self.blank += counted
            return " \t", None
        if pick == 2:
            value = ' "' + self.word() + "\\\né "
            return value, value
        value = self.word()
        return value, value

    def scalar(self, counted=True):
        """One scalar value and the text it is sent as on its own (None when it is not sent)."""
        pick = self.rng.randrange(8)
        if pick == 0:
            self.null += counted
            return None, None
        if pick == 1:
            value = self.rng.random() < 0.5
            return value, "true" if value else "false"
        if pick == 2:
            value = self.number()
            return value, str(value)
        if pick == 3:
            value = self.number() + 0.5
            return value, repr(value)
        return self.string(counted)

    def value(self, depth, keys, single=True):
        """Plain JSON DATA and what a walk of single values sends for it, in order.

        With ``single`` false the DATA stands inside one whole text: its words are counted there, its
        empty strings and nulls are not counted, and the returned texts are not used.
        """
        texts = []
        pick = self.rng.randrange(6) if depth < 3 else 5
        if pick == 0:
            value = {}
            names = [self.word(sent=keys or not single) for _ in range(self.rng.randrange(4))]
            if single and keys and self.rng.random() < 0.15:
                names.append("")
                self.empty += 1
            for name in names:
                item, inner = self.value(depth + 1, keys, single)
                value[name] = item
                texts += ([name] if keys and name else []) + inner
            return value, texts
        if pick == 1:
            value = []
            for _ in range(self.rng.randrange(4)):
                item, inner = self.value(depth + 1, keys, single)
                value.append(item)
                texts += inner
            return value, texts
        value, text = self.scalar(counted=single)
        return value, ([] if text is None else [text])

    def ordered(self, pairs):
        """An object with its fields in a random order and their texts in that order."""
        pairs = list(pairs)
        self.rng.shuffle(pairs)
        return {name: item for name, item, _ in pairs}, [text for _, _, texts in pairs for text in texts]

    def text_block(self):
        value, text = self.string()
        return {"text": value}, ([] if text is None else [text])

    def tool_call(self):
        ident, name = self.word(), self.word()
        argument, texts = self.value(0, keys=True)
        body, sent = self.ordered(
            [("toolUseId", ident, [ident]), ("name", name, [name]), ("input", argument, texts)]
        )
        return {"toolUse": body}, sent

    def tool_result(self):
        ident, status = self.word(), self.word()
        content = []
        for _ in range(self.rng.randrange(3)):
            if self.rng.random() < 0.5:
                content.append({"json": self.value(1, keys=False, single=False)[0]})
            else:
                content.append({"text": self.word()})
        body, sent = self.ordered(
            [
                ("toolUseId", ident, [ident]),
                ("status", status, [status]),
                ("content", content, [compact(content)]),
            ]
        )
        return {"toolResult": body}, sent

    def reasoning(self):
        pairs = []
        if self.rng.random() < 0.8:
            inner = []
            if self.rng.random() < 0.8:
                if self.rng.random() < 0.6:
                    self.empty += 1
                    inner.append(("text", "", []))
                else:
                    value = self.word()
                    inner.append(("text", value, [value]))
            if self.rng.random() < 0.8:
                value = self.word() + "QUJD" * 40
                inner.append(("signature", value, [value]))
            pairs.append(("reasoningText", *self.ordered(inner)))
        if not pairs or self.rng.random() < 0.3:
            value = self.word()
            pairs.append(("redactedContent", value, [value]))
        body, sent = self.ordered(pairs)
        return {"reasoningContent": body}, sent

    def other_block(self):
        """A block of no known closed shape: sent whole, so its words are counted inside that one text."""
        pick = self.rng.randrange(8)
        item = self.value(1, keys=False, single=False)[0] if pick < 6 else None
        if pick == 0:
            block = {"image": {"format": self.word(), "source": item}}
        elif pick == 1:
            block = {"text": self.word(), "extra": item}
        elif pick == 2:
            block = {
                "toolUse": {"toolUseId": self.word(), "name": self.word(), "input": item, "kind": self.word()}
            }
        elif pick == 3:
            block = {
                "toolResult": {
                    "toolUseId": self.word(),
                    "status": self.word(),
                    "content": item or self.word(),
                }
            }
            if type(block["toolResult"]["content"]) is list:
                block["toolResult"]["content"] = {"items": block["toolResult"]["content"]}
        elif pick == 4:
            block = {"reasoningContent": {"reasoningText": {"text": self.word()}, "kind": item}}
        elif pick == 5:
            block = {"toolUse": {"toolUseId": self.word(), "name": self.word(), "input": (self.word(), item)}}
        elif pick == 6:
            block = [self.word()]
        else:
            block = self.word()
        return block, [compact(block)]

    def message(self):
        blocks, sent = [], []
        for _ in range(self.rng.randrange(6)):
            maker = self.rng.choice(
                [
                    self.text_block,
                    self.tool_call,
                    self.tool_call,
                    self.tool_result,
                    self.reasoning,
                    self.other_block,
                ]
            )
            mark = (self.blank, self.empty, self.null)
            block, texts = maker()
            if self.blank != mark[0]:
                # A single value of only white space: the block is one whole text and counts nothing.
                self.blank, self.empty, self.null = mark
                texts = [compact(block)]
            blocks.append(block)
            sent += texts
        role = self.rng.choice(["user", "assistant"])
        pairs = [("role", role, [role]), ("content", blocks, sent)]
        if self.rng.random() < 0.6:
            pairs.append(("metadata", *self.value(0, keys=False)))
        unknown = self.rng.random() < 0.2
        if unknown:
            # A field that the form does not know.
            pairs.append((self.word(sent=False), *self.value(1, keys=False)))
        message, texts = self.ordered(pairs)
        if self.blank or unknown:
            # A value of only white space outside the content, or a field that the form does not know: the
            # message is one whole text, with every name in it.
            self.cores, self.unsent = self.cores + self.unsent, []
            self.empty = self.null = 0
            texts = [compact(message)]
        return message, texts


@pytest.mark.parametrize("seed", range(400))
def test_every_value_of_a_generated_message_is_covered_by_exactly_one_text(seed):
    maker = Maker(seed)
    message, expected = maker.message()
    before = copy.deepcopy(message)
    texts, empty, null = plan_of(message)
    assert texts == expected, "the texts, their spelling and their order"
    assert (empty, null) == (maker.empty, maker.null)
    # Every string, number and tool-call key that was placed stands in exactly one text, exactly once.
    for core in maker.cores:
        assert sum(text.count(core) for text in texts) == 1, core
    # The names of fields outside a tool call and outside a whole text are structure and are never sent.
    assert all(core not in text for core in maker.unsent for text in texts)
    assert all(type(text) is str and text for text in texts), "no empty text is sent"
    # No text is larger than the whole message was, so the size limit refuses nothing new.
    limit = len(compact(message).encode())
    assert all(len(text.encode()) <= limit for text in texts)
    assert message == before and plan_of(message) == (texts, empty, null)


def test_the_generated_messages_reach_every_kind_of_part():
    kinds = set()
    for seed in range(400):
        message, _ = Maker(seed).message()
        kinds.update("message:" + key for key in message if key in ("role", "content", "metadata"))
        kinds.update("message:other" for key in message if key not in ("role", "content", "metadata"))
        for block in message["content"]:
            kinds.add("block:" + (next(iter(block)) if type(block) is dict and len(block) == 1 else "other"))
    assert kinds >= {
        "message:metadata",
        "message:other",
        "block:text",
        "block:toolUse",
        "block:toolResult",
        "block:reasoningContent",
        "block:image",
        "block:other",
    }


# ── the capture path and the history helper send one text for one result ─────────────────────────────────────

QUESTION = "ordinary user question"
RECORD = {"zeta": 'café "quoted"', "alpha": [3, 1.5, None, False], "note": "second line\nhere"}


@tool
def get_declared_record(request_ref: str) -> dict:
    """Return the fixture's stored record of one request."""
    return {
        "status": "success",
        "content": [{"json": dict(RECORD, request_ref=request_ref)}, {"text": "kept"}],
    }


class ReadToolModel(wiring.ModelFixture):
    """One call of the read tool, then a final text."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1:
            use = {"toolUseId": "ordinary-read", "name": "get_declared_record"}
            arguments = json.dumps({"request_ref": "declared-request"})
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": use}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ordinary complete"}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def turn(monkeypatch, client, *, history=None, model=None):
    """One chat turn through the real backend, the SDK's tool cycle and the real checker."""
    wiring.configure_guard(monkeypatch, client)
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [get_declared_record])
    model = ReadToolModel() if model is None else model
    backend = binding.ToolBindingBackend(Settings(), model=model)
    agents, build = [], backend.build_agent

    def built(**kwargs):
        agents.append(build(**kwargs))
        agents[-1].callback_handler = null_callback_handler
        return agents[-1]

    monkeypatch.setattr(backend, "build_agent", built)
    result = backend.chat_with_tools(QUESTION, history=history)
    shown = [call["content"][0]["text"]["text"] for call in client.calls]
    records = [(r["boundary"], r["outcome"]) for r in (result.input_admission or {}).get("records", [])]
    return SimpleNamespace(
        result=result, backend=backend, model=model, agents=agents, shown=shown, records=records
    )


def test_a_stored_tool_result_is_sent_as_the_capture_path_sent_it(monkeypatch):
    first = turn(monkeypatch, wiring.ClientFixture(deny=lambda text: False))
    assert first.result.final_text == "ordinary complete" and first.records == [
        ("OWNER_PROMPT", "ALLOW"),
        ("TOOL_RESULT", "ALLOW"),
    ]
    captured = first.shown[1]
    stored = _record_messages(first.agents[0].messages)
    (holder,) = [m for m in stored if any("toolResult" in block for block in m["content"])]
    (result,) = [block["toolResult"] for block in holder["content"]]
    # The capture path and the history helper call one function, and the texts are byte-equal.
    assert captured == binding._tool_result_admission_text(result["content"]) == compact(result["content"])
    assert texts_of(holder).count(captured) == 1
    assert '"zeta":"café \\"quoted\\""' in captured and captured.index('"zeta"') < captured.index('"alpha"')
    # The same conversation as the earlier conversation of a next turn: the stored result is sent as that
    # same text, and every other stored value in the text the helper derives.
    client = wiring.ClientFixture(deny=lambda text: False)
    second = turn(monkeypatch, client, history=stored)
    expected = [text for message in stored for text in texts_of(message)]
    assert second.shown[: 1 + len(expected)] == [QUESTION, *expected]
    assert second.shown.count(captured) == 2, "once from the history and once for this turn's fresh result"
    assert second.records[: 1 + len(expected)] == [
        ("OWNER_PROMPT", "ALLOW"),
        *[("HISTORY_TEXT", "ALLOW")] * len(expected),
    ]
    assert second.result.final_text == "ordinary complete"


def together(*tokens):
    """A verdict that refuses only a text that holds two of the tokens together."""
    return lambda text: sum(token in text for token in tokens) >= 2


def test_earlier_conversation_blocked_only_as_one_json_text_is_admitted(monkeypatch):
    history = [
        {"role": "user", "content": [{"text": "An earlier question about " + RESULT_ONE + "."}]},
        {
            "role": "assistant",
            "content": [{"text": "An earlier answer."}, {"text": "[tool results] " + RESULT_TWO}],
        },
        {"role": "user", "content": [{"text": RESULT_ONE}, {"text": RESULT_TWO}]},
        {"role": "assistant", "content": [{"text": "A second answer."}]},
    ]
    before = copy.deepcopy(history)
    client = wiring.ClientFixture(deny=together(RESULT_ONE, RESULT_TWO))
    # Only the third message holds both tokens when it is one compact JSON text.
    assert [client.deny(compact(message)) for message in history] == [False, False, True, False]
    seen = turn(monkeypatch, client, history=history)
    # No text of the earlier conversation holds both tokens, so the turn is not held.
    assert seen.result.final_text == "ordinary complete" and seen.backend._input_hold is False
    assert seen.shown[:11] == [
        QUESTION,
        "user",
        "An earlier question about " + RESULT_ONE + ".",
        "assistant",
        "An earlier answer.",
        "[tool results] " + RESULT_TWO,
        "user",
        RESULT_ONE,
        RESULT_TWO,
        "assistant",
        "A second answer.",
    ]
    assert history == before


@pytest.mark.parametrize("held", range(1, 10))
def test_a_refused_text_of_the_earlier_conversation_holds_the_turn(monkeypatch, held):
    history = [
        {"role": "user", "content": [{"text": "An earlier question."}]},
        {"role": "assistant", "content": [{"text": "An earlier answer."}, {"text": "[tool results] kept"}]},
        {"role": "user", "content": [{"text": "A second question."}, {"text": "Its second part."}]},
        {"role": "assistant", "content": []},
    ]
    sends = []

    def nth(text):
        sends.append(text)
        return len(sends) == 1 + held

    seen = turn(monkeypatch, wiring.ClientFixture(deny=nth), history=history)
    # The question passes; the refused text of the history holds before any model call.
    assert len(sends) == 1 + held and seen.model.calls == 0 and seen.agents == []
    assert seen.backend._input_hold is True
    assert seen.records == [
        ("OWNER_PROMPT", "ALLOW"),
        *[("HISTORY_TEXT", "ALLOW")] * (held - 1),
        ("HISTORY_TEXT", "BLOCKED"),
    ]


class MaskingClient(wiring.ClientFixture):
    """Allows every text it is shown; the ``held``-th text is answered as one masked name."""

    def __init__(self, held):
        super().__init__(deny=lambda text: False)
        self.held = held

    def apply_guardrail(self, **kwargs):
        if len(self.calls) + 1 != self.held:
            return super().apply_guardrail(**kwargs)
        self.calls.append(kwargs)
        return masked_answer(kwargs["content"][0]["text"]["text"], "ordinaryselected")


@pytest.mark.parametrize("held", [1, 2, 5, 8])
def test_a_masked_text_of_the_earlier_conversation_holds_the_turn(monkeypatch, held):
    """History is admitted unchanged or not at all: a text that the guardrail would change holds."""
    history = [
        {"role": "user", "content": [{"text": "An earlier question."}]},
        {"role": "assistant", "content": [{"text": "An earlier answer."}, {"text": "[tool results] kept"}]},
        {"role": "user", "content": [{"text": "A second question."}, {"text": "Its second part."}]},
    ]
    before = copy.deepcopy(history)
    seen = turn(monkeypatch, MaskingClient(1 + held), history=history)
    assert seen.model.calls == 0 and seen.agents == [] and seen.backend._input_hold is True
    assert seen.records == [
        ("OWNER_PROMPT", "ALLOW"),
        *[("HISTORY_TEXT", "ALLOW")] * (held - 1),
        ("HISTORY_TEXT", "ANONYMIZED"),
    ]
    assert len(seen.shown) == 1 + held and history == before


@pytest.mark.parametrize(
    ("earlier", "reason"),
    [
        (lambda: two_results("b" * 40000, "c" * 40000), "INPUT_LIMIT"),
        (lambda: UNENCODABLE["a_field_name_outside_the_content"], "INVALID_INPUT"),
    ],
    ids=["over_the_size_limit", "not_encodable"],
)
def test_an_earlier_message_that_the_checker_refuses_as_one_text_holds_the_turn(monkeypatch, earlier, reason):
    history = [
        {"role": "user", "content": [{"text": "An earlier question."}]},
        earlier(),
        {"role": "assistant", "content": [{"text": "An earlier answer."}]},
    ]
    seen = turn(monkeypatch, wiring.ClientFixture(deny=lambda text: False), history=history)
    # The question and the first earlier message pass; the refused message holds before any model call.
    assert seen.model.calls == 0 and seen.agents == [] and seen.backend._input_hold is True
    assert seen.shown == [QUESTION, "user", "An earlier question."]
    records = seen.result.input_admission["records"]
    assert [(record["boundary"], record["outcome"], record["reason"]) for record in records] == [
        ("OWNER_PROMPT", "ALLOW", "NATIVE_NONE"),
        ("HISTORY_TEXT", "ALLOW", "NATIVE_NONE"),
        ("HISTORY_TEXT", "ALLOW", "NATIVE_NONE"),
        ("HISTORY_TEXT", "UNKNOWN", reason),
    ]


def test_a_refusal_that_depends_on_the_name_of_an_unknown_field_holds_the_turn(monkeypatch):
    """An earlier message with a field that the form does not know reaches the check as one whole text."""
    earlier = {"role": "assistant", "content": [{"text": "An earlier answer."}], UNKNOWN_FIELD: "kept"}
    history = [{"role": "user", "content": [{"text": "An earlier question."}]}, earlier]
    before = copy.deepcopy(history)
    client = wiring.ClientFixture(deny=lambda text: UNKNOWN_FIELD in text)
    seen = turn(monkeypatch, client, history=history)
    # The question and the first earlier message pass; the whole second message is refused for the name.
    assert seen.model.calls == 0 and seen.agents == [] and seen.backend._input_hold is True
    assert seen.shown == [QUESTION, "user", "An earlier question.", compact(earlier)]
    assert seen.records == [
        ("OWNER_PROMPT", "ALLOW"),
        ("HISTORY_TEXT", "ALLOW"),
        ("HISTORY_TEXT", "ALLOW"),
        ("HISTORY_TEXT", "BLOCKED"),
    ]
    assert history == before


def test_a_refusal_that_depends_on_a_name_inside_a_redacted_value_holds_the_turn(monkeypatch):
    """An earlier reasoning block whose redacted value is an object reaches the check as one whole text."""
    block = {"reasoningContent": {"redactedContent": {REDACTED_NAME: 7}}}
    earlier = {"role": "assistant", "content": [block, {"text": "An earlier answer."}]}
    history = [{"role": "user", "content": [{"text": "An earlier question."}]}, earlier]
    before = copy.deepcopy(history)
    client = wiring.ClientFixture(deny=lambda text: REDACTED_NAME in text)
    seen = turn(monkeypatch, client, history=history)
    # The question, the first earlier message and the role pass; the whole block is refused for the name.
    assert seen.model.calls == 0 and seen.agents == [] and seen.backend._input_hold is True
    assert seen.shown == [QUESTION, "user", "An earlier question.", "assistant", compact(block)]
    assert seen.records == [
        ("OWNER_PROMPT", "ALLOW"),
        ("HISTORY_TEXT", "ALLOW"),
        ("HISTORY_TEXT", "ALLOW"),
        ("HISTORY_TEXT", "ALLOW"),
        ("HISTORY_TEXT", "BLOCKED"),
    ]
    assert history == before


def test_earlier_conversation_that_json_cannot_carry_still_holds_with_its_own_line(monkeypatch, caplog):
    history = [{"role": "user", "content": [{"image": {"source": {"bytes": b"declared"}}}]}]
    client = wiring.ClientFixture(deny=lambda text: False)
    with caplog.at_level("WARNING"):
        seen = turn(monkeypatch, client, history=history)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("AMLAS_INPUT_HOLD")]
    assert lines == [NOT_SERIALIZABLE_LINE]
    assert seen.shown == [QUESTION] and seen.model.calls == 0 and seen.backend._input_hold is True


def test_earlier_conversation_is_not_derived_when_checks_are_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        binding, "_history_admission_texts", lambda message: calls.append(message) or ["unused"]
    )
    client = wiring.ClientFixture(deny=lambda text: True)
    wiring.configure_guard(monkeypatch, client)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [get_declared_record])
    backend = binding.ToolBindingBackend(Settings(), model=ReadToolModel())
    history = [{"role": "user", "content": [{"text": "An earlier question."}]}]
    result = backend.chat_with_tools(QUESTION, history=history)
    assert result.final_text == "ordinary complete" and calls == [] and client.calls == []


# ── the restore after a replaced worker ──────────────────────────────────────────────────────────────────────


def selected(response, guard):
    """The recorded answer as an answer of the selected guardrail ``guard``."""
    if guard is not None:
        response["assessments"][0]["appliedGuardrailDetails"]["guardrailId"] = guard
    return response


def blocked_answer(text, guard=None):
    """One content filter blocked the text; the service returns its own notice as output."""
    response = selected(recorded_none_response(text), guard)
    response["action"] = "GUARDRAIL_INTERVENED"
    response["actionReason"] = "Guardrail blocked."
    for entry in response["assessments"][0]["contentPolicy"]["filters"]:
        if entry["type"] == "PROMPT_ATTACK":
            entry.update(action="BLOCKED", confidence="MEDIUM", detected=True)
    response["outputs"] = [{"text": NOTICE}]
    return response


def masked_answer(text, guard=None):
    """The whole text was reported as one name; the service returns its placeholder."""
    response = selected(recorded_none_response(text), guard)
    response["action"] = "GUARDRAIL_INTERVENED"
    response["actionReason"] = "Guardrail masked."
    for entry in response["assessments"][0]["sensitiveInformationPolicy"]["piiEntities"]:
        if entry["type"] == "NAME":
            entry.update(action="ANONYMIZED", detected=True, match=text)
    response["outputs"] = [{"text": "{NAME}"}]
    return response


class ScriptedGuard:
    """Stands in for the guardrail client. ``verdict(text)`` picks each answer; every sent text is recorded."""

    def __init__(self, verdict=None):
        self.verdict = verdict if verdict is not None else (lambda _text: None)
        self.texts = []

    def close(self):
        pass

    def apply_guardrail(self, **kwargs):
        text = kwargs["content"][0]["text"]["text"]
        self.texts.append(text)
        answer = self.verdict(text)
        return recorded_none_response(text) if answer is None else answer(text)


def conversation():
    """Four stored messages of a run up to its fine-tuning call; every value is the test's own."""
    return [
        {"role": "user", "content": [{"text": "Fine-tune one licensed coding model."}]},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "", "signature": SIGNATURE_ONE}}},
                {"reasoningContent": {"reasoningText": {"text": "", "signature": SIGNATURE_TWO}}},
                {
                    "toolUse": {
                        "toolUseId": "declared-read-one",
                        "name": "list_declared_profiles",
                        "input": {},
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "declared-read-two",
                        "name": "get_declared_profile",
                        "input": {
                            "profile_id": "declared-profile",
                            "plan_only": True,
                            "limits": {"rows": 3, "share": 0.5},
                            "tags": ["declared", ""],
                            "note": None,
                        },
                    }
                },
            ],
            "metadata": {"usage": {"inputTokens": 1200, "outputTokens": 80}, "metrics": {"latencyMs": 900}},
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "declared-read-one",
                        "status": "success",
                        "content": [{"text": json.dumps({"profiles": [RESULT_ONE]})}],
                    }
                },
                {
                    "toolResult": {
                        "toolUseId": "declared-read-two",
                        "status": "success",
                        "content": [{"json": {"profile": RESULT_TWO, "rows": 3}}],
                    }
                },
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
                            "model_ref": MODEL_REF,
                            "preparation_ref": PREPARATION_REF,
                            "recipe_id": "full_v1",
                        },
                    }
                },
            ],
        },
    ]


def completion_text(task):
    """The content list of the completion result that the restore builds, as its one text."""
    event = {
        "schema_version": 1,
        "event_type": "original_sft_completed",
        "data_only": True,
        "request_ref": "sft_" + task["operation_digest"][:32],
        "tool_use_id": task["pending_tool_use_id"],
        "status": "verified_completed",
    }
    return compact([{"text": json.dumps(event, sort_keys=True, separators=(",", ":"))}])


def stored_texts():
    """The texts of the four stored messages, written out: a checkpoint keeps every object's keys sorted."""
    return [
        # the owner prompt and its role
        "Fine-tune one licensed coding model.",
        "user",
        # the assistant message with two signatures and two tool-call records
        SIGNATURE_ONE,
        SIGNATURE_TWO,
        "list_declared_profiles",
        "declared-read-one",
        "limits",
        "rows",
        "3",
        "share",
        "0.5",
        "note",
        "plan_only",
        "true",
        "profile_id",
        "declared-profile",
        "tags",
        "declared",
        "get_declared_profile",
        "declared-read-two",
        "900",
        "1200",
        "80",
        "assistant",
        # the user message with two tool results
        compact([{"text": json.dumps({"profiles": [RESULT_ONE]})}]),
        "success",
        "declared-read-one",
        compact([{"json": {"profile": RESULT_TWO, "rows": 3}}]),
        "success",
        "declared-read-two",
        "user",
        # the assistant message that calls the fine-tuning tool
        "Submitting the fine-tuning job.",
        "model_ref",
        MODEL_REF,
        "preparation_ref",
        PREPARATION_REF,
        "recipe_id",
        "full_v1",
        "submit_qwen_sft",
        SFT_USE,
        "assistant",
    ]


def completed_texts(task):
    """The stored texts and the completion message that the restore adds, in the order it builds it."""
    return [*stored_texts(), "user", SFT_USE, "success", completion_text(task)]


def declared_task(messages):
    """The restore's declared DATA input, not an enrolled TASK."""
    return {
        "schema_version": 2,
        "logical_run_id": "0" * 32,
        "private_store_correlation": "declared-component-store",
        "source_refs": [],
        "causal_parent_ref": None,
        "operation_digest": "1" * 64,
        "native_job_name": "declared-original-sft",
        "native_job_arn": "declared-original-identifier",
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


def restoring(monkeypatch, guard, *, messages=None, complete=True, checks=True):
    """A chat engine at the restore boundary: the real admission and checker, inert agent and context."""
    task = declared_task(conversation() if messages is None else messages)
    selected = private_checker(guard) if checks else None
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
    if not checks:
        monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "false")
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
    return SimpleNamespace(
        engine=engine,
        backend=backend,
        task=task,
        context=InertContext(),
        effects=effects,
        guard=guard,
        prefix=checkpoint._task_causal_prefix(task),
    )


def restore(work, site):
    """Run one restore at ``site`` and return the hold that stopped it, or None."""
    try:
        if site == "pending":
            work.engine.load_normal_snapshot(work.task)
        else:
            work.engine.claim_normal_completion(work.context, work.task)
    except _InputAdmissionHeldError as error:
        return error
    return None


BLOCKED_WHOLE = {
    "assistant_message_with_signatures_and_tool_call_records": (1, (SIGNATURE_ONE, SIGNATURE_TWO)),
    "user_message_with_two_tool_results": (2, (RESULT_ONE, RESULT_TWO)),
}


@pytest.mark.parametrize("site", ["pending", "completed"])
@pytest.mark.parametrize("shape", sorted(BLOCKED_WHOLE))
def test_the_restore_admits_a_history_whose_message_is_blocked_only_as_one_json_text(
    monkeypatch, site, shape
):
    position, tokens = BLOCKED_WHOLE[shape]
    refuses = together(*tokens)
    guard = ScriptedGuard(lambda text: blocked_answer if refuses(text) else None)
    work = restoring(monkeypatch, guard, complete=site == "completed")
    # The message as one compact JSON text holds both tokens; no other stored message does.
    assert [refuses(compact(message)) for message in work.prefix] == [index == position for index in range(4)]
    assert restore(work, site) is None, "the restore was held"
    assert work.backend._input_hold is False and "snapshot_load" in work.effects
    assert not any(refuses(text) for text in guard.texts)
    assert all(receipt.outcome == "ALLOW" for receipt in work.backend._input_receipts)
    # What was loaded is the stored conversation, unchanged: both signatures and both results are in it.
    loaded = _record_messages(work.engine._normal_agent.messages)
    assert loaded[:4] == work.prefix and work.task["sdk_snapshot"]["data"]["messages"] == conversation()
    assert compact(loaded[1]).count(SIGNATURE_ONE) == 1 and compact(loaded[2]).count(RESULT_TWO) == 1


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_the_restore_sends_these_texts_in_this_order(monkeypatch, site):
    guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, complete=site == "completed")
    assert restore(work, site) is None
    if site == "pending":
        expected = stored_texts()
        boundaries = ["HISTORY_TEXT"] * len(expected)
    else:
        history = completed_texts(work.task)
        # The completion result first, as the JSON of the whole result; then both passes over the history.
        result = {
            "toolUseId": SFT_USE,
            "status": "success",
            "content": json.loads(completion_text(work.task)),
        }
        expected = [compact(result), *history, *history]
        boundaries = ["TOOL_RESULT", *["HISTORY_TEXT"] * (2 * len(history))]
    assert guard.texts == expected
    assert [receipt.boundary for receipt in work.backend._input_receipts] == boundaries[-128:]
    assert {
        receipt.tool_name for receipt in work.backend._input_receipts if receipt.boundary == "HISTORY_TEXT"
    } == {"UNKNOWN"}
    # Derived from the stored messages by the one helper, message by message.
    assert stored_texts() == [text for message in work.prefix for text in texts_of(message)]
    assert work.backend._input_hold is False and "snapshot_load" in work.effects


def test_a_stored_json_item_is_sent_in_the_key_order_of_the_checkpoint():
    """A checkpoint keeps every object's keys sorted; the first check saw the key order that the tool wrote."""
    content = [{"json": {"zeta": 1, "alpha": {"beta": 2, "aleph": 3}}}, {"text": "kept"}]
    captured = binding._tool_result_admission_text(content)
    assert captured == '[{"json":{"zeta":1,"alpha":{"beta":2,"aleph":3}}},{"text":"kept"}]'
    messages = conversation()
    messages[2]["content"][1]["toolResult"]["content"] = content
    stored = checkpoint._task_causal_prefix(declared_task(messages))[2]
    # The stored text is the compact JSON of the stored content list: the same value with sorted keys.
    sent = '[{"json":{"alpha":{"aleph":3,"beta":2},"zeta":1}},{"text":"kept"}]'
    assert texts_of(stored).count(sent) == 1 and captured not in texts_of(stored)
    assert json.loads(sent) == json.loads(captured) == content
    # Content that already has that key order is the same text at both checks.
    assert binding._tool_result_admission_text(json.loads(sent)) == sent


def single_sends(site):
    count = len(stored_texts()) if site == "pending" else 1 + 2 * (len(stored_texts()) + 4)
    return [(site, send) for send in range(1 if site == "pending" else 2, count + 1)]


@pytest.mark.parametrize("answer", [blocked_answer, masked_answer], ids=["blocked", "masked"])
@pytest.mark.parametrize(("site", "send"), [*single_sends("pending"), *single_sends("completed")])
def test_a_block_or_a_mask_on_any_single_text_holds_the_restore(monkeypatch, site, send, answer):
    sends = []

    def verdict(text):
        sends.append(text)
        return answer if len(sends) == send else None

    guard = ScriptedGuard(verdict)
    work = restoring(monkeypatch, guard, complete=site == "completed")
    error = restore(work, site)
    assert isinstance(error, _InputAdmissionHeldError), "the refused text did not hold the restore"
    # The restore stops at the refused text: nothing after it is sent and nothing is loaded.
    assert len(guard.texts) == send and work.backend._input_hold is True
    assert "snapshot_load" not in work.effects and "agent_build" not in work.effects
    first_pass = 1 + len(stored_texts()) + 4
    assert ("completion_cas" in work.effects) is (site == "completed" and send > first_pass)
    last = work.backend._input_receipts[-1]
    assert (last.boundary, last.outcome) == (
        "HISTORY_TEXT",
        "BLOCKED" if answer is blocked_answer else "ANONYMIZED",
    )


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_a_single_text_over_the_size_limit_is_refused_and_holds(monkeypatch, site):
    messages = conversation()
    messages[0]["content"][0]["text"] = "a" * (MAX_INPUT_BYTES + 1)
    guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, messages=messages, complete=site == "completed")
    assert isinstance(restore(work, site), _InputAdmissionHeldError)
    last = work.backend._input_receipts[-1]
    assert (last.boundary, last.outcome, last.reason) == ("HISTORY_TEXT", "UNKNOWN", "INPUT_LIMIT")
    assert "a" * (MAX_INPUT_BYTES + 1) not in guard.texts and "snapshot_load" not in work.effects


def texts_before_the_third_message():
    """The stored texts of the first two messages, written out in ``stored_texts``."""
    texts = stored_texts()
    return texts[: texts.index(compact([{"text": json.dumps({"profiles": [RESULT_ONE]})}]))]


@pytest.mark.parametrize("site", ["pending", "completed"])
@pytest.mark.parametrize(
    ("size", "admitted"),
    [(MAX_INPUT_BYTES, True), (MAX_INPUT_BYTES + 1, False)],
    ids=["at_the_limit", "one_byte_over"],
)
def test_message_over_the_limit_is_sent_whole_and_refused(monkeypatch, site, size, admitted):
    """Two stored tool results that each fit the limit alone: their message is measured as one text."""
    messages = conversation()
    messages[2] = sized(size)
    guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, messages=messages, complete=site == "completed")
    first, second = (compact(block["toolResult"]["content"]) for block in work.prefix[2]["content"])
    assert len(compact(work.prefix[2]).encode()) == size
    assert max(len(first.encode()), len(second.encode())) < MAX_INPUT_BYTES
    held = restore(work, site)
    if admitted:
        # 65,536 bytes reach the checks: each result is sent as the text of its first check.
        assert held is None and "snapshot_load" in work.effects
        assert guard.texts.count(first) == guard.texts.count(second) == (1 if site == "pending" else 2)
        return
    assert isinstance(held, _InputAdmissionHeldError) and work.backend._input_hold is True
    last = work.backend._input_receipts[-1]
    assert (last.boundary, last.outcome, last.reason) == ("HISTORY_TEXT", "UNKNOWN", "INPUT_LIMIT")
    # The guardrail was asked about every text before that message and about no text of it.
    offset = 0 if site == "pending" else 1
    assert guard.texts[offset:] == texts_before_the_third_message()
    assert "snapshot_load" not in work.effects and "completion_cas" not in work.effects


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_a_stored_message_that_cannot_be_encoded_is_sent_whole_and_refused(monkeypatch, site):
    """A field name that strict UTF-8 cannot encode stands in no single text; the whole message is refused."""
    messages = conversation()
    messages[2]["metadata"] = {"usage": {"declared" + LONE_SURROGATE + "name": 5}}
    guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, messages=messages, complete=site == "completed")
    assert isinstance(restore(work, site), _InputAdmissionHeldError) and work.backend._input_hold is True
    last = work.backend._input_receipts[-1]
    assert (last.boundary, last.outcome, last.reason) == ("HISTORY_TEXT", "UNKNOWN", "INVALID_INPUT")
    offset = 0 if site == "pending" else 1
    assert guard.texts[offset:] == texts_before_the_third_message()
    assert "snapshot_load" not in work.effects and "completion_cas" not in work.effects


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_a_refusal_that_depends_on_the_name_of_an_unknown_field_holds_the_restore(monkeypatch, site):
    """A stored message with a field that the form does not know reaches the check as one whole text."""
    messages = conversation()
    messages[2][UNKNOWN_FIELD] = "kept"
    guard = ScriptedGuard(lambda text: blocked_answer if UNKNOWN_FIELD in text else None)
    work = restoring(monkeypatch, guard, messages=messages, complete=site == "completed")
    held = restore(work, site)
    assert isinstance(held, _InputAdmissionHeldError) and work.backend._input_hold is True
    last = work.backend._input_receipts[-1]
    assert (last.boundary, last.outcome) == ("HISTORY_TEXT", "BLOCKED")
    # The guardrail was asked about every text before that message and then about the whole message.
    offset = 0 if site == "pending" else 1
    assert guard.texts[offset:] == [*texts_before_the_third_message(), compact(work.prefix[2])]
    assert "snapshot_load" not in work.effects and "completion_cas" not in work.effects
    # The cause of the hold names the third stored message.
    assert (held.data["message_position"], held.data["message_role"]) == (3, "user")


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_a_refusal_that_depends_on_a_name_inside_a_redacted_value_holds_the_restore(monkeypatch, site):
    """A stored reasoning block whose redacted value is an object reaches the check as one whole text."""
    messages = conversation()
    block = {"reasoningContent": {"redactedContent": {REDACTED_NAME: 7}}}
    messages[1]["content"].insert(0, block)
    guard = ScriptedGuard(lambda text: blocked_answer if REDACTED_NAME in text else None)
    work = restoring(monkeypatch, guard, messages=messages, complete=site == "completed")
    assert work.prefix[1]["content"][0] == block
    held = restore(work, site)
    assert isinstance(held, _InputAdmissionHeldError) and work.backend._input_hold is True
    last = work.backend._input_receipts[-1]
    assert (last.boundary, last.outcome) == ("HISTORY_TEXT", "BLOCKED")
    # The guardrail was asked about the two texts of the first message and then about the whole block.
    offset = 0 if site == "pending" else 1
    assert guard.texts[offset:] == [*stored_texts()[:2], compact(block)]
    assert "snapshot_load" not in work.effects and "completion_cas" not in work.effects
    # The cause of the hold names the second stored message.
    assert (held.data["message_position"], held.data["message_role"]) == (2, "assistant")


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_with_checks_disabled_the_restore_makes_no_guardrail_call(monkeypatch, site):
    guard = ScriptedGuard(lambda text: blocked_answer)
    work = restoring(monkeypatch, guard, complete=site == "completed", checks=False)
    assert restore(work, site) is None
    assert guard.texts == [] and work.backend._input_receipts == [] and "snapshot_load" in work.effects
    assert _record_messages(work.engine._normal_agent.messages)[:4] == work.prefix


@pytest.mark.parametrize("site", ["pending", "completed"])
def test_a_latched_hold_still_stops_the_restore_before_any_check(monkeypatch, site):
    guard = ScriptedGuard()
    work = restoring(monkeypatch, guard, complete=site == "completed")
    work.backend._input_hold = True
    assert isinstance(restore(work, site), _InputAdmissionHeldError)
    assert guard.texts == [] and work.effects == []


# ── every history admission uses the one helper ──────────────────────────────────────────────────────────────


def history_admissions(source):
    """Every call in ``source`` that names the history kind: its function, whether its text is one of the
    helper's texts, and whether the call serializes anything itself."""
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(isinstance(arg, ast.Constant) and arg.value == "HISTORY_TEXT" for arg in node.args):
            continue
        names = {arg.id for arg in node.args if isinstance(arg, ast.Name)}
        scope, loops = node, []
        while not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = parents[scope]
            if isinstance(scope, ast.For) and isinstance(scope.target, ast.Name) and scope.target.id in names:
                loops.append(scope.iter)
        derived = []
        for origin in loops:
            if isinstance(origin, ast.Name):
                # A name is followed to every assignment it has in the same function.
                assigned = [
                    item.value
                    for item in ast.walk(scope)
                    if isinstance(item, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == origin.id for target in item.targets
                    )
                ]
                derived.append(bool(assigned) and all(is_helper_call(value) for value in assigned))
            else:
                derived.append(is_helper_call(origin))
        serialized = any(isinstance(item, ast.Attribute) and item.attr == "dumps" for item in ast.walk(node))
        found.append((scope.name, derived == [True], serialized))
    return found


def is_helper_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_history_admission_texts"
        and len(node.args) == 1
        and not node.keywords
    )


def test_every_history_admission_derives_its_texts_with_the_one_helper():
    engine = history_admissions(inspect.getsource(chat_engine))
    backend = history_admissions(inspect.getsource(binding))
    assert sorted(engine) == [
        ("assemble_normal_completion", True, False),
        ("claim_normal_completion", True, False),
        ("load_normal_snapshot", True, False),
    ]
    assert backend == [("_chat_with_tools_once", True, False)]
    # No other module of the package admits history, and each function is defined once.
    assert '"HISTORY_TEXT"' not in inspect.getsource(research_runtime_profile)
    assert inspect.getsource(binding).count("def _history_admission_texts(") == 1
    assert inspect.getsource(binding).count("def _tool_result_admission_text(") == 1
    # The capture path and the helper serialize a content list through the one function.
    capture = inspect.getsource(ToolBindingBackend._guard_new_tool_projection)
    assert 'text = _tool_result_admission_text(result["content"])' in capture and "json.dumps" not in capture
    helper = inspect.getsource(binding._history_block_texts)
    assert "_tool_result_admission_text(item)" in helper and "json.dumps" not in helper


def test_the_source_reader_refuses_an_admission_that_builds_its_text_another_way():
    """The reader above is shown to fail: three ways of admitting history without the helper."""
    cases = {
        "whole_message": (
            "def restore(backend, messages):\n"
            "    for message in messages:\n"
            "        backend._admit_input(json.dumps(message), 'HISTORY_TEXT', immutable=True)\n"
        ),
        "another_derivation": (
            "def restore(backend, messages):\n"
            "    for message in messages:\n"
            "        for text in other_texts(message):\n"
            "            backend._admit_input(text, 'HISTORY_TEXT', immutable=True)\n"
        ),
        "a_name_assigned_twice": (
            "def restore(backend, message):\n"
            "    texts = _history_admission_texts(message)\n"
            "    texts = [json.dumps(message)]\n"
            "    for text in texts:\n"
            "        backend._admit_input(text, 'HISTORY_TEXT', immutable=True)\n"
        ),
        "the_helper": (
            "def restore(backend, messages):\n"
            "    for message in messages:\n"
            "        for text in _history_admission_texts(message):\n"
            "            backend._admit_input(text, 'HISTORY_TEXT', immutable=True)\n"
        ),
        "the_helper_through_a_name_and_a_wrapper": (
            "def restore(backend, messages):\n"
            "    for position, message in enumerate(messages, 1):\n"
            "        texts = _history_admission_texts(message)\n"
            "        for text in texts:\n"
            "            admit(backend, text, 'HISTORY_TEXT', position, message)\n"
        ),
    }
    assert {name: history_admissions(source) for name, source in cases.items()} == {
        "whole_message": [("restore", False, True)],
        "another_derivation": [("restore", False, False)],
        "a_name_assigned_twice": [("restore", False, False)],
        "the_helper": [("restore", True, False)],
        "the_helper_through_a_name_and_a_wrapper": [("restore", True, False)],
    }


def test_the_restore_sites_and_the_earlier_conversation_share_the_helper(monkeypatch):
    """Behaviour, not only source: a marker text from a replaced helper reaches the check at every site."""
    marker = " ".join(("derived", "by", "the", "one", "helper"))
    monkeypatch.setattr(binding, "_history_admission_texts", lambda message: [marker])
    for site, count in (("pending", 4), ("completed", 10)):
        guard = ScriptedGuard()
        work = restoring(monkeypatch, guard, complete=site == "completed")
        assert restore(work, site) is None
        assert [text for text in guard.texts if text == marker] == [marker] * count
        assert len(guard.texts) == count + (site == "completed")
    client = wiring.ClientFixture(deny=lambda text: False)
    history = [{"role": "user", "content": [{"text": "An earlier question."}]}]
    seen = turn(monkeypatch, client, history=history)
    assert seen.shown[:2] == [QUESTION, marker]


# ── the research results at the restore ──────────────────────────────────────────────────────────────────────


def research_restore(monkeypatch):
    """A claimed TASK with one settled research result per unit of the research plan, then the fine-tuning call."""
    plan = research_runtime_profile._plan()
    prefix = [{"role": "user", "content": [{"text": "fresh declared request"}]}]
    for index, unit in enumerate(plan.units):
        identifier = "declared-use-" + str(index)
        page = {"text": 'declared source "' + str(index) + '" café', "zeta": index, "alpha": None}
        prefix.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": identifier,
                            "name": "research_fetch_urls",
                            "input": {"urls": list(unit.urls), "max_pages": 3},
                        }
                    }
                ],
            }
        )
        prefix.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": identifier,
                            "status": "success",
                            "content": [{"json": {"pages": [page]}}],
                        }
                    }
                ],
            }
        )
    prefix.append(
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": SFT_USE, "name": "submit_qwen_sft", "input": {}}}],
        }
    )
    task = declared_task(prefix)
    snapshot = copy.deepcopy(task["sdk_snapshot"])
    snapshot["data"]["messages"].append({"role": "user", "content": []})
    context = SimpleNamespace(
        _completion_receipt=("declared-token", json.dumps({"sdk_snapshot": snapshot}), "declared-task")
    )
    monkeypatch.setattr(checkpoint, "current_normal_run", lambda: context)
    sent = []
    state = {
        "phase": "UNUSED",
        "pending": False,
        "units": {unit.unit_id: unit for unit in plan.units},
        "checker": object(),
    }
    backend = SimpleNamespace(
        _research_started=False,
        _normal_original_result={
            "job_name": task["native_job_name"],
            "job_arn": task["native_job_arn"],
            "trained": True,
            "full_training_complete": True,
        },
        _input_lock=threading.RLock(),
        _input_cycle=CYCLE,
        _qualify_research_binding=lambda **kwargs: state,
        _planning_result_payload=ToolBindingBackend._planning_result_payload,
        _admit_input=lambda text, boundary, **kwargs: sent.append((text, boundary, kwargs)),
    )
    restored = research_runtime_profile._restore_runtime_research_profile(
        SimpleNamespace(tool_backend=backend), context, task, "declared-token", snapshot
    )
    return SimpleNamespace(
        restored=restored, sent=sent, state=state, prefix=checkpoint._task_causal_prefix(task)
    )


def test_the_research_results_are_sent_once_more_in_the_text_of_their_first_check(monkeypatch):
    work = research_restore(monkeypatch)
    assert work.restored is True and work.state["phase"] == "CONSUMED"
    holders = [m for m in work.prefix if any("toolResult" in block for block in m["content"])]
    results = [block["toolResult"] for message in holders for block in message["content"]]
    assert len(results) == len(work.sent) >= 2
    for holder, result, (text, boundary, options) in zip(holders, results, work.sent, strict=True):
        # The kind, the strictness and the tool name of this check are as they were.
        assert boundary == "TOOL_RESULT" and options == {
            "immutable": True,
            "tool_name": "research_fetch_urls",
        }
        # Its text is the content list's text: the one the capture path sent for the fresh result and the
        # one the history check sends for the stored message, byte for byte.
        assert text == binding._tool_result_admission_text(result["content"]) == compact(result["content"])
        assert texts_of(holder).count(text) == 1
        assert text != compact(result) and '"toolUseId"' not in text and "café" in text
    # Only the serialization of that check changed: the source reads the one function.
    source = inspect.getsource(research_runtime_profile._restore_runtime_research_profile)
    assert 'text = _tool_result_admission_text(result["content"])' in source
