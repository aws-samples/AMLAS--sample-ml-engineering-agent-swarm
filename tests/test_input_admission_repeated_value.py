"""A detected value that stands more than once in one checked text.

The native service reports one personal-data record for every occurrence it detects, and it masks only the
occurrences it detected. Observed for one guardrail in one Region, with ``outputScope`` FULL on the INPUT
side: an address that stood twice in one text gave two records with the same match; a number that stood
once on its own and once inside a longer identifier gave one record, one placeholder, and the second copy
unchanged.

The checker forwards the service's own masked text when one of two comparisons reads it as the checked text
with reported values replaced by the placeholder of their own category. The first comparison takes one
record for each value and expects every copy of every reported value replaced, the longest value first. The
second takes one record for each detected occurrence and reads the two texts side by side: nothing else may
have changed, and a value needs at least as many replacements as records. A copy that the service did not
replace stays readable in the forwarded text: that is the service's verdict. An answer that neither
comparison reads stays UNKNOWN with MASK_INVALID and supplies no text.

Template text that itself reads as an address was reported by the same guardrail as one match that begins
with an opening brace; inside the compact JSON of a content list the match began at a closing brace. Only
the first comparison reads a match that begins with an opening brace.

The answers are scripted in the shape of the recorded answers of
``test_input_admission_supplementary_characters.py``. Every value is the test's own, and the number is built
at run time. The checker tests call no native API, model or network. The full turns (a read tool result, an
owner prompt, a research page and a text that must stay unchanged) run the real backend callbacks and SDK
with a scripted model and guard. These tests say nothing about what a deployed guardrail detects, or about
a text that the service has already masked.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import logging
import random
from collections import Counter

import pytest
import test_input_admission_wiring as wiring
import test_research_result_supplementary_characters as research
from strands import tool
from strands.handlers.callback_handler import null_callback_handler
from test_input_admission_supplementary_characters import (
    CYCLE,
    GUARD,
    MODULE,
    SUPPLEMENTARY,
    RecordingGuardClient,
    private_checker,
    recorded_none_response,
    utf16_units,
)
from test_normal_research_profile import TRAIN_URL
from test_runtime_prompt_entry import inert_runtime_setup  # noqa: F401

from amlas.config.settings import Settings
from amlas.tui.chat import tool_binding_backend as binding

ADDRESS = "jordan@example.com"
OTHER_ADDRESS = "morgan@example.com"
NUMBER = "7" * 12  # built here, so the file holds no number of that length
IDENTIFIER = "arn:aws:sagemaker:us-west-2:" + NUMBER + ":training-job/fixture-job"
BANK = "US_BANK_ACCOUNT_NUMBER"
BANK_PLACEHOLDER = "{" + BANK + "}"
LIMIT = 65536

TWICE = "Write to " + ADDRESS + " today, or to " + ADDRESS + " tomorrow."
TWO_RECORDS = [("EMAIL", ADDRESS), ("EMAIL", ADDRESS)]
ONE_RECORD = [("EMAIL", ADDRESS)]
BOTH_MASKED = "Write to {EMAIL} today, or to {EMAIL} tomorrow."
ALONE_AND_INSIDE = "The job ran in account " + NUMBER + ". Its name is " + IDENTIFIER + "."
FORM = "The form shows {EMAIL} where " + ADDRESS + " was typed."


def mask_occurrences(text, value, token, positions):
    """``text`` with the occurrences of ``value`` at the given zero-based positions replaced by ``token``."""
    head, *rest = text.split(value)
    return head + "".join((token if index in positions else value) + part for index, part in enumerate(rest))


def replaced(*parts):
    """A text and its answer from the same parts: every ``(category, value)`` part stands as its placeholder.

    The answer is then the text with each reported value replaced once and nothing else changed.
    """
    text = "".join(part if type(part) is str else part[1] for part in parts)
    answer = "".join(part if type(part) is str else "{" + part[0] + "}" for part in parts)
    return text, answer


def none_answer(text, guard=GUARD):
    """The recorded NONE answer for ``text``, from the guard identity ``guard``."""
    response = recorded_none_response(text)
    details = response["assessments"][0]["appliedGuardrailDetails"]
    details["guardrailId"] = guard
    details["guardrailArn"] = details["guardrailArn"].rsplit("/", 1)[0] + "/" + guard
    return response


def masked_answer(text, records, outputs, guard=GUARD):
    """A masked answer: one detected ANONYMIZED entry for every reported occurrence, and ``outputs``."""
    response = none_answer(text, guard)
    response["action"] = "GUARDRAIL_INTERVENED"
    response["actionReason"] = "Guardrail masked."
    policy = response["assessments"][0]["sensitiveInformationPolicy"]
    reported = {category for category, _ in records}
    policy["piiEntities"] = [entry for entry in policy["piiEntities"] if entry["type"] not in reported] + [
        {"type": category, "action": "ANONYMIZED", "detected": True, "match": match}
        for category, match in records
    ]
    response["outputs"] = outputs
    return response


def decide(text, records, outputs):
    """One check of ``text`` against the scripted masked answer; the service is shown the text once."""
    response = masked_answer(text, records, outputs)
    client = RecordingGuardClient(response)
    decision = private_checker(client).check(text, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert len(client.calls) == 1 and client.calls[0]["content"] == [{"text": {"text": text}}]
    return response, decision


def assert_forwarded(response, decision, *values):
    """The service's own output object is forwarded, and the receipt holds no reported value."""
    data = decision.receipt.to_dict()
    assert (decision.outcome, data["reason"]) == ("ANONYMIZED", "NATIVE_MASKED")
    assert decision.forward_text is response["outputs"][0]["text"]
    for value in values:
        assert value not in json.dumps(data) and value not in repr(decision)
    return data


def assert_refused(decision, *values):
    data = decision.receipt.to_dict()
    assert (decision.outcome, data["reason"]) == ("UNKNOWN", "MASK_INVALID")
    assert decision.forward_text is None
    for value in values:
        assert value not in json.dumps(data) and value not in repr(decision)


def test_same_value_reported_twice_and_masked_twice_is_forwarded():
    response, decision = decide(TWICE, TWO_RECORDS, [{"text": BOTH_MASKED}])
    data = assert_forwarded(response, decision, ADDRESS)
    assert decision.forward_text == BOTH_MASKED
    assert "PII:EMAIL" in data["categories"] and len(data["categories"]) == 13
    assert data["byte_count"] == len(TWICE.encode())


@pytest.mark.parametrize("masked", [0, 1], ids=["the_copy_on_its_own", "the_copy_inside_the_identifier"])
def test_value_present_twice_and_masked_once_is_forwarded_as_returned(masked):
    output = mask_occurrences(ALONE_AND_INSIDE, NUMBER, "{" + BANK + "}", {masked})
    response, decision = decide(ALONE_AND_INSIDE, [(BANK, NUMBER)], [{"text": output}])
    data = assert_forwarded(response, decision)
    assert "PII:" + BANK in data["categories"]
    # The copy that the service did not replace is forwarded as the service returned it.
    assert decision.forward_text.count("{" + BANK + "}") == 1 and decision.forward_text.count(NUMBER) == 1
    assert (IDENTIFIER in decision.forward_text) is (masked == 0)
    # The receipt names the category and never the value.
    assert NUMBER not in json.dumps(data) and NUMBER not in repr(decision)


@pytest.mark.parametrize("copies", [2, 3])
def test_one_record_for_every_occurrence_is_still_accepted(copies):
    text = "Reply to " + " or ".join([ADDRESS] * copies) + " by noon."
    response, decision = decide(text, [("EMAIL", ADDRESS)], [{"text": text.replace(ADDRESS, "{EMAIL}")}])
    assert_forwarded(response, decision, ADDRESS)
    assert decision.forward_text.count("{EMAIL}") == copies


@pytest.mark.parametrize(
    ("copies", "masked", "records"),
    [(4, {0, 2}, 2), (4, {1}, 1), (3, {0, 1, 2}, 3), (5, {0, 4}, 1), (3, {2}, 1)],
    ids=["two_of_four", "one_of_four", "three_of_three", "two_of_five_one_record", "last_of_three"],
)
def test_three_or_more_occurrences_with_a_mix_of_masked_and_unmasked(copies, masked, records):
    text = "Seen: " + ", ".join([NUMBER] * copies) + "."
    output = mask_occurrences(text, NUMBER, "{" + BANK + "}", masked)
    response, decision = decide(text, [(BANK, NUMBER)] * records, [{"text": output}])
    assert_forwarded(response, decision)
    assert decision.forward_text.count("{" + BANK + "}") == len(masked)
    assert decision.forward_text.count(NUMBER) == copies - len(masked)


@pytest.mark.parametrize("repeated", [False, True], ids=["each_once", "one_of_them_twice"])
def test_two_different_values_of_the_same_category(repeated):
    text = "From " + ADDRESS + " to " + OTHER_ADDRESS + (", copy to " + ADDRESS if repeated else "") + "."
    records = [("EMAIL", ADDRESS), ("EMAIL", OTHER_ADDRESS)] + ([("EMAIL", ADDRESS)] if repeated else [])
    output = text.replace(ADDRESS, "{EMAIL}").replace(OTHER_ADDRESS, "{EMAIL}")
    response, decision = decide(text, records, [{"text": output}])
    assert_forwarded(response, decision, ADDRESS, OTHER_ADDRESS)
    assert decision.forward_text.count("{EMAIL}") == (3 if repeated else 2)


FULL_NAME = ("NAME", "Jordan Lee")
FIRST_NAME = ("NAME", "Jordan")
HANDLE = ("USERNAME", "Jordan")
LAST_NAME = ("NAME", "Lee")
START_OF_ITS_CATEGORY = (FULL_NAME, " wrote to ", FIRST_NAME, ".")
START_OF_ANOTHER_CATEGORY = (FULL_NAME, " wrote to ", HANDLE, ".")
END_OF_ITS_CATEGORY = (LAST_NAME, " wrote to ", FULL_NAME, ".")
NESTED = [
    pytest.param(START_OF_ITS_CATEGORY, [FULL_NAME, FIRST_NAME], id="start_of_its_category_reported_second"),
    pytest.param(
        START_OF_ANOTHER_CATEGORY, [FULL_NAME, HANDLE], id="start_of_another_category_reported_second"
    ),
    pytest.param(
        START_OF_ANOTHER_CATEGORY, [HANDLE, FULL_NAME], id="start_of_another_category_reported_first"
    ),
    pytest.param(END_OF_ITS_CATEGORY, [FULL_NAME, LAST_NAME], id="end_of_its_category_reported_second"),
    pytest.param(END_OF_ITS_CATEGORY, [LAST_NAME, FULL_NAME], id="end_of_its_category_reported_first"),
]


@pytest.mark.parametrize(("parts", "records"), NESTED)
def test_value_inside_another_reported_value_is_refused_whichever_is_reported_first(parts, records):
    """A reported value that is the start or the end of another reported value, in either record order.

    Every answer here replaces each reported value once and changes nothing else, and the comparison alone
    would read it so. Only the rule on a value inside another value refuses it; the service decides the
    order of its records. One order has no case: the start of a value of its own category, reported first.
    The comparison then reads the shorter value where the longer one stands and refuses without that rule.
    """
    text, answer = replaced(*parts)
    _, decision = decide(text, records, [{"text": answer}])
    assert_refused(decision, "Jordan", "Lee")

    # The control: the same answer is forwarded once the inner value stands inside no other value.
    def apart(part):
        return part if type(part) is str or part == FULL_NAME else (part[0], "Morgan")

    text, answer = replaced(*(apart(part) for part in parts))
    response, decision = decide(text, [apart(record) for record in records], [{"text": answer}])
    assert_forwarded(response, decision, "Jordan", "Morgan")
    assert decision.forward_text == answer and answer.count("{") == 2


@pytest.mark.parametrize(
    ("text", "records"),
    [
        (ADDRESS, [("EMAIL", ADDRESS)]),
        (ADDRESS + OTHER_ADDRESS, [("EMAIL", ADDRESS), ("EMAIL", OTHER_ADDRESS)]),
        (ADDRESS + ADDRESS, TWO_RECORDS),
    ],
    ids=["one_value", "two_values_back_to_back", "one_value_twice_with_a_record_for_each_copy"],
)
def test_text_that_is_nothing_but_its_reported_values_is_forwarded(text, records):
    """Records as long in total as the text can all be met; only a longer total is refused at once.

    The first comparison reads the first two answers as well. It refuses two records for one value, so the
    third answer is read by the second comparison alone and shows where its early refusal ends.
    """
    assert sum(len(value) for _, value in records) == len(text)
    answer = "{EMAIL}" * len(records)
    response, decision = decide(text, records, [{"text": answer}])
    assert_forwarded(response, decision, ADDRESS, OTHER_ADDRESS)
    assert decision.forward_text == answer


@pytest.mark.parametrize(
    ("masked", "records"),
    [({0, 1}, 2), ({0}, 1), ({1}, 1)],
    ids=["both_masked", "first_masked", "second_masked"],
)
def test_repeated_value_with_supplementary_characters_before_between_and_after(masked, records):
    """The walk steps over whole characters, and the coverage count stays in UTF-16 code units."""
    text = " ".join([SUPPLEMENTARY, ADDRESS, SUPPLEMENTARY + SUPPLEMENTARY, ADDRESS, SUPPLEMENTARY])
    assert utf16_units(text) == len(text) + 4
    output = mask_occurrences(text, ADDRESS, "{EMAIL}", masked)
    response, decision = decide(text, [("EMAIL", ADDRESS)] * records, [{"text": output}])
    assert response["guardrailCoverage"]["textCharacters"]["total"] == utf16_units(text)
    data = assert_forwarded(response, decision)
    assert decision.forward_text.count(SUPPLEMENTARY) == 4
    assert decision.forward_text.count("{EMAIL}") == len(masked)
    assert (data["character_count"], data["byte_count"]) == (len(text), len(text.encode()))


def test_supplementary_text_with_a_code_point_coverage_count_is_still_refused_before_the_walk():
    """A control of the coverage check, which comes before the comparison and is not part of it.

    The answer would be forwarded if the coverage total were compared with the number of code points.
    """
    text = " ".join([SUPPLEMENTARY, ADDRESS, SUPPLEMENTARY + SUPPLEMENTARY, ADDRESS, SUPPLEMENTARY])
    response = masked_answer(text, TWO_RECORDS, [{"text": text.replace(ADDRESS, "{EMAIL}")}])
    response["guardrailCoverage"]["textCharacters"] = {"guarded": len(text), "total": len(text)}
    client = RecordingGuardClient(response)
    decision = private_checker(client).check(text, boundary="TOOL_RESULT", opaque_cycle=CYCLE)
    assert (decision.outcome, decision.receipt.reason) == ("UNKNOWN", "NATIVE_INVALID")
    assert decision.forward_text is None


@pytest.mark.parametrize("change", ["dropped", "replaced_by_two_other_characters"])
def test_supplementary_character_changed_by_the_answer_is_refused(change):
    text = " ".join([SUPPLEMENTARY, ADDRESS, SUPPLEMENTARY, ADDRESS, SUPPLEMENTARY])
    good = text.replace(ADDRESS, "{EMAIL}")
    middle = " " + SUPPLEMENTARY + " "
    assert good.count(middle) == 1
    output = good.replace(middle, "  " if change == "dropped" else " ?? ")
    _, decision = decide(text, TWO_RECORDS, [{"text": output}])
    assert_refused(decision, ADDRESS)


def refusal(name, text, records, outputs):
    return pytest.param(text, records, outputs, id=name)


LONE_SURROGATE = "\ud800"  # a code point that strict UTF-8 cannot encode
REFUSED = [
    # A change outside a substitution.
    refusal("changed_character", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED.replace("today", "Today")}]),
    refusal("dropped_character", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED.replace(" today", "today")}]),
    refusal("dropped_tail", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED.removesuffix(" tomorrow.")}]),
    refusal("added_character", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED.replace("today", "today!")}]),
    refusal("added_tail", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED + " Thanks."}]),
    refusal("added_head", TWICE, TWO_RECORDS, [{"text": "Note: " + BOTH_MASKED}]),
    # The same six changes with one record for the value and every copy replaced: the form the first
    # comparison takes. It refuses them by its equality alone; with two records it never gets that far.
    refusal(
        "changed_character_with_one_record",
        TWICE,
        ONE_RECORD,
        [{"text": BOTH_MASKED.replace("today", "Today")}],
    ),
    refusal(
        "dropped_character_with_one_record",
        TWICE,
        ONE_RECORD,
        [{"text": BOTH_MASKED.replace(" today", "today")}],
    ),
    refusal(
        "dropped_tail_with_one_record", TWICE, ONE_RECORD, [{"text": BOTH_MASKED.removesuffix(" tomorrow.")}]
    ),
    refusal(
        "added_character_with_one_record",
        TWICE,
        ONE_RECORD,
        [{"text": BOTH_MASKED.replace("today", "today!")}],
    ),
    refusal("added_tail_with_one_record", TWICE, ONE_RECORD, [{"text": BOTH_MASKED + " Thanks."}]),
    refusal("added_head_with_one_record", TWICE, ONE_RECORD, [{"text": "Note: " + BOTH_MASKED}]),
    # A placeholder that does not stand for a reported value of its own category.
    refusal(
        "placeholder_where_the_text_held_other_words",
        TWICE,
        TWO_RECORDS,
        [{"text": BOTH_MASKED.replace("today", "{EMAIL}")}],
    ),
    refusal(
        "placeholder_of_an_unreported_category",
        TWICE,
        TWO_RECORDS,
        [{"text": BOTH_MASKED.replace("{EMAIL}", "{NAME}", 1)}],
    ),
    refusal(
        "placeholders_of_two_reported_categories_exchanged",
        "Jordan wrote from " + ADDRESS + ".",
        [("NAME", "Jordan"), ("EMAIL", ADDRESS)],
        [{"text": "{EMAIL} wrote from {NAME}."}],
    ),
    refusal(
        "placeholder_in_other_letters",
        TWICE,
        TWO_RECORDS,
        [{"text": BOTH_MASKED.replace("{EMAIL}", "{email}", 1)}],
    ),
    refusal(
        "placeholder_without_braces",
        TWICE,
        TWO_RECORDS,
        [{"text": BOTH_MASKED.replace("{EMAIL}", "EMAIL", 1)}],
    ),
    refusal(
        "part_of_a_value_replaced",
        TWICE,
        TWO_RECORDS,
        [{"text": TWICE.replace(ADDRESS, "{EMAIL}", 1).replace("jordan@", "{EMAIL}@")}],
    ),
    refusal(
        "more_than_the_value_replaced",
        "Write to " + ADDRESS + " today.",
        [("EMAIL", ADDRESS)],
        [{"text": "Write to {EMAIL}."}],
    ),
    refusal(
        "value_and_placeholder_changed_places",
        FORM,
        [("EMAIL", ADDRESS)],
        [{"text": "The form shows " + ADDRESS + " where {EMAIL} was typed."}],
    ),
    # Fewer substitutions than records.
    refusal(
        "fewer_substitutions_than_records",
        TWICE,
        TWO_RECORDS,
        [{"text": TWICE.replace(ADDRESS, "{EMAIL}", 1)}],
    ),
    refusal(
        "more_records_than_occurrences", TWICE, [*TWO_RECORDS, ("EMAIL", ADDRESS)], [{"text": BOTH_MASKED}]
    ),
    refusal(
        "literal_placeholder_counted_as_a_substitution",
        FORM,
        TWO_RECORDS,
        [{"text": FORM.replace(ADDRESS, "{EMAIL}")}],
    ),
    refusal("records_longer_in_total_than_the_text", ADDRESS, TWO_RECORDS, [{"text": "{EMAIL}"}]),
    # Absent or nested values.
    refusal("absent_value", TWICE, [*TWO_RECORDS, ("EMAIL", OTHER_ADDRESS)], [{"text": BOTH_MASKED}]),
    refusal(
        "absent_value_beside_a_value_with_one_record",
        "Write to " + ADDRESS + " today.",
        [("EMAIL", ADDRESS), ("EMAIL", OTHER_ADDRESS)],
        [{"text": "Write to {EMAIL} today."}],
    ),
    refusal(
        "value_inside_another_value",
        "jordan wrote from " + ADDRESS + ".",
        [("EMAIL", ADDRESS), ("NAME", "jordan")],
        [{"text": "{NAME} wrote from {EMAIL}."}],
    ),
    refusal(
        "value_inside_another_value_reported_first",
        "jordan wrote from " + ADDRESS + ".",
        [("NAME", "jordan"), ("EMAIL", ADDRESS)],
        [{"text": "{NAME} wrote from {EMAIL}."}],
    ),
    refusal(
        "value_inside_another_value_with_a_record_between",
        "jordan wrote from " + ADDRESS + " to " + OTHER_ADDRESS + ".",
        [("NAME", "jordan"), ("EMAIL", OTHER_ADDRESS), ("EMAIL", ADDRESS)],
        [{"text": "{NAME} wrote from {EMAIL} to {EMAIL}."}],
    ),
    refusal(
        "same_value_under_two_categories",
        "Call " + NUMBER + " now, or " + NUMBER + " later.",
        [(BANK, NUMBER), ("PHONE", NUMBER)],
        [{"text": "Call {" + BANK + "} now, or {PHONE} later."}],
    ),
    # The output itself.
    refusal("unchanged_output", TWICE, TWO_RECORDS, [{"text": TWICE}]),
    # Placeholder text that the checked text holds, reported as a match: replacing it changes nothing.
    refusal(
        "unchanged_output_with_a_value_that_is_its_own_placeholder",
        FORM,
        [("EMAIL", "{EMAIL}")],
        [{"text": FORM}],
    ),
    refusal("no_output", TWICE, TWO_RECORDS, []),
    refusal("two_outputs", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED}, {"text": BOTH_MASKED}]),
    refusal("outputs_that_are_not_a_list", TWICE, TWO_RECORDS, ({"text": BOTH_MASKED},)),
    refusal("output_entry_that_is_not_a_mapping", TWICE, TWO_RECORDS, [["text"]]),
    refusal("output_with_a_second_key", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED, "note": "ordinary"}]),
    refusal("output_that_is_not_text", TWICE, TWO_RECORDS, [{"text": [BOTH_MASKED]}]),
    refusal("empty_output", TWICE, TWO_RECORDS, [{"text": ""}]),
    refusal(
        "output_with_a_surrogate_code_point", TWICE, TWO_RECORDS, [{"text": BOTH_MASKED + LONE_SURROGATE}]
    ),
]


@pytest.mark.parametrize(("text", "records", "outputs"), REFUSED)
def test_answer_that_is_not_a_substitution_only_edit_is_refused(text, records, outputs):
    _, decision = decide(text, records, outputs)
    assert_refused(decision, ADDRESS, OTHER_ADDRESS, NUMBER)


# Template text that itself reads as an address: a braced word, a full stop, a braced word, the domain.
TEMPLATE = "{first}.{last}@example.com"
TEMPLATE_END = TEMPLATE[TEMPLATE.rindex("}") :]
PATTERN = "Name a mailbox by the pattern " + TEMPLATE + " and note it."
ASK = " Ask " + ADDRESS + " first; " + ADDRESS + " decides."
ONE_RECORD_EACH = [("EMAIL", TEMPLATE), ("EMAIL", ADDRESS)]
BRACED_WORD = "The key {draft} is set by " + ADDRESS + "."


def content_list(text):
    """The compact JSON of a content list with one text block, the form a tool result is checked in."""
    return json.dumps([{"text": text}], ensure_ascii=False, separators=(",", ":"))


def every_copy_masked(text, records):
    """``text`` with every copy of every reported value replaced by its placeholder, the longest first."""
    for category, value in sorted(records, key=lambda record: len(record[1]), reverse=True):
        text = text.replace(value, "{" + category + "}")
    return text


@pytest.mark.parametrize(
    ("text", "records"),
    [
        (PATTERN, [("EMAIL", TEMPLATE)]),
        (PATTERN + ASK, ONE_RECORD_EACH),
        (BRACED_WORD, [("EMAIL", ADDRESS), ("USERNAME", "{draft}")]),
    ],
    ids=["template_alone", "template_beside_a_value_that_stands_twice", "braced_word_beside_another_value"],
)
def test_reported_value_that_begins_with_an_opening_brace_is_forwarded(text, records):
    """One record for each value and every copy replaced: the first comparison reads the answer.

    The second comparison steps over the opening brace as equal text and never reads such a value as
    replaced, so these answers supply no text without the first comparison.
    """
    assert any(value.startswith("{") for _, value in records) and len(set(records)) == len(records)
    answer = every_copy_masked(text, records)
    response, decision = decide(text, records, [{"text": answer}])
    assert_forwarded(response, decision, *(value for _, value in records))
    assert decision.forward_text == answer
    assert all(value not in answer for _, value in records)


@pytest.mark.parametrize(
    ("text", "records"),
    [
        (PATTERN, [("EMAIL", TEMPLATE_END)]),
        (PATTERN + ASK, [("EMAIL", TEMPLATE_END), ("EMAIL", ADDRESS)]),
        (PATTERN + ASK, [("EMAIL", TEMPLATE_END), *TWO_RECORDS]),
    ],
    ids=["template_alone", "beside_a_value_that_stands_twice", "beside_a_value_with_a_record_for_each_copy"],
)
def test_reported_value_that_begins_at_a_closing_brace_in_a_content_list_is_forwarded(text, records):
    """Inside a content list the match for the template began at its last closing brace.

    Both comparisons read the first two answers. The first refuses two records for one value, so the
    second comparison alone reads the third.
    """
    assert TEMPLATE_END.startswith("}") and TEMPLATE.endswith(TEMPLATE_END)
    checked = content_list(text)
    answer = every_copy_masked(checked, records)
    response, decision = decide(checked, records, [{"text": answer}])
    assert_forwarded(response, decision, TEMPLATE_END, ADDRESS)
    assert decision.forward_text == answer
    # The masked content list is still one text block; the template's braced words stay as they were.
    assert json.loads(answer) == [{"text": every_copy_masked(text, records)}]
    assert answer.count("{EMAIL}") == text.count(TEMPLATE_END) + text.count(ADDRESS)
    assert "{first}.{last{EMAIL}" in answer


def test_overlapping_values_with_only_one_of_them_replaced_are_forwarded():
    """Two reported values overlap in the text and the service replaced the longer one.

    The first comparison reads the answer: once the longer value is replaced, the other no longer stands
    in the text. The second refuses it, because one record is left without a replacement of its own.
    """
    text = "See " + ADDRESS + "/team today."
    records = [("EMAIL", ADDRESS), ("URL", "example.com/team")]
    response, decision = decide(text, records, [{"text": "See {EMAIL}/team today."}])
    assert_forwarded(response, decision, ADDRESS)
    assert decision.forward_text == every_copy_masked(text, records) == "See {EMAIL}/team today."


EQUAL_LENGTH = ADDRESS[ADDRESS.index("@") + 1 :] + "/team/a"  # as long as the address
ACROSS = "See " + ADDRESS + "/team/a today."  # the address, and the other value across its end


@pytest.mark.parametrize(
    "address_first", [True, False], ids=["address_reported_first", "path_reported_first"]
)
def test_overlapping_values_of_equal_length_are_read_in_record_order(address_first):
    """Two reported values of equal length overlap in the text, so only one of them can be replaced.

    The first comparison replaces the longest value first and, of two values of equal length, the one
    reported first. It reads the answer that replaced that value and no other. The second comparison
    reads neither answer: one record is left without a replacement of its own.
    """
    address, path = ("EMAIL", ADDRESS), ("URL", EQUAL_LENGTH)
    assert len(ADDRESS) == len(EQUAL_LENGTH) and ADDRESS not in EQUAL_LENGTH and EQUAL_LENGTH not in ADDRESS
    assert ACROSS.count(ADDRESS) == ACROSS.count(EQUAL_LENGTH) == 1
    answers = {address: ACROSS.replace(ADDRESS, "{EMAIL}"), path: ACROSS.replace(EQUAL_LENGTH, "{URL}")}
    assert answers[address] != answers[path] and "{URL}" not in answers[address]
    reported_first, reported_second = (address, path) if address_first else (path, address)
    records = [reported_first, reported_second]
    response, decision = decide(ACROSS, records, [{"text": answers[reported_first]}])
    assert_forwarded(response, decision, ADDRESS, EQUAL_LENGTH)
    assert decision.forward_text == answers[reported_first]
    _, decision = decide(ACROSS, records, [{"text": answers[reported_second]}])
    assert_refused(decision, ADDRESS, EQUAL_LENGTH)


@pytest.mark.parametrize(
    ("records", "copies"),
    [([("EMAIL", TEMPLATE), *TWO_RECORDS], 2), (ONE_RECORD_EACH, 1)],
    ids=["a_record_for_each_copy_of_the_other_value", "one_copy_of_the_other_value_left"],
)
def test_answer_that_neither_comparison_reads_supplies_no_text(records, copies):
    """A value that begins with an opening brace beside a repeated value in the second comparison's form.

    The first comparison refuses two records for one value and a copy left unreplaced. The second never
    reads a value that begins with an opening brace as replaced. Neither reads the answer, although it
    differs from the checked text only by placeholders for reported values.
    """
    text = PATTERN + ASK
    answer = text.replace(TEMPLATE, "{EMAIL}").replace(ADDRESS, "{EMAIL}", copies)
    assert answer.count("{EMAIL}") == 1 + copies and answer.count(ADDRESS) == 2 - copies
    _, decision = decide(text, records, [{"text": answer}])
    assert_refused(decision, ADDRESS, TEMPLATE)


def sized(filler, masked_bytes):
    """A text of ``filler`` characters and the number, and its masked form of ``masked_bytes`` bytes."""
    room = masked_bytes - len(" " + BANK_PLACEHOLDER)
    width = len(filler.encode())
    text = filler * (room // width) + "y" * (room % width) + " " + NUMBER
    masked = text.replace(NUMBER, BANK_PLACEHOLDER)
    assert len(masked.encode()) == masked_bytes and len(text.encode()) <= LIMIT
    return text, masked


TWO_BYTES = "\u00e9"  # one character of two UTF-8 bytes and one UTF-16 code unit


@pytest.mark.parametrize("filler", ["x", TWO_BYTES], ids=["one_byte_characters", "two_byte_characters"])
def test_masked_text_is_forwarded_at_the_size_limit_and_refused_one_byte_above_it(filler):
    """The limit is on the UTF-8 bytes of the masked form; the checked text fits in every case."""
    # At the limit the masked form is forwarded.
    text, masked = sized(filler, LIMIT)
    response, decision = decide(text, [(BANK, NUMBER)], [{"text": masked}])
    assert_forwarded(response, decision, NUMBER)
    assert decision.forward_text == masked
    # One byte more is refused. With two-byte characters the form is far below the limit in characters,
    # so only the count of bytes refuses it.
    text, masked = sized(filler, LIMIT + 1)
    if filler == TWO_BYTES:
        assert len(masked) < LIMIT // 2 + len(BANK_PLACEHOLDER) + 2
    else:
        assert len(masked) == LIMIT + 1
    _, decision = decide(text, [(BANK, NUMBER)], [{"text": masked}])
    assert_refused(decision, NUMBER)


def test_text_at_the_size_limit_whose_masked_form_is_longer_is_refused():
    """The checked text is exactly at the limit and is sent; its masked form is above the limit."""
    text = "x" * (LIMIT - len(NUMBER) - 1) + " " + NUMBER
    masked = text.replace(NUMBER, BANK_PLACEHOLDER)
    assert len(text) == len(text.encode()) == LIMIT
    assert len(masked) == LIMIT + len(BANK_PLACEHOLDER) - len(NUMBER) > LIMIT
    _, decision = decide(text, [(BANK, NUMBER)], [{"text": masked}])
    assert_refused(decision, NUMBER)


def test_literal_placeholder_in_the_original_is_no_substitution():
    """Placeholder text that the checked text already held is stepped over like any other text."""
    response, decision = decide(FORM, [("EMAIL", ADDRESS)], [{"text": FORM.replace(ADDRESS, "{EMAIL}")}])
    assert_forwarded(response, decision, ADDRESS)
    assert decision.forward_text.count("{EMAIL}") == 2


def test_value_that_is_its_own_placeholder_is_read_by_the_first_comparison():
    """The checked text holds placeholder text, and the answer reports that text as a match of its own.

    The first comparison replaces the match by the same text, so its copy still stands in the output and
    the answer is read, in either record order. The second comparison steps over the match as equal text
    and leaves its record without a replacement.
    """
    answer = FORM.replace(ADDRESS, "{EMAIL}")
    for records in itertools.permutations([("EMAIL", "{EMAIL}"), ("EMAIL", ADDRESS)]):
        response, decision = decide(FORM, list(records), [{"text": answer}])
        assert_forwarded(response, decision, ADDRESS)
        assert decision.forward_text == answer and answer.count("{EMAIL}") == 2


MIXED = "Jordan wrote from " + ADDRESS + " about " + NUMBER + ", then again from " + ADDRESS + "."
MIXED_RECORDS = [("EMAIL", ADDRESS), ("NAME", "Jordan"), ("EMAIL", ADDRESS), (BANK, NUMBER)]
MIXED_MASKED = "{NAME} wrote from {EMAIL} about {" + BANK + "}, then again from {EMAIL}."
ORDERS = sorted(set(itertools.permutations(MIXED_RECORDS)))


def test_record_order_does_not_change_the_decision():
    assert len(ORDERS) == 12
    for order in ORDERS:
        response, decision = decide(MIXED, list(order), [{"text": MIXED_MASKED}])
        assert_forwarded(response, decision, ADDRESS, NUMBER)
        _, decision = decide(MIXED, list(order), [{"text": MIXED_MASKED.replace("{NAME}", "Jordan")}])
        assert_refused(decision, ADDRESS, NUMBER)


def test_name_inside_an_inserted_token_does_not_corrupt_the_check():
    """A reported value that also stands inside a placeholder's own letters is not searched for there."""
    text = "MAIL wrote from " + ADDRESS + "."
    response, decision = decide(
        text, [("EMAIL", ADDRESS), ("NAME", "MAIL")], [{"text": "{NAME} wrote from {EMAIL}."}]
    )
    assert_forwarded(response, decision, ADDRESS)


def test_unchanged_output_supplies_no_text_even_without_a_record():
    """The checker never calls the comparison without a record; the comparison refuses that call itself."""
    source = importlib.import_module(MODULE)
    assert source._service_mask({"outputs": [{"text": TWICE}]}, TWICE, []) is None
    assert source._service_mask({"outputs": [{"text": BOTH_MASKED}]}, TWICE, []) is None


def test_records_longer_in_total_than_the_text_are_refused_before_the_values_are_compared():
    """Every record needs a part of the text of its own, so a longer total is refused at once."""
    source = importlib.import_module(MODULE)

    class Counted(str):
        """A value that counts how often another value is searched for inside it."""

        searches = 0

        def __contains__(self, other):
            Counted.searches += 1
            return str.__contains__(self, other)

    # The control: two values that fit are compared with each other, and the answer is accepted.
    fitting = [("EMAIL", Counted("first@example.com")), ("EMAIL", Counted("second@example.com"))]
    text = "From first@example.com to second@example.com."
    output = "From {EMAIL} to {EMAIL}."
    assert source._service_mask({"outputs": [{"text": output}]}, text, fitting) is output
    assert Counted.searches > 0
    # Four records of 62 characters each cannot all be met in a text of 64 characters.
    Counted.searches = 0
    long_values = [("EMAIL", Counted("a" * 60 + f"{index:02d}")) for index in range(4)]
    assert source._service_mask({"outputs": [{"text": "{EMAIL}"}]}, "x" * 64, long_values) is None
    assert Counted.searches == 0


# Generated answers: what the first comparison reads is forwarded, and nothing else but a substitution.

SOCIAL = "-".join(["9" * 3, "9" * 2, "9" * 4])  # built here, like the number
CARD = " ".join(["4" * 4] * 4)
ROUTE = ".".join(["192", "0", "2", "44"])
DEFAULT_KINDS = {
    "EMAIL",
    "PHONE",
    "NAME",
    "US_SOCIAL_SECURITY_NUMBER",
    "CREDIT_DEBIT_CARD_NUMBER",
    BANK,
    "IP_ADDRESS",
}
POOL = [
    ("EMAIL", ADDRESS),
    ("EMAIL", OTHER_ADDRESS),
    ("EMAIL", TEMPLATE),  # begins with an opening brace
    ("EMAIL", TEMPLATE_END),  # begins with a closing brace
    ("EMAIL", "{user}@example.com"),
    ("EMAIL", "first.{last}@example.com"),  # holds braces
    ("EMAIL", "{EMAIL}"),  # its own placeholder
    ("PHONE", "555-0100"),
    ("PHONE", NUMBER[:9]),  # the same value under two kinds
    ("NAME", "Jordan Lee"),
    ("NAME", "Jordan"),  # inside another value
    ("NAME", "Lee"),
    ("NAME", "MAIL"),  # letters of a placeholder
    ("NAME", "nana"),  # overlaps itself in "nanana"
    ("NAME", "anan"),  # as long as "nana", and across it in "nanana"
    ("NAME", "{draft}"),
    ("NAME", "example.com/team"),  # overlaps an address that "/team" follows
    ("US_SOCIAL_SECURITY_NUMBER", SOCIAL),
    ("CREDIT_DEBIT_CARD_NUMBER", CARD),
    (BANK, NUMBER),
    (BANK, NUMBER[:9]),
    ("IP_ADDRESS", ROUTE),
    ("IP_ADDRESS", ROUTE[4:]),
]
WORDS = [
    " ",
    " and ",
    ", then ",
    ". ",
    "/team ",
    " wrote to ",
    "nanana ",
    "{",
    "}",
    "{} ",
    "{EMAIL} ",  # placeholder text that the checked text itself holds
    "{NAME}",
    "{URL} ",
    "MAIL ",
    SUPPLEMENTARY,
    " " + SUPPLEMENTARY + " ",
]
# Two values of equal length that stand across one another in "nanana": only one of them can be replaced.
EQUAL_LENGTH_PAIR = [("NAME", "nana"), ("NAME", "anan")]


def generated_answer(rng):
    """A text, its records and a service answer, all from the test's own values and words.

    The text holds one to four planted values of the seven default kinds, and in half of the texts one of
    them once more: values that begin with or hold a brace, values inside or across one another, a value
    that is its own placeholder, placeholder text and characters outside the Basic Multilingual Plane. One
    text in ten also holds two values of equal length across one another. One text in five is the compact
    JSON of a content list. The answer replaces every copy of the reported values, or copies found from
    left to right. For three answers in ten one more change is then tried: a character dropped or
    exchanged, a word added inside, at the end or at the start, or the first address placeholder exchanged
    for another kind's. The records stand once for each value, once for each replaced copy, or one fewer;
    one answer in ten has a record more.
    """

    def pick(items):
        return items[rng.randrange(len(items))]

    planted = [pick(POOL) for _ in range(1 + rng.randrange(4))]
    if rng.random() < 0.5:
        planted.append(pick(planted))
    text = pick(WORDS) + "".join(value + pick(WORDS) for _, value in planted)
    if rng.random() < 0.1:
        planted += EQUAL_LENGTH_PAIR
        text += "nanana" + pick(WORDS)
    if rng.random() < 0.2:
        text = content_list(text)
    reported = [record for record in POOL if record[1] in text and (record in planted or rng.random() < 0.2)]
    copies = Counter()
    if rng.random() < 0.4:
        answer = every_copy_masked(text, reported)
        copies.update({record: text.count(record[1]) for record in reported})
    else:
        share, parts, at = pick([1.0, 0.8, 0.5]), [], 0
        while at < len(text):
            fits = [record for record in reported if text.startswith(record[1], at)]
            if fits and rng.random() < share:
                category, value = pick(fits)
                copies[category, value] += 1
                parts.append("{" + category + "}")
                at += len(value)
            else:
                parts.append(text[at])
                at += 1
        answer = "".join(parts)
    form = rng.randrange(3)
    records = list(reported) if form == 0 else [record for record in reported for _ in range(copies[record])]
    if form == 2 and records:
        del records[rng.randrange(len(records))]
    if rng.random() < 0.1:
        records.append(pick(POOL))
    if not records:
        records.append(pick(reported))
    if rng.random() < 0.5:
        records.reverse()
    if rng.random() < 0.3 and answer:
        at, change = rng.randrange(len(answer)), rng.randrange(6)
        if change == 0:
            answer = answer[:at] + answer[at + 1 :]
        elif change == 1:
            answer = answer[:at] + pick(WORDS) + answer[at:]
        elif change == 2:
            answer = answer[:at] + "x" + answer[at + 1 :]
        elif change == 3:
            answer = answer.replace("{EMAIL}", "{NAME}", 1)
        elif change == 4:
            answer += pick(WORDS)
        else:
            answer = pick(WORDS) + answer
    return text, records, answer


def first_comparison_expects(text, records):
    """The text that the restated first comparison expects, or None where it reads no answer at all."""
    values = [value for _, value in records]
    if len(set(records)) != len(records) or any(value not in text for value in values):
        return None
    if any(one in other or other in one for one, other in itertools.combinations(values, 2)):
        return None
    return every_copy_masked(text, records)


def first_comparison_reads(text, records, answer):
    """The first comparison, restated: one record for each value, and every copy of every value replaced."""
    return bool(answer) and answer != text and answer == first_comparison_expects(text, records)


def substitution_only_edit(text, records, answer, equal_text_first=False):
    """A complete search: ``answer`` is ``text`` with copies of reported values replaced by their placeholder.

    Nothing else may differ, no reported value may stand inside another, and every value needs at least as
    many replacements as records. The search tries equal text and every replacement at every place. With
    ``equal_text_first`` it restates the second comparison: a replacement is tried only where the two
    texts differ.
    """
    need = Counter(records)
    pairs = sorted(need)
    values = [value for _, value in pairs]
    if not answer or answer == text:
        return False
    if any(one in other or other in one for one, other in itertools.combinations(values, 2)):
        return False
    start = (0, 0, (0,) * len(pairs))
    seen, stack = {start}, [start]
    while stack:
        i, j, done = stack.pop()
        if j == len(answer):
            if i == len(text) and all(count >= need[pair] for count, pair in zip(done, pairs, strict=True)):
                return True
            continue
        equal = i < len(text) and text[i] == answer[j]
        steps = [(i + 1, j + 1, done)] if equal else []
        for index, (category, value) in enumerate(() if equal and equal_text_first else pairs):
            token = "{" + category + "}"
            if answer.startswith(token, j) and text.startswith(value, i):
                count = min(done[index] + 1, need[category, value])
                steps.append((i + len(value), j + len(token), (*done[:index], count, *done[index + 1 :])))
        for step in steps:
            if step not in seen:
                seen.add(step)
                stack.append(step)
    return False


GENERATED = 4000


def test_generated_answers_that_the_first_comparison_reads_are_all_forwarded():
    """No answer that the first comparison reads may be refused, whatever else the checker learns to read.

    The same answers show two more things. The checker forwards an answer exactly when the restated first
    comparison or the restated second comparison reads it. And the second comparison reads every
    substitution-only edit that a complete search finds, unless a reported value begins with an opening
    brace.
    """
    source = importlib.import_module(MODULE)
    assert {kind for kind, _ in POOL} == DEFAULT_KINDS
    rng = random.Random(14)  # nosec B311 - seeded test data, not a secret
    seen = Counter()
    for _ in range(GENERATED):
        text, records, answer = generated_answer(rng)
        case = (text, records, answer)
        response = {"outputs": [{"text": answer}]}
        forwarded = source._service_mask(response, text, list(records))
        expected = first_comparison_expects(text, records)
        first = first_comparison_reads(text, records, answer)
        second = substitution_only_edit(text, records, answer, equal_text_first=True)
        edit = substitution_only_edit(text, records, answer)
        brace = any(value.startswith("{") for _, value in records)
        longer = expected is not None and answer != expected
        order = expected is not None and expected != first_comparison_expects(text, records[::-1])
        assert forwarded is None or forwarded is response["outputs"][0]["text"], case
        assert forwarded is not None or not first, case
        assert (forwarded is not None) == (first or second), case
        assert second or not edit or brace, case
        seen["first comparison reads"] += first
        seen["first comparison reads, a value begins with a brace"] += first and brace
        seen["first comparison reads, no substitution-only edit"] += first and not edit
        seen["forwarded beside the first comparison"] += forwarded is not None and not first
        seen["refused"] += forwarded is None
        seen["two records for one value"] += len(set(records)) != len(records)
        seen["content list"] += text.startswith("[{")
        seen["outside the plane"] += SUPPLEMENTARY in text
        seen["the record order decides, read"] += order and first
        seen["the record order decides, refused"] += order and forwarded is None
        seen["a value that is its own placeholder, read"] += first and ("EMAIL", "{EMAIL}") in records
        seen["more after the first comparison's text"] += longer and answer.startswith(expected)
        seen["more before the first comparison's text"] += longer and answer.endswith(expected)
    # The generator reaches every class of answer often enough for the checks above to mean something.
    assert seen["first comparison reads"] >= 600 and seen["forwarded beside the first comparison"] >= 240
    assert seen["first comparison reads, a value begins with a brace"] >= 150
    assert seen["first comparison reads, no substitution-only edit"] >= 20
    assert seen["refused"] >= 1500 and seen["two records for one value"] >= 600
    assert seen["content list"] >= 450 and seen["outside the plane"] >= 1000
    assert seen["the record order decides, read"] >= 15
    assert seen["the record order decides, refused"] >= 50
    assert seen["a value that is its own placeholder, read"] >= 50
    assert seen["more after the first comparison's text"] >= 45
    assert seen["more before the first comparison's text"] >= 40


# Full turns: the real backend callbacks and SDK, a scripted model and a scripted guard.

QUESTION = "ordinary user question"
SELECTED = "ordinaryselected"  # the guard identity that the wiring fixture selects
HOLD_TEXT = "Model input was held at the admission boundary."
ADMISSION_HOLD = "AMLAS_INPUT_HOLD phase=input_admission reason=admission_refused"
ALLOWED = ("ALLOW", "NATIVE_NONE")
MASKED = ("ANONYMIZED", "NATIVE_MASKED")
REQUEST = "fixture-request"


class MaskingGuard(wiring.ClientFixture):
    """NONE for a text without the number; else ``records`` records and the chosen copies masked."""

    def __init__(self, records, masked):
        super().__init__(deny=lambda text: False)
        self.records, self.masked = records, masked

    def apply_guardrail(self, **kwargs):
        self.calls.append(kwargs)
        text = kwargs["content"][0]["text"]["text"]
        if NUMBER not in text:
            return none_answer(text, SELECTED)
        output = mask_occurrences(text, NUMBER, "{" + BANK + "}", self.masked)
        return masked_answer(text, [(BANK, NUMBER)] * self.records, [{"text": output}], SELECTED)


class ReadToolModel(wiring.ModelFixture):
    """One call of the job-read tool, then a final text."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        if self.calls == 1:
            use = {"toolUseId": "ordinary-read", "name": "get_coding_preparation"}
            arguments = json.dumps({"request_ref": REQUEST})
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {"toolUse": use}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"toolUse": {"input": arguments}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            assert self.calls == 2
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ordinary complete"}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}


@tool
def get_coding_preparation(request_ref: str) -> dict:
    """Return the fixture's stored record of one job."""
    record = {"request_ref": request_ref, "job_arn": IDENTIFIER, "note": "Ran in account " + NUMBER + "."}
    return {"status": "success", "content": [{"json": record}]}


def turn(monkeypatch, caplog, *, guard, history=None, question=QUESTION):
    wiring.configure_guard(monkeypatch, guard)
    monkeypatch.setattr(binding, "get_safe_llm_tools", lambda: [get_coding_preparation])
    model = ReadToolModel()
    backend = binding.ToolBindingBackend(Settings(), model=model)
    agents, build = [], backend.build_agent

    def built(**kwargs):
        agents.append(build(**kwargs))
        agents[-1].callback_handler = null_callback_handler
        return agents[-1]

    monkeypatch.setattr(backend, "build_agent", built)
    with caplog.at_level(logging.WARNING):
        result = backend.chat_with_tools(question, history=history)
    admission = result.input_admission or {}
    return {
        "turn": result,
        "backend": backend,
        "model": model,
        "agents": agents,
        "status": admission.get("status"),
        "records": [(r["boundary"], r["outcome"], r["reason"]) for r in admission.get("records", [])],
        "tool_names": [r["tool_name"] for r in admission.get("records", [])],
        "hold_lines": [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("AMLAS_INPUT_HOLD")
        ],
    }


@pytest.mark.parametrize(
    ("records", "masked"),
    [(2, {0, 1}), (1, {1}), (1, {0})],
    ids=["two_records_both_copies_masked", "one_record_the_copy_in_the_note_masked", "one_record_first_copy"],
)
def test_repeated_value_in_a_read_tool_result_is_replaced_and_the_turn_goes_on(
    monkeypatch, caplog, records, masked
):
    guard = MaskingGuard(records, masked)
    seen = turn(monkeypatch, caplog, guard=guard)
    # The turn is not held: the model is called again and gives its final text.
    assert seen["turn"].final_text == "ordinary complete" and seen["model"].calls == 2
    assert seen["status"] == "admitted" and seen["backend"]._input_hold is False
    assert seen["records"] == [("OWNER_PROMPT", *ALLOWED), ("TOOL_RESULT", *MASKED)]
    assert seen["tool_names"] == ["UNKNOWN", "get_coding_preparation"]
    assert seen["hold_lines"] == []
    # The guard was shown the tool result once, as the compact JSON of its content list.
    shown = [call["content"][0]["text"]["text"] for call in guard.calls]
    assert len(shown) == 2 and shown[0] == QUESTION and shown[1].count(NUMBER) == 2
    # The committed result keeps its shape and holds the service's substitution.
    results = [
        block["toolResult"]
        for message in seen["agents"][0].messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert len(results) == 1 and results[0]["status"] == "success"
    assert results[0]["toolUseId"] == "ordinary-read"
    (block,) = results[0]["content"]
    assert set(block) == {"json"} and set(block["json"]) == {"request_ref", "job_arn", "note"}
    assert block["json"]["request_ref"] == REQUEST
    expected = mask_occurrences(shown[1], NUMBER, "{" + BANK + "}", masked)
    assert json.dumps(results[0]["content"], ensure_ascii=False, separators=(",", ":")) == expected
    # The model's second call received that result: every masked copy as the placeholder.
    sent = json.dumps(seen["model"].messages[-1], ensure_ascii=False)
    assert sent.count("{" + BANK + "}") == len(masked) and sent.count(NUMBER) == 2 - len(masked)
    assert NUMBER not in json.dumps(seen["turn"].input_admission)


@pytest.mark.parametrize("copies", [1, 2], ids=["value_once", "value_twice"])
def test_repeated_value_in_an_immutable_text_still_holds(monkeypatch, caplog, copies):
    """A text that must stay unchanged holds when the service masks it; its receipt reads as a mask."""
    earlier = "Earlier I gave " + " and again ".join([NUMBER] * copies) + "."
    history = [
        {"role": "user", "content": [{"text": earlier}]},
        {"role": "assistant", "content": [{"text": "ordinary earlier reply"}]},
    ]
    before = copy.deepcopy(history)
    guard = MaskingGuard(copies, set(range(copies)))
    seen = turn(monkeypatch, caplog, guard=guard, history=history)
    assert seen["turn"].final_text == HOLD_TEXT
    assert seen["status"] == "held" and seen["backend"]._input_hold is True
    # The earlier turn is checked as its texts: its role name passes, then its text is masked. The receipt
    # says what the service did, a mask; the hold is the caller's rule for unchanged text.
    assert seen["records"] == [
        ("OWNER_PROMPT", *ALLOWED),
        ("HISTORY_TEXT", *ALLOWED),
        ("HISTORY_TEXT", *MASKED),
    ]
    assert seen["hold_lines"] == [ADMISSION_HOLD]
    assert seen["model"].calls == 0 and seen["agents"] == []
    assert history == before
    assert NUMBER not in json.dumps(seen["turn"].input_admission)


def test_repeated_value_in_the_owner_prompt_is_masked_and_the_turn_goes_on(monkeypatch, caplog):
    question = "Use account " + NUMBER + " for the job and " + NUMBER + " for the report."
    guard = MaskingGuard(2, {0, 1})
    seen = turn(monkeypatch, caplog, guard=guard, question=question)
    assert seen["turn"].final_text == "ordinary complete" and seen["model"].calls == 2
    assert seen["status"] == "admitted" and seen["backend"]._input_hold is False
    assert seen["records"] == [("OWNER_PROMPT", *MASKED), ("TOOL_RESULT", *MASKED)]
    assert seen["hold_lines"] == []
    # The guard was shown the prompt as given; the model's first call received the service's substitution.
    assert guard.calls[0]["content"] == [{"text": {"text": question}}]
    first = json.dumps(seen["model"].messages[0], ensure_ascii=False)
    assert first.count("{" + BANK + "}") == 2 and NUMBER not in first
    assert NUMBER not in json.dumps(seen["agents"][0].messages, ensure_ascii=False)


PAGE = "Contact " + ADDRESS + " about the corpus. Send corrections to " + ADDRESS + " as well."


class RepeatedAddressGuard(research.CoverageClient):
    """The research turn's scripted guard, with one EMAIL record for every copy of the address."""

    def apply_guardrail(self, **kwargs):
        text = kwargs["content"][0]["text"]["text"]
        response = super().apply_guardrail(**kwargs)
        if ADDRESS in text:
            response["action"] = "GUARDRAIL_INTERVENED"
            response["outputs"] = [{"text": text.replace(ADDRESS, "{EMAIL}")}]
            response["assessments"][0]["sensitiveInformationPolicy"] = {
                "piiEntities": [
                    {"type": "EMAIL", "action": "ANONYMIZED", "detected": True, "match": ADDRESS}
                    for _ in range(text.count(ADDRESS))
                ],
                "regexes": [],
            }
        return response


def test_repeated_value_in_a_research_page_is_masked_and_the_turn_goes_on(monkeypatch, caplog):
    """A fetched page that names one address twice: the research step forwards the masked page."""
    guard = RepeatedAddressGuard()
    pages = {True: PAGE, False: research.SIGN_PAGE}
    seen = research.turn(monkeypatch, caplog, pages=lambda url: pages[url == TRAIN_URL], client=guard)
    selected, state = seen["selected"], seen["state"]
    shown = [call["content"][0]["text"]["text"] for call in guard.calls]
    assert [text.count(ADDRESS) for text in shown if ADDRESS in text] == [2]
    # The turn is not held: the masked page is forwarded and the ordinary tool after research runs.
    assert selected.model.calls == 5 and len(selected.fetches) == 5
    assert seen["status"] == "admitted" and selected.backends[0]._input_hold is False
    assert seen["records"] == [
        ("OWNER_PROMPT", *ALLOWED),
        ("TOOL_RESULT", *ALLOWED),
        ("TOOL_RESULT", *MASKED),
        ("TOOL_RESULT", *ALLOWED),
        ("TOOL_RESULT", *ALLOWED),
    ]
    assert seen["results"] == research.COMPLETED and seen["hold_lines"] == []
    assert state["phase"] == "CONSUMED" and state["pending"] is False
    # A masked unit is forwarded; only an unchanged strict NONE counts as admitted source evidence.
    assert sorted(state["attempted"]) == sorted(state["forwarded"]) == research.UNITS
    assert sorted(state["admitted"]) == ["benchmark-card", "model-card"] and state["blocked"] == set()
    for stored in (selected.agents[0].messages, selected.model.messages):
        text = json.dumps(stored, ensure_ascii=False)
        assert text.count("{EMAIL}") >= 2 and ADDRESS not in text
