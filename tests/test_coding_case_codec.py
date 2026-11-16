"""Pure data-only coding-case freeze, original-field replay and role isolation."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from amlas.data_eng.coding_case_codec import freeze_humaneval_cases
from amlas.data_eng.coding_case_manifest import validate_coding_case_bundle
from amlas.data_eng.coding_case_profile import CodingCaseSource

SOURCE_VALUES = {
    "repo_id": "openai/openai_humaneval",
    "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",  # pragma: allowlist secret (public source pin)
    "filename": "openai_humaneval/test-00000-of-00001.parquet",
    "sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",  # pragma: allowlist secret (public source pin)
    "byte_count": 83920,
    "format": "parquet",
    "license": "MIT",
    "card_sha256": "f81cb59f7dae73d1ee1069d7de3ba1c0f9d2bd80f4df6d6990d12bf488e5b5ef",  # pragma: allowlist secret (public source pin)
}  # public source pins; pragma: allowlist secret


def source(**changes):
    return CodingCaseSource(**{**SOURCE_VALUES, **changes})


def rows():
    return [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": f'def synthetic_{i}():\n    """Synthetic public fixture {i}."""\n',
            "entry_point": f"synthetic_{i}",
            "canonical_solution": f"    return {i}  # PRIVATE_REFERENCE_{i}\n",
            "test": f"def check(candidate):\n    assert candidate() == {i}  # PRIVATE_TEST_{i}\n",
        }
        for i in range(164)
    ]


def obj(raw):
    return json.loads(raw)


def records(raw):
    return [obj(line) for line in raw.splitlines()]


def encoded(v):
    return (
        json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def rewrite_envelopes(bundle, name, body):
    bundle = dict(bundle)
    bundle[name] = body
    manifest = obj(bundle["CASE_MANIFEST.json"])
    if name in manifest["files"]:
        manifest["files"][name]["sha256"] = hashlib.sha256(body).hexdigest()
        manifest["files"][name]["byte_count"] = len(body)
    bundle["CASE_MANIFEST.json"] = encoded(manifest)
    commit = obj(bundle["COMMITTED.json"])
    for filename in commit["files"]:
        commit["files"][filename]["sha256"] = hashlib.sha256(bundle[filename]).hexdigest()
        commit["files"][filename]["byte_count"] = len(bundle[filename])
    bundle["COMMITTED.json"] = encoded(commit)
    return bundle


def test_complete_deterministic_minimal_bundle_and_public_private_roles():
    value = rows()
    before = copy.deepcopy(value)
    bundle = freeze_humaneval_cases(value, source=source())
    assert list(bundle) == [
        "PROMPTS.jsonl",
        "ORACLES.jsonl",
        "EXCLUSIONS.json",
        "CASE_MANIFEST.json",
        "COMMITTED.json",
    ]
    assert bundle == freeze_humaneval_cases(value, source=source()) and value == before
    prompts = records(bundle["PROMPTS.jsonl"])
    oracles = records(bundle["ORACLES.jsonl"])
    assert len(prompts) == len(oracles) == 164
    assert all(set(p) == {"case_id", "prompt", "entry_point"} for p in prompts)
    assert all(set(p) == {"case_id", "test", "canonical_solution"} for p in oracles)
    assert (
        b"PRIVATE_TEST" not in bundle["PROMPTS.jsonl"] and b"PRIVATE_REFERENCE" not in bundle["PROMPTS.jsonl"]
    )
    assert prompts[0]["case_id"] == "HumanEval/0" and prompts[-1]["case_id"] == "HumanEval/163"
    manifest = obj(bundle["CASE_MANIFEST.json"])
    assert manifest["case_count"] == 164 and len(manifest["cases"]) == 164
    assert (
        manifest["metric_profile"] == "pass_at_1_descriptive"
        and manifest["execution_status"] == "not_executed"
    )
    assert (
        manifest["files"]["PROMPTS.jsonl"]["role"] == "public_prompt"
        and manifest["files"]["ORACLES.jsonl"]["role"] == "private_oracle"
    )
    assert validate_coding_case_bundle(bundle, source=source())["case_count"] == 164


@pytest.mark.parametrize(
    "field,value",
    [
        ("repo_id", "other/source"),
        ("revision", "main"),
        ("filename", "../evil.py"),
        ("sha256", "f" * 64),
        ("byte_count", True),
        ("byte_count", 83919),
        ("format", "jsonl"),
        ("license", "unknown"),
        ("card_sha256", "f" * 64),
    ],
)
def test_fixed_source_pin_type_license_and_identity_refusal(field, value):
    with pytest.raises(ValueError):
        freeze_humaneval_cases(rows(), source=source(**{field: value}))


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "order",
        "extra_row",
        "null",
        "bool",
        "extra_field",
        "missing_field",
        "unknown_id",
        "entrypoint",
        "source_grants",
        "oversize",
    ],
)
def test_closed_complete_original_source_rows_refuse(kind):
    value = rows()
    if kind == "missing":
        value.pop()
    elif kind == "duplicate":
        value[1]["task_id"] = value[0]["task_id"]
    elif kind == "order":
        value[0], value[1] = value[1], value[0]
    elif kind == "extra_row":
        value.append(copy.deepcopy(value[-1]))
    elif kind == "null":
        value[0]["test"] = None
    elif kind == "bool":
        value[0]["canonical_solution"] = False
    elif kind == "extra_field":
        value[0]["approved"] = True
    elif kind == "missing_field":
        value[0].pop("prompt")
    elif kind == "unknown_id":
        value[0]["task_id"] = "HumanEval/999"
    elif kind == "entrypoint":
        value[0]["entry_point"] = "__import__('os')"
    elif kind == "source_grants":
        value[0]["role"] = "administrator"
    else:
        value[0]["prompt"] = "x" * 8193
    with pytest.raises(ValueError):
        freeze_humaneval_cases(value, source=source())


def test_code_strings_remain_inert_and_unicode_bytes_preserved():
    value = rows()
    value[0]["prompt"] += "# λ漢字\n"
    value[0]["test"] = "raise SystemExit('PRIVATE_TEST_never_execute')"
    bundle = freeze_humaneval_cases(value, source=source())
    assert records(bundle["PROMPTS.jsonl"])[0]["prompt"] == value[0]["prompt"]
    assert records(bundle["ORACLES.jsonl"])[0]["test"] == value[0]["test"]


@pytest.mark.parametrize(
    "kind",
    [
        "prompt",
        "oracle",
        "id",
        "extra_file",
        "missing_file",
        "projection",
        "lineage",
        "role",
        "grant",
        "duplicate_json",
        "nonfinite",
    ],
)
def test_independent_replay_refuses_mutations_even_outer_hashes_recomputed(kind):
    bundle = freeze_humaneval_cases(rows(), source=source())
    if kind in ("prompt", "id"):
        values = records(bundle["PROMPTS.jsonl"])
        if kind == "prompt":
            values[0]["prompt"] = values[0]["prompt"].replace("Synthetic", "synthetic")
        else:
            values[0]["case_id"] = "HumanEval/1"
        bundle = rewrite_envelopes(bundle, "PROMPTS.jsonl", b"".join(encoded(v) for v in values))
    elif kind == "oracle":
        values = records(bundle["ORACLES.jsonl"])
        values[0]["test"] = values[0]["test"].replace("PRIVATE_TEST", "private_test")
        bundle = rewrite_envelopes(bundle, "ORACLES.jsonl", b"".join(encoded(v) for v in values))
    elif kind == "extra_file":
        bundle["unexpected.pkl"] = b"data"
    elif kind == "missing_file":
        bundle.pop("ORACLES.jsonl")
    elif kind == "projection":
        value = obj(bundle["EXCLUSIONS.json"])
        value["code_pair_sha256"][0] = "f" * 64
        bundle = rewrite_envelopes(bundle, "EXCLUSIONS.json", encoded(value))
    elif kind in ("lineage", "role", "grant"):
        value = obj(bundle["CASE_MANIFEST.json"])
        if kind == "lineage":
            value["cases"][0]["field_sha256"]["test"] = "f" * 64
        elif kind == "role":
            value["files"]["ORACLES.jsonl"]["role"] = "public_prompt"
        else:
            value["approved"] = True
        bundle = rewrite_envelopes(bundle, "CASE_MANIFEST.json", encoded(value))
    elif kind == "duplicate_json":
        bundle["COMMITTED.json"] = b'{"schema_version":1,"schema_version":1}\n'
    else:
        bundle["COMMITTED.json"] = b'{"schema_version":NaN}\n'
    with pytest.raises(ValueError):
        validate_coding_case_bundle(bundle, source=source())


def test_exclusions_use_existing_exact_codepair_domain_not_unknown_wrappers():
    bundle = freeze_humaneval_cases(rows(), source=source())
    excluded = obj(bundle["EXCLUSIONS.json"])
    pair = {
        "old_contents": rows()[0]["prompt"],
        "new_contents": rows()[0]["prompt"] + rows()[0]["canonical_solution"],
    }
    # CommitPack _digest uses sorted compact UTF8 JSON without terminal newline.
    expected = hashlib.sha256(
        json.dumps(pair, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert expected in excluded["code_pair_sha256"]
    assert excluded["canonical_sha256"] == excluded["input_sha256"] == []
    coverage = obj(bundle["CASE_MANIFEST.json"])["coverage"]
    assert (
        coverage["pretraining_contamination"] == "unknown"
        and coverage["near_or_semantic_overlap"] == "not_verified"
    )
    assert "source_authenticated" not in obj(bundle["CASE_MANIFEST.json"])


def test_complete_selection_exceeding_aggregate_bound_never_truncates():
    value = rows()
    for row in value:
        row["prompt"] = "x" * 8192
    with pytest.raises(ValueError, match="complete decoded"):
        freeze_humaneval_cases(value, source=source())


def test_deep_unknown_metadata_is_bounded_before_json_allocation():
    from amlas.data_eng.coding_case_manifest import _object

    with pytest.raises(ValueError):
        _object(b'{"unexpected":' + b"[" * 64 + b"0" + b"]" * 64 + b"}")


def test_metadata_only_codec_does_not_authenticate_coherently_replaced_raw_source():
    original = freeze_humaneval_cases(rows(), source=source())
    value = rows()
    value[0]["test"] = value[0]["test"].replace("PRIVATE_TEST", "private_test")
    changed = freeze_humaneval_cases(value, source=source())
    assert changed != original
    # Only the later managed raw byte/hash gate can establish that decoded rows came
    # from the pinnedParquet. Puremetadata validates consistent projections only.
    assert (
        validate_coding_case_bundle(changed, source=source())["raw_bytes_origin"]
        == "caller_integrity_gate_required"
    )


def test_exclusion_object_preserves_existing_versioned_preparation_schema():
    excluded = obj(freeze_humaneval_cases(rows(), source=source())["EXCLUSIONS.json"])
    assert set(excluded) == {
        "schema_version",
        "repositories",
        "commits",
        "canonical_sha256",
        "input_sha256",
        "code_pair_sha256",
    }
    assert type(excluded["schema_version"]) is int and excluded["schema_version"] == 1
