"""Parent verifies original private pins/counts; ordered guest digests remain descriptive."""

import dataclasses
import hashlib
import json

import pytest
from test_qwen_sft_backend import HEADER, SESSION, backend, catalog, encoded, install_output
from test_qwen_sft_receipt_summary import expected_digest

from amlas.data_eng.commitpack_preparation import COVERAGE, SUPERVISION_KIND


def prepared_backend():
    b, n, s, inputs = backend()
    old = inputs.prepared.preparation_ref
    _, raw = s.objects[old.uri.split("/", 3)[3]]
    commit = json.loads(raw)
    splits = commit["counts"]["split_counts"]
    retained = sum(splits.values())
    counts = {
        "source": 100,
        "normalized": 100,
        "normalization_rejected": 0,
        "excluded_origins": 0,
        "included_origins": 100,
        "retained_representatives": retained,
        "duplicate_origins": 100 - retained,
        "split_counts": splits,
    }
    commit["counts"] = counts
    commit["coverage"] = dict(COVERAGE)
    raw = encoded(commit)
    ref = dataclasses.replace(old, sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw))
    s.install(ref, raw)
    inputs = dataclasses.replace(inputs, prepared=dataclasses.replace(inputs.prepared, preparation_ref=ref))
    h = b.submit(inputs=inputs, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    _, receipt, final, put = install_output(b, n, s, h)
    summary = {
        "schema_version": 1,
        "summary_profile": "commitpack_replay_summary_v1",
        "digest_domain": "amlas.commitpack.canonical_sha256_split.v1",
        "counts": counts,
        "coverage": dict(COVERAGE),
        "supervision_kind": SUPERVISION_KIND,
        "exclusion_policy_sha256": inputs.prepared.exclusions_ref.sha256,
        "splits": {
            name: {
                "record_count": count,
                "ordered_canonical_sha256_sha256": expected_digest(name, ["a" * 64] * count),
            }
            for name, count in splits.items()
        },
    }
    receipt["input_summary"].update(
        preparation_profile_id=HEADER["profile_id"],
        preparation_policy_version=HEADER["policy_version"],
        commitpack=summary,
    )
    return b, n, s, h, receipt, final, put


def publish(receipt, final, put):
    ref = put("TRAINING.json", encoded(receipt))
    final["files"]["TRAINING.json"] = {"sha256": ref.sha256, "byte_count": ref.byte_count}
    put("COMMITTED.json", encoded(final))


@pytest.mark.parametrize(
    "change", ["missing", "extra", "count", "version", "hash", "exclusions", "source", "profile", "coverage"]
)
def test_cp_parent_refuses_selfconsistent_rehashed_summary_drift(change):
    b, _n, _s, h, receipt, final, put = prepared_backend()
    cp = receipt["input_summary"]["commitpack"]
    if change == "missing":
        del receipt["input_summary"]["commitpack"]
    elif change == "extra":
        cp["approved"] = True
    elif change == "count":
        cp["counts"]["source"] += 1
    elif change == "version":
        cp["schema_version"] = True
    elif change == "hash":
        cp["splits"]["train"]["ordered_canonical_sha256_sha256"] = "z" * 64
    elif change == "exclusions":
        cp["exclusion_policy_sha256"] = "b" * 64
    elif change == "source":
        receipt["input_summary"]["preparation_commit_sha256"] = "b" * 64
    elif change == "profile":
        receipt["input_summary"]["preparation_profile_id"] = "caller-profile"
    else:
        cp["coverage"]["unknown"] = True
    publish(receipt, final, put)
    assert b.artifacts(h)["success"] is False


def test_parent_does_not_claim_independent_ordered_digest_authentication():
    b, _n, _s, h, receipt, final, put = prepared_backend()
    receipt["input_summary"]["commitpack"]["splits"]["train"]["ordered_canonical_sha256_sha256"] = "b" * 64
    publish(receipt, final, put)
    assert b.artifacts(h)["success"] is True


def test_parent_binds_conservation_preserving_split_count_transfer():
    from amlas.training.qwen_sft import _validate_commitpack_summary

    b, _n, _s, h, receipt, final, put = prepared_backend()
    summary = receipt["input_summary"]["commitpack"]
    counts = summary["counts"]["split_counts"]
    counts["train"] -= 1
    counts["validation"] += 1
    for name in ("train", "validation"):
        summary["splits"][name] = {
            "record_count": counts[name],
            "ordered_canonical_sha256_sha256": expected_digest(name, ["a" * 64] * counts[name]),
        }
    _validate_commitpack_summary(summary)
    publish(receipt, final, put)
    assert b.artifacts(h)["success"] is False
