"""Process-local memory of the verified CommitPack replay; fake S3, no model or service calls.

Every pass still reads and hashes every exact-version body and runs every qualification callback; only an
identical re-derivation from identical verified bytes and parameters is skipped.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import Counter

import pytest
from test_managed_coding_preparation import encoded, sha
from test_managed_coding_preparation import fixture as base_fixture
from test_managed_commitpack_preparation import complete_outputs, staged_fixture

from amlas.backends.adapters import coding_preparation as adapter
from amlas.data_eng import commitpack_preparation as prep
from amlas.data_eng import commitpack_profile as profile
from amlas.sagemaker_entrypoints.coding_preparation import OUTPUT_FILES

SESSION = "test_session"


@pytest.fixture
def fixture(monkeypatch):
    return base_fixture.__wrapped__(monkeypatch)


@pytest.fixture(autouse=True)
def fresh_memo():
    memo = getattr(adapter, "_REPLAY_MEMO", None)
    if memo is not None:
        memo.clear()
    yield
    if memo is not None:
        memo.clear()


class Spy:
    """Counts the replay and the per-row normalization it repeats."""

    def __init__(self, monkeypatch):
        self.replays = 0
        self.normalized = 0
        real_replay, real_normalize = prep.replay_commitpack_preparation, profile.normalize_commitpack_row

        def replay(*args, **kwargs):
            self.replays += 1
            return real_replay(*args, **kwargs)

        def normalize(*args, **kwargs):
            self.normalized += 1
            return real_normalize(*args, **kwargs)

        monkeypatch.setattr(prep, "replay_commitpack_preparation", replay)
        monkeypatch.setattr(profile, "normalize_commitpack_row", normalize)


def prepared(fixture, monkeypatch):
    value = staged_fixture(fixture, monkeypatch)
    handle = value[0].submit(sources=(value[4],), catalog=value[3], runtime_session_id=SESSION)
    _result, committed = complete_outputs(value, handle)
    return value, handle, committed


def rewrite(value, handle, committed, name, mutate):
    """Replace one body with a changed one and re-pin its committed descriptor (a rehashed substitution)."""
    s3, prefix = value[2], handle.extra["output_prefix"]
    rows = [json.loads(line) for line in s3.values[prefix + name][0].splitlines()]
    mutate(rows)
    body = b"".join(encoded(record) for record in rows)
    s3.add(prefix + name, body)
    committed["files"][name].update(byte_count=len(body), sha256=sha(body))
    s3.add(prefix + "COMMITTED.json", encoded(committed))


def reads(s3, start):
    return Counter(
        (kind, kw.get("Key"), kw.get("VersionId"), kw.get("Prefix"))
        for kind, kw in s3.calls[start:]
        if kind in ("get", "head", "list")
    )


# --- (1) N identical passes call the replay once -------------------------------------------------------------


def test_identical_passes_replay_once(fixture, monkeypatch):
    value, handle, _committed = prepared(fixture, monkeypatch)
    spy = Spy(monkeypatch)
    first = value[0].artifacts(handle)
    assert first["success"] is True and first["counts"]["source"] == 100
    cold = spy.normalized
    assert spy.replays == 1 and cold >= 100
    later = [value[0].artifacts(handle) for _ in range(6)]
    assert all(item == first for item in later)
    assert spy.replays == 1 and spy.normalized == cold


def test_every_pass_still_reads_and_qualifies_everything(fixture, monkeypatch):
    value, handle, _committed = prepared(fixture, monkeypatch)
    backend, s3 = value[0], value[2]
    qualified = []
    real = backend._qualify_io
    monkeypatch.setattr(backend, "_qualify_io", lambda: (qualified.append(1), real())[1])
    spy = Spy(monkeypatch)
    passes = []
    for _ in range(3):
        start, count = len(s3.calls), len(qualified)
        assert backend.artifacts(handle)["success"] is True
        passes.append((reads(s3, start), len(qualified) - count))
    assert spy.replays == 1
    assert passes[1] == passes[0] and passes[2] == passes[0]
    assert sum(1 for key in passes[0][0] if key[0] == "get") >= 6  # all six bodies plus the committed marker
    assert passes[0][1] > 20


# --- (2) any change in the key components misses and recomputes ------------------------------------------------


def test_changed_body_one_byte_misses_and_recomputes(fixture, monkeypatch):
    value, handle, committed = prepared(fixture, monkeypatch)
    spy = Spy(monkeypatch)
    first = value[0].artifacts(handle)
    assert value[0].artifacts(handle) == first and spy.replays == 1

    def append_character(rows):
        rows[0]["origin_record"]["source_fields"]["message"] += "x"

    rewrite(value, handle, committed, "lineage.jsonl", append_character)
    changed = value[0].artifacts(handle)
    assert changed["success"] is True and spy.replays == 2
    assert changed["sidecar_refs"]["lineage"]["sha256"] != first["sidecar_refs"]["lineage"]["sha256"]
    assert value[0].artifacts(handle) == changed and spy.replays == 2


def test_new_version_id_of_identical_bytes_misses(fixture, monkeypatch):
    value, handle, _committed = prepared(fixture, monkeypatch)
    spy = Spy(monkeypatch)
    first = value[0].artifacts(handle)
    prefix = handle.extra["output_prefix"]
    value[2].add(prefix + "membership.jsonl", value[2].values[prefix + "membership.jsonl"][0])
    again = value[0].artifacts(handle)
    assert again["success"] is True and spy.replays == 2
    assert (
        again["sidecar_refs"]["membership"]["version_id"] != first["sidecar_refs"]["membership"]["version_id"]
    )
    assert value[0].artifacts(handle) == again and spy.replays == 2


def test_other_split_body_change_misses(fixture, monkeypatch):
    value, handle, committed = prepared(fixture, monkeypatch)
    spy = Spy(monkeypatch)
    assert value[0].artifacts(handle)["success"] is True
    rewrite(value, handle, committed, "train.jsonl", lambda rows: rows.reverse())
    assert value[0].artifacts(handle)["success"] is False  # reordered rows are not the replayed split
    assert spy.replays == 2


def replay_key(**changes):
    names = list(OUTPUT_FILES)
    refs = {
        name: {
            "uri": "s3://bucket/prefix/" + name,
            "version_id": "v1",
            "sha256": sha(name.encode()),
            "byte_count": 10 + index,
            "format": "json" if name == "REPORT.json" else "jsonl",
        }
        for index, name in enumerate(names)
    }
    records = {name: [{}] * 3 for name in names if name != "REPORT.json"}
    sources = (
        {
            "repo_id": "bigcode/commitpackft",
            "revision": "a" * 40,
            "filename": "data/python/data.jsonl",
            "artifact_ref": {
                "uri": "s3://bucket/source",
                "version_id": "s1",
                "sha256": "b" * 64,
                "byte_count": 5,
                "format": "jsonl",
            },
        },
    )
    parameters = {"refs": refs, "records": records, "sources": sources, "source_counts": {"c" * 64: 3}}
    parameters.update(changes)
    return adapter._replay_key(**parameters), parameters


def test_key_is_stable_and_ignores_dict_order():
    key, parameters = replay_key()
    assert len(key) == 64 and set(key) <= set("0123456789abcdef")
    shuffled = {
        "refs": dict(reversed(list(parameters["refs"].items()))),
        "records": dict(reversed(list(parameters["records"].items()))),
        "sources": parameters["sources"],
        "source_counts": dict(parameters["source_counts"]),
    }
    assert adapter._replay_key(**shuffled) == key


@pytest.mark.parametrize("name", list(OUTPUT_FILES))
@pytest.mark.parametrize("field", ["sha256", "version_id", "uri", "byte_count"])
def test_key_binds_every_body_identity(name, field):
    base, parameters = replay_key()
    refs = {key: dict(ref) for key, ref in parameters["refs"].items()}
    refs[name][field] = refs[name][field] + 1 if field == "byte_count" else refs[name][field] + "0"
    assert replay_key(refs=refs)[0] != base


@pytest.mark.parametrize("name", [name for name in OUTPUT_FILES if name != "REPORT.json"])
def test_key_binds_record_counts(name):
    base, parameters = replay_key()
    records = dict(parameters["records"])
    records[name] = [{}] * 4
    assert replay_key(records=records)[0] != base


def test_key_binds_sources_and_private_counts_and_order():
    base, parameters = replay_key()
    sources = parameters["sources"]
    changed = ({**sources[0], "revision": "d" * 40},)
    assert replay_key(sources=changed)[0] != base
    deeper = ({**sources[0], "artifact_ref": {**sources[0]["artifact_ref"], "version_id": "s2"}},)
    assert replay_key(sources=deeper)[0] != base
    assert replay_key(sources=(sources[0], {**sources[0], "filename": "other"}))[0] != base
    assert (
        replay_key(sources=(sources[0], {**sources[0], "filename": "other"}))[0]
        != replay_key(sources=({**sources[0], "filename": "other"}, sources[0]))[0]
    )
    assert replay_key(source_counts={"c" * 64: 4})[0] != base
    assert replay_key(source_counts={"c" * 64: 3, "d" * 64: 0})[0] != base


def test_key_binds_module_constants_the_replay_reads(monkeypatch):
    base, _parameters = replay_key()
    for owner, name, value in (
        (profile, "ARTIFACT_SHA256", "e" * 64),
        (profile, "REVISION", "f" * 40),
        (profile, "FILENAME", "other.jsonl"),
        (profile, "DATASET_REPO", "other/repo"),
        (prep, "SOURCE_BYTES", 1),
        (prep, "POLICY_VERSION", "other.policy"),
        (prep, "PROFILE_ID", "other.profile"),
        (prep, "MAX_ROWS", 7),
        (prep, "SUPERVISION_KIND", "other"),
        (prep, "COVERAGE", {**prep.COVERAGE, "tokenization": "applied"}),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(owner, name, value)
            assert replay_key()[0] != base, name


@pytest.mark.parametrize(
    "parameters",
    [
        {"source_counts": {1: 3}},
        {"source_counts": {"c" * 64: 3.5}},
        {"sources": [{"repo_id": "x"}]},
        {"sources": ({"repo_id": float("nan")},)},
        {"source_counts": None},
    ],
)
def test_key_refuses_inputs_the_replay_would_refuse_so_they_bypass_the_memory(parameters):
    assert replay_key(**parameters)[0] is None


# --- (3) tampered bodies still fail after a primed pass -------------------------------------------------------


@pytest.mark.parametrize(
    "tamper", ["source_count", "coverage", "origin", "membership", "payload", "report_bool", "foreign_source"]
)
def test_primed_memory_never_masks_a_rehashed_substitution(fixture, monkeypatch, tamper):
    value, handle, committed = prepared(fixture, monkeypatch)
    assert value[0].artifacts(handle)["success"] is True
    prefix, s3 = handle.extra["output_prefix"], value[2]
    if tamper == "source_count":
        committed["counts"]["source"] = 99
        s3.add(prefix + "COMMITTED.json", encoded(committed))
    elif tamper == "coverage":
        committed["coverage"] = {**prep.COVERAGE, "tokenization": "ready"}
        s3.add(prefix + "COMMITTED.json", encoded(committed))
    elif tamper == "report_bool":
        raw = s3.values[prefix + "REPORT.json"][0]
        report = json.loads(raw)
        report["counts"]["source"] = True
        body = encoded(report)
        s3.add(prefix + "REPORT.json", body)
        committed["files"]["REPORT.json"].update(byte_count=len(body), sha256=sha(body))
        s3.add(prefix + "COMMITTED.json", encoded(committed))
    else:
        name = {
            "origin": "lineage.jsonl",
            "foreign_source": "lineage.jsonl",
            "membership": "membership.jsonl",
            "payload": "train.jsonl",
        }[tamper]

        def mutate(rows):
            if tamper == "origin":
                rows[0]["origin_record"]["source_fields"]["new_contents"] = "changed code"
            elif tamper == "foreign_source":
                rows[0]["source_provenance"]["artifact_version_id"] = "foreign"
            elif tamper == "membership":
                rows[0]["split"] = "test"
            else:
                rows[0]["messages"][1]["content"] = "changed code"

        rewrite(value, handle, committed, name, mutate)
    result = value[0].artifacts(handle)
    assert result["success"] is False and result["prepared"] is False


@pytest.mark.parametrize("name", ["lineage.jsonl", "membership.jsonl", "train.jsonl", "REPORT.json"])
def test_changed_body_with_stale_descriptor_fails_before_any_memory_lookup(fixture, monkeypatch, name):
    value, handle, _committed = prepared(fixture, monkeypatch)
    spy = Spy(monkeypatch)
    assert value[0].artifacts(handle)["success"] is True and spy.replays == 1
    prefix, s3 = handle.extra["output_prefix"], value[2]
    raw = s3.values[prefix + name][0]
    s3.add(prefix + name, raw[:-2] + (b"0\n" if raw[-2:-1] != b"0" else b"1\n"))
    result = value[0].artifacts(handle)
    assert result["success"] is False
    assert spy.replays == 1


def test_a_failing_replay_is_not_remembered(fixture, monkeypatch):
    value, handle, _committed = prepared(fixture, monkeypatch)
    real = prep.replay_commitpack_preparation
    calls = []

    def flaky(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("CommitPack replayed lineage differs")
        return real(**kwargs)

    monkeypatch.setattr(prep, "replay_commitpack_preparation", flaky)
    assert value[0].artifacts(handle)["success"] is False
    assert value[0].artifacts(handle)["success"] is True
    assert value[0].artifacts(handle)["success"] is True
    assert len(calls) == 2


# --- (4) bounded memory -----------------------------------------------------------------------------------


def make(value):
    return {"counts": {"n": value}, "canonical_sha256_by_split": {"train": [str(value)] * 3}, "kind": "x"}


def test_memory_holds_at_most_the_configured_number_of_results():
    assert adapter.REPLAY_MEMO_ENTRIES == 2
    memo = adapter._ReplayMemo(entries=2)
    calls = []

    def fetch(key):
        return memo.fetch(key, lambda: (calls.append(key), make(key))[1])

    for key in "abcde":
        fetch(key)
        assert len(memo) <= 2
    assert len(memo) == 2 and calls == list("abcde")
    fetch("e"), fetch("d")
    assert calls == list("abcde")  # the two newest are held
    fetch("a")
    assert calls == [*list("abcde"), "a"]  # evicted long ago
    fetch("d")  # touched after e, so it outlived e
    fetch("a"), fetch("d")
    assert calls == [*list("abcde"), "a"]
    fetch("e")
    assert calls[-1] == "e" and len(memo) == 2


def test_the_module_memory_is_bounded_process_local_and_not_persisted():
    assert isinstance(adapter._REPLAY_MEMO, adapter._ReplayMemo)
    assert adapter._REPLAY_MEMO._entries == adapter.REPLAY_MEMO_ENTRIES
    source = open(adapter.__file__, encoding="utf-8").read()
    assert "pickle" not in source and "shelve" not in source


def test_zero_or_invalid_size_is_refused():
    for entries in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            adapter._ReplayMemo(entries=entries)


# --- (5) results cannot be mutated by callers -------------------------------------------------------------


def test_miss_and_hit_return_private_copies():
    memo = adapter._ReplayMemo(entries=2)
    first = memo.fetch("k", lambda: make(1))
    first["counts"]["n"] = 99
    first["canonical_sha256_by_split"]["train"].append("tampered")
    second = memo.fetch("k", lambda: make(2))
    assert second == make(1)
    second["counts"].clear()
    third = memo.fetch("k", lambda: make(3))
    assert third == make(1) and third is not second
    assert third["canonical_sha256_by_split"]["train"] is not second["canonical_sha256_by_split"]["train"]


def test_plain_results_are_kept_as_one_immutable_snapshot():
    memo = adapter._ReplayMemo(entries=2)
    memo.fetch("k", lambda: make(1))
    assert type(memo._held["k"]) is bytes


@pytest.mark.parametrize(
    "shape",
    [
        {"t": (1, 2)},
        {"f": 1.5},
        {1: "a"},
        {"s": {1, 2}},
        {"b": b"x"},
        {"i": float("inf")},
        {"text": "é\u2028"},
    ],
)
def test_results_that_are_not_plain_json_are_kept_and_returned_unchanged(shape):
    memo = adapter._ReplayMemo(entries=2)
    first = memo.fetch("k", lambda: copy.deepcopy(shape))
    second = memo.fetch("k", lambda: {"never": "computed"})
    third = memo.fetch("k", lambda: {"never": "computed"})
    assert first == shape and second == shape and third == shape and second is not third


def test_a_caller_mutating_a_returned_artifacts_result_does_not_reach_the_next_pass(fixture, monkeypatch):
    value, handle, _committed = prepared(fixture, monkeypatch)
    first = value[0].artifacts(handle)
    expected_counts = json.loads(json.dumps(first["counts"]))
    first["counts"]["source"] = -1
    first["coverage"]["tokenization"] = "ready"
    second = value[0].artifacts(handle)
    assert second["success"] is True and second["counts"] == expected_counts
    assert second["coverage"] == prep.COVERAGE


# --- (6) concurrency ---------------------------------------------------------------------------------------


def test_threads_asking_the_same_key_compute_once_and_get_private_correct_results():
    memo = adapter._ReplayMemo(entries=2)
    entered, release, calls = threading.Event(), threading.Event(), []

    def compute():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(10)
        return make(7)

    out = [None] * 6

    def worker(index):
        out[index] = memo.fetch("k", compute)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    threads[0].start()
    assert entered.wait(10)
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.2)
    assert len(calls) == 1 and all(item is None for item in out)  # nobody sees a partial result
    release.set()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert len(calls) == 1
    assert all(item == make(7) for item in out) and len({id(item) for item in out}) == 6


def test_a_blocked_key_does_not_block_other_keys():
    memo = adapter._ReplayMemo(entries=2)
    entered, release = threading.Event(), threading.Event()

    def slow():
        entered.set()
        assert release.wait(10)
        return make(1)

    owner = threading.Thread(target=lambda: memo.fetch("slow", slow))
    owner.start()
    assert entered.wait(10)
    started = time.monotonic()
    assert memo.fetch("other", lambda: make(2)) == make(2)
    assert time.monotonic() - started < 2
    release.set()
    owner.join(10)
    assert not owner.is_alive()


@pytest.mark.parametrize("failure", [ValueError("replay refused"), SystemExit(3)])
def test_a_failed_owner_releases_waiters_and_leaves_nothing_behind(failure):
    memo = adapter._ReplayMemo(entries=2)
    entered, release, calls = threading.Event(), threading.Event(), []

    def compute():
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
            raise failure
        return make(5)

    outcomes = {}

    def worker(name):
        try:
            outcomes[name] = memo.fetch("k", compute)
        except BaseException as error:  # the owner's own failure
            outcomes[name] = error

    owner = threading.Thread(target=worker, args=("owner",))
    owner.start()
    assert entered.wait(10)
    waiters = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(3)]
    for thread in waiters:
        thread.start()
    time.sleep(0.2)
    release.set()
    for thread in (owner, *waiters):
        thread.join(10)
        assert not thread.is_alive()
    assert outcomes["owner"] is failure
    assert all(outcomes[f"w{i}"] == make(5) for i in range(3))
    assert len(calls) == 2 and len(memo) == 1


def test_a_failure_every_time_is_raised_to_every_caller_and_never_stored():
    memo = adapter._ReplayMemo(entries=2)
    calls = []

    def compute():
        calls.append(1)
        raise ValueError("replay refused")

    errors = []

    def worker():
        try:
            memo.fetch("k", compute)
        except ValueError as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert len(errors) == 4 and len(memo) == 0
    assert memo.fetch("k", lambda: make(1)) == make(1)


def test_a_waiter_never_waits_past_its_bound():
    memo = adapter._ReplayMemo(entries=2, wait_seconds=0.2)
    entered, release = threading.Event(), threading.Event()

    def slow():
        entered.set()
        assert release.wait(10)
        return make(1)

    owner = threading.Thread(target=lambda: memo.fetch("k", slow))
    owner.start()
    assert entered.wait(10)
    started = time.monotonic()
    assert memo.fetch("k", lambda: make(1)) == make(1)  # computed by the waiter itself after the bound
    assert time.monotonic() - started < 5
    release.set()
    owner.join(10)
    assert not owner.is_alive()
    assert len(memo) == 1
