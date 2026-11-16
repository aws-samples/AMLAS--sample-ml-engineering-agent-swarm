# scripts/ci_shard

## Purpose

Own the tools that split the python phase of the full CI into several pytest containers on one builder and merge
their results into the files the single pipeline writes. The 23 gates, the 5 phase codes and the files the terminal
observer and the qualifier read do not change.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory. The tools are CI
assets of the builder, not product code: nothing under `src/` imports them.

## Local Contracts

- 2026-11-07. `full_ci_shard_plan.py` (standard library, Python 3.9 compatible) reads the ids of the
  unchanged `collection` gate (`collection.log`, checked against `collection.metadata.json` byte for byte) and a
  durations asset built from a baseline JUnit file. The plan is module units, weighted by seconds and spread by
  longest-first scheduling; a module heavier than half of the ideal shard load is split to single tests; a single
  test of at least 300 s (the child-process timeout of the longest test) gets a shard of its own. Every id lands in
  exactly one non-empty shard, bytes are deterministic, and a count above the number of units is refused. The plan
  file is canonical JSON and its sha256 is the pin that shards and merger check.
- 2026-11-07. `full_ci_shard.py` is the pytest plugin of one shard (`-p full_ci_shard`). It keeps the items
  of the plan's shard, refuses `-k`, `-m`, `--deselect`, `--ignore`, `--lf`, `--ff`, `--nf`, `--sw`, `-x`,
  `--maxfail` and `--collect-only`, requires the full collection to hash to the plan's, and writes
  `shard-report.json` (plan sha256, full and selected inventories, per-id outcome ledger, terminal-reporter counts,
  exit status, wall seconds, cgroup memory peak) and `execution-collection.metadata.json` for the selected set. It
  changes no test behavior and writes no report for a refused run.
- 2026-11-07. `full_ci_shard_merge.py` writes `full-pytest.xml`, `pytest.log`, `pytest.exit`,
  `execution-collection.metadata.json` (byte-equal to `collection.metadata.json`) and `shards.metadata.json` into
  `out/` only after every check passed: plan pinned and canonical; each shard built on that plan and on the same full
  collection; no id missing, duplicated, extra or run by the wrong shard; ledger and JUnit list the same ids; skipped,
  passed and log counts agree; the observer's own log regular expression reads the same totals from the merged log.
  Each plan carries a per-run nonce, so shard outputs of an earlier run that used the same ids are refused. The
  summary line is the last line shaped like pytest's, so output printed after it does not refuse a shard. A refusal
  (`MergeError`) leaves `out/` unchanged. A shard that failed with a structurally sound report is merged with its
  failure elements kept and `pytest.exit` 1; output the checks cannot reconcile (for example a call failure plus a
  teardown error of one test) is refused, and the orchestrator then writes `pytest.exit` 1 and the reason.
  An expected failure (xfail) is counted apart from skips: the JUnit `skipped` attribute must equal ledger skips plus
  xfails, and passed, skipped, xfailed and xpassed must account for every testcase of a run that exits 0.
- 2026-11-07. `shard_orchestrator.py` starts one container per shard at once with the single pipeline's
  image and flags (`--network none`, read-only root, same mounts and seeds), changing only the per-shard name and
  cidfile, `/ci-out` read-only, `/ci-shard` and `/ci-run` per shard, and the shard environment. It waits until every
  container has stopped, then merges. Shard output and cache directories must be new. A shard still running at
  the deadline (4 h unless `shard-params.json` says otherwise) is killed and counts as 124; a container that does not
  stop within 15 minutes is an error. A refused plan or merge, or any exception inside the phase, writes
  `pytest.exit` 1 and a `pytest.log` that names the reason. A failed shard fails the run: no shard-level re-run, no
  `--lf`, no deselection.
- 2026-11-07. `derive_sharded_assets.py` builds the gates script (the original without its pytest gate),
  the shard script and the sharded `full-ci.native.py` from the pinned originals by exact, counted replacements and
  writes the new pins; any change in an original is a refusal, including any step after the original's pytest gate
  other than its closing source check. The sharded native sets the python phase code to 1 before the orchestrator
  runs, so an exception cannot leave the gates container's 0 in place.
- 2026-11-07. `shard_observer_check.py` repeats the observer's assertions about `out/` with explicit pins
  (not the run-specific pids, QA clone, AWS actor and container checks) and adds checks for `shards.metadata.json`;
  `shard_equality_check.py` compares a sharded `out/` with a single-process baseline `out/` and exits 1 on any
  difference.

## Work Guidance

Keep the host-side tools to the standard library and Python 3.9 syntax: the builder host runs them with its system
Python. The plugin runs inside the CI image. Change the plan, merge or observer-check rules with a failing test first;
a new shard-induced failure becomes a plan rule (keep tests together) and not a test change.

## Verification

```bash
python -m pytest -q tests/test_ci_shard_plan.py tests/test_ci_shard_merge.py tests/test_ci_shard_plugin.py tests/test_ci_shard_orchestrator.py
python -m ruff check scripts/ci_shard tests/test_ci_shard_*.py
```

These tests use synthetic fixtures and a fake `docker`; they prove the tools against the file formats, not a sharded
run of the full suite. The equality of a real sharded run with the single-process run is recorded in the CI evidence.

## Child DOX Index

| Path | Covers |
| --- | --- |
