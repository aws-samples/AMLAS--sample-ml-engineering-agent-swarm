# Qwen TCP boundary controls

## Purpose

Exercise the fixed SFT-only TCP denial policy without launching training.

## Ownership

Inherits ../AGENTS.md; downloader policy/tests remain separate.

## Local Contracts

- Fakes require exact network mask3/filesystem0, no port grants, native-final state,
  single-thread/NNP1 and supported ABI before create/restrict; syscall failures refuse.
- Owned ruleset closure runs in an isolated child with no concurrent descriptor reuse.
- Numeric denial checks accept only errno1/13; reachable/timeout/EINVAL/resource errors refuse.
- Source ordering tests preserve original FD3/4 handoff before policy and Python exec after denial checks.
- The private production main gate is behaviorally tested with install/check/proof failures
  requiring zero exec-sentinel calls; the real fixed Command is supplied behind that gate.
- Test green grants no native/GPU/network admission.

## Work Guidance

Keep source-owned targets and production policy fixed; use no arbitrary endpoints.

## Verification

Run the parent locked/offline Rust workspace tests.

## Child DOX Index

None.
