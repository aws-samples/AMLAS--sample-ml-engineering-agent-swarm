# src/amlas/memory

## Purpose

Connect agent hooks to managed memory and session state.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `hooks.py:MemoryClientProto` — Minimal surface of `bedrock_agentcore.memory.MemoryClient` we actually use.
- `hooks.py:MemoryHookProvider` — HookProvider that pulls history on init and persists new messages on turn boundaries.
- `hooks.py:_record_messages` — the messages handed to `create_event` are copies without the SDK's top-level
  `tracking_id` (present on every live message from the locked Strands 1.65.0), so the Memory client receives the
  shape Strands 1.42.0 produced. Nothing else is removed, a nested key of that name stays, and the live SDK
  messages are never modified. The helper is local to this package and imports nothing from `tui/chat`.
  `tests/test_memory_hook_record_form.py` pins this with a recording client and a real SDK run; it says nothing
  about what a real Memory resource accepts.
- Owned files: `__init__.py`, `hooks.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
