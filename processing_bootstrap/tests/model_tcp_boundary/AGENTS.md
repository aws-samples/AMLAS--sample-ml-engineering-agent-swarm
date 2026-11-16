# Raw model TCP boundary tests

## Purpose

Own fixed TCP syscall and controlled kernel oracles.

## Ownership

Inherits ../AGENTS.md; test-only files are excluded from production images.

## Local Contracts

- Private cfg(test) syscall fakes cannot select production ports or grant authority.
- FD ownership checks run in isolated exact-test children; controlled descriptor reuse
  demonstrates that raw numbers are not historical ownership evidence.
- The kernel probe verifies/saves the original bounding mask, validates final state
  against it, and requires an identical unrestricted dropped-child connection/bind
  baseline before attributing restricted80/bind denial to this module.
- Actual kernel refusal is recorded as unsupported/refused, never a policy success.
- Fixed loopback controls only; no external endpoints, AWS or credential values.

## Work Guidance

Use red-first exact ABI/layout/order/refusal/FD cleanup assertions.

## Verification

Run the parent locked Rust workspace checks on developer hosts.

## Child DOX Index

None.
