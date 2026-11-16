# CASE parent output controls

## Purpose

Own actual-FD tests of compiled CASE-only Processing parent ownership.

## Ownership

Inherits ../AGENTS.md; cases.rs is included by main.rs only under cfg(test).

## Local Contracts

- Keep this module nested so Cargo does not discover it as an integration crate.
- Isolated account-only child fixtures require CaseFreeze root0:0/0755 parent;
  legacy Processing profiles retain USER:USER/0755. Invalid parents refuse without
  repair. Original parent/output inodes and strict private0700 final FD remain.
- These compiled tests execute no worker, privilege drop or native ML and grant
  no SageMaker mount, built-image or case-publication admission.

## Work Guidance

Preserve all existing assertions and actual filesystem/descriptor observations.

## Verification

Run default/native locked offline Rust workspace tests on the account build host.

## Child DOX Index

None.
