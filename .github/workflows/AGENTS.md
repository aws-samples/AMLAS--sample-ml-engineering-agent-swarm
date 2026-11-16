# .github/workflows

## Purpose

Run public-release checks with immutable action references and minimal permissions.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- The complete Python verify step provisions `scripts/qwen_cpu_test.requirements.lock` and the five immutable public metadata files with `provision_qwen_cpu_tests.sh`, transports its fixed environment through `GITHUB_ENV` and runs the unchanged module-pytest command in the next step. The same step transports `AMLAS_TEST_HUMANEVAL_PARQUET`, the path of the pinned public evaluation file that the provisioning script fetches and verifies, and stops when the provisioned environment does not carry it. CPU dependencies are audited separately with strict hashed requirements and OSV (official CPU local version is absent from PyPI); GPU/runtime requirements and required Rust profiles remain unchanged.

- Maintain the interfaces and file formats owned by the modules listed below; parent contracts apply.
- Dependency audit inputs retain uv-exported hashes; `pip-audit --disable-pip`
  audits that exact pre-resolved lock without constructing a nested resolver environment.
- The fine-tune image lock `src/amlas/training/qwen_sft.requirements.lock` is audited in its
  own step with the same strict hashed-requirements OSV command as the CPU test lock.
- Owned files: `public-release.yml`.
- The independent `processing-bootstrap` job runs the digest-pinned Rust vendor
  image directly; it does not build or commit a Docker image. Preparation copies
  its public toolchain into workspace tmp and adds the exact-toolchain formatter
  there. Both strict profiles then use that same mounted toolchain in a network-
  disabled container with read-only source and separate output directories.
- Check the strict default and explicit `sagemaker-native-v1` feature separately,
  with locked offline tests/builds and distinct target directories. Neither
  profile's checks may be skipped or overwrite the other's release artifact.
- The `processing-bootstrap` job's network-disabled container runs tests with
  `RUST_TEST_THREADS=1`, drops SETPCAP, NET_RAW, MKNOD and SETFCAP to use the
  processing image's capability set, and sets the open-file limit to 65536. The
  binary's unit tests share one fixed directory and its lock. The native startup
  test refuses any capability outside that set. The descriptor handoff test needs
  an open-file limit above 4096.
- The `verify` job has a 180-minute limit; `processing-bootstrap` has a 30-minute limit.

- `public-release.yml` supplies required source and regression checks. Its success
  alone is not public-release qualification: the root's GitHub release gate also
  requires the exact sanitized candidate's deployed core workflow and existing-feature
  client/integration checks before any GitHub publication. Retain those receipts in
  sibling internal devdocs; never publish cloud credentials, model/data artifacts,
  private evidence or account-bound receipts into GitHub workflow artifacts.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
