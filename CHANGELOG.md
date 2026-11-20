# Changelog

This file lists notable changes for users and maintainers. The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

Each release has a version `N.NN` held in the root `VERSION` file. `N` is the release line. `NN` is a counter of at least two digits that rises by one with every release that changes what a user or maintainer sees. The package version is the same number written as `N.<NN as an integer>.0`. For example, release `0.06` is package version `0.6.0`.

The numbers `0.01` to `0.05` were assigned afterwards. Releases `0.01` to `0.05` and the first publication of `0.06` carried the package version `0.1.0`.

## [Unreleased]

## [0.08] - 2026-11-20

### Added

- Added a requirement in `tests/test_changelog_shape.py` for the introduction of `CHANGELOG.md` to keep its note on release numbers assigned afterwards word for word, exactly once, and as one paragraph.

### Fixed

- Fixed possible timing-dependent failures of `tests/test_case_unknown_observation_budget.py` and `tests/test_normal_research_profile.py` by taking the read counts, setting the native state and replacing `adapter.artifacts` under the backend's `_warm_lock`, and by changing the policy or guard version before completing the fixture.
- Fixed timing-dependent failures of `tests/test_normal_run_console_silent.py`, in which a busy machine let the SDK add a measured `timeToFirstByteMs` to the messages of one of two compared runs, by setting the clock reading in `strands.event_loop.streaming` to zero for the scripted model through the fixture `scripted_model_has_no_measured_first_byte_time` and by adding a test with a slow first stream event that requires unchanged message metadata.
- Fixed timing-dependent failures in `tests/test_case_unknown_observation_budget.py`, `tests/test_model_unknown_observation_budget.py` and `tests/test_runtime_ambiguous_model_capture.py` by replacing the fixed pause after the eighth failed or unknown read with a wait for `work.api_returned` under the existing 5 s bound.
- Fixed a timing-dependent failure of the invalid-payload case in `tests/test_warm_model_refusal_diagnostics.py` by completing the fixture and changing the stored payload under the backend's `_warm_lock`.
- Fixed a timing-dependent failure in `tests/test_warm_model_closure.py` by wrapping the store's `_submission_lock` in `TurnTakingLock` so the second same-catalog call waited for the first to release the lock before taking the real lock.
- Fixed timing-dependent failures in `tests/test_sdk_record_form.py` and `tests/test_normal_completion_sdk_none_probe.py`, in which a busy machine let the SDK add a measured `timeToFirstByteMs` to scripted messages, by giving both files the fixture `scripted_model_has_no_measured_first_byte_time` that `tests/test_normal_run_console_silent.py` uses.
- Fixed an order-dependent failure of tests that build a coding report after a test that used the `native` fixture of `tests/test_coding_case_native.py` in the same process, by loading `amlas.arbiter.coding` and `amlas.evaluation.coding_pair` in that fixture before it changes the case profile's source hash.

## [0.07] - 2026-11-16

### Changed

- Changed the container recipes `Dockerfile`, `Dockerfile.processing`, `Dockerfile.model-acquisition` and `Dockerfile.coding-pair` to copy the root `VERSION` file, so the application in those images reports the package version instead of `unknown`.
- Changed the README clone command, CI badge and project layout and the GitLab synchronization job to the repository's new name, `AMLAS--sample-ml-engineering-agent-swarm`.
- Changed the release rules in `AGENTS.md` to tie human approval to the content the approver saw instead of an exact tree hash; changes limited to conflict-free teammate merges or edits required by the standing release rules need no new approval when the release checks pass on the new tree, while other content changes still do.

### Fixed

- Fixed timing-dependent failures of the continuation tests by making `start_original_target` in `tests/case_prep_continuation_fixtures.py`, `tests/test_normal_research_profile.py`, `tests/test_runtime_pair_original_capture.py` and `tests/test_sdk_pending_handoff.py` wait until the run had read or retained the original before reading its state.

## [0.06] - 2026-11-16

### Added

- Added `inspect_huggingface_dataset`, `acquire_huggingface_dataset`, `get_dataset_acquisition`, `inspect_acquired_dataset` and `get_dataset_inspection` for pinned public datasets, acquiring one file from an exact revision through SageMaker Processing by default.
- Added `list_coding_preparation_profiles`, `prepare_coding_dataset` and `get_coding_preparation` for coding-data preparation using canonical conversation JSONL `openai.explicit_conversations.v1` and CommitPackFT `bigcode.commitpackft.python.change_sft.v1` profiles.
- Added `list_model_weight_catalog`, `acquire_model_weights` and `get_model_weight_acquisition` for model weights, with one catalog entry, `qwen3_5_9b`, for Qwen/Qwen3.5-9B.
- Added `submit_qwen_sft`, `get_qwen_sft` and `reload_qwen_sft` for LoRA fine-tuning, with `fit_v1` using 32 optimizer steps, `full_v1` using 1024, and bounded training-history summaries.
- Added `list_coding_case_catalog`, `freeze_coding_cases`, `get_coding_case_freeze`, `bind_coding_generation_spec`, `submit_coding_generation`, `get_coding_generation`, `resolve_coding_evaluation_spec`, `submit_coding_pair` and `get_coding_pair` for Arbiter-graded base and tuned model evaluation on the same 164 HumanEval cases.
- Added `data_preparation_flow`, `coding_case_freeze_flow`, `qwen_sft_flow`, `coding_generation_flow` and `coding_pair_flow`, raising the registered flow count from 21 to 26.
- Added backend selectors `DATA_ACQUISITION_BACKEND`, `DATA_INSPECTION_BACKEND`, `DATA_PREPARATION_BACKEND`, `MODEL_WEIGHT_ACQUISITION_BACKEND`, `QWEN_SFT_BACKEND`, `CODING_CASE_FREEZE_BACKEND`, `CODING_GENERATION_BACKEND` and `CODING_PAIR_BACKEND` for the new dataset, model-weight, fine-tuning, case-freeze, generation and paired-evaluation ports.
- Added managed-worker container recipes `Dockerfile.processing`, `Dockerfile.inspection`, `Dockerfile.preparation`, `Dockerfile.model-acquisition`, `Dockerfile.qwen_sft`, `Dockerfile.case-freeze` and `Dockerfile.coding-pair` for dataset, model and evaluation jobs.
- Added prompt-run checkpoints, continuation after worker replacement, runtime recovery records in final reports, and opt-in training-completion wakeups through `AMLAS_NORMAL_CALLBACK_ENABLED=true`, off by default.
- Added SageMaker instance preference lists for fine-tuning, reload and generation jobs, capacity-refusal details naming what held instance types, and generation or paired-evaluation resubmission after refusal.
- Added log lines naming the stage where a prompt run waited or was refused.
- Added `execution_provider` to managed flow results and plan previews, identifying the provider that ran the SageMaker operations.
- Added tools in `scripts/ci_shard/` that split full CI's Python phase across several pytest containers and merged results into the files written by a single run.
- Added hash-locked CPU test environment installation through `scripts/provision_qwen_cpu_tests.sh`; both CI paths ran it before Python tests and also ran `processing_bootstrap` Rust checks.
- Added runtime image deployment by digest through `IMAGE_DIGEST` for `scripts/lib/deploy_agentcore_runtime.sh`.

### Changed

- Changed `scripts/setup.sh` runtime image builds to require a Linux EC2 host in the selected AWS account, with `--skip-agent-runtime` used on other hosts.
- Changed `scripts/setup.sh` to publish the SageMaker execution role ARN and selected data-acquisition and data-inspection images to SSM parameters.
- Changed runtime deployment to pack part of an environment exceeding the 4,000-byte service limit into one variable, expanded when the runtime starts.
- Raised the minimum dependency version for the `agents` extra to `botocore>=1.43.90`.
- Changed prompt runs to retain up to 100 chat messages, suppress agent stdout text, retry transient model failures, and bound polling attempts when operation state cannot be read.
- Changed baseline training results to report `waiting`, `failed` or `unassessed`, and deploy results to report `waiting` during endpoint creation.
- Changed the data collection agent prompt to cover inspection and pinned acquisition of public datasets.
- Updated both CI paths to provision the pinned public evaluation file for Python tests through `scripts/provision_qwen_cpu_tests.sh`, exported as `AMLAS_TEST_HUMANEVAL_PARQUET`, and run `processing_bootstrap` tests with one test thread (`RUST_TEST_THREADS=1`); moved the GitLab `verify` and `processing-bootstrap` jobs to large runners.
- Changed `amlas --version`, `amlas info version` and `amlas.__version__` to report the package version of the release, read from the root `VERSION` file, instead of a fixed `0.1.0`.

### Fixed

- Fixed cold runtime routes that ran before AgentCore configuration was loaded.
- Added final reports naming the job state for prompt runs whose fine-tuning jobs ended Failed or Stopped after a worker stop.
- Changed resume requests arriving after the original worker finished the run to receive completed responses.
- Fixed coding-task specification tools returning classification stubs by making the tools state the next call to make.
- Fixed baseline training registration to follow the `passes_acceptance` result of the Arbiter report.

## [0.05] - 2026-10-19

### Added

- Added the AWS reference architecture diagram `architecture.svg` to the README.

### Fixed

- Fixed the README solution summary rendering as a quote block, making it an ordinary paragraph.

## [0.04] - 2026-10-15

### Added

- Added maintained-directory `AGENTS.md` contracts, expanding from 4 files to 63, root `CLAUDE.md`, and `scripts/dox_lint.py` as the first check in both CI paths.
- Added `amlas.api.run_prompt` to send one natural-language prompt through the chat engine from the Python API.
- Added `text_generation` experiment tasks and solution surveys listing only text-generation models for generation tasks.

### Changed

- Changed chat and agent defaults to Claude Sonnet 5.5, with the separate reviewer using GPT-6.1 Sol through `AMLAS_MODEL_REVIEWER`.

### Fixed

- Fixed the omission of plural numeric prediction objectives from generated specifications.

## [0.03] - 2026-10-14

### Added

- Added `scripts/setup.sh --deployment-prefix`, defaulting to `amlas`, with lowercase names of at most 24 characters for multiple deployments sharing one account.
- Added `scripts/lib/prepare_host.sh` to prepare fresh Linux hosts for setup, with `--check` reporting missing requirements.
- Added `scripts/setup.sh --ask-tavily-key` to request an optional Tavily key and supported keyless Tavily Search, which `AMLAS_TAVILY_KEYLESS=false` turned off.
- Added `research_fetch_urls` to read user-supplied public URLs, limited to three pages and 256 KiB per page.
- Added setup checks for the selected Region's five-trail CloudTrail limit before infrastructure writes.

### Changed

- Changed runtime deployment to report success only after the DEFAULT endpoint reached READY on the accepted runtime version.
- Changed README installation sections to cover prerequisites, host preparation, optional research access, developer verification and teardown, with a CI badge and sample deployment wording.

### Fixed

- Fixed the default SageMaker pipeline template failing to compile because its Preprocess step definition was incomplete.
- Fixed empty AWS profile values loaded from `.env` breaking CLI discovery of deployed application resources.
- Added missing test-suite prerequisites to the GitHub Actions workflow and removed its unsupported temporary-directory setting.

## [0.02] - 2026-10-13

### Added

- Added `scripts/setup.sh` options `--profile`, `--account-id`, `--region`, `--runtime-name` and `--runtime-role-name` to select AWS deployment settings, the managed runtime and its execution role.
- Added `scripts/setup.sh --preflight-only` for read-only account, service and model metadata checks without AWS writes, and `scripts/setup.sh --skip-harness` to omit HarnessStack.
- Added `AMLAS_RUNTIME_POLL_ATTEMPTS`, defaulting to 60, and `AMLAS_RUNTIME_POLL_SECONDS`, defaulting to 10, to control runtime deployment readiness waits.
- Added a `tests/` regression suite, GitHub Actions workflow, GitLab CI pipeline and root `AGENTS.md` repository contract.

### Changed

- Changed Region resolution to explicit argument, `AWS_REGION`, `AMLAS_AWS_REGION`, `AGENTICML_AWS_REGION`, `AWS_DEFAULT_REGION`, then selected profile, and CDK resource naming to use the deployment Region.
- Changed setup to publish the runtime ARN to SSM only after the runtime was verified READY.
- Changed free-form chat to one tool-bound agent, removed raw Bedrock Converse chat, and added one warning when the local keyword responder answered without an LLM.
- Changed the managed-runtime image to Linux ARM64 with CPU PyTorch wheels through `runtime-cpu`, and added README image-build requirements.

### Fixed

- Fixed default relative output failures in the managed runtime.

## [0.01] - 2026-09-15

### Added

- Added a chat REPL, Textual TUI through `amlas-ui`, `amlas` CLI and Python API over one shared capability set.
- Added 21 registered flows covering research, business framing, solution survey, data collection, data quality, imputation, feature engineering, baseline training, hyperparameter tuning, evaluation, readiness, deployment, monitoring and retraining.
- Added 25 backend ports, each with SageMaker and local adapters selected through `*_BACKEND` environment settings.
- Added a deterministic Arbiter that graded models and made accept or reject decisions without an LLM.
- Added offline data-quality profiling, missingness diagnosis, imputation, synthetic-data generation and local training that run without AWS access.
- Added Plan Mode as a read-only preview with cost and impact estimates before spend-incurring actions.
- Added `scripts/setup.sh` to deploy CDK stacks and the managed agent runtime, and `scripts/teardown.sh` to remove them.
