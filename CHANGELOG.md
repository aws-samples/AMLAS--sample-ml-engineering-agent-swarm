# Changelog

This file lists notable changes for users and maintainers. The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

Each release has a version `N.NN` held in the root `VERSION` file. `N` is the release line. `NN` is a counter of at least two digits that rises by one with every release that changes what a user or maintainer sees. The package version is the same number written as `N.<NN as an integer>.0`. For example, release `0.06` is package version `0.6.0`.

The numbers `0.01` to `0.05` were assigned afterwards. Releases `0.01` to `0.05` and the first publication of `0.06` carried the package version `0.1.0`.

## [Unreleased]

## [0.14] - 2026-11-26

### Added

- Added `tests/test_history_admission_form.py` to check how stored messages were sent to the guardrail's input check during a restore and for the earlier conversation of a chat turn. Text blocks were sent as their text. Tool results were sent as the compact JSON of their content lists. Machine-written values were sent on their own in stored order. A message with an unknown field was sent whole. A stored message that the scripted guardrail blocked only as one whole JSON text was admitted when none of its separate texts was blocked. For 400 generated messages, every non-empty string, number and tool-call key stood in exactly one sent text. A blocked or masked text held the restore or the turn. The cases used scripted guardrail answers.
- Added `tests/test_input_admission_repeated_value.py` to check how the input check handled a detected value that appeared more than once in the checked text. The guardrail's masked text was forwarded in cases where both copies were masked and where only one was masked. An answer that changed other text was refused. The turn continued after a masked owner prompt, research page or read-only tool result was forwarded. A masked text that had to stay unchanged held the turn. The cases used scripted guardrail answers.

### Changed

- Changed `get_prompt_run_status` and `resume_prompt_run` to say why a restore was held when the deployed guardrail's input check stopped the stored conversation of a run after its worker was replaced. A `resume_prompt_run` request that ended with `resume_state` `failed` because of that hold sent one more `prompt_run_status` event after it. In that event `found` was false; a caller that had read `found` false as nothing to resume should check `run_state` for `restore_held`. The status answer kept its keys and values and added the same three keys: `run_state` with `restore_held`, `message` and `restore_hold`. The three keys gave the position and kind of the held message. They named each content filter or personal-data entity the guardrail reported, with each filter's category and confidence. They contained no matched value or identifier. The only checked text they contained was the held message's role name, `user` or `assistant`. The run stayed resumable. The held restore wrote no report. The fine-tuning job, its results and every stored report were kept. The restore of a run with a finished fine-tuning job checked the stored conversation twice and, as before, recorded its completion claim between the two passes. A restore held in the second pass had already recorded that claim. The claim started nothing and changed no other stored data; the next restore took the same claim. The runtime process that ran the restore kept the cause until the next restore of that run started in the same process. `tests/test_normal_restore_held_cause.py` checked these answers with scripted guardrail answers.

### Fixed

- Fixed runs that were held when a deployed guardrail masked a prompt, a research page or the result of a read-only tool that contained the same detected value more than once: when the masked answer passed the other input checks, the text went on in the guardrail's masked form and the run continued. A masked answer that failed one of those checks still held the run.
- Fixed prompt runs that stayed held at the restore after their worker was replaced during fine-tuning because a deployed guardrail gave a whole stored message a verdict that none of its parts carried. A blocked, masked or unassessed text still held the run.

## [0.13] - 2026-11-25

### Added

- Added `tests/test_normal_run_ended_status.py` to check the status and resume answers of runs that ended before fine-tuning started. The cases included a run whose last response was its waiting text, reports that could not be verified, records in the fine-tuning stage, another session, a deployment whose approved actions did not include fine-tuning, and a status poll made before the run's worker had finished.
- Added `tests/test_held_report_guardrail_assessment.py` to check the report of a prompt run that was held after a guardrail stop on its first model call or on a call after a tool cycle, with a blocked policy trace on the input side or the output side, and of a run that was held for another reason after an earlier model call carried a trace. The tests checked that the report data carried `guardrail_assessment` with its records, that the action stayed `input_admission_held` with the fixed message and no `guardrail_outcome`, that a later `get_prompt_report` request returned the same report, and that a stop without a trace named no guardrail. Two fixtures had the shape that a stopped call and an unstopped call of the model service returned, and the tests checked that the report of such a stop named the blocked entity in its records. One case checked that a personal-data policy whose `regexes` value was a list with an entry, or was not a list, kept its entity records with the status `partial`.
- Added a test to `tests/test_bedrock_guardrail_stop_without_trace.py` that checked that a guardrail stop on the model call that resumed a run, with a trace that named only a masked entity, was reported as `guardrail_intervened` with the provider's text and held nothing. Two cases in `tests/test_held_report_guardrail_assessment.py` checked the same for the first model call of a run and for a call after a tool cycle. Three more cases there checked through the runtime entry, for the first model call, a call after a tool cycle and the call that resumed the run, that such a run had ended with its report kept: a later `get_prompt_report` request returned the same report, the status and resume requests started no work, and the same prompt started a new run.
- Added `tests/test_normal_resumed_run_held.py` to check a guardrail stop in the worker that resumed a normal run after fine-tuning. The tests covered a stop on the first model call and a stop after a tool cycle that submitted a job. They checked the held report, its sentence about earlier actions, and the runtime recovery record. They checked later report, status, and resume replies and the order of the steps that ended the run. They also checked that the resume answered `failed` and stored no report when the backend, `Agent`, claim, snapshot, system prompt, approval, or run record changed. They checked that a hold set by the runtime's own check before a model call ended the run with the same held report and no model call. They also checked that the held report for a stop with the trace form the model service returned for a blocked e-mail address had a `partial` assessment with one record for the blocked entity.
- Added `base_pass_count_range`, `tuned_pass_count_range` and `delta_pass_count_range` to the summary of the paired coding evaluation. A case that ends in `timeout` or `infrastructure_error` still counts as not passed, and the new fields give the pass counts and the difference if such cases had passed.
- Added `result_statement` and `next_step` to the result of `get_coding_pair` for an evaluated pair: fixed sentences with the pass counts, the cases without a verdict, the range of the difference and the time limit.
- Added `tests/test_paired_guest_wrapper.py` with nine tests. Seven parsed the two remote texts of the paired coding evaluation without running them and checked the redirect before the answer and the restore before the reply, the `finally` of the value call, the answer's source byte for byte, that no module was bound beyond `json`, `math` and two private names, that the parent sent the built texts, and that no call followed a failed load in its session. Two checked that the parent took a reply only when it was the whole standard output and ended the case as `infrastructure_error` when other output stood before or after a reply.
- Added `tests/test_input_admission_supplementary_characters.py` to check that the input check admitted text with a character outside the Basic Multilingual Plane when the answer showed full coverage and no intervention. The cases covered one, two or five such characters. The scripted answers counted each such character as two 16-bit code units. A masked answer for text with such a character was forwarded in its masked form. The input check returned UNKNOWN with NATIVE_INVALID when `guarded` was below `total` or when `total` matched the number of code points. It gave the same answer when `total` was above or below the text's length in 16-bit code units. Text with a surrogate code point gave UNKNOWN with INVALID_INPUT without a guardrail call.
- Added `tests/test_research_result_supplementary_characters.py` to check that three research results with five fetched pages were admitted when each page held a character outside the Basic Multilingual Plane. The research plan was consumed, and the turn continued to the next tool. A masked page reached the model without the name and address, and the turn continued. A prompt with such a character was admitted and the turn continued. A research page with a short coverage count held the turn with the fixed hold text.

### Changed

- Changed `get_prompt_run_status` to add `run_state` `ended`, `ended_action`, and `message` for a run that ended before fine-tuning started, once its stored report was verified. For such a run, the last event of `resume_prompt_run` changed from `unavailable` with `found` false to that status answer with `found` true. A caller that had read `found` false as nothing to resume should check for `run_state` `ended`. When the stored report could not be verified, the status answer was `unavailable`. A status poll added the three keys only once the run's worker had finished, because report requests answered `unavailable` until then. Both cases of `tests/test_normal_run_held_final_state.py` required the three new keys and a resume request that ended with that status.
- Changed the README subsection `Held runs and evaluation timeouts` to describe guardrail stops on the first model call of a prompt run, on a call after tools ran and in the worker that took over after fine-tuning, including the held report, `get_prompt_run_status` and `resume_prompt_run` replies and where the fine-tuning job's output stayed. It covered holds without a guardrail stop, the `AMLAS_INPUT_HOLD` runtime log line, stops with the trace off, which reports carried `guardrail_assessment` and what it named, what was kept and how to start again when a stop's trace named no blocked policy, and the paired coding evaluation's range fields, fixed sentences of `get_coding_pair` and answers that wrote to standard output.

### Fixed

- Fixed the report of a prompt run that was held after a guardrail stop on its first model call or on a call after a tool cycle: the report data carried `guardrail_assessment` when the guardrail returned a trace, as the report of a held resumed call already did. A run that was held for another reason after its model calls carried the assessment of that turn in the same way.
- Fixed `guardrail_assessment` for a guardrail trace whose personal-data policy had `piiEntities` and no `regexes` list, the form the model service returned for a blocked e-mail address. The assessment kept the entity records with the status `partial`, where it had the status `invalid` and no record.
- Fixed a resumed run whose model input was held at the admission boundary during its continuation: the run ended with the held report, readable through `get_prompt_report`, where the resume had answered `failed` and no report had described the stop.
- Fixed the paired coding evaluation recording `infrastructure_error` for an answer that wrote to standard output. The answer's output is discarded while it loads and runs, and the case gets the verdict of its test.
- Fixed an input check that rejected valid guardrail answers and held runs when the checked text contained a character outside the Basic Multilingual Plane, such as an emoji.

## [0.12] - 2026-11-24

### Fixed

- Fixed `processing-bootstrap` in `.github/workflows/public-release.yml` so public project maintainers could still run the job when Amazon ECR Public refused the pull and Docker Hub served the same pinned compiler image. A first step fetched the image from Amazon ECR Public, asked Docker Hub for the same tag and digest only when that pull failed, printed which source answered, and failed when both refused.

## [0.11] - 2026-11-22

### Added

- Added `tests/test_chat_llm_absent_warning.py` with four tests that checked that `_warn_chat_llm_absent` logged one `WARNING` naming `strands-agents` and `pyproject.toml [project.dependencies]`, with no version comparison operator, dotted number, or number, operator, or extras bracket directly after the package name. They also checked that `[project.dependencies]` declared `strands-agents` exactly once, that a second call logged nothing, and that the next call after `reset_degradation_warning` logged the same text again.

### Changed

- Changed `_warn_chat_llm_absent` in `src/amlas/tui/chat/engine.py` to name `strands-agents` in place of `strands-agents>=1.1`, leaving the minimum version only in `[project.dependencies]` in `pyproject.toml`. The warning was logged when no real `strands` package could be imported and the local keyword fallback answered free-form chat.

### Fixed

- Fixed a timing-dependent failure in `tests/test_public_dns_pipe.py`, where wall-clock budgets of 1 s or less for a real forked resolver process could make a correct run on a busy machine fail with "public DNS validation exceeded the retrieval timeout" or let a case that accepted any refusal pass on that timeout. Cases that checked a rule used a 60 s `LIVENESS` budget that only ended a hung case, and every expected refusal from `_checked_addresses` was matched by its reason. The cases that called `_decode_dns_wire` directly still expected only `PublicFetchError`. In stall cases the resolver process slept for `STALL` and the tests required a refusal within `LIVENESS`. The partial-frame cases and the timeout case checked the waits the parent asked for without reading a clock. The partial-frame cases required the first wait to be the poll that met the partial frame and all later waits to total less than 1 s. The timeout case required each poll to ask for no more than the case's budget and all waits to total less than 1 s. The timeout case held `Process.start` until the resolver process reported that it ignored `SIGTERM` and required exactly one `SIGKILL`, and a fixture ended any resolver process left behind by a failing case.

## [0.10] - 2026-11-21

### Added

- Added coverage in `tests/test_bedrock_guardrail_stop_without_trace.py` and the new `tests/test_normal_run_held_final_state.py` for a guardrail stop on the model call that resumed a normal run, with and without a trace naming the blocked policy. The tests checked that the report stayed readable, status stayed `enrolled` with `resume_ready` false, and report, status, and resume requests started no work and wrote nothing, with or without resume memory.
- Added `tests/test_input_admission_unencodable_text.py` to check full chat turns with tool text, tool JSON, history, or prompts that could not be encoded as UTF-8. With an input guardrail configured, the turn was held and neither the guardrail nor the model received that text. Without one, the turns completed with the text as given, and ordinary text was admitted with a guardrail configured.
- Added `tests/test_caller_history_stays_unchanged.py` to check that chat turns left caller history dictionaries unchanged, that the builder gave the SDK an equal list for a supplied history and no messages for an empty or absent one, and that no agent build of an uninterrupted or restored normal run, or of a normal run that was resumed while its job still ran, received a history.
- Added the subsection `Held runs and evaluation timeouts` to `README.md` to explain what a held run shows, how to start again, how text that cannot be encoded as UTF-8 is handled, and how `timeout` and `infrastructure_error` outcomes bound the pass counts of the paired coding evaluation.

### Changed

- Updated `src/amlas/evaluation/AGENTS.md` to explain that `coding_pair_runner.py` records `timeout` when a case has no verdict by its limit, including a remote call that does not return in time or an answer that runs too long, and that nothing is tried again.
- Updated the `uv` tool version from 0.8.8 to 0.13.4 in `Dockerfile`, `.gitlab-ci.yml`, and `.github/workflows/public-release.yml`.
- Raised the minimum versions in `pyproject.toml` to `strands-agents` 1.48.0, `python-dotenv` 1.2.2, `lightgbm` 4.6.0, `gitpython` 3.2.1, `pillow` 13.0.0, `pyasn1` 0.6.5, `python-multipart` 0.0.31, `setuptools` 84.1.0, `bedrock-agentcore-starter-toolkit` 0.1.13, `pyarrow` 23.0.1, and `pytest` 9.0.3. The versions in `uv.lock` did not change, and `tests/test_repository_policy.py` was updated to check the new `pytest` minimum.
- Regenerated `uv.lock` with `uv` 0.13.4. The lock revision went from 3 to 5, environment markers were written in a shorter form, and four `greenlet` 3.4.0 wheel entries for s390x and riscv64 were dropped. No package version or source changed, and the lock kept all 208 package entries.

### Fixed

- Fixed `build_agent` in `src/amlas/_build.py` to pass a new dictionary per history message to the SDK, so that a caller's history dictionaries stay unchanged across chat turns. The copy is one level deep.

## [0.09] - 2026-11-20

### Added

- Added a test in `tests/test_normal_model_campaign_flow.py` for an overlapping model-weight submission to return `busy` without a second create.
- Added a test in `tests/test_sdk_record_form.py` that gives scripted model calls a fixed measured `timeToFirstByteMs` and requires stored records to keep it, `normal_conversation_size` to count it, and two runs to give equal records.

### Fixed

- Fixed six test failures in `tests/test_generated_coding_spec_tool_connection.py` with `--basetemp` outside the system temp folder by using the fixture `spec_folder` to set `AMLAS_WORKSPACE_DIR` to the test's folder.

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
