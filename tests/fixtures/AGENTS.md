# tests/fixtures

## Purpose

Own bounded regression captures containing no executable workload or model tensors.

## Ownership

Inherits ../AGENTS.md; child contracts identify capture provenance and exclusions.

## Local Contracts

- Fixture bytes are test data, never authority or native admission.

- `sft_producer_cold_graph.py` is the neutral cold-process product bootstrap
  fixture. It parses only synthetic metadata/byte streams and invokes ordinary
  read-only product verification; no model, candidate, benchmark oracle or hostile
  process is executed. Captured fixture state contains no credentials or grants.

- `sagemaker_training_state_change_real.json` holds seven real DescribeTrainingJob records
  (October 2026 and a GPU test run of 2026-11-10) placed in the documented "SageMaker Training Job
  State Change" envelope, plus the measured invocation pattern of the deployed EventBridge target.
  The record shapes, statuses, instance types, times, durations and byte counts are the recorded
  ones. Every job name, the account, the bucket, session, role and image names, every digest and
  commit value and the free text are made-up values. It is shape data for the callback tests, not
  a record of a run.

- `sagemaker_instance_preference_readback_real.json` holds the stored request ResourceConfig and the
  DescribeTrainingJob ResourceConfig, status and secondary status sequence of four real training jobs of a
  GPU test run of 2026-11-10 (three preference-list jobs and one fixed-type job), read with botocore
  1.43.127. Job names are neutral; no account, bucket or role value is kept. It is readback shape data, not a
  record of a run.

- `sagemaker_training_tag_read_real.json` (added 2026-11-11) holds real records of the first real normal-path
  run of 2026-11-11 UTC and of one real preference-list job: the SageMaker calls of the runtime execution role from
  CloudTrail (71 DescribeTrainingJob answered, 71 ListTags on the fine-tune job refused with AccessDenied, 3 ListTags
  on a processing job answered), the refusal text, the held report's diagnostic, the stored native request of the
  fine-tune with its DescribeTrainingJob and ListTags answers after completion, and every distinct DescribeTrainingJob
  answer of the list job from Pending to Stopped (botocore 1.43.127, read every 10 s). The account id is replaced by
  the word ACCOUNT. The stage name, every job name with its digest or timestamp part, the bucket, role and image
  names, the session, deployment and role-session ids, the image digests, the manifest digest and the source commit
  argument are made-up values with the length of the real ones, so each job name keeps its
  length and every record keeps its keys and value shapes. The record datetimes are the recorded ones.
  It is shape data for the tag-read and readback tests, not a record of a run.

## Work Guidance

Preserve reviewed capture bytes and keep generated/private artifacts outside source.

## Verification

Use the owning regression and recursive DOX lint.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `model_weight_manifest/AGENTS.md` | Public pinned model index metadata oracle |
