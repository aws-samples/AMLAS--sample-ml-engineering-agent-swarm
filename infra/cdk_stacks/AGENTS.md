# infra/cdk_stacks

## Purpose

Own the maintained infra/cdk_stacks boundary and its checked-in inputs and outputs.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

<!-- normal-callback-stack-optin-contract:begin -->
- Normal callback opt-in defaults false and preserves complete legacy templates.
  True binds current callback role Arn/RoleId, exact selected runtime/version/route,
  GetAgentRuntime plus GetAgentRuntimeEndpoint and InvokeAgentRuntime on their
  required exact runtime/endpoint resources. The same jobs table gains
  NEW_AND_OLD_IMAGES and one batch1/LATEST mapping to the existing callback; arm it
  before a fresh run. Keep actual SageMaker rules separate, no new service/role or
  self-minted authority. Offline synth grants no native authorization or wake proof.
<!-- normal-callback-stack-optin-contract:end -->
- [Added and changed 2026-11-10] With the opt-in true,
  AsyncTrainingStack writes the validated literal runtime ARN from its constructor
  into the callback role policy (GetAgentRuntime, GetAgentRuntimeEndpoint,
  InvokeAgentRuntime) and the Lambda `AGENTCORE_RUNTIME_ARN` value. It still declares
  the `AgentRuntimeArn` parameter with its earlier shape (String, the literal as
  default, the same description), and nothing reads it: no `Ref`, no `Fn::Join`, no
  `RuntimeArnSupplied` condition. Why the literal: CloudFormation applies a parameter
  default only when a stack is created and reuses the stored value on later updates,
  so enabling the callback on an existing stack kept the stored empty value and IAM
  refused the empty Resource. Why the parameter stays declared: a template without it
  drops the stored value, so a later deploy with the opt-in off would read the empty
  default and grant InvokeAgentRuntime on every runtime of the account and Region
  where the stack held one runtime ARN before; and CloudFormation refuses a deploy
  command that passes a parameter the template does not declare. A stored or passed
  value has no effect while the opt-in is true; CloudFormation keeps it without any
  policy check, and the next legacy deploy reads it. Pass the runtime ARN or nothing:
  an explicitly empty value passed during the opt-in is stored, and a later legacy
  deploy then uses the account/Region fallback. The legacy path (opt-in false or
  absent) keeps the parameter, the condition and the account/Region fallback
  unchanged. Do not make an opt-in consumer read the parameter, do not remove it from
  the opt-in template, and do not carry a synth-time value to an existing stack
  through a parameter default.

- RuntimeStack grants CASE ownership-tag reads through a separate ListTags-only
  `SageMakerProcessingTagRead` statement scoped to the existing hyphen deployment
  prefix processing-job ARN. Preserve global control-plane actions and all other
  statements; the exact prefix wildcard has its own IAM5 ownership-read rationale.

- [Added 2026-11-11] RuntimeStack grants the fine-tune, reload and generation
  ownership-tag reads through a separate ListTags-only `SageMakerTrainingTagRead` statement
  scoped to the hyphen deployment prefix training-job ARN, with its own IAM5 ownership-read
  rationale. Describe and stop keep their statement; no other action is added.

- AgentCoreRuntimeStack runtime ECR lifecycle selects only UNTAGGED images over
  the existing count10 threshold. Tagged runtime/worker manifests are excluded;
  digest references alone are not retention tags. Preserve repository scanning,
  names, deletion behavior, IAM and all other settings. Removing the last tag
  makes an image eligible for untagged cleanup; live policy/recovery is separate.

- Runtime and Harness Bedrock invocation grants use the same admitted model
  catalog, including reviewer models. Keep Harness guardrails and forbidden
  compute actions unchanged when adding model support.
- `_iam.py:cfn_getatt_arn` — Return cdk-nag's canonical ``<LogicalId.Arn>`` rendering for an L2.
- `_iam.py:acknowledge_iam5_resources` — Acknowledge exact residual IAM5 resource patterns with review evidence.
- `_kms.py:make_data_key` — Return a customer-managed CMK for a data stack — create by default, import if BYO.
- `_s3.py:make_access_log_bucket` — Create a retained, private sink for S3 server-access logs.
- `_tags.py:cost_tag_values` — Resolve the cost-allocation tag values from env (safe defaults; no org-specific value).
- `_tags.py:apply_cost_tags` — Stamp project/environment/owner cost-allocation tags on every taggable resource (WA-12).
- Owned files: `__init__.py`, `_iam.py`, `_kms.py`, `_s3.py`, `_tags.py`, `agentcore_runtime_stack.py`, `async_training_stack.py`, `backend_iam_stack.py`, `detections_stack.py`, `guardrail_stack.py`, `harness_stack.py`, `lake_stack.py`, `model_registry_stack.py`, `network_stack.py`, `research_scan_stack.py`, `runtime_stack.py`, `secrets_stack.py`.

- The SageMaker execution role owns paired broker CodeInterpreter permissions independently of the Runtime role. Grant only GetCodeInterpreter and Start/Invoke/Stop/GetCodeInterpreterSession on the selected Region AWS-owned built-in interpreter ARN, with RequestedRegion restriction; do not grant creation/deletion or custom interpreters. Source policy does not prove live IAM admission or runtime containment.

- AsyncTrainingStack and RuntimeStack preserve their existing Training terminal rule and add a separate Processing terminal rule to the same callback target. ProcessingJobName is limited to the selected hyphen deployment prefix plus dash; only Completed/Failed/Stopped from aws.sagemaker match. The target's generated Lambda permission is bound to that rule ARN; execution roles, callback environment and handler are unchanged. This source event route does not activate campaign mode or qualify native ingress: legacy mode denies Processing, and campaign mode still requires independently qualified private services/original handoff before any delivery. Event payload is never lifecycle or approval authority.
- [Added 2026-11-10] Both Training terminal rules carry the same `TrainingJobName` prefix filter as the Processing rule (`<hyphen deployment prefix>-`). Every product training job name starts that way (SFT and reload `-sft-`, generation `-gen-`, tabular `<prefix>-<operation>-<suffix>`). Why: without the filter every deployment's callback in the account received every other deployment's training events and answered `binding_missing` after several control-plane reads. A prefix filter cannot separate a deployment whose prefix is another's prefix plus a dash and more characters (`amlas` and `amlas-demo`).

- `guardrail_stack.py` explicitly enables PII evaluation on INPUT and OUTPUT for the unchanged seven entities; each direction and legacy Action uses the existing PiiDefaultAction parameter. Preserve its ANONYMIZE default/BLOCK option, HIGH content default, PROMPT_ATTACK input strength/outputNONE, messaging and current SSM/Output version attributes. Synthesized-template proof does not establish selected native protection or an admission grant. <!-- PII_DIRECTION_CONTRACT -->

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
