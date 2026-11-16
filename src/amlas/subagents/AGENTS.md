# src/amlas/subagents

## Purpose

Declare bounded specialist agents and their reusable tools/prompts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep specialist tool/prompt declarations centralized and bounded; wrappers must reflect those declarations.
- Dataset acquisition belongs to the data-collection specialist, with public
  metadata discovery on research/data-collection specialists. Leads keep their
  orchestration boundary and do not acquire datasets directly.
- Collection owns opaque acquisition polling as well as gated submission;
  accepted jobs and verified artifacts remain distinct. Metadata inspection
  remains available to research specialists without acquisition authority.
- Collection also dispatches fixed managed inspection and polls its opaque refs.
  The specialist never parses raw acquired files in its agent process.
- `business_goal_advisor.py:build_business_goal_advisor` — Build the business_goal_advisor sub-agent (R1.a) — Stage-1 problem/KPI/go-no-go reasoning.
- `compute_placement.py:agent_tool_names` — Return the tool names bound to a Strands ``Agent`` (best-effort, version-tolerant).
- `data_collection_agent.py:build_data_collection_agent` — Build the data_collection_agent sub-agent (DE.a) — Stage-4 lake inventory via QueryBackend (read).
- `data_integrity.py:build_data_integrity_agent` — Build the Data Integrity sub-agent.
- `declarations.py:tool_name_of` — The name a ``@tool`` object registers itself under.
- `declarations.py:tool_contract_of` — ``(name, description, input_schema)`` for a ``@tool``, readable offline.
- Owned files: `__init__.py`, `algo_hypothesis.py`, `algorithm_specialist.py`, `business_goal_advisor.py`, `compute_placement.py`, `data_collection_agent.py`, `data_integrity.py`, `declarations.py`, `deployment_agent.py`, `devops_git_agent.py`, `dq_assessment.py`, `feature_hypothesis.py`, `fine_tune.py`, `framing.py`, `governance_agent.py`, `improvement_agent.py`, `imputation_strategist.py`, `labeling_strategist.py`, `llm_extrapolation.py`, `monitoring_agent.py`, `problem_framing_advisor.py`, `readiness_review_agent.py`, `registry.py`, `research_curator.py`, `research_discovery.py`, `research_scout.py`, `sagemaker_ops.py`, `solution_fit_advisor.py`, `solution_survey_agent.py`, `spec_writer.py`, `synthetic_data.py`, `tabular_fe.py`, `trainer_tuner.py`, `vision_fe.py`.

- Only the collection specialist receives preparation dispatch/poll tools; the
  Data Engineering Lead delegates and never computes. Preparation state is transported
  only for the same conversation acquisition store, with fresh caller policy.

- Collection and trainer declarations expose list_model_weight_catalog,
  acquire_model_weights and get_model_weight_acquisition. Resource/image/role
  authority is never a model schema field. Collection receives only a typed
  model store matching its acquisition session and a fresh caller-policy copy.

- Trainer/tuner and fine-tune specialists expose fixed submit_qwen_sft,
  get_qwen_sft and reload_qwen_sft alongside existing compatibility tools. Only
  opaque same-conversation refs and fixed recipes are public inputs; role/image/
  source/native handles/approval remain outside model schemas. Fit32 is a pilot,
  full1024/floors and separate reload are distinct; neither implies quality.

- Collection receives read-only `list_coding_preparation_profiles` to query existing source compatibility before acquisition. Its public optional source pins are DATA; catalog values, policies, payloads and native readiness stay outside its model schema. Discovery adds no preparation profile or dataset recommendation.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `fine_tune_a2a/AGENTS.md` | Expose the separate managed fine-tuning sub-runtime and its invocation boundary. |
