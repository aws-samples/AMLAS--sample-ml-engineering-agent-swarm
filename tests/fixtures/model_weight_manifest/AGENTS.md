# tests/fixtures/model_weight_manifest

## Purpose

Own one bounded public safetensors index metadata capture for pure validation.

## Ownership

Inherits ../AGENTS.md; this directory contains no tensor weights, dataset or code.

## Local Contracts

- qwen3_5_9b_index.json is the exact 79,657-byte official Qwen/Qwen3.5-9B index
  at revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. The model card declares
  Apache-2.0; this is provenance, not a broader artifact copyright conclusion.
- Source: https://huggingface.co/Qwen/Qwen3.5-9B/raw/c202236235762e1c871ad0ccb60c8ee5ba337b9a/model.safetensors.index.json
- SHA256: 26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5.
- The capture describes 775 tensor names and four shard files; it contains no tensors.
  Metadata integrity cannot establish acquisition, safe loading, trust or readiness.

## Work Guidance

Keep the original bytes; version a new capture instead of regenerating this oracle.

## Verification

Run tests/test_model_weight_manifest.py and recursive DOX lint.

## Child DOX Index

None.
