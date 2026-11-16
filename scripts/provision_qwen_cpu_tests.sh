#!/usr/bin/env bash
# Install the closed CPU closure, exact metadata and the pinned public evaluation file; source environment.sh afterwards.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  printf '%s\n' 'usage: provision_qwen_cpu_tests.sh NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
output=$1
mkdir -m 700 "$output"
output=$(cd "$output" && pwd -P)
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
uv pip install --target "$output/site" --require-hashes --no-deps --only-binary :all: -r "$project/scripts/qwen_cpu_test.requirements.lock"
PYTHONPATH="$project/src${PYTHONPATH:+:$PYTHONPATH}" uv run --frozen --no-sync python "$project/scripts/qwen_cpu_test_metadata.py" --destination "$output/metadata" > "$output/metadata-receipt.json"
# The one public evaluation file, fetched and verified while the network is still allowed.
case_file="$output/humaneval/test-00000-of-00001.parquet"
uv run --frozen --no-sync python "$project/scripts/qwen_cpu_test_case_file.py" --destination "$case_file"
{
  printf 'export PYTHONPATH=%q\n' "$output/site:$project/src${PYTHONPATH:+:$PYTHONPATH}"
  printf 'export AMLAS_QWEN_TEST_METADATA=%q\n' "$output/metadata"
  printf 'export AMLAS_TEST_HUMANEVAL_PARQUET=%q\n' "$case_file"
  printf '%s\n' 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1'
} > "$output/environment.sh"
