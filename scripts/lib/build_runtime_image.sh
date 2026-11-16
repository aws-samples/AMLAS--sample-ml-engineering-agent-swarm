#!/usr/bin/env bash
# Build and push the AgentCore runtime image (linux/arm64) to the ECR
# repository created by AgentCoreRuntimeStack. Called by scripts/setup.sh.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Honor an explicitly selected profile; otherwise use the standard AWS
# credential chain. Account and region are always resolved at runtime.
if [ -n "${AWS_PROFILE:-}" ]; then
  export AWS_PROFILE
fi
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_common.sh"
NON_INTERACTIVE=1
configure_deployment_namespace
resolve_aws_region
REGION="$AWS_REGION"
# Never trust an environment account as caller proof. The shared guard rechecks STS.
ACCT="${AMLAS_AWS_ACCOUNT_ID:-${AWS_ACCOUNT_ID:-}}"
if [ -z "$ACCT" ]; then
  ACCT="$(aws sts get-caller-identity --region "$REGION" --query Account --output text)" || exit 1
fi
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/require_account_build_host.sh"
require_account_build_host "$ACCT" "$REGION" || exit 1
# Keep the short-lived ECR authorization cache in a run-scoped workspace directory.
VERIFIED_PLUGIN_CONFIG="${DOCKER_CONFIG:-${HOME:-}/.docker}"
CONFIG_ROOT="${TMPDIR:-$PWD/output/deploy/tmp}"
mkdir -p "$CONFIG_ROOT" || exit 1
AUTH_CONFIG="$(mktemp -d "$CONFIG_ROOT/amlas-image-auth.XXXXXX")" || exit 1
[ -n "$AUTH_CONFIG" ] && [ -d "$AUTH_CONFIG" ] && [ ! -L "$AUTH_CONFIG" ] || exit 1
chmod 700 "$AUTH_CONFIG" || exit 1
[ -r "$AUTH_CONFIG" ] && [ -w "$AUTH_CONFIG" ] && [ -x "$AUTH_CONFIG" ] || exit 1
export DOCKER_CONFIG="$AUTH_CONFIG"
# Host preparation may install buildx only in the previous workspace config.
# Retain just that already-guarded executable; do not copy shared auth/config JSON.
if [ -f "$VERIFIED_PLUGIN_CONFIG/cli-plugins/docker-buildx" ]; then
  mkdir -p "$DOCKER_CONFIG/cli-plugins" || exit 1
  cp "$VERIFIED_PLUGIN_CONFIG/cli-plugins/docker-buildx" "$DOCKER_CONFIG/cli-plugins/docker-buildx" || exit 1
  chmod 700 "$DOCKER_CONFIG/cli-plugins/docker-buildx" || exit 1
fi
REPO="$AMLAS_DEPLOYMENT_PREFIX-runtime"
REGISTRY="${ACCT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO}"
# The image tag is parameterized so a rebuild can publish a new version without
# editing this helper. The deploy helper reads the same IMAGE_TAG value.
TAG="${IMAGE_TAG:-v1}"

trap 'docker --config "$DOCKER_CONFIG" --host "$AMLAS_VERIFIED_DOCKER_HOST" logout "$REGISTRY" >/dev/null 2>&1 || true' EXIT
START=$(date +%s)

echo "=== ECR LOGIN ==="
aws ecr get-login-password --region "${REGION}" \
  | docker --config "$DOCKER_CONFIG" --host "$AMLAS_VERIFIED_DOCKER_HOST" login --username AWS --password-stdin "${REGISTRY}" 2>&1 || exit $?

echo "=== BUILDX BUILD + PUSH (linux/arm64) ==="
# Single-arch arm64 (AgentCore requirement). --provenance=false keeps a plain
# image manifest (not an OCI index) so describe-images/inspect report arm64.
docker --config "$DOCKER_CONFIG" --host "$AMLAS_VERIFIED_DOCKER_HOST" buildx build --builder default \
  --platform linux/arm64 \
  --provenance=false \
  -t "${IMAGE_URI}:${TAG}" \
  -t "${IMAGE_URI}:latest" \
  --push \
  -f Dockerfile . 2>&1

BUILD_RC=$?
END=$(date +%s)
echo "=== BUILD_RC=${BUILD_RC} DURATION_SEC=$((END-START)) ==="
if [ "$BUILD_RC" -ne 0 ]; then
  echo "ERROR: runtime image build failed (exit $BUILD_RC)." >&2
  exit "$BUILD_RC"
fi
docker --config "$DOCKER_CONFIG" --host "$AMLAS_VERIFIED_DOCKER_HOST" logout "$REGISTRY" >/dev/null 2>&1 || exit 1
echo "IMAGE_URI=${IMAGE_URI}:${TAG}"
echo "RUNTIME_IMAGE_PUSH_DONE"
