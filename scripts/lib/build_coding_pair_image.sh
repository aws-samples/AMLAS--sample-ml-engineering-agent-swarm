#!/usr/bin/env bash
# Build a reviewable AMD64 worker locally on the verified account EC2 daemon.
# Publishing is a separate reviewed action; this helper never pushes a tag.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACCT=${AMLAS_AWS_ACCOUNT_ID:?explicit target account required}
REGION=${AMLAS_AWS_REGION:?explicit target Region required}
SOURCE=${AMLAS_SOURCE_COMMIT:?exact source commit required}
IMAGE=${AMLAS_CODING_PAIR_BUILD_TAG:?explicit local review tag required}
[[ "$SOURCE" =~ ^[a-f0-9]{40}$ ]] || { echo 'Invalid exact source commit.' >&2; exit 1; }
[[ "$IMAGE" =~ ^[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || { echo 'Invalid local review image tag.' >&2; exit 1; }
source "$ROOT/scripts/lib/require_account_build_host.sh"
require_account_build_host "$ACCT" "$REGION" || exit 1
[ "$(uname -m)" = x86_64 ] || { echo 'Paired image requires the verified native x86 EC2 builder.' >&2; exit 1; }
[ "$(git -C "$ROOT" rev-parse HEAD)" = "$SOURCE" ] || { echo 'Source checkout differs from selected commit.' >&2; exit 1; }
[ -z "$(git -C "$ROOT" status --porcelain)" ] || { echo 'Source checkout must be clean before image construction.' >&2; exit 1; }
docker --host "$AMLAS_VERIFIED_DOCKER_HOST" buildx build --builder "$AMLAS_VERIFIED_IMAGE_BUILDER" \
    --platform linux/amd64 --provenance=false --load \
    --build-arg "AMLAS_SOURCE_COMMIT=$SOURCE" \
    --tag "$IMAGE" --file "$ROOT/Dockerfile.coding-pair" "$ROOT"
docker --host "$AMLAS_VERIFIED_DOCKER_HOST" image inspect "$IMAGE" \
    --format '{{json .Architecture}} {{json .Config.User}} {{json .Config.Entrypoint}} {{json .Config.Labels}}'
printf 'PAIRED_IMAGE_BUILD_DONE source=%s tag=%s\n' "$SOURCE" "$IMAGE"
