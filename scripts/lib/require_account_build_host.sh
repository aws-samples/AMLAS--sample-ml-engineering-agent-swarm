#!/usr/bin/env bash
# Sourceable; no native calls until the function is invoked.
require_account_build_host() {
  local helper binding account region instance docker_host builder
  helper="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/account_image_build.py" || return 1
  binding="$(python3 "$helper" --account "$1" --region "$2" --shell-binding "${@:3}")" || return 1
  IFS=$'\t' read -r account region instance docker_host builder <<< "$binding"
  [ "$account" = "$1" ] && [ "$region" = "$2" ] && [ -n "$instance" ] || return 1
  if [ "${3:-}" = "--location-only" ]; then return 0; fi
  [ "$docker_host" = 'unix:///var/run/docker.sock' ] && [ "$builder" = default ] || return 1
  export AMLAS_VERIFIED_DOCKER_HOST="$docker_host" AMLAS_VERIFIED_IMAGE_BUILDER="$builder"
}
