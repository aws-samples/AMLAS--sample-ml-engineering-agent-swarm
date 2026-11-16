"""Operator-only preparation selector snapshot. Never imported by TUI startup.

A deployment selects none, the resource triple (image, source commit, admission
digest) or all seven values. With the triple, preparation runs only on a frozen
case of the same conversation, which supplies its governed exclusions
(``prepare_coding_dataset(case_request_ref=...)``); without a case the runtime
refuses with the four exclusions selectors named as missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from urllib.parse import urlsplit

SUFFIXES = (
    "image_uri",
    "source_commit",
    "admission_digest",
    "exclusions_uri",
    "exclusions_version_id",
    "exclusions_sha256",
    "exclusions_byte_count",
)
KEYS = tuple("AMLAS_DATA_PREPARATION_" + suffix.upper() for suffix in SUFFIXES)
TRIPLE = KEYS[:3]
EXCLUSIONS = KEYS[3:]


def validate(values: dict[str, str], account: str, region: str) -> dict[str, str]:
    if not any(values.values()):
        return {}
    if any(not values.get(key) for key in TRIPLE):
        raise ValueError("preparation selector tuple is incomplete")
    excluded = [bool(values.get(key)) for key in EXCLUSIONS]
    if any(excluded) and not all(excluded):
        raise ValueError("preparation exclusions selector tuple is incomplete")
    if any(
        type(value) is not str or len(value) > 4096 or any(ord(char) < 32 for char in value)
        for value in values.values()
    ):
        raise ValueError("preparation selector encoding refused")
    image = re.fullmatch(
        r"([0-9]{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com/[a-z0-9][a-z0-9._/-]*@sha256:([a-f0-9]{64})",
        values[KEYS[0]],
    )
    if (
        not image
        or image[1] != account
        or image[2] != region
        or not re.fullmatch(r"[a-f0-9]{40}", values[KEYS[1]])
        or values[KEYS[2]] != image[3]
    ):
        raise ValueError("preparation immutable image/source admission refused")
    if not any(excluded):
        return {key: values[key] for key in TRIPLE}
    uri = urlsplit(values[KEYS[3]])
    if (
        uri.scheme != "s3"
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", uri.netloc)
        or not uri.path.startswith("/")
        or uri.path == "/"
        or uri.query
        or uri.fragment
        or "\\" in values[KEYS[3]]
        or any(part in {".", "..", ""} for part in uri.path[1:].split("/"))
    ):
        raise ValueError("preparation exclusions URI refused")
    if (
        values[KEYS[4]] == "null"
        or len(values[KEYS[4]]) > 1024
        or not re.fullmatch(r"[a-f0-9]{64}", values[KEYS[5]])
    ):
        raise ValueError("preparation exclusions immutable identity refused")
    if not values[KEYS[6]].isdigit() or not 1 <= int(values[KEYS[6]]) <= 2 * 1024 * 1024:
        raise ValueError("preparation exclusions byte count refused")
    return {key: values[key] for key in KEYS}


def _parameter(name: str, region: str) -> str | None:
    """One exact SSM read; absent is None, and any other failure refuses."""
    result = subprocess.run(
        ["aws", "ssm", "get-parameter", "--name", name, "--region", region, "--output", "json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        if "ParameterNotFound" in result.stderr:
            return None
        raise ValueError("preparation selected parameter read refused")
    if len(result.stdout) > 16384:
        raise ValueError("preparation selected parameter response refused")
    try:
        response = json.loads(result.stdout)
        parameter = response["Parameter"]
        if parameter["Name"] != name or type(parameter["Value"]) is not str:
            raise ValueError
        return parameter["Value"].strip()
    except (ValueError, KeyError, TypeError):
        raise ValueError("preparation selected parameter identity refused") from None


def _parameter_name(prefix: str, key: str) -> str:
    return prefix + "/data-preparation-" + SUFFIXES[KEYS.index(key)].replace("_", "-")


def resolve(account: str, region: str, prefix: str) -> dict[str, str]:
    if (
        not re.fullmatch(r"[0-9]{12}", account)
        or not re.fullmatch(r"[a-z0-9-]+", region)
        or not re.fullmatch(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", prefix)
    ):
        raise ValueError("preparation selected operator namespace refused")
    explicit = {key: os.environ.get(key, "").strip() for key in KEYS}
    # Never combine a new ENV image with an old SSM admission/exclusions tuple.
    if any(explicit.values()):
        selected = validate(explicit, account, region)
        # A triple-only selection publishes three parameters. Exclusions left in SSM
        # from an earlier selection would sit beside it, and a later SSM-only read
        # would combine values never selected together; refuse before any write.
        if KEYS[3] not in selected and any(
            _parameter(_parameter_name(prefix, key), region) is not None for key in EXCLUSIONS
        ):
            raise ValueError("preparation exclusions selectors remain beside a triple-only selection")
        return selected
    selected = {}
    for key in KEYS:
        value = _parameter(_parameter_name(prefix, key), region)
        selected[key] = "" if value is None else value
    return validate(selected, account, region)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--env-only", action="store_true")
    args = parser.parse_args()
    try:
        values = (
            validate({key: os.environ.get(key, "").strip() for key in KEYS}, args.account, args.region)
            if args.env_only
            else resolve(args.account, args.region, args.prefix)
        )
        for key, value in values.items():
            print(key + "=" + value)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        print("Preparation release configuration refused before writes.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
