"""Operator-only model selector transport; configured values grant no admission."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys

SUFFIXES = ("image_uri", "source_commit", "admission_digest")
KEYS = tuple("AMLAS_MODEL_WEIGHT_ACQUISITION_" + suffix.upper() for suffix in SUFFIXES)


def _target(account: str, region: str, prefix: str) -> None:
    if (
        not re.fullmatch(r"[0-9]{12}", account)
        or account == "0" * 12
        or not re.fullmatch(r"[a-z0-9-]+", region)
        or not re.fullmatch(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", prefix)
    ):
        raise ValueError("model selected operator namespace refused")


def validate(values: dict[str, str], account: str, region: str) -> dict[str, str]:
    if not any(values.values()):
        return {}
    if set(values) != set(KEYS) or any(not values.get(key) for key in KEYS):
        raise ValueError("model selector tuple is incomplete")
    if any(
        type(value) is not str or len(value) > 4096 or any(ord(char) < 32 for char in value)
        for value in values.values()
    ):
        raise ValueError("model selector encoding refused")
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
        raise ValueError("model immutable image/source selector refused")
    return values


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite model parameter response")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite model parameter response")
    return number


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model parameter response key")
        result[key] = value
    return result


def resolve(account: str, region: str, prefix: str) -> dict[str, str]:
    _target(account, region, prefix)
    explicit = {key: os.environ.get(key, "").strip() for key in KEYS}
    if any(explicit.values()):
        return validate(explicit, account, region)
    selected = {}
    for index, key in enumerate(KEYS):
        suffix = SUFFIXES[index]
        name = prefix + "/model-weight-acquisition-" + suffix.replace("_", "-")
        result = subprocess.run(
            ["aws", "ssm", "get-parameter", "--name", name, "--region", region, "--output", "json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode:
            if re.match(
                r"^(?:aws: \[ERROR\]: )?An error occurred \(ParameterNotFound\) when calling the GetParameter operation:",
                result.stderr.lstrip(" \t\r\n\f\v"),
            ):
                selected[key] = ""
                continue
            raise ValueError("model selected parameter read refused")
        if len(result.stdout) > 16384:
            raise ValueError("model selected parameter response refused")
        try:
            response = json.loads(
                result.stdout,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
            parameter = response["Parameter"]
            if parameter["Name"] != name or type(parameter["Value"]) is not str:
                raise ValueError
            value = parameter["Value"].strip()
            if not value:
                raise ValueError("successful model parameter response is blank")
            selected[key] = value
        except (ValueError, KeyError, TypeError):
            raise ValueError("model selected parameter identity refused") from None
    return validate(selected, account, region)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--env-only", action="store_true")
    args = parser.parse_args()
    try:
        _target(args.account, args.region, args.prefix)
        values = (
            validate({key: os.environ.get(key, "").strip() for key in KEYS}, args.account, args.region)
            if args.env_only
            else resolve(args.account, args.region, args.prefix)
        )
        for key, value in values.items():
            print(key + "=" + value)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        print("Model acquisition release configuration refused before writes.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
