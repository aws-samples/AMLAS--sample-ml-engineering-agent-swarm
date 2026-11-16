"""Lossless runtime-environment pack for the managed runtime's 4,000-byte limit.

AgentCore refuses a runtime environment whose key and value lengths sum to more
than 4,000 bytes. When a selection is larger, the deploy control client folds the
closed stage-pin family and a frozen approved-action list into one key,
``AMLAS_RUNTIME_ENV_PACK``, with the sha256 of that text in
``AMLAS_RUNTIME_ENV_PACK_SHA256``. ``amlas/__init__.py`` checks and expands it before
any reader runs and removes both keys, so every reader and child process sees exactly
the keys and values the deployment selected. Admission digests stay literal values;
nothing is derived. Stdlib only and free of package imports: the deploy loads this
file by path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, MutableMapping

PACK_KEY = "AMLAS_RUNTIME_ENV_PACK"
PACK_SHA256_KEY = "AMLAS_RUNTIME_ENV_PACK_SHA256"
PACK_VERSION = 1
# AgentCore's maximum: the sum of key and value lengths. Counted here in UTF-8 bytes.
LIMIT_BYTES = 4000

STAGES = (
    "DATA_ACQUISITION",
    "DATA_INSPECTION",
    "DATA_PREPARATION",
    "CODING_PAIR",
    "CODING_GENERATION",
    "CODING_CASE_FREEZE",
    "QWEN_SFT",
    "MODEL_WEIGHT_ACQUISITION",
)
_COMMON = {"i": "IMAGE_URI", "c": "SOURCE_COMMIT", "d": "ADMISSION_DIGEST"}
_EXTRA = {
    "CODING_PAIR": {"l": "LAUNCHER_SHA256"},
    "DATA_PREPARATION": {
        "xu": "EXCLUSIONS_URI",
        "xv": "EXCLUSIONS_VERSION_ID",
        "xs": "EXCLUSIONS_SHA256",
        "xn": "EXCLUSIONS_BYTE_COUNT",
    },
}
FIELDS = {stage: {**_COMMON, **_EXTRA.get(stage, {})} for stage in STAGES}
_SLOTS = {
    f"AMLAS_{stage}_{suffix}": (stage, code)
    for stage, fields in FIELDS.items()
    for code, suffix in fields.items()
}

APPROVED_ACTIONS_KEY = "AMLAS_RUNTIME_APPROVED_ACTIONS"
# A profile names one exact list, frozen across releases. It is never the live
# closed action set, so adding an action to the product cannot widen it; a changed
# list gets a new profile name. Only the byte-identical canonical text is replaced.
APPROVED_ACTION_PROFILES = {
    "campaign_v1": (
        "acquire_huggingface_dataset",
        "acquire_model_weights",
        "coding_case_freeze_flow",
        "coding_generation_flow",
        "coding_pair_flow",
        "data_collection_flow",
        "data_preparation_flow",
        "freeze_coding_cases",
        "inspect_acquired_dataset",
        "prepare_coding_dataset",
        "qwen_sft_flow",
        "reload_qwen_sft",
        "resolve_coding_evaluation_spec",
        "run_processing_job",
        "submit_coding_generation",
        "submit_coding_pair",
        "submit_qwen_sft",
    )
}
_DIGEST = re.compile(r"@sha256:[a-f0-9]{64}")
_IMAGE = re.compile(r"(.+)(@sha256:[a-f0-9]{64})")


def environment_bytes(environment: Mapping[str, str]) -> int:
    """The size AgentCore checks: the sum of key and value lengths."""
    return sum(len(key.encode("utf-8")) + len(value.encode("utf-8")) for key, value in environment.items())


def is_packed_key(key: str) -> bool:
    """True for a key of the closed stage-pin family."""
    return key in _SLOTS


def payload_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_actions(names: tuple[str, ...]) -> str:
    return json.dumps(list(names), separators=(",", ":"))


def _clean(value: object) -> bool:
    return type(value) is str and not any(ord(char) < 32 or ord(char) == 127 for char in value)


def _string_map(environment: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(environment, Mapping) or not all(
        type(key) is str and type(value) is str for key, value in environment.items()
    ):
        raise ValueError("runtime environment must be a string map")
    return dict(environment)


def _repository(stages: dict[str, dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for fields in stages.values():
        match = _IMAGE.fullmatch(fields.get("i", ""))
        if match:
            counts[match[1]] = counts.get(match[1], 0) + 1
    if not counts:
        return ""
    repository = min(counts, key=lambda name: (-counts[name], name))
    return repository if counts[repository] > 1 else ""


def pack(environment: Mapping[str, str]) -> dict[str, str]:
    """Fold the closed family into one key; the result always unpacks to the input."""
    source = _string_map(environment)
    if PACK_KEY in source or PACK_SHA256_KEY in source:
        raise ValueError("runtime environment already carries a pack")
    result: dict[str, str] = {}
    stages: dict[str, dict[str, str]] = {}
    profile = ""
    canonical = {_canonical_actions(names): name for name, names in APPROVED_ACTION_PROFILES.items()}
    for key, value in source.items():
        slot = _SLOTS.get(key)
        if slot is not None:
            if not _clean(value):
                raise ValueError("stage pin value cannot be packed")
            stages.setdefault(slot[0], {})[slot[1]] = value
        elif key == APPROVED_ACTIONS_KEY and value in canonical:
            profile = canonical[value]
        else:
            result[key] = value
    if not stages and not profile:
        return source
    repository = _repository(stages)
    if repository:
        for fields in stages.values():
            match = _IMAGE.fullmatch(fields.get("i", ""))
            if match and match[1] == repository:
                fields["i"] = match[2]
    document: dict[str, object] = {"v": PACK_VERSION}
    if repository:
        document["r"] = repository
    if profile:
        document["a"] = profile
    if stages:
        document["s"] = stages
    text = json.dumps(document, separators=(",", ":"), sort_keys=True)
    result[PACK_KEY] = text
    result[PACK_SHA256_KEY] = payload_sha256(text)
    if unpack(result) != source:
        raise ValueError("runtime environment pack is not lossless")
    return result


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = dict(pairs)
    if len(document) != len(pairs):
        raise ValueError("pack repeats a name")
    return document


def _refuse_constant(_name: str) -> object:
    raise ValueError("pack carries a non-finite number")


def _decode(text: str) -> dict[str, str]:
    if type(text) is not str or len(text.encode("utf-8")) > LIMIT_BYTES:
        raise ValueError("pack text is not a bounded string")
    try:
        document = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_refuse_constant)
    except (TypeError, RecursionError, json.JSONDecodeError):
        raise ValueError("pack is not valid JSON") from None
    if type(document) is not dict or not set(document) <= {"v", "r", "a", "s"}:
        raise ValueError("pack has an unknown shape")
    if type(document.get("v")) is not int or document["v"] != PACK_VERSION:
        raise ValueError("pack version is not supported")
    if "a" not in document and "s" not in document:
        raise ValueError("pack carries no values")
    repository = document.get("r")
    if repository is not None and (not _clean(repository) or not repository):
        raise ValueError("pack repository is invalid")
    restored: dict[str, str] = {}
    if "a" in document:
        names = APPROVED_ACTION_PROFILES.get(document["a"]) if type(document["a"]) is str else None
        if names is None:
            raise ValueError("pack names an unknown approved-action profile")
        restored[APPROVED_ACTIONS_KEY] = _canonical_actions(names)
    stages = document.get("s", {})
    if type(stages) is not dict or ("s" in document and not stages):
        raise ValueError("pack stage table is invalid")
    used_repository = False
    for stage, fields in stages.items():
        allowed = FIELDS.get(stage)
        if allowed is None:
            raise ValueError("pack names an unknown stage")
        if type(fields) is not dict or not fields:
            raise ValueError("pack stage entry is invalid")
        for code, value in fields.items():
            if code not in allowed:
                raise ValueError("pack names a field the stage does not have")
            if not _clean(value):
                raise ValueError("pack value is not a clean string")
            if code == "i" and value.startswith("@"):
                if repository is None or not _DIGEST.fullmatch(value):
                    raise ValueError("pack image digest has no repository")
                value = repository + value
                used_repository = True
            restored[f"AMLAS_{stage}_{allowed[code]}"] = value
    if repository is not None and not used_repository:
        raise ValueError("pack repository is unused")
    return restored


def _checked(source: dict[str, str]) -> tuple[dict[str, str], str]:
    """Remove and verify the pack and its digest; return the restored keys and the digest."""
    text = source.pop(PACK_KEY)
    digest = source.pop(PACK_SHA256_KEY, None)
    if type(digest) is not str or digest != payload_sha256(text if type(text) is str else ""):
        raise ValueError("pack digest is missing or differs")
    restored = _decode(text)
    for key in restored:
        if key in source:
            raise ValueError(f"{key} is present both plain and packed")
    return restored, digest


def unpack(environment: Mapping[str, str]) -> dict[str, str]:
    """Return the plain environment the pack was made from."""
    source = _string_map(environment)
    if PACK_KEY not in source:
        if PACK_SHA256_KEY in source:
            raise ValueError("pack digest has no pack")
        return source
    restored, _digest = _checked(source)
    source.update(restored)
    return source


def expand_environ(environ: MutableMapping[str, str] | None = None) -> tuple[int, str] | None:
    """Expand a pack in place, all or nothing; both pack keys are removed.

    Returns the number of restored keys and the payload digest, or None when no
    pack is present. A second call finds no pack and changes nothing.
    """
    target = os.environ if environ is None else environ
    if PACK_KEY not in target:
        if PACK_SHA256_KEY in target:
            raise ValueError("pack digest has no pack")
        return None
    source = _string_map(dict(target))
    restored, digest = _checked(source)
    for key, value in restored.items():
        target[key] = value
    del target[PACK_KEY]
    del target[PACK_SHA256_KEY]
    return len(restored), digest


def main(argv: list[str] | None = None) -> int:
    """Read a JSON string map on stdin; print it packed, unpacked or its size."""
    args = sys.argv[1:] if argv is None else argv
    if args not in (["pack"], ["unpack"], ["bytes"]):
        print("usage: _runtime_env_pack.py pack|unpack|bytes < environment.json", file=sys.stderr)
        return 2
    try:
        environment = _string_map(json.load(sys.stdin))
        if args == ["bytes"]:
            print(environment_bytes(environment))
        else:
            selected = pack(environment) if args == ["pack"] else unpack(environment)
            print(json.dumps(selected, sort_keys=True))
    except (ValueError, TypeError) as error:
        print(f"{PACK_KEY} refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
