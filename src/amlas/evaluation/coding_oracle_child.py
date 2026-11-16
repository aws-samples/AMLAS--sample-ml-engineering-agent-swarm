"""Fixed official oracle exec child: no SDK imports, inherited roles or candidate exec."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coding_oracle import decode, encode
from coding_oracle_catalog import verify_pinned_case


def _object(line):
    if not line or len(line) > 65536 or not line.endswith(b"\n"):
        raise ValueError("bounded child frame required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate child frame key")
            result[key] = value
        return result

    value = json.loads(
        line,
        object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite child frame")),
    )
    if type(value) is not dict:
        raise ValueError("child frame object required")
    return value


def main():
    if any(name.startswith(("AWS_", "AMLAS_", "BEDROCK", "TAVILY")) for name in os.environ):
        raise ValueError("oracle child cannot inherit provider configuration")
    packet = _object(sys.stdin.buffer.readline(65537))
    if set(packet) != {"prompt", "oracle"}:
        raise ValueError("fixed original oracle packet required")
    prompt, oracle = packet["prompt"], packet["oracle"]
    verify_pinned_case(prompt, oracle)
    calls = 0

    def proxy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 512:
            raise ValueError("fixed original oracle call bound exceeded")
        raw = (
            json.dumps(
                {"kind": "call", "value": encode((args, kwargs))}, separators=(",", ":"), allow_nan=False
            ).encode()
            + b"\n"
        )
        if len(raw) > 65536:
            raise ValueError("fixed oracle call frame exceeded")
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        response = _object(sys.stdin.buffer.readline(65537))
        if set(response) != {"value"}:
            raise ValueError("fixed oracle response differs")
        return decode(response["value"])

    result = {"kind": "verdict", "outcome": "incomplete", "oracle_completed": False, "exit_code": None}
    try:
        random.seed(12345)
        namespace = {}
        # Exact source-owned hashes were checked before this credential-free exec.
        # Prompt namespace preserves fixed helper bindings such as case33.
        exec(compile(prompt["prompt"], prompt["case_id"] + "/trusted-prompt", "exec"), namespace)
        namespace[prompt["entry_point"]] = proxy
        exec(compile(oracle["test"], prompt["case_id"] + "/trusted-tests", "exec"), namespace)
        namespace["check"](proxy)
        result.update(outcome="passed", oracle_completed=True, exit_code=0)
    except AssertionError:
        result.update(outcome="failed", oracle_completed=True, exit_code=1)
    except Exception:
        result.update(outcome="runtime_error", oracle_completed=False, exit_code=None)
    sys.stdout.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
