"""Pre-core diagnostic attribution never relaxes inventory/provider refusal."""

import json
from types import SimpleNamespace

import pytest

from amlas.sagemaker_entrypoints import qwen_sft_train as worker


@pytest.mark.parametrize(
    "point,expected", [("inventory", "provider_inventory"), ("credentials", "provider_credentials")]
)
def test_precore_provider_failure_has_only_fixed_stage(monkeypatch, point, expected):
    module = SimpleNamespace()

    def read(_fd):
        if point == "inventory":
            raise RuntimeError("private-provider-path-content")
        return {"inventory": {}, "sha256": "a" * 64}

    def observe(_sealed):
        raise RuntimeError("private-provider-path-content")

    module._read_provider_inventory = read
    module._credential_observation_v2 = observe
    monkeypatch.setattr(worker, "_provider_module", lambda: module)
    monkeypatch.setattr(worker.os, "fstat", lambda _fd: SimpleNamespace(st_dev=1, st_ino=2, st_size=3))
    for key in worker.FORBIDDEN:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(Exception) as found:
        worker._provider_observation("guard")
    result = json.loads(worker._format_refusal(found.value))
    assert result.get("stage") == expected
    assert result.get("cause_type") == "RuntimeError"
    assert "private-provider" not in json.dumps(result)


def test_known_metadata_refusal_is_closed_code_not_exception_text():
    def fail():
        raise RuntimeError("model metadata endpoint isolation refused")

    with pytest.raises(Exception) as found:
        worker._precore("provider_credentials", fail)
    result = json.loads(worker._format_refusal(found.value))
    assert result.get("refusal_code") == "metadata_reachable"
    assert "model metadata endpoint" not in json.dumps(result)


def test_unknown_refusal_never_echoes_message():
    def fail():
        raise RuntimeError("private-provider-path-content")

    with pytest.raises(Exception) as found:
        worker._precore("provider_credentials", fail)
    result = json.loads(worker._format_refusal(found.value))
    assert result.get("refusal_code") == "unknown"
    assert "private-provider" not in json.dumps(result)


def test_unknown_metadata_errno_is_numeric_only_and_refusal_preserved():
    import errno

    def fail():
        try:
            raise OSError(errno.EADDRNOTAVAIL, "private-native-network-value")
        except OSError:
            raise RuntimeError("model metadata endpoint observation unverified") from None

    with pytest.raises(Exception) as found:
        worker._precore("provider_credentials", fail)
    result = json.loads(worker._format_refusal(found.value))
    assert result.get("observation_kind") == "metadata_connect"
    assert result.get("native_errno") == errno.EADDRNOTAVAIL
    assert result["status"] == "fixed_qwen_operation_refused"
    assert "private-native" not in json.dumps(result)
