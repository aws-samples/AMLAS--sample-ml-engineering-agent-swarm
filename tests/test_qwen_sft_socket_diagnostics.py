"""Fixed socket diagnostics explain only an existing EINVAL refusal; they never admit."""

import errno
import json

import pytest

from amlas.sagemaker_entrypoints import qwen_sft_train as worker

LITERAL = "model metadata endpoint observation unverified"
STAGES = ("socket_create", "timeout_setup", "connect", "close")


def original_refusal(number=errno.EINVAL, literal=LITERAL, stage="provider_credentials"):
    def fail():
        try:
            raise OSError(number, "private-error-value")
        except OSError:
            raise RuntimeError(literal) from None

    with pytest.raises(worker.FixedPrecoreError) as found:
        worker._precore(stage, fail)
    error = found.value
    result = json.loads(worker._format_refusal(error))
    assert result["status"] == "fixed_qwen_operation_refused"
    assert result["stage"] == stage
    assert result["cause_type"] == "RuntimeError"
    assert "private-error-value" not in json.dumps(result)
    return error, result


class FixedSocket:
    def __init__(self, index, failures, calls):
        self.index = index
        self.failures = failures
        self.calls = calls

    def call(self, stage, value=None):
        self.calls.append((self.index, stage, value))
        error = self.failures.get((self.index, stage))
        if error is not None:
            raise error

    def settimeout(self, value):
        assert value == 0.5
        self.call("timeout_setup", value)

    def connect(self, address):
        assert address == (("169.254.169.254", "169.254.170.2")[self.index], 80)
        self.call("connect", address)

    def close(self):
        self.call("close")


def sockets(monkeypatch, failures=None):
    failures = failures or {}
    calls = []
    count = []

    def create(family, kind, protocol):
        assert (family, kind, protocol) == (
            worker.socket.AF_INET,
            worker.socket.SOCK_STREAM,
            worker.socket.IPPROTO_TCP,
        )
        index = len(count)
        count.append(index)
        assert index < 2
        calls.append((index, "socket_create", None))
        error = failures.get((index, "socket_create"))
        if error is not None:
            raise error
        return FixedSocket(index, failures, calls)

    monkeypatch.setattr(worker.socket, "socket", create)
    monkeypatch.setattr(
        worker.socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("diagnostic DNS forbidden")
    )
    monkeypatch.setattr(
        worker.socket, "create_connection", lambda *_a, **_k: pytest.fail("wrapper forbidden")
    )
    return calls


def test_fixed_numeric_success_records_reachable_but_original_refusal_remains(monkeypatch):
    calls = sockets(monkeypatch)
    error, result = original_refusal()
    assert result["refusal_code"] == "provider_observation_unknown"
    assert result["native_errno"] == errno.EINVAL
    records = result.get("socket_diagnostics")
    assert records is not None, "original EINVAL refusal must have bounded socket diagnostics"
    assert len(records) == 8
    assert [record["stage"] for record in records] == list(STAGES) * 2
    assert [record["address_index"] for record in records] == [0] * 4 + [1] * 4
    assert records[2]["outcome"] == records[6]["outcome"] == "reachable"
    assert all(record["errno_name"] == "none" for record in records)
    assert sum(stage == "close" for _, stage, _ in calls) == 2
    assert error.refusal_code == "provider_observation_unknown"
    assert set(records[0]) == {"address_index", "stage", "outcome", "errno_name"}
    encoded = json.dumps(result)
    assert "169.254" not in encoded
    assert "attested" not in encoded and "accepted" not in encoded


@pytest.mark.parametrize("stage", STAGES)
def test_each_einval_stage_is_closed_and_socket_always_closed(monkeypatch, stage):
    calls = sockets(monkeypatch, {(0, stage): OSError(errno.EINVAL, "private-stage-value")})
    _, result = original_refusal()
    records = result.get("socket_diagnostics")
    assert records is not None, "original EINVAL failure lost its substage"
    record = next(item for item in records if item["address_index"] == 0 and item["stage"] == stage)
    assert record["outcome"] == "failed"
    assert record["errno_name"] == "EINVAL"
    assert sum(index == 0 and point == "close" for index, point, _ in calls) == (
        0 if stage == "socket_create" else 1
    )
    assert "private-stage-value" not in json.dumps(result)
    assert sum(point == "socket_create" for _, point, _ in calls) == 2


@pytest.mark.parametrize("number", [errno.EMFILE, errno.ENOMEM, errno.EACCES, errno.EPERM, 1234])
def test_errno_categories_are_observations_not_acceptance(monkeypatch, number):
    sockets(monkeypatch, {(0, "connect"): OSError(number, "private-native-value")})
    _, result = original_refusal()
    records = result.get("socket_diagnostics")
    assert records is not None
    record = next(item for item in records if item["address_index"] == 0 and item["stage"] == "connect")
    assert record["errno_name"] == errno.errorcode.get(number, "unknown")
    assert record["outcome"] == "failed"
    assert "private-native-value" not in json.dumps(result)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("error", [ValueError("private-unexpected"), KeyboardInterrupt("private-signal")])
def test_unexpected_diagnostic_exception_never_masks_original(monkeypatch, stage, error):
    sockets(monkeypatch, {(0, stage): error})
    _, result = original_refusal()
    records = result.get("socket_diagnostics")
    assert records is not None
    record = next(item for item in records if item["address_index"] == 0 and item["stage"] == stage)
    assert record["errno_name"] == "unknown"
    assert record["outcome"] == "failed"
    assert "private-unexpected" not in json.dumps(result)
    assert "private-signal" not in json.dumps(result)


def test_timeout_is_bounded_observation_and_close_failure_retained(monkeypatch):
    calls = sockets(
        monkeypatch,
        {(0, "connect"): TimeoutError(), (0, "close"): OSError(errno.EINVAL, "private-close-value")},
    )
    _, result = original_refusal()
    records = result.get("socket_diagnostics")
    assert records is not None
    assert records[2] == {
        "address_index": 0,
        "stage": "connect",
        "outcome": "timed_out",
        "errno_name": "ETIMEDOUT",
    }
    assert records[3] == {"address_index": 0, "stage": "close", "outcome": "failed", "errno_name": "EINVAL"}
    assert sum(index == 0 and stage == "close" for index, stage, _ in calls) == 1


@pytest.mark.parametrize(
    "number,literal,stage",
    [
        (errno.EADDRNOTAVAIL, LITERAL, "provider_credentials"),
        (errno.EINVAL, "private-wrong-message", "provider_credentials"),
        (errno.EINVAL, LITERAL, "core_import"),
    ],
)
def test_diagnostic_trigger_is_exact_original_refusal_only(monkeypatch, number, literal, stage):
    calls = []

    def unexpected_socket(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unexpected diagnostic")

    monkeypatch.setattr(worker.socket, "socket", unexpected_socket)
    _, result = original_refusal(number, literal, stage)
    assert "socket_diagnostics" not in result
    assert calls == [], "nonmatching original refusal must not attempt any socket"


@pytest.mark.parametrize(
    "bad",
    [
        [{"address_index": 0, "stage": "private-value", "outcome": "failed", "errno_name": "EINVAL"}],
        [{"address_index": True, "stage": "connect", "outcome": "reachable", "errno_name": "none"}],
        [{"address_index": 0, "stage": "connect", "outcome": "failed", "errno_name": "private-value"}],
        [{"address_index": 0, "stage": "socket_create", "outcome": "reachable", "errno_name": "none"}],
        [
            {
                "address_index": 0,
                "stage": "connect",
                "outcome": "reachable",
                "errno_name": "none",
                "private": "content",
            }
        ],
        "private-value",
    ],
)
def test_formatter_discards_nonclosed_diagnostic_objects(bad):
    error = worker.FixedPrecoreError("provider_credentials", "RuntimeError", "provider_observation_unknown")
    error.observation_kind = "metadata_connect"
    error.native_errno = errno.EINVAL
    error.socket_diagnostics = bad
    assert "socket_diagnostics" not in json.loads(worker._format_refusal(error))


def test_formatter_preserves_only_closed_original_einval_diagnostics():
    error = worker.FixedPrecoreError("provider_credentials", "RuntimeError", "provider_observation_unknown")
    error.observation_kind = "metadata_connect"
    error.native_errno = errno.EINVAL
    records = [{"address_index": 0, "stage": "connect", "outcome": "failed", "errno_name": "EINVAL"}]
    error.socket_diagnostics = records
    assert json.loads(worker._format_refusal(error)).get("socket_diagnostics") == records


def test_one_argument_oserror_is_unknown_not_native_errno(monkeypatch):
    sockets(monkeypatch, {(0, "close"): OSError(errno.EINVAL)})
    _, result = original_refusal()
    records = result["socket_diagnostics"]
    assert records[3] == {"address_index": 0, "stage": "close", "outcome": "failed", "errno_name": "unknown"}
