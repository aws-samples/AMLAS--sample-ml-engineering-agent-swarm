"""Strict neutral original DATA provider fixtures for normal warm SDK tests."""

from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

from botocore.session import Session
from botocore.validate import validate_parameters
from strands.models import Model
from test_managed_dataset_acquisition import NativeProcessing, NativeS3, adapter

from amlas.flows import dataset_acquisition_flow


class VirtualInitialCreateClock:
    """Explicit bulk-fake pacing clock; real deadlines/events stay untouched."""

    def __init__(self):
        self.now = 10.0
        self.waits = []
        self.reads = 0
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            self.reads += 1
            return self.now

    def sleep(self, seconds):
        assert type(seconds) in (int, float) and seconds == 1.25
        with self.lock:
            self.waits.append(seconds)
            self.now += seconds

    def assert_effects(self, effects):
        with self.lock:
            assert effects == 66
            assert len(self.waits) == effects - 1
            assert self.waits == [1.25] * (effects - 1)
            assert self.now == 10.0 + (effects - 1) * 1.25


def install_bulk_initial_create_clock(monkeypatch):
    """Opt in only for the two 65-original inert SDK volume fixtures."""
    from amlas.backends.adapters import _util

    clock = VirtualInitialCreateClock()
    monkeypatch.setattr(_util, "_PROCESSING_CREATE_STATES", {})
    monkeypatch.setattr(_util, "_processing_create_now", clock.monotonic)
    monkeypatch.setattr(_util, "_processing_create_sleep", clock.sleep)
    return clock


class MultipleNativeData(NativeProcessing):
    def __init__(self):
        super().__init__()
        self.requests = {}
        self.states = {}
        self.observations = []
        self.observation_states = []

    def create_processing_job(self, **kwargs):
        shape = Session().get_service_model("sagemaker").operation_model("CreateProcessingJob").input_shape
        validate_parameters(kwargs, shape)
        assert kwargs["ProcessingJobName"] not in self.requests
        assert kwargs["AppSpecification"]["ContainerEntrypoint"] == [
            "/usr/local/bin/amlas-processing-init",
            "acquisition",
        ]
        self.requests[kwargs["ProcessingJobName"]] = copy.deepcopy(kwargs)
        self.states[kwargs["ProcessingJobName"]] = "InProgress"
        self.calls.append(copy.deepcopy(kwargs))
        return {"ProcessingJobArn": self.arn(kwargs["ProcessingJobName"])}

    def describe_processing_job(self, **kwargs):
        shape = Session().get_service_model("sagemaker").operation_model("DescribeProcessingJob").input_shape
        validate_parameters(kwargs, shape)
        name = kwargs["ProcessingJobName"]
        request = self.requests[name]
        self.observations.append(name)
        self.observation_states.append((name, self.states[name]))
        return {
            "ProcessingJobArn": self.arn(name),
            "ProcessingJobName": name,
            "ProcessingJobStatus": self.states[name],
            "FailureReason": "",
            "AppSpecification": request["AppSpecification"],
            "ProcessingOutputConfig": request["ProcessingOutputConfig"],
            "RoleArn": request["RoleArn"],
            "Environment": request.get("Environment", {}),
        }


def configured_data(monkeypatch):
    native, s3 = MultipleNativeData(), NativeS3()
    value = adapter(sm=native, s3=s3)
    monkeypatch.setattr(
        dataset_acquisition_flow,
        "_resolve_backend",
        lambda: SimpleNamespace(adapter=value, adapter_name=value.backend_name),
    )
    return SimpleNamespace(native=native, s3=s3, adapter=value)


class SequentialProducerStream(Model):
    """Actual SDK stream chooses registered tools; never creates private records."""

    def __init__(self, *, concurrent=False):
        self.concurrent = concurrent
        self.calls = 0
        self.messages_seen = []

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "neutral-multi-original"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary installed SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        yield {"messageStart": {"role": "assistant"}}
        calls = []
        if self.calls == 1:
            calls.append(
                (
                    "original-model-use",
                    "acquire_model_weights",
                    {"catalog_id": "qwen3_5_9b", "catalog_version": "1"},
                )
            )
            if self.concurrent:
                calls.extend(self.data_calls())
        elif self.calls == 3 and not self.concurrent:
            assert messages[-1]["role"] == "user"
            event = json.loads(messages[-1]["content"][0]["text"])
            assert (
                event["event_type"] == "original_model_acquisition_completed" and event["data_only"] is True
            )
            calls.extend(self.data_calls())
        if calls:
            for index, (identifier, name, params) in enumerate(calls):
                yield {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {"toolUseId": identifier, "name": name}},
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {"toolUse": {"input": json.dumps(params)}},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": index}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "Original producers remain pending."},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    @staticmethod
    def data_calls():
        return [
            (
                "original-data-one",
                "acquire_huggingface_dataset",
                {"repo_id": "fixture/open-code", "revision": "b" * 40, "filename": "data/train.jsonl"},
            ),
            (
                "original-data-two",
                "acquire_huggingface_dataset",
                {"repo_id": "fixture/open-code", "revision": "b" * 40, "filename": "data/train-extra.jsonl"},
            ),
        ]
