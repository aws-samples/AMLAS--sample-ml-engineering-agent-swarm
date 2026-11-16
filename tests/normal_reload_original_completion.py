"""Verification-only inert original completion; imports pinned239 Source helpers.
No workflow entrypoint, model decisions, cold tool IDs, stores seeded or live fallback.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from types import SimpleNamespace
from typing import ClassVar

ARMED = True


class GuardOnly:
    def __init__(self, client):
        self._client = client

    def apply_guardrail(self, **values):
        return self._client.apply_guardrail(**values)

    def close(self):
        return self._client.close()


class ClosedClients:
    """No saved live SDK fallback; only exact inert objects plus explicit guard."""

    def __init__(self, inert, guard_factory):
        self.inert = dict(inert)
        self.guard_factory = guard_factory

    def client(self, service, **kwargs):
        if kwargs.get("region_name", "us-west-2") != "us-west-2":
            raise RuntimeError("SDK region differs")
        if service == "bedrock-runtime":
            return GuardOnly(self.guard_factory())
        if service not in self.inert:
            raise RuntimeError("unexpected SDK service")
        return self.inert[service]

    def resource(self, *args, **kwargs):
        raise RuntimeError("resource API forbidden")

    def setup_default_session(self, *args, **kwargs):
        raise RuntimeError("default Session mutation forbidden")


def install_live_channels(
    monkeypatch,
    state,
    *,
    prebuilt_model,
    guard_factory,
    guardrail_id,
    guardrail_version,
    production_build_agent,
    public_checked,
    public_request,
):
    """Root supplies preconstructed live channels and original Source functions."""
    if not ARMED:
        raise RuntimeError("UNARMED controlled-ML verification")
    import boto3

    from amlas.tools.research import public_fetch
    from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

    config = prebuilt_model.get_config()
    if (
        config.get("model_id") != "us.anthropic.claude-sonnet-5-5"
        or config.get("guardrail_id") != guardrail_id
        or config.get("guardrail_version") != guardrail_version
        or not guardrail_id
        or not guardrail_version
    ):
        raise RuntimeError("selected live model/guard configuration differs")
    if (
        not callable(guard_factory)
        or not callable(production_build_agent)
        or not callable(public_checked)
        or not callable(public_request)
    ):
        raise RuntimeError("original live channel prerequisites missing")
    monkeypatch.setenv("AMLAS_GUARDRAIL_ID", guardrail_id)
    monkeypatch.setenv("AMLAS_GUARDRAIL_VERSION", guardrail_version)
    monkeypatch.setenv("AMLAS_GUARDRAIL_ENABLED", "true")
    monkeypatch.setattr(public_fetch, "_checked_addresses", public_checked)
    monkeypatch.setattr(public_fetch, "_request_public", public_request)

    def build(backend, **kwargs):
        backend._model = prebuilt_model
        agent = production_build_agent(backend, **kwargs)
        state.work.agents.append(agent)
        return agent

    monkeypatch.setattr(ToolBindingBackend, "build_agent", build)
    clients = ClosedClients(
        {
            "sagemaker": state.native,
            "s3": state.objects,
            "sts": state.ports,
            "iam": state.ports,
            "dynamodb": state.ports,
        },
        guard_factory,
    )

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def client(self, service, **kwargs):
            return clients.client(service, **kwargs)

        def resource(self, *args, **kwargs):
            return clients.resource(*args, **kwargs)

    monkeypatch.setattr(boto3, "client", clients.client)
    monkeypatch.setattr(boto3, "Session", Session)
    monkeypatch.setattr(boto3.session, "Session", Session)
    monkeypatch.setattr(boto3, "resource", clients.resource)
    monkeypatch.setattr(boto3, "setup_default_session", clients.setup_default_session)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    return clients


class OriginalCompletion:
    """Finish only observed original fake jobs; never select or submit a stage."""

    STORE_NAMES: ClassVar[dict[str, str]] = {
        "MODEL": "_model_acquisition_store",
        "DATA": "_acquisition_store",
        "CASE": "_case_freeze_store",
        "PREP": "_preparation_store",
        "SFT": "_sft_store",
        "GEN": "_generation_store",
        "PAIR": "_pair_store",
    }

    def __init__(self, state, monkeypatch):
        self.state = state
        self.monkeypatch = monkeypatch
        self.accepted = {}
        self.done = set()
        self.owners = {}
        self.lock = threading.RLock()

    @staticmethod
    def _canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _find(self, name):
        from amlas.backends.types import JobHandle
        from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

        matches = []
        for engine in self.state.work.engines:
            backend = engine.tool_backend
            if type(backend) is not ToolBindingBackend:
                raise RuntimeError("original backend type differs")
            for kind, attr in self.STORE_NAMES.items():
                store = getattr(backend, attr)
                if store is None:
                    continue
                for value in store._records.values():
                    handle = value if type(value) is JobHandle else getattr(value, "handle", None)
                    if type(handle) is JobHandle and handle.id == name:
                        matches.append((kind, handle, value))
        if len(matches) > 1:
            raise RuntimeError("original provider identity is ambiguous")
        return matches[0] if matches else None

    def _complete(self, provider, operation, name):
        if operation not in {"Processing", "Training"}:
            raise RuntimeError("unexpected provider operation")
        key = (id(provider), operation, name)
        with self.lock:
            if key not in self.accepted:
                matches = [item for item in self.accepted if item[1:] == (operation, name)]
                if (
                    provider is not self.state.native
                    or len(matches) != 1
                    or matches[0][0] != id(self.state.work.native)
                ):
                    raise RuntimeError("unaccepted provider original")
                key = matches[0]
                provider = self.state.work.native
            original = self._find(name)
            if original is None:
                if key in self.done:
                    raise RuntimeError("original provider no longer owned")
                return
            kind, handle, record = original
            request, arn = self.accepted[key]
            if operation == "Training":
                current = next((value for value in provider.calls if value["TrainingJobName"] == name), None)
            elif hasattr(provider, "processing"):
                current = provider.processing.get(name)
            elif type(provider.requests) is dict:
                current = provider.requests.get(name)
            else:
                current = next(
                    (value for value in provider.requests if value["ProcessingJobName"] == name), None
                )
            if current is None or self._canonical(current) != self._canonical(request):
                raise RuntimeError("accepted provider request changed")
            if handle.arn != arn:
                raise RuntimeError("original provider ARN differs")
            if kind == "MODEL" and provider is not self.state.work.native:
                raise RuntimeError("MODEL provider differs")
            if kind == "DATA" and provider is not self.state.data.native:
                raise RuntimeError("DATA provider differs")
            if kind not in {"MODEL", "DATA"} and provider is not self.state.native:
                raise RuntimeError("managed provider differs")
            if key in self.done:
                if self.owners[key] != (kind, id(handle)):
                    raise RuntimeError("original provider record changed")
                return
            self._publish(kind, handle, record, request)
            self.owners[key] = (kind, id(handle))
            self.done.add(key)

    def _publish(self, kind, handle, record, request):
        import normal_sft_cold_fixture as f

        from amlas.backends.ports import QwenSftBackend

        state = self.state
        if kind == "MODEL":
            f.complete_fixture(
                self.monkeypatch,
                state.directory / ("model-output-" + handle.id),
                SimpleNamespace(handle=handle, adapter=state.work.adapter, native=state.work.native),
            )
            for key, (version, raw) in state.work.s3.objects.items():
                state.objects.install(key, raw, version)
        elif kind == "DATA":
            f.publish_data(state, record)
        elif kind == "CASE":
            bundle = f.freeze_managed_case_bundle(
                f.official_case_data()[1], source=handle.extra["source"], header=handle.extra["header"]
            )
            for name, raw in bundle.items():
                state.objects.install(handle.extra["output_prefix"] + name, raw)
            state.native.states[handle.id] = "Completed"
        elif kind == "PREP":
            f.publish_preparation(state, handle)
        elif kind == "SFT":
            if request["TrainingJobName"] != handle.id or handle.extra.get("recipe_id") not in {
                "fit_v1",
                "full_v1",
            }:
                raise RuntimeError("original SFT request differs")
            if handle.extra.get("operation") == "reload":
                from test_runtime_sft_reload_originals import publish_reload

                publish_reload(handle, state.objects)
            else:
                # The existing constructor selects calls[0]; provide only this accepted request.
                view = SimpleNamespace(calls=[copy.deepcopy(request)], state="InProgress")
                f.sft_fixture.install_output(
                    state.adapters[QwenSftBackend],
                    view,
                    state.objects,
                    handle,
                    recipe=handle.extra["recipe_id"],
                )
            state.native.states[handle.id] = "Completed"
        elif kind == "GEN":
            ref = handle.extra["run_binding"]["prompts_ref"]
            uri = ref["uri"].split("/", 3)
            response = state.objects.get_object(
                Bucket=uri[2],
                Key=uri[3],
                VersionId=ref["version_id"],
                ExpectedBucketOwner=handle.extra["catalog"].account_id,
            )
            body = response["Body"]
            try:
                raw = body.read(ref["byte_count"] + 1)
            finally:
                body.close()
            if (
                response.get("VersionId") != ref["version_id"]
                or len(raw) != ref["byte_count"]
                or hashlib.sha256(raw).hexdigest() != ref["sha256"]
            ):
                raise RuntimeError("original GEN prompt artifact differs")
            prompts = [json.loads(line) for line in raw.splitlines()]
            if len(prompts) != 164:
                raise RuntimeError("original GEN prompt count differs")
            f.publish_generation(state.objects, handle, prompts)
            state.native.states[handle.id] = "Completed"
        elif kind == "PAIR":
            f.publish_pair(state, handle)
        else:
            raise RuntimeError("unsupported controlled producer")

    def attach(self):
        from amlas.backends.registry import REGISTRY

        for provider in (self.state.work.native, self.state.data.native, self.state.native):
            for operation, create_name, describe_name, id_key, arn_key in (
                (
                    "Processing",
                    "create_processing_job",
                    "describe_processing_job",
                    "ProcessingJobName",
                    "ProcessingJobArn",
                ),
                (
                    "Training",
                    "create_training_job",
                    "describe_training_job",
                    "TrainingJobName",
                    "TrainingJobArn",
                ),
            ):
                if not hasattr(provider, create_name):
                    continue
                create = getattr(provider, create_name)
                describe = getattr(provider, describe_name)

                def accepted(
                    *_args,
                    _provider=provider,
                    _operation=operation,
                    _create=create,
                    _id_key=id_key,
                    _arn_key=arn_key,
                    **values,
                ):
                    if _args:
                        raise TypeError("positional SDK arguments forbidden")
                    result = _create(**values)
                    if type(result) is not dict or type(result.get(_arn_key)) is not str:
                        raise RuntimeError("fake SDK did not accept original")
                    key = (id(_provider), _operation, values[_id_key])
                    with self.lock:
                        if key in self.accepted:
                            raise RuntimeError("fake original was recreated")
                        self.accepted[key] = (copy.deepcopy(values), result[_arn_key])
                    return result

                def observe(
                    *_args,
                    _provider=provider,
                    _operation=operation,
                    _describe=describe,
                    _id_key=id_key,
                    **values,
                ):
                    if _args:
                        raise TypeError("positional SDK arguments forbidden")
                    self._complete(_provider, _operation, values[_id_key])
                    return _describe(**values)

                self.monkeypatch.setattr(provider, create_name, accepted)
                self.monkeypatch.setattr(provider, describe_name, observe)

        def resolve(port, *args, **kwargs):
            if port not in self.state.adapters:
                raise RuntimeError("unexpected backend port")
            adapter = self.state.adapters[port]
            return SimpleNamespace(adapter=adapter, adapter_name=adapter.backend_name)

        self.monkeypatch.setattr(REGISTRY, "resolve", resolve)
        return self
