"""Lambda handler — EventBridge → AgentCore Runtime resume.

Port of `sample-mlops-agent/lambda/callback/handler.py` (verbatim pattern;
paths + names adapted for Amlas).

Trigger: EventBridge rule on source=aws.sagemaker, detail-type="SageMaker Training
Job State Change". Handler pulls the DynamoDB row for the job, extracts the
runtime_session_id, and calls `bedrock-agentcore:InvokeAgentRuntime` so our agent
resumes in the same session.

Deploy as a Lambda with env vars:
    AGENTCORE_RUNTIME_ARN  = arn:aws:bedrock-agentcore:...:runtime/<id>
    JOBS_TABLE             = amlas-training-jobs
    AWS_REGION             = <runtime AWS region provided by Lambda>
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import boto3

# ── Pure builders (no AWS) ────────────────────────────────────────────────────
# These construct the EXACT kwargs / payload / envelopes the handler passes to
# boto3 / returns. They are pure functions (string/dict construction only — no
# network, no boto3) so the byte-identical-contract property callers can exercise
# them DIRECTLY without faking any external service (Phase 8 de-mock).


def build_resume_payload(job_name: str, status: str) -> bytes:
    """The utf-8 JSON payload sent to ``invoke_agent_runtime`` (pure)."""
    return json.dumps({"event": "training_complete", "job_name": job_name, "status": status}).encode("utf-8")


def build_invoke_kwargs(runtime_arn: str, session_id: str, job_name: str, status: str) -> dict[str, Any]:
    """The exact kwargs for ``bedrock-agentcore.invoke_agent_runtime`` (pure)."""
    return {
        "agentRuntimeArn": runtime_arn,
        "runtimeSessionId": session_id,
        "qualifier": "DEFAULT",
        "payload": build_resume_payload(job_name, status),
    }


def build_update_item_kwargs(jobs_table: str, job_name: str, status: str) -> dict[str, Any]:
    """The exact kwargs for ``dynamodb.update_item`` (pure)."""
    return {
        "TableName": jobs_table,
        "Key": {"job_name": {"S": job_name}},
        "UpdateExpression": "SET #s = :s",
        "ExpressionAttributeNames": {"#s": "status"},
        "ExpressionAttributeValues": {":s": {"S": status}},
    }


def build_success_envelope(session_id: str, job_name: str) -> dict[str, Any]:
    """The 200-path response envelope (pure)."""
    return {
        "statusCode": 200,
        "body": f"resumed session={session_id} for job={job_name}",
    }


@contextmanager
def _normal_callback_bootstrap():
    """Fresh server-only provider; its clients are owned independently of injected ports."""
    config = _campaign_config()
    role_arn = os.environ.get("AMLAS_CALLBACK_ROLE_ARN", "").strip()
    role_id = os.environ.get("AMLAS_CALLBACK_ROLE_ID", "").strip()
    _NativeCallbackRouteQualifier.validate_role(config, role_arn, role_id)
    clients, qualifier = [], None
    try:
        try:
            from botocore.config import Config

            options = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10)
            sts = boto3.client("sts", region_name=config.region, config=options)
            clients.append(sts)
            control = boto3.client("bedrock-agentcore-control", region_name=config.region, config=options)
            if control is not sts:
                clients.append(control)
            qualifier = _NativeCallbackRouteQualifier(
                config,
                expected_role_arn=role_arn,
                expected_role_id=role_id,
                sts_client=sts,
                control_client=control,
            )
        except Exception:
            raise CallbackQualificationError("native callback provider unavailable") from None
        try:
            with campaign_callback_scope(qualifier):
                yield
        finally:
            qualifier.close()
    finally:
        for client in reversed(clients):
            try:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            except Exception:
                # Cleanup attempts never mask the original setup/dispatch result.
                # Do not log provider errors or infer successful socket cleanup.
                pass


def _mode1_dispatch(event):
    if type(event) is dict and "Records" in event:
        result = _normal_stream_handler(event)
        if (
            type(result) is not dict
            or type(result.get("statusCode")) is not int
            or result.get("statusCode") != 200
            or result.get("delivery_status") not in ("delivered", "ignored")
        ):
            raise RuntimeError("normal readiness delivery unresolved")
        return result
    return _campaign_handler(event)


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point. Non-blocking — drains response stream then returns."""
    token = _LAMBDA_CONTEXT.set(context)
    try:
        return _handle(event)
    finally:
        _LAMBDA_CONTEXT.reset(token)


def _handle(event: dict[str, Any]) -> dict[str, Any]:
    mode = os.environ.get("AMLAS_CAMPAIGN_CALLBACK_MODE", "").strip()
    if mode:
        if mode != "1":
            return _campaign_result(403, "configuration_refused")
        if type(_CAMPAIGN_SERVICES.get()) is _CallbackServices:
            return _mode1_dispatch(event)
        try:
            config = _campaign_config()
            if type(event) is dict and "Records" in event:
                _bounded_stream_event(event)
                records = event.get("Records")
                if type(records) is not list or not 1 <= len(records) <= 16:
                    raise ValueError("bounded readiness batch required")
                candidates = [_normal_readiness_candidate(record) for record in records]
                if not any(candidate is not None for candidate in candidates):
                    return _normal_result(200, "ignored")
            else:
                _event_identity(event, config)
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
            if type(event) is dict and "Records" in event:
                raise RuntimeError("normal readiness delivery unresolved") from None
            return _campaign_result(400, "input_refused")
        try:
            with _normal_callback_bootstrap():
                return _mode1_dispatch(event)
        except CallbackQualificationError:
            if type(event) is dict and "Records" in event:
                raise RuntimeError("normal readiness delivery unresolved") from None
            return _campaign_result(403, "qualification_missing")
    detail = event.get("detail", {}) or {}
    job_name = detail.get("TrainingJobName")
    status = detail.get("TrainingJobStatus")
    if not job_name or not status:
        return {"statusCode": 400, "body": "event missing TrainingJobName or TrainingJobStatus"}

    # Lambda provides AWS_REGION. Keep the callback stdlib + provided boto3 only:
    # importing amlas.config would transitively require YAML/other optional deps.
    region = os.environ.get("AWS_REGION", "").strip()
    if not region:
        raise ValueError("AWS_REGION is required for the training callback; Lambda normally provides it.")
    jobs_table = os.environ["JOBS_TABLE"]
    runtime_arn = os.environ["AGENTCORE_RUNTIME_ARN"]

    ddb = boto3.client("dynamodb", region_name=region)
    row = ddb.get_item(TableName=jobs_table, Key={"job_name": {"S": job_name}}).get("Item")
    if not row:
        return {"statusCode": 404, "body": f"no DynamoDB row for {job_name}"}

    session_id = row.get("runtime_session_id", {}).get("S")
    if not session_id:
        return {"statusCode": 500, "body": f"row missing runtime_session_id for {job_name}"}

    # Update status row first — belt-and-braces for observability
    ddb.update_item(**build_update_item_kwargs(jobs_table, job_name, status))

    # Invoke the runtime with the preserved session ID
    client = boto3.client("bedrock-agentcore", region_name=region)
    response = client.invoke_agent_runtime(**build_invoke_kwargs(runtime_arn, session_id, job_name, status))

    # Drain the response body so the agent's training_complete handling is fully
    # observed before we return (LAMB-01). The SDK returns the body under the
    # "response" key as a StreamingBody (has .read()); align with the proven
    # dispatch.parse_invoke_response handling rather than iterating chunks. Fall
    # back to iterating chunk lines if it isn't a StreamingBody.
    body = response.get("response")
    if body is not None:
        reader = getattr(body, "read", None)
        if callable(reader):
            reader()  # fully consume the StreamingBody
        else:
            # Intentional full drain (LAMB-01): the invoke response is a finite HTTP
            # stream and the Lambda itself is bounded at 2 minutes (CDK timeout).
            for _ in body:  # nosemgrep: pass-body-range
                pass
    return {"statusCode": 200, "body": f"resumed session={session_id} for job={job_name}"}


# Opt-in source-only campaign route; native qualification implementation is not
# supplied by configuration, SDK clients, Lambda context or these Python types.
_CAMPAIGN_SERVICES = ContextVar("amlas_private_callback_services", default=None)
# The Lambda context of the current invocation (None when the handler is called directly).
_LAMBDA_CONTEXT = ContextVar("amlas_callback_lambda_context", default=None)
# SageMaker sends two terminal events for one Completed or Failed job; their deliveries start 6 to 878 ms
# apart and reach InvokeAgentRuntime within about a second of each other. Two near-simultaneous first
# calls on a session without a microVM have created two replacement microVMs, each restoring the run.
# The delivery that did not itself create the terminal hint therefore waits before it resumes, within
# the invocation's remaining time minus a reserve for the invoke, the drain and the requalification.
_TWIN_RESUME_DELAY_SECONDS = 40.0
_TWIN_RESUME_RESERVE_SECONDS = 50.0
_twin_pause = time.sleep


def _twin_delay():
    """Seconds the second delivery waits: bounded by the remaining Lambda time, 0 without a context."""
    remaining = getattr(_LAMBDA_CONTEXT.get(), "get_remaining_time_in_millis", None)
    if not callable(remaining):
        return 0.0
    try:
        budget = remaining() / 1000.0 - _TWIN_RESUME_RESERVE_SECONDS
    except Exception:
        return 0.0
    if type(budget) is not float or not math.isfinite(budget):
        return 0.0
    return max(0.0, min(_TWIN_RESUME_DELAY_SECONDS, budget))


_HINT_FIELDS = (
    "schema_version",
    "record_kind",
    "account_id",
    "region",
    "deployment_id",
    "kind",
    "job_name",
    "job_arn",
    "campaign_ref",
    "operation_digest",
    "transport_session_id",
)
_WAKE_FIELDS = (
    "schema_version",
    "event_id",
    "campaign_ref",
    "account_id",
    "region",
    "deployment_id",
    "kind",
    "operation_digest",
    "job_arn",
)


class CallbackQualificationError(ValueError):
    """Unknown/missing current independently approved callback route or role."""


def _exact_text(value, maximum=256):
    if type(value) is not str or not 0 < len(value.encode()) <= maximum or any(ord(c) < 32 for c in value):
        raise ValueError("bounded exact callback data required")
    return value


def _sha(value):
    if type(value) is not str or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("exact callback digest required")
    return value


@dataclass(frozen=True)
class CallbackRouteConfig:
    account_id: str
    region: str
    deployment_id: str
    jobs_table: str
    runtime_arn: str
    qualifier: str
    runtime_version: str
    route_sha256: str


@dataclass(frozen=True)
class CallbackRouteObservation:
    account_id: str
    region: str
    deployment_id: str
    runtime_arn: str
    qualifier: str
    runtime_version: str
    route_sha256: str
    observed_at: float
    expires_at: float
    process_id: int
    issuer: object


class _NativeCallbackRouteQualifier:
    """Lexical native metadata provider; no cached or serializable qualification."""

    @staticmethod
    def validate_role(config, role_arn, role_id):
        if (
            type(role_arn) is not str
            or re.fullmatch(
                r"arn:aws:iam::" + config.account_id + r":role/[A-Za-z0-9_+=,.@/-]{1,512}",
                role_arn,
            )
            is None
            or type(role_id) is not str
            or re.fullmatch(r"AROA[A-Z0-9]{17}", role_id) is None
        ):
            raise CallbackQualificationError("private callback role binding unavailable")

    def __init__(
        self,
        config,
        *,
        expected_role_arn,
        expected_role_id,
        sts_client,
        control_client,
        clock=time.monotonic,
    ):
        if type(config) is not CallbackRouteConfig or config != _campaign_config() or not callable(clock):
            raise CallbackQualificationError("current callback configuration required")
        self.validate_role(config, expected_role_arn, expected_role_id)
        self.config, self.role_arn, self.role_id = config, expected_role_arn, expected_role_id
        self.sts, self.control, self.clock = sts_client, control_client, clock
        self.process_id, self.thread_id = os.getpid(), threading.get_ident()
        self.active, self.highwater = True, None
        self.role_selectors = self._selected_role()
        if any(self.role_selectors) and self.role_selectors != (self.role_arn, self.role_id):
            raise CallbackQualificationError("current callback role selectors differ")
        self._now()
        self._transport(self.sts)
        self._transport(self.control)

    @staticmethod
    def _selected_role():
        return tuple(
            os.environ.get(key, "").strip() for key in ("AMLAS_CALLBACK_ROLE_ARN", "AMLAS_CALLBACK_ROLE_ID")
        )

    def close(self):
        self.active = False

    def _now(self):
        if not self.active or os.getpid() != self.process_id or threading.get_ident() != self.thread_id:
            raise CallbackQualificationError("expired or inherited callback provider refused")
        value = self.clock()
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (self.highwater is not None and value < self.highwater)
        ):
            raise CallbackQualificationError("finite current callback clock required")
        self.highwater = value
        return value

    def _current(self, config):
        self._now()
        if (
            config != self.config
            or _campaign_config() != self.config
            or self._selected_role() != self.role_selectors
        ):
            raise CallbackQualificationError("callback configuration changed")

    def _transport(self, client):
        meta = getattr(client, "meta", None)
        options = getattr(meta, "config", None)
        retries = getattr(options, "retries", None)
        if (
            getattr(meta, "region_name", None) != self.config.region
            or type(retries) is not dict
            or type(retries.get("total_max_attempts")) is not int
            or retries["total_max_attempts"] != 1
        ):
            raise CallbackQualificationError("selected single-attempt callback metadata client required")
        for field, limit in (("connect_timeout", 5), ("read_timeout", 10)):
            value = getattr(options, field, None)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= limit:
                raise CallbackQualificationError("bounded callback metadata transport required")

    def _read(self, config, client, method, **kwargs):
        self._current(config)
        self._transport(client)
        try:
            result = getattr(client, method)(**kwargs)
        except Exception:
            raise CallbackQualificationError("current callback metadata unavailable") from None
        self._current(config)
        self._transport(client)
        metadata = result.get("ResponseMetadata") if type(result) is dict else None
        if (
            type(metadata) is not dict
            or type(metadata.get("HTTPStatusCode")) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata.get("RetryAttempts")) is not int
            or metadata["RetryAttempts"] != 0
        ):
            raise CallbackQualificationError("current callback metadata unqualified")
        return result

    def _endpoint(self, config, runtime_id):
        result = self._read(
            config,
            self.control,
            "get_agent_runtime_endpoint",
            agentRuntimeId=runtime_id,
            endpointName=config.qualifier,
        )
        if any(
            type(result.get(key)) is not str or result[key] != expected
            for key, expected in (
                ("agentRuntimeArn", config.runtime_arn),
                ("name", config.qualifier),
                ("status", "READY"),
                ("liveVersion", config.runtime_version),
            )
        ) or result.get("targetVersion") not in (None, config.runtime_version):
            raise CallbackQualificationError("current callback endpoint differs")

    def qualify_current(self, config, identity):
        try:
            self._current(config)
            before = self._now()
            caller = self._read(config, self.sts, "get_caller_identity")
            arn, user = caller.get("Arn"), caller.get("UserId")
            role_name = self.role_arn.rsplit("/", 1)[1]
            prefix = "arn:aws:sts::" + config.account_id + ":assumed-role/" + role_name + "/"
            session = arn[len(prefix) :] if type(arn) is str and arn.startswith(prefix) else None
            if (
                caller.get("Account") != config.account_id
                or type(session) is not str
                or re.fullmatch(r"[A-Za-z0-9_+=,.@-]{2,64}", session) is None
                or user != self.role_id + ":" + session
            ):
                raise CallbackQualificationError("current callback principal differs")
            runtime_id = config.runtime_arn.rsplit("/", 1)[1]
            self._endpoint(config, runtime_id)
            runtime = self._read(
                config,
                self.control,
                "get_agent_runtime",
                agentRuntimeId=runtime_id,
                agentRuntimeVersion=config.runtime_version,
            )
            if any(
                type(runtime.get(key)) is not str or runtime[key] != expected
                for key, expected in (
                    ("agentRuntimeArn", config.runtime_arn),
                    ("agentRuntimeVersion", config.runtime_version),
                    ("status", "READY"),
                )
            ):
                raise CallbackQualificationError("current callback runtime differs")
            self._endpoint(config, runtime_id)
            self._current(config)
            observed = self._now()
            if observed - before >= 60:
                raise CallbackQualificationError("callback metadata qualification expired")
            return CallbackRouteObservation(
                config.account_id,
                config.region,
                config.deployment_id,
                config.runtime_arn,
                config.qualifier,
                config.runtime_version,
                config.route_sha256,
                observed,
                observed + 30,
                self.process_id,
                self,
            )
        except CallbackQualificationError:
            raise
        except Exception:
            raise CallbackQualificationError("current callback provider unavailable") from None


@dataclass
class _CallbackServices:
    route: Any
    dynamodb_client: Any
    runtime_client: Any
    clock: Any
    process_id: int
    thread_id: int
    active: bool = True
    highwater: float | None = None

    def now(self):
        if not self.active or self.process_id != os.getpid() or self.thread_id != threading.get_ident():
            raise CallbackQualificationError("expired/inherited/cross-thread callback context refused")
        value = self.clock()
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (self.highwater is not None and value < self.highwater)
        ):
            raise CallbackQualificationError("finite nondecreasing callback clock required")
        self.highwater = value
        return value


@contextmanager
def campaign_callback_scope(
    route_qualifier, *, dynamodb_client=None, runtime_client=None, clock=time.monotonic
):
    """Private bootstrap/test seam; injected objects are not authentication proof."""
    if not callable(getattr(route_qualifier, "qualify_current", None)) or not callable(clock):
        raise CallbackQualificationError("current private route/role qualifier required")
    services = _CallbackServices(
        route_qualifier, dynamodb_client, runtime_client, clock, os.getpid(), threading.get_ident()
    )
    token = _CAMPAIGN_SERVICES.set(services)
    try:
        yield
    finally:
        services.active = False
        _CAMPAIGN_SERVICES.reset(token)


def _campaign_config():
    values = {
        name: os.environ.get(env_name, "").strip()
        for name, env_name in (
            ("account_id", "AMLAS_AWS_ACCOUNT_ID"),
            ("region", "AWS_REGION"),
            ("deployment_id", "AMLAS_DEPLOYMENT_ID"),
            ("jobs_table", "JOBS_TABLE"),
            ("runtime_arn", "AGENTCORE_RUNTIME_ARN"),
            ("qualifier", "AGENTCORE_RUNTIME_QUALIFIER"),
            ("runtime_version", "AGENTCORE_RUNTIME_VERSION"),
            ("route_sha256", "AMLAS_CALLBACK_ROUTE_SHA256"),
        )
    }
    if (
        not re.fullmatch("[0-9]{12}", values["account_id"])
        or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-[0-9]+", values["region"])
        or str(uuid.UUID(values["deployment_id"])) != values["deployment_id"]
        or not re.fullmatch(r"[A-Za-z0-9_.-]{3,255}", values["jobs_table"])
        or not re.fullmatch(
            r"arn:aws:bedrock-agentcore:"
            + values["region"]
            + ":"
            + values["account_id"]
            + r":runtime/[A-Za-z0-9_][A-Za-z0-9_-]{1,127}",
            values["runtime_arn"],
        )
        or not re.fullmatch(r"[A-Za-z0-9_]{1,48}", values["qualifier"])
        or not re.fullmatch(r"[1-9][0-9]{0,18}", values["runtime_version"])
    ):
        raise ValueError("closed selected callback route configuration required")
    _sha(values["route_sha256"])
    return CallbackRouteConfig(**values)


def _event_identity(event, config):
    if type(event) is not dict or type(event.get("source")) is not str or event["source"] != "aws.sagemaker":
        raise ValueError("unsupported event source")
    if type(event.get("detail-type")) is not str:
        raise ValueError("exact managed event type required")
    kind = {
        "SageMaker Processing Job State Change": "Processing",
        "SageMaker Training Job State Change": "Training",
    }.get(event.get("detail-type"))
    if kind is None or type(event.get("detail")) is not dict:
        raise ValueError("unsupported managed job event")
    for name, expected in (("account", config.account_id), ("region", config.region)):
        if type(event.get(name)) is not str or event[name] != expected:
            raise ValueError("event deployment scope differs")
    event_id = _exact_text(event.get("id"), 128)
    detail = event["detail"]
    name = _exact_text(detail.get(kind + "JobName"), 63)
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", name):
        raise ValueError("native job name differs")
    arn = _exact_text(detail.get(kind + "JobArn"), 2048)
    expected_arn = f"arn:aws:sagemaker:{config.region}:{config.account_id}:{'processing-job' if kind == 'Processing' else 'training-job'}/{name}"
    status = detail.get(kind + "JobStatus")
    if type(status) is not str or status not in ("Completed", "Failed", "Stopped") or arn != expected_arn:
        raise ValueError("native job/status identity differs")
    resources = event.get("resources")
    if (
        type(resources) is not list
        or not 1 <= len(resources) <= 8
        or any(type(value) is not str for value in resources)
        or arn not in resources
    ):
        raise ValueError("event resource identity differs")
    for resource in resources:
        _exact_text(resource, 2048)
    return {"event_id": event_id, "kind": kind, "job_name": name, "job_arn": arn}


def _qualify(services, config, identity):
    before = services.now()
    try:
        observation = services.route.qualify_current(config, dict(identity))
    except Exception:
        raise CallbackQualificationError("current private route/role evidence unavailable") from None
    after = services.now()
    if type(observation) is not CallbackRouteObservation:
        raise CallbackQualificationError("current private route evidence unknown")
    for name in (
        "account_id",
        "region",
        "deployment_id",
        "runtime_arn",
        "qualifier",
        "runtime_version",
        "route_sha256",
    ):
        value = getattr(observation, name)
        if type(value) is not str or value != getattr(config, name):
            raise CallbackQualificationError("current callback route/role differs")
    if (
        type(observation.process_id) is not int
        or observation.process_id != services.process_id
        or observation.issuer is not services.route
        or type(observation.observed_at) not in (int, float)
        or type(observation.expires_at) not in (int, float)
        or not math.isfinite(observation.observed_at)
        or not math.isfinite(observation.expires_at)
        or not before <= observation.observed_at <= after
        or not observation.observed_at < observation.expires_at <= observation.observed_at + 60
        or after >= observation.expires_at
    ):
        raise CallbackQualificationError("current route issuer/process/deadline refused")


def _client(services, name, config):
    current = services.dynamodb_client if name == "dynamodb" else services.runtime_client
    if current is None:
        from botocore.config import Config

        current = boto3.client(
            name,
            region_name=config.region,
            config=Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=10),
        )
    retries = getattr(getattr(getattr(current, "meta", None), "config", None), "retries", None)
    if (
        type(retries) is not dict
        or type(retries.get("total_max_attempts")) is not int
        or retries["total_max_attempts"] != 1
    ):
        raise CallbackQualificationError("callback transport requires one wire attempt")
    transport_config = getattr(getattr(current, "meta", None), "config", None)
    for field, limit in (("connect_timeout", 5), ("read_timeout", 10)):
        value = getattr(transport_config, field, None)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= limit:
            raise CallbackQualificationError("bounded exact callback connect/read timeouts required")
    return current


def wake_job_lookup_key(kind, job_name):
    """Pure exact selector for untrusted routing DATA, never authentication."""
    if type(kind) is not str or kind not in ("Processing", "Training"):
        raise ValueError("closed native job kind required")
    name = _exact_text(job_name, 63)
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", name):
        raise ValueError("closed native job name required")
    return "CAMPAIGN_WAKE_JOB#" + kind + "#" + name


def encode_wake_job_binding(body):
    """Canonical frozen eleven-field DATA codec for Root pre-Create publication.

    This validates structural consistency only. Protected intent/publisher origin,
    table retention and current qualification remain the publication owner's gates.
    """
    if type(body) is not dict or set(body) != set(_HINT_FIELDS):
        raise ValueError("closed eleven-field binding required")
    if type(body["schema_version"]) is not int or body["schema_version"] != 1:
        raise ValueError("binding schema differs")
    if type(body["record_kind"]) is not str or body["record_kind"] != "campaign_wake_job_v1":
        raise ValueError("binding kind differs")
    for name in _HINT_FIELDS[1:]:
        _exact_text(body[name], 2048 if name == "job_arn" else 256)
    if not re.fullmatch("[0-9]{12}", body["account_id"]) or not re.fullmatch(
        r"[a-z]{2}(?:-[a-z]+)+-[0-9]+", body["region"]
    ):
        raise ValueError("binding scope differs")
    if str(uuid.UUID(body["deployment_id"])) != body["deployment_id"]:
        raise ValueError("binding deployment differs")
    wake_job_lookup_key(body["kind"], body["job_name"])
    resource = "processing-job" if body["kind"] == "Processing" else "training-job"
    if (
        body["job_arn"]
        != f"arn:aws:sagemaker:{body['region']}:{body['account_id']}:{resource}/{body['job_name']}"
    ):
        raise ValueError("binding native identity differs")
    if not re.fullmatch(r"campaign_[a-f0-9]{64}_[a-f0-9]{64}", body["campaign_ref"]):
        raise ValueError("binding original reference differs")
    _sha(body["operation_digest"])
    if not 33 <= len(body["transport_session_id"]) <= 256:
        raise ValueError("binding transport correlation differs")
    raw = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    if len(raw) > 8192:
        raise ValueError("binding byte cap exceeded")
    return raw


def _hint(row, identity, config):
    if type(row) is not dict or set(row) != {"job_name", "private_record"}:
        raise ValueError("closed lookup envelope required")
    key = wake_job_lookup_key(identity["kind"], identity["job_name"])
    if (
        row["job_name"] != {"S": key}
        or type(row["private_record"]) is not dict
        or set(row["private_record"]) != {"B"}
    ):
        raise ValueError("lookup identity differs")
    raw = row["private_record"]["B"]
    if type(raw) is not bytes or not 0 < len(raw) <= 8192:
        raise ValueError("bounded exact lookup bytes required")

    def pairs(values):
        result = {}
        for name, value in values:
            if name in result:
                raise ValueError("duplicate lookup field")
            result[name] = value
        return result

    body = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite lookup")),
    )
    if (
        type(body) is not dict
        or set(body) != set(_HINT_FIELDS)
        or type(body["schema_version"]) is not int
        or body["schema_version"] != 1
    ):
        raise ValueError("closed lookup body required")
    if type(body["record_kind"]) is not str or body["record_kind"] != "campaign_wake_job_v1":
        raise ValueError("lookup kind differs")
    for name, expected in (
        ("account_id", config.account_id),
        ("region", config.region),
        ("deployment_id", config.deployment_id),
        ("kind", identity["kind"]),
        ("job_name", identity["job_name"]),
        ("job_arn", identity["job_arn"]),
    ):
        if type(body[name]) is not str or body[name] != expected:
            raise ValueError("original lookup native scope differs")
    reference = _exact_text(body["campaign_ref"], 138)
    if not re.fullmatch(r"campaign_[a-f0-9]{64}_[a-f0-9]{64}", reference):
        raise ValueError("original lookup reference differs")
    _sha(body["operation_digest"])
    session = _exact_text(body["transport_session_id"])
    if not 33 <= len(session) <= 256:
        raise ValueError("original transport correlation differs")
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    if canonical != raw:
        raise ValueError("canonical lookup bytes differ")
    return body


def _campaign_result(code, delivery_status, body=None):
    result = {
        "statusCode": code,
        "delivery_status": delivery_status,
        "lifecycle_verified": False,
        "quality_success": False,
    }
    if body is not None:
        result.update(
            campaign_ref=body["campaign_ref"],
            job_arn=body["job_arn"],
            operation_digest=body["operation_digest"],
        )
    return result


# The resume action answers with one closed first event. A normal wake is acknowledged only by
# that event in a positive state; the runtime is our own code behind IAM, so a body can only cause
# retries unless it carries exactly this event, bound to this delivery's run.
_RESUME_STATES = frozenset(
    {"restoring", "continuing", "completed", "original_active", "failed_transient", "failed"}
)
_RESUME_ACKNOWLEDGED = frozenset({"continuing", "completed"})
_RESUME_EVENT_BYTES = 64 * 1024


def _resume_event_state(text, run_id):
    """Closed schema of the first resume event: exact keys, known state, this delivery's run."""

    def pairs(values):
        result = {}
        for name, value in values:
            if name in result:
                raise ValueError("duplicate resume event field")
            result[name] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite resume event value")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    base = {"event", "state", "run_id"}
    if type(value) is not dict or set(value) not in (base, base | {"diagnostic"}):
        raise ValueError("resume event schema differs")
    state = value["state"]
    if (
        type(value["event"]) is not str
        or value["event"] != "resume_state"
        or type(state) is not str
        or state not in _RESUME_STATES
        or type(value["run_id"]) is not str
        or value["run_id"] != run_id
    ):
        raise ValueError("resume event fields differ")
    if "diagnostic" in value:
        diagnostic = value["diagnostic"]
        if (
            not state.startswith("failed")
            or type(diagnostic) is not dict
            or set(diagnostic) != {"restore_stage", "exception_class"}
            or any(type(item) is not str or not 0 < len(item) <= 64 for item in diagnostic.values())
        ):
            raise ValueError("resume event diagnostic differs")
    return state


def _first_resume_state(stream, services, deadline, run_id):
    """Return the state of the first complete event line, reading one byte at a time.

    The runtime keeps the response open after its events, and a read of n bytes on a live
    stream blocks until n bytes arrive, so only single-byte reads cannot wait past the
    event. Reading stops at the first complete event: the rest of the stream is never read.
    """
    count, line = 0, bytearray()
    while True:
        if services.now() >= deadline:
            raise TimeoutError("original delivery drain deadline")
        chunk = stream.read(1)
        if services.now() >= deadline or type(chunk) is not bytes or len(chunk) > 1:
            raise TimeoutError("original delivery drain unsupported")
        if not chunk:
            raise ValueError("resume event absent")
        count += 1
        if count > _RESUME_EVENT_BYTES:
            raise ValueError("resume event exceeds bound")
        if chunk != b"\n":
            line.extend(chunk)
            continue
        text = bytes(line).decode("utf-8").strip()
        line.clear()
        if not text or text.startswith(":"):
            continue  # SSE separator or comment: not an event
        if text.startswith("data:"):
            text = text[5:].strip()
        elif not text.startswith("{"):
            raise ValueError("resume event framing differs")
        if not text:
            raise ValueError("resume event absent")
        return _resume_event_state(text, run_id)


def _drain(response, services, current_qualification, *, resume_run_id=None):
    # Own the returned closeable object before any clock/qualification/interface
    # operation can fail. Body content remains untrusted delivery data: it is discarded,
    # except that a resume delivery (resume_run_id) reads only its first closed event.
    stream = response.get("response") if type(response) is dict else None
    close = getattr(stream, "close", None)
    deadline = None
    state = None
    try:
        current_qualification()  # post-Invoke, before all body reads
        # The deadline bounds the body read: the qualification above has its own transport bounds.
        deadline, count = services.now() + 20, 0
        if not callable(getattr(stream, "read", None)) or not callable(close):
            raise ValueError("bounded delivery stream required")
        metadata = response.get("ResponseMetadata")
        if (
            type(metadata) is not dict
            or type(metadata.get("HTTPStatusCode")) is not int
            or metadata["HTTPStatusCode"] != 200
            or type(metadata.get("RetryAttempts")) is not int
            or metadata["RetryAttempts"] != 0
        ):
            raise ValueError("successful nonretried delivery response evidence required")
        if resume_run_id is not None:
            state = _first_resume_state(stream, services, deadline, resume_run_id)
        else:
            while True:
                if services.now() >= deadline:
                    raise TimeoutError("original delivery drain deadline")
                chunk = stream.read(65536)
                if services.now() >= deadline or type(chunk) is not bytes:
                    raise TimeoutError("original delivery drain unsupported")
                count += len(chunk)
                if count > 1024 * 1024:
                    raise ValueError("original delivery response exceeds bound")
                if not chunk:
                    break
    finally:
        if callable(close):
            close()  # exactly one owned cleanup attempt, including all denials
    if deadline is not None and services.now() >= deadline:
        raise TimeoutError("original delivery close exceeded measured deadline")
    return state


_NORMAL_WAKE_FIELDS = {
    "schema_version",
    "record_kind",
    "logical_run_id",
    "transport_session_id",
    "account_id",
    "region",
    "deployment_id",
    "table_id",
    "kind",
    "job_name",
    "job_arn",
    "operation_digest",
}


def _normal_wake_job_lookup_key(kind, job_name):
    key = wake_job_lookup_key(kind, job_name)
    return key.replace("CAMPAIGN_WAKE_JOB#", "PROMPT_WAKE_JOB#", 1)


def _normal_hint(row, identity, config):
    """Validate routing DATA only; this record grants no permission."""
    if type(row) is not dict or set(row) != {"job_name", "record", "record_sha256"}:
        raise ValueError("invalid normal routing envelope")
    for name in ("job_name", "record", "record_sha256"):
        if type(row[name]) is not dict or set(row[name]) != {"S"} or type(row[name]["S"]) is not str:
            raise ValueError("invalid normal routing field")
    key = _normal_wake_job_lookup_key(identity["kind"], identity["job_name"])
    if row["job_name"]["S"] != key:
        raise ValueError("conflicting normal routing key")
    raw = _exact_text(row["record"]["S"], 8192)
    digest = _sha(row["record_sha256"]["S"])
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
        raise ValueError("changed normal routing bytes")

    def pairs(values):
        result = {}
        for name, value in values:
            if name in result:
                raise ValueError("duplicate normal routing field")
            result[name] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite normal routing value")

    body = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    if type(body) is not dict or set(body) != _NORMAL_WAKE_FIELDS:
        raise ValueError("invalid normal routing body")
    if type(body["schema_version"]) is not int or body["schema_version"] != 1:
        raise ValueError("invalid normal routing schema")
    if body["record_kind"] != "normal_prompt_wake_job_v1":
        raise ValueError("invalid normal routing kind")
    for name in _NORMAL_WAKE_FIELDS - {"schema_version"}:
        _exact_text(body[name], 512 if name == "job_arn" else 256)
    if not re.fullmatch(r"[a-f0-9]{32}", body["logical_run_id"]):
        raise ValueError("invalid normal logical run")
    if len(body["transport_session_id"]) < 33:
        raise ValueError("invalid normal transport session")
    if str(uuid.UUID(body["table_id"])) != body["table_id"]:
        raise ValueError("invalid normal table incarnation")
    _sha(body["operation_digest"])
    expected = (
        config.account_id,
        config.region,
        config.deployment_id,
        identity["kind"],
        identity["job_name"],
        identity["job_arn"],
    )
    actual = tuple(
        body[name] for name in ("account_id", "region", "deployment_id", "kind", "job_name", "job_arn")
    )
    if actual != expected:
        raise ValueError("conflicting normal routing identity")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if canonical != raw:
        raise ValueError("noncanonical normal routing body")
    return body


def _normal_table(description, body, config):
    if type(description) is not dict or type(description.get("Table")) is not dict:
        raise ValueError("invalid normal table description")
    table = description["Table"]
    if (
        table.get("TableStatus") != "ACTIVE"
        or table.get("TableName") != config.jobs_table
        or table.get("TableArn")
        != f"arn:aws:dynamodb:{config.region}:{config.account_id}:table/{config.jobs_table}"
        or table.get("TableId") != body["table_id"]
        or table.get("KeySchema") != [{"AttributeName": "job_name", "KeyType": "HASH"}]
    ):
        raise ValueError("changed normal table binding")
    definitions = table.get("AttributeDefinitions")
    if type(definitions) is not list:
        raise ValueError("invalid normal table attributes")
    key_definitions = [
        value for value in definitions if type(value) is dict and value.get("AttributeName") == "job_name"
    ]
    if key_definitions != [{"AttributeName": "job_name", "AttributeType": "S"}]:
        raise ValueError("invalid normal table key")


def _normal_result(code, delivery_status, body=None):
    result = _campaign_result(code, delivery_status)
    if body is not None:
        result.update(
            run_id=body["logical_run_id"], job_arn=body["job_arn"], operation_digest=body["operation_digest"]
        )
    return result


def _retain_normal_terminal_hint(row, body, identity, status, config, services, ddb):
    """Retain immutable terminal routing DATA; this grants no readiness."""
    if type(status) is not str or status not in ("Completed", "Failed", "Stopped"):
        raise ValueError("normal terminal status differs")
    hint = {name: body[name] for name in _NORMAL_WAKE_FIELDS - {"transport_session_id"}}
    hint.update(
        record_kind="normal_prompt_terminal_hint_v1",
        status=status,
        normal_route_sha256=row["record_sha256"]["S"],
    )
    raw = json.dumps(hint, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if not 0 < len(raw.encode("utf-8")) <= 8192:
        raise ValueError("normal terminal hint byte bound exceeded")
    item = {
        "job_name": {"S": "PROMPT_TERMINAL_HINT#" + identity["kind"] + "#" + identity["job_name"]},
        "record": {"S": raw},
        "record_sha256": {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
    }

    def current():
        _qualify(services, config, identity)
        description = ddb.describe_table(TableName=config.jobs_table)
        _qualify(services, config, identity)
        _normal_table(description, body, config)
        _qualify(services, config, identity)
        selected = ddb.get_item(
            TableName=config.jobs_table,
            Key={"job_name": row["job_name"]},
            ConsistentRead=True,
        )
        _qualify(services, config, identity)
        if type(selected) is not dict or selected.get("Item") != row:
            raise ValueError("normal terminal route changed")
        _normal_hint(selected["Item"], identity, config)
        description = ddb.describe_table(TableName=config.jobs_table)
        _qualify(services, config, identity)
        _normal_table(description, body, config)
        _qualify(services, config, identity)

    current()
    created = False
    try:
        ddb.put_item(
            TableName=config.jobs_table,
            Item=item,
            ConditionExpression="attribute_not_exists(job_name)",
        )
        created = True
    except PermissionError:
        raise
    except Exception as error:
        response = getattr(error, "response", None)
        fields = response.get("Error") if type(response) is dict else None
        code = fields.get("Code") if type(fields) is dict else None
        if code in {
            "AccessDenied",
            "AccessDeniedException",
            "UnauthorizedOperation",
            "UnrecognizedClientException",
            "InvalidClientTokenId",
            "ExpiredToken",
            "ExpiredTokenException",
        }:
            raise
    finally:
        current()
    current()
    confirmed = ddb.get_item(
        TableName=config.jobs_table,
        Key={"job_name": item["job_name"]},
        ConsistentRead=True,
    )
    current()
    if type(confirmed) is not dict or confirmed.get("Item") != item:
        raise ValueError("normal terminal hint publication unconfirmed")
    current()
    # A confirmed hint that this delivery did not create proves publication, not that this delivery is first.
    return current, created


def _invoke_normal(body, identity, config, services, current_route):
    kwargs = {
        "agentRuntimeArn": config.runtime_arn,
        "runtimeSessionId": body["transport_session_id"],
        "qualifier": config.qualifier,
        "payload": json.dumps(
            {"action": "resume_prompt_run", "run_id": body["logical_run_id"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    }
    try:
        client = _client(services, "bedrock-agentcore", config)
        _qualify(services, config, identity)
    except CallbackQualificationError:
        return _normal_result(403, "qualification_unavailable", body)
    except Exception:
        return _normal_result(503, "setup_unavailable", body)
    try:
        current_route()
        response = client.invoke_agent_runtime(**kwargs)
        state = _drain(response, services, current_route, resume_run_id=body["logical_run_id"])
        current_route()
    except Exception:
        return _normal_result(202, "outcome_unknown", body)
    if state not in _RESUME_ACKNOWLEDGED:
        # restoring, original_active and failed* did not start or continue the run: retry.
        return _normal_result(202, "outcome_unknown", body)
    return _normal_result(200, "delivered", body)


def _stream_body(item, *, ascii_json=True):
    if type(item) is not dict or set(item) != {"job_name", "record", "record_sha256"}:
        raise ValueError("bounded stream row envelope differs")
    for name in item:
        if type(item[name]) is not dict or set(item[name]) != {"S"} or type(item[name]["S"]) is not str:
            raise ValueError("stream scalar envelope differs")
    raw = _exact_text(item["record"]["S"], 160 * 1024)
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != _sha(item["record_sha256"]["S"]):
        raise ValueError("stream row digest differs")

    def pairs(values):
        result = {}
        for name, value in values:
            if name in result:
                raise ValueError("duplicate stream DATA field")
            result[name] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite stream DATA")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    if (
        type(value) is not dict
        or json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_json, allow_nan=False)
        != raw
    ):
        raise ValueError("stream row canonical DATA differs")
    return value


def _stream_journal(item):
    fields = {
        "job_name",
        "runtime_session_id",
        "record_kind",
        "operation_digest",
        "record_revision",
        "record_state",
        "status",
        "private_record",
    }
    if type(item) is not dict or set(item) != fields:
        raise ValueError("original journal envelope differs")
    result = {}
    for name in fields - {"record_revision", "private_record"}:
        if type(item[name]) is not dict or set(item[name]) != {"S"}:
            raise ValueError("original journal scalar differs")
        result[name] = _exact_text(item[name]["S"], 256)
    if result["record_kind"] != "amlas-private-sft-operation-v1":
        raise ValueError("unrelated journal kind")
    if type(item["record_revision"]) is not dict or set(item["record_revision"]) != {"N"}:
        raise ValueError("journal revision type differs")
    revision = item["record_revision"]["N"]
    if type(revision) is not str or not re.fullmatch("0|[1-9][0-9]{0,18}", revision):
        raise ValueError("journal revision differs")
    result["record_revision"] = revision
    _sha(result["operation_digest"])
    if str(uuid.UUID(result["runtime_session_id"])) != result["runtime_session_id"]:
        raise ValueError("journal correlation differs")
    blob = item["private_record"]
    if (
        type(blob) is not dict
        or set(blob) != {"B"}
        or type(blob["B"]) not in (bytes, str)
        or not 0 < len(blob["B"]) <= 600 * 1024
    ):
        raise ValueError("opaque journal bound differs")
    return result


def _normal_stream_linkage(body):
    required = {
        "schema_version",
        "logical_run_id",
        "session_sha256",
        "scope",
        "table_id",
        "private_store_correlation",
        "admitted_prompt",
        "history",
        "stage",
        "sft",
        "task_checkpoint_ref",
    }
    if (
        not required.issubset(body)
        or type(body["schema_version"]) is not int
        or body["schema_version"] != 1
        or body["stage"] != "sft"
        or type(body["scope"]) is not dict
        or type(body["history"]) is not list
        or type(body["sft"]) is not dict
    ):
        raise ValueError("normal readiness canonical row schema differs")
    if not re.fullmatch("[a-f0-9]{32}", _exact_text(body["logical_run_id"], 32)):
        raise ValueError("normal readiness logical run differs")
    _sha(body["session_sha256"])
    for name in ("private_store_correlation", "table_id"):
        value = _exact_text(body[name], 36)
        if str(uuid.UUID(value)) != value:
            raise ValueError("normal readiness original correlation differs")
    prompt = body["admitted_prompt"]
    if (
        type(prompt) is not str
        or not prompt.strip()
        or len(json.dumps(prompt, ensure_ascii=True).encode("utf-8")) > 160 * 1024
    ):
        raise ValueError("normal readiness admitted prompt differs")
    # Bookkeeping appended after the TASK transition (history, the continuation, the report pointers of
    # the original worker and of a replacement) is not part of the immutable linkage.
    bookkeeping = {
        "history",
        "continuation",
        "claim",
        "report",
        "report_checkpoint_ref",
        "resumed_report_checkpoint_ref",
    }
    immutable = {key: value for key, value in body.items() if key not in bookkeeping}
    return json.dumps(immutable, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normal_readiness_candidate(record):
    if (
        type(record) is not dict
        or record.get("eventSource") != "aws:dynamodb"
        or record.get("eventName") not in ("INSERT", "MODIFY")
    ):
        return None
    data = record.get("dynamodb")
    if type(data) is not dict or data.get("StreamViewType") != "NEW_AND_OLD_IMAGES":
        raise ValueError("complete stream images required")
    keys, new, old = data.get("Keys"), data.get("NewImage"), data.get("OldImage")
    if (
        type(keys) is not dict
        or set(keys) != {"job_name"}
        or type(keys["job_name"]) is not dict
        or set(keys["job_name"]) != {"S"}
    ):
        raise ValueError("stream primary key differs")
    key = _exact_text(keys["job_name"]["S"], 256)
    if type(new) is not dict or new.get("job_name") != keys["job_name"]:
        raise ValueError("stream new row key differs")
    if record["eventName"] == "MODIFY" and (type(old) is not dict or old.get("job_name") != keys["job_name"]):
        raise ValueError("stream original row key differs")
    if key.startswith("PROMPT_RUN#"):
        body = _stream_body(new)
        if not re.fullmatch("PROMPT_RUN#[a-f0-9]{32}", key) or body.get("logical_run_id") != key[11:]:
            raise ValueError("normal stream run identity differs")
        if body.get("stage") == "enrolled":
            # Enrollment precedes any SFT association, so it is never TASK readiness;
            # refusing it would hold an unlimited-retry stream behind every new run.
            return None
        if body.get("stage") != "sft":
            raise ValueError("normal stream run identity differs")
        prior = _stream_body(old).get("task_checkpoint_ref") if old is not None else None
        task = body.get("task_checkpoint_ref")
        if task is None or json.dumps(task, sort_keys=True, allow_nan=False) == json.dumps(
            prior, sort_keys=True, allow_nan=False
        ):
            return None
        fields = {"uri", "version_id", "sha256", "byte_count", "format"}
        if (
            type(task) is not dict
            or set(task) != fields
            or task["format"] != "json"
            or type(task["byte_count"]) is not int
            or not 0 < task["byte_count"] <= 160 * 1024
        ):
            raise ValueError("normal TASK descriptor differs")
        _sha(task["sha256"])
        _exact_text(task["uri"], 4096)
        version = _exact_text(task["version_id"], 1024)
        if version.lower() == "null":
            raise ValueError("normal TASK version differs")
        association = body.get("sft")
        if type(association) is not dict:
            raise ValueError("normal original association absent")
        return {
            "key": key,
            "task": task,
            "linkage": _normal_stream_linkage(body),
            "run_id": body["logical_run_id"],
            "job_name": association.get("native_job_name"),
            "job_arn": association.get("native_job_arn"),
            "operation_digest": association.get("operation_digest"),
            "table_id": body.get("table_id"),
        }
    if new.get("record_kind") != {"S": "amlas-private-sft-operation-v1"}:
        return None
    body = _stream_journal(new)
    prior = _stream_journal(old) if old is not None else None
    if (prior is None and body["record_revision"] == "0") or (
        prior is not None
        and (prior["record_revision"], prior["operation_digest"])
        == (body["record_revision"], body["operation_digest"])
    ):
        return None
    return {
        "key": key,
        "journal": body,
        "job_name": body["job_name"],
        "operation_digest": body["operation_digest"],
    }


def _normal_stream_table(description, config, stream_arn, table_id=None):
    if type(description) is not dict or type(description.get("Table")) is not dict:
        raise ValueError("native stream table absent")
    table = description["Table"]
    _normal_table(
        description, {"table_id": table_id if table_id is not None else table.get("TableId")}, config
    )
    actual_id = table.get("TableId")
    if (
        type(actual_id) is not str
        or str(uuid.UUID(actual_id)) != actual_id
        or table.get("LatestStreamArn") != stream_arn
        or table.get("StreamSpecification") != {"StreamEnabled": True, "StreamViewType": "NEW_AND_OLD_IMAGES"}
    ):
        raise ValueError("native stream binding differs")


def _deliver_normal_readiness(record, candidate, config, services, ddb):
    name = _exact_text(candidate["job_name"], 63)
    if not re.fullmatch("[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", name):
        raise ValueError("readiness native name differs")
    arn = "arn:aws:sagemaker:" + config.region + ":" + config.account_id + ":training-job/" + name
    if "job_arn" in candidate and candidate["job_arn"] != arn:
        raise ValueError("readiness native ARN differs")
    identity = {
        "event_id": _exact_text(record.get("eventID"), 128),
        "kind": "Training",
        "job_name": name,
        "job_arn": arn,
    }
    stream_arn = _exact_text(record.get("eventSourceARN"), 2048)
    if record.get("awsRegion") != config.region:
        raise ValueError("readiness stream region differs")
    table_id = candidate.get("table_id")

    def native():
        _qualify(services, config, identity)
        description = ddb.describe_table(TableName=config.jobs_table)
        _qualify(services, config, identity)
        _normal_stream_table(description, config, stream_arn, table_id)
        _qualify(services, config, identity)

    def read(key):
        native()
        response = ddb.get_item(
            TableName=config.jobs_table, Key={"job_name": {"S": key}}, ConsistentRead=True
        )
        native()
        if type(response) is not dict or type(response.get("Item")) is not dict:
            raise ValueError("current readiness row unavailable")
        return response["Item"]

    producer = read(candidate["key"])
    if "journal" in candidate:
        if _stream_journal(producer) != candidate["journal"]:
            raise ValueError("current original journal revision differs")
    else:
        producer_body = _stream_body(producer)
        if (
            producer.get("job_name") != {"S": candidate["key"]}
            or _normal_stream_linkage(producer_body) != candidate["linkage"]
            or json.dumps(producer_body.get("task_checkpoint_ref"), sort_keys=True, allow_nan=False)
            != json.dumps(candidate["task"], sort_keys=True, allow_nan=False)
        ):
            raise ValueError("current normal TASK original linkage differs")
    route = read(_normal_wake_job_lookup_key("Training", name))
    body = _normal_hint(route, identity, config)
    table_id = body["table_id"]
    native()
    if body["operation_digest"] != candidate["operation_digest"] or (
        "run_id" in candidate and body["logical_run_id"] != candidate["run_id"]
    ):
        raise ValueError("readiness original route differs")
    run_key = "PROMPT_RUN#" + body["logical_run_id"]
    run = producer if candidate["key"] == run_key else read(run_key)
    run_body = _stream_body(run)
    scope = run_body.get("scope")
    if (
        type(scope) is not dict
        or any(
            scope.get(key) != value
            for key, value in (
                ("account_id", config.account_id),
                ("region", config.region),
                ("deployment_id", config.deployment_id),
                ("jobs_table", config.jobs_table),
            )
        )
        or run_body.get("session_sha256")
        != hashlib.sha256(body["transport_session_id"].encode("utf-8")).hexdigest()
    ):
        raise ValueError("current readiness native scope/session differs")
    association = run_body.get("sft")
    if (
        run_body.get("logical_run_id") != body["logical_run_id"]
        or run_body.get("table_id") != table_id
        or type(association) is not dict
        or (
            association.get("native_job_name"),
            association.get("native_job_arn"),
            association.get("operation_digest"),
        )
        != (name, arn, body["operation_digest"])
    ):
        raise ValueError("current readiness enrollment differs")
    if (
        "journal" in candidate
        and run_body.get("private_store_correlation") != candidate["journal"]["runtime_session_id"]
    ):
        raise ValueError("current readiness journal correlation differs")
    hint_key = "PROMPT_TERMINAL_HINT#Training#" + name
    hint = read(hint_key)
    hint_body = _stream_body(hint, ascii_json=False)
    status = hint_body.get("status")
    if type(status) is not str or status not in ("Completed", "Failed", "Stopped"):
        raise ValueError("retained terminal status differs")
    expected = {key: body[key] for key in _NORMAL_WAKE_FIELDS - {"transport_session_id"}}
    expected.update(
        record_kind="normal_prompt_terminal_hint_v1",
        status=status,
        normal_route_sha256=route["record_sha256"]["S"],
    )
    raw = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    expected_item = {
        "job_name": {"S": hint_key},
        "record": {"S": raw},
        "record_sha256": {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
    }
    if len(raw.encode("utf-8")) > 8192 or hint != expected_item:
        raise ValueError("current terminal hint linkage differs")
    pinned = {candidate["key"]: producer, route["job_name"]["S"]: route, run_key: run, hint_key: hint}

    def current():
        for key, item in pinned.items():
            if read(key) != item:
                raise ValueError("readiness current row changed")
        native()

    current()
    return _invoke_normal(body, identity, config, services, current)


def _bounded_stream_event(event):
    def plain(value, depth=0):
        if depth > 24:
            raise ValueError("stream DATA depth bound exceeded")
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("stream DATA key type differs")
            for item in value.values():
                plain(item, depth + 1)
        elif type(value) is list:
            for item in value:
                plain(item, depth + 1)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("stream DATA type differs")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("stream DATA nonfinite value")

    plain(event)
    measured = 0
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode(event):
        measured += len(chunk.encode("utf-8"))
        if measured > 4 * 1024 * 1024:
            raise ValueError("total stream event byte bound exceeded")


def _normal_stream_handler(event):
    services = _CAMPAIGN_SERVICES.get()
    if type(services) is not _CallbackServices:
        return _normal_result(403, "qualification_missing")
    try:
        _bounded_stream_event(event)
        config = _campaign_config()
        records = event.get("Records")
        if type(records) is not list or not 1 <= len(records) <= 16:
            raise ValueError("bounded readiness batch required")
        candidates = [(record, _normal_readiness_candidate(record)) for record in records]
        selected = [(record, candidate) for record, candidate in candidates if candidate is not None]
        if not selected:
            return _normal_result(200, "ignored")
        ddb = _client(services, "dynamodb", config)
        result = _normal_result(200, "ignored")
        for record, candidate in selected:
            result = _deliver_normal_readiness(record, candidate, config, services, ddb)
            if result.get("delivery_status") != "delivered":
                return result
        return result
    except CallbackQualificationError:
        return _normal_result(403, "qualification_unavailable")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        return _normal_result(409, "readiness_refused")
    except Exception:
        return _normal_result(503, "readiness_unavailable")


def _deliver_normal(row, identity, config, services, ddb, terminal_status):
    try:
        body = _normal_hint(row, identity, config)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        return _normal_result(400, "lookup_invalid")
    try:
        _qualify(services, config, identity)
        description = ddb.describe_table(TableName=config.jobs_table)
        _qualify(services, config, identity)
    except CallbackQualificationError:
        return _normal_result(403, "qualification_unavailable")
    except Exception:
        return _normal_result(503, "lookup_unavailable")
    try:
        _normal_table(description, body, config)
        _qualify(services, config, identity)
    except CallbackQualificationError:
        return _normal_result(403, "qualification_unavailable")
    except (ValueError, TypeError, KeyError, UnicodeError):
        return _normal_result(409, "lookup_invalid")
    try:
        current_route, created = _retain_normal_terminal_hint(
            row, body, identity, terminal_status, config, services, ddb
        )
    except CallbackQualificationError:
        return _normal_result(403, "qualification_unavailable", body)
    except Exception:
        return _normal_result(503, "terminal_hint_unavailable", body)
    if not created:
        delay = _twin_delay()
        if delay > 0:
            _twin_pause(delay)
    return _invoke_normal(body, identity, config, services, current_route)


def _campaign_handler(event):
    services = _CAMPAIGN_SERVICES.get()
    if type(services) is not _CallbackServices:
        return _campaign_result(403, "qualification_missing")
    try:
        config = _campaign_config()
        identity = _event_identity(event, config)
    except (ValueError, TypeError, AttributeError, UnicodeError):
        return _campaign_result(400, "input_refused")
    try:
        _qualify(services, config, identity)
        ddb = _client(services, "dynamodb", config)
        _qualify(services, config, identity)
        normal_response = ddb.get_item(
            TableName=config.jobs_table,
            ConsistentRead=True,
            Key={"job_name": {"S": _normal_wake_job_lookup_key(identity["kind"], identity["job_name"])}},
        )
        _qualify(services, config, identity)
        if type(normal_response) is not dict:
            return _normal_result(503, "lookup_unavailable")
        if "Item" in normal_response:
            return _deliver_normal(
                normal_response["Item"],
                identity,
                config,
                services,
                ddb,
                event["detail"][identity["kind"] + "JobStatus"],
            )
        _qualify(services, config, identity)
        response = ddb.get_item(
            TableName=config.jobs_table,
            ConsistentRead=True,
            Key={"job_name": {"S": "CAMPAIGN_WAKE_JOB#" + identity["kind"] + "#" + identity["job_name"]}},
        )
        _qualify(services, config, identity)
    except CallbackQualificationError:
        return _campaign_result(403, "qualification_refused")
    except Exception:
        return _campaign_result(503, "lookup_unavailable")
    row = response.get("Item") if type(response) is dict else None
    if row is None:
        return _campaign_result(404, "binding_missing")
    try:
        body = _hint(row, identity, config)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        return _campaign_result(409, "binding_refused")
    wake = {name: body[name] for name in _WAKE_FIELDS if name not in ("event_id",)}
    wake["event_id"] = identity["event_id"]
    kwargs = {
        "agentRuntimeArn": config.runtime_arn,
        "runtimeSessionId": body["transport_session_id"],
        "qualifier": config.qualifier,
        "payload": json.dumps(wake, sort_keys=True, separators=(",", ":")).encode(),
    }
    try:
        client = _client(services, "bedrock-agentcore", config)
        _qualify(services, config, identity)
    except CallbackQualificationError:
        return _campaign_result(403, "qualification_refused", body)
    except Exception:
        return _campaign_result(503, "setup_unavailable", body)
    try:
        response = client.invoke_agent_runtime(**kwargs)
        _drain(response, services, lambda: _qualify(services, config, identity))
        _qualify(services, config, identity)
    except Exception:
        return _campaign_result(202, "outcome_unknown", body)
    return _campaign_result(200, "delivered", body)
