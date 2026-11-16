"""Fresh sequential native metadata reads; no approval or identity attestation.

Resource-policy text is preserved as returned UTF8 bytes. IAM trust is the defined
canonical encoding of the SDK-decoded JSON object, not raw HTTP bytes. Fixed role
selectors identify metadata to inspect; they do not prove an effective co-owner
cohort. Native reader permissions and transport qualification remain integration
gates. No observations, grants or witnesses are cached or serialized.

SDK validation supplies structural type/required-field checks, not every modeled
pattern. Qualification additionally checks consumed identity/status/protocol/
version/tag/trust pins explicitly. Endpoint id is not admission authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass

from .campaign_admission import (
    _ACCOUNT,
    _ENDPOINT_NAME,
    _PREFIX,
    _REGION,
    _RUNTIME,
    _VERSION,
    AdmissionManifest,
    AdmissionObservation,
    RoleIncarnation,
    _native_fields,
    _policy,
    _role_fields,
    _uuid,
)


class NativeAdmissionReadError(ValueError):
    """Unusable or changed native metadata; no partial observation is exposed."""


def _trust_value(value, depth=0):
    if depth > 32:
        raise NativeAdmissionReadError("bounded SDK trust depth required")
    if type(value) is dict:
        if len(value) > 2048 or any(type(key) is not str for key in value):
            raise NativeAdmissionReadError("closed bounded SDK trust object required")
        for child in value.values():
            _trust_value(child, depth + 1)
    elif type(value) is list:
        if len(value) > 4096:
            raise NativeAdmissionReadError("bounded SDK trust array required")
        for child in value:
            _trust_value(child, depth + 1)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise NativeAdmissionReadError("JSON primitive SDK trust data required")
    elif type(value) is float and not math.isfinite(value):
        raise NativeAdmissionReadError("finite SDK trust numbers required")


def canonical_sdk_trust(document):
    """Defined UTF8 JSON-object projection; never raw native HTTP trust bytes."""
    if type(document) is not dict:
        raise NativeAdmissionReadError("decoded native SDK trust object required")
    _trust_value(document)
    try:
        raw = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        _policy(raw, hashlib.sha256(raw).hexdigest())
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise NativeAdmissionReadError("bounded canonical SDK trust encoding refused") from None
    return raw


@dataclass(frozen=True)
class NativeAdmissionSelectors:
    account_id: str
    region: str
    deployment_id: str
    deployment_prefix: str
    runtime_arn: str
    runtime_id: str
    runtime_version: str
    endpoint_arn: str
    endpoint_name: str
    runtime_execution_role: RoleIncarnation
    co_owner_roles: tuple[RoleIncarnation, ...]

    def __post_init__(self):
        try:
            if type(self.account_id) is not str or not _ACCOUNT.fullmatch(self.account_id):
                raise ValueError
            if type(self.region) is not str or not _REGION.fullmatch(self.region):
                raise ValueError
            _uuid(self.deployment_id)
            if type(self.deployment_prefix) is not str or not _PREFIX.fullmatch(self.deployment_prefix):
                raise ValueError
            if type(self.runtime_arn) is not str:
                raise ValueError
            match = _RUNTIME.fullmatch(self.runtime_arn)
            if not match or (match[1], match[2], match[3]) != ("aws", self.region, self.account_id):
                raise ValueError
            if self.runtime_id != self.runtime_arn.rsplit("/", 1)[1]:
                raise ValueError
            if type(self.runtime_version) is not str or not _VERSION.fullmatch(self.runtime_version):
                raise ValueError
            if type(self.endpoint_name) is not str or not _ENDPOINT_NAME.fullmatch(self.endpoint_name):
                raise ValueError
            if self.endpoint_arn != self.runtime_arn + "/runtime-endpoint/" + self.endpoint_name:
                raise ValueError
            if type(self.co_owner_roles) is not tuple or not 1 <= len(self.co_owner_roles) <= 16:
                raise ValueError
            roles = (self.runtime_execution_role, *self.co_owner_roles)
            for role in roles:
                _role_fields(role, self.account_id, "aws")
                if canonical_sdk_trust(json.loads(role.trust_policy)) != role.trust_policy:
                    raise ValueError
            if len({role.role_arn for role in self.co_owner_roles}) != len(self.co_owner_roles):
                raise ValueError
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
            raise NativeAdmissionReadError("closed fixed unapproved native selectors required") from None

    @classmethod
    def from_manifest(cls, manifest):
        """Copy inspection selectors only; never approve the unapproved manifest."""
        if type(manifest) is not AdmissionManifest:
            raise NativeAdmissionReadError("exact unapproved server manifest required")
        try:
            _native_fields(manifest)
            return cls(
                account_id=manifest.account_id,
                region=manifest.region,
                deployment_id=manifest.deployment_id,
                deployment_prefix=manifest.deployment_prefix,
                runtime_arn=manifest.runtime_arn,
                runtime_id=manifest.runtime_arn.rsplit("/", 1)[1],
                runtime_version=manifest.runtime_version,
                endpoint_arn=manifest.endpoint_arn,
                endpoint_name=manifest.endpoint_name,
                runtime_execution_role=manifest.runtime_execution_role,
                co_owner_roles=manifest.co_owner_roles,
            )
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise NativeAdmissionReadError("native selector projection refused") from None


class NativeAdmissionReader:
    def __init__(self, selectors, *, control_client, iam_client):
        if type(selectors) is not NativeAdmissionSelectors:
            raise NativeAdmissionReadError("exact private native selectors required")
        selectors.__post_init__()
        self._process_id = os.getpid()
        self._selectors = selectors
        self._control, self._iam = control_client, iam_client
        self._transports()

    def _process(self):
        if os.getpid() != self._process_id:
            raise NativeAdmissionReadError("inherited native reader requires independent construction")

    def _transports(self):
        self._process()
        expected = (
            (
                self._control,
                "bedrock-agentcore-control",
                {self._selectors.region},
                f"https://bedrock-agentcore-control.{self._selectors.region}.amazonaws.com",
            ),
            (self._iam, "iam", {"aws-global", "us-east-1"}, "https://iam.amazonaws.com"),
        )
        try:
            for client, service, regions, endpoint in expected:
                meta = client.meta
                retries = meta.config.retries
                if (
                    meta.service_model.service_name != service
                    or meta.region_name not in regions
                    or meta.endpoint_url != endpoint
                    or type(retries) is not dict
                    or type(retries.get("total_max_attempts")) is not int
                    or retries["total_max_attempts"] != 1
                ):
                    raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError):
            raise NativeAdmissionReadError("selected one-attempt native read transports required") from None

    def _read(self, client, operation, method, kwargs):
        self._process()
        self._transports()
        try:
            from botocore.validate import validate_parameters

            model = client.meta.service_model.operation_model(operation)
            validate_parameters(kwargs, model.input_shape)
            response = getattr(client, method)(**kwargs)
            self._process()
            if type(response) is not dict or set(response) - (
                set(model.output_shape.members) | {"ResponseMetadata"}
            ):
                raise ValueError
            metadata = response.get("ResponseMetadata")
            if (
                type(metadata) is not dict
                or type(metadata.get("HTTPStatusCode")) is not int
                or metadata["HTTPStatusCode"] != 200
                or type(metadata.get("RetryAttempts")) is not int
                or metadata["RetryAttempts"] != 0
            ):
                raise ValueError
            projected = {key: value for key, value in response.items() if key != "ResponseMetadata"}
            if operation == "GetAgentRuntimeEndpoint" and projected.get("targetVersion") is None:
                projected.pop("targetVersion", None)
            if operation == "GetRole":
                role = projected.get("Role")
                if type(role) is not dict:
                    raise ValueError
                trust = canonical_sdk_trust(role.get("AssumeRolePolicyDocument"))
                projected["Role"] = {**role, "AssumeRolePolicyDocument": trust.decode("utf-8")}
            validate_parameters(projected, model.output_shape)
            return response
        except Exception:
            raise NativeAdmissionReadError("native read refused at " + operation) from None

    def _role(self, selected):
        path_name = selected.role_arn.split(":role/", 1)[1]
        name = path_name.rsplit("/", 1)[-1]
        path = "/" + path_name[: -len(name)]
        response = self._read(self._iam, "GetRole", "get_role", {"RoleName": name})
        role = response["Role"]
        if (role.get("Arn"), role.get("RoleId"), role.get("RoleName"), role.get("Path")) != (
            selected.role_arn,
            selected.role_id,
            name,
            path,
        ):
            raise NativeAdmissionReadError("native role incarnation/identity differs")
        raw = canonical_sdk_trust(role.get("AssumeRolePolicyDocument"))
        if raw != selected.trust_policy or hashlib.sha256(raw).hexdigest() != selected.trust_sha256:
            raise NativeAdmissionReadError("native canonical trust differs from selected pin")
        return RoleIncarnation(role["Arn"], role["RoleId"], raw, hashlib.sha256(raw).hexdigest())

    def _resource_policy(self, arn):
        response = self._read(self._control, "GetResourcePolicy", "get_resource_policy", {"resourceArn": arn})
        text = response.get("policy")
        if type(text) is not str or not text.strip():
            raise NativeAdmissionReadError("native resource policy evidence is unknown")
        try:
            raw = text.encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            _policy(raw, digest)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise NativeAdmissionReadError("native raw resource policy bytes refused") from None
        return raw, digest

    def read_current(self):
        """Return fresh untrusted metadata from sequential, non-atomic native reads."""
        self._process()
        self._transports()
        s = self._selectors
        runtime = self._read(
            self._control,
            "GetAgentRuntime",
            "get_agent_runtime",
            {"agentRuntimeId": s.runtime_id, "agentRuntimeVersion": s.runtime_version},
        )
        if (
            runtime.get("agentRuntimeArn") != s.runtime_arn
            or runtime.get("agentRuntimeId") != s.runtime_id
            or runtime.get("agentRuntimeVersion") != s.runtime_version
            or runtime.get("roleArn") != s.runtime_execution_role.role_arn
            or runtime.get("status") != "READY"
            or type(runtime.get("protocolConfiguration")) is not dict
            or runtime["protocolConfiguration"].get("serverProtocol") != "HTTP"
        ):
            raise NativeAdmissionReadError("native runtime is not the selected qualified runtime")
        endpoint = self._read(
            self._control,
            "GetAgentRuntimeEndpoint",
            "get_agent_runtime_endpoint",
            {"agentRuntimeId": s.runtime_id, "endpointName": s.endpoint_name},
        )
        if (
            endpoint.get("agentRuntimeEndpointArn") != s.endpoint_arn
            or endpoint.get("agentRuntimeArn") != s.runtime_arn
            or endpoint.get("name") != s.endpoint_name
            or endpoint.get("status") != "READY"
            or endpoint.get("liveVersion") != s.runtime_version
            or endpoint.get("targetVersion") not in (None, s.runtime_version)
        ):
            raise NativeAdmissionReadError("native endpoint is not the selected qualified endpoint")
        tags = self._read(
            self._control, "ListTagsForResource", "list_tags_for_resource", {"resourceArn": s.runtime_arn}
        ).get("tags")
        if type(tags) is not dict or (
            tags.get("amlas:deployment-id"),
            tags.get("amlas:deployment-prefix"),
        ) != (s.deployment_id, s.deployment_prefix):
            raise NativeAdmissionReadError("native deployment tag binding differs or is missing")
        runtime_policy, runtime_sha = self._resource_policy(s.runtime_arn)
        endpoint_policy, endpoint_sha = self._resource_policy(s.endpoint_arn)
        execution = self._role(s.runtime_execution_role)
        roles = tuple(self._role(role) for role in s.co_owner_roles)
        observation = AdmissionObservation(
            account_id=s.account_id,
            region=s.region,
            deployment_id=tags["amlas:deployment-id"],
            deployment_prefix=tags["amlas:deployment-prefix"],
            runtime_arn=runtime["agentRuntimeArn"],
            runtime_execution_role=execution,
            endpoint_arn=endpoint["agentRuntimeEndpointArn"],
            endpoint_name=endpoint["name"],
            runtime_version=runtime["agentRuntimeVersion"],
            endpoint_live_version=endpoint["liveVersion"],
            # None means unreported target (absent or null), not a fabricated observation.
            endpoint_target_version=endpoint.get("targetVersion"),
            runtime_policy=runtime_policy,
            runtime_policy_sha256=runtime_sha,
            endpoint_policy=endpoint_policy,
            endpoint_policy_sha256=endpoint_sha,
            co_owner_roles=roles,
        )
        try:
            _native_fields(observation)
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
            raise NativeAdmissionReadError("closed current native observation refused") from None
        self._process()
        return observation
