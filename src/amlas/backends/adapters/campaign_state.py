"""Conditional campaign records in the existing selected deployment jobs table.

No table/IAM creation, registry changes, training Create call or failure-row delete.
Native STS binds execution-role/account scope, not the invoking human. Fresh action
policy and complete native/artifact validation remain the lifecycle owner's gates.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from ...control_plane.campaign_state import (
    CampaignOperationIntent,
    CampaignStateCatalog,
    CampaignStateError,
    OriginalJobReservation,
    TrustedExecutionBinding,
    _binding_fields,
    _decode_private_record,
    _encode_private_record,
    _mint_binding,
    _native_identity,
    operation_digest,
)
from ...control_plane.sft_state import SftSubmissionSnapshot

logger = logging.getLogger(__name__)
_KIND = "amlas-private-sft-operation-v1"
_STATUS = {
    "RESERVED": "PENDING",
    "CREATE_ATTEMPTED": "PENDING",
    "ACCEPTED": "RUNNING",
    "AMBIGUOUS": "SUBMISSION_AMBIGUOUS",
    "REJECTED": "SUBMISSION_REJECTED",
}


class DynamoCampaignStateBackend:
    def __init__(
        self,
        catalog: CampaignStateCatalog,
        *,
        dynamodb_client: Any = None,
        sts_client: Any = None,
        sagemaker_client: Any = None,
    ):
        if type(catalog) is not CampaignStateCatalog:
            raise CampaignStateError("typed server deployment catalog required")
        self.catalog = catalog
        self._ddb, self._sts, self._sm = dynamodb_client, sts_client, sagemaker_client

    def _clients(self):
        if self._ddb is None or self._sts is None:
            import boto3

            self._ddb = self._ddb or boto3.client("dynamodb", region_name=self.catalog.region)
            self._sts = self._sts or boto3.client("sts", region_name=self.catalog.region)
        return self._ddb, self._sts

    def _verified_scope(self):
        _ddb, sts = self._clients()
        try:
            value = sts.get_caller_identity()
            role_name = self.catalog.execution_role_arn.rsplit("/", 1)[-1]
            expected = "arn:aws:sts::" + self.catalog.account_id + ":assumed-role/" + role_name + "/"
            if (
                type(value) is not dict
                or value.get("Account") != self.catalog.account_id
                or type(value.get("UserId")) is not str
                or value["UserId"].split(":", 1)[0] != self.catalog.execution_role_id
                or type(value.get("Arn")) is not str
                or not value["Arn"].startswith(expected)
                or not value["Arn"][len(expected) :]
            ):
                raise CampaignStateError("native execution role scope differs")
        except Exception as error:
            raise CampaignStateError("native execution role verification failed") from error

    def bind_execution(self, *, campaign_id: str, runtime_session_id: str) -> TrustedExecutionBinding:
        self._verified_scope()
        return _mint_binding(self.catalog, campaign_id, runtime_session_id)

    def _scope(self, binding):
        fields = _binding_fields(binding)
        expected = dataclasses.asdict(self.catalog)
        if {k: fields[k] for k in expected} != expected:
            raise CampaignStateError("execution binding belongs to another deployment role")
        self._verified_scope()

    def _item(self, reservation):
        return {
            "job_name": {"S": reservation.native_job_name},
            "runtime_session_id": {"S": reservation.binding.runtime_session_id},
            "record_kind": {"S": _KIND},
            "operation_digest": {"S": reservation.operation_digest},
            "record_revision": {"N": str(reservation.revision)},
            "record_state": {"S": reservation.disposition},
            "status": {"S": _STATUS[reservation.disposition]},
            "private_record": {"B": _encode_private_record(reservation)},
        }

    def reserve_original_job(
        self, binding: TrustedExecutionBinding, intent: CampaignOperationIntent
    ) -> OriginalJobReservation:
        self._scope(binding)
        if type(intent) is not CampaignOperationIntent or _binding_fields(intent.binding) != _binding_fields(
            binding
        ):
            raise CampaignStateError("same-scope typed private intent required")
        digest = operation_digest(intent)
        name, arn = _native_identity(intent)
        record = OriginalJobReservation(binding, intent, digest, name, arn)
        try:
            self._clients()[0].put_item(
                TableName=self.catalog.jobs_table,
                Item=self._item(record),
                ConditionExpression="attribute_not_exists(job_name)",
            )
            return dataclasses.replace(record, _persisted_body=_encode_private_record(record))
        except Exception as error:
            if (
                getattr(error, "response", {}).get("Error", {}).get("Code")
                != "ConditionalCheckFailedException"
            ):
                raise CampaignStateError("private reservation write failed") from error
            logger.info("Campaign reservation reused after conditional write conflict")
            current = self.load_original(binding, name)
            if current.operation_digest != digest:
                raise CampaignStateError("reserved native name belongs to another complete intent") from error
            return dataclasses.replace(current, reused=True)

    def load_original(self, binding: TrustedExecutionBinding, job_name: str) -> OriginalJobReservation:
        self._scope(binding)
        if (
            type(job_name) is not str
            or not re.fullmatch(re.escape(self.catalog.deployment_prefix) + r"-sft-[a-f0-9]{18,57}", job_name)
            or len(job_name) > 63
        ):
            raise CampaignStateError("original owned native job selector required")
        try:
            result = self._clients()[0].get_item(
                TableName=self.catalog.jobs_table, Key={"job_name": {"S": job_name}}, ConsistentRead=True
            )
            item = result.get("Item")
            keys = {
                "job_name",
                "runtime_session_id",
                "record_kind",
                "operation_digest",
                "record_revision",
                "record_state",
                "status",
                "private_record",
            }
            if (
                type(item) is not dict
                or set(item) != keys
                or item["record_kind"] != {"S": _KIND}
                or item["job_name"] != {"S": job_name}
                or item["runtime_session_id"] != {"S": binding.runtime_session_id}
                or set(item["private_record"]) != {"B"}
            ):
                raise CampaignStateError("private durable record is missing or differs")
            record = _decode_private_record(item["private_record"]["B"], binding)
            if (
                record.native_job_name != job_name
                or item["operation_digest"] != {"S": record.operation_digest}
                or item["record_revision"] != {"N": str(record.revision)}
                or item["record_state"] != {"S": record.disposition}
                or set(item["status"]) != {"S"}
                or item["status"]["S"] not in set(_STATUS.values()) | {"Completed", "Failed", "Stopped"}
            ):
                raise CampaignStateError("private durable envelope differs")
            return record
        except Exception as error:
            raise CampaignStateError("authenticated private record read failed") from error

    def _transition(self, binding, old, new):
        self._scope(binding)
        if type(old) is not OriginalJobReservation or _binding_fields(old.binding) != _binding_fields(
            binding
        ):
            raise CampaignStateError("original same-scope reservation required")
        prior_body = old._persisted_body or _encode_private_record(old)
        prior_record = _decode_private_record(prior_body, binding)
        if _encode_private_record(prior_record) != _encode_private_record(old):
            raise CampaignStateError("original attached snapshot or record differs")
        new_body = _encode_private_record(new)
        try:
            self._clients()[0].update_item(
                TableName=self.catalog.jobs_table,
                Key={"job_name": {"S": old.native_job_name}},
                ConditionExpression="operation_digest = :digest AND record_revision = :revision AND record_state = :old AND private_record = :prior_body",
                UpdateExpression="SET record_revision = :next, record_state = :new, private_record = :body, #status = :status",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":digest": {"S": old.operation_digest},
                    ":revision": {"N": str(old.revision)},
                    ":old": {"S": old.disposition},
                    ":next": {"N": str(new.revision)},
                    ":new": {"S": new.disposition},
                    ":body": {"B": new_body},
                    ":prior_body": {"B": prior_body},
                    ":status": {"S": _STATUS[new.disposition]},
                },
            )
            return dataclasses.replace(new, _persisted_body=new_body)
        except Exception as error:
            raise CampaignStateError(
                "conditional private transition failed; original record retained"
            ) from error

    def mark_create_attempted(
        self, binding: TrustedExecutionBinding, reservation: OriginalJobReservation
    ) -> OriginalJobReservation:
        if type(reservation) is not OriginalJobReservation or reservation.disposition != "RESERVED":
            raise CampaignStateError("only an original reserved operation may attempt Create")
        return self._transition(
            binding,
            reservation,
            dataclasses.replace(
                reservation, disposition="CREATE_ATTEMPTED", revision=reservation.revision + 1, reused=False
            ),
        )

    def attach_submission_snapshot(
        self,
        binding: TrustedExecutionBinding,
        reservation: OriginalJobReservation,
        snapshot: SftSubmissionSnapshot,
    ) -> OriginalJobReservation:
        if (
            type(reservation) is not OriginalJobReservation
            or reservation.disposition != "CREATE_ATTEMPTED"
            or reservation.submission_snapshot is not None
            or type(snapshot) is not SftSubmissionSnapshot
        ):
            raise CampaignStateError("original attempted operation requires one closed snapshot")
        return self._transition(
            binding,
            reservation,
            dataclasses.replace(
                reservation, submission_snapshot=snapshot, revision=reservation.revision + 1, reused=False
            ),
        )

    def record_submission_outcome(
        self,
        binding: TrustedExecutionBinding,
        reservation: OriginalJobReservation,
        *,
        disposition: str,
        native_job_arn: str | None = None,
        error_code: str | None = None,
    ) -> OriginalJobReservation:
        if (
            type(reservation) is not OriginalJobReservation
            or reservation.disposition not in ("CREATE_ATTEMPTED", "AMBIGUOUS")
            or disposition not in ("ACCEPTED", "AMBIGUOUS", "REJECTED")
        ):
            raise CampaignStateError("original attempted submission disposition required")
        if disposition == "ACCEPTED":
            if native_job_arn != reservation.native_job_arn or error_code is not None:
                raise CampaignStateError("accepted native identity differs from original reservation")
        elif (
            native_job_arn is not None
            or type(error_code) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", error_code)
        ):
            raise CampaignStateError("bounded unresolved/rejected error class required")
        new = dataclasses.replace(
            reservation,
            disposition=disposition,
            revision=reservation.revision + 1,
            error_code=error_code,
            reused=False,
        )
        return self._transition(binding, reservation, new)

    def observe_original(self, binding: TrustedExecutionBinding, job_name: str) -> dict:
        record = self.load_original(binding, job_name)
        base = {
            "observed": False,
            "status": "unresolved",
            "disposition": record.disposition,
            "job_name": record.native_job_name,
            "job_arn": record.native_job_arn,
            "trained": False,
            "artifacts_verified": False,
            "quality_success": False,
        }
        if record.disposition in ("RESERVED", "REJECTED"):
            return {**base, "status": record.disposition.lower()}
        try:
            if self._sm is None:
                import boto3

                self._sm = boto3.client("sagemaker", region_name=self.catalog.region)
            native = self._sm.describe_training_job(TrainingJobName=record.native_job_name)
            model = record.intent.catalog
            if (
                native.get("TrainingJobName") != record.native_job_name
                or native.get("TrainingJobArn") != record.native_job_arn
                or native.get("RoleArn") != model.execution_role_arn
                or native.get("AlgorithmSpecification", {}).get("TrainingImage") != model.image_uri
                or native.get("TrainingJobStatus")
                not in ("InProgress", "Completed", "Failed", "Stopping", "Stopped")
            ):
                raise CampaignStateError("original native SFT identity differs")
            response = self._sm.list_tags(ResourceArn=record.native_job_arn)
            tags = response.get("Tags")
            expected = [
                {"Key": "amlas:" + k, "Value": v}
                for k, v in (
                    ("deployment-id", model.deployment_id),
                    ("deployment-prefix", model.deployment_prefix),
                    ("runtime-session-id", binding.runtime_session_id),
                )
            ]
            if (
                type(tags) is not list
                or response.get("NextToken")
                or sorted(tags, key=lambda v: v["Key"]) != sorted(expected, key=lambda v: v["Key"])
            ):
                raise CampaignStateError("original native SFT ownership tags differ")
            return {
                **base,
                "observed": True,
                "status": "observed",
                "native_status": native["TrainingJobStatus"],
            }
        except Exception as error:
            logger.warning("Original campaign native observation unresolved (%s)", type(error).__name__)
            return {**base, "error_type": type(error).__name__}
