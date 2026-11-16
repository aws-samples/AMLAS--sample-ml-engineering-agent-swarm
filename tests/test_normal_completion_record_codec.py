"""Declared continuation row DATA; no runtime/native/claim authority is supplied.

Exercise only the actual row decoder and a strict inert GetItem transport. These
cases do not prove original qualification, completion CAS, SDK load or recovery.
"""

from __future__ import annotations

import copy

import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from test_normal_checkpoint_cas_contract import TABLE_ID, _declared_enrollment, _item

from amlas.control_plane import normal_run_checkpoint as checkpoint

DDB_MODEL = Session().get_service_model("dynamodb")


def _declared_sft_record(*, continuation=True, epoch=1):
    context, record, _initial = _declared_enrollment()
    digest = "3" * 64
    record.update(
        stage="sft",
        sft={
            "model_ref": "model_acq_" + "4" * 32,
            "preparation_ref": "prep_" + "5" * 32,
            "intent": {"operation": "train"},
            "operation_digest": digest,
            "native_job_name": "declared-codec-original",
            "native_job_arn": (
                "arn:aws:sagemaker:us-west-2:111122223333:training-job/declared-codec-original"
            ),
        },
    )
    if continuation:
        record["task_checkpoint_ref"] = {
            "uri": (
                "s3://declared-codec-bucket/sessions/"
                + record["private_store_correlation"]
                + "/prompt-runs/"
                + context.run_id
                + "/tasks/"
                + "6" * 64
                + ".json"
            ),
            "version_id": "declared-codec-version",
            "sha256": "6" * 64,
            "byte_count": 64,
            "format": "json",
        }
        record["continuation"] = {
            "schema_version": 1,
            "epoch": epoch,
            "token": "7" * 32,
            "status": "consumed-ready",
            "operation_digest": digest,
            "sdk_snapshot": {
                "scope": "agent",
                "schema_version": "1.0",
                "created_at": "2026-01-01T00:00:00+00:00",
                "data": {
                    "messages": [{"role": "user", "content": [{"text": "Declared codec DATA."}]}],
                    "state": {},
                    "conversation_manager_state": {},
                    "interrupt_state": {},
                },
                "app_data": {},
            },
        }
    return context, record


class DeclaredGetPort:
    def __init__(self, context, item):
        self.context = context
        self.item = copy.deepcopy(item)
        self.calls = 0

    def get_item(self, **kwargs):
        validate_parameters(kwargs, DDB_MODEL.operation_model("GetItem").input_shape)
        assert kwargs == {
            "TableName": self.context.catalog.jobs_table,
            "Key": {"job_name": self.item["job_name"]},
            "ConsistentRead": True,
        }
        self.calls += 1
        return {"Item": copy.deepcopy(self.item)}


def _decode(context, record):
    item = _item(context, record)
    port = DeclaredGetPort(context, item)
    boundaries = []

    def data_boundary():
        # Unit DATA plumbing only: no runtime/native qualification is asserted.
        boundaries.append("decode")

    loaded = checkpoint._load(context, port, TABLE_ID, data_boundary)
    assert boundaries == ["decode", "decode"] and port.calls == 1
    assert port.item == item
    return loaded


@pytest.mark.parametrize("epoch", [1, 2])
def test_declared_continuation_row_decodes_without_granting_authority(epoch):
    context, record = _declared_sft_record(epoch=epoch)
    item, decoded = _decode(context, record)
    assert item == _item(context, record) and decoded == record


@pytest.mark.parametrize("stage", ["enrolled", "sft"])
def test_legacy_row_without_continuation_keeps_decoder_acceptance(stage):
    if stage == "enrolled":
        context, record, _initial = _declared_enrollment()
    else:
        context, record = _declared_sft_record(continuation=False)
    assert "continuation" not in record
    _item_value, decoded = _decode(context, record)
    assert decoded == record


@pytest.mark.parametrize(
    "fault",
    [
        "extra_field",
        "missing_or_bad_schema",
        "bool_epoch",
        "zero_epoch",
        "invalid_token",
        "invalid_status",
        "digest_mismatch",
        "task_absent",
        "non_sft",
        "invalid_snapshot",
        "invalid_task_descriptor",
        "null_continuation",
        "non_dict_continuation",
        "zero_task_bytes",
    ],
)
def test_malformed_continuation_data_refuses(fault):
    context, original = _declared_sft_record()
    variants = []
    if fault == "missing_or_bad_schema":
        absent = copy.deepcopy(original)
        absent["continuation"].pop("schema_version")
        variants.append(absent)
        for version in (True, 2):
            changed = copy.deepcopy(original)
            changed["continuation"]["schema_version"] = version
            variants.append(changed)
    elif fault == "invalid_task_descriptor":
        for field, value in (("format", "csv"), ("byte_count", 160 * 1024 + 1)):
            changed = copy.deepcopy(original)
            changed["task_checkpoint_ref"][field] = value
            variants.append(changed)
    else:
        changed = copy.deepcopy(original)
        data = changed["continuation"]
        if fault == "extra_field":
            data["unrecognized"] = "Declared unsupported DATA."
        elif fault == "bool_epoch":
            data["epoch"] = True
        elif fault == "zero_epoch":
            data["epoch"] = 0
        elif fault == "invalid_token":
            data["token"] = "A" * 32
        elif fault == "invalid_status":
            data["status"] = "unknown"
        elif fault == "digest_mismatch":
            data["operation_digest"] = "8" * 64
        elif fault == "task_absent":
            changed.pop("task_checkpoint_ref")
        elif fault == "non_sft":
            changed["stage"] = "enrolled"
            changed.pop("sft")
            changed.pop("task_checkpoint_ref")
        elif fault == "invalid_snapshot":
            data["sdk_snapshot"]["scope"] = "unsupported"
        elif fault == "null_continuation":
            changed["continuation"] = None
        elif fault == "non_dict_continuation":
            changed["continuation"] = []
        elif fault == "zero_task_bytes":
            changed["task_checkpoint_ref"]["byte_count"] = 0
        else:
            raise AssertionError("unsupported declared test case")
        variants.append(changed)
    for record in variants:
        with pytest.raises(ValueError):
            _decode(context, record)
