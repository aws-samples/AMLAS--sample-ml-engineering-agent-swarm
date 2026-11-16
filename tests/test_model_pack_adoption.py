"""Native-shaped tiny artifact adoption tests; no model tensors or provider calls."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy

ACCOUNT = "1234" * 3
SESSION = "consumer-session"
PACK = "model_pack_" + "a" * 64
BUCKET = "fixture-private-model-bucket"


class S3:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.copies = 0

    def put(self, key, raw):
        version = "v" + str(len(self.objects) + 1)
        self.objects[(key, version)] = raw
        return DatasetArtifactRef(
            "s3://" + BUCKET + "/" + key, version, hashlib.sha256(raw).hexdigest(), len(raw), "json"
        )

    def get_object(self, **kw):
        self.calls.append(("get", deepcopy(kw)))
        assert kw["ExpectedBucketOwner"] == ACCOUNT and kw["Bucket"] == BUCKET
        raw = self.objects[(kw["Key"], kw["VersionId"])]
        return {"Body": io.BytesIO(raw), "VersionId": kw["VersionId"], "ContentLength": len(raw)}

    def head_object(self, **kw):
        self.calls.append(("head", deepcopy(kw)))
        assert kw["ExpectedBucketOwner"] == ACCOUNT
        raw = self.objects[(kw["Key"], kw["VersionId"])]
        import base64

        return {
            "VersionId": kw["VersionId"],
            "ContentLength": len(raw),
            "ChecksumType": "FULL_OBJECT",
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(raw).digest()).decode(),
        }

    def put_object(self, **kw):
        assert kw["IfNoneMatch"] == "*" and kw["ExpectedBucketOwner"] == ACCOUNT
        ref = self.put(kw["Key"], kw["Body"])
        return {"VersionId": ref.version_id}


class DDB:
    def __init__(self):
        self.rows = {}
        self.puts = 0

    def describe_table(self, **kw):
        return {
            "Table": {
                "TableName": kw["TableName"],
                "TableId": "table-incarnation",
                "TableArn": "arn:aws:dynamodb:us-west-2:" + ACCOUNT + ":table/jobs",
                "TableStatus": "ACTIVE",
                "KeySchema": [{"AttributeName": "job_name", "KeyType": "HASH"}],
            }
        }

    def get_item(self, **kw):
        return (
            {"Item": deepcopy(self.rows.get(kw["Key"]["job_name"]["S"]))}
            if kw["Key"]["job_name"]["S"] in self.rows
            else {}
        )

    def put_item(self, **kw):
        assert kw["ConditionExpression"] == "attribute_not_exists(job_name)"
        key = kw["Item"]["job_name"]["S"]
        assert key not in self.rows
        self.rows[key] = deepcopy(kw["Item"])
        self.puts += 1
        return {}

    def update_item(self, **kw):
        key = kw["Key"]["job_name"]["S"]
        assert self.rows[key]["record"] == kw["ExpressionAttributeValues"][":prior"]
        self.rows[key]["record"] = deepcopy(kw["ExpressionAttributeValues"][":body"])
        self.rows[key]["record_sha256"] = deepcopy(kw["ExpressionAttributeValues"][":sha"])
        return {}


class STS:
    def get_caller_identity(self):
        return {
            "Account": ACCOUNT,
            "Arn": "arn:aws:sts::" + ACCOUNT + ":assumed-role/runtime/current",
            "UserId": "AROATEST:current",
        }


class Model:
    def __init__(self, result):
        self.result = result
        self.reads = 0

    def artifacts(self, handle):
        self.reads += 1
        return deepcopy(self.result)

    def qualified_io(self, fn):
        from contextlib import nullcontext

        return nullcontext()


class Copier:
    def __init__(self, s3):
        self.s3 = s3
        self.copies = 0

    def _copy(self, ref, key, catalog, deadline):
        self.copies += 1
        old = ref.uri.split("/", 3)[3]
        raw = self.s3.objects[(old, ref.version_id)]
        out = self.s3.put(key, raw)
        from dataclasses import replace

        return replace(out, format=ref.format)

    def _raw(self, ref, catalog, deadline, collect=False):
        raw = self.s3.objects[(ref.uri.split("/", 3)[3], ref.version_id)]
        assert hashlib.sha256(raw).hexdigest() == ref.sha256 and len(raw) == ref.byte_count
        return raw if collect else None

    def _verified(self, ref, catalog, deadline):
        self._raw(ref, catalog, deadline)


def subject():
    try:
        m = importlib.import_module("amlas.control_plane.model_pack_adoption")
    except ModuleNotFoundError as error:
        if error.name != "amlas.control_plane.model_pack_adoption":
            raise
        raise AssertionError("missing-symbol: ModelPackAdoptionProvider") from None
    assert callable(getattr(m, "ModelPackAdoptionProvider", None)), (
        "missing-symbol: ModelPackAdoptionProvider"
    )
    return m


def setup(monkeypatch):
    m = subject()
    s3 = S3()
    ddb = DDB()
    source = s3.put("sessions/producer/model/file.json", b"ordinary model metadata")
    manifests = {
        k: s3.put("sessions/producer/model/" + k + ".json", b"ordinary provenance")
        for k in ("commit", "lineage", "receipt")
    }
    result = {
        "success": True,
        "acquired": True,
        "file_refs": [{**source.__dict__, "path": "config.json"}],
        "manifest_refs": {k: v.__dict__ for k, v in manifests.items()},
    }
    producer = {
        "request_ref": "model_acq_" + "b" * 32,
        "id": "fixture-lab-model-acq-" + "b" * 12,
        "arn": "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":processing-job/fixture-lab-model-acq-" + "b" * 12,
        "backend": "model_processing",
        "bindings": {"runtime_session_id": "producer-session"},
        "bucket": BUCKET,
    }
    catalog = {
        "schema_version": 1,
        "packs": [
            {
                "pack_ref": PACK,
                "catalog_id": "qwen3_5_9b",
                "catalog_version": "1",
                "producer": producer,
                "artifacts": result,
            }
        ],
    }
    raw = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode()
    catref = s3.put("model-packs/catalog.json", raw)
    scope = {
        "account_id": ACCOUNT,
        "region": "us-west-2",
        "bucket": BUCKET,
        "jobs_table": "jobs",
        "execution_role_arn": "arn:aws:iam::" + ACCOUNT + ":role/runtime",
        "execution_role_id": "AROATEST",
        "deployment_id": "fd0d82ab-4e03-44c0-9bf1-908af5b2b031",
        "deployment_prefix": "fixture-lab",
    }
    model = Model(result)
    copier = Copier(s3)
    monkeypatch.setattr(
        m,
        "_decode_original",
        lambda row, scope: SimpleNamespace(id=row["id"], arn=row["arn"], extra={"bindings": row["bindings"]}),
    )
    provider = m.ModelPackAdoptionProvider(
        catalog_ref=catref, scope=scope, s3=s3, ddb=ddb, sts=STS(), model_adapter=model, copy_adapter=copier
    )
    return m, provider, s3, ddb, model, copier, producer


def test_adoption_owns_new_refs_and_keeps_original_attribution(monkeypatch):
    _, p, _s3, ddb, model, copy, producer = setup(monkeypatch)
    before = deepcopy(producer)
    out = p.adopt(PACK, SESSION, ApprovalPolicy(mode=ApprovalMode.AUTO))
    assert out["origin"] == "verified_pack_adoption" and out["runtime_session_id"] == SESSION
    assert out["producer"] == before and producer == before
    assert all(v["uri"].startswith("s3://" + BUCKET + "/sessions/" + SESSION + "/") for v in out["file_refs"])
    assert copy.copies == 4 and ddb.puts == 1 and model.reads == 1
    again = p.adopt(PACK, SESSION, ApprovalPolicy(mode=ApprovalMode.AUTO))
    assert again == out and copy.copies == 4


def test_restore_is_read_only_and_rechecks_owned_record(monkeypatch):
    _, p, _s3, _ddb, _model, copy, _ = setup(monkeypatch)
    out = p.adopt(PACK, SESSION, ApprovalPolicy(mode=ApprovalMode.AUTO))
    n = copy.copies
    assert p.restore(out, SESSION, ApprovalPolicy(mode=ApprovalMode.AUTO)) == out and copy.copies == n
    changed = deepcopy(out)
    changed["runtime_session_id"] = "foreign"
    with pytest.raises(ValueError):
        p.restore(changed, SESSION, ApprovalPolicy(mode=ApprovalMode.AUTO))
    assert copy.copies == n


def test_changed_exact_source_version_refuses_without_copy(monkeypatch):
    _, p, _s3, _ddb, model, copy, _ = setup(monkeypatch)
    model.result["file_refs"][0]["version_id"] = "unrecorded"
    with pytest.raises(ValueError):
        p.adopt(PACK, SESSION, ApprovalPolicy(mode=ApprovalMode.AUTO))
    assert copy.copies == 0


def test_no_policy_or_unknown_pack_cannot_copy(monkeypatch):
    _, p, _s3, ddb, _model, copy, _ = setup(monkeypatch)
    with pytest.raises(ValueError):
        p.adopt(PACK, SESSION, None)
    with pytest.raises(ValueError):
        p.adopt("model_pack_" + "c" * 64, SESSION, ApprovalPolicy(mode=ApprovalMode.AUTO))
    assert copy.copies == 0 and not ddb.rows
