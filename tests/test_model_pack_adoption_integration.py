"""Real original codec/model adapter over nine tiny Source-fixture byte streams.

No tensor, native provider, model process or authority graph is read or restored.
The only substitutions are external byte/service ports and source expected byte
pins supplied by the existing model_acquisition_fixtures contract.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import model_acquisition_fixtures as f
import pytest
from botocore.session import Session
from botocore.validate import validate_parameters
from test_model_pack_adoption import DDB, STS

from amlas.backends.adapters import model_weight_acquisition as model_backend
from amlas.backends.types import DatasetArtifactRef
from amlas.config import externalized
from amlas.control_plane import model_pack_adoption as adoption
from amlas.control_plane import producer_snapshot
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.data_eng import model_acquisition_schema as schema
from amlas.data_eng import model_weight_manifest

PACK = "model_pack_" + "a" * 64
CONSUMER = "consumer-session"


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class Objects(f.S3):
    def __init__(self):
        super().__init__()
        self.writes = 0
        self.catalog_version_drift = False
        self.on_catalog_read = None
        self.catalog_key = "model-packs/catalog.json"

    def install(self, key, raw, format="json"):
        self.writes += 1
        version = "immutable-" + str(self.writes)
        assert key not in self.objects
        self.objects[key] = (version, raw)
        return DatasetArtifactRef(
            "s3://" + f.BUCKET + "/" + key, version, hashlib.sha256(raw).hexdigest(), len(raw), format
        )

    def put_object(self, **values):
        validate_parameters(
            values, Session().get_service_model("s3").operation_model("PutObject").input_shape
        )
        assert values["Bucket"] == f.BUCKET and values["ExpectedBucketOwner"] == f.ACCOUNT
        assert values["IfNoneMatch"] == "*"
        return {"VersionId": self.install(values["Key"], values["Body"]).version_id}

    def get_object(self, **values):
        validate_parameters(
            values, Session().get_service_model("s3").operation_model("GetObject").input_shape
        )
        assert values["Bucket"] == f.BUCKET and values["ExpectedBucketOwner"] == f.ACCOUNT
        reply = super().get_object(**values)
        if values["Key"] == self.catalog_key:
            if self.catalog_version_drift:
                reply["VersionId"] = "foreign-version"
            if self.on_catalog_read is not None:
                self.on_catalog_read()
        return reply


class Native(f.Processing):
    def __init__(self):
        super().__init__()
        self.describes = 0

    def describe_processing_job(self, **values):
        self.describes += 1
        return super().describe_processing_job(**values)


class Rows(DDB):
    def __init__(self):
        super().__init__()
        self.table_id = "original-table-incarnation"
        self.updates = 0
        self.update_mode = "normal"

    def describe_table(self, **values):
        reply = super().describe_table(**values)
        reply["Table"]["TableId"] = self.table_id
        return reply

    def get_item(self, **values):
        validate_parameters(
            values, Session().get_service_model("dynamodb").operation_model("GetItem").input_shape
        )
        assert values["ConsistentRead"] is True
        return super().get_item(**values)

    def put_item(self, **values):
        validate_parameters(
            values, Session().get_service_model("dynamodb").operation_model("PutItem").input_shape
        )
        return super().put_item(**values)

    def update_item(self, **values):
        validate_parameters(
            values, Session().get_service_model("dynamodb").operation_model("UpdateItem").input_shape
        )
        self.updates += 1
        key = values["Key"]["job_name"]["S"]
        assert self.rows[key]["record_sha256"] == values["ExpressionAttributeValues"][":prior_sha"]
        if self.update_mode == "before_commit":
            raise TimeoutError("Neutral uncertain conditional publication.")
        result = super().update_item(**values)
        if self.update_mode == "foreign_commit":
            body = json.loads(self.rows[key]["record"]["S"])
            body["pack_ref"] = "model_pack_" + "c" * 64
            raw = wire(body)
            self.rows[key]["record"] = {"S": raw.decode()}
            self.rows[key]["record_sha256"] = {"S": hashlib.sha256(raw).hexdigest()}
        if self.update_mode != "normal":
            raise TimeoutError("Neutral uncertain conditional publication.")
        return result


class Identity(STS):
    def __init__(self):
        self.role_id = "AROATEST"
        self.reads = 0

    def get_caller_identity(self):
        self.reads += 1
        reply = super().get_caller_identity()
        reply["UserId"] = self.role_id + ":current"
        return reply


class Copies:
    """Tiny external server-copy port; exact-version bytes never leave the fixture."""

    def __init__(self, objects):
        self.objects = objects
        self.calls = 0
        self.fail_next = False

    @contextmanager
    def qualified_io(self, current):
        current()
        yield
        current()

    def _copy(self, ref, key, catalog, deadline):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise TimeoutError("Neutral uncertain copy.")
        original = ref.uri.split("/", 3)[3]
        version, raw = self.objects.objects[original]
        assert version == ref.version_id and len(raw) == ref.byte_count
        assert hashlib.sha256(raw).hexdigest() == ref.sha256
        return self.objects.install(key, raw, ref.format)

    def _raw(self, ref, catalog, deadline, collect=False):
        reply = self.objects.get_object(
            Bucket=f.BUCKET,
            Key=ref.uri.split("/", 3)[3],
            VersionId=ref.version_id,
            ExpectedBucketOwner=f.ACCOUNT,
        )
        with reply["Body"] as body:
            raw = body.read(ref.byte_count + 1)
        if (
            reply["VersionId"] != ref.version_id
            or len(raw) != ref.byte_count
            or hashlib.sha256(raw).hexdigest() != ref.sha256
        ):
            raise ValueError("Exact tiny staged object changed.")
        return raw if collect else None

    def _verified(self, ref, catalog, deadline):
        self._raw(ref, catalog, deadline)


def real_case(monkeypatch, tmp_path):
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")
    f.patch_expected(monkeypatch, [model_backend, schema, model_weight_manifest])
    native, objects = Native(), Objects()
    backend = f.configured_backend(model_backend, native, objects)
    handle = backend.submit(catalog_id=f.CATALOG, catalog_version=f.VERSION, runtime_session_id=f.SESSION)
    f.publish_fixture(monkeypatch, tmp_path, backend, handle)
    native.state = "Completed"
    observed = backend.artifacts(handle)
    assert observed["success"] is True and observed["acquired"] is True
    assert len(observed["file_refs"]) == 9
    producer = producer_snapshot._capture(handle, "model_acq_" + "b" * 32, model=True)
    selected = {key: deepcopy(observed[key]) for key in ("success", "acquired", "file_refs", "manifest_refs")}
    row = {
        "pack_ref": PACK,
        "catalog_id": f.CATALOG,
        "catalog_version": f.VERSION,
        "producer": producer,
        "artifacts": selected,
    }
    catalog_ref = objects.install(objects.catalog_key, wire({"schema_version": 1, "packs": [row]}))
    scope = {
        "account_id": f.ACCOUNT,
        "region": f.REGION,
        "bucket": f.BUCKET,
        "jobs_table": "jobs",
        "execution_role_arn": "arn:aws:iam::" + f.ACCOUNT + ":role/runtime",
        "execution_role_id": "AROATEST",
        "deployment_id": f.DEPLOYMENT,
        "deployment_prefix": "fixture-lab",
    }
    rows, identity, copies = Rows(), Identity(), Copies(objects)
    provider = adoption.ModelPackAdoptionProvider(
        catalog_ref=catalog_ref,
        scope=scope,
        s3=objects,
        ddb=rows,
        sts=identity,
        model_adapter=backend,
        copy_adapter=copies,
        scope_getter=lambda: (scope, catalog_ref),
    )
    objects.reads.clear()
    return SimpleNamespace(
        provider=provider,
        rows=rows,
        identity=identity,
        copies=copies,
        objects=objects,
        backend=backend,
        native=native,
        handle=handle,
        producer=producer,
        scope=scope,
        catalog_ref=catalog_ref,
        row=row,
    )


def policy():
    return ApprovalPolicy(mode=ApprovalMode.AUTO, max_auto_cost=None)


def payload_reads(case):
    originals = {
        case.handle.extra["output_prefix"] + entry["path"] for entry in case.row["artifacts"]["file_refs"]
    }
    return [entry for entry in case.objects.reads if entry["Key"] in originals]


def test_genuine_codec_and_adapter_own_committed_nine_file_adoption(monkeypatch, tmp_path):
    case = real_case(monkeypatch, tmp_path)
    original = deepcopy(case.producer)
    out = case.provider.adopt(PACK, CONSUMER, policy())
    assert out["producer"] == original and case.producer == original
    assert len(out["file_refs"]) == 9 and case.copies.calls == 12
    assert all(
        ref["uri"].startswith("s3://" + f.BUCKET + "/sessions/" + CONSUMER + "/") for ref in out["file_refs"]
    )
    assert len(case.native.requests) == 1 and case.rows.puts == case.rows.updates == 1
    assert next(iter(case.rows.rows.values()))["record"]["S"].find('"status":"COMMITTED"') >= 0
    assert case.provider.restore(out, CONSUMER, policy()) == out and case.copies.calls == 12


def test_same_original_handle_reuses_only_completed_private_byteproof(monkeypatch, tmp_path):
    case = real_case(monkeypatch, tmp_path)
    out = case.provider.adopt(PACK, CONSUMER, policy())
    assert len(payload_reads(case)) == 9
    describes, identities = case.native.describes, case.identity.reads
    assert case.provider.restore(out, CONSUMER, policy()) == out
    assert case.native.describes > describes and case.identity.reads > identities
    assert len(payload_reads(case)) == 9, "same_original_source_handle_byteproof_not_reused"
    assert case.copies.calls == 12 and len(case.native.requests) == 1


@pytest.mark.parametrize("change", ["version", "hash", "license", "catalog", "foreign_source"])
def test_changed_real_original_source_refuses_before_copy(monkeypatch, tmp_path, change):
    case = real_case(monkeypatch, tmp_path)
    if change == "version":
        key = case.handle.extra["output_prefix"] + case.row["artifacts"]["file_refs"][0]["path"]
        _, raw = case.objects.objects[key]
        case.objects.objects[key] = ("foreign-version", raw)
    elif change == "hash":
        key = case.handle.extra["output_prefix"] + case.row["artifacts"]["file_refs"][0]["path"]
        version, raw = case.objects.objects[key]
        case.objects.objects[key] = (version, bytes([raw[0] ^ 1]) + raw[1:])
    elif change == "license":
        prefix = case.handle.extra["output_prefix"]
        old_version, raw = case.objects.objects[prefix + "LINEAGE.json"]
        lineage = json.loads(raw)
        lineage["expected_manifest"]["license"] = "unreviewed-license"
        changed = schema.encoded(lineage)
        receipt = schema.encoded(
            schema.receipt(f.CATALOG, f.VERSION, case.handle.extra["bindings"], f.evidence(), changed)
        )
        committed = schema.encoded(
            schema.commit(f.CATALOG, f.VERSION, case.handle.extra["bindings"], changed, receipt)
        )
        case.objects.objects[prefix + "LINEAGE.json"] = (old_version, changed)
        for name, body in (("MODEL_ACQUIRED.json", receipt), ("COMMITTED.json", committed)):
            version, _ = case.objects.objects[prefix + name]
            case.objects.objects[prefix + name] = (version, body)
    else:
        key = case.objects.catalog_key
        version, raw = case.objects.objects[key]
        catalog = json.loads(raw)
        if change == "catalog":
            catalog["packs"][0]["catalog_version"] = "2"
        else:
            catalog["packs"][0]["producer"]["bindings"]["source_commit"] = "d" * 40
        changed = wire(catalog)
        ref = DatasetArtifactRef(
            case.catalog_ref.uri, "changed-catalog", hashlib.sha256(changed).hexdigest(), len(changed), "json"
        )
        case.objects.objects[key] = (ref.version_id, changed)
        case.provider = adoption.ModelPackAdoptionProvider(
            catalog_ref=ref,
            scope=case.scope,
            s3=case.objects,
            ddb=case.rows,
            sts=case.identity,
            model_adapter=case.backend,
            copy_adapter=case.copies,
        )
    with pytest.raises(ValueError):
        case.provider.adopt(PACK, CONSUMER, policy())
    assert case.copies.calls == 0 and case.rows.puts == 0 and len(case.native.requests) == 1


@pytest.mark.parametrize("change", ["catalog_hash", "catalog_version", "role", "table", "policy"])
def test_current_catalog_scope_and_policy_drift_refuse_copy(monkeypatch, tmp_path, change):
    case = real_case(monkeypatch, tmp_path)
    selected = policy()
    assert len(case.provider.candidates(selected)) == 1
    if change == "catalog_hash":
        version, raw = case.objects.objects[case.objects.catalog_key]
        case.objects.objects[case.objects.catalog_key] = (version, raw.replace(b'"packs"', b'"pocks"', 1))
    elif change == "catalog_version":
        case.objects.catalog_version_drift = True
    elif change == "role":
        case.identity.role_id = "RECREATEDROLE"
    elif change == "table":
        case.rows.table_id = "recreated-table-incarnation"
    else:
        case.objects.on_catalog_read = lambda: setattr(selected, "mode", ApprovalMode.ASK)
    with pytest.raises(ValueError):
        case.provider.adopt(PACK, CONSUMER, selected)
    assert case.copies.calls == 0 and case.rows.puts == 0


@pytest.mark.parametrize("status", ["RESERVED", "UNKNOWN"])
def test_unresolved_reservation_never_replays_copy(monkeypatch, tmp_path, status):
    case = real_case(monkeypatch, tmp_path)
    case.copies.fail_next = True
    with pytest.raises(TimeoutError):
        case.provider.adopt(PACK, CONSUMER, policy())
    assert case.rows.puts == 1 and case.copies.calls == 1
    row = next(iter(case.rows.rows.values()))
    value = json.loads(row["record"]["S"])
    value["status"] = status
    raw = wire(value)
    row["record"] = {"S": raw.decode()}
    row["record_sha256"] = {"S": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError):
        case.provider.adopt(PACK, CONSUMER, policy())
    assert case.copies.calls == 1 and case.rows.puts == 1


@pytest.mark.parametrize("change", ["version", "same_size_hash"])
def test_restore_refuses_exact_staged_object_drift(monkeypatch, tmp_path, change):
    case = real_case(monkeypatch, tmp_path)
    out = case.provider.adopt(PACK, CONSUMER, policy())
    key = out["file_refs"][0]["uri"].split("/", 3)[3]
    version, raw = case.objects.objects[key]
    case.objects.objects[key] = (
        ("foreign-stage-version", raw) if change == "version" else (version, bytes([raw[0] ^ 1]) + raw[1:])
    )
    with pytest.raises((ValueError, AssertionError)):
        case.provider.restore(out, CONSUMER, policy())
    assert case.copies.calls == 12 and case.rows.puts == case.rows.updates == 1


@pytest.mark.parametrize("mode", ["after_commit", "before_commit", "foreign_commit"])
def test_ambiguous_cas_accepts_only_exact_original_readback(monkeypatch, tmp_path, mode):
    case = real_case(monkeypatch, tmp_path)
    case.rows.update_mode = mode
    if mode == "after_commit":
        out = case.provider.adopt(PACK, CONSUMER, policy())
        assert out["producer"] == case.producer
    else:
        with pytest.raises(ValueError):
            case.provider.adopt(PACK, CONSUMER, policy())
    assert case.rows.updates == case.rows.puts == 1 and case.copies.calls == 12
