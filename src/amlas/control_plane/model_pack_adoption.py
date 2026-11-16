"""Version-pinned model-pack adoption using existing S3 and conditional job rows."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict
from types import SimpleNamespace

from ..backends.types import DatasetArtifactRef
from .producer_snapshot import _approved_read, _keys, _producer, _ref

_MAX = 131072


def _bytes(value):
    from .normal_run_checkpoint import _checkpoint_data

    _checkpoint_data(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if not 0 < len(raw) <= _MAX:
        raise ValueError("bounded model adoption DATA required")
    return raw


def _decode(raw):
    from .normal_run_checkpoint import _pairs

    if type(raw) is not bytes or not 0 < len(raw) <= _MAX:
        raise ValueError("model adoption byte bound differs")
    value = json.loads(raw, object_pairs_hook=_pairs)
    if _bytes(value) != raw:
        raise ValueError("canonical adoption bytes required")
    return value


def _decode_original(row, scope):
    old = {**scope, "runtime_session_id": row["bindings"]["runtime_session_id"]}
    return _producer(row, old, model=True)


class ModelPackAdoptionProvider:
    def __init__(self, *, catalog_ref, scope, s3, ddb, sts, model_adapter, copy_adapter, scope_getter=None):
        if type(catalog_ref) is not DatasetArtifactRef or catalog_ref.format != "json":
            raise ValueError("typed configured pack catalog required")
        _keys(
            scope,
            {
                "account_id",
                "region",
                "bucket",
                "jobs_table",
                "execution_role_arn",
                "execution_role_id",
                "deployment_id",
                "deployment_prefix",
            },
        )
        if scope_getter is not None and not callable(scope_getter):
            raise ValueError("current adoption scope getter required")
        self._scope_getter = scope_getter
        self.catalog_ref = catalog_ref
        self.scope = dict(scope)
        self.s3, self.ddb, self.sts = s3, ddb, sts
        self.model_adapter, self.copy_adapter = model_adapter, copy_adapter
        self._table_id = None
        self._original_handles = {}
        self._handle_lock = threading.Lock()

    def _require_scope(self):
        if self._scope_getter is not None and self._scope_getter() != (self.scope, self.catalog_ref):
            raise ValueError("configured adoption scope changed")

    def _current(self, policy=None, qualify=None):
        if policy is not None:
            _approved_read(policy)
        self._require_scope()
        if qualify is not None:
            qualify()
        a = self.sts.get_caller_identity()
        s = self.scope
        role = s["execution_role_arn"].rsplit("/", 1)[1]
        if (
            a.get("Account") != s["account_id"]
            or not a.get("Arn", "").startswith(
                "arn:aws:sts::" + s["account_id"] + ":assumed-role/" + role + "/"
            )
            or a.get("UserId", "").split(":", 1)[0] != s["execution_role_id"]
        ):
            raise ValueError("current adoption role differs")
        t = self.ddb.describe_table(TableName=s["jobs_table"])["Table"]
        if (
            t.get("TableStatus") != "ACTIVE"
            or t.get("TableName") != s["jobs_table"]
            or t.get("TableArn")
            != "arn:aws:dynamodb:" + s["region"] + ":" + s["account_id"] + ":table/" + s["jobs_table"]
            or t.get("KeySchema") != [{"AttributeName": "job_name", "KeyType": "HASH"}]
            or not t.get("TableId")
        ):
            raise ValueError("current adoption table differs")
        if self._table_id is not None and self._table_id != t["TableId"]:
            raise ValueError("adoption table incarnation changed")
        self._table_id = t["TableId"]
        if qualify is not None:
            qualify()
        if policy is not None:
            _approved_read(policy)
        self._require_scope()

    def _checker(self, policy=None, qualify=None):
        """The qualifier of one provider call: calling it is the full `_current`, as every request bracket needs.

        A pack is read in 1 MiB chunks, and a full `_current` per chunk bracket (an STS read, a table description and
        the caller's qualifier, twice) made one 19.3 GB pass take 88 minutes. Between two chunks the callback runs the
        local steps (policy, configured scope) and asks the caller's own qualifier for its cheap variant; the
        remote steps (identity, table incarnation) run once per interval, counted from the start of the last full one.
        """
        from ..backends.adapters.coding_preparation import record_step
        from .normal_run_checkpoint import _record_recheck

        def check(remote):
            if remote:
                return self._current(policy, qualify)
            if policy is not None:
                _approved_read(policy)
            self._require_scope()
            if qualify is not None:
                record_step(qualify)
            if policy is not None:
                _approved_read(policy)
            self._require_scope()

        return _record_recheck(check)

    def _read(self, ref, current):
        _ref(asdict(ref))
        s = self.scope
        if (
            not ref.uri.startswith("s3://" + s["bucket"] + "/")
            or any(part in ("", ".", "..") for part in ref.uri.split("/", 3)[3].split("/"))
            or any(char in ref.uri for char in ("?", "#", "\\"))
            or ref.format != "json"
            or ref.byte_count > _MAX
        ):
            raise ValueError("configured pack metadata scope differs")
        current()
        x = self.s3.get_object(
            Bucket=s["bucket"],
            Key=ref.uri.split("/", 3)[3],
            VersionId=ref.version_id,
            ExpectedBucketOwner=s["account_id"],
        )
        body = x["Body"]
        try:
            current()
            raw = body.read(ref.byte_count + 1)
            current()
        finally:
            body.close()
        if (
            x.get("VersionId") != ref.version_id
            or x.get("ContentLength") != ref.byte_count
            or len(raw) != ref.byte_count
            or hashlib.sha256(raw).hexdigest() != ref.sha256
        ):
            raise ValueError("model pack exact metadata differs")
        return _decode(raw)

    def _catalog(self, current):
        value = self._read(self.catalog_ref, current)
        _keys(value, {"schema_version", "packs"})
        if (
            value["schema_version"] != 1
            or type(value["packs"]) is not list
            or not 1 <= len(value["packs"]) <= 8
        ):
            raise ValueError("bounded configured model packs required")
        seen = set()
        for row in value["packs"]:
            _keys(row, {"pack_ref", "catalog_id", "catalog_version", "producer", "artifacts"})
            if (
                not re.fullmatch("model_pack_[a-f0-9]{64}", row["pack_ref"])
                or row["pack_ref"] in seen
                or (row["catalog_id"], row["catalog_version"]) != ("qwen3_5_9b", "1")
            ):
                raise ValueError("configured model pin differs")
            seen.add(row["pack_ref"])
        return value["packs"]

    def candidates(self, policy=None):
        _approved_read(policy)

        def current():
            self._current(policy)

        return [
            {k: r[k] for k in ("catalog_id", "catalog_version", "pack_ref")} for r in self._catalog(current)
        ]

    def _source(self, pack, current):
        rows = [r for r in self._catalog(current) if r["pack_ref"] == pack]
        if len(rows) != 1:
            raise ValueError("unknown configured model pack")
        row = rows[0]
        producer_bytes = _bytes(row["producer"])
        current()
        with self._handle_lock:
            retained = self._original_handles.get(pack)
            if retained is None:
                handle = _decode_original(row["producer"], self.scope)
                if len(self._original_handles) >= 8:
                    raise ValueError("bounded original model handles exceeded")
                self._original_handles[pack] = (producer_bytes, handle)
            else:
                if retained[0] != producer_bytes:
                    raise ValueError("original configured model producer changed")
                handle = retained[1]
        current()
        with self.model_adapter.qualified_io(current):
            result = self.model_adapter.artifacts(handle)
        current()
        observed = {
            "success": result.get("success"),
            "acquired": result.get("acquired"),
            "file_refs": result.get("file_refs"),
            "manifest_refs": result.get("manifest_refs"),
        }
        if (
            observed != {k: row["artifacts"].get(k) for k in observed}
            or observed["success"] is not True
            or observed["acquired"] is not True
        ):
            raise ValueError("original verified model versions changed")
        return row

    def _key(self, pack, session):
        if (
            type(session) is not str
            or not re.fullmatch("[A-Za-z0-9_-]{1,128}", session)
            or type(pack) is not str
            or not re.fullmatch("model_pack_[a-f0-9]{64}", pack)
        ):
            raise ValueError("owned adoption selector differs")
        return "MODEL_ADOPTION#" + session + "#" + pack[11:]

    def _item(self, key, current):
        current()
        item = self.ddb.get_item(
            TableName=self.scope["jobs_table"], Key={"job_name": {"S": key}}, ConsistentRead=True
        ).get("Item")
        current()
        if item is None:
            return None
        _keys(item, {"job_name", "record", "record_sha256"})
        raw = item["record"]["S"].encode()
        if item["job_name"] != {"S": key} or hashlib.sha256(raw).hexdigest() != item["record_sha256"]["S"]:
            raise ValueError("adoption record hash differs")
        return item, _decode(raw)

    def _verify_staged(self, snapshot, current):
        _keys(
            snapshot,
            {
                "schema_version",
                "origin",
                "pack_ref",
                "catalog_id",
                "catalog_version",
                "runtime_session_id",
                "producer",
                "original_artifacts",
                "file_refs",
                "manifest_refs",
                "adoption_ref",
            },
        )
        if snapshot["schema_version"] != 1 or snapshot["origin"] != "verified_pack_adoption":
            raise ValueError("adoption origin differs")
        prefix = (
            "s3://"
            + self.scope["bucket"]
            + "/sessions/"
            + snapshot["runtime_session_id"]
            + "/model-adoption/"
            + snapshot["pack_ref"]
            + "/"
        )
        refs = [*snapshot["file_refs"], *snapshot["manifest_refs"].values(), snapshot["adoption_ref"]]
        for value in refs:
            ref = DatasetArtifactRef(
                **{k: value[k] for k in ("uri", "version_id", "sha256", "byte_count", "format")}
            )
            _ref(asdict(ref))
            if (
                not ref.uri.startswith(prefix)
                or any(part in ("", ".", "..") for part in ref.uri[len(prefix) :].split("/"))
                or any(char in ref.uri for char in ("?", "#", "\\"))
            ):
                raise ValueError("adoption consumer namespace differs")
            current()
            self.copy_adapter._verified(ref, SimpleNamespace(**self.scope), time.monotonic() + 3600)
            current()
        metadata = self._read(DatasetArtifactRef(**snapshot["adoption_ref"]), current)
        if metadata != {k: v for k, v in snapshot.items() if k != "adoption_ref"}:
            raise ValueError("adoption commit contents differ")

    def _scopes(self, current):
        stack = ExitStack()
        if hasattr(self.copy_adapter, "qualified_io"):
            stack.enter_context(self.copy_adapter.qualified_io(current))
        return stack

    def adopt(self, pack, session, policy, qualify=None):
        _approved_read(policy)
        key = self._key(pack, session)
        current = self._checker(policy, qualify)
        current()
        found = self._item(key, current)
        if found is not None:
            if found[1].get("status") != "COMMITTED":
                raise ValueError("prior adoption outcome unresolved; copy replay refused")
            return self.restore(found[1]["snapshot"], session, policy, qualify)
        row = self._source(pack, current)
        reservation = {
            "schema_version": 1,
            "status": "RESERVED",
            "pack_ref": pack,
            "runtime_session_id": session,
            "catalog_ref": asdict(self.catalog_ref),
            "table_id": self._table_id,
        }
        raw = _bytes(reservation)
        item = {
            "job_name": {"S": key},
            "record": {"S": raw.decode()},
            "record_sha256": {"S": hashlib.sha256(raw).hexdigest()},
        }
        current()
        try:
            self.ddb.put_item(
                TableName=self.scope["jobs_table"],
                Item=item,
                ConditionExpression="attribute_not_exists(job_name)",
            )
        except Exception:
            raise ValueError("adoption reservation unresolved; no copy authority") from None
        current()
        if self._item(key, current) != (item, reservation):
            raise ValueError("adoption reservation readback differs")
        prefix = "sessions/" + session + "/model-adoption/" + pack + "/"
        deadline = time.monotonic() + 3600
        with self._scopes(current):
            files = []
            manifests = {}
            for value in row["artifacts"]["file_refs"]:
                ref = DatasetArtifactRef(
                    **{k: value[k] for k in ("uri", "version_id", "sha256", "byte_count", "format")}
                )
                current()
                copied = self.copy_adapter._copy(
                    ref, prefix + value["path"], SimpleNamespace(**self.scope), deadline
                )
                current()
                files.append({**asdict(copied), "path": value["path"]})
            for name, value in row["artifacts"]["manifest_refs"].items():
                ref = DatasetArtifactRef(**value)
                current()
                copied = self.copy_adapter._copy(
                    ref, prefix + name + ".json", SimpleNamespace(**self.scope), deadline
                )
                current()
                manifests[name] = asdict(copied)
            snapshot = {
                "schema_version": 1,
                "origin": "verified_pack_adoption",
                "pack_ref": pack,
                "catalog_id": row["catalog_id"],
                "catalog_version": row["catalog_version"],
                "runtime_session_id": session,
                "producer": row["producer"],
                "original_artifacts": row["artifacts"],
                "file_refs": files,
                "manifest_refs": manifests,
            }
            raw = _bytes(snapshot)
            sha = hashlib.sha256(raw).hexdigest()
            current()
            written = self.s3.put_object(
                Bucket=self.scope["bucket"],
                Key=prefix + "adoption.json",
                Body=raw,
                IfNoneMatch="*",
                ExpectedBucketOwner=self.scope["account_id"],
            )
            current()
            snapshot["adoption_ref"] = asdict(
                DatasetArtifactRef(
                    "s3://" + self.scope["bucket"] + "/" + prefix + "adoption.json",
                    written["VersionId"],
                    sha,
                    len(raw),
                    "json",
                )
            )
            self._verify_staged(snapshot, current)
        final = {**reservation, "status": "COMMITTED", "snapshot": snapshot}
        after = _bytes(final)
        current()
        try:
            self.ddb.update_item(
                TableName=self.scope["jobs_table"],
                Key={"job_name": item["job_name"]},
                ConditionExpression="#record = :prior AND record_sha256 = :prior_sha",
                UpdateExpression="SET #record = :body, record_sha256 = :sha",
                ExpressionAttributeNames={"#record": "record"},
                ExpressionAttributeValues={
                    ":prior": item["record"],
                    ":prior_sha": item["record_sha256"],
                    ":body": {"S": after.decode()},
                    ":sha": {"S": hashlib.sha256(after).hexdigest()},
                },
            )
        except Exception:
            observed = self._item(key, current)
            if observed is None or observed[1] != final:
                raise ValueError("adoption publication unresolved; no replay") from None
        current()
        observed = self._item(key, current)
        if observed is None or observed[1] != final:
            raise ValueError("adoption committed record differs")
        return snapshot

    def restore(self, snapshot, session, policy, qualify=None):
        _approved_read(policy)
        current = self._checker(policy, qualify)
        if type(snapshot) is not dict or snapshot.get("runtime_session_id") != session:
            raise ValueError("foreign adoption consumer")
        key = self._key(snapshot.get("pack_ref"), session)
        found = self._item(key, current)
        if (
            found is None
            or found[1].get("status") != "COMMITTED"
            or found[1].get("snapshot") != snapshot
            or found[1].get("catalog_ref") != asdict(self.catalog_ref)
            or found[1].get("table_id") != self._table_id
        ):
            raise ValueError("owned committed adoption unavailable")
        row = self._source(snapshot["pack_ref"], current)
        if (snapshot["catalog_id"], snapshot["catalog_version"]) != (
            row["catalog_id"],
            row["catalog_version"],
        ):
            raise ValueError("adoption catalog identity changed")
        original_files = row["artifacts"]["file_refs"]
        staged_files = snapshot["file_refs"]
        if type(staged_files) is not list or len(staged_files) != len(original_files):
            raise ValueError("adoption model file inventory changed")
        by_path = {ref["path"]: ref for ref in staged_files}
        if len(by_path) != len(staged_files) or set(by_path) != {ref["path"] for ref in original_files}:
            raise ValueError("adoption model paths changed")
        for original in original_files:
            staged = by_path[original["path"]]
            if any(staged.get(key) != original[key] for key in ("sha256", "byte_count", "format")):
                raise ValueError("adoption model bytes differ from original")
        original_manifests = row["artifacts"]["manifest_refs"]
        staged_manifests = snapshot["manifest_refs"]
        if type(staged_manifests) is not dict or set(staged_manifests) != set(original_manifests):
            raise ValueError("adoption manifest inventory changed")
        for name, original in original_manifests.items():
            if any(
                staged_manifests[name].get(key) != original[key] for key in ("sha256", "byte_count", "format")
            ):
                raise ValueError("adoption provenance differs from original")
        if row["producer"] != snapshot["producer"] or row["artifacts"] != snapshot["original_artifacts"]:
            raise ValueError("adoption original attribution changed")
        with self._scopes(current):
            self._verify_staged(snapshot, current)
        current()
        return json.loads(_bytes(snapshot))
