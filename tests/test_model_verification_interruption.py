"""A run-row write inside a model byte pass is the caller's own recheck, never a refused artifact.

The real prompt runs to its accepted original SFT over the produced nine-payload model pack (`normal_sft_cold_fixture`),
then the real `NormalRunContext.load_qualified_task` hydrates the parent graph. Its model parent is verified by a fresh
adapter whose SDK calls are each bracketed by the load's own run-row recheck. The model payload HEAD and GET carry an
ETag and a CRC64NVME checksum, as S3 returns them, so the first full hash leaves the process-wide byte proof and every
later load runs a pinned pass from it (the shape of a real run). A forward run-row write (a report pointer, a write every
run still has ahead of it) lands immediately before the first run-row read after the observation line or after model
verification line N (N = 0..8). The load must then end with its recheck's own designed text and one closed
`model_verification_interrupted` line; no `model_artifact_refused` line is written and the process-wide proof stays.
Transient SDK faults and byte refusals inside the same pass stay refusals and fail closed; their refused line names the
exception class from a closed vocabulary, the closed native error code and the innermost product code site, never the
message. The original worker keeps polling in its own thread meanwhile; every observation here is of the loader's thread.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import inspect
import io
import json
import logging
import re
import sys
import threading
from pathlib import Path

from botocore.exceptions import ClientError, EndpointConnectionError
from normal_sft_cold_fixture import close, original_precommit, setup
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas.backends.adapters import model_weight_acquisition as mwa
from amlas.control_plane import normal_run_checkpoint as nrc
from amlas.control_plane import normal_run_parents

RECHECK_TEXT = "normal original TASK selection changed"
PROGRESS = re.compile(
    r"normal_model_progress run_id=(?P<run>[0-9a-f]{32}) phase=(?P<phase>[a-z_]+) state=(?P<state>[A-Z]+)"
    r" verified_payloads=(?P<count>[0-9])(?P<rest>.*)"
)
REFUSED_CAUSE = re.compile(
    r" exception_class=(?P<cls>[A-Za-z]+) native_code=(?P<code>[A-Za-z0-9]+) site=(?P<site>[a-z0-9_]+\.py:[0-9]+)"
    r" cause_class=(?P<cause>[A-Za-z]+)"
)
INTERRUPTED = re.compile(r" reason=(?P<reason>[a-z_]+) site=(?P<site>[a-z0-9_]+\.py:[0-9]+)")
AMLAS = Path(mwa.__file__).resolve().parents[2]


def _site(function, text, index=0):
    """`file.py:line` of the index-th source line of `function` that contains `text`."""
    source, start = inspect.getsourcelines(function)
    lines = [start + number for number, line in enumerate(source) if text in line]
    return Path(inspect.getsourcefile(function)).name + ":" + str(lines[index])


def _statement_has(site, text):
    """True when the product statement at `file:line` (that line or up to three above it) contains `text`."""
    name, line = site.rsplit(":", 1)
    files = list(AMLAS.rglob(name))
    assert len(files) == 1, site
    source = files[0].read_text(encoding="utf-8").splitlines()
    number = int(line) - 1
    return any(text in source[number - back] for back in range(4) if number - back >= 0)


def _progress(lines):
    return [match.groupdict() for match in map(PROGRESS.fullmatch, lines) if match is not None]


class Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append((record.thread, record.getMessage()))


def _fingerprinted(raw):
    """The ETag and the CRC64NVME-style checksum field S3 returns for a stored object (deterministic stand-ins)."""
    digest = hashlib.sha256(raw).digest()
    return {
        "ETag": '"' + digest[:16].hex() + '"',
        "ChecksumType": "FULL_OBJECT",
        "ChecksumCRC64NVME": base64.b64encode(digest[:8]).decode(),
    }


class World:
    """One real accepted original and its run row; a hook runs before each loader-thread SDK call it is named for."""

    def __init__(self, state, context, monkeypatch):
        self.state, self.context, self.ports = state, context, state.ports
        self.key = "PROMPT_RUN#" + context.run_id
        with self.ports.lock:
            self.item = copy.deepcopy(self.ports.items[self.key])
        self.handler = Lines()
        logging.getLogger(nrc.__name__).addHandler(self.handler)
        self.hook, self.thread = None, None
        self.discards, self.passes = [], []
        world = self
        actual_get, actual_identity = self.ports.get_item, self.ports.get_caller_identity

        def get_item(**kwargs):
            if kwargs["Key"]["job_name"]["S"] == world.key and world.mine():
                world.hook("row_read")
            return actual_get(**kwargs)

        def get_caller_identity(**kwargs):
            if world.mine() and world.active():
                world.hook("sts")
            return actual_identity(**kwargs)

        monkeypatch.setattr(self.ports, "get_item", get_item)
        monkeypatch.setattr(self.ports, "get_caller_identity", get_caller_identity)
        self.monkeypatch = monkeypatch
        real_artifacts = mwa.ProcessingModelWeightAcquisitionBackend._artifacts
        real_discard = mwa.ProcessingModelWeightAcquisitionBackend._discard_byte_proof

        def artifacts(adapter, handle):
            if threading.get_ident() != world.thread:
                return real_artifacts(adapter, handle)
            content = adapter._content_binding(handle)
            entry = {
                "active": True,
                "prefix": handle.extra["output_prefix"],
                "shared": mwa._PROCESS_PROOFS.get(content),
                "epoch": mwa._PROCESS_PROOFS.epoch(content),
            }
            world.passes.append(entry)
            world.patch_s3(adapter._clients(handle.extra["region"])[1])
            try:
                entry["result"] = real_artifacts(adapter, handle)
                return entry["result"]
            except BaseException as error:
                entry["raised"] = error
                raise
            finally:
                entry["active"] = False
                entry["shared_after"] = mwa._PROCESS_PROOFS.get(content)
                entry["epoch_after"] = mwa._PROCESS_PROOFS.epoch(content)

        real_public = mwa.ProcessingModelWeightAcquisitionBackend.artifacts

        def public(adapter, handle):
            try:
                return real_public(adapter, handle)
            finally:
                if threading.get_ident() == world.thread and world.passes:
                    # Balanced exit of the per-handle observation (lock released, user count back to zero).
                    world.passes[-1]["observations_after"] = len(adapter._observations)

        def discard(adapter, handle):
            if threading.get_ident() == world.thread:
                world.discards.append(id(handle))
            return real_discard(adapter, handle)

        monkeypatch.setattr(mwa.ProcessingModelWeightAcquisitionBackend, "_artifacts", artifacts)
        monkeypatch.setattr(mwa.ProcessingModelWeightAcquisitionBackend, "artifacts", public)
        monkeypatch.setattr(mwa.ProcessingModelWeightAcquisitionBackend, "_discard_byte_proof", discard)
        self._patched = set()

    def mine(self):
        return self.hook is not None and threading.get_ident() == self.thread

    def active(self):
        return bool(self.passes) and self.passes[-1]["active"]

    def patch_s3(self, s3):
        """Fingerprint the model payloads (ETag and checksum on HEAD and GET) and route loader HEADs through the hook."""
        if id(s3) in self._patched:
            return
        self._patched.add(id(s3))
        monkeypatch = self.monkeypatch
        world, actual_head, actual_get = self, s3.head_object, s3.get_object

        def model_key(kwargs):
            return any(kwargs["Key"].startswith(entry["prefix"]) for entry in world.passes)

        def raw_of(key, version):
            stored = s3.ports.objects.get((key, version)) if hasattr(s3, "ports") else None
            return stored if stored is not None else s3.objects[(key, version)]

        def head_object(**kwargs):
            change = None
            if world.mine() and world.active() and model_key(kwargs):
                change = world.hook("model_head", kwargs=kwargs)
            value = actual_head(**kwargs)
            if model_key(kwargs):
                value = {**value, **_fingerprinted(raw_of(kwargs["Key"], value["VersionId"]))}
            return change(value) if callable(change) else value

        def get_object(**kwargs):
            value = actual_get(**kwargs)
            if model_key(kwargs):
                raw = value["Body"].read()
                value["Body"].close()
                value = {**value, "ETag": _fingerprinted(raw)["ETag"], "Body": io.BytesIO(raw)}
            return value

        monkeypatch.setattr(s3, "head_object", head_object)
        monkeypatch.setattr(s3, "get_object", get_object)

    def close(self):
        logging.getLogger(nrc.__name__).removeHandler(self.handler)

    def lines(self):
        return [message for thread, message in self.handler.records if thread == self.thread]

    def reset_row(self):
        with self.ports.lock:
            self.ports.items[self.key] = copy.deepcopy(self.item)

    def forward_write(self):
        """The original worker's report-pointer write, as the product would land it (a valid, newer row)."""
        with self.ports.lock:
            item = self.ports.items[self.key]
            record = json.loads(item["record"]["S"])
            record["report_checkpoint_ref"] = {
                "uri": "s3://neutral-fixture-bucket/reports/" + "c" * 64 + ".json",
                "version_id": "neutral-report-version",
                "sha256": "c" * 64,
                "byte_count": 64,
                "format": "json",
            }
            raw = nrc._canonical(record)
            self.ports.items[self.key] = {
                **item,
                "record": {"S": raw},
                "record_sha256": {"S": hashlib.sha256(raw.encode("utf-8")).hexdigest()},
            }

    def position(self):
        """('observation', None) inside the status read, ('verification', N) after verification line N, else None."""
        lines = _progress(self.lines())
        if not lines or not self.active():
            return None
        last = lines[-1]
        if last["phase"] == "model_native_observation":
            return ("observation", None)
        if last["phase"] == "model_artifact_verification":
            return ("verification", int(last["count"]))
        return None

    def load(self, hook=None):
        self.reset_row()
        self.handler.records.clear()
        self.discards.clear()
        self.passes.clear()
        self.thread, self.hook = threading.get_ident(), hook or (lambda *args, **kwargs: None)
        try:
            with nrc.activate_normal_run(self.context):
                self.context.load_qualified_task(policy=self.state.backend.approval_policy)
        except Exception as error:
            return error
        finally:
            self.hook = None
        return None


def _world(monkeypatch, tmp_path, body):
    async def scenario():
        state = setup(monkeypatch, tmp_path)
        actual_publish = normal_run_parents.publish_parent_graph
        contexts = []

        def publish(store, model_ref, preparation_ref, *, run_id, qualify):
            raw = actual_publish(store, model_ref, preparation_ref, run_id=run_id, qualify=qualify)
            contexts.append(nrc.current_normal_run())
            return raw

        monkeypatch.setattr(normal_run_parents, "publish_parent_graph", publish)
        try:
            await original_precommit(state, monkeypatch)
            world = World(state, contexts[0], monkeypatch)
            try:
                assert world.load() is None, "baseline load refused"
                assert world.load() is None, "warm load refused"
                assert len(world.passes) == 1 and world.passes[0]["shared"] is not None, (
                    "a warm load must run its model pass from the process-wide byte proof"
                )
                body(world)
            finally:
                world.close()
        finally:
            await close(state)

    asyncio.run(scenario())


def _report(name, value):
    print(name + " " + json.dumps(value, sort_keys=True, default=str), file=sys.__stderr__, flush=True)


TARGETS = [("observation", None)] + [("verification", count) for count in range(9)]


def test_run_row_write_at_every_verification_position_is_typed_never_refused(monkeypatch, tmp_path):
    row_recheck = _site(nrc.NormalRunContext.load_qualified_task, RECHECK_TEXT, 0)
    failures = []

    def body(world):
        for target in TARGETS:
            assert world.load() is None, "re-warm load refused"
            fired = []

            def hook(kind, *, target=target, fired=fired, **_):
                if kind == "row_read" and not fired and world.position() == target:
                    fired.append(True)
                    world.forward_write()

            error = world.load(hook)
            lines = _progress(world.lines())
            refused = [line for line in lines if line["phase"] == "model_artifact_refused"]
            typed = [line for line in lines if line["phase"] == "model_verification_interrupted"]
            raised = [entry for entry in world.passes if "raised" in entry]
            row = {
                "target": target,
                "fired": bool(fired),
                "refused": [(line["count"], line["rest"]) for line in refused],
                "typed": [(line["state"], line["count"], line["rest"]) for line in typed],
                "error": None if error is None else (type(error).__name__, list(error.args)),
                "discards": len(world.discards),
                "passes": len(world.passes),
                "returned": [entry.get("result", {}).get("status") for entry in world.passes],
                "shared_kept": [
                    entry["shared"] is not None and entry["shared_after"] is entry["shared"] for entry in raised
                ],
                "epoch_kept": [entry["epoch_after"] == entry["epoch"] for entry in raised],
                "observations_left": [entry.get("observations_after") for entry in raised],
            }
            _report("POSITION", row)
            match = INTERRUPTED.fullmatch(typed[0]["rest"]) if len(typed) == 1 else None
            expected = ("UNKNOWN", "0") if target[1] is None else ("SUCCEEDED", str(target[1]))
            ok = (
                fired
                and refused == []
                and match is not None
                and match["reason"] == "run_record_changed"
                and match["site"] == row_recheck
                and (typed[0]["state"], typed[0]["count"]) == expected
                and type(error) is ValueError
                and error.args == (RECHECK_TEXT,)
                and world.discards == []
                and len(world.passes) == 1
                and row["shared_kept"] == [True]
                and row["epoch_kept"] == [True]
                and row["observations_left"] == [0]
            )
            if not ok:
                failures.append(row)

    _world(monkeypatch, tmp_path, body)
    assert failures == [], f"positions where a run-row write became a refusal: {json.dumps(failures[:3])}"


def _client_error(code, status, operation):
    return ClientError(
        {
            "Error": {"Code": code, "Message": "neutral fixture fault"},
            "ResponseMetadata": {"HTTPStatusCode": status, "RequestId": "0" * 8 + "-0000-0000-0000-" + "0" * 12},
        },
        operation,
    )


def _faults():
    sdk_call = _site(mwa.ProcessingModelWeightAcquisitionBackend._sdk_io, "result = method(**kwargs)")
    payload = mwa.ProcessingModelWeightAcquisitionBackend._payload_observed
    return {
        "s3_503": (
            "model_head",
            lambda: _client_error("SlowDown", 503, "HeadObject"),
            ("ClientError", "SlowDown", sdk_call),
        ),
        "s3_connection_error": (
            "model_head",
            lambda: EndpointConnectionError(endpoint_url="https://s3.neutral.invalid"),
            ("EndpointConnectionError", "none", sdk_call),
        ),
        "sts_throttle": (
            "sts",
            lambda: _client_error("Throttling", 400, "GetCallerIdentity"),
            ("ClientError", "Throttling", "get_caller_identity("),
        ),
        "dynamodb_throttle": (
            "row_read",
            lambda: _client_error("ProvisionedThroughputExceededException", 400, "GetItem"),
            ("ClientError", "ProvisionedThroughputExceededException", "get_item("),
        ),
        "version_changed": (
            "model_head_version",
            None,
            ("ValueError", "none", _site(payload, '"model original verified version changed"')),
        ),
        "size_differs": (
            "model_head_size",
            None,
            ("ValueError", "none", _site(payload, '"model exact object size differs"')),
        ),
    }


def test_transient_and_byte_faults_inside_the_pass_stay_refusals_and_name_their_cause(monkeypatch, tmp_path):
    failures = []

    def body(world):
        for name, (kind, make, (cls, code, site)) in _faults().items():
            assert world.load() is None, "re-warm load refused"
            fired = []

            def hook(seen, *, kwargs=None, kind=kind, make=make, fired=fired):
                if fired or world.position() != ("verification", 2):
                    return None
                if kind == seen:
                    fired.append(True)
                    raise make()
                if seen == "model_head" and kind.startswith("model_head_") and "VersionId" not in kwargs:
                    fired.append(True)
                    if kind == "model_head_version":
                        return lambda value: {**value, "VersionId": "neutral-replaced-version"}
                    return lambda value: {**value, "ContentLength": value["ContentLength"] + 1}
                return None

            error = world.load(hook)
            lines = _progress(world.lines())
            refused = [line for line in lines if line["phase"] == "model_artifact_refused"]
            typed = [line for line in lines if line["phase"] == "model_verification_interrupted"]
            failed = [entry for entry in world.passes if entry.get("result", {}).get("status") == "verification_failed"]
            row = {
                "fault": name,
                "fired": bool(fired),
                "refused": [(line["count"], line["rest"]) for line in refused],
                "typed": [line["rest"] for line in typed],
                "error": None if error is None else type(error).__name__,
                "discards": len(world.discards),
                "returned": [entry["result"].get("error_type") for entry in failed],
            }
            _report("FAULT", row)
            match = REFUSED_CAUSE.fullmatch(refused[0]["rest"]) if len(refused) == 1 else None
            ok = (
                fired
                and match is not None
                and typed == []
                and refused[0]["count"] == "2"
                and match["cls"] == cls
                and match["code"] == code
                and match["cause"] == "none"
                and (match["site"] == site if site[-1].isdigit() else _statement_has(match["site"], site))
                and "neutral fixture fault" not in json.dumps(world.handler.records)
                and error is not None
                and world.discards != []
                and [entry["result"]["error_type"] for entry in failed] == [cls]
            )
            if not ok:
                failures.append(row)

    _world(monkeypatch, tmp_path, body)
    assert failures == [], f"faults whose refusal lost its cause or its fail-closed outcome: {json.dumps(failures)}"
