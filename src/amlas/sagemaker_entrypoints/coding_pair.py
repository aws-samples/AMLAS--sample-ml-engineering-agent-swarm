"""Fixed AWS Processing broker entrypoint; original raw pins precede any oracle child."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import stat
from pathlib import Path

from ..arbiter.coding import CodingCasePin, CodingEvaluationBinding, CodingEvaluatorFactory
from ..backends.types import DatasetArtifactRef, DatasetInspectionSource, VerifiedPreparationSource
from ..control_plane.coding_case_state import VerifiedCaseEvaluationSource
from ..control_plane.generation_state import VerifiedCodingGenerationSource
from ..control_plane.paired_state import PairedCodingInputs
from ..data_eng.inspection_schema import decode_object
from ..evaluation.coding_pair import run_paired, validate_pair_artifacts, verify_pair_result
from ..evaluation.coding_pair_runner import BUDGETS, paired_runner_sha256, verify_oracle_launcher
from ..training.coding_generation import canonical_bytes

INPUT = Path("/opt/ml/processing/input")
OUTPUT = Path("/opt/ml/processing/output")


def _read(path, *, limit, sha=None, size=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= limit:
            raise ValueError("fixed paired input file bound differs")
        parts, total, digest = [], 0, hashlib.sha256()
        while True:
            part = os.read(fd, min(65536, limit - total + 1))
            if not part:
                break
            total += len(part)
            if total > limit:
                raise ValueError("fixed paired input byte bound exceeded")
            digest.update(part)
            parts.append(part)
        after = os.fstat(fd)
        if (
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or total != info.st_size
            or (size is not None and total != size)
            or (sha is not None and digest.hexdigest() != sha)
        ):
            raise ValueError("original paired input changed")
        return b"".join(parts)
    finally:
        os.close(fd)


def _source(value):
    origin = value["original_source"]
    inspection = origin["inspection_source"]
    inspection = DatasetInspectionSource(
        **{**inspection, "artifact_ref": DatasetArtifactRef(**inspection["artifact_ref"])}
    )
    prepared = VerifiedPreparationSource(
        **{
            **origin,
            "inspection_source": inspection,
            "manifest_refs": tuple(tuple(row) for row in origin["manifest_refs"]),
        }
    )
    return VerifiedCaseEvaluationSource(
        value["runtime_session_id"],
        value["producer_job_arn"],
        tuple(tuple(row) for row in value["producer_header"]),
        prepared,
        tuple((name, DatasetArtifactRef(**ref)) for name, ref in value["file_refs"]),
    )


def _generation(value):
    return VerifiedCodingGenerationSource(
        value["runtime_session_id"],
        value["producer_job_arn"],
        value["evaluation_id"],
        value["arm"],
        tuple((name, DatasetArtifactRef(**ref)) for name, ref in value["file_refs"]),
    )


def load_packet(input_root, *, raw_binding, source_commit, runtime_sha256, oracle_launcher_sha256):
    installed = verify_oracle_launcher(oracle_launcher_sha256)
    run = decode_object(raw_binding, limit=262144)
    wanted = {
        "schema_version",
        "operation",
        "source_commit",
        "session",
        "account_id",
        "region",
        "deployment_id",
        "deployment_prefix",
        "evaluation_binding",
        "spec_sha256",
        "spec_descriptor",
        "cases_source",
        "generation_sources",
        "original_refs",
        "staged_refs",
        "staged_spec_ref",
        "oracle_launcher_sha256",
    }
    if (
        set(run) != wanted
        or type(run["schema_version"]) is not int
        or run["schema_version"] != 1
        or run["operation"] != "paired_eval"
        or run["source_commit"] != source_commit
        or run["oracle_launcher_sha256"] != installed
        or not re.fullmatch("[a-f0-9]{40}", source_commit)
    ):
        raise ValueError("fixed closed paired RUN differs")
    binding = CodingEvaluationBinding.model_validate(run["evaluation_binding"])
    if (
        binding.runtime_sha256 != runtime_sha256
        or binding.runner_sha256 != paired_runner_sha256(oracle_launcher_sha256=installed)
        or binding.budgets_sha256 != hashlib.sha256(canonical_bytes(BUDGETS)).hexdigest()
    ):
        raise ValueError("fixed paired image/runner/budget binding differs")
    spec = run["spec_descriptor"]
    if set(spec) != {"sha256", "byte_count", "format"} or spec["format"] != "json":
        raise ValueError("original canonical paired spec descriptor differs")
    spec_bytes = _read(
        Path(input_root) / "bindings/SPEC.json", limit=65536, sha=spec["sha256"], size=spec["byte_count"]
    )
    cases = _source(run["cases_source"])
    if set(run["generation_sources"]) != {"base", "tuned"}:
        raise ValueError("closed original generation sources required")
    base, tuned = (_generation(run["generation_sources"][arm]) for arm in ("base", "tuned"))
    inputs = PairedCodingInputs(cases, base, tuned, spec_bytes, run["spec_sha256"])
    payloads = {}
    for group, values in (("cases", cases.file_refs), ("base", base.file_refs), ("tuned", tuned.file_refs)):
        expected = {name for name, _ref in values}
        folder = Path(input_root) / group
        if (
            {path.name for path in folder.iterdir()} != expected
            or set(run["staged_refs"][group]) != expected
            or set(run["original_refs"][group]) != expected
        ):
            raise ValueError("fixed original paired input inventory differs")
        for name, original_ref in values:
            staged = DatasetArtifactRef(**run["staged_refs"][group][name])
            if run["original_refs"][group][name] != dataclasses.asdict(original_ref) or (
                staged.sha256,
                staged.byte_count,
                staged.format,
            ) != (original_ref.sha256, original_ref.byte_count, original_ref.format):
                raise ValueError("original staged paired reference differs")
            payloads[group + "/" + name] = _read(
                folder / name,
                limit=12 * 1024**2
                if name == "CANDIDATES.jsonl"
                else 2 * 1024**2
                if group == "cases"
                else 262144,
                sha=staged.sha256,
                size=staged.byte_count,
            )
    observed, rows, candidates = validate_pair_artifacts(
        inputs,
        payloads,
        runner_sha256=binding.runner_sha256,
        runtime_sha256=binding.runtime_sha256,
        budgets_sha256=binding.budgets_sha256,
    )
    if observed != binding:
        raise ValueError("original frozen paired binding differs")
    return run, binding, rows, candidates


def _write(directory, name, raw):
    root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fd = os.open(
            name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=root
        )
        try:
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("paired publication write incomplete")
                view = view[count:]
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            digest, count = hashlib.sha256(), 0
            while True:
                part = os.read(fd, 65536)
                if not part:
                    break
                digest.update(part)
                count += len(part)
            if count != len(raw) or digest.hexdigest() != hashlib.sha256(raw).hexdigest():
                raise ValueError("paired publication same-FD bytes differ")
        finally:
            os.close(fd)
        os.fsync(root)
    finally:
        os.close(root)
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "format": "jsonl" if name.endswith(".jsonl") else "json",
    }


def publish_result(directory, *, binding, result, cases, source_commit):
    pins = tuple(
        CodingCasePin(case_id=case["prompt"]["case_id"], oracle_sha256=case["oracle_sha256"])
        for case in cases
    )
    report = CodingEvaluatorFactory().build_report(
        binding=binding, case_pins=pins, receipts=result["receipts"]
    )
    from ..evaluation.coding_pair import PairedTraceAccumulator

    trace = PairedTraceAccumulator(binding)
    trace_root = Path(directory) / "INVOCATIONS"
    entries = []
    for arm in ("base", "tuned"):
        for index in range(164):
            for category in ("session", "calls"):
                folder = trace_root / arm / f"{index:03d}" / category
                if not folder.exists():
                    continue
                for path in sorted(folder.iterdir()):
                    raw = _read(path, limit=65536)
                    trace.add(path.relative_to(directory).as_posix(), decode_object(raw, limit=65536))
                    entries.append(
                        canonical_bytes(
                            {
                                "path": path.relative_to(directory).as_posix(),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "byte_count": len(raw),
                            }
                        )
                        + b"\n"
                    )
    if len(entries) > 1000000:
        raise ValueError("paired trace record bound exceeded")
    trace_raw = b"".join(entries) or b"\n"
    if len(trace_raw) > 64 * 1024**2:
        raise ValueError("paired trace manifest bound exceeded")
    trace_descriptor = {
        **_write(directory, "INVOCATIONS_MANIFEST.jsonl", trace_raw),
        "record_count": len(entries),
    }
    receipts = b"".join(canonical_bytes(row.model_dump(mode="json")) + b"\n" for row in result["receipts"])
    evaluation = {
        "schema_version": 1,
        "operation": "paired_eval",
        "source_commit": source_commit,
        "binding": binding.model_dump(mode="json"),
        "case_count": 164,
        "receipt_count": 328,
        "quality_success": False,
        "native_admission": False,
        "cleanup_complete": result["cleanup_complete"],
        "summary": report.describe(),
        "invocations_manifest": trace_descriptor,
    }
    evaluation_raw = canonical_bytes(evaluation) + b"\n"
    files = {
        "RECEIPTS.jsonl": _write(directory, "RECEIPTS.jsonl", receipts),
        "EVALUATION.json": _write(directory, "EVALUATION.json", evaluation_raw),
    }
    commit = {
        "schema_version": 1,
        "operation": "paired_eval",
        "source_commit": source_commit,
        "case_count": 164,
        "quality_success": False,
        "native_admission": False,
        "files": files,
        "invocations_manifest": trace_descriptor,
    }
    commit_raw = canonical_bytes(commit) + b"\n"
    verified = verify_pair_result(
        {"RECEIPTS.jsonl": receipts, "EVALUATION.json": evaluation_raw, "COMMITTED.json": commit_raw},
        binding=binding,
        source_commit=source_commit,
        invocations=(
            trace_descriptor["sha256"],
            trace_descriptor["byte_count"],
            trace_descriptor["record_count"],
        ),
        trace=trace.finish(),
    )
    _write(directory, "COMMITTED.json", commit_raw)
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--binding-byte-count", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if (
        args.source_commit != os.environ.get("AMLAS_CODING_PAIR_SOURCE_COMMIT")
        or not re.fullmatch("[a-f0-9]{64}", args.binding_sha256)
        or not 0 < args.binding_byte_count <= 262144
    ):
        raise ValueError("fixed image paired source/binding differs")
    if os.geteuid() == 0 or os.getegid() == 0:
        raise ValueError("paired broker requires image-owned unprivileged execution")
    installed = verify_oracle_launcher(os.environ.get("AMLAS_CODING_PAIR_LAUNCHER_SHA256"))
    raw = _read(
        INPUT / "bindings/RUN.json", limit=262144, sha=args.binding_sha256, size=args.binding_byte_count
    )
    declared = decode_object(raw, limit=262144)["evaluation_binding"]
    # Image digest is read back from the original fixed native request by the consumer.
    run, binding, cases, candidates = load_packet(
        INPUT,
        raw_binding=raw,
        source_commit=args.source_commit,
        runtime_sha256=declared["runtime_sha256"],
        oracle_launcher_sha256=installed,
    )
    import boto3
    from botocore.config import Config

    options = Config(connect_timeout=2, read_timeout=10, retries={"total_max_attempts": 1})
    session = boto3.Session(region_name=run["region"])
    if session.client("sts", config=options).get_caller_identity().get("Account") != run["account_id"]:
        raise ValueError("native paired account differs")
    resource = session.client("bedrock-agentcore-control", config=options).get_code_interpreter(
        codeInterpreterId="aws.codeinterpreter.v1"
    )
    expected_interpreter_arn = (
        "arn:"
        + session.get_partition_for_region(run["region"])
        + ":bedrock-agentcore:"
        + run["region"]
        + ":aws:code-interpreter/aws.codeinterpreter.v1"
    )
    if (
        type(resource) is not dict
        or resource.get("codeInterpreterId") != "aws.codeinterpreter.v1"
        or resource.get("codeInterpreterArn") != expected_interpreter_arn
        or resource.get("status") != "READY"
        or resource.get("executionRoleArn")
        or resource.get("filesystemConfigurations")
        or resource.get("networkConfiguration", {}).get("networkMode") not in (None, "SANDBOX")
    ):
        raise ValueError("fixed original interpreter resource differs")
    if not OUTPUT.is_dir() or any(OUTPUT.iterdir()):
        raise ValueError("paired native output must start empty")
    trace = OUTPUT / "INVOCATIONS"
    trace.mkdir(mode=0o700)
    result = run_paired(
        binding=binding,
        cases=cases,
        candidates=candidates,
        client=session.client("bedrock-agentcore", config=options),
        cleanup_client=session.client("bedrock-agentcore", config=options),
        evidence_root=trace,
    )
    publish_result(OUTPUT, binding=binding, result=result, cases=cases, source_commit=args.source_commit)


if __name__ == "__main__":
    main()
