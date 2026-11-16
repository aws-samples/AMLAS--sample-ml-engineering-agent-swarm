"""Expected-PASS offline image contract; fixture replay never uses network or AWS."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    assert os.geteuid() == 10001 and os.getegid() == 10001, "artifact code must run as fixed non-root user"
    output = Path("/opt/ml/processing/output")
    fixture = Path("/fixture")
    metadata = json.loads((fixture / "metadata.json").read_bytes())
    payload = (fixture / "data.csv").read_bytes()
    # Pinned public fixture content digest, not a credential.
    expected = "d9e0cbb5b8850ef7efc1955f7400b0462fa2cef6dfe517b64ed6fcf7ea712c58"  # pragma: allowlist secret
    assert len(payload) == 894 and hashlib.sha256(payload).hexdigest() == expected
    assert metadata["sha"] == "87ecf163bedca9d80598b528940a9c4f99e14c11"  # pragma: allowlist secret
    from amlas.data_eng import hub_dataset as core
    from amlas.sagemaker_entrypoints import dataset_acquisition as entrypoint

    core._metadata = lambda *args, **kwargs: metadata

    def replay(url, sink, **kwargs):
        assert kwargs["expected_size"] == 894 and kwargs["max_bytes"] == core.MAX_FILE_BYTES
        sink.write(payload)
        framed = b"blob " + str(len(payload)).encode() + b"\0" + payload
        return {
            "byte_count": len(payload),
            "sha256": expected,
            "git_blob_sha1": hashlib.sha1(framed, usedforsecurity=False).hexdigest(),
        }

    core.public_fetch.stream_public_binary = replay
    result = core.acquire_huggingface_dataset_local(
        "lhoestq/demo1", "87ecf163bedca9d80598b528940a9c4f99e14c11", "data/test.csv", output_dir=str(output)
    )
    before = output.stat()
    evidence = {
        "euid": os.geteuid(),
        "gid": os.getegid(),
        "output_uid": before.st_uid,
        "output_gid": before.st_gid,
        "output_mode": oct(before.st_mode & 0o777),
        "success": result["success"],
        "error_stage": result.get("error_stage"),
        "error_type": result.get("error_type"),
        "network": "none",
        "fixture_bytes": len(payload),
    }
    print(json.dumps(evidence, sort_keys=True), flush=True)
    assert result["success"] is True, (
        "Processing fixed non-root worker must commit into its declared output path"
    )
    assert result["sha256"] == expected and Path(result["commit_manifest_path"]).is_file()
    # The native entrypoint must publish a verified root receipt, not just return success.
    entrypoint._acquire_local = lambda **kwargs: result
    entrypoint.run_acquisition(
        repo_id="lhoestq/demo1",
        revision=metadata["sha"],
        filename="data/test.csv",
        output_dir=str(output),
        job_name="fixture-contract",
        runtime_session_id="fixture-session",
        deployment_id="fixture-id",
        deployment_prefix="fixture-prefix",
        image_uri="fixture-image",
        source_commit="b" * 40,
    )
    assert (output / "ACQUIRED.json").is_file()
    print("PROCESSING-OUTPUT-CONTRACT-PASS", flush=True)


if __name__ == "__main__":
    main()
