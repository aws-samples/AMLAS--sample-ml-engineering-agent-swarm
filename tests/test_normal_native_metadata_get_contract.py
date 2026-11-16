"""Cold provider implements latest metadata GET and retains exact-version data.

Fresh harmless metadata only; Source S3 SDK input-shape checks remain mandatory.
"""

from __future__ import annotations

from types import SimpleNamespace

import test_paired_coding_backend as pair_fixture
from model_acquisition_fixtures import ACCOUNT, BUCKET
from normal_sft_cold_fixture import NativeObjects
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup


def test_latest_metadata_get_returns_pinned_original_version(monkeypatch):
    monkeypatch.setattr(pair_fixture, "ACCOUNT", ACCOUNT)
    ports = SimpleNamespace(objects={})
    provider = NativeObjects(ports)
    key = "sessions/declared-source/acquisition/declared-original/ACQUIRED.json"
    raw = b'{"schema_version":1,"source":"declared"}'
    ref = provider.install(key, raw, "declared-metadata-version")
    result = provider.get_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=ACCOUNT)
    try:
        assert result["VersionId"] == ref.version_id and result["ContentLength"] == len(raw)
        assert result["Body"].read() == raw
    finally:
        result["Body"].close()
    exact = provider.get_object(Bucket=BUCKET, Key=key, VersionId=ref.version_id, ExpectedBucketOwner=ACCOUNT)
    try:
        assert exact["Body"].read() == raw and exact["VersionId"] == ref.version_id
    finally:
        exact["Body"].close()
