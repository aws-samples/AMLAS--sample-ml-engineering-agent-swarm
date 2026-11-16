"""One native S3 namespace includes immutable publisher and staged objects.

Only fresh declared source bytes; no model loading or AWS calls.
"""

from types import SimpleNamespace

import pytest
import test_paired_coding_backend as pair_fixture
from model_acquisition_fixtures import ACCOUNT, BUCKET
from normal_sft_cold_fixture import NativeObjects


def provider(monkeypatch):
    monkeypatch.setattr(pair_fixture, "ACCOUNT", ACCOUNT)
    prefix = "sessions/declared-model/acquisition/declared-original/files/"
    source = {
        (prefix + "weight-" + str(index) + ".safetensors", "declared-version"): b"declared-inert-bytes"
        for index in range(9)
    }
    return NativeObjects(SimpleNamespace(objects=source)), prefix, source


def test_shared_publisher_model_inventory_is_listed_once(monkeypatch):
    objects, prefix, source = provider(monkeypatch)
    result = objects.list_objects_v2(Bucket=BUCKET, Prefix=prefix, ExpectedBucketOwner=ACCOUNT, MaxKeys=10)
    assert result["IsTruncated"] is False
    assert {item["Key"] for item in result["Contents"]} == {key for key, _ in source}
    assert len(result["Contents"]) == 9


@pytest.mark.parametrize("exact", [False, True])
def test_shared_publisher_head_retains_pinned_metadata(monkeypatch, exact):
    objects, prefix, source = provider(monkeypatch)
    key = prefix + "weight-0.safetensors"
    arguments = {"Bucket": BUCKET, "Key": key, "ExpectedBucketOwner": ACCOUNT}
    if exact:
        arguments["VersionId"] = "declared-version"
    result = objects.head_object(**arguments)
    assert result == {
        "VersionId": "declared-version",
        "ContentLength": len(source[(key, "declared-version")]),
    }
