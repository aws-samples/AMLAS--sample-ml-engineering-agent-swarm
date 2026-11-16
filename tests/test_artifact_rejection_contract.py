"""Finite immutable rejection DATA cannot classify ordinary uncertainty."""

import pytest

from amlas.backends.adapters.artifact_rejection import (
    ImmutableArtifactRejectionError,
    is_closed_rejection,
    rejection_data,
)


@pytest.mark.parametrize("reason", ["artifact_hash_mismatch","closed_manifest_invalid"])
def test_exact_source_rejection_projects_only_fixed_data_and_legacy_error_type(reason):
    result=rejection_data(ImmutableArtifactRejectionError(reason))
    assert result=={"error_type":"ValueError","verification_rejection":{"schema_version":1,"reason":reason,"data_only":True}}
    assert is_closed_rejection(result["verification_rejection"])


@pytest.mark.parametrize("error", [ValueError("ordinary budget uncertainty"),TimeoutError("ordinary transport uncertainty"),OSError("ordinary cleanup uncertainty")])
def test_generic_errors_never_mint_closed_rejection(error):
    assert rejection_data(error)=={}


def test_subclass_never_mints_source_exact_type_projection():
    class OtherError(ImmutableArtifactRejectionError):
        pass
    assert rejection_data(OtherError("artifact_hash_mismatch"))=={}


@pytest.mark.parametrize("value", [None,{}, {"schema_version":True,"reason":"artifact_hash_mismatch","data_only":True}, {"schema_version":1,"reason":"UNKNOWN","data_only":True}, {"schema_version":1,"reason":"artifact_hash_mismatch","data_only":1}, {"schema_version":1,"reason":"artifact_hash_mismatch","data_only":True,"accepted":True}])
def test_marker_shape_remains_closed_data_only(value):
    assert not is_closed_rejection(value)
