"""Finite source-only immutable artifact rejection; uncertainty never becomes a grant."""

from __future__ import annotations

_REASONS = frozenset({"artifact_hash_mismatch", "closed_manifest_invalid"})


class ImmutableArtifactRejectionError(ValueError):
    """Raised only by conclusive source-controlled immutable output guards."""

    def __init__(self, reason: str):
        if type(reason) is not str or reason not in _REASONS:
            raise ValueError("closed artifact rejection reason required")
        self.reason = reason
        super().__init__("immutable original artifact rejected")


def rejection_data(error):
    if type(error) is not ImmutableArtifactRejectionError:
        return {}
    return {"error_type": "ValueError", "verification_rejection": {"schema_version": 1, "reason": error.reason, "data_only": True}}


def is_closed_rejection(value):
    return (
        type(value) is dict
        and set(value) == {"schema_version", "reason", "data_only"}
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and type(value["reason"]) is str
        and value["reason"] in _REASONS
        and value["data_only"] is True
    )
