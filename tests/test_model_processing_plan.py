"""Neutral fixed-request planning, before any native effect or durable journal."""

from copy import deepcopy

import model_acquisition_fixtures as f
import pytest

from amlas.backends.adapters.model_weight_acquisition import ProcessingModelWeightAcquisitionBackend


@pytest.fixture(autouse=True)
def namespace(monkeypatch):
    from amlas.config import externalized

    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *args, **kwargs: "")
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", f.DEPLOYMENT)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "fixture-lab")
    monkeypatch.setenv("AMLAS_AGENT_REMOTE_FETCH", "1")


def adapter():
    sm, s3 = f.Processing(), f.S3()
    return (
        f.configured_backend(
            type(
                "Module",
                (),
                {"ProcessingModelWeightAcquisitionBackend": ProcessingModelWeightAcquisitionBackend},
            ),
            sm,
            s3,
        ),
        sm,
        s3,
    )


def plan(backend, *, job_name=None):
    return backend.prepare_submission(
        catalog_id=f.CATALOG,
        catalog_version=f.VERSION,
        runtime_session_id=f.SESSION,
        job_name=job_name,
    )


def test_prepare_has_no_sdk_reads_or_create_and_fixed_source_request():
    backend, sm, s3 = adapter()
    backend._clients = lambda *args: pytest.fail("planning attempted SDK client access")
    value = plan(backend)
    assert sm.requests == [] and s3.reads == []
    assert value.extra["accepted"] is False and value.extra["submission_ambiguous"] is False
    assert value.extra["request"]["ProcessingJobName"] == value.id
    assert value.extra["request"]["AppSpecification"]["ContainerEntrypoint"] == [
        "/usr/local/bin/amlas-processing-init",
        "model-acquisition",
    ]
    assert value.extra["bindings"]["runtime_session_id"] == f.SESSION


def test_original_prepared_request_dispatch_does_not_remint_or_mutate_plan():
    backend, sm, _ = adapter()
    value = plan(backend)
    original = deepcopy(value)
    restored = plan(backend, job_name=value.id)
    assert restored == value
    submitted = backend.submit_prepared(value)
    assert value == original
    assert submitted.id == value.id and submitted.arn == value.arn
    assert submitted.extra["accepted"] is True
    assert sm.requests == [original.extra["request"]]


@pytest.mark.parametrize("field", ["role", "arguments", "output", "environment", "source", "extra"])
def test_prepared_drift_refuses_before_client_or_create(field):
    backend, sm, _ = adapter()
    value = plan(backend)
    if field == "role":
        value.extra["request"]["RoleArn"] = f.ROLE + "-other"
    elif field == "arguments":
        value.extra["request"]["AppSpecification"]["ContainerArguments"].append("ordinary-extra")
    elif field == "output":
        value.extra["request"]["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"] += "other/"
    elif field == "environment":
        value.extra["request"]["Environment"]["ORDINARY_OTHER_FIELD"] = "ordinary"
    elif field == "source":
        value.extra["bindings"]["source_commit"] = "c" * 40
    else:
        value.extra["ordinary_other_field"] = "ordinary"
    backend._clients = lambda *args: pytest.fail("drift reached SDK client access")
    with pytest.raises(ValueError):
        backend.submit_prepared(value)
    assert sm.requests == []


@pytest.mark.parametrize("name", ["ordinary-other-job", "fixture-lab-model-acq-short", None, True])
def test_fixed_original_job_name_grammar(name):
    backend, sm, _ = adapter()
    if name is None:
        assert plan(backend).id.startswith("fixture-lab-model-acq-")
    else:
        with pytest.raises(ValueError):
            plan(backend, job_name=name)
    assert sm.requests == []


def test_prepared_create_ambiguity_retains_exact_original_identity():
    backend, sm, _ = adapter()
    value = plan(backend)
    sm.ambiguous_submit = True
    submitted = backend.submit_prepared(value)
    assert submitted.id == value.id and submitted.arn == value.arn
    assert submitted.extra["submission_ambiguous"] is True
    assert submitted.extra["accepted"] is False
    assert value.extra["submission_ambiguous"] is False
    assert len(sm.requests) == 1


@pytest.mark.parametrize("when", ["bucket_read", "effect_qualifier"])
def test_prepared_current_configuration_is_rechecked_at_actual_create(monkeypatch, when):
    backend, sm, s3 = adapter()
    value = plan(backend)

    def changed_configuration():
        monkeypatch.setenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", "ordinary.example")

    if when == "bucket_read":
        read = s3.get_bucket_versioning

        def changed_read(**kwargs):
            result = read(**kwargs)
            changed_configuration()
            return result

        monkeypatch.setattr(s3, "get_bucket_versioning", changed_read)
        qualifier = None
    else:
        qualifier = changed_configuration
    with pytest.raises(ValueError):
        backend.submit_prepared(value, before_create=qualifier)
    assert sm.requests == [] and value.extra["accepted"] is False


def test_effect_qualification_refusal_is_not_an_ambiguous_native_create():
    backend, sm, _ = adapter()
    value = plan(backend)

    def refused():
        raise PermissionError("ordinary current boundary refusal")

    with pytest.raises(PermissionError):
        backend.submit_prepared(value, before_create=refused)
    assert sm.requests == [] and value.extra["submission_ambiguous"] is False
