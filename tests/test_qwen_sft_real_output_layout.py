"""Fixed SFT and reload jobs: the artifact root real SageMaker returns, and the refusals around it.

Real DescribeTrainingJob answers for completed jobs with CompressionType NONE carry
ModelArtifacts.S3ModelArtifacts = <OutputDataConfig.S3OutputPath>/<TrainingJobName>/output/model, with no
trailing separator. The simulated provider returned <S3OutputPath>/model/ with one, and the adapter accepted
only that. These tests feed the real shape through ``SageMakerQwenSftBackend.artifacts`` and pin the refusals.
"""

from __future__ import annotations

import pytest
from test_qwen_sft_backend import (
    BUCKET,
    SESSION,
    backend,
    catalog,
    install_output,
    output_prefix,
    real_description,
)

from amlas.backends.types import DatasetArtifactRef


def _prefix(*args, **kwargs):
    from amlas.backends.adapters.qwen_sft import _native_uncompressed_output_prefix

    return _native_uncompressed_output_prefix(*args, **kwargs)


# (S3OutputPath, TrainingJobName, S3ModelArtifacts) of seven completed jobs as DescribeTrainingJob returned them
# (SFT train, diagnostic train, reload, three generation jobs and one smoke job). The bucket, the session segment
# and the job names are made up; the structure and the missing trailing separator are as recorded.
_R = "s3://" + BUCKET + "/sessions/recorded-session/"
RECORDED = [
    (_R + "sft/" + job + "/output", job, _R + "sft/" + job + "/output/" + job + "/output/model")
    for job in ("example-sft-fit-batch-01", "example-sft-full-a7a7a7a-01")
] + [
    (_R + family + "/" + job + "/output", job, _R + family + "/" + job + "/output/" + job + "/output/model")
    for family, job in (
        ("diagnostics", "example-sft-records-a7a7a7a-01"),
        ("reload", "example-reload-restored-b8b8b8b-01"),
        ("generation", "example-generation-base-c9c9c9c-01"),
        ("generation", "example-generation-tuned-b8b8b8b-01"),
        ("generation", "example-gen-f6f6f6f6f6f6f6f6"),
    )
]


@pytest.mark.parametrize("output,job,artifacts", RECORDED, ids=[r[1] for r in RECORDED])
def test_recorded_real_descriptions_qualify_to_the_job_bound_key_prefix(output, job, artifacts):
    # the made-up bucket, session and job names must not have changed the recorded structure
    assert artifacts == output + "/" + job + "/output/model" and not artifacts.endswith("/")
    prefix = _prefix(artifacts, s3_output_path=output, job_name=job, bucket=BUCKET)
    assert prefix == output.split("/", 3)[3] + "/" + job + "/output/model/"
    assert prefix.endswith("/model/") and not prefix.endswith("//")
    # one trailing separator more or less is the only difference the qualification absorbs
    assert _prefix(artifacts + "/", s3_output_path=output, job_name=job, bucket=BUCKET) == prefix


@pytest.mark.parametrize("slash", [False, True], ids=["no-trailing-slash", "one-trailing-slash"])
def test_real_description_shape_flows_through_artifacts(slash):
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    real_description(n, slash=slash)
    install_output(b, n, s, h, layout="native")
    listings = []
    original_list = s.list_objects_v2
    s.list_objects_v2 = lambda **kw: listings.append(kw["Prefix"]) or original_list(**kw)
    result = b.artifacts(h)
    assert result["success"] is True and result["trained"] is True, result
    assert listings == [output_prefix(n.calls[0], "native")] and listings[0].endswith("/output/model/")
    assert result["checkpoint_ref"].uri == (
        "s3://" + BUCKET + "/" + output_prefix(n.calls[0], "native") + "CHECKPOINT.json"
    )
    assert result["quality_success"] is False and result["full_training_complete"] is False


@pytest.mark.parametrize("slash", [False, True], ids=["no-trailing-slash", "one-trailing-slash"])
def test_simulated_provider_layout_still_qualifies_with_or_without_the_separator(slash):
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    original = n.describe_training_job

    def described(**kw):
        value = original(**kw)
        uri = n.calls[0]["OutputDataConfig"]["S3OutputPath"] + "/model"
        value["ModelArtifacts"] = {"S3ModelArtifacts": uri + ("/" if slash else "")}
        return value

    n.describe_training_job = described
    install_output(b, n, s, h)
    assert b.artifacts(h)["trained"] is True


def _refusal(variant, request, job):
    output = request["OutputDataConfig"]["S3OutputPath"]
    good = output + "/" + job + "/output/model"
    other_bucket = "s3://" + "other-test-bucket" + "/" + good.split("/", 3)[3]
    return {
        "wrong-bucket": good.replace("s3://" + BUCKET + "/", "s3://other-test-bucket/", 1) + "/",
        "wrong-bucket-same-key": other_bucket + "/",
        "wrong-root": "s3://" + BUCKET + "/unrelated/" + job + "/output/model/",
        "prefix-sibling": output + "-other/" + job + "/output/model/",
        "root-only": output + "/",
        "empty": "",
        "other-job": output + "/" + job + "-other/output/model/",
        "foreign-job-name": output + "/someone-else/output/model/",
        "missing-output-directory": output + "/" + job + "/model/",
        "extra-segment": good + "/extra/",
        "extra-leading-segment": output + "/extra/" + job + "/output/model/",
        "compressed-archive": output + "/" + job + "/output/model.tar.gz",
        "compressed-archive-slash": output + "/" + job + "/output/model.tar.gz/",
        "model-prefix-sibling": output + "/" + job + "/output/model-other/",
        "dot": output + "/" + job + "/output/./model/",
        "dotdot": output + "/" + job + "/output/../model/",
        "dotdot-escape": output + "/../" + job + "/output/model/",
        "double-slash": output + "//" + job + "/output/model/",
        "double-trailing": good + "//",
        "triple-trailing": good + "///",
        "query": good + "?versionId=1/",
        "fragment": good + "#summary/",
        "backslash": good + chr(92) + "summary/",
        "https-scheme": good.replace("s3://", "https://", 1) + "/",
        "uppercase-scheme": good.replace("s3://", "S3://", 1) + "/",
        "whitespace": good + " /",
        "number": 5,
        "null": None,
        "list": [good],
    }[variant]


REFUSALS = [
    "wrong-bucket",
    "wrong-bucket-same-key",
    "wrong-root",
    "prefix-sibling",
    "root-only",
    "empty",
    "other-job",
    "foreign-job-name",
    "missing-output-directory",
    "extra-segment",
    "extra-leading-segment",
    "compressed-archive",
    "compressed-archive-slash",
    "model-prefix-sibling",
    "dot",
    "dotdot",
    "dotdot-escape",
    "double-slash",
    "double-trailing",
    "triple-trailing",
    "query",
    "fragment",
    "backslash",
    "https-scheme",
    "uppercase-scheme",
    "whitespace",
    "number",
    "null",
    "list",
]


@pytest.mark.parametrize("variant", REFUSALS)
def test_other_artifact_roots_are_refused_before_any_object_read(monkeypatch, variant):
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    install_output(b, n, s, h, layout="native")
    request = n.calls[0]
    value = _refusal(variant, request, request["TrainingJobName"])
    original = n.describe_training_job

    def described(**kw):
        observed = original(**kw)
        observed["ModelArtifacts"] = {"S3ModelArtifacts": value}
        return observed

    n.describe_training_job = described
    reads = []
    for name in ("list_objects_v2", "head_object", "get_object"):
        method = getattr(s, name)
        monkeypatch.setattr(s, name, lambda *a, _m=method, _n=name, **kw: reads.append(_n) or _m(*a, **kw))
    result = b.artifacts(h)
    assert result["success"] is False and result["trained"] is False
    assert result["status"] == "verification_failed"
    assert not reads, "an unqualified artifact root reached the object store: " + repr(reads)


def test_missing_model_artifacts_field_is_refused_before_any_object_read(monkeypatch):
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    install_output(b, n, s, h, layout="native")
    original = n.describe_training_job

    def described(**kw):
        observed = original(**kw)
        observed.pop("ModelArtifacts", None)
        return observed

    n.describe_training_job = described
    reads = []
    monkeypatch.setattr(s, "list_objects_v2", lambda **kw: reads.append(kw) or {"Contents": []})
    result = b.artifacts(h)
    assert result["trained"] is False and result["status"] == "verification_failed" and not reads


def test_a_listing_with_objects_outside_the_job_prefix_is_still_refused():
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    real_description(n, slash=True)
    install_output(b, n, s, h, layout="native")
    extra = output_prefix(n.calls[0], "native") + "unexpected.pkl"
    s.install(DatasetArtifactRef("s3://" + BUCKET + "/" + extra, "x", "0" * 64, 1, "json"), b"x")
    assert b.artifacts(h)["trained"] is False


def test_objects_at_the_simulated_prefix_do_not_satisfy_a_real_description():
    # The description names the real layout; objects exist only under the simulated one. Nothing may be read from there.
    b, n, s, i = backend()
    h = b.submit(inputs=i, catalog=catalog(), recipe_id="fit_v1", runtime_session_id=SESSION)
    real_description(n)
    install_output(b, n, s, h, layout="controlled")
    result = b.artifacts(h)
    assert result["trained"] is False and result["status"] == "verification_failed"


def _allowed(output, job):
    return output + "/" + job + "/output/model"


@pytest.mark.parametrize(
    "job",
    ["", ".", "..", "a/b", "-job", "job_name", "job name", "x" * 64, "job\u0430", "job?x", "job#x"],
    ids=lambda v: repr(v)[:14],
)
def test_helper_refuses_a_job_name_outside_the_service_grammar_even_when_the_value_matches(job):
    output = "s3://" + BUCKET + "/sessions/x/sft/" + (job or "empty") + "/output"
    with pytest.raises(ValueError, match="layout unqualified"):
        _prefix(_allowed(output, job), s3_output_path=output, job_name=job, bucket=BUCKET)


@pytest.mark.parametrize(
    "output",
    [
        "s3://" + BUCKET + "/sessions/x/sft/job-1/output/",
        "s3://" + BUCKET + "//sessions/x/sft/job-1/output",
        "s3://" + BUCKET + "/sessions/./sft/job-1/output",
        "s3://" + BUCKET + "/sessions/../sft/job-1/output",
        "s3://" + BUCKET + "/sessions//sft/job-1/output",
        "s3://" + BUCKET + "/",
        "s3://" + BUCKET,
        "s3://" + BUCKET + "-other/sessions/x/sft/job-1/output",
        "s3://other-bucket/sessions/x/sft/job-1/output",
        "https://" + BUCKET + "/sessions/x/sft/job-1/output",
        "",
    ],
    ids=lambda v: v[-24:] or "empty",
)
def test_helper_refuses_a_request_output_path_outside_the_catalog_bucket_scope(output):
    with pytest.raises(ValueError, match="layout unqualified"):
        _prefix(_allowed(output, "job-1"), s3_output_path=output, job_name="job-1", bucket=BUCKET)


@pytest.mark.parametrize(
    "bucket", ["", "a", "ab", "A" * 10, "a/b", "bucket_name", "-bucket", "bucket-", "x" * 64]
)
def test_helper_refuses_a_bucket_outside_the_service_grammar(bucket):
    output = "s3://" + bucket + "/sessions/x/sft/job-1/output"
    with pytest.raises(ValueError, match="layout unqualified"):
        _prefix(_allowed(output, "job-1"), s3_output_path=output, job_name="job-1", bucket=bucket)


def test_bucket_that_is_a_prefix_of_another_bucket_does_not_qualify():
    output = "s3://" + BUCKET + "-other/sessions/x/sft/job-1/output"
    with pytest.raises(ValueError, match="layout unqualified"):
        _prefix(_allowed(output, "job-1"), s3_output_path=output, job_name="job-1", bucket=BUCKET)


def test_simulated_layout_is_the_only_second_member_and_stays_job_bound_by_the_output_path():
    output = "s3://" + BUCKET + "/sessions/x/sft/job-1/output"
    assert _prefix(output + "/model/", s3_output_path=output, job_name="job-1", bucket=BUCKET).endswith(
        "/sft/job-1/output/model/"
    )
    for other in ("/job-1/model", "/output/model", "/job-2/output/model", "/model/model", "/model.tar.gz"):
        with pytest.raises(ValueError, match="layout unqualified"):
            _prefix(output + other, s3_output_path=output, job_name="job-1", bucket=BUCKET)


@pytest.mark.parametrize(
    "job", ["j" * 63, "a--b", "J1", "1"], ids=["longest", "hyphen-run", "mixed-case", "digit"]
)
def test_helper_accepts_the_longest_job_and_bucket_names_the_services_allow(job):
    bucket = "b" * 63
    output = "s3://" + bucket + "/sessions/x/sft/" + job + "/output"
    expected = "sessions/x/sft/" + job + "/output/" + job + "/output/model/"
    for suffix in ("", "/"):
        got = _prefix(_allowed(output, job) + suffix, s3_output_path=output, job_name=job, bucket=bucket)
        assert got == expected
