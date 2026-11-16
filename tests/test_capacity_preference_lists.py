"""Pinned instance preference lists for fine-tune, reload and generation, and the quota of one job per type.

The account allows one training job per ml.g6e type, and SageMaker refuses a create when any listed type is at
that limit. Fine-tune (full and fit) lists ml.g6e.16xlarge, 12xlarge, 8xlarge and 4xlarge in that order; the new
reload contract lists ml.g6e.12xlarge alone, sent as a one-entry list; base and tuned generation list
ml.g6e.16xlarge, 4xlarge and 8xlarge in that order (changed 2026-11-11; before, the reload listed 16xlarge and
12xlarge and generation 4xlarge and 8xlarge). The
reload and generation lists share no type, so the two can be submitted in either order. Historical requests keep
their exact fixed type. Every observation of a preference-list job names SelectedInstanceType and
SelectedInstanceCount. A generation capacity rejection names the one listed type SageMaker refused for and who
holds it. Strict installed-SDK fakes, no AWS call.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import threading

import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters
from test_generation_capacity_rejection_retry import _real_adapter
from test_managed_generation_product import (  # noqa: F401  (pytest fixture)
    CASE_REF,
    MODEL_REF,
    SFT_REF,
    state,
)
from test_qwen_sft_backend import ACCOUNT, MODELS, SESSION, catalog, ref
from test_qwen_sft_durable_submission import durable, submit
from test_qwen_sft_reload_resources import contract_intent, ordinary_reload, parts

from amlas.backends.types import DatasetArtifactRef, JobHandle, JobState, JobStatus
from amlas.control_plane.campaign_state import CampaignStateError, _json, validate_submission_snapshot
from amlas.control_plane.generation_state import fixed_generation_request
from amlas.control_plane.sft_state import (
    GENERATION_INSTANCE_PREFERENCES,
    PINNED_INSTANCE_PREFERENCES,
    RELOAD_INSTANCE_PREFERENCES,
    RELOAD_PREFERENCE_CONTRACT,
    RELOAD_RESOURCE_CONTRACT,
    RELOAD_RESOURCE_CONTRACTS,
    TRAIN_INSTANCE_PREFERENCES,
    fixed_resource_config,
    fixed_sft_request,
)

TRAIN = ("ml.g6e.16xlarge", "ml.g6e.12xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge")
RELOAD = ("ml.g6e.12xlarge",)
GENERATION = ("ml.g6e.16xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge")
# The reload and generation lists of 2026-11-10, replaced on 2026-11-11 and refused now.
EARLIER_RELOAD = ("ml.g6e.16xlarge", "ml.g6e.12xlarge")
EARLIER_GENERATION = ("ml.g6e.4xlarge", "ml.g6e.8xlarge")
# Lists no request may carry: reordered, extended, shortened and prototype lists, the generation list of the
# first port (ml.g6e.4xlarge, 8xlarge, 16xlarge), which shared ml.g6e.16xlarge with the reload, and the earlier
# reload and generation lists. Every list this file refused before 2026-11-11 is still here by its value (the
# earlier reload list plus 8xlarge equals TRAIN[:3]); the last five entries are new.
OTHER_LISTS = [
    tuple(reversed(TRAIN)),
    ("ml.g6e.12xlarge", "ml.g6e.16xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge"),
    (*TRAIN, "ml.g6e.2xlarge"),
    TRAIN[:3],
    ("ml.g6e.12xlarge", "ml.g6e.16xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge", "ml.g6e.2xlarge"),
    ("ml.g6e.4xlarge", "ml.g6e.8xlarge", "ml.g6e.16xlarge", "ml.g6e.2xlarge"),
    ("ml.g6e.4xlarge", "ml.g6e.8xlarge", "ml.g6e.16xlarge"),
    (*EARLIER_GENERATION, "ml.g6e.12xlarge"),
    ("ml.g6e.8xlarge", "ml.g6e.4xlarge", "ml.g6e.16xlarge"),
    tuple(reversed(EARLIER_RELOAD)),
    tuple(reversed(EARLIER_GENERATION)),
    EARLIER_RELOAD[:1],
    EARLIER_GENERATION[:1],
    EARLIER_RELOAD,
    EARLIER_GENERATION,
    (*GENERATION, "ml.g6e.12xlarge"),
    (*RELOAD, "ml.g6e.8xlarge"),
    GENERATION[:2],
    ("ml.g6e.16xlarge", "ml.g6e.8xlarge", "ml.g6e.4xlarge"),
]


def _listed(request):
    return tuple(item["InstanceType"] for item in request["ResourceConfig"]["InstancePreferences"])


def _run_ref(family, job):
    c = catalog()
    return DatasetArtifactRef(
        "s3://" + c.bucket + "/sessions/" + SESSION + "/" + family + "/" + job + "/input/bindings/RUN.json",
        "version-1",
        "a" * 64,
        10,
        "json",
    )


# --- list constants and order -----------------------------------------------------------------------------------


def test_pinned_lists_have_exactly_the_chosen_members_in_order():
    assert TRAIN_INSTANCE_PREFERENCES == TRAIN
    assert RELOAD_INSTANCE_PREFERENCES == RELOAD
    assert GENERATION_INSTANCE_PREFERENCES == GENERATION
    assert PINNED_INSTANCE_PREFERENCES == (TRAIN, RELOAD, GENERATION)
    assert all("ml.g6e.2xlarge" not in listed for listed in PINNED_INSTANCE_PREFERENCES)
    assert "ml.g6e.12xlarge" not in GENERATION_INSTANCE_PREFERENCES
    # 2026-11-11: the reload names ml.g6e.12xlarge alone; generation tries 16xlarge first, then 4xlarge, 8xlarge.
    assert RELOAD_INSTANCE_PREFERENCES == ("ml.g6e.12xlarge",)
    assert GENERATION_INSTANCE_PREFERENCES == ("ml.g6e.16xlarge", "ml.g6e.4xlarge", "ml.g6e.8xlarge")
    assert (
        EARLIER_RELOAD not in PINNED_INSTANCE_PREFERENCES
        and EARLIER_GENERATION not in PINNED_INSTANCE_PREFERENCES
    )


def test_reload_and_generation_lists_share_no_type():
    """A rejected reload is never submitted again, so no generation of the run may hold one of its types."""
    assert not set(RELOAD_INSTANCE_PREFERENCES) & set(GENERATION_INSTANCE_PREFERENCES)
    assert set(RELOAD_INSTANCE_PREFERENCES) | set(GENERATION_INSTANCE_PREFERENCES) == set(
        TRAIN_INSTANCE_PREFERENCES
    )


@pytest.mark.parametrize("listed", [TRAIN, RELOAD, GENERATION])
def test_each_pinned_list_builds_one_instance_with_100_gib(listed):
    config = fixed_resource_config(listed)
    assert config == {
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
        "InstancePreferences": [{"InstanceType": name} for name in listed],
    }
    validate_parameters(config, MODELS["sagemaker"].shape_for("ResourceConfig"))


@pytest.mark.parametrize("listed", OTHER_LISTS)
def test_a_list_with_other_members_or_order_is_refused(listed):
    with pytest.raises(ValueError, match="preference list differs"):
        fixed_resource_config(listed)


# --- new contract and new snapshot shapes -------------------------------------------------------------------------


def test_new_reload_contract_lists_the_reload_preferences():
    assert RELOAD_PREFERENCE_CONTRACT == "reload_instance_preferences_v1"
    assert RELOAD_RESOURCE_CONTRACTS == ("reload_single_gpu_v1", "reload_instance_preferences_v1")
    c = catalog()
    job = c.deployment_prefix + "-sft-lists"
    request = fixed_sft_request(
        c,
        SESSION,
        job,
        "reload",
        "full_v1",
        _run_ref("sft", job),
        resource_contract=RELOAD_PREFERENCE_CONTRACT,
    )
    assert _listed(request) == RELOAD and "InstanceType" not in request["ResourceConfig"]


ONE_ENTRY_RELOAD = {
    "InstanceCount": 1,
    "VolumeSizeInGB": 100,
    "InstancePreferences": [{"InstanceType": "ml.g6e.12xlarge"}],
}


def test_one_type_reload_is_sent_as_a_one_entry_preference_list():
    """The reload names one type and keeps the list shape (2026-11-11).

    The service model of the locked SDK and the API reference allow one to five InstancePreferences entries, so
    the request is valid as a list, and the reload's status and artifacts keep naming the selected instance
    through the path that fine-tune and generation use. The fixed-type shape stays with the historical records
    that carry it. No real one-entry create is on record yet: the account's list jobs carried two to five types.
    """
    bounds = MODELS["sagemaker"].shape_for("InstancePreferenceList").metadata
    assert (bounds["min"], bounds["max"]) == (1, 5)
    assert fixed_resource_config(RELOAD_INSTANCE_PREFERENCES) == ONE_ENTRY_RELOAD
    c = catalog()
    job = c.deployment_prefix + "-sft-lists"
    request = fixed_sft_request(
        c,
        SESSION,
        job,
        "reload",
        "full_v1",
        _run_ref("sft", job),
        resource_contract=RELOAD_PREFERENCE_CONTRACT,
    )
    assert request["ResourceConfig"] == ONE_ENTRY_RELOAD
    validate_parameters(request, MODELS["sagemaker"].operation_model("CreateTrainingJob").input_shape)


def test_one_entry_reload_is_created_recorded_restored_and_read_back():
    """The ordinary reload sends the one-entry list, its record restores, and status() and artifacts() name ml.g6e.12xlarge."""
    p, checkpoint, selected, reader = parts()
    h = ordinary_reload(p, checkpoint, selected, reader)
    assert h.extra["submission_state"] == "accepted"
    assert p[1].calls[-1]["ResourceConfig"] == ONE_ENTRY_RELOAD
    r = p[4].load_original(p[5], h.id)
    assert r.intent.reload_resource_contract == RELOAD_PREFERENCE_CONTRACT and r.disposition == "ACCEPTED"
    assert json.loads(r.submission_snapshot.request_json)["ResourceConfig"] == ONE_ENTRY_RELOAD
    creates = len(p[1].calls)
    restored = p[0].restore_original(r, fresh_inputs=p[3])
    assert restored.id == h.id and len(p[1].calls) == creates
    assert restored.extra["request"]["ResourceConfig"] == ONE_ENTRY_RELOAD
    # The reload job is still training, as in the SFT status test below (the fixture's jobs read Completed).
    p[1].state = "InProgress"
    for handle in (h, restored):
        observed = p[0].status(handle)
        assert observed.state is JobState.RUNNING
        assert observed.detail == "SelectedInstanceType=ml.g6e.12xlarge SelectedInstanceCount=1"
        assert observed.raw["ResourceConfig"]["SelectedInstanceType"] == "ml.g6e.12xlarge"
        result = p[0].artifacts(handle)
        assert result["status"] == "RUNNING" and result["operation"] == "reload"
        assert (result["SelectedInstanceType"], result["SelectedInstanceCount"]) == ("ml.g6e.12xlarge", 1)
    assert len(p[1].calls) == creates


@pytest.mark.parametrize("recipe", ["fit_v1", "full_v1"])
def test_new_training_snapshot_lists_the_training_preferences(recipe):
    c = catalog()
    job = c.deployment_prefix + "-sft-lists"
    request = fixed_sft_request(c, SESSION, job, "train", recipe, _run_ref("sft", job))
    assert _listed(request) == TRAIN and "InstanceType" not in request["ResourceConfig"]


@pytest.mark.parametrize("arm", ["base", "tuned"])
def test_generation_requests_list_the_generation_preferences(arm):
    c = catalog()
    job = c.deployment_prefix + "-gen-lists"
    request = fixed_generation_request(c, SESSION, job, _run_ref("generation", job), arm)
    assert _listed(request) == GENERATION and "InstanceType" not in request["ResourceConfig"]


@pytest.mark.parametrize("contract", ["reload_single_gpu_preferences_v1", "reload_preferences_v1"])
def test_other_reload_contract_names_are_refused(contract):
    c = catalog()
    job = c.deployment_prefix + "-sft-lists"
    with pytest.raises(ValueError, match="resource contract differs"):
        fixed_sft_request(
            c, SESSION, job, "reload", "full_v1", _run_ref("sft", job), resource_contract=contract
        )
    p, checkpoint, _selected, _reader = parts()
    before = len(p[6].calls), len(p[1].calls), list(p[2].mutations)
    with pytest.raises(CampaignStateError):
        contract_intent(p, checkpoint, contract=contract)
    assert (len(p[6].calls), len(p[1].calls), p[2].mutations) == before


# --- historical acceptance unchanged ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation,contract,historical,instance_type",
    [
        ("train", None, True, "ml.g6e.12xlarge"),
        ("reload", None, False, "ml.g6e.12xlarge"),
        ("reload", RELOAD_RESOURCE_CONTRACT, False, "ml.g6e.4xlarge"),
    ],
)
def test_historical_contracts_keep_their_exact_fixed_request(operation, contract, historical, instance_type):
    c = catalog()
    job = c.deployment_prefix + "-sft-lists"
    request = fixed_sft_request(
        c,
        SESSION,
        job,
        operation,
        "full_v1",
        _run_ref("sft", job),
        resource_contract=contract,
        historical=historical,
    )
    assert request["ResourceConfig"] == {
        "InstanceType": instance_type,
        "InstanceCount": 1,
        "VolumeSizeInGB": 100,
    }


def test_historical_fixed_training_snapshot_and_new_snapshot_both_validate():
    p = durable()
    h = submit(p)
    r = p[4].load_original(p[5], h.id)
    validate_submission_snapshot(r)
    assert _listed(json.loads(r.submission_snapshot.request_json)) == TRAIN
    request = json.loads(r.submission_snapshot.request_json)
    request["ResourceConfig"] = {"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}
    validate_submission_snapshot(
        dataclasses.replace(
            r,
            submission_snapshot=dataclasses.replace(
                r.submission_snapshot, request_json=_json(request).decode()
            ),
        )
    )


# --- refusal of a reordered or extended list ------------------------------------------------------------------


@pytest.mark.parametrize("listed", [*OTHER_LISTS, RELOAD, GENERATION])
def test_training_snapshot_with_another_list_is_refused(listed):
    p = durable()
    h = submit(p)
    r = p[4].load_original(p[5], h.id)
    request = json.loads(r.submission_snapshot.request_json)
    request["ResourceConfig"]["InstancePreferences"] = [{"InstanceType": name} for name in listed]
    changed = dataclasses.replace(
        r,
        submission_snapshot=dataclasses.replace(r.submission_snapshot, request_json=_json(request).decode()),
    )
    with pytest.raises(CampaignStateError, match="original fixed submission request differs"):
        validate_submission_snapshot(changed)


@pytest.mark.parametrize("listed", [*OTHER_LISTS, TRAIN, GENERATION])
def test_reload_snapshot_with_another_list_cannot_restore(listed):
    p, checkpoint, selected, reader = parts()
    h = ordinary_reload(p, checkpoint, selected, reader)
    r = p[4].load_original(p[5], h.id)
    assert r.intent.reload_resource_contract == RELOAD_PREFERENCE_CONTRACT
    request = json.loads(r.submission_snapshot.request_json)
    assert _listed(request) == RELOAD
    request["ResourceConfig"]["InstancePreferences"] = [{"InstanceType": name} for name in listed]
    changed = dataclasses.replace(
        r,
        submission_snapshot=dataclasses.replace(r.submission_snapshot, request_json=_json(request).decode()),
    )
    before = p[1].reads, len(p[1].calls)
    with pytest.raises(CampaignStateError):
        p[0].restore_original(changed, fresh_inputs=p[3])
    assert (p[1].reads, len(p[1].calls)) == before


# --- status details and observation records name the selected instance ----------------------------------------------


@pytest.mark.parametrize("selected", TRAIN)
def test_sft_status_and_observation_name_the_selected_instance(selected, monkeypatch):
    p = durable()
    h = submit(p)
    native = p[1]
    describe = native.describe_training_job

    def described(**kw):
        value = describe(**kw)
        value["ResourceConfig"]["SelectedInstanceType"] = selected
        return value

    monkeypatch.setattr(native, "describe_training_job", described)
    observed = p[0].status(h)
    assert observed.state is JobState.RUNNING
    assert observed.detail == "SelectedInstanceType=" + selected + " SelectedInstanceCount=1"
    assert observed.raw["ResourceConfig"]["SelectedInstanceType"] == selected
    result = p[0].artifacts(h)
    assert result["status"] == "RUNNING"
    assert (result["SelectedInstanceType"], result["SelectedInstanceCount"]) == (selected, 1)


def test_fixed_type_status_names_no_selection():
    p = durable()
    h = submit(p)
    fixed = {"InstanceType": "ml.g6e.12xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}
    p[1].calls[0]["ResourceConfig"] = copy.deepcopy(fixed)
    h.extra["request"] = {**h.extra["request"], "ResourceConfig": copy.deepcopy(fixed)}
    observed = p[0].status(h)
    assert observed.state is JobState.RUNNING and observed.detail == ""
    assert not {"SelectedInstanceType", "SelectedInstanceCount"} & set(p[0].artifacts(h))


def test_unknown_observation_names_no_selection(monkeypatch):
    p = durable()
    h = submit(p)
    monkeypatch.setattr(p[1], "describe_training_job", lambda **kw: {})
    observed = p[0].status(h)
    assert observed.state is JobState.UNKNOWN
    assert not {"SelectedInstanceType", "SelectedInstanceCount"} & set(p[0].artifacts(h))


# --- one training job per type in the account -------------------------------------------------------------------


def _quota(name):
    return ClientError(
        {
            "Error": {
                "Code": "ResourceLimitExceeded",
                "Message": "The account-level service limit '"
                + name
                + " for training job usage' is 1 Instances, with current utilization of 1 Instances"
                + " and a request delta of 1 Instances.",
            }
        },
        "CreateTrainingJob",
    )


def _wanted(request):
    """(requested types, fixed) of a create request: its one fixed type, or its listed types in order."""
    resource = request["ResourceConfig"]
    if "InstanceType" in resource:
        return [resource["InstanceType"]], True
    return [item["InstanceType"] for item in resource["InstancePreferences"]], False


class Account:
    """The quota of one training job per ml.g6e type, as SageMaker applied it on 2026-11-10.

    Observed (CloudTrail CreateTrainingJob and DescribeTrainingJob, a test account in us-west-2, 2026-11-10):
    - A create is refused with ResourceLimitExceeded when ANY requested type is at the account limit, and the
      message names that type, even when other listed types are free. A list of 12xlarge, 16xlarge and
      2xlarge was refused naming 2xlarge (a fixed 2xlarge job was Pending) while 12xlarge and 16xlarge were free;
      the same request was accepted 23 s later, 2 s after that job stopped. Earlier a list starting 12xlarge,
      8xlarge, 4xlarge was refused naming 8xlarge, the first listed type at its limit.
    - A fixed-type job holds its type from the create on, also while it is Pending.
    - A list job that has not selected a type holds nothing: a fixed 4xlarge create and a list create
      were accepted while a list job that listed their types waited without a selection.
    Not measured: a create that lists a type a list job has already selected. This stand-in counts the selected
    type as held, like a fixed type; the pinned lists are chosen so that a reload and a generation never depend on
    that reading (they share no type).

    `outside` types are held by jobs of the account that do not belong to the run under test.
    """

    def __init__(self, outside=()):
        self.held = {name: "outside-account-job" for name in outside}
        self.waiting = {}

    def admit(self, request, *, select=True):
        """Refuse naming the first requested type at its limit; otherwise register the job. Returns its type or None."""
        wanted, fixed = _wanted(request)
        blocked = [name for name in wanted if name in self.held]
        if blocked:
            raise _quota(blocked[0])
        job = request["TrainingJobName"]
        if fixed or select:
            self.held[wanted[0]] = job
            return wanted[0]
        self.waiting[job] = wanted
        return None

    def select(self, job):
        """SageMaker gives a waiting list job the first of its listed types that no job holds; None when none is free."""
        free = [name for name in self.waiting[job] if name not in self.held]
        if not free:
            return None
        del self.waiting[job]
        self.held[free[0]] = job
        return free[0]

    def end(self, job):
        self.held = {name: holder for name, holder in self.held.items() if holder != job}
        self.waiting.pop(job, None)


def _in_account(native, account):
    """Put a durable SFT stand-in into the account: its creates meet the same quota rule. Returns the attempts.

    The stand-in describes the first listed type as selected, which is the type `Account.admit` registers.
    """
    create, attempts = native.create_training_job, []

    def guarded(**kw):
        attempts.append(kw["TrainingJobName"])
        account.admit(kw)
        return create(**kw)

    native.create_training_job = guarded
    return attempts


class QuotaNative:
    """SageMaker stand-in for generation jobs inside one `Account`.

    A list job selects its first listed type at the create (every listed type is free, or the create is refused);
    with `select=False` it stays Pending without a selection until `select(name)`. Descriptions add
    SelectedInstanceType and SelectedInstanceCount once a type is selected, as real preference-list descriptions do.
    """

    def __init__(self, account=None, *, outside=(), select=True):
        from types import SimpleNamespace

        self.meta = SimpleNamespace(region_name="us-west-2")
        self.account = account if account is not None else Account(outside)
        self.select_on_create = select
        self.calls, self.jobs, self.states, self.selected = [], {}, {}, {}

    def create_training_job(self, **kw):
        validate_parameters(kw, MODELS["sagemaker"].operation_model("CreateTrainingJob").input_shape)
        self.calls.append(copy.deepcopy(kw))
        selected = self.account.admit(kw, select=self.select_on_create)
        name = kw["TrainingJobName"]
        self.jobs[name], self.states[name], self.selected[name] = copy.deepcopy(kw), "InProgress", selected
        return {"TrainingJobArn": "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/" + name}

    def select(self, name):
        self.selected[name] = self.account.select(name)
        return self.selected[name]

    def describe_training_job(self, **kw):
        name = kw["TrainingJobName"]
        request, selected = self.jobs[name], self.selected[name]
        resource = copy.deepcopy(request["ResourceConfig"])
        if "InstancePreferences" in resource and selected is not None:
            resource.update(SelectedInstanceType=selected, SelectedInstanceCount=resource["InstanceCount"])
        return {
            **{key: value for key, value in request.items() if key != "Tags"},
            "ResourceConfig": resource,
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/" + name,
            "TrainingJobStatus": self.states[name],
            "SecondaryStatus": "Pending" if selected is None else "Training",
        }

    def list_tags(self, **kw):
        return {"Tags": copy.deepcopy(self.jobs[kw["ResourceArn"].rsplit("/", 1)[-1]]["Tags"])}

    def complete(self, name):
        self.states[name] = "Completed"
        self.account.end(name)


def _tuned_checkpoint(store, s3, monkeypatch):
    from amlas.control_plane.sft_state import VerifiedSftCheckpointSource

    cp = ref("CHECKPOINT.json", raw=b"{}")
    cfg = ref("adapter_config.json", raw=b"{}")
    tensor = ref("adapter_model.safetensors", raw=b"abc", fmt="safetensors")
    for artifact, raw in ((cp, b"{}"), (cfg, b"{}"), (tensor, b"abc")):
        s3.install(artifact, raw)
    checkpoint = VerifiedSftCheckpointSource(
        SESSION,
        "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/owned-original",
        "full_v1",
        "d" * 40,
        cp,
        (("CHECKPOINT.json", cp), ("adapter_config.json", cfg), ("adapter_model.safetensors", tensor)),
    )
    monkeypatch.setattr(store._sft, "_verified_generation_checkpoint", lambda value, model: checkpoint)


class _SftHolder:
    """SFT port of this run: one original whose description shows a selected type (or none yet)."""

    def __init__(self, store, name, listed, selected, native_state="RUNNING", operation="train"):
        self.state, self.selected, self.observed = native_state, selected, []
        handle = JobHandle(
            name,
            "qwen_sft",
            "arn:aws:sagemaker:us-west-2:" + ACCOUNT + ":training-job/" + name,
            {
                "submission_state": "accepted",
                "operation": operation,
                "request": {"ResourceConfig": fixed_resource_config(listed)},
            },
        )
        sft = store._sft
        sft._lock, sft._records, sft._adapter = threading.RLock(), {"sft_" + "c" * 32: handle}, self

    def status(self, handle, **private):
        self.observed.append(handle.id)
        resource = dict(handle.extra["request"]["ResourceConfig"])
        if self.selected is not None:
            resource.update(SelectedInstanceType=self.selected, SelectedInstanceCount=1)
        return JobStatus(JobState(self.state), "", {"ResourceConfig": resource})


AFTER_THAT_JOB = "Nothing was created. Submit this arm again with the same arguments after that job ends."


# --- reload and generation in one message: their lists share no type ----------------------------------------------


@pytest.mark.parametrize("reload_type", RELOAD)
def test_reload_then_generation_in_one_message_are_both_created(state, reload_type):  # noqa: F811
    account = Account()
    p, checkpoint, selected, reader = parts()
    attempts = _in_account(p[1], account)
    reload_handle = ordinary_reload(p, checkpoint, selected, reader)
    assert reload_handle.extra["submission_state"] == "accepted" and attempts == [reload_handle.id]
    assert _listed(p[1].calls[-1]) == RELOAD and account.held == {"ml.g6e.12xlarge": reload_handle.id}
    # The one type the reload lists is not a generation type.
    account.held = {reload_type: reload_handle.id}
    native = QuotaNative(account)
    store, spec = _real_adapter(state, native)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    assert base["status"] == "submitted" and base["accepted"] is True and "next_step" not in base
    assert _listed(native.calls[-1]) == GENERATION
    assert account.held == {reload_type: reload_handle.id, "ml.g6e.16xlarge": base["job_name"]}


@pytest.mark.parametrize("generation_type", GENERATION)
def test_generation_then_reload_in_one_message_are_both_created(state, generation_type):  # noqa: F811
    account = Account()
    native = QuotaNative(account)
    store, spec = _real_adapter(state, native)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    assert base["status"] == "submitted" and account.held == {"ml.g6e.16xlarge": base["job_name"]}
    # Whichever listed type SageMaker gives the generation, it is not a reload type.
    account.held = {generation_type: base["job_name"]}
    p, checkpoint, selected, reader = parts()
    attempts = _in_account(p[1], account)
    reload_handle = ordinary_reload(p, checkpoint, selected, reader)
    assert reload_handle.extra["submission_state"] == "accepted" and attempts == [reload_handle.id]
    assert account.held == {generation_type: base["job_name"], "ml.g6e.12xlarge": reload_handle.id}
    assert p[4].load_original(p[5], reload_handle.id).disposition == "ACCEPTED"


@pytest.mark.parametrize("reload_ended", [False, True])
@pytest.mark.parametrize("base_ended", [False, True])
@pytest.mark.parametrize("base_type", GENERATION)
@pytest.mark.parametrize("order", ["reload_first", "generation_first"])
def test_every_interleaving_of_reload_and_base_then_tuned_is_created_or_submitted_again(
    state,  # noqa: F811
    monkeypatch,
    order,
    base_type,
    base_ended,
    reload_ended,
):
    """Reload and base generation in either order, then the tuned generation, under one job per type.

    No create that is never submitted again is refused: the reload is accepted in both orders whichever listed
    type the base generation holds, and the base generation is accepted whether or not the reload holds
    ml.g6e.12xlarge. The tuned generation is created at once when the base generation has ended, and is refused
    naming the type the base generation holds while it runs; the same tuned call after the base generation ends
    is a real create. Whether the reload still runs changes nothing.
    """
    account = Account()
    native = QuotaNative(account)
    store, spec = _real_adapter(state, native)
    _tuned_checkpoint(store, state[3], monkeypatch)
    p, checkpoint, selected, reader = parts()
    attempts = _in_account(p[1], account)
    made = {}

    def reload():
        made["reload"] = ordinary_reload(p, checkpoint, selected, reader)
        assert made["reload"].extra["submission_state"] == "accepted"

    def base():
        made["base"] = store.submit(MODEL_REF, CASE_REF, spec)
        assert made["base"]["status"] == "submitted" and made["base"]["accepted"] is True
        assert "next_step" not in made["base"] and _listed(native.calls[-1]) == GENERATION
        # Whichever listed type SageMaker gives the base generation.
        name = made["base"]["job_name"]
        account.end(name)
        account.held[base_type], native.selected[name] = name, base_type

    for create in (reload, base) if order == "reload_first" else (base, reload):
        create()
    reload_id, base_name = made["reload"].id, made["base"]["job_name"]
    assert attempts == [reload_id] and _listed(p[1].calls[-1]) == RELOAD
    assert p[4].load_original(p[5], reload_id).disposition == "ACCEPTED"
    assert account.held == {"ml.g6e.12xlarge": reload_id, base_type: base_name}
    assert store.observe(made["base"]["request_ref"])["SelectedInstanceType"] == base_type
    if reload_ended:
        account.end(reload_id)
    if base_ended:
        native.complete(base_name)
    creates = len(native.calls)
    tuned = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    if base_ended:
        assert tuned["status"] == "submitted" and tuned["accepted"] is True and "next_step" not in tuned
        assert len(native.calls) == creates + 1
    else:
        assert tuned["status"] == "submission_rejected" and tuned["accepted"] is False
        assert len(native.calls) == creates + 1 and list(native.jobs) == [base_name]
        assert tuned["blocking_instance_type"] == base_type and tuned["holding_job_name"] == base_name
        assert tuned["type_holders"] == {
            name: (base_name if name == base_type else None) for name in GENERATION
        }
        assert tuned["next_step"] == (
            "Job " + base_name + " of this run holds the " + base_type + " capacity. " + AFTER_THAT_JOB
        )
        native.complete(base_name)
        again = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
        assert again["status"] == "submitted" and again["request_ref"] == tuned["request_ref"]
        assert "next_step" not in again and len(native.calls) == creates + 2
    assert sorted(native.jobs) == sorted([base_name, tuned["job_name"]])
    observed = store.observe(tuned["request_ref"])
    assert observed["status"] == "RUNNING" and observed["accepted"] is True
    assert (observed["SelectedInstanceType"], observed["SelectedInstanceCount"]) == ("ml.g6e.16xlarge", 1)
    # The reload was sent once and never refused; the generation creates never named its type.
    assert attempts == [reload_id] and p[4].load_original(p[5], reload_id).disposition == "ACCEPTED"
    expected_held = {"ml.g6e.16xlarge": tuned["job_name"]}
    if not reload_ended:
        expected_held["ml.g6e.12xlarge"] = reload_id
    assert account.held == expected_held


# --- base and tuned generation share a list ------------------------------------------------------------------------


def test_tuned_generation_waits_for_the_type_base_holds(state, monkeypatch):  # noqa: F811
    native = QuotaNative()
    store, spec = _real_adapter(state, native)
    _tuned_checkpoint(store, state[3], monkeypatch)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    assert base["status"] == "submitted" and "next_step" not in base
    assert store.observe(base["request_ref"])["SelectedInstanceType"] == "ml.g6e.16xlarge"
    tuned = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    assert tuned["status"] == "submission_rejected" and tuned["accepted"] is False and len(native.calls) == 2
    assert _listed(native.calls[-1]) == GENERATION and list(native.jobs) == [base["job_name"]]
    assert store._records[tuned["request_ref"]].extra["rejection_type"] == "ml.g6e.16xlarge"
    assert (
        tuned["instance_types"] == list(GENERATION) and tuned["blocking_instance_type"] == "ml.g6e.16xlarge"
    )
    assert tuned["type_holders"] == {
        "ml.g6e.16xlarge": base["job_name"],
        "ml.g6e.4xlarge": None,
        "ml.g6e.8xlarge": None,
    }
    assert tuned["waiting_job_names"] == [] and tuned["holding_job_name"] == base["job_name"]
    assert tuned["next_step"] == (
        "Job " + base["job_name"] + " of this run holds the ml.g6e.16xlarge capacity. " + AFTER_THAT_JOB
    )
    # The base generation ends and frees ml.g6e.16xlarge: the same tuned call is a real create.
    native.complete(base["job_name"])
    again = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    assert (
        again["status"] == "submitted"
        and again["request_ref"] == tuned["request_ref"]
        and len(native.calls) == 3
    )
    assert "next_step" not in again and "blocking_instance_type" not in again
    observed = store.observe(again["request_ref"])
    assert (observed["SelectedInstanceType"], observed["SelectedInstanceCount"]) == ("ml.g6e.16xlarge", 1)


def test_two_list_jobs_accepted_before_either_selects_are_both_created(state, monkeypatch):  # noqa: F811
    """A Pending list job without a selection did not block another list create (observed 2026-11-10).

    The later selection of two different types is this stand-in's reading, not a platform record.
    """
    native = QuotaNative(select=False)
    store, spec = _real_adapter(state, native)
    _tuned_checkpoint(store, state[3], monkeypatch)
    base = store.submit(MODEL_REF, CASE_REF, spec)
    tuned = store.submit(MODEL_REF, CASE_REF, spec, SFT_REF)
    assert base["status"] == tuned["status"] == "submitted" and len(native.calls) == 2
    assert base["job_name"] != tuned["job_name"]
    for submitted in (base, tuned):
        waiting = store.observe(submitted["request_ref"])
        assert waiting["status"] == "RUNNING"
        assert not {"SelectedInstanceType", "SelectedInstanceCount"} & set(waiting)
    seen = {}
    for submitted in (base, tuned):
        native.select(submitted["job_name"])
        observed = store.observe(submitted["request_ref"])
        assert observed["status"] == "RUNNING" and observed["SelectedInstanceCount"] == 1
        seen[submitted["arm"]] = observed["SelectedInstanceType"]
    assert seen == {"base": "ml.g6e.16xlarge", "tuned": "ml.g6e.4xlarge"}


def test_a_job_outside_the_run_on_one_listed_type_refuses_generation_until_it_ends(state):  # noqa: F811
    """16xlarge and 4xlarge are free, yet a job on ml.g6e.8xlarge refuses the create; the note names only 8xlarge."""
    native = QuotaNative(outside=("ml.g6e.8xlarge",))
    store, spec = _real_adapter(state, native)
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and first["accepted"] is False
    assert native.jobs == {} and len(native.calls) == 1
    assert first["blocking_instance_type"] == "ml.g6e.8xlarge" and first["holding_job_name"] is None
    assert first["type_holders"] == {name: None for name in GENERATION} and first["waiting_job_names"] == []
    assert (
        first["next_step"]
        == "Another job in the account holds the ml.g6e.8xlarge capacity. " + AFTER_THAT_JOB
    )
    native.account.end("outside-account-job")
    again = store.submit(MODEL_REF, CASE_REF, spec)
    assert again["status"] == "submitted" and again["request_ref"] == first["request_ref"]
    assert store.observe(again["request_ref"])["SelectedInstanceType"] == "ml.g6e.16xlarge"


def test_a_reload_of_this_run_holds_no_generation_type_and_is_not_observed(state):  # noqa: F811
    native = QuotaNative(outside=("ml.g6e.4xlarge",))
    store, spec = _real_adapter(state, native)
    holder = _SftHolder(
        store, "owned-sft-reload", RELOAD_INSTANCE_PREFERENCES, "ml.g6e.12xlarge", operation="reload"
    )
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and first["holding_job_name"] is None
    assert first["type_holders"] == {name: None for name in GENERATION} and holder.observed == []
    assert (
        first["next_step"]
        == "Another job in the account holds the ml.g6e.4xlarge capacity. " + AFTER_THAT_JOB
    )


def test_holder_check_considers_every_listed_type(state):  # noqa: F811
    """A fine-tune of this run lists the generation types too; the type it selected is the one it holds."""
    native = QuotaNative()
    store, _spec = _real_adapter(state, native)
    other = "gen_" + "f" * 32
    for selected in GENERATION:
        _SftHolder(store, "owned-sft-holder", TRAIN_INSTANCE_PREFERENCES, selected)
        holders, waiting = store._active_holders(GENERATION_INSTANCE_PREFERENCES, other)
        assert holders == {name: ("owned-sft-holder" if name == selected else None) for name in GENERATION}
        assert waiting == []
    # A selected type outside the generation list is not a generation holder.
    for selected in RELOAD:
        _SftHolder(store, "owned-sft-holder", TRAIN_INSTANCE_PREFERENCES, selected)
        assert store._active_holders(GENERATION_INSTANCE_PREFERENCES, other) == (
            {name: None for name in GENERATION},
            [],
        )
    # An ended job holds nothing.
    _SftHolder(
        store, "owned-sft-holder", TRAIN_INSTANCE_PREFERENCES, "ml.g6e.4xlarge", native_state="SUCCEEDED"
    )
    assert store._active_holders(GENERATION_INSTANCE_PREFERENCES, other) == (
        {name: None for name in GENERATION},
        [],
    )


def test_a_waiting_list_job_of_this_run_is_reported_but_never_named_as_the_holder(state):  # noqa: F811
    """A list job without a selection holds no type: the note names the job that holds the blocking type."""
    native = QuotaNative(outside=("ml.g6e.4xlarge",))
    store, spec = _real_adapter(state, native)
    _SftHolder(store, "owned-sft-waiting", TRAIN_INSTANCE_PREFERENCES, None, native_state="PENDING")
    first = store.submit(MODEL_REF, CASE_REF, spec)
    assert first["status"] == "submission_rejected" and first["blocking_instance_type"] == "ml.g6e.4xlarge"
    assert first["type_holders"] == {name: None for name in GENERATION}
    assert first["waiting_job_names"] == ["owned-sft-waiting"] and first["holding_job_name"] is None
    assert (
        first["next_step"]
        == "Another job in the account holds the ml.g6e.4xlarge capacity. " + AFTER_THAT_JOB
    )


# --- fine-tune and reload: a job on any listed type rejects the one create for good ---------------------------------


@pytest.mark.parametrize("held", TRAIN)
def test_a_fixed_type_holder_of_any_listed_type_rejects_the_fine_tune_and_it_is_never_created_again(held):
    """Every listed type blocks: one job of the account on any of them rejects the fine-tune for good.

    The SFT path never submits a rejected job again (test_qwen_sft_durable_submission), also after the other job
    ends; a launch must keep other jobs off ml.g6e.16xlarge, 12xlarge, 8xlarge and 4xlarge until the fine-tune is
    created.
    """
    p = durable()
    account = Account(outside=(held,))
    attempts = _in_account(p[1], account)
    first = submit(p)
    account.end("outside-account-job")
    handles = [first, submit(p), submit(p)]
    assert [h.extra["submission_state"] for h in handles] == ["rejected"] * 3
    assert len({h.id for h in handles}) == 1 and len(attempts) == 1 and p[1].calls == []
    assert p[4].load_original(p[5], first.id).disposition == "REJECTED"


def test_a_job_on_an_unlisted_type_does_not_refuse_the_fine_tune():
    p = durable()
    attempts = _in_account(p[1], Account(outside=("ml.g6e.2xlarge",)))
    handle = submit(p)
    assert handle.extra["submission_state"] == "accepted" and len(attempts) == 1
    assert _listed(p[1].calls[-1]) == TRAIN


@pytest.mark.parametrize("held", RELOAD)
def test_a_job_on_a_reload_type_rejects_the_reload_and_it_is_never_created_again(held):
    p, checkpoint, selected, reader = parts()
    account = Account(outside=(held,))
    attempts = _in_account(p[1], account)
    first = ordinary_reload(p, checkpoint, selected, reader)
    account.end("outside-account-job")
    again = ordinary_reload(p, checkpoint, selected, reader)
    assert first.extra["submission_state"] == again.extra["submission_state"] == "rejected"
    assert first.id == again.id and len(attempts) == 1
    assert p[4].load_original(p[5], first.id).disposition == "REJECTED"


@pytest.mark.parametrize("held", [*GENERATION, "ml.g6e.2xlarge"])
def test_a_job_on_a_generation_type_does_not_refuse_the_reload(held):
    p, checkpoint, selected, reader = parts()
    attempts = _in_account(p[1], Account(outside=(held,)))
    handle = ordinary_reload(p, checkpoint, selected, reader)
    assert handle.extra["submission_state"] == "accepted" and len(attempts) == 1
    assert _listed(p[1].calls[-1]) == RELOAD
