"""One finite ordinary SDK graph, backed only by inert native response fixtures.

No store records, approval witnesses, journal rows or private task identifiers are
seeded. The SDK submits every original through the registered tools. Provider
output publication below models native job completion; it grants no native AWS,
worker-loss, executable-oracle, ML quality or full C1-C4/R1-R4 acceptance.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import threading
from types import SimpleNamespace

import boto3
import model_acquisition_fixtures as model_fixture
import test_managed_dataset_acquisition as data_fixture
import test_normal_run_checkpoint as normal_fixture
import test_paired_coding_backend as pair_fixture
import test_qwen_sft_backend as sft_fixture
from botocore.exceptions import ClientError
from botocore.handlers import handle_copy_source_param
from botocore.session import Session
from botocore.validate import validate_parameters
from multi_producer_fixtures import MultipleNativeData
from strands import Agent
from strands.models import Model
from test_coding_preparation_worker import PROOF
from test_commitpack_profile import row as commitpack_row
from test_managed_dataset_acquisition import NativeS3, populate
from test_managed_generation_product import spec_text
from test_normal_research_profile import CASE_URLS, MODEL_URLS, TRAIN_URL
from test_qwen_sft_backend import described_resource_config
from test_runtime_connected_warm_model import close_original_worker
from test_runtime_generation_originals import publish_generation
from test_runtime_warm_model_continuation import complete_fixture, tool_payload

from amlas import app as runtime
from amlas.backends.adapters import coding_generation, dataset_acquisition, model_weight_acquisition, qwen_sft
from amlas.backends.adapters.coding_case_freeze import ProcessingCodingCaseFreezeBackend
from amlas.backends.adapters.coding_pair import SageMakerPairedCodingBackend
from amlas.backends.adapters.coding_preparation import ProcessingCodingPreparationBackend
from amlas.backends.ports import (
    CodingCaseFreezeBackend,
    CodingGenerationBackend,
    CodingPreparationBackend,
    PairedCodingBackend,
    QwenSftBackend,
)
from amlas.backends.registry import REGISTRY
from amlas.backends.types import DatasetArtifactRef
from amlas.control_plane import sft_state
from amlas.control_plane.normal_run_checkpoint import NormalRunContext
from amlas.data_eng import (
    coding_case_profile,
    commitpack_preparation,
    commitpack_profile,
    model_acquisition_schema,
    model_weight_manifest,
)
from amlas.data_eng.coding_case_managed import freeze_managed_case_bundle
from amlas.flows import dataset_acquisition_flow, model_weight_acquisition_flow
from amlas.tools.research import public_fetch
from amlas.training import qwen_sft as training_qwen_sft
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend

SFT_USE = "cold-original-sft"
FINAL_TEXT = "The original paired evaluation is complete; fixture quality remains false."
FIT_FINAL_TEXT = "The original fit pilot saved its checkpoint; fixture quality remains false."
FIT_STATUS_USE = "cold-fit-status"
# Returned by post_sft_stage when the script ends with its final text right after the SFT result.
REPORT_NOW = object()
ACTIONS = [
    "run_processing_job",
    "data_collection_flow",
    "coding_case_freeze_flow",
    "data_preparation_flow",
    "qwen_sft_flow",
    "coding_generation_flow",
    "coding_pair_flow",
]


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def payloads(messages):
    return {
        block["toolResult"]["toolUseId"]: tool_payload(block["toolResult"])
        for message in messages
        for block in message["content"]
        if "toolResult" in block
    }


class FullGraphModel(Model):
    """Choose the next registered tool from admitted DATA, never from warm RAM."""

    # The recipe of the one original SFT this script submits, and the text that ends the script.
    original_recipe_id = "full_v1"
    final_text = FINAL_TEXT

    def __init__(self):
        self.calls = 0
        self.messages_seen = []

    def sft_arguments(self, known, recipe_id="full_v1"):
        return {
            "model_request_ref": known["cold-original-model"]["request_ref"],
            "preparation_request_ref": known["cold-original-preparation"]["request_ref"],
            "recipe_id": recipe_id,
        }

    def sft_stage(self, known):
        """Batch to emit once the preparation is ready and no SFT result exists yet."""
        return [(SFT_USE, "submit_qwen_sft", self.sft_arguments(known))]

    def post_sft_stage(self, known):
        """Optional batch to emit after the admitted SFT result and before the spec is bound.

        REPORT_NOW ends the script with its final text instead of binding the spec.
        """
        return None

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": "neutral-finite-cold-sft-component"}

    def structured_output(self, *args, **kwargs):
        raise AssertionError("ordinary SDK stream required")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.messages_seen.append(copy.deepcopy(messages))
        known = payloads(messages)
        case = known.get("cold-original-cases", {})
        completed_case = (
            case.get("event_type") == "original_case_completed"
            and case.get("status") == "verified_completed"
            and case.get("data_only") is True
        )
        case_reference = case.get("request_ref") if completed_case else case.get("case_ref")
        case_observations = [value for use, value in known.items() if use.startswith("cold-case-status-")]
        case_observed = case_observations[-1] if case_observations else case
        case_ready = completed_case or (
            case_observed.get("frozen") is True
            and case_observed.get("case_ref") == case_reference
            and case_observed.get("case_count") == 164
        )
        selected = []
        for identifier, urls in (
            ("cold-research-model", MODEL_URLS),
            ("cold-research-training", (TRAIN_URL,)),
            ("cold-research-cases", CASE_URLS),
        ):
            if identifier not in known:
                selected = [(identifier, "research_fetch_urls", {"urls": list(urls), "max_pages": 3})]
                break
        else:
            if "cold-original-model" not in known:
                selected = [
                    (
                        "cold-original-model",
                        "acquire_model_weights",
                        {"catalog_id": "qwen3_5_9b", "catalog_version": "1"},
                    ),
                    (
                        "cold-original-training-data",
                        "acquire_huggingface_dataset",
                        {
                            "repo_id": commitpack_profile.DATASET_REPO,
                            "revision": commitpack_profile.REVISION,
                            "filename": commitpack_profile.FILENAME,
                        },
                    ),
                    (
                        "cold-original-case-data",
                        "acquire_huggingface_dataset",
                        {
                            "repo_id": coding_case_profile.SOURCE_REPO,
                            "revision": coding_case_profile.SOURCE_REVISION,
                            "filename": coding_case_profile.SOURCE_FILENAME,
                        },
                    ),
                ]
            elif "cold-original-cases" not in known:
                selected = [
                    (
                        "cold-original-cases",
                        "freeze_coding_cases",
                        {"acquisition_request_ref": known["cold-original-case-data"]["request_ref"]},
                    )
                ]
            elif not case_ready:
                selected = [
                    (
                        "cold-case-status-" + str(self.calls),
                        "get_coding_case_freeze",
                        {"case_ref": case_reference},
                    )
                ]
            elif "cold-original-preparation" not in known:
                selected = [
                    (
                        "cold-original-preparation",
                        "prepare_coding_dataset",
                        {
                            "acquisition_request_refs": [known["cold-original-training-data"]["request_ref"]],
                            "case_request_ref": case_reference,
                        },
                    )
                ]
            elif SFT_USE not in known:
                selected = self.sft_stage(known)
            elif "cold-original-spec" not in known:
                assert (
                    known[SFT_USE].get("event_type") == "original_sft_completed"
                    and known[SFT_USE].get("status") == "verified_completed"
                    and known[SFT_USE].get("data_only") is True
                ), "SDK saw pending original instead of its admitted completed result"
                staged = self.post_sft_stage(known)
                selected = (
                    []
                    if staged is REPORT_NOW
                    else staged
                    or [("cold-original-spec", "bind_coding_generation_spec", {"yaml_text": spec_text()})]
                )
            elif "cold-original-resolved-spec" not in known:
                selected = [
                    (
                        "cold-original-resolved-spec",
                        "resolve_coding_evaluation_spec",
                        {
                            "case_request_ref": case_reference,
                            "spec_ref": known["cold-original-spec"]["spec_ref"],
                        },
                    )
                ]
            elif "cold-original-generation-base" not in known:
                selected = [
                    (
                        "cold-original-generation-" + arm,
                        "submit_coding_generation",
                        {
                            "model_request_ref": known["cold-original-model"]["request_ref"],
                            "case_request_ref": case_reference,
                            "spec_ref": known["cold-original-resolved-spec"]["spec_ref"],
                            "sft_request_ref": known[SFT_USE]["request_ref"] if arm == "tuned" else "",
                        },
                    )
                    for arm in ("base", "tuned")
                ]
            elif "cold-original-pair" not in known:
                selected = [
                    (
                        "cold-original-pair",
                        "submit_coding_pair",
                        {
                            "case_request_ref": case_reference,
                            "base_generation_ref": known["cold-original-generation-base"]["request_ref"],
                            "tuned_generation_ref": known["cold-original-generation-tuned"]["request_ref"],
                            "resolved_spec_ref": known["cold-original-resolved-spec"]["spec_ref"],
                        },
                    )
                ]
        yield {"messageStart": {"role": "assistant"}}
        if selected:
            for index, (identifier, name, arguments) in enumerate(selected):
                yield {
                    "contentBlockStart": {
                        "contentBlockIndex": index,
                        "start": {"toolUse": {"toolUseId": identifier, "name": name}},
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": index,
                        "delta": {"toolUse": {"input": json.dumps(arguments)}},
                    }
                }
                yield {"contentBlockStop": {"contentBlockIndex": index}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": self.final_text}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }


MIXED_PARTS = ("sft", "other", "gen")


class MixedBatchModel(FullGraphModel):
    """First SFT turn: ONE assistant message mixing submit_qwen_sft with other tools, in the given order.

    After the whole batch is refused the model complies and calls submit_qwen_sft alone.
    """

    def __init__(self, order=MIXED_PARTS):
        super().__init__()
        assert sorted(order) == sorted(MIXED_PARTS)
        self.order = tuple(order)

    def sft_stage(self, known):
        if "mixed-sft" in known:
            return super().sft_stage(known)
        case = known["cold-original-cases"]
        case_ref = case.get("request_ref") or case.get("case_ref")
        parts = {
            "sft": ("mixed-sft", "submit_qwen_sft", self.sft_arguments(known)),
            "other": ("mixed-other", "get_coding_case_freeze", {"case_ref": case_ref}),
            "gen": (
                "mixed-gen",
                "submit_coding_generation",
                {
                    "model_request_ref": known["cold-original-model"]["request_ref"],
                    "case_request_ref": case_ref,
                    "spec_ref": "spec_" + "0" * 32,
                    "sft_request_ref": "",
                },
            ),
        }
        return [parts[name] for name in self.order]


class SpecFirstModel(FullGraphModel):
    """Bind and resolve the specification before the SFT, so a replacement meets references from before the stop."""

    def sft_stage(self, known):
        case = known["cold-original-cases"]
        case_ref = case.get("request_ref") or case.get("case_ref")
        if "cold-original-spec" not in known:
            return [("cold-original-spec", "bind_coding_generation_spec", {"yaml_text": spec_text()})]
        if "cold-original-resolved-spec" not in known:
            return [
                (
                    "cold-original-resolved-spec",
                    "resolve_coding_evaluation_spec",
                    {"case_request_ref": case_ref, "spec_ref": known["cold-original-spec"]["spec_ref"]},
                )
            ]
        return super().sft_stage(known)


class RepeatSftModel(FullGraphModel):
    """After the admitted SFT result, call submit_qwen_sft again in its own message with the given recipe."""

    recipe_id = "full_v1"

    def post_sft_stage(self, known):
        if "repeat-sft" in known:
            return None
        return [("repeat-sft", "submit_qwen_sft", self.sft_arguments(known, self.recipe_id))]


class RepeatDifferentSftModel(RepeatSftModel):
    recipe_id = "fit_v1"


class FitOnlyModel(FullGraphModel):
    """The pilot script: one fit_v1 original SFT, one status read of it, then the final text.

    No specification, reload, generation or paired evaluation follows.
    """

    original_recipe_id = "fit_v1"
    final_text = FIT_FINAL_TEXT

    def sft_stage(self, known):
        return [(SFT_USE, "submit_qwen_sft", self.sft_arguments(known, "fit_v1"))]

    def post_sft_stage(self, known):
        if FIT_STATUS_USE in known:
            return REPORT_NOW
        return [(FIT_STATUS_USE, "get_qwen_sft", {"request_ref": known[SFT_USE]["request_ref"]})]


def model_for_variant(variant):
    """Scripted model for a named variant; the empty name is the unchanged single-SFT script."""
    if not variant:
        return FullGraphModel()
    if variant.startswith("mixed-"):
        return MixedBatchModel(tuple(variant.split("-")[1:]))
    return {
        "repeat-same": RepeatSftModel,
        "repeat-different": RepeatDifferentSftModel,
        "spec-first": SpecFirstModel,
        "fit-only": FitOnlyModel,
    }[variant]()


def waiting_description(resource, state):
    """Description fields of a preference-list job that never left the waiting phase.

    No SelectedInstanceType or SelectedInstanceCount, and only Starting and Pending before a Stopped or Failed
    transition: the shape of a real job that SageMaker stopped while it waited for capacity.
    """
    statuses = ("Starting", "Pending") + ((state,) if state in ("Stopped", "Failed") else ())
    return {
        "ResourceConfig": copy.deepcopy(resource),
        "SecondaryStatus": statuses[-1],
        "SecondaryStatusTransitions": [{"Status": status, "StatusMessage": "neutral"} for status in statuses],
    }


class NativeJobs:
    """Exact native request shapes and retained original responses, no execution."""

    meta = SimpleNamespace(region_name="us-west-2")

    def __init__(self):
        self.calls = []
        self.processing = {}
        self.states = {}
        self.reads = []
        self.stops = []
        self.duplicates = []
        # Training jobs described as preference-list jobs that SageMaker has not given an instance (yet, or ever);
        # with unselected_training every training job created from now on joins them.
        self.unselected = set()
        self.unselected_training = False
        self.model = Session().get_service_model("sagemaker")

    def check(self, operation, values):
        validate_parameters(values, self.model.operation_model(operation).input_shape)

    def arn(self, kind, name):
        return "arn:aws:sagemaker:us-west-2:" + model_fixture.ACCOUNT + ":" + kind + "/" + name

    def refuse_duplicate(self, name, operation):
        """A name already created: the service answers ResourceInUse for the deterministic generation and pair names.

        A repeat of any other native original is a defect of the caller and stays an assertion.
        """
        if name not in self.states and name not in self.processing:
            return
        self.duplicates.append(name)
        if "-gen-" in name or "-pair-" in name:
            raise ClientError({"Error": {"Code": "ResourceInUse", "Message": "name in use"}}, operation)
        raise AssertionError("a native original was recreated")

    def create_training_job(self, **values):
        self.check("CreateTrainingJob", values)
        name = values["TrainingJobName"]
        self.refuse_duplicate(name, "CreateTrainingJob")
        self.calls.append(copy.deepcopy(values))
        self.states[name] = "InProgress"
        if self.unselected_training:
            self.unselected.add(name)
        return {"TrainingJobArn": self.arn("training-job", name)}

    def describe_training_job(self, **values):
        self.check("DescribeTrainingJob", values)
        name = values["TrainingJobName"]
        request = next(value for value in self.calls if value["TrainingJobName"] == name)
        self.reads.append(name)
        return {
            **{key: value for key, value in request.items() if key != "Tags"},
            "ResourceConfig": described_resource_config(request["ResourceConfig"]),
            "TrainingJobArn": self.arn("training-job", name),
            "TrainingJobStatus": self.states[name],
            "ModelArtifacts": {"S3ModelArtifacts": request["OutputDataConfig"]["S3OutputPath"] + "/model/"},
            **(
                waiting_description(request["ResourceConfig"], self.states[name])
                if name in self.unselected
                else {}
            ),
        }

    def create_processing_job(self, **values):
        self.check("CreateProcessingJob", values)
        name = values["ProcessingJobName"]
        self.refuse_duplicate(name, "CreateProcessingJob")
        self.processing[name] = copy.deepcopy(values)
        self.states[name] = "InProgress"
        return {"ProcessingJobArn": self.arn("processing-job", name)}

    def describe_processing_job(self, **values):
        self.check("DescribeProcessingJob", values)
        name = values["ProcessingJobName"]
        self.reads.append(name)
        return {
            **{key: value for key, value in self.processing[name].items() if key != "Tags"},
            "ProcessingJobArn": self.arn("processing-job", name),
            "ProcessingJobStatus": self.states[name],
        }

    def list_tags(self, **values):
        self.check("ListTags", values)
        name = values["ResourceArn"].rsplit("/", 1)[1]
        request = self.processing.get(name) or next(
            value for value in self.calls if value["TrainingJobName"] == name
        )
        return {"Tags": copy.deepcopy(request["Tags"])}

    def stop_training_job(self, **values):
        self.stops.append(values)
        raise AssertionError("cold continuation must preserve its native original")

    stop_processing_job = stop_training_job


class NativeObjects(pair_fixture.Objects):
    """Reuse strict versioned native S3 staging; expose existing publisher view."""

    def __init__(self, ports):
        super().__init__()
        self.ports = ports
        self.key_errors = []

    def _note_key_error(self, operation, error, lookup):
        frame = error.__traceback__
        while frame.tb_next is not None:
            frame = frame.tb_next
        self.key_errors.append(
            {
                "operation": operation,
                "function": frame.tb_frame.f_code.co_name,
                "file": frame.tb_frame.f_code.co_filename.rsplit("/", 1)[-1],
                "line": frame.tb_lineno,
                "self_tuple_present": lookup in self.objects,
                "ports_tuple_present": lookup in self.ports.objects,
            }
        )

    def install(self, key, raw, version=None):
        if type(key) is DatasetArtifactRef:
            key, version = key.uri.split("/", 3)[3], key.version_id
        ref = super().install(key, raw, version)
        self.objects[key] = (ref, raw)
        return ref

    def copy_object(self, **values):
        source = values["CopySource"]
        wire = copy.deepcopy(values)
        handle_copy_source_param(wire)
        self.check("CopyObject", wire)
        lookup = (source["Key"], source["VersionId"])
        try:
            raw = self.objects[lookup]
            return {"VersionId": self.install(values["Key"], raw).version_id}
        except KeyError as error:
            self._note_key_error("CopyObject", error, lookup)
            raise

    def put_object(self, **values):
        self.check("PutObject", values)
        if "IfNoneMatch" in values:
            assert values["IfNoneMatch"] == "*" and values["Key"] not in self.versions
        raw = values["Body"]
        assert type(raw) is bytes
        return {"VersionId": self.install(values["Key"], raw).version_id}

    def get_bucket_location(self, **values):
        self.check("GetBucketLocation", values)
        return {"LocationConstraint": "us-west-2"}

    def get_object(self, **values):
        lookup = (values["Key"], values.get("VersionId"))
        try:
            if lookup in self.ports.objects:
                return self.ports.get_object(**values)
            if values.get("VersionId") is None:
                # Native metadata discovery GET may resolve latest; its returned
                # exact version is then validated/pinned by the Source consumer.
                self.check("GetObject", values)
                version = self.versions[values["Key"]]
                return super().get_object(**{**values, "VersionId": version})
            return super().get_object(**values)
        except KeyError as error:
            self._note_key_error("GetObject", error, lookup)
            raise

    def _latest_publisher_object(self, key):
        versions = [
            (version, raw) for (source_key, version), raw in self.ports.objects.items() if source_key == key
        ]
        if len(versions) != 1:
            raise ValueError("fresh immutable publisher latest version is ambiguous")
        return versions[0]

    def list_objects_v2(self, **values):
        self.check("ListObjectsV2", values)
        names = sorted(
            key
            for key in set(self.versions) | {key for key, _ in self.ports.objects}
            if key.startswith(values["Prefix"])
        )
        items = []
        for key in names[: values["MaxKeys"]]:
            if key in self.versions:
                raw = self.objects[(key, self.versions[key])]
            else:
                _version, raw = self._latest_publisher_object(key)
            items.append({"Key": key, "Size": len(raw)})
        return {"IsTruncated": len(names) > values["MaxKeys"], "Contents": items}

    def head_object(self, **values):
        self.check("HeadObject", values)
        if "VersionId" not in values and values["Key"] not in self.versions:
            version, raw = self._latest_publisher_object(values["Key"])
            return {"VersionId": version, "ContentLength": len(raw)}
        lookup = (values["Key"], values.get("VersionId"))
        try:
            native = self.ports.objects.get(lookup)
            if native is not None:
                return {"VersionId": values["VersionId"], "ContentLength": len(native)}
            return super().head_object(**values)
        except KeyError as error:
            self._note_key_error("HeadObject", error, lookup)
            raise


def official_case_data():
    """Read only exact public fixture DATA; no candidate/oracle is executed."""
    import io
    import os
    from pathlib import Path

    import pyarrow.parquet as pq

    from amlas.evaluation.coding_oracle_catalog import verify_pinned_case

    path = Path(os.environ["AMLAS_TEST_HUMANEVAL_PARQUET"])
    with path.open("rb") as stream:
        raw = stream.read(coding_case_profile.SOURCE_BYTES + 1)
    if len(raw) != coding_case_profile.SOURCE_BYTES or sha(raw) != coding_case_profile.SOURCE_SHA256:
        raise ValueError("exact pinned public CASE fixture bytes required")
    table = pq.ParquetFile(io.BytesIO(raw)).read()
    if table.num_rows != 164 or set(table.column_names) != coding_case_profile.SOURCE_FIELDS:
        raise ValueError("complete pinned public CASE fixture schema required")
    records = table.to_pylist()
    for row in records:
        if (
            type(row) is not dict
            or set(row) != coding_case_profile.SOURCE_FIELDS
            or any(type(value) is not str for value in row.values())
        ):
            raise ValueError("closed pinned public CASE fixture records required")
        verify_pinned_case(
            {"case_id": row["task_id"], "prompt": row["prompt"], "entry_point": row["entry_point"]},
            {
                "case_id": row["task_id"],
                "test": row["test"],
                "canonical_solution": row["canonical_solution"],
            },
        )
    if tuple(row["task_id"] for row in records) != coding_case_profile.CASE_IDS:
        raise ValueError("ordered pinned public CASE fixture population required")
    return raw, records


def setup(monkeypatch, directory, *, captured=None, stale_cas=False, variant=None):
    """Configure existing builders/adapters and a fresh normal fixture policy."""
    task_published = threading.Event()
    original_save_task = NormalRunContext.save_admitted_task

    def save_task(context, *args, **kwargs):
        result = original_save_task(context, *args, **kwargs)
        task_published.set()
        return result

    monkeypatch.setattr(NormalRunContext, "save_admitted_task", save_task)
    monkeypatch.setattr(pair_fixture, "ACCOUNT", model_fixture.ACCOUNT)
    monkeypatch.setattr(sft_fixture, "ACCOUNT", model_fixture.ACCOUNT)
    monkeypatch.setattr(sft_fixture, "BUCKET", model_fixture.BUCKET)
    monkeypatch.setattr(sft_fixture, "SOURCE", "a" * 40)
    monkeypatch.setattr(
        sft_fixture,
        "IMAGE",
        model_fixture.ACCOUNT + ".dkr.ecr.us-west-2.amazonaws.com/source@sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        sft_fixture,
        "HEADER",
        {
            **sft_fixture.HEADER,
            "deployment_id": model_fixture.DEPLOYMENT,
            "deployment_prefix": "fixture-lab",
            "image_uri": sft_fixture.IMAGE,
            "source_commit": sft_fixture.SOURCE,
        },
    )
    monkeypatch.setattr(normal_fixture, "ACCOUNT", model_fixture.ACCOUNT)
    # Keep the existing explicit normal runtime role/RoleId fixture distinct
    # from all trainer roles; these are source fixture configuration, not rows.
    monkeypatch.setenv("AMLAS_MODEL_WEIGHT_ACQUISITION_IMAGE_URI", model_fixture.IMAGE)
    monkeypatch.setenv("AMLAS_MODEL_WEIGHT_ACQUISITION_SOURCE_COMMIT", model_fixture.COMMIT)
    monkeypatch.setenv("AMLAS_DATA_ACQUISITION_IMAGE_URI", data_fixture.IMAGE)
    monkeypatch.setenv("AMLAS_DATA_ACQUISITION_SOURCE_COMMIT", "c" * 40)
    monkeypatch.setenv(
        "AMLAS_MODEL_WEIGHT_ACQUISITION_ADMISSION_DIGEST", model_fixture.IMAGE.rsplit(":", 1)[1]
    )
    generator = normal_fixture.normal_work.__wrapped__(monkeypatch, directory)
    work, ports, configured_model = next(generator)
    model = model_for_variant(os.environ.get("AMLAS_COLD_MODEL_VARIANT", "") if variant is None else variant)
    monkeypatch.setattr(configured_model, "stream", model.stream)
    work.finite_model = model
    pending_phase_trace = []
    observed_backends = set()
    original_build_agent = ToolBindingBackend.build_agent

    def note_pending_phase(phase, error):
        frame = error.__traceback__
        while frame.tb_next is not None:
            frame = frame.tb_next
        pending_phase_trace.append(
            {
                "phase": phase,
                "error_type": type(error).__name__,
                "function": frame.tb_frame.f_code.co_name,
                "file": frame.tb_frame.f_code.co_filename.rsplit("/", 1)[-1],
                "line": frame.tb_lineno,
            }
        )

    def observe_build_agent(backend, **kwargs):
        if id(backend) not in observed_backends:
            original_complete = backend._complete_original_before_commit
            original_mark = backend._mark_pending_pause

            async def observe_complete(event, witness):
                try:
                    return await original_complete(event, witness)
                except Exception as error:
                    note_pending_phase("complete_original_before_commit", error)
                    raise

            def observe_mark(event):
                try:
                    return original_mark(event)
                except Exception as error:
                    note_pending_phase("mark_pending_pause", error)
                    raise

            monkeypatch.setattr(backend, "_complete_original_before_commit", observe_complete)
            monkeypatch.setattr(backend, "_mark_pending_pause", observe_mark)
            observed_backends.add(id(backend))
        return original_build_agent(backend, **kwargs)

    monkeypatch.setattr(ToolBindingBackend, "build_agent", observe_build_agent)
    native, objects = NativeJobs(), NativeObjects(ports)
    data = SimpleNamespace(native=MultipleNativeData(), s3=NativeS3())
    data.adapter = dataset_acquisition.ProcessingDatasetAcquisitionBackend(
        region=data_fixture.REGION,
        role_arn=data_fixture.ROLE,
        image_uri=data_fixture.IMAGE,
        session_bucket=model_fixture.BUCKET,
        source_commit="c" * 40,
        sagemaker_client=data.native,
        s3_client=data.s3,
    )
    monkeypatch.setattr(
        dataset_acquisition_flow,
        "_resolve_backend",
        lambda: SimpleNamespace(adapter=data.adapter, adapter_name=data.adapter.backend_name),
    )
    adapters = {
        dataset_acquisition.DataAcquisitionBackend: data.adapter,
        CodingCaseFreezeBackend: ProcessingCodingCaseFreezeBackend(
            sagemaker_client=native, s3_client=objects
        ),
        CodingPreparationBackend: ProcessingCodingPreparationBackend(
            sagemaker_client=native, s3_client=objects
        ),
        QwenSftBackend: qwen_sft.SageMakerQwenSftBackend(sagemaker_client=native, s3_client=objects),
        CodingGenerationBackend: coding_generation.SageMakerCodingGenerationBackend(
            sagemaker_client=native, s3_client=objects
        ),
        PairedCodingBackend: SageMakerPairedCodingBackend(sagemaker_client=native, s3_client=objects),
    }
    preparation_trace = []
    original_preparation_submit = adapters[CodingPreparationBackend].submit

    def observe_preparation_submit(**kwargs):
        try:
            return original_preparation_submit(**kwargs)
        except Exception as error:
            frame = error.__traceback__
            while frame.tb_next is not None:
                frame = frame.tb_next
            preparation_trace.append(
                {
                    "error_type": type(error).__name__,
                    "function": frame.tb_frame.f_code.co_name,
                    "file": frame.tb_frame.f_code.co_filename.rsplit("/", 1)[-1],
                    "line": frame.tb_lineno,
                }
            )
            raise

    monkeypatch.setattr(adapters[CodingPreparationBackend], "submit", observe_preparation_submit)
    original_resolve = REGISTRY.resolve
    monkeypatch.setattr(
        REGISTRY,
        "resolve",
        lambda port, *args, **kwargs: (
            SimpleNamespace(adapter=adapters[port], adapter_name=adapters[port].backend_name)
            if port in adapters
            else original_resolve(port, *args, **kwargs)
        ),
    )
    model_fixture.patch_expected(
        monkeypatch,
        [
            model_weight_acquisition,
            model_acquisition_schema,
            model_weight_manifest,
            qwen_sft,
            coding_generation,
            sft_state,
            training_qwen_sft,
        ],
    )
    training = b"".join(encoded(commitpack_row(index + 1)) for index in range(2000))
    case_raw, _case_records = official_case_data()
    monkeypatch.setattr(commitpack_profile, "ARTIFACT_SHA256", sha(training))
    monkeypatch.setattr(commitpack_preparation, "SOURCE_BYTES", len(training))
    for key, value in {
        "SAGEMAKER_EXECUTION_ROLE_ARN": model_fixture.ROLE,
        "AMLAS_RUNTIME_APPROVED_ACTIONS": json.dumps(ACTIONS),
        "AMLAS_CODING_PAIR_LAUNCHER_SHA256": "d" * 64,
    }.items():
        monkeypatch.setenv(key, value)
    for prefix in (
        "AMLAS_CODING_CASE_FREEZE",
        "AMLAS_DATA_PREPARATION",
        "AMLAS_QWEN_SFT",
        "AMLAS_CODING_GENERATION",
        "AMLAS_CODING_PAIR",
    ):
        for suffix, value in (
            ("IMAGE_URI", sft_fixture.IMAGE),
            ("SOURCE_COMMIT", sft_fixture.SOURCE),
            ("ADMISSION_DIGEST", "b" * 64),
        ):
            monkeypatch.setenv(prefix + "_" + suffix, value)
    urls = (*MODEL_URLS, TRAIN_URL, *CASE_URLS)
    monkeypatch.setattr(
        public_fetch, "_checked_addresses", lambda url, timeout: ("93.184.216.34",) if url in urls else ()
    )
    monkeypatch.setattr(
        public_fetch,
        "_request_public",
        lambda url, ips, timeout, max_bytes: (
            200,
            {"content-type": "text/plain"},
            b"Neutral independently licensed coding research fixture.",
        ),
    )

    async def short_wait(_backend):
        await asyncio.sleep(0.01)

    monkeypatch.setattr(ToolBindingBackend, "_wait_precommit_original", short_wait)
    actual_client = boto3.client

    def client(service, **kwargs):
        if service == "sagemaker":
            return native
        if service == "s3":
            return objects
        return actual_client(service, **kwargs)

    class NativeSession:
        def __init__(self, *args, **kwargs):
            pass

        def client(self, service, **kwargs):
            return client(service, **kwargs)

    # Warm MODEL and fresh SDK clients share one declared service inventory.
    # Keep native validators/response shape and all original Create assertions.
    original_describe_processing = native.describe_processing_job
    original_list_tags = native.list_tags
    original_describe_data = data.native.describe_processing_job

    def describe_shared_processing(**values):
        if any(
            request["ProcessingJobName"] == values["ProcessingJobName"] for request in work.native.requests
        ):
            return work.native.describe_processing_job(**values)
        return original_describe_processing(**values)

    def list_shared_tags(**values):
        name = values["ResourceArn"].rsplit("/", 1)[-1]
        selected = next(
            (request for request in work.native.requests if request["ProcessingJobName"] == name), None
        )
        if selected is not None:
            native.check("ListTags", values)
            return {"Tags": copy.deepcopy(selected["Tags"])}
        return original_list_tags(**values)

    def describe_complete_data(**values):
        result = original_describe_data(**values)
        request = data.native.requests[values["ProcessingJobName"]]
        return {
            **result,
            "ProcessingResources": copy.deepcopy(request["ProcessingResources"]),
            "StoppingCondition": copy.deepcopy(request["StoppingCondition"]),
        }

    monkeypatch.setattr(data.native, "describe_processing_job", describe_complete_data)

    monkeypatch.setattr(native, "describe_processing_job", describe_shared_processing)
    monkeypatch.setattr(native, "list_tags", list_shared_tags)

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setattr(boto3, "Session", NativeSession)
    monkeypatch.setattr(boto3.session, "Session", NativeSession)
    state = SimpleNamespace(
        task_published=task_published,
        work=work,
        ports=ports,
        model=model,
        native=native,
        objects=objects,
        data=data,
        adapters=adapters,
        training=training,
        case_raw=case_raw,
        generator=generator,
        directory=directory,
        loaded=[],
        invoked=[],
        rejected_cas=[],
        preparation_trace=preparation_trace,
        pending_phase_trace=pending_phase_trace,
    )
    if captured is not None:
        ports.items = copy.deepcopy(captured["ddb"])
        ports.objects = {
            (item["key"], item["version"]): item["raw"] for item in captured["checkpoint_objects"]
        }
        native.calls = copy.deepcopy(captured["training_requests"])
        native.processing = copy.deepcopy(captured["processing_requests"])
        native.states = copy.deepcopy(captured["states"])
        native.unselected = set(captured.get("unselected", ()))
        # Fresh provider clients describe captured originals; no store is seeded.
        work.adapter = model_fixture.configured_backend(model_weight_acquisition, native, objects)
        monkeypatch.setattr(model_weight_acquisition_flow, "_resolve_backend", lambda: work.adapter)
        data.adapter = dataset_acquisition.ProcessingDatasetAcquisitionBackend(
            region=data_fixture.REGION,
            role_arn=data_fixture.ROLE,
            image_uri=data_fixture.IMAGE,
            session_bucket=model_fixture.BUCKET,
            source_commit="c" * 40,
            sagemaker_client=native,
            s3_client=objects,
        )
        adapters[dataset_acquisition.DataAcquisitionBackend] = data.adapter
        for item in captured["stage_objects"]:
            objects.install(item["key"], item["raw"], item["version"])
        original_load, original_invoke = Agent.load_snapshot, Agent.invoke_async

        def load(agent, snapshot):
            state.loaded.append(copy.deepcopy(snapshot.to_dict()))
            return original_load(agent, snapshot)

        async def invoke(agent, prompt=None, **kwargs):
            state.invoked.append({"prompt": prompt, "messages": copy.deepcopy(agent.messages)})
            assert prompt is None, (
                "cold original must continue load_snapshot with invoke_async(None), without goal replay or orphan repair"
            )
            return await original_invoke(agent, prompt, **kwargs)

        monkeypatch.setattr(Agent, "load_snapshot", load)
        monkeypatch.setattr(Agent, "invoke_async", invoke)
        if stale_cas:
            original_update = ports.update_item

            def rejected_update(**values):
                if values["Key"]["job_name"]["S"].startswith("PROMPT_RUN#"):
                    normal_fixture.checked("dynamodb", "UpdateItem", values)
                    assert values.get("ConditionExpression"), (
                        "completion needs an actual conditional checkpoint write"
                    )
                    state.rejected_cas.append(copy.deepcopy(values))
                    raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
                return original_update(**values)

            monkeypatch.setattr(ports, "update_item", rejected_update)
    return state


async def wait_for(state, reader, predicate, stage, *, timeout=60):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate() and not reader.done():
        assert asyncio.get_running_loop().time() < deadline, "ordinary source did not reach " + stage
        await asyncio.sleep(0.005)
    terminal = state.events[-1] if state.events else {}
    response = terminal.get("response") or {}
    response_data = response.get("data") or {}
    admission = response_data.get("input_admission") or {}
    guardrail = response_data.get("guardrail_outcome") or {}
    reader_error = reader.exception() if reader.done() and not reader.cancelled() else None
    preparation = next(
        (value for use, value in reversed(state.work.originals) if use == "cold-original-preparation"),
        {},
    )
    preparation_status = {}
    for key in ("status", "error_type", "error_code", "code"):
        value = preparation.get(key)
        preparation_status[key] = (
            value if type(value) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,95}", value) else None
        )
    for key in ("success", "executed", "accepted"):
        value = preparation.get(key)
        preparation_status[key] = value if type(value) is bool else None
    decision = preparation.get("decision")
    decision = decision.get("decision") if type(decision) is dict else None
    preparation_status["decision"] = decision if decision in ("approved", "denied") else None
    assert predicate(), {
        "ordinary_source_prerequisite": stage,
        "worker_done": reader.done(),
        "model_calls": state.model.calls,
        "original_result_count": len(state.work.originals),
        "preparation_status": preparation_status,
        "preparation_trace": state.preparation_trace,
        "pending_phase_trace": state.pending_phase_trace,
        "s3_key_errors": state.objects.key_errors,
        "terminal": {key: terminal.get(key) for key in ("event", "status", "error_type", "worker_finished")},
        "response_action": response.get("action"),
        "input_admission": {
            key: admission.get(key) for key in ("status", "data_only", "native_admission", "quality_success")
        },
        "guardrail_action": guardrail.get("action"),
        "reader_error_type": type(reader_error).__name__ if reader_error is not None else None,
    }


def publish_data(state, record):
    """Native completed data response for the exact code-generated original."""
    handle, data = record.handle, state.data
    populate(handle, data.s3)
    prefix = handle.extra["output_prefix"]
    raw = state.training if handle.extra["repo_id"] == commitpack_profile.DATASET_REPO else state.case_raw
    extension = "jsonl" if handle.extra["filename"].endswith(".jsonl") else "parquet"
    data_path = "acquisition-test/data." + extension
    receipt = json.loads(data.s3.objects[prefix + "ACQUIRED.json"])
    lineage = json.loads(data.s3.objects[prefix + receipt["lineage"]])
    committed = json.loads(data.s3.objects[prefix + receipt["commit_manifest"]])
    for value in (receipt, lineage):
        value.update(
            repo_id=handle.extra["repo_id"],
            revision=handle.extra["revision"],
            filename=handle.extra["filename"],
        )
    receipt.update(runtime_session_id=handle.extra["runtime_session_id"], data=data_path)
    lineage.update(
        source_uri="https://huggingface.co/datasets/"
        + handle.extra["repo_id"]
        + "/resolve/"
        + handle.extra["revision"]
        + "/"
        + handle.extra["filename"],
        sha256=sha(raw),
        byte_count=len(raw),
        format=extension,
        local_path="/opt/ml/processing/output/" + data_path,
        artifact_uri="/opt/ml/processing/output/" + data_path,
    )
    lineage_raw = encoded(lineage)
    committed.update(data="data." + extension, sha256=sha(raw), lineage_sha256=sha(lineage_raw))
    committed_raw = encoded(committed)
    receipt["commit_manifest_sha256"] = sha(committed_raw)
    for name, body in (
        ("ACQUIRED.json", encoded(receipt)),
        (receipt["lineage"], lineage_raw),
        (receipt["commit_manifest"], committed_raw),
        (data_path, raw),
    ):
        data.s3.objects[prefix + name] = body
        state.objects.install(prefix + name, body, "fixture-version")
    data.s3.heads[prefix + data_path] = {"ContentLength": len(raw), "VersionId": "fixture-version"}
    data.native.states[handle.id] = "Completed"


def publish_preparation(state, handle):
    source = handle.extra["sources"][0]
    artifact = source.inspection_source.artifact_ref
    parsed = [
        commitpack_preparation.CommitPackParsedRow(
            json.loads(line),
            commitpack_profile.CommitPackSource(
                source.repo_id,
                source.revision,
                source.filename,
                artifact.uri,
                artifact.version_id,
                artifact.sha256,
                index,
            ),
            sha(line + b"\n"),
        )
        for index, line in enumerate(state.training.splitlines())
    ]
    exclusions = json.loads(
        state.objects.objects[handle.extra["catalog"].exclusions_ref.uri.split("/", 3)[3]][1]
    )
    prepared = commitpack_preparation.build_commitpack_preparation(parsed, exclusion_policy=exclusions)
    assert prepared["status"] == "complete", "native preparation fixture must preserve genuine source replay"
    assert prepared["report"]["counts"]["split_counts"]["train"] >= 1000
    files = {}
    values = {
        name + ".jsonl": b"".join(encoded(row) for row in rows) for name, rows in prepared["splits"].items()
    }
    values.update(
        {
            "membership.jsonl": b"".join(encoded(row) for row in prepared["membership"]),
            "lineage.jsonl": b"".join(encoded(row) for row in prepared["lineage"]),
            "REPORT.json": encoded(prepared["report"]),
        }
    )
    for name, raw in values.items():
        state.objects.install(handle.extra["output_prefix"] + name, raw)
        files[name] = {
            "byte_count": len(raw),
            "sha256": sha(raw),
            "format": "json" if name.endswith(".json") else "jsonl",
            "record_count": len(raw.splitlines()),
        }
    committed = {
        "schema_version": 2,
        **handle.extra["header"],
        "privilege_profile": "sagemaker_native_ceiling_v1",
        "selection_sha256": handle.extra["selection_sha256"],
        "exclusions_sha256": handle.extra["exclusions_sha256"],
        "files": files,
        "counts": prepared["report"]["counts"],
        "coverage": prepared["report"]["coverage"],
        "guest_evidence": {
            "privilege_proof": PROOF,
            "credential_observation": "env_and_home_file_presence",
            "network_observation": "fixed_public_and_metadata_tcp_denial",
        },
    }
    state.objects.install(handle.extra["output_prefix"] + "COMMITTED.json", encoded(committed))
    state.native.states[handle.id] = "Completed"


async def prepare_original_submission(state, monkeypatch):
    """Drive only the exact fresh upstream path, leaving SFT outcome unrequired."""
    received = []

    async def read():
        async for event in runtime.agent_invocation({"prompt": state.work.literal}):
            received.append(event)
        return received

    reader = asyncio.create_task(read())
    state.reader, state.events = reader, received
    await wait_for(
        state,
        reader,
        lambda: (
            state.work.engines
            and len(state.work.engines[0].tool_backend._acquisition_store._records) == 2
            and state.work.engines[0].tool_backend._model_acquisition_store._records
        ),
        "original model and two acquired data records",
    )
    backend = state.work.engines[0].tool_backend
    model_record = next(iter(backend._model_acquisition_store._records.values()))
    complete_fixture(
        monkeypatch,
        state.directory / "native-model",
        SimpleNamespace(handle=model_record.handle, adapter=state.work.adapter, native=state.work.native),
    )
    for key, (version, raw) in state.work.s3.objects.items():
        state.objects.install(key, raw, version)
    for record in backend._acquisition_store._records.values():
        publish_data(state, record)
    await wait_for(state, reader, lambda: backend._case_freeze_store._records, "original case native record")
    case_handle = next(iter(backend._case_freeze_store._records.values()))
    bundle = freeze_managed_case_bundle(
        official_case_data()[1], source=case_handle.extra["source"], header=case_handle.extra["header"]
    )
    for name, raw in bundle.items():
        state.objects.install(case_handle.extra["output_prefix"] + name, raw)
    state.native.states[case_handle.id] = "Completed"
    await wait_for(
        state, reader, lambda: backend._preparation_store._records, "original preparation native record"
    )
    preparation = next(iter(backend._preparation_store._records.values())).handle
    publish_preparation(state, preparation)
    return reader, received, backend


async def original_precommit(state, monkeypatch):
    """Run the real prompt to accepted SFT, keeping its original result uncommitted."""
    reader, received, backend = await prepare_original_submission(state, monkeypatch)
    await wait_for(
        state,
        reader,
        lambda: (
            backend._sft_store._records
            and any(value.kind == "sft" for value in backend._warm_pending_producers.values())
        ),
        "accepted PRECOMMIT original SFT",
    )
    await wait_for(state, reader, state.task_published.is_set, "successful admitted TASK publication")
    agent = state.work.agents[0]
    original = next(iter(backend._sft_store._records.values()))
    assert len(backend._warm_graph()) == 7 and len(state.native.calls) == 1
    assert original.extra["submission_state"] == "accepted"
    assert original.extra["recipe_id"] == state.model.original_recipe_id
    assert SFT_USE not in payloads(agent.messages), (
        "freeze boundary must precede original SFT result history append"
    )
    assert any(
        block.get("toolUse", {}).get("toolUseId") == SFT_USE for block in agent.messages[-1]["content"]
    )
    state.backend, state.agent, state.original = backend, agent, original
    state.run_id = next(event["run_id"] for event in received if event["event"] == "run_started")
    state.prefix = copy.deepcopy(agent.messages)
    return state


def cloud_data(state, *, completed=False):
    """Transport DATA only; never export SDK RAM/history/stores/policy/witnesses."""
    if completed:
        sft_fixture.install_output(
            state.adapters[QwenSftBackend],
            state.native,
            state.objects,
            state.original,
            recipe=state.original.extra["recipe_id"],
        )
    processing = {
        **copy.deepcopy(state.native.processing),
        **copy.deepcopy(state.data.native.requests),
        **{value["ProcessingJobName"]: copy.deepcopy(value) for value in state.work.native.requests},
    }
    states = {
        **copy.deepcopy(state.native.states),
        **copy.deepcopy(state.data.native.states),
        **{value["ProcessingJobName"]: state.work.native.state for value in state.work.native.requests},
    }
    states[state.original.id] = "Completed" if completed else "InProgress"
    return {
        "run_id": state.run_id,
        "ddb": copy.deepcopy(state.ports.items),
        "checkpoint_objects": [
            {"key": key, "version": version, "raw": raw}
            for (key, version), raw in state.ports.objects.items()
        ],
        "stage_objects": [
            {"key": key[0], "version": key[1], "raw": raw}
            for key, raw in state.objects.objects.items()
            if type(key) is tuple
        ],
        "training_requests": copy.deepcopy(state.native.calls),
        "processing_requests": processing,
        "states": states,
        "original_job": state.original.id,
        **({"unselected": sorted(state.native.unselected)} if state.native.unselected else {}),
    }


async def finish_generation_pair(state, reader, *, report_timeout=60, deadline=None):
    """Only native completion outputs; all successor submits remain in SDK tools."""
    backend = state.work.engines[-1].tool_backend
    generation_wait = (
        {} if deadline is None else {"timeout": max(0, deadline - asyncio.get_running_loop().time())}
    )
    await wait_for(
        state,
        reader,
        lambda: len(backend._generation_store._records) == 2,
        "two actual registered generation originals",
        **generation_wait,
    )
    for handle in tuple(backend._generation_store._records.values()):
        ref = handle.extra["run_binding"]["prompts_ref"]
        uri = ref["uri"].split("/", 3)
        response = state.objects.get_object(
            Bucket=uri[2],
            Key=uri[3],
            VersionId=ref["version_id"],
            ExpectedBucketOwner=handle.extra["catalog"].account_id,
        )
        body = response["Body"]
        try:
            raw = body.read(ref["byte_count"] + 1)
        finally:
            body.close()
        assert (
            response.get("VersionId") == ref["version_id"]
            and response.get("ContentLength") == ref["byte_count"]
            and len(raw) == ref["byte_count"]
            and hashlib.sha256(raw).hexdigest() == ref["sha256"]
        ), "generation_original_prompt_artifact_differs"
        prompts = [json.loads(line) for line in raw.splitlines()]
        assert len(prompts) == 164, "generation_original_prompt_count_differs"
        publish_generation(state.objects, handle, prompts)
        state.native.states[handle.id] = "Completed"
    pair_wait = {} if deadline is None else {"timeout": max(0, deadline - asyncio.get_running_loop().time())}
    await wait_for(
        state, reader, lambda: backend._pair_store._records, "actual registered paired original", **pair_wait
    )
    handle = next(iter(backend._pair_store._records.values()))
    publish_pair(state, handle)
    remaining = report_timeout if deadline is None else max(0, deadline - asyncio.get_running_loop().time())
    return await asyncio.wait_for(reader, remaining)


def publish_pair(state, handle):
    """Existing 328-receipt neutral provider output; no candidate/oracle execution."""
    from test_paired_coding_runner import population

    from amlas.arbiter.coding import CodingCasePin, CodingEvaluatorFactory
    from amlas.evaluation.coding_oracle_catalog import EXPECTED_CASES
    from amlas.evaluation.coding_pair import run_paired
    from amlas.training.coding_generation import canonical_bytes

    cases, candidates = population()
    cases = tuple(
        {**case, "oracle_sha256": EXPECTED_CASES[index]["oracle_record_sha256"]}
        for index, case in enumerate(cases)
    )
    candidates = {
        arm: tuple(
            {**row, "source": None, "source_sha256": None, "outcome": "generation_error"} for row in rows
        )
        for arm, rows in candidates.items()
    }

    class NoExecution:
        def __getattr__(self, name):
            raise AssertionError("neutral unresolved candidate must not execute")

    root = state.directory / "native-paired-output"
    root.mkdir(exist_ok=True)
    binding = handle.extra["evaluation_binding"]
    result = run_paired(
        binding=binding,
        cases=cases,
        candidates=candidates,
        client=NoExecution(),
        cleanup_client=NoExecution(),
        evidence_root=root,
        clock=lambda: 1.0,
        sleep=lambda _seconds: None,
    )
    assert len(result["receipts"]) == 328 and result["cleanup_complete"] is True
    pins = tuple(
        CodingCasePin(case_id=row["case_id"], oracle_sha256=row["oracle_record_sha256"])
        for row in EXPECTED_CASES
    )
    report = CodingEvaluatorFactory().build_report(
        binding=binding, case_pins=pins, receipts=result["receipts"]
    )
    receipts = b"".join(canonical_bytes(row.model_dump(mode="json")) + b"\n" for row in result["receipts"])
    trace_raw = b"\n"
    trace = {"sha256": sha(trace_raw), "byte_count": 1, "format": "jsonl", "record_count": 0}
    evaluation = {
        "schema_version": 1,
        "operation": "paired_eval",
        "source_commit": sft_fixture.SOURCE,
        "binding": binding.model_dump(mode="json"),
        "case_count": 164,
        "receipt_count": 328,
        "quality_success": False,
        "native_admission": False,
        "cleanup_complete": True,
        "summary": report.describe(),
        "invocations_manifest": trace,
    }
    values = {
        "RECEIPTS.jsonl": receipts,
        "EVALUATION.json": canonical_bytes(evaluation) + b"\n",
        "INVOCATIONS_MANIFEST.jsonl": trace_raw,
    }
    files = {
        name: {
            "sha256": sha(raw),
            "byte_count": len(raw),
            "format": "jsonl" if name.endswith(".jsonl") else "json",
        }
        for name, raw in values.items()
        if name != "INVOCATIONS_MANIFEST.jsonl"
    }
    values["COMMITTED.json"] = (
        canonical_bytes(
            {
                "schema_version": 1,
                "operation": "paired_eval",
                "source_commit": sft_fixture.SOURCE,
                "case_count": 164,
                "quality_success": False,
                "native_admission": False,
                "files": files,
                "invocations_manifest": trace,
            }
        )
        + b"\n"
    )
    for name, raw in values.items():
        state.objects.install(handle.extra["output_prefix"] + name, raw)
    state.native.states[handle.id] = "Completed"


async def close(state):
    if hasattr(state, "reader"):
        await close_original_worker(state.work, state.reader)
    state.generator.close()
