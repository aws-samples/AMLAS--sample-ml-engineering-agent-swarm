"""Fixed source-owned Qwen SFT; no model-supplied grants or executable loaders."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from amlas.data_eng.model_weight_manifest import expected_model_weight_manifest

TEMPLATE_SHA256 = (
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"  # pragma: allowlist secret
)
NORMALIZATION_VERSION = "qwen_tool_arguments_v1"
RELOAD_CONTRACT_VERSION = "qwen3_5_9b.safe_adapter_reload.v1"
MAX_SEQUENCE_TOKENS = 4096

FIXED_STAGES = frozenset(
    {
        "input_verify",
        "base_input_verify",
        "ml_import",
        "gpu_probe",
        "tokenizer_load",
        "train_tokenize",
        "validation_tokenize",
        "model_load",
        "adapter_attach",
        "optimizer_train",
        "greedy_probe",
        "checkpoint_save",
        "reload_load",
        "reload_generate",
        "training_receipt",
        "final_readback",
        "final_publication",
    }
)


CHECKPOINT_SUBSTAGES = frozenset(
    {
        "evidence",
        "output_authority",
        "output_empty",
        "framework_save",
        "model_card",
        "file_inventory",
        "file_readback",
        "tensor_header",
        "checkpoint_schema",
        "receipt_commit",
    }
)


class CheckpointSaveError(ValueError):
    """Closed source-owned save substage; no exception text or authority."""

    def __init__(self, substage: str, cause_type: str):
        if substage not in CHECKPOINT_SUBSTAGES or not cause_type.isidentifier() or len(cause_type) > 64:
            raise ValueError("fixed checkpoint diagnostic required")
        self.checkpoint_substage = substage
        self.checkpoint_reason = substage + "_refused"
        self.cause_type = cause_type
        super().__init__("fixed checkpoint save refused")


class FixedStageError(RuntimeError):
    """Source-owned stage and class only; native exception text is never logged."""

    def __init__(self, stage: str, cause_type: str) -> None:
        if stage not in FIXED_STAGES or not cause_type.isidentifier() or len(cause_type) > 64:
            raise ValueError("fixed diagnostic stage/class required")
        self.stage = stage
        self.cause_type = cause_type
        super().__init__("fixed operation refused")


def fixed_runtime_fingerprint(torch_module) -> dict[str, str]:
    def version(value):
        if not isinstance(value, str):
            return "unknown"
        plain = str.__str__(value)
        return (
            plain
            if re.fullmatch(r"[0-9]{1,3}\.[0-9]{1,3}(?:\.[0-9]{1,3})?(?:\+cu[0-9]{2,4})?", plain)
            else "unknown"
        )

    return {
        "torch_version": version(getattr(torch_module, "__version__", None)),
        "cuda_version": version(getattr(getattr(torch_module, "version", None), "cuda", None)),
    }


def run_fixed_stage(stage: str, function):
    if type(stage) is not str or stage not in FIXED_STAGES:
        raise ValueError("fixed diagnostic stage required")
    try:
        return function()
    except Exception as error:
        if type(error).__name__ == "ConversationCompatibilityError":
            raise
        wrapped = FixedStageError(
            stage, error.cause_type if type(error) is CheckpointSaveError else type(error).__name__
        )
        if stage == "checkpoint_save" and type(error) is CheckpointSaveError:
            wrapped.checkpoint_substage = error.checkpoint_substage
            wrapped.checkpoint_reason = error.checkpoint_reason
        raise wrapped from None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _bounded_json(value: Any) -> None:
    pending = [(value, 0)]
    count = 0
    while pending:
        part, depth = pending.pop()
        count += 1
        if count > 500000 or depth > 64:
            raise ValueError("conversation JSON bound exceeded")
        kind = type(part)
        if kind is dict:
            if any(type(key) is not str for key in part):
                raise ValueError("conversation key differs")
            pending.extend((child, depth + 1) for pair in part.items() for child in pair)
        elif kind is list:
            pending.extend((child, depth + 1) for child in part)
        elif kind is float:
            if not math.isfinite(part):
                raise ValueError("nonfinite conversation value")
        elif kind not in (str, int, bool, type(None)):
            raise ValueError("conversation object differs")
    if len(_json(value).encode()) > 32 * 1024 * 1024:
        raise ValueError("conversation bytes exceeded")


def _argument_object(text: Any) -> dict[str, Any]:
    if type(text) is not str or len(text.encode()) > 1024 * 1024:
        raise ValueError("tool argument encoding differs")
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("tool argument depth exceeded")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate tool argument")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite tool argument")

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except (RecursionError, TypeError):
        raise ValueError("tool argument refused") from None
    if type(result) is not dict:
        raise ValueError("tool argument mapping required")
    _bounded_json(result)
    return result


class ConversationCompatibilityError(ValueError):
    def __init__(self, reason_counts: dict[str, int]):
        if (
            not reason_counts
            or set(reason_counts) - {"developer_role", "historical_reasoning_unsupported"}
            or any(type(count) is not int or count <= 0 for count in reason_counts.values())
        ):
            raise ValueError("conversation compatibility reason differs")
        self.reason_counts = dict(reason_counts)
        super().__init__("model_template_compatibility: " + ",".join(sorted(reason_counts)))


def normalize_row(value: Any) -> dict[str, Any]:
    _bounded_json(value)
    if type(value) is not dict or set(value) != {"messages", "tools"}:
        raise ValueError("conversation row fields differ")
    if type(value["messages"]) is not list or not value["messages"] or type(value["tools"]) is not list:
        raise ValueError("conversation row shape differs")
    if len(value["messages"]) > 16384 or len(value["tools"]) > 256:
        raise ValueError("conversation row count exceeded")
    result = json.loads(_json(value))
    names = set()
    for tool in result["tools"]:
        if type(tool) is not dict or set(tool) != {"type", "function"} or tool["type"] != "function":
            raise ValueError("tool schema differs")
        fn = tool["function"]
        if (
            type(fn) is not dict
            or not set(fn) <= {"name", "parameters", "description", "strict"}
            or type(fn.get("name")) is not str
            or not fn["name"]
            or type(fn.get("parameters")) is not dict
        ):
            raise ValueError("tool function schema differs")
        if fn["name"] in names:
            raise ValueError("duplicate tool name")
        names.add(fn["name"])
    last_user = max(
        (
            index
            for index, message in enumerate(result["messages"])
            if type(message) is dict and message.get("role") == "user"
        ),
        default=-1,
    )
    pending = {}
    assistant_payload = False
    user_seen = False
    for index, message in enumerate(result["messages"]):
        if type(message) is not dict or not set(message) <= {
            "role",
            "content",
            "reasoning_content",
            "tool_calls",
            "tool_call_id",
            "name",
        }:
            raise ValueError("message fields differ")
        role = message.get("role")
        if role == "developer":
            raise ConversationCompatibilityError({"developer_role": 1})
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("message role differs")
        if message.get("content") is not None and type(message.get("content")) is not str:
            raise ValueError("message content differs")
        if message.get("reasoning_content") is not None and type(message.get("reasoning_content")) is not str:
            raise ValueError("message reasoning differs")
        if role == "system" and index != 0:
            raise ValueError("system ordering differs")
        if role == "user":
            user_seen = True
        calls = message.get("tool_calls")
        if calls is not None:
            if role != "assistant" or type(calls) is not list:
                raise ValueError("tool calls differ")
            for call in calls:
                if (
                    type(call) is not dict
                    or set(call) != {"id", "type", "function"}
                    or call["type"] != "function"
                    or type(call["id"]) is not str
                    or not call["id"]
                    or call["id"] in pending
                ):
                    raise ValueError("tool call identity differs")
                fn = call["function"]
                if type(fn) is not dict or set(fn) != {"name", "arguments"} or fn["name"] not in names:
                    raise ValueError("tool call function differs")
                fn["arguments"] = _argument_object(fn["arguments"])
                pending[call["id"]] = fn["name"]
        if role == "tool":
            identity = message.get("tool_call_id")
            if identity not in pending or message.get("name") != pending[identity]:
                raise ValueError("orphan tool response")
            pending.pop(identity)
        elif pending and role not in ("assistant",):
            raise ValueError("unanswered tool call")
        if role == "assistant":
            if index < last_user and (
                (message.get("reasoning_content") or "").strip()
                or "</think>" in (message.get("content") or "")
            ):
                raise ConversationCompatibilityError({"historical_reasoning_unsupported": 1})
            payload = bool(
                (message.get("content") or "").strip()
                or (message.get("reasoning_content") or "").strip()
                or calls
            )
            if not payload:
                raise ValueError("empty assistant supervision")
            assistant_payload = True
    if pending or not user_seen or not assistant_payload:
        raise ValueError("incomplete supervised conversation")
    return result


def instrument_template(template: str) -> str:
    if type(template) is not str or hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("official template hash differs")
    block = """        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""
    replacement = """        {{- '<|im_start|>' + message.role + '\\n' }}
        {%- generation %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- content }}
        {%- endif %}"""
    if template.count(block) != 1:
        raise ValueError("official assistant branch differs")
    result = template.replace(block, replacement)
    end = "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
    if result.count(end) != 1:
        raise ValueError("official assistant end differs")
    return result.replace(
        end,
        "        {{- '<|im_end|>\\n' }}\n        {%- endgeneration %}\n    {%- elif message.role == \"tool\" %}",
    )


def encode_supervised(row: Any, tokenizer: Any) -> dict[str, Any]:
    from transformers.utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices

    normalized = normalize_row(row)
    template = tokenizer.chat_template
    tracked = instrument_template(template)
    kwargs = {"tools": normalized["tools"], "add_generation_prompt": False}
    canonical = tokenizer.apply_chat_template(
        normalized["messages"], chat_template=template, tokenize=False, **kwargs
    )
    rendered, spans = _render_with_assistant_indices(
        _compile_jinja_template(tracked),
        normalized["messages"],
        normalized["tools"],
        None,
        False,
        **tokenizer.special_tokens_map,
    )
    if canonical.encode() != rendered.encode():
        raise ValueError("official render bytes differ")
    original = tokenizer.apply_chat_template(
        normalized["messages"], chat_template=template, return_dict=True, **kwargs
    )["input_ids"]
    encoded = tokenizer.apply_chat_template(
        normalized["messages"],
        chat_template=tracked,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    measured = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = measured["offset_mapping"]
    if ids != original or ids != measured["input_ids"] or len(ids) > MAX_SEQUENCE_TOKENS:
        raise ValueError("token equality or truncation refusal")
    masks = [int(end > start and any(a <= start and end <= b for a, b in spans)) for start, end in offsets]
    if masks != encoded["assistant_masks"] or not any(masks):
        raise ValueError("actual assistant spans differ or empty")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [token if active else -100 for token, active in zip(ids, masks, strict=True)],
        "rendered": rendered,
        "assistant_char_spans": spans,
        "normalization_version": NORMALIZATION_VERSION,
        "original_row_sha256": hashlib.sha256(_json(row).encode()).hexdigest(),
    }


def pad_batch(rows: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    if not rows or type(pad_token_id) is not int:
        raise ValueError("batch padding requires fixed tokenizer")
    width = max(len(row["input_ids"]) for row in rows)
    if width > MAX_SEQUENCE_TOKENS:
        raise ValueError("batch sequence exceeded")
    return {
        key: torch.tensor([row[key] + [fill] * (width - len(row[key])) for row in rows], dtype=torch.long)
        for key, fill in [("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)]
    }


def recipe(identity: str) -> dict[str, Any]:
    if identity not in ("fit_v1", "full_v1"):
        raise ValueError("fixed recipe identity differs")
    return {
        "recipe_id": identity,
        "optimizer_steps": 32 if identity == "fit_v1" else 1024,
        "minimum_supervised_tokens_processed": 0 if identity == "fit_v1" else 1000000,
        "minimum_unique_corpus_examples": 0 if identity == "fit_v1" else 1000,
        "minimum_unique_supervised_tokens": 0 if identity == "fit_v1" else 1000000,
        "micro_batch_size": 1,
        "gradient_accumulation": 8,
        "seed": 12345,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "learning_rate": 0.0002,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "quality_success": False,
    }


LANGUAGE_LINEAR_SUFFIXES = frozenset(
    {
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


def attach_language_lora(model: Any) -> Any:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration

    if not isinstance(model, Qwen3_5ForConditionalGeneration):
        raise ValueError("fixed native model class required")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = [
        name
        for name, child in model.named_modules()
        if name.startswith("model.language_model.layers.")
        and name.rsplit(".", 1)[-1] in LANGUAGE_LINEAR_SUFFIXES
        and type(child) is torch.nn.Linear
    ]
    if not targets:
        raise ValueError("fixed language linears missing")
    torch.manual_seed(12345)
    configuration = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
        use_dora=False,
        use_rslora=False,
    )
    wrapped = get_peft_model(model, configuration)
    selected = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    if not selected or any("language_model.layers." not in name or "lora_" not in name for name in selected):
        raise ValueError("nonlanguage trainable tensor refused")
    return wrapped


def optimizer_update(model: Any, optimizer: Any, batches: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not batches:
        raise ValueError("empty optimizer accumulation")
    counts = [
        int(((batch["labels"][:, 1:] != -100) & (batch["attention_mask"][:, 1:] != 0)).sum().item())
        for batch in batches
    ]
    total = sum(counts)
    if total <= 0 or any(count <= 0 for count in counts):
        raise ValueError("empty shifted supervision")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_sum = 0.0
    for batch, count in zip(batches, counts, strict=True):
        loss = model(**batch, use_cache=False).loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise ValueError("nonfinite training loss")
        weighted = loss * (count / total)
        weighted.backward()
        loss_sum += float(loss.detach().item()) * count
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    ):
        raise ValueError("nonfinite training gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise ValueError("nonfinite gradient norm")
    optimizer.step()
    return {
        "optimizer_steps": 1,
        "supervised_tokens": total,
        "loss": loss_sum / total,
        "gradient_norm": float(norm.item()),
    }


def _checkpoint_json(value: Any) -> bytes:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > 262144:
        raise ValueError("checkpoint metadata exceeded")
    return raw


def _fd_bytes(descriptor: int, limit: int, *, collect: bool = False) -> tuple[dict[str, Any], bytes | None]:
    import os
    import stat
    import time

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= limit:
        raise ValueError("checkpoint regular file/size refused")
    digest = hashlib.sha256()
    count = 0
    parts = []
    deadline = time.monotonic() + 300
    while count <= before.st_size:
        if time.monotonic() >= deadline:
            raise ValueError("checkpoint hash deadline exceeded")
        chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - count + 1), count)
        if not chunk:
            break
        count += len(chunk)
        if count > before.st_size:
            raise ValueError("checkpoint byte count exceeded")
        digest.update(chunk)
        if collect:
            parts.append(chunk)
    after = os.fstat(descriptor)

    def signature(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if signature(before) != signature(after) or count != before.st_size:
        raise ValueError("checkpoint file changed during hash")
    return {"sha256": digest.hexdigest(), "byte_count": count}, b"".join(parts) if collect else None


def _checkpoint_named(directory: int, descriptor: int, name: str) -> None:
    import os
    import stat

    value = os.fstat(descriptor)
    named = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if (value.st_dev, value.st_ino) != (named.st_dev, named.st_ino) or not stat.S_ISREG(named.st_mode):
        raise ValueError("checkpoint original name changed")


def _checkpoint_root_named(directory: int, path: Any) -> None:
    import os

    named = _open_directory_original(path)
    try:
        if (os.fstat(directory).st_dev, os.fstat(directory).st_ino) != (
            os.fstat(named).st_dev,
            os.fstat(named).st_ino,
        ):
            raise ValueError("checkpoint original root changed")
    finally:
        os.close(named)


def _hash_file(path: Any, limit: int) -> dict[str, Any]:
    import os
    from pathlib import Path

    path = Path(path)
    directory = _open_directory_original(path.parent)
    descriptor = None
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        value, _ = _fd_bytes(descriptor, limit)
        _checkpoint_named(directory, descriptor, path.name)
        _checkpoint_root_named(directory, path.parent)
        return value
    except OSError:
        raise ValueError("checkpoint original file refused") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _checkpoint_evidence(value: Any) -> dict[str, Any]:
    import re

    required = {"recipe_id", "optimizer_steps", "supervised_tokens", "greedy_input_ids", "greedy_output_ids"}
    optional = {"source_commit", "stage_manifest_sha256"}
    if type(value) is not dict or not required <= set(value) or set(value) - required - optional:
        raise ValueError("checkpoint evidence fields differ")
    recipe(value["recipe_id"])
    for name in ("optimizer_steps", "supervised_tokens"):
        if type(value[name]) is not int or not 0 <= value[name] <= 10**12:
            raise ValueError("actual checkpoint counter required")
    query, output = value["greedy_input_ids"], value["greedy_output_ids"]
    for batch, limit in ((query, 64), (output, 80)):
        if (
            type(batch) is not list
            or len(batch) != 1
            or type(batch[0]) is not list
            or not 0 < len(batch[0]) <= limit
        ):
            raise ValueError("checkpoint greedy token shape differs")
        if any(type(token) is not int or not 0 <= token < 262144 for token in batch[0]):
            raise ValueError("checkpoint greedy token ID refused")
    if output[0][: len(query[0])] != query[0] or not 1 <= len(output[0]) - len(query[0]) <= 16:
        raise ValueError("checkpoint greedy generation bound differs")
    result = dict(value)
    for name, length in (("source_commit", 40), ("stage_manifest_sha256", 64)):
        part = value.get(name)
        if part is not None and (
            type(part) is not str or not re.fullmatch("[a-f0-9]{" + str(length) + "}", part)
        ):
            raise ValueError("checkpoint source provenance differs")
        result[name] = part
    return result


def _checkpoint_metadata(value: Any) -> dict[str, Any]:
    import re

    fixed = {
        "schema_version",
        "catalog_id",
        "catalog_version",
        "base_revision",
        "template_sha256",
        "normalization_version",
        "tensor_keys",
        "files",
        "status",
        "reload_pending",
        "quality_success",
    }
    evidence_keys = {
        "recipe_id",
        "optimizer_steps",
        "supervised_tokens",
        "greedy_input_ids",
        "greedy_output_ids",
        "source_commit",
        "stage_manifest_sha256",
    }
    if type(value) is not dict or set(value) != fixed | evidence_keys:
        raise ValueError("checkpoint metadata fields differ")
    _checkpoint_evidence({name: value[name] for name in evidence_keys})
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["catalog_id"] != "qwen3_5_9b"
        or value["catalog_version"] != "1"
        or value["base_revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # pragma: allowlist secret
        or value["template_sha256"] != TEMPLATE_SHA256
        or value["normalization_version"] != NORMALIZATION_VERSION
        or value["status"] != "checkpoint_saved"
        or value["quality_success"] is not False
        or value["reload_pending"] is not True
    ):
        raise ValueError("checkpoint source binding differs")
    files = value["files"]
    if type(files) is not dict or set(files) != {"adapter_model.safetensors", "adapter_config.json"}:
        raise ValueError("checkpoint file fields differ")
    for name, ref in files.items():
        limit = 2 * 1024**3 if name.endswith(".safetensors") else 262144
        if (
            type(ref) is not dict
            or set(ref) != {"sha256", "byte_count"}
            or type(ref["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", ref["sha256"])
            or type(ref["byte_count"]) is not int
            or not 0 < ref["byte_count"] <= limit
        ):
            raise ValueError("checkpoint file descriptor differs")
    keys = value["tensor_keys"]
    if type(keys) is not dict or not 0 < len(keys) <= 8192:
        raise ValueError("checkpoint tensor inventory differs")
    for name, descriptor in keys.items():
        if (
            type(name) is not str
            or len(name) > 256
            or "language_model.layers." not in name
            or not name.endswith((".lora_A.weight", ".lora_B.weight"))
            or type(descriptor) is not dict
            or set(descriptor) != {"shape", "dtype"}
            or descriptor["dtype"] not in ("F32", "BF16")
            or type(descriptor["shape"]) is not list
            or len(descriptor["shape"]) != 2
            or any(type(size) is not int or not 0 < size <= 262144 for size in descriptor["shape"])
        ):
            raise ValueError("checkpoint tensor keyshape differs")
        shape = descriptor["shape"]
        if (
            (name.endswith(".lora_A.weight") and shape[0] != 16)
            or (name.endswith(".lora_B.weight") and shape[1] != 16)
            or max(shape) > 16384
        ):
            raise ValueError("checkpoint fixed LoRA rank/dimension differs")
    return value


def save_adapter_checkpoint(
    model: Any, output: Any, evidence: dict[str, Any], *, output_fd: int | None = None
) -> dict[str, Any]:
    import os
    import stat
    from pathlib import Path

    from safetensors import safe_open

    try:
        evidence = _checkpoint_evidence(evidence)
    except Exception as error:
        raise CheckpointSaveError("evidence", type(error).__name__) from None
    output = Path(output)
    try:
        directory = _open_directory_original(output) if output_fd is None else os.dup(output_fd)
    except Exception as error:
        raise CheckpointSaveError("output_authority", type(error).__name__) from None
    descriptors = {}
    substage = "output_authority"
    try:
        _checkpoint_root_named(directory, output)
        substage = "output_empty"
        if os.listdir(directory):
            raise ValueError("checkpoint requires empty fixed directory")
        # The library receives only the already-opened image-owned output authority.
        alias = "/proc/self/fd/" + str(directory)
        substage = "framework_save"
        model.save_pretrained(alias, safe_serialization=True, save_embedding_layers=False)
        substage = "model_card"
        if "README.md" in os.listdir(directory):
            readme = os.stat("README.md", dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(readme.st_mode) or readme.st_nlink != 1 or readme.st_uid != os.geteuid():
                raise ValueError("framework model-card file refused")
            os.unlink("README.md", dir_fd=directory)
        substage = "file_inventory"
        if set(os.listdir(directory)) != {"adapter_model.safetensors", "adapter_config.json"}:
            raise ValueError("unexpected framework checkpoint files")
        substage = "file_readback"
        files = {}
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            descriptors[name] = descriptor
            os.fchmod(descriptor, 0o600)
            files[name], _ = _fd_bytes(descriptor, 2 * 1024**3 if name.endswith(".safetensors") else 262144)
            _checkpoint_named(directory, descriptor, name)
        substage = "tensor_header"
        keys = {}
        with safe_open(
            "/proc/self/fd/" + str(descriptors["adapter_model.safetensors"]), framework="pt", device="cpu"
        ) as tensors:
            for name in tensors.keys():
                tensor = tensors.get_slice(name)
                keys[name] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype()}
        substage = "checkpoint_schema"
        metadata = _checkpoint_metadata(
            {
                "schema_version": 1,
                "catalog_id": "qwen3_5_9b",
                "catalog_version": "1",
                "base_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",  # pragma: allowlist secret
                "template_sha256": TEMPLATE_SHA256,
                "normalization_version": NORMALIZATION_VERSION,
                **evidence,
                "tensor_keys": keys,
                "files": files,
                "status": "checkpoint_saved",
                "reload_pending": True,
                "quality_success": False,
            }
        )
        substage = "receipt_commit"
        raw = _checkpoint_json(metadata)
        receipt = os.open(
            "CHECKPOINT.json",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        descriptors["CHECKPOINT.json"] = receipt
        with os.fdopen(os.dup(receipt), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        receipt_expected = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
        for _ in range(2):
            for name, descriptor in descriptors.items():
                actual, _ = _fd_bytes(descriptor, 2 * 1024**3 if name.endswith(".safetensors") else 262144)
                if actual != (receipt_expected if name == "CHECKPOINT.json" else files[name]):
                    raise ValueError("checkpoint retained bytes changed")
                _checkpoint_named(directory, descriptor, name)
            _checkpoint_root_named(directory, output)
        return metadata
    except Exception as error:
        raise CheckpointSaveError(substage, type(error).__name__) from None
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        os.close(directory)


def _safe_adapter_header(descriptor: int, expected: dict[str, Any]) -> None:
    import os
    import struct

    from amlas.data_eng.inspection_schema import decode_object

    prefix = os.pread(descriptor, 8, 0)
    if len(prefix) != 8:
        raise ValueError("safetensor header prefix refused")
    count = struct.unpack("<Q", prefix)[0]
    size = os.fstat(descriptor).st_size
    if not 0 < count <= 2 * 1024**2 or 8 + count >= size:
        raise ValueError("safetensor header bound refused")
    raw = os.pread(descriptor, count, 8)
    if len(raw) != count:
        raise ValueError("safetensor header count refused")
    header = decode_object(raw, limit=2 * 1024**2)
    metadata = header.pop("__metadata__", None)
    if metadata != {"format": "pt"} or set(header) != set(expected):
        raise ValueError("safetensor fixed header fields differ")
    spans = []
    for name, value in header.items():
        if (
            type(value) is not dict
            or set(value) != {"dtype", "shape", "data_offsets"}
            or {key: value[key] for key in ("dtype", "shape")} != expected[name]
        ):
            raise ValueError("safetensor fixed header shape differs")
        offsets = value["data_offsets"]
        if type(offsets) is not list or len(offsets) != 2 or any(type(item) is not int for item in offsets):
            raise ValueError("safetensor offsets differ")
        shape = value["shape"]
        item_size = 4 if value["dtype"] == "F32" else 2
        start, end = offsets
        if not 0 <= start < end <= size - 8 - count or end - start != shape[0] * shape[1] * item_size:
            raise ValueError("safetensor storage span refused")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("safetensor overlapping or missing span refused")
        cursor = end
    if cursor != size - 8 - count:
        raise ValueError("safetensor trailing bytes refused")


def reload_adapter_checkpoint(
    base_model: Any, adapter: Any, *, expected_ref: dict[str, Any] | None = None
) -> Any:
    import os
    from pathlib import Path

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration

    if not isinstance(base_model, Qwen3_5ForConditionalGeneration):
        raise ValueError("fixed native reload class required")
    adapter = Path(adapter)
    directory = _open_directory_original(adapter)
    descriptors = {}
    try:
        wanted = {"adapter_model.safetensors", "adapter_config.json", "CHECKPOINT.json"}
        if set(os.listdir(directory)) != wanted:
            raise ValueError("checkpoint inventory differs")
        refs, raws = {}, {}
        for name in wanted:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            descriptors[name] = descriptor
            refs[name], raw = _fd_bytes(
                descriptor,
                2 * 1024**3 if name.endswith(".safetensors") else 262144,
                collect=not name.endswith(".safetensors"),
            )
            raws[name] = raw
            _checkpoint_named(directory, descriptor, name)
        value = _checkpoint_metadata(_decode_plain_metadata(raws["CHECKPOINT.json"]))
        if expected_ref is not None and refs["CHECKPOINT.json"] != {
            key: expected_ref.get(key) for key in ("sha256", "byte_count")
        }:
            raise ValueError("externally pinned checkpoint differs from consumed metadata")
        vocabulary = base_model.get_input_embeddings().num_embeddings
        if any(
            token >= vocabulary
            for batch in (value["greedy_input_ids"], value["greedy_output_ids"])
            for row in batch
            for token in row
        ):
            raise ValueError("checkpoint probe token outside native vocabulary")
        _safe_adapter_header(descriptors["adapter_model.safetensors"], value["tensor_keys"])
        for name in value["files"]:
            if refs[name] != value["files"][name]:
                raise ValueError("checkpoint exact bytes differ")
        alias = "/proc/self/fd/" + str(descriptors["adapter_model.safetensors"])
        with safe_open(alias, framework="pt", device="cpu") as tensors:
            actual = {
                name: {
                    "shape": tensors.get_slice(name).get_shape(),
                    "dtype": tensors.get_slice(name).get_dtype(),
                }
                for name in tensors.keys()
            }
        if actual != value["tensor_keys"]:
            raise ValueError("checkpoint tensor keyshape differs")
        config = _decode_plain_metadata(raws["adapter_config.json"])
        targets = config.get("target_modules")
        if (
            config.get("peft_type") != "LORA"
            or config.get("task_type") != "CAUSAL_LM"
            or config.get("r") != 16
            or config.get("lora_alpha") != 32
            or config.get("lora_dropout") != 0.0
            or config.get("bias") != "none"
            or config.get("modules_to_save") is not None
            or config.get("use_dora")
            or config.get("use_rslora")
            or type(targets) is not list
            or not targets
            or any(type(name) is not str for name in targets)
            or len(targets) != len(set(targets))
            or any(name.rsplit(".", 1)[-1] not in LANGUAGE_LINEAR_SUFFIXES for name in targets)
        ):
            raise ValueError("checkpoint fixed LoRA configuration differs")
        import torch

        source_targets = {
            name
            for name, child in base_model.named_modules()
            if name.startswith("model.language_model.layers.")
            and name.rsplit(".", 1)[-1] in LANGUAGE_LINEAR_SUFFIXES
            and type(child) is torch.nn.Linear
        }
        source_suffixes = {name.rsplit(".", 1)[-1] for name in source_targets}
        if set(targets) not in (source_targets, source_suffixes):
            raise ValueError("checkpoint complete source target inventory differs")
        # PEFT condenses large saved target lists to suffixes. Serialized targets
        # only verify metadata; the current native inventory selects attachment.
        fixed_targets = sorted(source_targets)
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        fixed_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=fixed_targets,
            inference_mode=True,
        )
        model = get_peft_model(base_model, fixed_config)
        expected = get_peft_model_state_dict(model)
        if set(actual) != set(expected) or any(
            tuple(actual[name]["shape"]) != tuple(expected[name].shape) for name in actual
        ):
            raise ValueError("checkpoint native adapter header keyshape differs")
        state = load_file(alias, device="cpu")
        if set(state) != set(expected) or any(
            tuple(state[name].shape) != tuple(expected[name].shape) for name in state
        ):
            raise ValueError("checkpoint native adapter keyshape differs")
        loaded_result = set_peft_model_state_dict(model, state)
        loaded = get_peft_model_state_dict(model)
        if loaded_result.unexpected_keys or any(
            not torch.equal(loaded[name].detach().cpu(), state[name]) for name in state
        ):
            raise ValueError("checkpoint actual native adapter application differs")
        for _ in range(2):
            for name, descriptor in descriptors.items():
                measured, _ = _fd_bytes(descriptor, 2 * 1024**3 if name.endswith(".safetensors") else 262144)
                if measured != refs[name]:
                    raise ValueError("checkpoint retained bytes changed during load")
                _checkpoint_named(directory, descriptor, name)
            _checkpoint_root_named(directory, adapter)
        model.eval()
        return model
    except OSError:
        raise ValueError("checkpoint original filesystem refused") from None
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
        os.close(directory)


def _open_directory_original(path: Any) -> int:
    import os
    from pathlib import Path

    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("original absolute staging path required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except OSError:
        os.close(fd)
        raise ValueError("staging original components refused") from None


def _read_exact_relative(
    root_fd: int,
    name: str,
    ref: dict[str, Any],
    limit: int,
    read_bytes: bool,
    *,
    nonblocking: bool = False,
) -> bytes | None:
    import os
    import stat
    import time

    if (
        "/" in name
        or name in (".", "..")
        or type(ref) is not dict
        or type(ref.get("byte_count")) is not int
        or not 0 < ref["byte_count"] <= limit
        or type(ref.get("sha256")) is not str
        or len(ref["sha256"]) != 64
    ):
        raise ValueError("fixed input reference differs")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    if nonblocking:
        flags |= os.O_NONBLOCK
    fd = os.open(name, flags, dir_fd=root_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != ref["byte_count"]:
            raise ValueError("staged regular byte count differs")
        deadline = time.monotonic() + 900
        digest = hashlib.sha256()
        count = 0
        parts = []
        while True:
            if time.monotonic() >= deadline:
                raise ValueError("staging hash deadline exceeded")
            chunk = os.read(fd, min(1024 * 1024, ref["byte_count"] - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > ref["byte_count"]:
                raise ValueError("staging bytes exceeded")
            digest.update(chunk)
            if read_bytes:
                parts.append(chunk)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            count != ref["byte_count"]
            or digest.hexdigest() != ref["sha256"]
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError("staged immutable reference changed")
        return b"".join(parts) if read_bytes else None
    finally:
        os.close(fd)


def _decode_plain_metadata(raw: bytes) -> dict[str, Any]:
    from amlas.data_eng.inspection_schema import decode_object

    return decode_object(raw, limit=262144)


PREPARED_V2_PROFILE = "nvidia.open_swe_traces.v1.2.singleton_derived_ids.v2"
PREPARED_V2_POLICY = "coding-preparation.open-swe.v2"
PREPARED_V2_LINKAGE = "derived_not_original_source_verified"
_PREPARED_V2_REVISION = "1db674173119d71df45c22007af99e214600711a"  # pragma: allowlist secret
_PREPARED_V2_CARD = (
    "8fd748230ebacb5a07b01c0299b3d99c1249567f47677c1c935163f843652341"  # pragma: allowlist secret
)
_PREPARED_COVERAGE = {
    "near_dedup": "not_implemented",
    "exclusion_scope": "supplied_instance_ids_and_patch_digests_only",
}


def _prepared_digest(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _prepared_records(raw: bytes) -> list[dict[str, Any]]:
    records = []
    for line in raw.splitlines():
        if not line or len(line) > 32 * 1024**2:
            raise ValueError("derived preparation record bytes exceeded")
        depth = 0
        quoted = escaped = False
        for char in line:
            if quoted:
                if escaped:
                    escaped = False
                elif char == 92:
                    escaped = True
                elif char == 34:
                    quoted = False
            elif char == 34:
                quoted = True
            elif char in (91, 123):
                depth += 1
                if depth > 64:
                    raise ValueError("derived preparation record depth exceeded")
            elif char in (93, 125):
                depth -= 1

        def pairs(items):
            value = {}
            for key, part in items:
                if key in value:
                    raise ValueError("duplicate derived preparation record field")
                value[key] = part
            return value

        def constant(_):
            raise ValueError("finite derived preparation record required")

        try:
            value = json.loads(line, object_pairs_hook=pairs, parse_constant=constant)
        except (UnicodeError, RecursionError):
            raise ValueError("derived preparation record decode refused") from None
        _bounded_json(value)
        if type(value) is not dict:
            raise ValueError("derived preparation record object required")
        records.append(value)
    return records


def _validate_preparation_v2(committed: dict[str, Any], bodies: dict[str, bytes]) -> dict[str, Any]:
    import re
    from collections import Counter
    from urllib.parse import urlsplit

    def integer(value, positive=False):
        if type(value) is not int or not (1 if positive else 0) <= value <= 100000:
            raise ValueError("derived preparation integer bound differs")

    def sha(value):
        if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("derived preparation digest differs")

    def text(value, maximum=2048):
        if type(value) is not str or not 0 < len(value) <= maximum or any(ord(char) < 32 for char in value):
            raise ValueError("derived preparation text differs")

    counts = committed["counts"]
    if type(counts) is not dict or set(counts) != {
        "source_count",
        "retained_representatives",
        "primary_drop_count",
        "unresolved_record_count",
        "split_counts",
    }:
        raise ValueError("derived preparation count schema differs")
    for key in ("source_count", "retained_representatives", "primary_drop_count", "unresolved_record_count"):
        integer(counts[key])
    splits = counts["split_counts"]
    if type(splits) is not dict or set(splits) != {"train", "validation", "test"}:
        raise ValueError("derived preparation split schema differs")
    for count in splits.values():
        integer(count, True)
    if (
        not counts["source_count"]
        or counts["unresolved_record_count"] != 0
        or sum(splits.values()) != counts["retained_representatives"]
        or counts["source_count"] != counts["retained_representatives"] + counts["primary_drop_count"]
    ):
        raise ValueError("derived preparation original count conservation differs")
    if committed["coverage"] != _PREPARED_COVERAGE:
        raise ValueError("derived preparation coverage differs")
    report = _decode_plain_metadata(bodies["REPORT.json"])
    keys = {
        "schema_version",
        "status",
        "counts",
        "primary_drop_counts",
        "secret_pattern_counts",
        "coverage",
        "normalization_source_count",
        "normalization_accepted_count",
        "normalization_rejected_count",
        "normalization_rejection_reasons",
        "derived_edge_count",
        "linkage_status",
    }
    if (
        set(report) != keys
        or type(report["schema_version"]) is not int
        or report["schema_version"] != 2
        or report["status"] != "complete"
        or report["counts"] != counts
        or report["coverage"] != _PREPARED_COVERAGE
        or report["linkage_status"] != PREPARED_V2_LINKAGE
    ):
        raise ValueError("derived preparation report binding differs")
    report_counts = report["counts"]
    if type(report_counts) is not dict or set(report_counts) != set(counts):
        raise ValueError("derived report count fields differ")
    for key in ("source_count", "retained_representatives", "primary_drop_count", "unresolved_record_count"):
        integer(report_counts[key])
    if type(report_counts["split_counts"]) is not dict or set(report_counts["split_counts"]) != set(splits):
        raise ValueError("derived report split fields differ")
    for value in report_counts["split_counts"].values():
        integer(value, True)
    for name in (
        "normalization_source_count",
        "normalization_accepted_count",
        "normalization_rejected_count",
    ):
        integer(report[name])
    if (
        type(report["derived_edge_count"]) is not int
        or not 0 <= report["derived_edge_count"] <= 8192 * counts["source_count"]
    ):
        raise ValueError("derived preparation edge count bound differs")
    for name in ("primary_drop_counts", "secret_pattern_counts", "normalization_rejection_reasons"):
        mapping = report[name]
        if type(mapping) is not dict:
            raise ValueError("derived preparation reason map differs")
        for reason, count in mapping.items():
            if type(reason) is not str or not re.fullmatch(
                r"[a-z0-9_]{1,128}" if name == "normalization_rejection_reasons" else r"[a-z_]{1,64}", reason
            ):
                raise ValueError("derived preparation reason differs")
            integer(count, name == "normalization_rejection_reasons")
    core_drop_count = sum(report["primary_drop_counts"].values())
    if (
        report["normalization_source_count"] != counts["source_count"]
        or report["normalization_source_count"]
        != report["normalization_accepted_count"] + report["normalization_rejected_count"]
        or report["normalization_accepted_count"] != counts["retained_representatives"] + core_drop_count
        or sum(report["normalization_rejection_reasons"].values()) != report["normalization_rejected_count"]
        or core_drop_count + report["normalization_rejected_count"] != counts["primary_drop_count"]
    ):
        raise ValueError("derived preparation core/normalization conservation differs")
    lines = _prepared_records(bodies["lineage.jsonl"])
    members = _prepared_records(bodies["membership.jsonl"])
    if len(lines) != counts["source_count"] or len(members) != counts["retained_representatives"]:
        raise ValueError("derived preparation sidecar count differs")
    common = {
        "source_locator",
        "source_version_id",
        "source_sha256",
        "record_id",
        "decision",
        "cluster_id",
        "representative_id",
        "normalization",
    }
    prov_keys = {
        "profile_id",
        "dataset_repo",
        "revision",
        "config_name",
        "filename",
        "artifact_uri",
        "artifact_version_id",
        "artifact_sha256",
        "row_index",
        "data_license",
        "card_sha256",
        "coverage",
    }
    edge_keys = {
        "source_row_ordinal",
        "result_message_index",
        "assistant_message_index",
        "call_index",
        "explicit_originating_call_id",
        "original_result_id_state",
        "normalization_policy_version",
        "linkage_status",
    }
    rows, retained, original_ids = {}, {}, set()
    artifact_bindings = {}
    accepted = rejected = edge_count = 0
    rejected_reasons, core_reasons = Counter(), Counter()
    for entry in lines:
        norm = entry.get("normalization")
        if (
            type(norm) is not dict
            or norm.get("profile_id") != PREPARED_V2_PROFILE
            or norm.get("status") not in ("accepted", "rejected")
        ):
            raise ValueError("derived preparation normalization profile differs")
        status = norm["status"]
        if set(norm) != {
            "profile_id",
            "status",
            "normalized_record_sha256",
            "normalization_lineage",
            "source_provenance",
        } | ({"reason"} if status == "rejected" else set()):
            raise ValueError("derived preparation normalization fields differ")
        prov = norm["source_provenance"]
        if type(prov) is not dict or set(prov) != prov_keys:
            raise ValueError("derived preparation source fields differ")
        integer(prov["row_index"])
        text(prov["artifact_uri"], 4096)
        text(prov["artifact_version_id"], 1024)
        sha(prov["artifact_sha256"])
        uri = urlsplit(prov["artifact_uri"])
        if (
            uri.scheme != "s3"
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", uri.netloc)
            or not uri.path.startswith("/")
            or uri.query
            or uri.fragment
            or "\\" in prov["artifact_uri"]
            or any(part in ("", ".", "..") for part in uri.path[1:].split("/"))
        ):
            raise ValueError("derived preparation original artifact URI differs")
        if (
            prov["profile_id"] != PREPARED_V2_PROFILE
            or prov["dataset_repo"] != "nvidia/Open-SWE-Traces"
            or prov["revision"] != _PREPARED_V2_REVISION
            or prov["config_name"] != "v1.2"
            or prov["data_license"] != "CC-BY-4.0"
            or prov["card_sha256"] != _PREPARED_V2_CARD
            or prov["artifact_version_id"].lower() == "null"
            or type(prov["filename"]) is not str
            or not re.fullmatch(
                r"data/(openhands|sweagent|minisweagent)/(qwen38_27b|qwen38_flash_next)/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+\.parquet",
                prov["filename"],
            )
            or ".." in prov["filename"]
            or prov["coverage"]
            != "qualified singleton derived linkage; no original-link verification, pinned corpus acceptance or near-dedup claim"
        ):
            raise ValueError("derived preparation pinned original provenance differs")
        source_key = (
            prov["artifact_uri"],
            prov["artifact_version_id"],
            prov["artifact_sha256"],
            prov["filename"],
        )
        immutable_artifact = (prov["artifact_uri"], prov["artifact_version_id"])
        attribution = (prov["artifact_sha256"], prov["filename"])
        previous = artifact_bindings.setdefault(immutable_artifact, attribution)
        if previous != attribution:
            raise ValueError("derived immutable artifact attribution conflicts")
        source_key = immutable_artifact
        ordinal = prov["row_index"]
        ordinals = rows.setdefault(source_key, set())
        if ordinal in ordinals:
            raise ValueError("derived preparation original row repeats")
        ordinals.add(ordinal)
        if (
            entry.get("source_locator")
            != prov["artifact_uri"] + "#version=" + prov["artifact_version_id"] + "#row=" + str(ordinal)
            or entry.get("source_version_id") != prov["artifact_version_id"]
            or entry.get("source_sha256") != prov["artifact_sha256"]
        ):
            raise ValueError("derived preparation raw source attribution differs")
        edges = norm["normalization_lineage"]
        if type(edges) is not list or len(edges) > 8192:
            raise ValueError("derived preparation edge envelope differs")
        if status == "rejected":
            reason = norm["reason"]
            if (
                type(reason) is not str
                or reason not in report["normalization_rejection_reasons"]
                or norm["normalized_record_sha256"] is not None
                or edges
                or set(entry) != common
                or entry["decision"] != "normalization_" + reason
                or any(entry[key] is not None for key in ("record_id", "cluster_id", "representative_id"))
            ):
                raise ValueError("derived preparation rejected lineage differs")
            rejected += 1
            rejected_reasons[reason] += 1
            continue
        sha(norm["normalized_record_sha256"])
        if set(entry) != common | {"repository", "instance_id", "trajectory_id", "license"} or entry[
            "license"
        ] not in {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}:
            raise ValueError("derived preparation accepted lineage differs")
        for key in ("repository", "instance_id", "trajectory_id"):
            text(entry[key])
        sha(entry["record_id"])
        identity = {
            key: entry[key]
            for key in (
                "source_locator",
                "source_version_id",
                "source_sha256",
                "repository",
                "instance_id",
                "trajectory_id",
            )
        }
        if entry["record_id"] != _prepared_digest(identity) or entry["record_id"] in original_ids:
            raise ValueError("derived preparation original identity differs")
        original_ids.add(entry["record_id"])
        seen_calls, seen_results = set(), set()
        for edge in edges:
            if type(edge) is not dict or set(edge) != edge_keys:
                raise ValueError("derived preparation exact edge fields differ")
            for key in (
                "source_row_ordinal",
                "result_message_index",
                "assistant_message_index",
                "call_index",
            ):
                integer(edge[key])
            call = edge["explicit_originating_call_id"]
            text(call, 512)
            if (
                edge["source_row_ordinal"] != ordinal
                or edge["call_index"] != 0
                or not 0 <= edge["assistant_message_index"] < edge["result_message_index"] < 16384
                or call in seen_calls
                or edge["result_message_index"] in seen_results
                or edge["original_result_id_state"] != "absent"
                or edge["normalization_policy_version"] != PREPARED_V2_PROFILE
                or edge["linkage_status"] != PREPARED_V2_LINKAGE
            ):
                raise ValueError("derived preparation qualified edge differs")
            seen_calls.add(call)
            seen_results.add(edge["result_message_index"])
        decision = entry["decision"]
        if decision == "duplicate_exact":
            sha(entry["cluster_id"])
            sha(entry["representative_id"])
        elif decision != "retained" and (
            entry["cluster_id"] is not None or entry["representative_id"] is not None
        ):
            raise ValueError("derived dropped identity references differ")
        if decision == "retained":
            sha(entry["cluster_id"])
            sha(entry["representative_id"])
            if entry["representative_id"] != entry["record_id"]:
                raise ValueError("derived preparation retained identity differs")
            retained[entry["record_id"]] = entry
        elif type(decision) is str and decision in report["primary_drop_counts"]:
            core_reasons[decision] += 1
        else:
            raise ValueError("derived preparation core decision differs")
        accepted += 1
        edge_count += len(edges)
    if (
        any(not values or max(values) + 1 != len(values) for values in rows.values())
        or accepted != report["normalization_accepted_count"]
        or rejected != report["normalization_rejected_count"]
        or len(retained) != counts["retained_representatives"]
        or dict(core_reasons) != report["primary_drop_counts"]
        or dict(rejected_reasons) != report["normalization_rejection_reasons"]
        or edge_count != report["derived_edge_count"]
    ):
        raise ValueError("derived preparation original lineage conservation differs")
    seen_members = set()
    group_splits = {}
    split_members = {name: [] for name in splits}
    for member in members:
        if (
            set(member) != {"record_id", "group_id", "split"}
            or member["record_id"] not in retained
            or member["record_id"] in seen_members
            or member["split"] not in splits
        ):
            raise ValueError("derived preparation membership differs")
        sha(member["group_id"])
        previous_split = group_splits.setdefault(member["group_id"], member["split"])
        if previous_split != member["split"]:
            raise ValueError("derived membership group crosses held-out split")
        seen_members.add(member["record_id"])
        split_members[member["split"]].append(member["record_id"])
    if seen_members != set(retained):
        raise ValueError("derived preparation membership coverage differs")
    for split in splits:
        values = _prepared_records(bodies[split + ".jsonl"])
        if len(values) != splits[split] or len(split_members[split]) != splits[split]:
            raise ValueError("derived preparation split membership count differs")
        for record_id, value in zip(split_members[split], values, strict=True):
            if (
                set(value) != {"messages", "tools"}
                or _prepared_digest(value) != retained[record_id]["normalization"]["normalized_record_sha256"]
            ):
                raise ValueError("derived preparation normalized canonical digest differs")
            messages = value["messages"]
            if type(messages) is not list:
                raise ValueError("derived preparation canonical messages differ")
            for edge in retained[record_id]["normalization"]["normalization_lineage"]:
                if edge["result_message_index"] >= len(messages):
                    raise ValueError("derived preparation edge message index differs")
                assistant = messages[edge["assistant_message_index"]]
                result = messages[edge["result_message_index"]]
                calls = assistant.get("tool_calls")
                if (
                    assistant.get("role") != "assistant"
                    or type(calls) is not list
                    or len(calls) != 1
                    or type(calls[0]) is not dict
                    or calls[0].get("id") != edge["explicit_originating_call_id"]
                    or result.get("role") != "tool"
                    or result.get("tool_call_id") != edge["explicit_originating_call_id"]
                ):
                    raise ValueError("derived preparation canonical edge association differs")
    return {
        "source_count": report["normalization_source_count"],
        "accepted_count": accepted,
        "rejected_count": rejected,
        "rejection_reasons": dict(rejected_reasons),
        "derived_edge_count": edge_count,
        "linkage_status": PREPARED_V2_LINKAGE,
    }


def _commitpack_records(raw: bytes, expected_count: int) -> list[dict[str, Any]]:
    import io

    from amlas.data_eng.inspection_schema import decode_object

    if type(expected_count) is not int or not 0 < expected_count <= 100000:
        raise ValueError("CommitPack record count bound differs")
    records = []
    stream = io.BytesIO(raw)
    for index in range(expected_count + 1):
        line = stream.readline(32 * 1024**2 + 1)
        if not line:
            break
        if index == expected_count or len(line) > 32 * 1024**2:
            raise ValueError("CommitPack record count/byte bound exceeded")
        # The shared object decoder checks lexical depth before decoding; no splitlines allocation.
        value = decode_object(line, limit=32 * 1024**2)
        _bounded_json(value)
        records.append(value)
    if len(records) != expected_count:
        raise ValueError("CommitPack exact record count differs")
    return records


def _commitpack_count_preflight(committed: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "source",
        "normalization_rejected",
        "normalized",
        "excluded_origins",
        "included_origins",
        "retained_representatives",
        "duplicate_origins",
        "split_counts",
    }
    counts = committed["counts"]
    if type(counts) is not dict or set(counts) != keys:
        raise ValueError("CommitPack committed count fields differ")
    for key in keys - {"split_counts"}:
        if type(counts[key]) is not int or not 0 <= counts[key] <= 100000:
            raise ValueError("CommitPack committed count type/bound differs")
    splits = counts["split_counts"]
    if (
        type(splits) is not dict
        or set(splits) != {"train", "validation", "test"}
        or any(type(value) is not int or not 0 < value <= 100000 for value in splits.values())
        or not counts["source"]
        or counts["source"] != counts["normalization_rejected"] + counts["normalized"]
        or counts["normalized"] != counts["excluded_origins"] + counts["included_origins"]
        or counts["included_origins"] != counts["retained_representatives"] + counts["duplicate_origins"]
        or counts["retained_representatives"] != sum(splits.values())
    ):
        raise ValueError("CommitPack committed count conservation differs")
    return counts


PREPARED_COMMITPACK_PROFILE = "bigcode.commitpackft.python.change_sft.v1"
PREPARED_COMMITPACK_POLICY = "coding-preparation.commitpack.v3"
PREPARED_COMMITPACK_REPORT_LIMIT = 8 * 1024**2 + 262144


def _validate_preparation_commitpack(committed: dict[str, Any], bodies: dict[str, bytes]) -> dict[str, Any]:
    """Reconcile hash-pinned CommitPack metadata; source declarations grant no authority."""
    import re

    from amlas.data_eng.commitpack_preparation import replay_commitpack_preparation
    from amlas.data_eng.inspection_schema import decode_object

    try:
        counts = _commitpack_count_preflight(committed)
        splits = counts["split_counts"]
        for name in ("selection_sha256", "exclusions_sha256"):
            if type(committed[name]) is not str or not re.fullmatch(r"[a-f0-9]{64}", committed[name]):
                raise ValueError("CommitPack committed digest differs")
        report = decode_object(bodies["REPORT.json"], limit=PREPARED_COMMITPACK_REPORT_LIMIT)
        if committed["files"]["REPORT.json"]["record_count"] != 1:
            raise ValueError("CommitPack report record count differs")
        lineage = _commitpack_records(bodies["lineage.jsonl"], counts["source"])
        membership = _commitpack_records(bodies["membership.jsonl"], counts["retained_representatives"])
        if len(lineage) != counts["source"] or len(membership) != counts["retained_representatives"]:
            raise ValueError("CommitPack sidecar count differs")
        result = replay_commitpack_preparation(
            lineage_records=lineage,
            membership_records=membership,
            split_records={
                name: _commitpack_records(bodies[name + ".jsonl"], splits[name]) for name in splits
            },
            report=report,
        )
        if (
            type(result) is not dict
            or set(result)
            != {
                "counts",
                "canonical_sha256_by_split",
                "supervision_kind",
                "coverage",
                "exclusion_policy_sha256",
            }
            or _checkpoint_json(result["counts"]) != _checkpoint_json(counts)
            or _checkpoint_json(result["coverage"]) != _checkpoint_json(committed["coverage"])
            or result["exclusion_policy_sha256"] != committed["exclusions_sha256"]
            or result["supervision_kind"] != "human_commit_change_supervision"
        ):
            raise ValueError("CommitPack committed replay binding differs")
        return result
    except (KeyError, TypeError, UnicodeError, RecursionError):
        raise ValueError("CommitPack preparation refused") from None


COMMITPACK_SUMMARY_PROFILE = "commitpack_replay_summary_v1"
COMMITPACK_SUMMARY_DOMAIN = "amlas.commitpack.canonical_sha256_split.v1"


def _commitpack_summary_fields(counts, coverage, supervision, exclusion):
    from amlas.data_eng.commitpack_preparation import COUNT_KEYS, COVERAGE, MAX_ROWS, SUPERVISION_KIND

    if type(counts) is not dict or set(counts) != COUNT_KEYS:
        raise ValueError("bounded CommitPack summary counts differ")
    if any(
        type(value) is not int or not 0 <= value <= MAX_ROWS
        for name, value in counts.items()
        if name != "split_counts"
    ):
        raise ValueError("bounded CommitPack summary counts differ")
    splits = counts["split_counts"]
    if (
        type(splits) is not dict
        or set(splits) != {"train", "validation", "test"}
        or any(type(value) is not int or not 0 < value <= MAX_ROWS for value in splits.values())
    ):
        raise ValueError("bounded CommitPack summary split counts differ")
    if not (
        counts["source"] == counts["normalization_rejected"] + counts["normalized"]
        and counts["normalized"] == counts["excluded_origins"] + counts["included_origins"]
        and counts["included_origins"] == counts["retained_representatives"] + counts["duplicate_origins"]
        and counts["retained_representatives"] == sum(splits.values())
    ):
        raise ValueError("bounded CommitPack summary conservation differs")
    if (
        type(coverage) is not dict
        or _checkpoint_json(coverage) != _checkpoint_json(COVERAGE)
        or supervision != SUPERVISION_KIND
    ):
        raise ValueError("bounded CommitPack summary coverage differs")
    if type(exclusion) is not str or not re.fullmatch("[a-f0-9]{64}", exclusion):
        raise ValueError("bounded CommitPack summary exclusion differs")


def _compact_commitpack_replay(result: Any) -> dict[str, Any]:
    """Compact only after full private replay; every digest participates in original order."""
    import struct

    if type(result) is not dict or set(result) != {
        "counts",
        "canonical_sha256_by_split",
        "supervision_kind",
        "coverage",
        "exclusion_policy_sha256",
    }:
        raise ValueError("bounded CommitPack replay fields differ")
    _commitpack_summary_fields(
        result["counts"], result["coverage"], result["supervision_kind"], result["exclusion_policy_sha256"]
    )
    lists = result["canonical_sha256_by_split"]
    if type(lists) is not dict or set(lists) != {"train", "validation", "test"}:
        raise ValueError("bounded CommitPack replay split fields differ")
    splits = {}
    for name in ("train", "validation", "test"):
        values = lists[name]
        count = result["counts"]["split_counts"][name]
        if type(values) is not list or len(values) != count:
            raise ValueError("bounded CommitPack replay digest count differs")
        digest = hashlib.sha256(
            COMMITPACK_SUMMARY_DOMAIN.encode("ascii")
            + b"\0"
            + name.encode("ascii")
            + b"\0"
            + struct.pack(">Q", count)
        )
        for value in values:
            if type(value) is not str or not re.fullmatch("[a-f0-9]{64}", value):
                raise ValueError("bounded CommitPack replay digest differs")
            digest.update(bytes.fromhex(value))
        splits[name] = {"record_count": count, "ordered_canonical_sha256_sha256": digest.hexdigest()}
    return {
        "schema_version": 1,
        "summary_profile": COMMITPACK_SUMMARY_PROFILE,
        "digest_domain": COMMITPACK_SUMMARY_DOMAIN,
        "counts": result["counts"],
        "coverage": result["coverage"],
        "supervision_kind": result["supervision_kind"],
        "exclusion_policy_sha256": result["exclusion_policy_sha256"],
        "splits": splits,
    }


def _validate_commitpack_summary(value: Any) -> None:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "summary_profile",
        "digest_domain",
        "counts",
        "coverage",
        "supervision_kind",
        "exclusion_policy_sha256",
        "splits",
    }:
        raise ValueError("bounded CommitPack summary fields differ")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["summary_profile"] != COMMITPACK_SUMMARY_PROFILE
        or value["digest_domain"] != COMMITPACK_SUMMARY_DOMAIN
    ):
        raise ValueError("bounded CommitPack summary version differs")
    _commitpack_summary_fields(
        value["counts"], value["coverage"], value["supervision_kind"], value["exclusion_policy_sha256"]
    )
    if type(value["splits"]) is not dict or set(value["splits"]) != {"train", "validation", "test"}:
        raise ValueError("bounded CommitPack summary splits differ")
    for name, ref in value["splits"].items():
        if (
            type(ref) is not dict
            or set(ref) != {"record_count", "ordered_canonical_sha256_sha256"}
            or type(ref["record_count"]) is not int
            or ref["record_count"] != value["counts"]["split_counts"][name]
            or type(ref["ordered_canonical_sha256_sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", ref["ordered_canonical_sha256_sha256"])
        ):
            raise ValueError("bounded CommitPack summary split descriptor differs")


def validate_staged_inputs(binding: Any, base_directory: Any, prepared_directory: Any) -> dict[str, Any]:
    import os

    keys = {
        "schema_version",
        "catalog_id",
        "catalog_version",
        "recipe_id",
        "model_files",
        "preparation_header",
        "preparation_commit",
        "split_refs",
    }
    if (
        type(binding) is not dict
        or set(binding) != keys
        or type(binding["schema_version"]) is not int
        or binding["schema_version"] != 1
    ):
        raise ValueError("fixed staging binding fields differ")
    if binding["catalog_id"] != "qwen3_5_9b" or binding["catalog_version"] != "1":
        raise ValueError("fixed trainer catalog differs")
    expected = expected_model_weight_manifest(binding["catalog_id"], binding["catalog_version"])
    recipe(binding["recipe_id"])
    if _checkpoint_json(binding["model_files"]) != _checkpoint_json(expected["files"]):
        raise ValueError("staged model inventory differs from source")
    private_header = binding["preparation_header"]
    cp_expected = type(private_header) is dict and (
        private_header.get("profile_id"),
        private_header.get("policy_version"),
    ) == (PREPARED_COMMITPACK_PROFILE, PREPARED_COMMITPACK_POLICY)
    base = _open_directory_original(base_directory)
    prepared = None
    try:
        prepared = _open_directory_original(prepared_directory)
        if set(os.listdir(base)) != {item["path"] for item in expected["files"]}:
            raise ValueError("staged model extra/missing files")
        for ref in expected["files"]:
            _read_exact_relative(base, ref["path"], ref, 8 * 1024**3, False, nonblocking=cp_expected)
        raw = _read_exact_relative(
            prepared, "COMMITTED.json", binding["preparation_commit"], 262144, True, nonblocking=cp_expected
        )
        committed = _decode_plain_metadata(raw)
        header = {
            "job_name",
            "runtime_session_id",
            "deployment_id",
            "deployment_prefix",
            "image_uri",
            "source_commit",
            "profile_id",
            "policy_version",
        }
        expected_header = binding["preparation_header"]
        if (
            type(expected_header) is not dict
            or set(expected_header) != header
            or any(type(value) is not str or not value for value in expected_header.values())
        ):
            raise ValueError("private preparation header differs")
        if (
            set(committed)
            != {
                "schema_version",
                *header,
                "privilege_profile",
                "selection_sha256",
                "exclusions_sha256",
                "files",
                "counts",
                "coverage",
                "guest_evidence",
            }
            or type(committed["schema_version"]) is not int
            or committed["schema_version"] != 2
            or any(committed[name] != expected_header[name] for name in header)
            or committed["privilege_profile"] != "sagemaker_native_ceiling_v1"
            or (committed["profile_id"], committed["policy_version"])
            not in (
                ("openai.explicit_conversations.v1", "coding-preparation.v1"),
                (PREPARED_V2_PROFILE, PREPARED_V2_POLICY),
                (PREPARED_COMMITPACK_PROFILE, PREPARED_COMMITPACK_POLICY),
            )
        ):
            raise ValueError("preparation committed provenance differs")
        wanted = {
            "train.jsonl",
            "validation.jsonl",
            "test.jsonl",
            "membership.jsonl",
            "lineage.jsonl",
            "REPORT.json",
        }
        if (
            type(committed["files"]) is not dict
            or set(committed["files"]) != wanted
            or set(os.listdir(prepared)) != wanted | {"COMMITTED.json"}
        ):
            raise ValueError("preparation fixed output inventory differs")
        if type(binding["split_refs"]) is not dict or set(binding["split_refs"]) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError("prepared split refs differ")
        cp_counts = _commitpack_count_preflight(committed) if cp_expected else None
        if cp_counts is not None:
            expected_records = {name + ".jsonl": count for name, count in cp_counts["split_counts"].items()}
            expected_records.update(
                {
                    "lineage.jsonl": cp_counts["source"],
                    "membership.jsonl": cp_counts["retained_representatives"],
                    "REPORT.json": 1,
                }
            )
            for name in wanted:
                ref = committed["files"][name]
                if (
                    type(ref) is not dict
                    or type(ref.get("record_count")) is not int
                    or ref["record_count"] != expected_records[name]
                ):
                    raise ValueError("CommitPack descriptor record count differs before payload")
        results = {}
        derived_bodies = {}
        derived_total = 0
        for name in wanted:
            ref = committed["files"][name]
            if (
                type(ref) is not dict
                or set(ref) != {"byte_count", "sha256", "format", "record_count"}
                or type(ref["record_count"]) is not int
                or ref["record_count"] <= 0
                or ref["format"] != ("jsonl" if name.endswith(".jsonl") else "json")
            ):
                raise ValueError("prepared file descriptor differs")
            if committed["profile_id"] in (PREPARED_V2_PROFILE, PREPARED_COMMITPACK_PROFILE):
                if type(ref.get("byte_count")) is not int or ref["byte_count"] <= 0:
                    raise ValueError("derived preparation byte count differs")
                derived_total += ref["byte_count"]
                if derived_total > 256 * 1024**2:
                    raise ValueError("derived preparation total bytes exceeded")
            limit = (
                PREPARED_COMMITPACK_REPORT_LIMIT if cp_expected and name == "REPORT.json" else 256 * 1024**2
            )
            data = _read_exact_relative(prepared, name, ref, limit, True, nonblocking=cp_expected)
            if committed["profile_id"] in (PREPARED_V2_PROFILE, PREPARED_COMMITPACK_PROFILE):
                derived_bodies[name] = data
            if name.endswith(".jsonl"):
                cp_records = _commitpack_records(data, ref["record_count"]) if cp_expected else None
                lines = cp_records if cp_records is not None else data.splitlines()
                if not lines or len(lines) != ref["record_count"] or any(not line for line in lines):
                    raise ValueError("prepared record count differs")
                split = name.removesuffix(".jsonl")
                if split in binding["split_refs"]:
                    if (
                        ref != binding["split_refs"][split]
                        or committed["counts"]["split_counts"].get(split) != ref["record_count"]
                    ):
                        raise ValueError("private prepared split count/hash differs")
                    for line in lines:
                        if cp_records is None and len(line) > 32 * 1024 * 1024:
                            raise ValueError("prepared row bytes exceeded")
                        # Parse only after exact same-FD input integrity; no instruction/code execution.
                        value = line if cp_records is not None else json.loads(line)
                        _bounded_json(value)
                        if type(value) is not dict or set(value) != {"messages", "tools"}:
                            raise ValueError("prepared explicit row shape differs")
                    results[split + "_record_count"] = ref["record_count"]
        normalization = (
            _validate_preparation_v2(committed, derived_bodies)
            if committed["profile_id"] == PREPARED_V2_PROFILE
            else None
        )
        commitpack = (
            _validate_preparation_commitpack(committed, derived_bodies)
            if committed["profile_id"] == PREPARED_COMMITPACK_PROFILE
            else None
        )
        return {
            "catalog_id": binding["catalog_id"],
            "catalog_version": binding["catalog_version"],
            **results,
            "trusted": False,
            "test_used_for_training": False,
            "preparation_commit_sha256": binding["preparation_commit"]["sha256"],
            "preparation_profile_id": committed["profile_id"],
            "preparation_policy_version": committed["policy_version"],
            **({"normalization": normalization} if normalization is not None else {}),
            **({"commitpack": _compact_commitpack_replay(commitpack)} if commitpack is not None else {}),
        }
    except (OSError, KeyError, TypeError):
        raise ValueError("staging admission refused") from None
    finally:
        if prepared is not None:
            os.close(prepared)
        os.close(base)


def train_fixed_recipe(
    model: Any,
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    recipe_id: str,
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    import random
    import time

    import torch

    settings = recipe(recipe_id)
    if not training_rows or not validation_rows:
        raise ValueError("positive train/validation supervision required")
    for row in training_rows + validation_rows:
        if (
            not isinstance(row, dict)
            or any(name not in row for name in ("input_ids", "attention_mask", "labels"))
            or not len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])
            or not 1 < len(row["input_ids"]) <= MAX_SEQUENCE_TOKENS
            or not any(
                label != -100 and valid
                for label, valid in zip(row["labels"][1:], row["attention_mask"][1:], strict=True)
            )
        ):
            raise ValueError("complete shifted token supervision required")
    unique_rows = {}
    provenance_rows = {}
    for row in training_rows:
        token_signature = hashlib.sha256(
            _json({key: row[key] for key in ("input_ids", "attention_mask", "labels")}).encode()
        ).hexdigest()
        identity = row.get("original_row_sha256", token_signature)
        if identity in provenance_rows and provenance_rows[identity] != token_signature:
            raise ValueError("unique original corpus identity maps to changed supervision")
        provenance_rows[identity] = token_signature
        unique_rows[token_signature] = sum(
            label != -100 and bool(valid)
            for label, valid in zip(row["labels"][1:], row["attention_mask"][1:], strict=True)
        )
    unique_supervised_tokens = sum(unique_rows.values())
    if recipe_id == "full_v1" and (len(unique_rows) < 1000 or unique_supervised_tokens < 1000000):
        raise ValueError("unique accepted corpus floors not met before optimizer")
    selected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not selected or any("language_model.layers." not in name or "lora_" not in name for name in selected):
        raise ValueError("fixed language-only LoRA required")
    before = {name: parameter.detach().cpu().clone() for name, parameter in selected.items()}
    device = next(model.parameters()).device
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        list(selected.values()), lr=settings["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    order = list(range(len(training_rows)))
    random.shuffle(order)
    cursor = 0
    epochs = 0
    tokens = 0
    history = []
    validation_history = []
    best = float("inf")
    nonimproving = 0
    started = time.monotonic()
    would_stop = False

    def tensor_batch(rows):
        return {name: value.to(device) for name, value in pad_batch(rows, pad_token_id).items()}

    def validation():
        model.eval()
        loss_sum = 0.0
        count = 0
        with torch.no_grad():
            for row in validation_rows:
                batch = tensor_batch([row])
                supervised = int(
                    ((batch["labels"][:, 1:] != -100) & (batch["attention_mask"][:, 1:] != 0)).sum().item()
                )
                loss = model(**batch, use_cache=False).loss
                if loss is None or not bool(torch.isfinite(loss).item()):
                    raise ValueError("nonfinite validation loss")
                loss_sum += float(loss.item()) * supervised
                count += supervised
        if count <= 0:
            raise ValueError("empty validation supervision")
        return loss_sum / count

    for step in range(1, settings["optimizer_steps"] + 1):
        batches = []
        for _ in range(settings["gradient_accumulation"]):
            if cursor >= len(order):
                cursor = 0
                epochs += 1
                random.shuffle(order)
            batches.append(tensor_batch([training_rows[order[cursor]]]))
            cursor += 1
        update = optimizer_update(model, optimizer, batches)
        tokens += update["supervised_tokens"]
        history.append(
            {
                "step": step,
                "loss": update["loss"],
                "supervised_tokens": update["supervised_tokens"],
                "gradient_norm": update["gradient_norm"],
            }
        )
        if step % 64 == 0 or step == settings["optimizer_steps"]:
            loss = validation()
            improved = loss < best - 1e-4
            if improved:
                best = loss
                nonimproving = 0
            else:
                nonimproving += 1
            would_stop = nonimproving >= 8
            validation_history.append(
                {
                    "step": step,
                    "loss": loss,
                    "best_loss": best,
                    "nonimproving_evaluations": nonimproving,
                    "early_stop_would_trigger": would_stop,
                }
            )
            # The fixed step/token floors cannot be shortened by early stopping.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    changed = sum(
        not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in selected.items()
    )
    if changed <= 0:
        raise ValueError("no changed language adapter tensors")
    met = tokens >= settings["minimum_supervised_tokens_processed"]
    return {
        "schema_version": 1,
        "recipe_id": recipe_id,
        "optimizer_steps": settings["optimizer_steps"],
        "supervised_tokens": tokens,
        "unique_training_rows": len(unique_rows),
        "original_provenance_rows": len(provenance_rows),
        "unique_supervised_tokens": unique_supervised_tokens,
        "optimizer_exposure_tokens": tokens,
        "epochs_completed": epochs,
        "changed_tensor_count": changed,
        "train_history": history,
        "validation_history": validation_history,
        "early_stop_would_trigger": would_stop,
        "target_met": met,
        "status": "checkpoint_ready" if met else "training_target_not_met",
        "reload_pending": True,
        "quality_success": False,
        "vision_frozen": not any(
            parameter.requires_grad for name, parameter in model.named_parameters() if ".visual." in name
        ),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "optimizer_state_saved": False,
    }


FIXED_BASE = "/opt/ml/input/data/base_model"
FIXED_PREPARED = "/opt/ml/input/data/prepared"
FIXED_ADAPTER = "/opt/ml/input/data/adapter"
FIXED_OUTPUT = "/opt/ml/model"


def _require_native_readonly_directory(path: Any) -> None:
    import os
    from pathlib import Path

    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("native readonly original input path refused")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for component in (None, *path.parts[1:]):
            if component is not None:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            value = os.fstat(descriptor)
            if value.st_uid != 0 or value.st_mode & 0o022:
                raise ValueError("native readonly input ancestor/owner unqualified")
    except OSError:
        raise ValueError("native readonly input original components refused") from None
    finally:
        os.close(descriptor)


def _require_native_readonly_file(descriptor: int) -> None:
    import os
    import stat

    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_uid != 0 or value.st_mode & 0o133:
        raise ValueError("native readonly input file owner/mode unqualified")


class _HeldNativeInputs:
    """Private fixed input authority; never deserialized from model/caller JSON."""

    def __init__(self, path: str, refs: dict[str, Any]):
        import os
        from pathlib import Path

        if path not in (FIXED_BASE, FIXED_ADAPTER) or type(refs) is not dict or not refs:
            raise ValueError("fixed native input namespace differs")
        self.path = path
        self.directory = None
        self.files = {}
        self.refs = {name: {key: ref[key] for key in ("sha256", "byte_count")} for name, ref in refs.items()}
        try:
            _require_native_readonly_directory(Path(path))
            self.directory = _open_directory_original(path)
            if set(os.listdir(self.directory)) != set(refs):
                raise ValueError("native fixed input inventory differs")
            for name in refs:
                if type(name) is not str or "/" in name or name in (".", ".."):
                    raise ValueError("native fixed input file name refused")
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.directory)
                self.files[name] = descriptor
                _require_native_readonly_file(descriptor)
            self.verify()
        except Exception:
            self.close()
            raise

    @property
    def alias(self) -> str:
        if self.directory is None:
            raise ValueError("native input authority closed")
        return "/proc/self/fd/" + str(self.directory)

    def verify(self) -> None:
        import os

        if self.directory is None:
            raise ValueError("native input authority closed")
        _require_native_readonly_directory(self.path)
        _checkpoint_root_named(self.directory, self.path)
        if set(os.listdir(self.directory)) != set(self.files):
            raise ValueError("native fixed input inventory changed")
        for name, descriptor in self.files.items():
            _require_native_readonly_file(descriptor)
            actual, _ = _fd_bytes(descriptor, 8 * 1024**3)
            if actual != self.refs[name]:
                raise ValueError("native retained input bytes changed")
            _checkpoint_named(self.directory, descriptor, name)

    def close(self) -> None:
        import os

        for descriptor in self.files.values():
            os.close(descriptor)
        self.files = {}
        if self.directory is not None:
            os.close(self.directory)
            self.directory = None


def load_fixed_gpu_model(base_inputs: _HeldNativeInputs | None = None) -> Any:
    if type(base_inputs) is not _HeldNativeInputs or base_inputs.path != FIXED_BASE:
        raise ValueError("source-held base input authority required")
    base_inputs.verify()
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("native BF16 GPU capability unqualified")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_inputs.alias,
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",  # pragma: allowlist secret
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    )
    base_inputs.verify()
    return model.to("cuda:0")


def _fixed_output_fd() -> int:
    import fcntl
    import os
    import stat

    value = os.fstat(3)
    if not stat.S_ISDIR(value.st_mode) or (value.st_uid, value.st_gid, stat.S_IMODE(value.st_mode)) != (
        10001,
        10001,
        0o700,
    ):
        raise ValueError("fixed training output FD3 refused")
    named = _open_directory_original(FIXED_OUTPUT)
    try:
        if (value.st_dev, value.st_ino) != (os.fstat(named).st_dev, os.fstat(named).st_ino):
            raise ValueError("fixed training output name changed")
    finally:
        os.close(named)
    fcntl.flock(3, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.listdir(3):
        raise ValueError("fixed training output must start empty")
    return 3


def _publish_fixed_json(output_fd: int, name: str, value: Any) -> dict[str, Any]:
    import os

    if name not in {"TRAINING.json", "RELOAD.json", "COMMITTED.json"}:
        raise ValueError("fixed training receipt name refused")
    raw = _checkpoint_json(value)
    held = {}
    descriptor = None
    try:
        _checkpoint_root_named(output_fd, FIXED_OUTPUT)
        preceding = value.get("files") if name == "COMMITTED.json" and type(value) is dict else None
        if name == "COMMITTED.json":
            allowed = {
                "adapter_model.safetensors",
                "adapter_config.json",
                "CHECKPOINT.json",
                "TRAINING.json",
                "RELOAD.json",
            }
            if (
                type(preceding) is not dict
                or not preceding
                or not set(preceding) <= allowed
                or set(os.listdir(output_fd)) != set(preceding)
            ):
                raise ValueError("final training inventory differs")
            for prior_name, expected in preceding.items():
                prior = os.open(prior_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=output_fd)
                held[prior_name] = prior
                actual, _ = _fd_bytes(prior, 2 * 1024**3 if prior_name.endswith(".safetensors") else 262144)
                if type(expected) is not dict or actual != {
                    key: expected.get(key) for key in ("sha256", "byte_count")
                }:
                    raise ValueError("final training expected bytes differ")
                _checkpoint_named(output_fd, prior, prior_name)
        descriptor = os.open(
            name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=output_fd
        )
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        expected_raw = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
        for _ in range(2):
            actual, _ = _fd_bytes(descriptor, 262144)
            if actual != expected_raw:
                raise ValueError("training receipt same-FD bytes changed")
            _checkpoint_named(output_fd, descriptor, name)
            for prior_name, prior in held.items():
                actual, _ = _fd_bytes(prior, 2 * 1024**3 if prior_name.endswith(".safetensors") else 262144)
                if actual != {key: preceding[prior_name].get(key) for key in ("sha256", "byte_count")}:
                    raise ValueError("final training retained bytes changed")
                _checkpoint_named(output_fd, prior, prior_name)
            _checkpoint_root_named(output_fd, FIXED_OUTPUT)
        if name == "COMMITTED.json" and set(os.listdir(output_fd)) != set(preceding) | {name}:
            raise ValueError("final training extra file refused")
        return expected_raw
    except Exception:
        if descriptor is not None and name == "COMMITTED.json":
            try:
                # Invalidate only the original exclusively created write authority.
                # No pathname unlink/check race can target a substituted inode.
                os.ftruncate(descriptor, 0)
                os.fsync(descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for prior in held.values():
            os.close(prior)


def _load_split_encoded(
    prepared_fd: int, name: str, ref: dict[str, Any], tokenizer: Any
) -> list[dict[str, Any]]:
    raw = _read_exact_relative(prepared_fd, name, ref, 256 * 1024**2, True)
    lines = raw.splitlines()
    if len(lines) != ref["record_count"]:
        raise ValueError("training split record count differs")
    encoded = []
    reasons = {}
    for line in lines:
        try:
            encoded.append(encode_supervised(json.loads(line), tokenizer))
        except ConversationCompatibilityError as error:
            for reason, count in error.reason_counts.items():
                reasons[reason] = reasons.get(reason, 0) + count
    if reasons:
        raise ConversationCompatibilityError(reasons)
    return encoded


GENERATION_RECIPE_ID = "humaneval_greedy_v1"
FIXED_PROMPTS = "/opt/ml/input/data/prompts"
GENERATION_OUTPUT_LIMIT = 12 * 1024**2


def validate_generation_binding(value: Any, *, source_commit: str) -> dict[str, Any]:
    """Validate data-only pins; native input ownership is independently required."""
    import copy

    from amlas.training.coding_generation import CONTRACT_SHA256

    common = {
        "schema_version",
        "operation",
        "catalog_id",
        "catalog_version",
        "recipe_id",
        "recipe_version",
        "source_commit",
        "evaluation_id",
        "arm",
        "spec_sha256",
        "contract_sha256",
        "model_files",
        "prompts_ref",
    }
    if type(value) is not dict or value.get("arm") not in ("base", "tuned"):
        raise ValueError("fixed generation arm/binding required")
    wanted = common | ({"checkpoint_ref", "checkpoint_source_commit"} if value["arm"] == "tuned" else set())
    if set(value) != wanted:
        raise ValueError("fixed generation binding fields differ")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["operation"] != "generate"
        or value["catalog_id"] != "qwen3_5_9b"
        or value["catalog_version"] != "1"
        or value["recipe_id"] != GENERATION_RECIPE_ID
        or value["recipe_version"] != "1"
        or type(source_commit) is not str
        or not re.fullmatch("[a-f0-9]{40}", source_commit)
        or value["source_commit"] != source_commit
        or type(value["evaluation_id"]) is not str
        or not re.fullmatch("[A-Za-z0-9_-]{1,128}", value["evaluation_id"])
        or type(value["spec_sha256"]) is not str
        or not re.fullmatch("[a-f0-9]{64}", value["spec_sha256"])
        or value["contract_sha256"] != CONTRACT_SHA256
        or type(value["model_files"]) is not list
        or _checkpoint_json(value["model_files"])
        != _checkpoint_json(expected_model_weight_manifest("qwen3_5_9b", "1")["files"])
    ):
        raise ValueError("fixed generation source/contract/model pins differ")

    def reference(ref: Any, filename: str, fmt: str, limit: int) -> None:
        if (
            type(ref) is not dict
            or set(ref) != {"uri", "version_id", "sha256", "byte_count", "format"}
            or type(ref["uri"]) is not str
            or len(ref["uri"]) > 2048
            or not re.fullmatch(
                r"s3://[a-z0-9][a-z0-9.-]+/sessions/[A-Za-z0-9_-]+/[A-Za-z0-9_./-]+/" + re.escape(filename),
                ref["uri"],
            )
            or ".." in ref["uri"].split("/")
            or type(ref["version_id"]) is not str
            or ref["version_id"] in ("", "null")
            or len(ref["version_id"]) > 1024
            or any(not 33 <= ord(char) <= 126 for char in ref["version_id"])
            or type(ref["sha256"]) is not str
            or not re.fullmatch("[a-f0-9]{64}", ref["sha256"])
            or type(ref["byte_count"]) is not int
            or not 0 < ref["byte_count"] <= limit
            or ref["format"] != fmt
        ):
            raise ValueError("fixed generation immutable artifact reference differs")

    reference(value["prompts_ref"], "PROMPTS.jsonl", "jsonl", 2 * 1024**2)
    if value["arm"] == "tuned":
        reference(value["checkpoint_ref"], "CHECKPOINT.json", "json", 262144)
        if type(value["checkpoint_source_commit"]) is not str or not re.fullmatch(
            "[a-f0-9]{40}", value["checkpoint_source_commit"]
        ):
            raise ValueError("original training checkpoint source pin required")
    return copy.deepcopy(value)


def _publish_generation_artifacts(
    output_fd: int, *, candidates_raw: bytes, generation: dict, verify_inputs=None
) -> dict:
    import os

    if type(candidates_raw) is not bytes or not 0 < len(candidates_raw) <= GENERATION_OUTPUT_LIMIT:
        raise ValueError("generation candidate byte bound exceeded")
    if (
        type(generation) is not dict
        or type(generation.get("schema_version")) is not int
        or generation["schema_version"] != 1
        or generation.get("operation") != "generate"
        or type(generation.get("source_commit")) is not str
        or not re.fullmatch("[a-f0-9]{40}", generation["source_commit"])
        or type(generation.get("case_count")) is not int
        or generation["case_count"] != 164
        or generation.get("quality_success") is not False
        or generation.get("native_admission") is not False
    ):
        raise ValueError("fixed generation receipt differs")
    metadata_raw = _checkpoint_json(generation)
    held = {}
    commit_fd = None
    files = {}
    try:
        _checkpoint_root_named(output_fd, FIXED_OUTPUT)
        if os.listdir(output_fd):
            raise ValueError("generation output must start empty")
        for name, raw, limit, fmt in (
            ("CANDIDATES.jsonl", candidates_raw, GENERATION_OUTPUT_LIMIT, "jsonl"),
            ("GENERATION.json", metadata_raw, 262144, "json"),
        ):
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=output_fd,
            )
            held[name] = descriptor
            with os.fdopen(os.dup(descriptor), "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            expected = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
            actual, _ = _fd_bytes(descriptor, limit)
            if actual != expected:
                raise ValueError("generation same-FD publication differs")
            _checkpoint_named(output_fd, descriptor, name)
            files[name] = {**expected, "format": fmt}
        commit = {
            "schema_version": 1,
            "operation": "generate",
            "source_commit": generation["source_commit"],
            "case_count": 164,
            "quality_success": False,
            "native_admission": False,
            "files": files,
        }
        raw = _checkpoint_json(commit)
        for name, descriptor in held.items():
            actual, _ = _fd_bytes(descriptor, GENERATION_OUTPUT_LIMIT if name.endswith(".jsonl") else 262144)
            if actual != {key: files[name][key] for key in ("sha256", "byte_count")}:
                raise ValueError("generation retained bytes changed before commit")
            _checkpoint_named(output_fd, descriptor, name)
        if set(os.listdir(output_fd)) != set(files):
            raise ValueError("generation precommit inventory differs")
        if verify_inputs is not None:
            verify_inputs()
        os.fsync(output_fd)
        commit_fd = os.open(
            "COMMITTED.json",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=output_fd,
        )
        with os.fdopen(os.dup(commit_fd), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        expected_commit = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
        for _ in range(2):
            actual, _ = _fd_bytes(commit_fd, 262144)
            if actual != expected_commit:
                raise ValueError("generation commit bytes changed")
            _checkpoint_named(output_fd, commit_fd, "COMMITTED.json")
            for name, descriptor in held.items():
                actual, _ = _fd_bytes(
                    descriptor, GENERATION_OUTPUT_LIMIT if name.endswith(".jsonl") else 262144
                )
                if actual != {key: files[name][key] for key in ("sha256", "byte_count")}:
                    raise ValueError("generation retained bytes changed after commit")
                _checkpoint_named(output_fd, descriptor, name)
            _checkpoint_root_named(output_fd, FIXED_OUTPUT)
        if verify_inputs is not None:
            verify_inputs()
        if set(os.listdir(output_fd)) != set(files) | {"COMMITTED.json"}:
            raise ValueError("generation final inventory differs")
        os.fsync(output_fd)
        return {**files, "COMMITTED.json": {**expected_commit, "format": "json"}}
    except BaseException:
        if commit_fd is not None:
            try:
                os.ftruncate(commit_fd, 0)
                os.fsync(commit_fd)
            except OSError:
                pass
        raise
    finally:
        if commit_fd is not None:
            os.close(commit_fd)
        for descriptor in held.values():
            os.close(descriptor)


def managed_generation(
    stage_binding: Any, *, source_commit: str, privilege_proof: Any, stage_manifest_sha256: str
) -> dict:
    import io
    import os
    from dataclasses import asdict
    from pathlib import Path

    from amlas.data_eng.inspection_schema import decode_object, validate_native_privilege_proof
    from amlas.training.coding_generation import (
        CodingGenerationBinding,
        canonical_bytes,
        generate_humaneval_arm,
        prompts_sha256,
    )

    validate_native_privilege_proof(privilege_proof)
    binding = validate_generation_binding(stage_binding, source_commit=source_commit)
    if (
        os.environ.get("AMLAS_SOURCE_COMMIT") != source_commit
        or type(stage_manifest_sha256) is not str
        or not re.fullmatch("[a-f0-9]{64}", stage_manifest_sha256)
    ):
        raise ValueError("generation image/raw binding differs")
    _require_native_readonly_directory(FIXED_PROMPTS)
    prompts_directory = _open_directory_original(FIXED_PROMPTS)
    prompt_fd = None
    base_inputs = adapter_inputs = None
    try:
        if set(os.listdir(prompts_directory)) != {"PROMPTS.jsonl"}:
            raise ValueError("public prompt input inventory differs")
        prompt_fd = os.open(
            "PROMPTS.jsonl", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=prompts_directory
        )
        _require_native_readonly_file(prompt_fd)
        expected_prompt = {key: binding["prompts_ref"][key] for key in ("sha256", "byte_count")}
        actual, raw = _fd_bytes(prompt_fd, 2 * 1024**2, collect=True)
        if actual != expected_prompt:
            raise ValueError("public prompt bytes differ")
        rows = []
        stream = io.BytesIO(raw)
        for index in range(165):
            line = stream.readline(65537)
            if not line:
                break
            if index == 164 or len(line) > 65536 or not line.endswith(b"\n"):
                raise ValueError("public prompt row bound differs")
            rows.append(decode_object(line, limit=65536))
        prompts = tuple(rows)
        if prompts_sha256(prompts) != expected_prompt["sha256"]:
            raise ValueError("public prompt canonical population differs")
        if any(
            marker in row["prompt"]
            for row in prompts
            for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
        ):
            raise ValueError("public prompt contains native role markers")
        base_inputs = _HeldNativeInputs(FIXED_BASE, {item["path"]: item for item in binding["model_files"]})
        checkpoint = None
        if binding["arm"] == "tuned":
            directory = _open_directory_original(FIXED_ADAPTER)
            try:
                checkpoint_raw = _read_exact_relative(
                    directory, "CHECKPOINT.json", binding["checkpoint_ref"], 262144, True
                )
            finally:
                os.close(directory)
            checkpoint = _decode_plain_metadata(checkpoint_raw)
            _checkpoint_metadata(checkpoint)
            if checkpoint["source_commit"] != binding["checkpoint_source_commit"]:
                raise ValueError("original checkpoint producer source differs")
            adapter_inputs = _HeldNativeInputs(
                FIXED_ADAPTER, {**checkpoint["files"], "CHECKPOINT.json": binding["checkpoint_ref"]}
            )
        elif os.path.exists(FIXED_ADAPTER):
            raise ValueError("base generation cannot receive adapter input")
        output_fd = _fixed_output_fd()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            base_inputs.alias,
            revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",  # pragma: allowlist secret -- reviewed public Qwen model revision
            local_files_only=True,
            trust_remote_code=False,
        )
        model = load_fixed_gpu_model(base_inputs)
        if checkpoint is not None:
            model = reload_adapter_checkpoint(
                model, Path(FIXED_ADAPTER), expected_ref=binding["checkpoint_ref"]
            )
        model_hash = hashlib.sha256(_checkpoint_json(binding["model_files"])).hexdigest()
        if checkpoint is not None:
            model_hash = hashlib.sha256(
                (model_hash + binding["checkpoint_ref"]["sha256"]).encode()
            ).hexdigest()
        tokenizer_hash = hashlib.sha256(
            _checkpoint_json(
                [
                    ref
                    for ref in binding["model_files"]
                    if ref["role"] in ("tokenizer", "tokenizer_config", "chat_template")
                ]
            )
        ).hexdigest()
        bound = CodingGenerationBinding(
            binding["evaluation_id"],
            binding["arm"],
            expected_prompt["sha256"],
            model_hash,
            tokenizer_hash,
            binding["spec_sha256"],
        )
        result = generate_humaneval_arm(model=model, tokenizer=tokenizer, prompts=prompts, binding=bound)
        candidates_raw = b"".join(canonical_bytes(asdict(case)) + b"\n" for case in result.cases)
        generation = decode_object(result.manifest_bytes, limit=262144)
        generation.update(
            {
                "operation": "generate",
                "source_commit": source_commit,
                "stage_manifest_sha256": stage_manifest_sha256,
                "privilege_proof": privilege_proof,
                "quality_success": False,
                "native_admission": False,
            }
        )

        def verify_inputs():
            base_inputs.verify()
            if adapter_inputs is not None:
                adapter_inputs.verify()
            actual, _ = _fd_bytes(prompt_fd, 2 * 1024**2)
            if actual != expected_prompt:
                raise ValueError("generation retained prompt bytes changed")
            _checkpoint_named(prompts_directory, prompt_fd, "PROMPTS.jsonl")
            _checkpoint_root_named(prompts_directory, FIXED_PROMPTS)

        files = _publish_generation_artifacts(
            output_fd, candidates_raw=candidates_raw, generation=generation, verify_inputs=verify_inputs
        )
        return {
            "schema_version": 1,
            "operation": "generate",
            "arm": binding["arm"],
            "case_count": 164,
            "files": files,
            "quality_success": False,
            "native_admission": False,
        }
    finally:
        if adapter_inputs is not None:
            adapter_inputs.close()
        if base_inputs is not None:
            base_inputs.close()
        if prompt_fd is not None:
            os.close(prompt_fd)
        os.close(prompts_directory)


def validate_reload_binding(value: Any, *, source_commit: str, recipe_id: str) -> dict:
    """Closed descriptive native binding; parent compatibility is independently required."""
    import copy

    common = {"schema_version", "catalog_id", "catalog_version", "recipe_id", "model_files", "checkpoint_ref"}
    extra = {
        "operation",
        "recipe_version",
        "source_commit",
        "checkpoint_source_commit",
        "reload_contract_version",
        "compatibility_digest",
    }
    if (
        type(value) is not dict
        or type(value.get("schema_version")) is not int
        or value["schema_version"] not in (1, 2)
    ):
        raise ValueError("managed reload binding fields differ")
    version = value["schema_version"]
    if set(value) != common | (extra if version == 2 else set()):
        raise ValueError("managed reload binding fields differ")
    if (
        value["catalog_id"] != "qwen3_5_9b"
        or value["catalog_version"] != "1"
        or value["recipe_id"] != recipe_id
    ):
        raise ValueError("managed fixed binding differs")
    recipe(recipe_id)
    expected = expected_model_weight_manifest("qwen3_5_9b", "1")
    if _checkpoint_json(value["model_files"]) != _checkpoint_json(expected["files"]):
        raise ValueError("managed reload base inventory differs")
    ref = value["checkpoint_ref"]
    if (
        type(ref) is not dict
        or set(ref) != {"sha256", "byte_count"}
        or type(ref["sha256"]) is not str
        or not re.fullmatch("[a-f0-9]{64}", ref["sha256"])
        or type(ref["byte_count"]) is not int
        or not 0 < ref["byte_count"] <= 262144
    ):
        raise ValueError("managed reload checkpoint descriptor differs")
    if version == 2 and (
        value["operation"] != "reload"
        or value["recipe_version"] != "1"
        or value["source_commit"] != source_commit
        or type(value["checkpoint_source_commit"]) is not str
        or not re.fullmatch("[a-f0-9]{40}", value["checkpoint_source_commit"])
        or value["checkpoint_source_commit"] == source_commit
        or value["reload_contract_version"] != RELOAD_CONTRACT_VERSION
        or type(value["compatibility_digest"]) is not str
        or not re.fullmatch("[a-f0-9]{64}", value["compatibility_digest"])
    ):
        raise ValueError("managed reload compatibility source/contract differs")
    return copy.deepcopy(value)


def managed_operation(
    operation: str,
    recipe_id: str,
    *,
    source_commit: str,
    stage_binding: Any,
    privilege_proof: Any,
    stage_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    import os
    import re
    from pathlib import Path

    from amlas.data_eng.inspection_schema import validate_native_privilege_proof

    if (
        operation not in ("train", "reload", "generate")
        or type(stage_manifest_sha256) is not str
        or not re.fullmatch("[a-f0-9]{64}", stage_manifest_sha256)
        or not re.fullmatch("[a-f0-9]{40}", source_commit)
        or os.environ.get("AMLAS_SOURCE_COMMIT") != source_commit
    ):
        raise ValueError("managed source operation differs")
    validate_native_privilege_proof(privilege_proof)
    if operation == "generate":
        if recipe_id != GENERATION_RECIPE_ID:
            raise ValueError("fixed generation recipe differs")
        validate_generation_binding(stage_binding, source_commit=source_commit)
        return managed_generation(
            stage_binding,
            source_commit=source_commit,
            privilege_proof=privilege_proof,
            stage_manifest_sha256=stage_manifest_sha256,
        )
    recipe(recipe_id)
    if (
        type(stage_binding) is not dict
        or stage_binding.get("recipe_id") != recipe_id
        or stage_binding.get("catalog_id") != "qwen3_5_9b"
        or stage_binding.get("catalog_version") != "1"
    ):
        raise ValueError("managed fixed binding differs")
    if operation == "train":
        input_summary = run_fixed_stage(
            "input_verify",
            lambda: validate_staged_inputs(stage_binding, Path(FIXED_BASE), Path(FIXED_PREPARED)),
        )
    else:
        stage_binding = validate_reload_binding(
            stage_binding, source_commit=source_commit, recipe_id=recipe_id
        )
        expected = expected_model_weight_manifest("qwen3_5_9b", "1")
        base_fd = _open_directory_original(FIXED_BASE)
        adapter_fd = _open_directory_original(FIXED_ADAPTER)
        try:
            if set(os.listdir(base_fd)) != {item["path"] for item in expected["files"]}:
                raise ValueError("reload base inventory differs")
            for ref in expected["files"]:
                _read_exact_relative(base_fd, ref["path"], ref, 8 * 1024**3, False)
            checkpoint_raw = _read_exact_relative(
                adapter_fd, "CHECKPOINT.json", stage_binding["checkpoint_ref"], 262144, True
            )
            checkpoint = _decode_plain_metadata(checkpoint_raw)
            producer_source = (
                source_commit
                if stage_binding["schema_version"] == 1
                else stage_binding["checkpoint_source_commit"]
            )
            if checkpoint.get("recipe_id") != recipe_id or checkpoint.get("source_commit") != producer_source:
                raise ValueError("reload recipe binding differs")
        finally:
            os.close(base_fd)
            os.close(adapter_fd)
        input_summary = {"checkpoint_sha256": stage_binding["checkpoint_ref"]["sha256"]}
    base_inputs = run_fixed_stage(
        "base_input_verify",
        lambda: _HeldNativeInputs(FIXED_BASE, {item["path"]: item for item in stage_binding["model_files"]}),
    )
    adapter_inputs = None
    try:
        if operation == "reload":
            _checkpoint_metadata(checkpoint)
            adapter_inputs = _HeldNativeInputs(
                FIXED_ADAPTER, {**checkpoint["files"], "CHECKPOINT.json": stage_binding["checkpoint_ref"]}
            )
        output_fd = _fixed_output_fd()

        # Exact stage bytes/OS/source validation are complete before native ML load.
        def load_ml_runtime():
            import torch
            from transformers import AutoTokenizer

            return torch, AutoTokenizer

        torch, tokenizer_class = run_fixed_stage("ml_import", load_ml_runtime)
        print(_json({"fixed_runtime_observation": fixed_runtime_fingerprint(torch)}), flush=True)

        def probe_gpu():
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise ValueError("native GPU admission absent")

        run_fixed_stage("gpu_probe", probe_gpu)
        tokenizer = run_fixed_stage(
            "tokenizer_load",
            lambda: tokenizer_class.from_pretrained(
                base_inputs.alias,
                revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",  # pragma: allowlist secret
                local_files_only=True,
                trust_remote_code=False,
            ),
        )
        if operation == "reload":
            model = run_fixed_stage(
                "reload_load",
                lambda: reload_adapter_checkpoint(
                    load_fixed_gpu_model(base_inputs),
                    Path(FIXED_ADAPTER),
                    expected_ref=stage_binding["checkpoint_ref"],
                ),
            )
            model.eval()
            query = torch.tensor(checkpoint["greedy_input_ids"], dtype=torch.long, device="cuda:0")
            expected_ids = checkpoint["greedy_output_ids"]
            generated = run_fixed_stage(
                "reload_generate",
                lambda: model.generate(
                    input_ids=query,
                    max_new_tokens=len(expected_ids[0]) - len(checkpoint["greedy_input_ids"][0]),
                    do_sample=False,
                    use_cache=True,
                ).tolist(),
            )
            if generated != expected_ids:
                raise ValueError("fresh process greedy output equality refused")
            reload_fields = (
                {}
                if stage_binding["schema_version"] == 1
                else {
                    name: stage_binding[name]
                    for name in (
                        "checkpoint_source_commit",
                        "reload_contract_version",
                        "compatibility_digest",
                    )
                }
            )
            result = {
                "schema_version": stage_binding["schema_version"],
                **reload_fields,
                "operation": "reload",
                "source_commit": source_commit,
                "catalog_id": "qwen3_5_9b",
                "catalog_version": "1",
                "recipe_id": recipe_id,
                "checkpoint_sha256": input_summary["checkpoint_sha256"],
                "fresh_process_greedy_equal": True,
                "quality_success": False,
                "privilege_proof": privilege_proof,
            }
            ref = _publish_fixed_json(output_fd, "RELOAD.json", result)
            base_inputs.verify()
            if adapter_inputs is not None:
                adapter_inputs.verify()
            _publish_fixed_json(
                output_fd,
                "COMMITTED.json",
                {
                    "schema_version": stage_binding["schema_version"],
                    **reload_fields,
                    "operation": "reload",
                    "source_commit": source_commit,
                    "files": {"RELOAD.json": ref},
                    "quality_success": False,
                },
            )
            return result
        prepared_fd = _open_directory_original(FIXED_PREPARED)
        try:
            training = run_fixed_stage(
                "train_tokenize",
                lambda: _load_split_encoded(
                    prepared_fd, "train.jsonl", stage_binding["split_refs"]["train"], tokenizer
                ),
            )
            validation = run_fixed_stage(
                "validation_tokenize",
                lambda: _load_split_encoded(
                    prepared_fd, "validation.jsonl", stage_binding["split_refs"]["validation"], tokenizer
                ),
            )
        finally:
            os.close(prepared_fd)
        model = run_fixed_stage("model_load", lambda: load_fixed_gpu_model(base_inputs))
        model = run_fixed_stage("adapter_attach", lambda: attach_language_lora(model))
        result = run_fixed_stage(
            "optimizer_train",
            lambda: train_fixed_recipe(
                model, training, validation, recipe_id, pad_token_id=tokenizer.pad_token_id
            ),
        )
        if not result["target_met"]:
            raise ValueError("full actual supervised exposure target not met")
        model.eval()
        query = torch.tensor([validation[0]["input_ids"][:64]], dtype=torch.long, device="cuda:0")
        greedy = run_fixed_stage(
            "greedy_probe",
            lambda: model.generate(
                input_ids=query, max_new_tokens=16, do_sample=False, use_cache=True
            ).tolist(),
        )
        checkpoint = run_fixed_stage(
            "checkpoint_save",
            lambda: save_adapter_checkpoint(
                model,
                Path(FIXED_OUTPUT),
                {
                    "recipe_id": recipe_id,
                    "optimizer_steps": result["optimizer_steps"],
                    "supervised_tokens": result["supervised_tokens"],
                    "greedy_input_ids": query.tolist(),
                    "greedy_output_ids": greedy,
                    "source_commit": source_commit,
                    "stage_manifest_sha256": stage_manifest_sha256,
                },
                output_fd=output_fd,
            ),
        )
        training_receipt = {
            "schema_version": 1,
            "operation": "train",
            "source_commit": source_commit,
            "catalog_id": "qwen3_5_9b",
            "catalog_version": "1",
            "recipe_id": recipe_id,
            "input_summary": input_summary,
            "privilege_proof": privilege_proof,
            "metrics": result,
            "status": "checkpoint_saved",
            "reload_pending": True,
            "quality_success": False,
        }
        training_ref = run_fixed_stage(
            "training_receipt", lambda: _publish_fixed_json(output_fd, "TRAINING.json", training_receipt)
        )
        checkpoint_raw = _checkpoint_json(checkpoint)
        files = {
            **checkpoint["files"],
            "CHECKPOINT.json": {
                "sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
                "byte_count": len(checkpoint_raw),
            },
        }
        files["TRAINING.json"] = training_ref
        run_fixed_stage("final_readback", base_inputs.verify)
        run_fixed_stage(
            "final_publication",
            lambda: _publish_fixed_json(
                output_fd,
                "COMMITTED.json",
                {
                    "schema_version": 1,
                    "operation": "train",
                    "source_commit": source_commit,
                    "files": files,
                    "status": "checkpoint_saved",
                    "reload_pending": True,
                    "quality_success": False,
                },
            ),
        )
        return training_receipt
    finally:
        if adapter_inputs is not None:
            adapter_inputs.close()
        base_inputs.close()
