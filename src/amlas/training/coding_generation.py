"""Fixed held-object HumanEval generation; no loaders, execution or authority."""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
import re
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal

from ..data_eng.coding_case_profile import CASE_IDS, MAX_DECODED_BYTES, MAX_FIELD_BYTES
from ..data_eng.model_weight_manifest import expected_model_weight_manifest

_WRAPPER = (
    "Implement the following Python function. Return one complete Python module containing "
    "the requested entry point. Return Python source only, without Markdown fences or explanation.\n\n"
)
_CONTRACT = {
    "profile_id": "openai.humaneval.164.qwen.greedy.module.v1",
    "profile_version": "1",
    "case_count": 164,
    "seed": 12345,
    "do_sample": False,
    "context_tokens": 4096,
    "max_new_tokens": 1024,
    "output_bytes": 65536,
    "num_return_sequences": 1,
    "enable_thinking": False,
    "add_generation_prompt": True,
    "tools": (),
    "wrapper": _WRAPPER,
    "normalization": "tokenizer_decode_skip_special_tokens_false_cleanup_false_remove_only_terminal_eos",
    "context_policy": "input_plus_fixed1024_budget_at_most4096",
    "eos_policy": "single_tokenizer_eos_terminal_only_cap_always_truncated",
    "retention_policy": "bounded_partial_source_and_observed_overflow_hash_count",
    "greedy_policy": "fresh_GenerationConfig_single_beam_no_forced_or_suppressed_tokens",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


CONTRACT_BYTES = canonical_bytes({**_CONTRACT, "tools": []})
CONTRACT = MappingProxyType(_CONTRACT)
CONTRACT_SHA256 = digest(CONTRACT_BYTES)


@dataclass(frozen=True, slots=True)
class CodingGenerationBinding:
    evaluation_id: str
    arm: Literal["base", "tuned"]
    prompts_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    spec_sha256: str


@dataclass(frozen=True, slots=True)
class CodingGenerationCase:
    case_id: str
    prompt_sha256: str
    rendered_sha256: str
    input_tokens: int
    generated_tokens: int
    outcome: str
    source: str | None
    source_sha256: str | None
    source_byte_count: int
    failure_stage: str = "none"


@dataclass(frozen=True, slots=True)
class CodingGenerationResult:
    binding: CodingGenerationBinding
    contract_sha256: str
    cases: tuple[CodingGenerationCase, ...]
    manifest_bytes: bytes
    manifest_sha256: str


def _prompts(values: Any) -> tuple[dict[str, str], ...]:
    if type(values) is not tuple or len(values) != 164:
        raise ValueError("complete ordered164 public prompts required")
    checked, count = [], 0
    for index, row in enumerate(values):
        if type(row) is not dict or set(row) != {"case_id", "prompt", "entry_point"}:
            raise ValueError("public prompt role fields only; oracle fields refused")
        if (
            any(type(value) is not str or not value for value in row.values())
            or row["case_id"] != CASE_IDS[index]
        ):
            raise ValueError("public case IDs/order/string fields differ")
        entry = row["entry_point"]
        if not entry.isascii() or not entry.isidentifier() or keyword.iskeyword(entry):
            raise ValueError("plain Python entry point required")
        sizes = [len(value.encode("utf-8")) for value in row.values()]
        count += sum(sizes)
        if max(sizes) > MAX_FIELD_BYTES or count > MAX_DECODED_BYTES:
            raise ValueError("public prompt byte bound exceeded")
        checked.append(dict(row))
    return tuple(checked)


def prompts_sha256(values: Any) -> str:
    return digest(b"".join(canonical_bytes(row) + b"\n" for row in _prompts(values)))


def _binding(value: Any, rows: tuple[dict[str, str], ...]) -> CodingGenerationBinding:
    if type(value) is not CodingGenerationBinding or value.arm not in ("base", "tuned"):
        raise ValueError("closed generation binding required")
    if type(value.evaluation_id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value.evaluation_id):
        raise ValueError("bounded evaluation identity required")
    for name in ("prompts_sha256", "model_sha256", "tokenizer_sha256", "spec_sha256"):
        field = getattr(value, name)
        if type(field) is not str or not re.fullmatch(r"[a-f0-9]{64}", field):
            raise ValueError("exact generation binding hashes required")
    if value.prompts_sha256 != prompts_sha256(rows):
        raise ValueError("original public PROMPTS bytes differ")
    return value


def _template(tokenizer: Any) -> None:
    expected = next(
        item["sha256"]
        for item in expected_model_weight_manifest("qwen3_5_9b", "1")["files"]
        if item["role"] == "chat_template"
    )
    template = tokenizer.chat_template
    if type(template) is not str or len(template.encode()) > 16384 or digest(template.encode()) != expected:
        raise ValueError("source-pinned native Qwen chat template differs")


def _valid_module(source: str, entry: str) -> bool:
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry for node in tree.body
    )


def generate_humaneval_arm(
    *,
    model: Any,
    tokenizer: Any,
    prompts: tuple[dict[str, str], ...],
    binding: CodingGenerationBinding,
) -> CodingGenerationResult:
    """Generate all164 cases once from held objects; binding hashes are descriptive.

    No object loading or weight/tokenizer authentication occurs in this pure seam.
    Source-held native input ownership and qualification belong to its caller.
    """
    rows = _prompts(prompts)
    bound = _binding(binding, rows)
    _template(tokenizer)
    if any(
        marker in row["prompt"] for row in rows for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
    ):
        raise ValueError("public prompt contains native chat role/termination markers")
    import torch
    from transformers import GenerationConfig

    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if type(eos) is not int or type(pad) is not int or eos < 0 or pad < 0:
        raise ValueError("single nonnegative tokenizer EOS/pad identity required")
    configuration = GenerationConfig(
        max_new_tokens=1024,
        min_new_tokens=0,
        min_length=0,
        do_sample=False,
        num_beams=1,
        num_beam_groups=1,
        num_return_sequences=1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        encoder_no_repeat_ngram_size=0,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        forced_decoder_ids=None,
        bad_words_ids=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        eos_token_id=eos,
        pad_token_id=pad,
        return_dict_in_generate=False,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True,
    )
    setup_failed = False
    try:
        torch.manual_seed(12345)
        model.eval()
    except Exception:
        setup_failed = True
    cases = []
    for row in rows:
        prompt_hash = digest(canonical_bytes(row))
        rendered_hash = ""
        input_count = generated_count = 0
        source, source_hash, byte_count, outcome = None, None, 0, "generation_error"
        stage = "setup" if setup_failed else "render"
        if setup_failed:
            cases.append(
                CodingGenerationCase(
                    row["case_id"], prompt_hash, rendered_hash, 0, 0, "setup_error", None, None, 0, "setup"
                )
            )
            continue
        try:
            messages = [{"role": "user", "content": _WRAPPER + row["prompt"]}]
            rendered = tokenizer.apply_chat_template(
                messages, tools=[], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            if type(rendered) is not str or len(rendered.encode()) > 65536:
                raise ValueError("rendered prompt bound differs")
            rendered_hash = digest(rendered.encode())
            stage = "tokenize"
            encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt", truncation=False)
            if set(encoded) != {"input_ids", "attention_mask"}:
                raise ValueError("closed tokenizer tensor fields required")
            query, mask = encoded["input_ids"], encoded["attention_mask"]
            if (
                not torch.is_tensor(query)
                or not torch.is_tensor(mask)
                or query.ndim != 2
                or query.shape[0] != 1
                or query.shape != mask.shape
                or query.dtype != torch.long
            ):
                raise ValueError("one complete tokenizer sequence required")
            input_count = query.shape[1]
            if input_count + 1024 > 4096:
                outcome = "context_overflow"
            elif input_count <= 0 or not torch.all(mask == 1):
                raise ValueError("complete unpadded input required")
            else:
                query, mask = query.to(model.device), mask.to(model.device)
                stage = "generate"
                with torch.inference_mode():
                    output = model.generate(
                        generation_config=configuration,
                        input_ids=query,
                        attention_mask=mask,
                        max_new_tokens=1024,
                        do_sample=False,
                        use_cache=True,
                        num_return_sequences=1,
                    )
                if (
                    not torch.is_tensor(output)
                    or output.dtype != torch.long
                    or output.ndim != 2
                    or output.shape[0] != 1
                    or output.shape[1] < input_count
                    or output.shape[1] > input_count + 1024
                    or not torch.equal(output[:, :input_count], query)
                ):
                    raise ValueError("generation output prefix/shape/bound differs")
                generated_count = output.shape[1] - input_count
                suffix = output[0, input_count:].tolist()
                stage = "decode"
                terminal = bool(suffix) and suffix[-1] == eos
                malformed = eos in suffix[:-1] or (pad != eos and pad in suffix)
                content_tokens = suffix[:-1] if terminal and not malformed else suffix
                decoded = tokenizer.decode(
                    content_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if type(decoded) is not str:
                    raise ValueError("decoded source must be UTF8 text")
                raw = decoded.encode("utf-8")
                source_hash, byte_count = digest(raw), len(raw)
                if len(raw) > 65536:
                    outcome = "output_overflow"
                else:
                    source = decoded
                    if malformed:
                        outcome = "invalid_termination"
                    elif generated_count >= 1024:
                        outcome = "truncated"
                    elif not terminal:
                        outcome = "incomplete"
                    else:
                        outcome = (
                            "generated" if raw and _valid_module(decoded, row["entry_point"]) else "invalid"
                        )
                if outcome == "generated":
                    stage = "none"
        except Exception:
            outcome = "generation_error"
        cases.append(
            CodingGenerationCase(
                row["case_id"],
                prompt_hash,
                rendered_hash,
                input_count,
                generated_count,
                outcome,
                source,
                source_hash,
                byte_count,
                stage,
            )
        )
    manifest = (
        canonical_bytes(
            {
                "schema_version": 1,
                "binding": asdict(bound),
                "contract_sha256": CONTRACT_SHA256,
                "case_count": 164,
                "cases": [
                    {key: value for key, value in asdict(case).items() if key != "source"} for case in cases
                ],
            }
        )
        + b"\n"
    )
    return CodingGenerationResult(bound, CONTRACT_SHA256, tuple(cases), manifest, digest(manifest))
