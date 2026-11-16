"""Pytest plugin: run existing control tests on environments that went through the runtime pack.

Loaded only by ``test_runtime_env_budget.py`` in a nested run (``-p``). Each startup
reader of the packed keys receives the environment after ``pack`` and
``expand_environ``, as the managed runtime does, and the plugin asserts that the
expanded map equals the plain one before the reader runs. The counts written at the
end show which readers received a packed environment.
"""

from __future__ import annotations

import functools
import importlib
import json
import os

COUNTS: dict[str, int] = {}
READERS = (
    ("amlas.control_plane.sft_binding", "build_sft_store"),
    ("amlas.control_plane.generation_binding", "build_generation_store"),
    ("amlas.control_plane.paired_binding", "build_pair_store"),
    ("amlas.control_plane.case_binding", "build_case_freeze_store"),
    ("amlas.control_plane.preparation_binding", "build_preparation_store"),
    ("amlas.config.externalized", "model_weight_acquisition_selectors"),
)
# Modules that import a reader by name at import time hold their own reference.
HOLDERS = ("amlas.tui.chat.tool_binding_backend", "amlas.flows.model_weight_acquisition_flow")


def _count(name: str) -> None:
    COUNTS[name] = COUNTS.get(name, 0) + 1


def _through_pack(name: str, environment):
    """Return the environment the runtime would see after expansion, or the input if it cannot be packed."""
    from amlas import _runtime_env_pack as codec

    try:
        plain = dict(environment)
        packed = codec.pack(plain)
    except (TypeError, ValueError):
        _count(name + ":not-packable")
        return environment
    expanded = dict(packed)
    codec.expand_environ(expanded)
    if expanded != plain:
        raise AssertionError(f"{name}: expanded environment differs from the plain selection")
    _count(name + (":packed" if codec.PACK_KEY in packed else ":plain"))
    return expanded


def _reader(name: str, original):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if kwargs.get("env") is None:
            _through_pack(name + ":environ", os.environ)
        else:
            kwargs["env"] = _through_pack(name, kwargs["env"])
        return original(*args, **kwargs)

    return wrapped


def _policy(original):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        _through_pack("_runtime_prompt_policy:environ", os.environ)
        return original(*args, **kwargs)

    return wrapped


def pytest_configure(config):
    replaced = {}
    for module_name, attribute in READERS:
        module = importlib.import_module(module_name)
        original = getattr(module, attribute)
        replaced[id(original)] = (original, _reader(attribute, original))
        setattr(module, attribute, replaced[id(original)][1])
    for module_name in HOLDERS:
        module = importlib.import_module(module_name)
        for key, value in list(vars(module).items()):
            pair = replaced.get(id(value))
            if pair is not None and pair[0] is value:
                setattr(module, key, pair[1])
    app = importlib.import_module("amlas.app")
    app._runtime_prompt_policy = _policy(app._runtime_prompt_policy)


def pytest_unconfigure(config):
    target = os.environ.get("RUNTIME_ENV_PACK_CONTROLS_COUNTS")
    if target:
        with open(target, "w", encoding="utf-8") as stream:
            json.dump(COUNTS, stream, sort_keys=True)
