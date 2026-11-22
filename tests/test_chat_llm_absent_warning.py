"""The no-LLM chat warning names the package and where it is declared, and states no version.

A version requirement written into a log text goes stale with the next change of
``pyproject.toml``. The warning points at the declaration; only the declaration holds a number.
"""

from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path

import pytest

from amlas.tui.chat import engine

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "strands-agents"
PLACE = "pyproject.toml [project.dependencies]"
# The comparison operators of a version requirement, and a dotted run of numbers.
REQUIREMENT_OPERATOR = re.compile(r"===|~=|==|!=|<=|>=|<|>")
DOTTED_NUMBER = re.compile(r"\d+(?:\.\d+)+")
# A number, an operator or an extras bracket directly behind the package name.
SPECIFIER_AFTER_NAME = re.compile(re.escape(PACKAGE) + r"\s*[\d<>=!~\[]")


def logged_texts(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == engine.logger.name and record.levelno == logging.WARNING
    ]


@pytest.fixture
def warning_text(monkeypatch, caplog) -> str:
    monkeypatch.setattr(engine, "_WARNED_CHAT_LLM_ABSENT", False)
    caplog.set_level(logging.WARNING, logger=engine.logger.name)
    engine._warn_chat_llm_absent()
    texts = logged_texts(caplog)
    assert len(texts) == 1
    return texts[0]


def test_the_warning_names_the_package_and_where_it_is_declared(warning_text):
    assert PACKAGE in warning_text
    assert PLACE in warning_text


def test_the_warning_states_no_version_requirement(warning_text):
    assert REQUIREMENT_OPERATOR.findall(warning_text) == []
    assert DOTTED_NUMBER.findall(warning_text) == []
    assert SPECIFIER_AFTER_NAME.findall(warning_text) == []


def test_the_named_place_declares_the_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = [re.split(r"[\s\[<>=!~;]", entry, maxsplit=1)[0] for entry in project["dependencies"]]
    assert names.count(PACKAGE) == 1


def test_the_warning_is_logged_once_until_it_is_armed_again(monkeypatch, caplog):
    monkeypatch.setattr(engine, "_WARNED_CHAT_LLM_ABSENT", False)
    caplog.set_level(logging.WARNING, logger=engine.logger.name)
    engine._warn_chat_llm_absent()
    engine._warn_chat_llm_absent()
    assert len(logged_texts(caplog)) == 1
    engine.reset_degradation_warning()
    engine._warn_chat_llm_absent()
    texts = logged_texts(caplog)
    assert len(texts) == 2
    assert texts[0] == texts[1]
