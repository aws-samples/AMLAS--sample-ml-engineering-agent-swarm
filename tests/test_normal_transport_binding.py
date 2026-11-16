"""Server transport DATA ABI and SDK capture; no native/role/admission proof.

Actual pure prepare/configuration and SDK getter are used. Catalogs/identifiers
are fresh declarations; constructor data and matched hashes grant no authority.
No producer, model, current qualifier, native I/O or recovered snapshot is used.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from bedrock_agentcore.runtime import BedrockAgentCoreContext
from test_normal_checkpoint_cas_contract import _declared_enrollment
from test_runtime_busy_ownership import managed_report_context as managed_report_context
from test_runtime_prompt_entry import inert_runtime_setup as inert_runtime_setup

from amlas import app as runtime
from amlas.control_plane import normal_run_checkpoint as checkpoint

RUN_ID = "1" * 32
SESSION = "declared-original-sdk-transport-session-0123456789"


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def selected_role(monkeypatch):
    context, _record, _item = _declared_enrollment()
    catalog = context.catalog
    for name, value in {
        "AMLAS_AWS_ACCOUNT_ID": catalog.account_id,
        "AMLAS_AWS_REGION": catalog.region,
        "AWS_REGION": catalog.region,
        "AMLAS_DEPLOYMENT_ID": catalog.deployment_id,
        "AMLAS_DEPLOYMENT_PREFIX": catalog.deployment_prefix,
        "AMLAS_JOBS_TABLE": catalog.jobs_table,
        "AMLAS_RUNTIME_EXEC_ROLE_ARN": catalog.execution_role_arn,
        "AMLAS_RUNTIME_EXEC_ROLE_ID": catalog.execution_role_id,
    }.items():
        monkeypatch.setenv(name, value)
    return catalog


def test_server_prepare_retains_hash_matched_transport_without_ascii_or_uuid_pattern(selected_role):
    for transport in ("s" * 33, "s" * 256, "é" * 128):
        context = checkpoint.prepare_normal_run(RUN_ID, _hash(transport), transport_session_id=transport)
        assert context.catalog == selected_role
        assert context.run_id == RUN_ID and context.session_hash == _hash(transport)
        assert context.transport_session_id == transport
        direct = checkpoint.NormalRunContext(
            selected_role, RUN_ID, _hash(transport), transport_session_id=transport
        )
        assert direct.transport_session_id == transport and direct.session_hash == _hash(transport)
    omitted = checkpoint.prepare_normal_run(RUN_ID, _hash(SESSION))
    assert omitted.transport_session_id is None


def test_selected_role_rejects_invalid_transport_shapes_and_run_ids(selected_role):
    for transport in (True, [SESSION], "short", "s" * 257, "é" * 129, "s" * 33 + "\n"):
        session_hash = _hash(transport) if type(transport) is str else _hash(SESSION)
        with pytest.raises(ValueError):
            checkpoint.prepare_normal_run(RUN_ID, session_hash, transport_session_id=transport)
    for run_id in (True, [RUN_ID], "declared-invalid-run", "A" * 32):
        with pytest.raises(ValueError):
            checkpoint.prepare_normal_run(run_id, _hash(SESSION), transport_session_id=SESSION)


def test_selected_role_rejects_transport_hash_mismatch(selected_role):
    with pytest.raises(ValueError):
        checkpoint.prepare_normal_run(RUN_ID, _hash(SESSION + "-other"), transport_session_id=SESSION)


def test_absent_runtime_role_pair_preserves_legacy_none(monkeypatch):
    monkeypatch.delenv("AMLAS_RUNTIME_EXEC_ROLE_ARN", raising=False)
    monkeypatch.delenv("AMLAS_RUNTIME_EXEC_ROLE_ID", raising=False)
    assert checkpoint.prepare_normal_run(RUN_ID, _hash(SESSION), transport_session_id=SESSION) is None


def test_context_old_fourth_positional_argument_stays_process_id(selected_role):
    context = checkpoint.NormalRunContext(selected_role, RUN_ID, _hash(SESSION), 4321)
    assert context.process_id == 4321
    assert context.transport_session_id is None
    assert context.catalog == selected_role and context.session_hash == _hash(SESSION)


def test_actual_sdk_getter_hashes_original_once_and_public_payload_cannot_override(
    monkeypatch, managed_report_context
):
    managed_report_context.set_request_context("declared-transport-request", SESSION)
    actual_getter = BedrockAgentCoreContext.get_session_id
    observed = []

    def getter():
        value = actual_getter()
        observed.append(value)
        return value

    monkeypatch.setattr(BedrockAgentCoreContext, "get_session_id", staticmethod(getter))
    assert runtime._prompt_report_session_hash() == _hash(SESSION)
    assert observed == [SESSION]

    async def reject_override():
        return [
            event
            async for event in runtime.agent_invocation(
                {
                    "action": "get_prompt_run_status",
                    "run_id": RUN_ID,
                    "transport_session_id": "declared-public-override-session-0123456789",
                }
            )
        ]

    assert asyncio.run(reject_override()) == [{"event": "error", "error_code": "invalid_run_status_request"}]
    assert observed == [SESSION]
