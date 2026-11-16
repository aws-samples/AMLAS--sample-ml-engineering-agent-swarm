"""Source-owned TCP gate order; not native port/credential-isolation proof."""
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_fixed_model_tcp_policy_and_endpoint_denial_precede_worker_exec():
    source = (ROOT / "processing_bootstrap/src/main.rs").read_text()
    run = source[source.index("fn run()") :]
    model = run[run.index("if profile == WorkerProfile::ModelAcquisition {") :]
    model = model[:model.index("if profile == WorkerProfile::QwenSft {")]
    assert "apply_model_tcp_boundary(&native_state()?)" in model
    assert "verify_model_credential_endpoint_denial()" in model
    assert model.index("apply_model_tcp_boundary") < model.index("verify_model_credential_endpoint_denial") < model.index('Command::new("/usr/local/bin/python3.11")')
    assert "model_tcp_boundary_startup_proof" in model


def test_only_native_model_relative_endpoint_is_deferred_and_never_forwarded():
    source = (ROOT / "processing_bootstrap/src/main.rs").read_text()
    environment = source[source.index("fn child_environment") : source.index("fn bounded_os_text")]
    assert "defer_model_relative_credential_hint(profile, key, value)?" in environment
    assert "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" not in environment[environment.index("if key == \"AMLAS_SOURCE_COMMIT\"") :]
    assert "if !acquisition && credentials" in " ".join(environment.split())
