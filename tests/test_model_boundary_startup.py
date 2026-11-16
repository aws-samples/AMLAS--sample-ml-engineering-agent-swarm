"""Fixed model-only startup integration; native enforcement remains separate."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_model_inventory_capture_precedes_drop_and_policy_precedes_exec():
    source = (ROOT / "processing_bootstrap/src/main.rs").read_text()
    run = source[source.index("fn run()") :]
    for symbol in (
        "capture_provider_inventory()",
        "handoff_model_fds(output, inventory)",
        "apply_model_file_boundary",
    ):
        assert symbol in run, "missing-symbol: " + symbol
    capture = run.index("capture_provider_inventory()")
    drop = run.index("drop_native_privileges_for_profile(initial_bnd, profile)")
    handoff = run.index("handoff_model_fds(output, inventory)")
    policy = run.index("apply_model_file_boundary")
    execution = run.index('Command::new("/usr/local/bin/python3.11")')
    assert capture < drop < handoff < policy < execution
    assert "model_file_boundary_startup_proof" in run


def test_fixed_model_worker_requires_inventory_fd4_before_package_loading():
    source = (ROOT / "src/amlas/sagemaker_entrypoints/model_weight_acquisition.py").read_text()
    guard = source[source.index("def _guard_worker") : source.index("def _load_core")]
    assert "_read_provider_inventory(4)" in guard
    assert "_credential_observation_v2" in guard
    assert "descriptor <= 4" in guard or "int(name) <= 4" in guard
    main = source[source.index("def main()") :]
    assert "_credential_observation_v2" in main


def test_model_temp_paths_are_inside_only_the_fixed_output_tree():
    source = (ROOT / "processing_bootstrap/src/main.rs").read_text()
    environment = source[source.index("fn child_environment") : source.index("fn bounded_os_text")]
    assert "profile == WorkerProfile::ModelAcquisition" in environment
    assert environment.count('"/opt/ml/processing/output"') >= 2


def test_profile_drop_preserves_legacy_helper_and_permanent_native_guards():
    source = (ROOT / "processing_bootstrap/src/main.rs").read_text()
    legacy = source[source.index("fn drop_native_privileges(initial_bnd:") :]
    legacy = legacy[: legacy.index("fn native_startup_proof(")]
    assert "drop_native_privileges_for_profile(initial_bnd, WorkerProfile::CaseFreeze)" in legacy
    native = source[source.index("fn drop_native_privileges_for_profile(") :]
    native = native[: native.index("fn native_startup_proof_for_profile(")]
    for guard in (
        "setgroups(groups.len(), groups.as_ptr())",
        "setresgid(USER, USER, USER)",
        "setresuid(USER, USER, USER)",
        "capset(&header, data.as_ptr())",
        "validate_native_final_for_profile(&native_state()?, initial_bnd, profile)",
    ):
        assert guard in native, "missing permanent native guard: " + guard
    assert "WorkerProfile::CodingPair" in native
    assert "&[USER][..]" in native
    assert "&[][..]" in native
