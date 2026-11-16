"""Fixed Qwen inventory startup; no GPU/native admission claim."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_qwen_captures_before_drop_and_hands_same_sealed_fds_without_model_policy():
    source = (ROOT / "processing_bootstrap/src/main.rs").read_text()
    run = source[source.index("fn run()") :]
    capture = run[run.index("let provider_inventory =") : run.index("let output =")]
    assert "WorkerProfile::QwenSft" in capture, "Qwen missing original provider capture before drop"
    branch = run[run.index("if profile == WorkerProfile::QwenSft {") :]
    # Output initialization occurs first; require a second explicit post-drop branch.
    branch = branch[branch.index("drop_native_privileges_for_profile(initial_bnd, profile)") :]
    assert "if profile == WorkerProfile::QwenSft {" in branch, "Qwen missing sealed post-drop handoff"
    trainer = branch[branch.index("if profile == WorkerProfile::QwenSft {") :]
    trainer = trainer[: trainer.index("handoff_output(output)")]
    assert "handoff_model_fds(output, inventory)" in trainer
    assert "native_startup_proof(initial_bnd)" in trainer
    assert 'Command::new("/usr/local/bin/python3.11")' in trainer
    assert "apply_model_file_boundary" not in trainer, (
        "CUDA training cannot inherit unqualified downloader policy"
    )
