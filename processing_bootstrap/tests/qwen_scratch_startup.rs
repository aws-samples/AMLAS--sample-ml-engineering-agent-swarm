#[test]
fn source_fixed_scratch_is_separate_and_initialized_before_exec() {
    let source = include_str!("../src/main.rs");
    assert!(
        source.contains("/home/appuser/.cache/amlas-qwen-sft"),
        "fixed SFT TMPDIR must leave checkpoint output"
    );
    let block = source
        .rsplit("if profile == WorkerProfile::QwenSft {")
        .next()
        .unwrap();
    let handoff = block.find("handoff_model_fds(output, inventory)?").unwrap();
    let native = block.find("native_startup_proof(initial_bnd)?").unwrap();
    let scratch = block
        .find("initialize_qwen_scratch()?")
        .expect("fixed scratch initialized postdrop");
    let gate = block.find("return qwen_tcp_startup_gate(").unwrap();
    assert!(handoff < native && native < scratch && scratch < gate);
}
