#[test]
fn sft_stacks_tcp_policy_and_denials_after_fd_handoff_before_exec() {
    let source = include_str!("../src/main.rs");
    let block = source
        .rsplit("if profile == WorkerProfile::QwenSft {")
        .next()
        .unwrap();
    let handoff = block.find("handoff_model_fds(output, inventory)?").unwrap();
    let proof = block.find("native_startup_proof(initial_bnd)?").unwrap();
    let policy = block
        .find("apply_qwen_tcp_boundary(&native_state()?)")
        .expect("SFT TCP restriction must precede Python");
    let denials = block
        .find("verify_qwen_tcp_endpoint_denial")
        .expect("fixed native denials required");
    let metadata = block.find("qwen_tcp_boundary_startup_proof").unwrap();
    let exec = block
        .find("Command::new(\"/usr/local/bin/python3.11\")")
        .unwrap();
    assert!(
        handoff < proof
            && proof < policy
            && policy < denials
            && denials < metadata
            && metadata < exec
    );
    assert!(!block[..exec].contains("apply_model_tcp_boundary"));
    assert!(!block[..exec].contains("apply_model_file_boundary"));
}
