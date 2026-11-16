use super::*;
use std::cell::RefCell;

fn proof() -> QwenTcpBoundaryProof {
    QwenTcpBoundaryProof {
        profile: "qwen_sft_tcp_deny_v1",
        abi: 6,
        handled_access_net: 3,
        allowed_connect_tcp_ports: [],
        allowed_bind_tcp_ports: [],
    }
}
#[test]
fn policy_check_or_proof_failure_has_zero_exec_calls() {
    for failure in ["install", "checks", "proof"] {
        let events = RefCell::new(vec![]);
        let result = qwen_tcp_startup_gate(
            || {
                events.borrow_mut().push("install");
                if failure == "install" {
                    Err("fixed_install_failure")
                } else {
                    Ok(proof())
                }
            },
            || {
                events.borrow_mut().push("checks");
                if failure == "checks" {
                    Err("fixed_checks_failure")
                } else {
                    Ok(())
                }
            },
            |_| {
                events.borrow_mut().push("proof");
                if failure == "proof" {
                    Err("fixed_proof_failure")
                } else {
                    Ok(())
                }
            },
            || {
                events.borrow_mut().push("exec");
                Ok(())
            },
        );
        assert!(result.is_err());
        assert!(
            !events.borrow().contains(&"exec"),
            "refusal reached exec sentinel"
        );
        assert_eq!(
            events.into_inner(),
            match failure {
                "install" => vec!["install"],
                "checks" => vec!["install", "checks"],
                _ => vec!["install", "checks", "proof"],
            }
        );
    }
}
#[test]
fn all_fixed_checks_reach_exec_once_and_keep_its_failure() {
    let events = RefCell::new(vec![]);
    let result = qwen_tcp_startup_gate(
        || {
            events.borrow_mut().push("install");
            Ok(proof())
        },
        || {
            events.borrow_mut().push("checks");
            Ok(())
        },
        |value| {
            assert_eq!(value, &proof());
            events.borrow_mut().push("proof");
            Ok(())
        },
        || {
            events.borrow_mut().push("exec");
            Err("fixed_qwen_worker_exec")
        },
    );
    assert_eq!(result, Err("fixed_qwen_worker_exec"));
    assert_eq!(
        events.into_inner(),
        vec!["install", "checks", "proof", "exec"]
    );
}
