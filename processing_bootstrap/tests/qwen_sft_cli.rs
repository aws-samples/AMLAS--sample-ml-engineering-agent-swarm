//! Fixed training startup ABI; compiler fixtures never execute an ML worker.
use std::process::Command;

fn arguments(operation: &str, recipe: &str) -> Vec<String> {
    let mut result = vec!["qwen-sft".to_owned()];
    for (key, value) in [
        ("operation", operation.to_owned()),
        ("recipe-id", recipe.to_owned()),
        ("recipe-version", "1".to_owned()),
        ("source-commit", "c".repeat(40)),
        ("manifest-sha256", "a".repeat(64)),
        ("manifest-byte-count", "262144".to_owned()),
    ] {
        result.extend([format!("--{key}"), value]);
    }
    result
}

fn invoke(values: Vec<String>, hint: Option<&str>) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_amlas-processing-init"));
    command
        .args(values)
        .env_clear()
        .env("AMLAS_SOURCE_COMMIT", "c".repeat(40));
    if let Some(name) = hint {
        command.env(name, "fixture");
    }
    let output = command.output().unwrap();
    assert_eq!(output.status.code(), Some(78));
    String::from_utf8(output.stderr).unwrap()
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn fixed_train_and_reload_pairs_reach_the_trusted_code_gate() {
    for operation in ["train", "reload"] {
        for recipe in ["fit_v1", "full_v1"] {
            assert!(invoke(arguments(operation, recipe), None).contains("stage=fixed_open"));
        }
    }
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn training_startup_rejects_mutable_code_and_authority_selectors() {
    for flag in [
        "--url",
        "--image-uri",
        "--role-arn",
        "--model-path",
        "--output-dir",
        "--module",
        "--executable",
        "--fd",
        "--policy",
        "--operation",
    ] {
        let mut values = arguments("train", "fit_v1");
        values.extend([flag.to_owned(), "caller".to_owned()]);
        assert!(invoke(values, None).contains("stage=worker_arguments"));
    }
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn training_startup_checks_closed_binding_values_before_paths_or_output() {
    for (flag, invalid) in [
        ("--operation", "evaluate"),
        ("--recipe-id", "caller_recipe"),
        ("--recipe-version", "2"),
        ("--manifest-sha256", "caller"),
        ("--manifest-byte-count", "0"),
        ("--manifest-byte-count", "262145"),
        ("--manifest-byte-count", "01"),
        ("--manifest-byte-count", "-1"),
        ("--manifest-byte-count", "1.0"),
    ] {
        let mut values = arguments("train", "fit_v1");
        let index = values.iter().position(|value| value == flag).unwrap() + 1;
        values[index] = invalid.to_owned();
        assert!(invoke(values, None).contains("stage=worker_arguments"));
    }
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn training_provider_hints_refuse_before_environment_clearing() {
    for hint in [
        "AWS_ACCESS_KEY_ID",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_EC2_METADATA_DISABLED",
        "HTTPS_PROXY",
        "https_proxy",
    ] {
        let message = invoke(arguments("train", "fit_v1"), Some(hint));
        assert!(
            message.contains("stage=inspection_credential_hint")
                || message.contains("stage=inspection_proxy_hint")
        );
    }
}

#[cfg(not(feature = "sagemaker-native-v1"))]
#[test]
fn strict_build_has_no_training_profile() {
    assert!(invoke(arguments("train", "fit_v1"), None).contains("stage=fixed_profile"));
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn fixed_generation_operation_uses_only_its_source_owned_recipe() {
    assert!(invoke(arguments("generate", "humaneval_greedy_v1"), None).contains("stage=fixed_open"));
    for (operation, recipe) in [
        ("train", "humaneval_greedy_v1"),
        ("reload", "humaneval_greedy_v1"),
        ("generate", "fit_v1"),
        ("generate", "full_v1"),
    ] {
        assert!(invoke(arguments(operation, recipe), None).contains("stage=worker_arguments"));
    }
}
