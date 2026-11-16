//! Fixed model downloader arguments and pre-clear environment refusal.
use std::process::Command;

fn arguments() -> Vec<String> {
    let mut result = vec!["model-acquisition".to_owned()];
    for key in [
        "catalog-id",
        "catalog-version",
        "job-name",
        "runtime-session-id",
        "deployment-id",
        "deployment-prefix",
        "image-uri",
        "source-commit",
    ] {
        result.push(format!("--{key}"));
        result.push(match key {
            "source-commit" => "c".repeat(40),
            "catalog-id" => "qwen3_5_9b".to_owned(),
            "catalog-version" => "1".to_owned(),
            _ => "fixture".to_owned(),
        });
    }
    result
}

fn invoke(arguments: Vec<String>, hint: Option<&str>) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_amlas-processing-init"));
    command
        .args(arguments)
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
fn exact_model_pairs_reach_the_fixed_trusted_path_gate() {
    // The compiler fixture has no fixed image paths and executes no downloader.
    assert!(invoke(arguments(), None).contains("stage=fixed_open"));
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn model_profile_rejects_paths_grants_and_argument_drift() {
    for flag in [
        "--url",
        "--filename",
        "--entrypoint",
        "--output-dir",
        "--role-arn",
        "--byte-cap",
        "--fd",
        "--policy",
        "--target-column",
        "--catalog-id",
    ] {
        let mut values = arguments();
        values.extend([flag.to_owned(), "caller".to_owned()]);
        assert!(invoke(values, None).contains("stage=worker_arguments"));
    }
    let mut missing = arguments();
    missing.truncate(missing.len() - 2);
    assert!(invoke(missing, None).contains("stage=worker_arguments"));
    let mut control = arguments();
    control[2] = "caller\nvalue".to_owned();
    assert!(invoke(control, None).contains("stage=worker_arguments"));
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn model_provider_and_proxy_hints_refuse_before_environment_clearing() {
    for hint in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_EC2_METADATA_DISABLED",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ] {
        let message = invoke(arguments(), Some(hint));
        if hint == "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" {
            assert!(
                message.contains("stage=model_relative_credential_hint"),
                "malformed relative endpoint did not refuse specifically: {message}"
            );
        } else {
            assert!(
                message.contains("stage=inspection_credential_hint")
                    || message.contains("stage=inspection_proxy_hint"),
                "hint {hint} bypassed its pre-clear gate: {message}"
            );
        }
    }
}

#[cfg(not(feature = "sagemaker-native-v1"))]
#[test]
fn strict_build_has_no_model_downloader_profile() {
    assert!(invoke(arguments(), None).contains("stage=fixed_profile"));
}
