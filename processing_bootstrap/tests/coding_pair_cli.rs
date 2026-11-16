//! Compiled fixed pair startup must reach trusted paths and preserve provider refusal.
use std::process::Command;
fn invoke(extra: Option<(&str, &str)>, hint: Option<(&str, &str)>) -> String {
    let mut values = vec!["coding-pair".to_owned()];
    for (key, value) in [
        ("binding-sha256", "a".repeat(64)),
        ("binding-byte-count", "128".to_owned()),
        ("source-commit", "c".repeat(40)),
    ] {
        values.extend([format!("--{key}"), value]);
    }
    if let Some((flag, value)) = extra {
        values.extend([flag.to_owned(), value.to_owned()]);
    }
    let mut command = Command::new(env!("CARGO_BIN_EXE_amlas-processing-init"));
    command
        .args(values)
        .env_clear()
        .env("AMLAS_SOURCE_COMMIT", "c".repeat(40))
        .env("AMLAS_CODING_PAIR_SOURCE_COMMIT", "c".repeat(40))
        .env("AMLAS_CODING_PAIR_LAUNCHER_SHA256", "d".repeat(64));
    if let Some((key, value)) = hint {
        command.env(key, value);
    }
    let result = command.output().unwrap();
    assert_eq!(result.status.code(), Some(78));
    String::from_utf8(result.stderr).unwrap()
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn native_fixed_pair_reaches_trusted_path_gate() {
    assert!(invoke(None, None).contains("stage=fixed_open"));
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn native_pair_role_relative_uri_reaches_trusted_path_gate() {
    assert!(invoke(
        None,
        Some((
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "/v2/credentials/native-role"
        ))
    )
    .contains("stage=fixed_open"));
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn pair_rejects_other_credential_and_proxy_hints() {
    for key in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_EC2_METADATA_SERVICE_ENDPOINT",
        "HTTPS_PROXY",
    ] {
        let error = invoke(None, Some((key, "inert")));
        assert!(
            error.contains("stage=inspection_credential_hint")
                || error.contains("stage=inspection_proxy_hint"),
            "{key}: {error}"
        );
    }
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn pair_refuses_invalid_relative_uri() {
    for value in [
        "",
        "//other",
        "/v2/credentials/../other",
        "/v2/credentials/other?query",
        "/other",
    ] {
        assert!(invoke(
            None,
            Some(("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", value))
        )
        .contains("stage=pair_relative_credential_hint"));
    }
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn pair_denies_paths_policy_code_and_duplicate_selectors() {
    for flag in [
        "--entrypoint",
        "--output-dir",
        "--source-path",
        "--policy",
        "--oracle",
        "--binding-sha256",
    ] {
        assert!(invoke(Some((flag, "caller")), None).contains("stage=worker_arguments"));
    }
}
#[cfg(not(feature = "sagemaker-native-v1"))]
#[test]
fn strict_build_has_no_coding_pair_profile() {
    assert!(invoke(None, None).contains("stage=fixed_profile"));
}
