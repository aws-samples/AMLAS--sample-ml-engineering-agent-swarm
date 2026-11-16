//! Fixed primitive preparation CLI admission before any output mutation.
use std::process::Command;
fn invoke(extra: Option<(&str, &str)>) -> String {
    let mut arguments = vec!["preparation".to_owned()];
    for key in [
        "selection-sha256",
        "selection-bytes",
        "exclusions-sha256",
        "exclusions-bytes",
        "job-name",
        "runtime-session-id",
        "deployment-id",
        "deployment-prefix",
        "image-uri",
        "source-commit",
        "profile-id",
        "policy-version",
    ] {
        arguments.push(format!("--{key}"));
        arguments.push(match key {
            "source-commit" => "c".repeat(40),
            "profile-id" => "openai.explicit_conversations.v1".to_owned(),
            "policy-version" => "coding-preparation.v1".to_owned(),
            _ => "fixture".to_owned(),
        });
    }
    if let Some((key, value)) = extra {
        arguments.extend([key.to_owned(), value.to_owned()]);
    }
    let output = Command::new(env!("CARGO_BIN_EXE_amlas-processing-init"))
        .args(arguments)
        .env_clear()
        .env("AMLAS_SOURCE_COMMIT", "c".repeat(40))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(78));
    String::from_utf8(output.stderr).unwrap()
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn fixed_preparation_profile_accepts_exact_primitive_pairs() {
    // The compiler container has no fixed worker image; successful CLI admission
    // must reach the subsequent trusted-path check and never execute data code.
    assert!(invoke(None).contains("stage=fixed_open"));
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn fixed_preparation_profile_rejects_code_path_and_grant_flags() {
    for flag in [
        "--entrypoint",
        "--output-dir",
        "--source-path",
        "--policy",
        "--target-column",
    ] {
        assert!(invoke(Some((flag, "caller"))).contains("stage=worker_arguments"));
    }
}

#[cfg(not(feature = "sagemaker-native-v1"))]
#[test]
fn strict_build_has_no_preparation_profile() {
    assert!(invoke(None).contains("stage=fixed_profile"));
}
