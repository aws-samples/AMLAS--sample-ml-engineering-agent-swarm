//! Fixed source-bound case freezing without workload execution in compiler tests.
use std::process::Command;
fn invoke(extra: Option<(&str, &str)>) -> String {
    let mut values = vec!["case-freeze".to_owned()];
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
    let result = Command::new(env!("CARGO_BIN_EXE_amlas-processing-init"))
        .args(values)
        .env_clear()
        .env("AMLAS_SOURCE_COMMIT", "c".repeat(40))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(78));
    String::from_utf8(result.stderr).unwrap()
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn native_fixed_case_cli_reaches_original_trusted_path_gate() {
    assert!(invoke(None).contains("stage=fixed_open"));
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn case_cli_denies_code_path_or_grant_arguments() {
    for flag in [
        "--entrypoint",
        "--output-dir",
        "--source-path",
        "--policy",
        "--oracle",
        "--binding-sha256",
    ] {
        assert!(invoke(Some((flag, "caller"))).contains("stage=worker_arguments"));
    }
}
#[cfg(not(feature = "sagemaker-native-v1"))]
#[test]
fn strict_build_has_no_case_freeze_profile() {
    assert!(invoke(None).contains("stage=fixed_profile"));
}
