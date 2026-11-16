//! Actual one-thread compiled initializer; the Rust test harness never drops itself.
#![cfg(feature = "sagemaker-native-v1")]
use std::fs;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::process::Command;
extern "C" {
    fn setgroups(count: usize, groups: *const u32) -> i32;
    fn fchown(fd: i32, uid: u32, gid: u32) -> i32;
}
fn dir(path: &str) {
    fs::create_dir_all(path).unwrap();
    let fd = fs::File::open(path).unwrap();
    assert_eq!(unsafe { fchown(fd.as_raw_fd(), 0, 0) }, 0);
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
#[test]
fn fixed_pair_binary_initializes_fd_and_permanently_drops_before_child() {
    for path in [
        "/app",
        "/app/src",
        "/opt",
        "/opt/ml",
        "/opt/ml/processing",
        "/opt/ml/processing/input",
        "/opt/ml/processing/output",
        "/opt/venv/bin",
        "/opt/venv/lib/python3.11/site-packages",
        "/usr/local/bin",
        "/usr/local/lib/python3.11",
    ] {
        dir(path);
    }
    fs::copy(
        env!("CARGO_BIN_EXE_amlas-processing-init"),
        "/usr/local/bin/amlas-processing-init",
    )
    .unwrap();
    fs::set_permissions(
        "/usr/local/bin/amlas-processing-init",
        fs::Permissions::from_mode(0o555),
    )
    .unwrap();
    // Fixed OS stand-in only: no application or candidate code. Record actual child identity,
    // NNP, group and original output-FD writability through inherited fixed paths.
    fs::write("/usr/local/bin/python3.11", "#!/bin/sh\nset -eu\ntest \"$(id -u)\" = 10001\ntest \"$(id -g)\" = 10001\ntest \"$(id -G)\" = 10001\ntest \"$AWS_CONTAINER_CREDENTIALS_RELATIVE_URI\" = /v2/credentials/native-role\nawk '/^NoNewPrivs:/ {if ($2 != 1) exit 1; seen=1} END {if (!seen) exit 1}' /proc/self/status\nmkdir /opt/ml/processing/output/INVOCATIONS\nprintf PAIR_FIXED_CHILD_OK\n").unwrap();
    fs::set_permissions(
        "/usr/local/bin/python3.11",
        fs::Permissions::from_mode(0o555),
    )
    .unwrap();
    symlink("python3.11", "/usr/local/bin/python3").unwrap();
    symlink("python3", "/usr/local/bin/python").unwrap();
    symlink("/usr/local/bin/python", "/opt/venv/bin/python").unwrap();
    fs::write("/app/coding_pair_worker.py", "fixed-unused-wrapper").unwrap();
    fs::write("/app/processing_identity.py", "fixed-unused-identity").unwrap();
    let mut command = Command::new("/usr/local/bin/amlas-processing-init");
    unsafe {
        command.pre_exec(|| {
            if setgroups(0, std::ptr::null()) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let result = command
        .args([
            "coding-pair",
            "--binding-sha256",
            &"a".repeat(64),
            "--binding-byte-count",
            "128",
            "--source-commit",
            &"c".repeat(40),
        ])
        .env_clear()
        .env("AMLAS_SOURCE_COMMIT", "c".repeat(40))
        .env("AMLAS_CODING_PAIR_SOURCE_COMMIT", "c".repeat(40))
        .env("AMLAS_CODING_PAIR_LAUNCHER_SHA256", "d".repeat(64))
        .env(
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "/v2/credentials/native-role",
        )
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8(result.stdout).unwrap();
    assert!(text.contains("\"groups\":[10001]"));
    assert!(text.contains("\"cap_eff\":0"));
    assert!(text.contains("\"no_new_privs\":1"));
    assert!(text.contains("PAIR_FIXED_CHILD_OK"));
    assert!(fs::metadata("/opt/ml/processing/output/INVOCATIONS")
        .unwrap()
        .is_dir());
}
