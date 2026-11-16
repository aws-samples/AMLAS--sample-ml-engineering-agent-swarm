//! Fixed Linux AMD64 OS-only initialization. Never import workload code as root.
#[cfg(not(feature = "sagemaker-native-v1"))]
use amlas_processing_init::drop_bounding_capabilities;
#[cfg(feature = "sagemaker-native-v1")]
use amlas_processing_init::qwen_tcp_boundary::{
    apply_qwen_tcp_boundary, verify_qwen_tcp_endpoint_denial, QwenTcpBoundaryProof,
};
use amlas_processing_init::verify_named_directory;
#[cfg(feature = "sagemaker-native-v1")]
use amlas_processing_init::{
    model_fd_handoff::handoff_model_fds,
    model_file_boundary::{
        apply_model_file_boundary, capture_provider_inventory, FileBoundaryProof,
    },
    model_tcp_boundary::{apply_model_tcp_boundary, TcpBoundaryProof},
};
#[cfg(feature = "sagemaker-native-v1")]
use amlas_processing_init::{validate_native_final, validate_native_initial, NativeState};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CString, OsString};
use std::fs::{self, File};
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::CommandExt;
use std::process::Command;

type Result<T> = std::result::Result<T, &'static str>;
const USER: u32 = 10001;
const PARENT: &str = "/opt/ml/processing";
const DIRECTORY_FLAGS: i32 = 0o2600000; // O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC
const ACQUISITION: &[&str] = &[
    "repo-id",
    "revision",
    "filename",
    "job-name",
    "runtime-session-id",
    "deployment-id",
    "deployment-prefix",
    "image-uri",
    "source-commit",
];
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerProfile {
    Acquisition,
    Inspection,
    Preparation,
    CaseFreeze,
    CodingPair,
    ModelAcquisition,
    QwenSft,
}
const CASE_FREEZE: &[&str] = &["binding-sha256", "binding-byte-count", "source-commit"];
const QWEN_SFT: &[&str] = &[
    "operation",
    "recipe-id",
    "recipe-version",
    "source-commit",
    "manifest-sha256",
    "manifest-byte-count",
];
const MODEL_ACQUISITION: &[&str] = &[
    "catalog-id",
    "catalog-version",
    "job-name",
    "runtime-session-id",
    "deployment-id",
    "deployment-prefix",
    "image-uri",
    "source-commit",
];
const PREPARATION: &[&str] = &[
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
];
const INSPECTION: &[&str] = &[
    "sha256",
    "byte-count",
    "format",
    "job-name",
    "runtime-session-id",
    "deployment-id",
    "deployment-prefix",
    "image-uri",
    "source-commit",
    "source-version-id",
    "staged-version-id",
];

#[repr(C)]
#[derive(Default)]
struct StatVfs {
    bsize: u64,
    frsize: u64,
    blocks: u64,
    bfree: u64,
    bavail: u64,
    files: u64,
    ffree: u64,
    favail: u64,
    fsid: u64,
    flag: u64,
    namemax: u64,
    spare: [i32; 6],
}
#[repr(C)]
struct CapHeader {
    version: u32,
    pid: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CapData {
    effective: u32,
    permitted: u32,
    inheritable: u32,
}
extern "C" {
    fn openat(fd: i32, path: *const i8, flags: i32, ...) -> i32;
    fn fchown(fd: i32, uid: u32, gid: u32) -> i32;
    fn fchmod(fd: i32, mode: u32) -> i32;
    fn mkdirat(fd: i32, path: *const i8, mode: u32) -> i32;
    fn fstatvfs(fd: i32, out: *mut StatVfs) -> i32;
    fn flock(fd: i32, operation: i32) -> i32;
    fn setgroups(count: usize, groups: *const u32) -> i32;
    fn setresgid(real: u32, effective: u32, saved: u32) -> i32;
    fn setresuid(real: u32, effective: u32, saved: u32) -> i32;
    fn getresuid(real: *mut u32, effective: *mut u32, saved: *mut u32) -> i32;
    fn getresgid(real: *mut u32, effective: *mut u32, saved: *mut u32) -> i32;
    fn getgroups(count: i32, groups: *mut u32) -> i32;
    fn capset(header: *const CapHeader, data: *const CapData) -> i32;
    fn prctl(option: i32, ...) -> i32;
    fn close(fd: i32) -> i32;
    fn dup2(old: i32, new: i32) -> i32;
    fn fcntl(fd: i32, command: i32, ...) -> i32;
}

fn ids(expected: u32) -> Result<()> {
    let (mut a, mut b, mut c) = (0, 0, 0);
    if unsafe { getresuid(&mut a, &mut b, &mut c) } != 0
        || (a, b, c) != (expected, expected, expected)
    {
        return Err("uid_identity");
    }
    if unsafe { getresgid(&mut a, &mut b, &mut c) } != 0
        || (a, b, c) != (expected, expected, expected)
    {
        return Err("gid_identity");
    }
    Ok(())
}
fn open_at(parent: &File, name: &str, directory: bool) -> Result<File> {
    let name = CString::new(name).map_err(|_| "fixed_name")?;
    let flags = if directory {
        DIRECTORY_FLAGS
    } else {
        0o2400000
    };
    let fd = unsafe { openat(parent.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        return Err("fixed_open");
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}
fn root_owned(file: &File, directory: bool) -> Result<()> {
    let m = file.metadata().map_err(|_| "trusted_stat")?;
    if m.uid() != 0
        || m.gid() != 0
        || m.mode() & 0o022 != 0
        || (directory && !m.is_dir())
        || (!directory && !m.is_file())
    {
        return Err("trusted_code_owner");
    }
    Ok(())
}
fn fixed_root_path(path: &str, directory: bool) -> Result<File> {
    let mut current = File::open("/").map_err(|_| "root_open")?;
    root_owned(&current, true)?;
    let components: Vec<_> = path.split('/').filter(|s| !s.is_empty()).collect();
    for (i, component) in components.iter().enumerate() {
        let is_dir = i + 1 < components.len() || directory;
        current = open_at(&current, component, is_dir)?;
        root_owned(&current, is_dir)?;
    }
    Ok(current)
}
fn fixed_pair_interpreter() -> Result<()> {
    // The existing venv uses this exact immutable Python alias chain, never a caller path.
    fixed_root_path("/opt/venv/bin", true)?;
    fixed_root_path("/usr/local/bin", true)?;
    for (path, target) in [
        ("/opt/venv/bin/python", "/usr/local/bin/python"),
        ("/usr/local/bin/python", "python3"),
        ("/usr/local/bin/python3", "python3.11"),
    ] {
        let metadata = fs::symlink_metadata(path).map_err(|_| "pair_interpreter_alias")?;
        if !metadata.file_type().is_symlink()
            || metadata.uid() != 0
            || metadata.gid() != 0
            || fs::read_link(path)
                .map_err(|_| "pair_interpreter_alias")?
                .as_os_str()
                != target
        {
            return Err("pair_interpreter_alias");
        }
    }
    fixed_root_path("/usr/local/bin/python3.11", false)?;
    Ok(())
}
fn writable(file: &File) -> Result<()> {
    let mut v = StatVfs::default();
    if unsafe { fstatvfs(file.as_raw_fd(), &mut v) } != 0 || v.flag & 1 != 0 {
        return Err("readonly_filesystem");
    }
    Ok(())
}
fn empty(file: &File) -> Result<()> {
    let mut entries = fs::read_dir(format!("/proc/self/fd/{}", file.as_raw_fd()))
        .map_err(|_| "directory_read")?;
    if entries.next().is_some() {
        return Err("nonempty_output");
    }
    Ok(())
}
fn parent_identity(parent: &File) -> Result<()> {
    let named = open_at(&fixed_root_path("/opt/ml", true)?, "processing", true)?;
    let a = parent.metadata().map_err(|_| "parent_stat")?;
    let b = named.metadata().map_err(|_| "parent_name_stat")?;
    if (a.dev(), a.ino()) != (b.dev(), b.ino()) {
        return Err("parent_identity");
    }
    Ok(())
}
fn initialize_output(profile: WorkerProfile) -> Result<File> {
    let parent = open_at(&fixed_root_path("/opt/ml", true)?, "processing", true)?;
    let p = parent.metadata().map_err(|_| "parent_stat")?;
    let parent_owner = if matches!(
        profile,
        WorkerProfile::CaseFreeze | WorkerProfile::CodingPair
    ) {
        (0, 0)
    } else {
        (USER, USER)
    };
    if !p.is_dir()
        || (p.uid(), p.gid(), p.mode() & 0o7777) != (parent_owner.0, parent_owner.1, 0o755)
    {
        return Err("foreign_parent");
    }
    writable(&parent)?;
    parent_identity(&parent)?;
    let output = open_at(&parent, "output", true)?;
    if unsafe { flock(output.as_raw_fd(), 2 | 4) } != 0 {
        return Err("output_initialization_busy");
    }
    let m = output.metadata().map_err(|_| "output_stat")?;
    if !m.is_dir()
        || ![(0, 0, 0o755), (USER, USER, 0o700)].contains(&(m.uid(), m.gid(), m.mode() & 0o7777))
    {
        return Err("foreign_output");
    }
    writable(&output)?;
    empty(&output)?;
    parent_identity(&parent)?;
    verify_named_directory(&parent, &output)?;
    // Only this already-opened, empty target changes. No path repair or recursion.
    if unsafe { fchown(output.as_raw_fd(), USER, USER) } != 0 {
        return Err("output_fchown");
    }
    if unsafe { fchmod(output.as_raw_fd(), 0o700) } != 0 {
        return Err("output_fchmod");
    }
    let after = output.metadata().map_err(|_| "output_verify_stat")?;
    if (after.uid(), after.gid(), after.mode() & 0o7777) != (USER, USER, 0o700)
        || (after.dev(), after.ino()) != (m.dev(), m.ino())
    {
        return Err("output_verification");
    }
    empty(&output)?;
    parent_identity(&parent)?;
    verify_named_directory(&parent, &output)?;
    Ok(output) // Only the now-nonprivileged output FD/lease survives the drop.
}
#[cfg(test)]
fn case_parent_test_initialize(profile: WorkerProfile) -> Result<File> {
    initialize_output(profile)
}
#[cfg(test)]
#[path = "../tests/case_output_parent/cases.rs"]
mod case_output_parent_tests;

fn verify_model_name(parent: &File, output: &File) -> Result<()> {
    let named = open_at(parent, "model", true)?;
    let a = output.metadata().map_err(|_| "model_output_stat")?;
    let b = named.metadata().map_err(|_| "model_output_name_stat")?;
    if (a.dev(), a.ino()) != (b.dev(), b.ino()) {
        return Err("model_output_identity");
    }
    Ok(())
}
fn verify_model_parent(parent: &File) -> Result<()> {
    let named = fixed_root_path("/opt/ml", true)?;
    let a = parent.metadata().map_err(|_| "model_parent_stat")?;
    let b = named.metadata().map_err(|_| "model_parent_name_stat")?;
    if (a.dev(), a.ino()) != (b.dev(), b.ino()) {
        return Err("model_parent_identity");
    }
    Ok(())
}
fn initialize_model_output() -> Result<File> {
    let parent = fixed_root_path("/opt/ml", true)?;
    let p = parent.metadata().map_err(|_| "model_parent_stat")?;
    if (p.uid(), p.gid(), p.mode() & 0o7777) != (0, 0, 0o755) {
        return Err("foreign_model_parent");
    }
    verify_model_parent(&parent)?;
    let output = open_at(&parent, "model", true)?;
    if unsafe { flock(output.as_raw_fd(), 2 | 4) } != 0 {
        return Err("model_output_busy");
    }
    let before = output.metadata().map_err(|_| "model_output_stat")?;
    if !before.is_dir()
        || ![(0, 0, 0o755), (USER, USER, 0o700)].contains(&(
            before.uid(),
            before.gid(),
            before.mode() & 0o7777,
        ))
    {
        return Err("foreign_model_output");
    }
    writable(&output)?;
    empty(&output)?;
    verify_model_parent(&parent)?;
    verify_model_name(&parent, &output)?;
    if unsafe { fchown(output.as_raw_fd(), USER, USER) } != 0
        || unsafe { fchmod(output.as_raw_fd(), 0o700) } != 0
    {
        return Err("model_output_permissions");
    }
    let after = output.metadata().map_err(|_| "model_output_verify")?;
    if (after.uid(), after.gid(), after.mode() & 0o7777) != (USER, USER, 0o700)
        || (before.dev(), before.ino()) != (after.dev(), after.ino())
    {
        return Err("model_output_verification");
    }
    empty(&output)?;
    verify_model_parent(&parent)?;
    verify_model_name(&parent, &output)?;
    Ok(output)
}
fn args() -> Result<(WorkerProfile, Vec<String>)> {
    let values: Vec<String> = std::env::args().skip(1).collect();
    let profile = match values.first().map(String::as_str) {
        Some("acquisition") => WorkerProfile::Acquisition,
        Some("inspection") => WorkerProfile::Inspection,
        Some("preparation") if cfg!(feature = "sagemaker-native-v1") => WorkerProfile::Preparation,
        Some("model-acquisition") if cfg!(feature = "sagemaker-native-v1") => {
            WorkerProfile::ModelAcquisition
        }
        Some("qwen-sft") if cfg!(feature = "sagemaker-native-v1") => WorkerProfile::QwenSft,
        Some("case-freeze") if cfg!(feature = "sagemaker-native-v1") => WorkerProfile::CaseFreeze,
        Some("coding-pair") if cfg!(feature = "sagemaker-native-v1") => WorkerProfile::CodingPair,
        _ => return Err("fixed_profile"),
    };
    let required = match profile {
        WorkerProfile::Acquisition => ACQUISITION,
        WorkerProfile::Inspection => INSPECTION,
        WorkerProfile::Preparation => PREPARATION,
        WorkerProfile::CaseFreeze | WorkerProfile::CodingPair => CASE_FREEZE,
        WorkerProfile::ModelAcquisition => MODEL_ACQUISITION,
        WorkerProfile::QwenSft => QWEN_SFT,
    };
    let forwarded = &values[1..];
    if forwarded.len() % 2 != 0 || forwarded.len() > 26 {
        return Err("worker_arguments");
    }
    let mut seen = BTreeSet::new();
    for pair in forwarded.chunks_exact(2) {
        let flag = pair[0].strip_prefix("--").ok_or("worker_arguments")?;
        if (!required.contains(&flag)
            && (profile != WorkerProfile::Inspection || flag != "target-column"))
            || !seen.insert(flag)
            || pair[1].is_empty()
            || pair[1].len() > 1024
            || pair[1].chars().any(|c| c.is_control())
        {
            return Err("worker_arguments");
        }
    }
    if required.iter().any(|key| !seen.contains(key)) {
        return Err("worker_arguments");
    }
    if profile == WorkerProfile::QwenSft {
        for pair in forwarded.chunks_exact(2) {
            let valid = match pair[0].as_str() {
                "--operation" => ["train", "reload", "generate"].contains(&pair[1].as_str()),
                "--recipe-id" => {
                    ["fit_v1", "full_v1", "humaneval_greedy_v1"].contains(&pair[1].as_str())
                }
                "--recipe-version" => pair[1] == "1",
                "--manifest-sha256" => {
                    pair[1].len() == 64
                        && pair[1]
                            .bytes()
                            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                }
                "--manifest-byte-count" => pair[1].parse::<u32>().is_ok_and(|value| {
                    (1..=262144).contains(&value) && value.to_string() == pair[1]
                }),
                _ => true,
            };
            if !valid {
                return Err("worker_arguments");
            }
        }
    }
    if profile == WorkerProfile::QwenSft {
        let selected = |flag: &str| {
            forwarded
                .chunks_exact(2)
                .find(|pair| pair[0] == flag)
                .map(|pair| pair[1].as_str())
        };
        let valid_pair = match (selected("--operation"), selected("--recipe-id")) {
            (Some("generate"), Some("humaneval_greedy_v1")) => true,
            (Some("train" | "reload"), Some("fit_v1" | "full_v1")) => true,
            _ => false,
        };
        if !valid_pair {
            return Err("worker_arguments");
        }
    }
    if profile == WorkerProfile::CodingPair {
        for pair in forwarded.chunks_exact(2) {
            let valid = match pair[0].as_str() {
                "--binding-sha256" => {
                    pair[1].len() == 64
                        && pair[1]
                            .bytes()
                            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                }
                "--binding-byte-count" => pair[1].parse::<u32>().is_ok_and(|value| {
                    (1..=262144).contains(&value) && value.to_string() == pair[1]
                }),
                "--source-commit" => {
                    pair[1].len() == 40
                        && pair[1]
                            .bytes()
                            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                }
                _ => false,
            };
            if !valid {
                return Err("worker_arguments");
            }
        }
    }
    Ok((profile, forwarded.to_vec()))
}
// Deferral is limited to this fixed native profile. Child exec remains gated by
// installed kernel TCP restrictions plus actual endpoint-denial observations.
fn defer_model_relative_credential_hint(
    profile: WorkerProfile,
    key: &str,
    value: &str,
) -> Result<bool> {
    if !cfg!(feature = "sagemaker-native-v1")
        || profile != WorkerProfile::ModelAcquisition
        || key != "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
    {
        return Ok(false);
    }
    if value.is_empty()
        || value.len() > 4096
        || !value.starts_with('/')
        || value.starts_with("//")
        || value.contains("..")
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/_.-".contains(&c))
    {
        return Err("model_relative_credential_hint");
    }
    Ok(true)
}
fn pair_role_relative_uri(key: &str, value: &str) -> Result<bool> {
    if key != "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" {
        return Ok(false);
    }
    let prefix = "/v2/credentials/";
    if value.len() <= prefix.len()
        || value.len() > 4096
        || !value.starts_with(prefix)
        || !value[prefix.len()..]
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
    {
        return Err("pair_relative_credential_hint");
    }
    Ok(true)
}
fn child_environment(profile: WorkerProfile) -> Result<BTreeMap<OsString, OsString>> {
    let acquisition = profile == WorkerProfile::Acquisition;
    let remote_fetch = acquisition || profile == WorkerProfile::ModelAcquisition;
    let mut env: BTreeMap<OsString, OsString> = BTreeMap::new();
    let mut source = None;
    let mut count = 0;
    for (key, value) in std::env::vars_os() {
        count += 1;
        if count > 256 {
            return Err("environment_bound");
        }
        let key = key.to_str().ok_or("environment_encoding")?;
        let value = value.to_str().ok_or("environment_encoding")?;
        if key.len() > 256 || value.len() > 8192 {
            return Err("environment_bound");
        }
        let immutable_python_metadata = match key {
            "PYTHON_VERSION" => value == "3.11.15",
            "PYTHON_SHA256" => {
                // Public Python release source checksum, not a credential.
                // pragma: allowlist nextline secret
                value == "272179ddd9a2e41a0fc8e42e33dfbdca0b3711aa5abf372d3f2d51543d09b625"
            }
            _ => false,
        };
        let fixed_python_flags =
            ["PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"].contains(&key) && value == "1";
        if key.starts_with("LD_")
            || (key.starts_with("PYTHON") && !immutable_python_metadata && !fixed_python_flags)
        {
            return Err("loader_environment");
        }
        if !acquisition
            && [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
            ]
            .contains(&key)
        {
            return Err("inspection_proxy_hint");
        }
        let credentials = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_SECURITY_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
            "AWS_ROLE_SESSION_NAME",
            "AWS_CREDENTIAL_EXPIRATION",
            "AWS_EC2_METADATA_DISABLED",
            "AWS_EC2_METADATA_SERVICE_ENDPOINT",
            "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
            "AWS_EC2_METADATA_V1_DISABLED",
            "AWS_METADATA_SERVICE_TIMEOUT",
            "AWS_METADATA_SERVICE_NUM_ATTEMPTS",
            "AWS_CREDENTIAL_FILE",
            "BOTO_CONFIG",
        ]
        .contains(&key)
            || key.starts_with("AWS_CONTAINER_CREDENTIALS_")
            || key.starts_with("AWS_CONTAINER_AUTHORIZATION_");
        if !acquisition && credentials {
            if cfg!(feature = "sagemaker-native-v1")
                && profile == WorkerProfile::CodingPair
                && pair_role_relative_uri(key, value)?
            {
                // Only the native role metadata selector survives; no credentials or arbitrary endpoint.
                env.insert(key.into(), value.into());
            } else if !defer_model_relative_credential_hint(profile, key, value)? {
                return Err("inspection_credential_hint");
            }
        }
        if profile == WorkerProfile::CodingPair && key == "AMLAS_CODING_PAIR_SOURCE_COMMIT" {
            if value.len() != 40
                || !value
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                return Err("pair_source_commit");
            }
            env.insert(key.into(), value.into());
        }
        if profile == WorkerProfile::CodingPair && key == "AMLAS_CODING_PAIR_LAUNCHER_SHA256" {
            if value.len() != 64
                || !value
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                return Err("pair_launcher_sha256");
            }
            env.insert(key.into(), value.into());
        }
        if key == "AMLAS_SOURCE_COMMIT" {
            if value.len() != 40
                || !value
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                return Err("source_commit");
            }
            source = Some(value.to_owned());
        }
        if remote_fetch
            && ["AMLAS_AGENT_REMOTE_FETCH", "AMLAS_AGENT_HTTP_ALLOW_HOSTS"].contains(&key)
        {
            env.insert(key.into(), value.into());
        }
    }
    let source = source.ok_or("source_commit")?;
    if profile == WorkerProfile::CodingPair {
        if env
            .get(&OsString::from("AMLAS_CODING_PAIR_SOURCE_COMMIT"))
            .and_then(|v| v.to_str())
            != Some(source.as_str())
        {
            return Err("pair_source_commit");
        }
        if !env.contains_key(&OsString::from("AMLAS_CODING_PAIR_LAUNCHER_SHA256")) {
            return Err("pair_launcher_sha256");
        }
    }
    env.insert("AMLAS_SOURCE_COMMIT".into(), source.into());
    for (k, v) in [
        ("PATH", "/usr/local/bin:/usr/bin:/bin"),
        ("HOME", "/home/appuser"),
        (
            "TMPDIR",
            if profile == WorkerProfile::QwenSft {
                "/home/appuser/.cache/amlas-qwen-sft"
            } else if matches!(
                profile,
                WorkerProfile::ModelAcquisition | WorkerProfile::CodingPair
            ) {
                "/opt/ml/processing/output"
            } else {
                PARENT
            },
        ),
        (
            "AMLAS_WORKSPACE_DIR",
            if profile == WorkerProfile::QwenSft {
                "/opt/ml/model"
            } else if matches!(
                profile,
                WorkerProfile::ModelAcquisition | WorkerProfile::CodingPair
            ) {
                "/opt/ml/processing/output"
            } else {
                PARENT
            },
        ),
        ("LANG", "C.UTF-8"),
        ("PYTHONUNBUFFERED", "1"),
        ("PYTHONDONTWRITEBYTECODE", "1"),
    ] {
        env.insert(k.into(), v.into());
    }
    if !acquisition {
        env.insert("OPENBLAS_NUM_THREADS".into(), "1".into());
        env.insert("OMP_NUM_THREADS".into(), "1".into());
    }
    Ok(env)
}
fn bounded_os_text(path: &str, limit: u64) -> Result<String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|_| "os_status_open")?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "os_status_read")?;
    if bytes.len() as u64 > limit {
        return Err("os_status_bound");
    }
    String::from_utf8(bytes).map_err(|_| "os_status_encoding")
}
#[cfg(not(feature = "sagemaker-native-v1"))]
fn verify_drop() -> Result<()> {
    ids(USER)?;
    if unsafe { getgroups(0, std::ptr::null_mut()) } != 0 {
        return Err("supplementary_groups");
    }
    let status = bounded_os_text("/proc/self/status", 32768)?;
    for key in [
        "CapEff",
        "CapPrm",
        "CapInh",
        "CapAmb",
        "CapBnd",
        "NoNewPrivs",
    ] {
        let mut values = status
            .lines()
            .filter_map(|line| line.strip_prefix(&format!("{key}:")));
        let value = values.next().ok_or("privilege_status")?.trim();
        if values.next().is_some() {
            return Err("privilege_status");
        }
        if key == "NoNewPrivs" {
            if value != "1" {
                return Err("no_new_privileges");
            }
        } else if u64::from_str_radix(value, 16).map_err(|_| "capability_status")? != 0 {
            return Err("remaining_capability");
        }
    }
    Ok(())
}
#[cfg(not(feature = "sagemaker-native-v1"))]
fn drop_privileges() -> Result<()> {
    if unsafe { prctl(38, 1usize, 0usize, 0usize, 0usize) } != 0
        || unsafe { prctl(8, 0usize, 0usize, 0usize, 0usize) } != 0
        || unsafe { prctl(47, 4usize, 0usize, 0usize, 0usize) } != 0
    {
        return Err("privilege_setup");
    }
    let last: i32 = bounded_os_text("/proc/sys/kernel/cap_last_cap", 16)?
        .trim()
        .parse()
        .map_err(|_| "capability_range")?;
    if !(0..=63).contains(&last) {
        return Err("capability_range");
    }
    drop_bounding_capabilities(
        last as u32,
        |cap| unsafe { prctl(23, cap as usize, 0usize, 0usize, 0usize) },
        |cap| unsafe { prctl(24, cap as usize, 0usize, 0usize, 0usize) },
    )?;
    if unsafe { setgroups(0, std::ptr::null()) } != 0
        || unsafe { setresgid(USER, USER, USER) } != 0
        || unsafe { setresuid(USER, USER, USER) } != 0
    {
        return Err("identity_drop");
    }
    let header = CapHeader {
        version: 0x20080522,
        pid: 0,
    };
    let data = [CapData {
        effective: 0,
        permitted: 0,
        inheritable: 0,
    }; 2];
    if unsafe { capset(&header, data.as_ptr()) } != 0 {
        return Err("capability_drop");
    }
    verify_drop()
}
#[cfg(feature = "sagemaker-native-v1")]
fn native_state() -> Result<NativeState> {
    let status = bounded_os_text("/proc/self/status", 32768)?;
    let field = |key: &str| -> Result<&str> {
        let prefix = format!("{key}:");
        let mut values = status.lines().filter_map(|line| line.strip_prefix(&prefix));
        let value = values.next().ok_or("native_status_missing")?.trim();
        if values.next().is_some() {
            return Err("native_status_duplicate");
        }
        Ok(value)
    };
    let parse_ids = |key: &str| -> Result<[u32; 4]> {
        let values: Vec<u32> = field(key)?
            .split_whitespace()
            .map(|v| v.parse().map_err(|_| "native_status_ids"))
            .collect::<Result<_>>()?;
        values.try_into().map_err(|_| "native_status_ids")
    };
    let uid = parse_ids("Uid")?;
    let gid = parse_ids("Gid")?;
    let (mut a, mut b, mut c) = (0, 0, 0);
    if unsafe { getresuid(&mut a, &mut b, &mut c) } != 0 || uid[..3] != [a, b, c] {
        return Err("native_uid_status");
    }
    if unsafe { getresgid(&mut a, &mut b, &mut c) } != 0 || gid[..3] != [a, b, c] {
        return Err("native_gid_status");
    }
    let count = unsafe { getgroups(0, std::ptr::null_mut()) };
    if !(0..=128).contains(&count) {
        return Err("native_groups_bound");
    }
    let mut groups = vec![0; count as usize];
    if unsafe { getgroups(count, groups.as_mut_ptr()) } != count {
        return Err("native_groups_status");
    }
    let mut caps = [0; 5];
    for (i, key) in ["CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd"]
        .iter()
        .enumerate()
    {
        let text = field(key)?;
        if text.is_empty() || text.len() > 16 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("native_capability_status");
        }
        caps[i] = u64::from_str_radix(text, 16).map_err(|_| "native_capability_status")?;
    }
    Ok(NativeState {
        uid,
        gid,
        groups,
        caps,
        threads: field("Threads")?
            .parse()
            .map_err(|_| "native_thread_status")?,
        no_new_privs: field("NoNewPrivs")?
            .parse()
            .map_err(|_| "native_nnp_status")?,
        seccomp: field("Seccomp")?
            .parse()
            .map_err(|_| "native_seccomp_status")?,
        keep_caps: unsafe { prctl(7, 0usize, 0usize, 0usize, 0usize) },
        securebits: unsafe { prctl(27, 0usize, 0usize, 0usize, 0usize) },
    })
}
#[cfg(feature = "sagemaker-native-v1")]
fn validate_native_final_for_profile(
    state: &NativeState,
    initial_bnd: u64,
    profile: WorkerProfile,
) -> Result<()> {
    if profile != WorkerProfile::CodingPair {
        return validate_native_final(state, initial_bnd);
    }
    if state.groups != [USER] {
        return Err("pair_native_final_groups");
    }
    // Reuse every common final invariant; only this compiled broker has the existing oracle group.
    let mut common = state.clone();
    common.groups.clear();
    validate_native_final(&common, initial_bnd)
}
#[cfg(feature = "sagemaker-native-v1")]
fn drop_native_privileges(initial_bnd: u64) -> Result<()> {
    drop_native_privileges_for_profile(initial_bnd, WorkerProfile::CaseFreeze)
}
#[cfg(feature = "sagemaker-native-v1")]
fn native_startup_proof(initial_bnd: u64) -> Result<()> {
    native_startup_proof_for_profile(initial_bnd, WorkerProfile::CaseFreeze)
}
#[cfg(feature = "sagemaker-native-v1")]
fn drop_native_privileges_for_profile(initial_bnd: u64, profile: WorkerProfile) -> Result<()> {
    // Every setter failure refuses. No bounding DROP operation exists in this profile.
    if unsafe { prctl(38, 1usize, 0usize, 0usize, 0usize) } != 0
        || unsafe { prctl(39, 0usize, 0usize, 0usize, 0usize) } != 1
    {
        return Err("native_nnp_set");
    }
    if unsafe { prctl(8, 0usize, 0usize, 0usize, 0usize) } != 0
        || unsafe { prctl(7, 0usize, 0usize, 0usize, 0usize) } != 0
        || unsafe { prctl(47, 4usize, 0usize, 0usize, 0usize) } != 0
    {
        return Err("native_keep_ambient_set");
    }
    let groups = if profile == WorkerProfile::CodingPair {
        &[USER][..]
    } else {
        &[][..]
    };
    if unsafe { setgroups(groups.len(), groups.as_ptr()) } != 0
        || unsafe { setresgid(USER, USER, USER) } != 0
        || unsafe { setresuid(USER, USER, USER) } != 0
    {
        return Err("native_identity_drop");
    }
    let header = CapHeader {
        version: 0x20080522,
        pid: 0,
    };
    let data = [CapData {
        effective: 0,
        permitted: 0,
        inheritable: 0,
    }; 2];
    if unsafe { capset(&header, data.as_ptr()) } != 0 {
        return Err("native_capability_drop");
    }
    validate_native_final_for_profile(&native_state()?, initial_bnd, profile)
}
#[cfg(feature = "sagemaker-native-v1")]
fn native_startup_proof_for_profile(initial_bnd: u64, profile: WorkerProfile) -> Result<()> {
    let state = native_state()?;
    validate_native_final_for_profile(&state, initial_bnd, profile)?;
    // Fixed bounded kernel metadata only; guest claims never supply authority.
    let groups = if profile == WorkerProfile::CodingPair {
        "[10001]"
    } else {
        "[]"
    };
    println!("{{\"privilege_profile\":\"sagemaker_native_ceiling_v1\",\"initial_bnd\":\"{initial_bnd:016x}\",\"final_bnd\":\"{:016x}\",\"privilege_proof\":{{\"res_uid\":[10001,10001,10001],\"res_gid\":[10001,10001,10001],\"groups\":{groups},\"cap_eff\":0,\"cap_prm\":0,\"cap_inh\":0,\"cap_amb\":0,\"cap_bnd\":{},\"no_new_privs\":1,\"keep_caps\":0,\"seccomp\":2,\"fs_uid\":{},\"fs_gid\":{},\"securebits\":{}}}}}",state.caps[4],state.caps[4],state.uid[3],state.gid[3],state.securebits);
    Ok(())
}

#[cfg(feature = "sagemaker-native-v1")]
fn model_file_boundary_startup_proof(proof: &FileBoundaryProof) -> Result<()> {
    if proof.rule_identities.len() != 28 {
        return Err("model_file_proof_rule_count");
    }
    let rules = proof
        .rule_identities
        .iter()
        .map(|rule| format!(
            "{{\"rule_index\":{},\"device\":{},\"inode\":{},\"uid\":{},\"gid\":{},\"mode\":{},\"allowed_access_fs\":{}}}",
            rule.rule_index, rule.device, rule.inode, rule.uid, rule.gid, rule.mode, rule.allowed_access_fs
        ))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"model_file_boundary\":{{\"profile\":\"{}\",\"abi\":{},\"handled_access_fs\":{},\"original_output_device\":{},\"original_output_inode\":{},\"process_id\":{},\"process_dumpability\":{},\"rule_identities\":[{}]}},\"guest_claim_attested\":false}}",
        proof.profile, proof.abi, proof.handled_access_fs, proof.original_output_device,
        proof.original_output_inode, proof.process_id, proof.process_dumpability, rules
    );
    Ok(())
}

fn close_inherited_fds() -> Result<()> {
    let mut fds = Vec::new();
    for entry in fs::read_dir("/proc/self/fd").map_err(|_| "fd_inventory")? {
        let name = entry.map_err(|_| "fd_inventory")?.file_name();
        let fd: i32 = name
            .to_str()
            .ok_or("fd_inventory")?
            .parse()
            .map_err(|_| "fd_inventory")?;
        if fd > 2 && fd != 3 {
            fds.push(fd);
        }
    }
    // Inventory iterator has closed; EBADF for its own old FD is harmless.
    for fd in fds {
        if unsafe { close(fd) } != 0 && std::io::Error::last_os_error().raw_os_error() != Some(9) {
            return Err("inherited_fd_close");
        }
    }
    Ok(())
}
fn handoff_output(output: File) -> Result<()> {
    let info = output.metadata().map_err(|_| "output_handoff_stat")?;
    if !info.is_dir() || (info.uid(), info.gid(), info.mode() & 0o7777) != (USER, USER, 0o700) {
        return Err("output_handoff_identity");
    }
    let original = output.into_raw_fd();
    if original != 3 {
        if unsafe { dup2(original, 3) } != 3 {
            return Err("output_handoff_dup");
        }
        if unsafe { close(original) } != 0 {
            return Err("output_handoff_close");
        }
    }
    // FD3 is a duplicate of the SAME open-file-description: flock stays live.
    if unsafe { fcntl(3, 2, 0) } != 0 {
        return Err("output_handoff_exec");
    }
    Ok(())
}

#[cfg(feature = "sagemaker-native-v1")]
fn verify_model_credential_endpoint_denial() -> Result<()> {
    for ip in [[169, 254, 169, 254], [169, 254, 170, 2]] {
        let target = std::net::SocketAddr::from((ip, 80));
        match std::net::TcpStream::connect_timeout(&target, std::time::Duration::from_millis(500)) {
            Err(error) if matches!(error.raw_os_error(), Some(1 | 13)) => {}
            _ => return Err("model_credential_endpoint_denial"),
        }
    }
    Ok(())
}
#[cfg(feature = "sagemaker-native-v1")]
fn model_tcp_boundary_startup_proof(proof: &TcpBoundaryProof) -> Result<()> {
    if proof.profile != "raw_model_tcp_ports_v1"
        || proof.abi < 4
        || proof.handled_access_net != 3
        || proof.allowed_connect_tcp_ports != [53, 443]
        || !proof.allowed_bind_tcp_ports.is_empty()
    {
        return Err("model_tcp_proof_shape");
    }
    println!("{{\"model_tcp_boundary\":{{\"profile\":\"{}\",\"abi\":{},\"handled_access_net\":3,\"allowed_connect_tcp_ports\":[53,443],\"allowed_bind_tcp_ports\":[]}},\"fixed_http_metadata_endpoints_denied\":true,\"ip_filtering\":false,\"udp_filtering\":false,\"guest_claim_attested\":false}}", proof.profile, proof.abi);
    Ok(())
}
#[cfg(feature = "sagemaker-native-v1")]
fn qwen_tcp_boundary_startup_proof(proof: &QwenTcpBoundaryProof) -> Result<()> {
    if proof.profile != "qwen_sft_tcp_deny_v1"
        || proof.abi < 4
        || proof.handled_access_net != 3
        || !proof.allowed_connect_tcp_ports.is_empty()
        || !proof.allowed_bind_tcp_ports.is_empty()
    {
        return Err("qwen_tcp_proof_shape");
    }
    println!("{{\"qwen_tcp_boundary\":{{\"profile\":\"qwen_sft_tcp_deny_v1\",\"abi\":{},\"handled_access_net\":3,\"allowed_connect_tcp_ports\":[],\"allowed_bind_tcp_ports\":[]}},\"fixed_tcp_endpoint_permission_denials_observed\":true,\"udp_filtering\":false,\"ip_filtering\":false,\"existing_sockets_revoked\":false,\"guest_claim_attested\":false}}", proof.abi);
    Ok(())
}

#[cfg(feature = "sagemaker-native-v1")]
fn qwen_tcp_startup_gate(
    install: impl FnOnce() -> Result<QwenTcpBoundaryProof>,
    checks: impl FnOnce() -> Result<()>,
    proof: impl FnOnce(&QwenTcpBoundaryProof) -> Result<()>,
    launch: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let installed = install()?;
    checks()?;
    proof(&installed)?;
    launch()
}
#[cfg(all(test, feature = "sagemaker-native-v1"))]
#[path = "../tests/qwen_tcp_boundary/startup_gate.rs"]
mod qwen_tcp_gate_tests;

#[cfg(feature = "sagemaker-native-v1")]
fn initialize_qwen_scratch() -> Result<()> {
    ids(USER)?;
    let root = File::open("/").map_err(|_| "qwen_scratch_root")?;
    qwen_scratch_from_root(&root)
}
#[cfg(feature = "sagemaker-native-v1")]
fn qwen_scratch_from_root(root: &File) -> Result<()> {
    ids(USER)?;
    let mut nodes = Vec::new();
    let mut accepted = Vec::new();
    let mut parent = root;
    for (name, owner, mode, create) in [
        ("home", 0, 0o755, false),
        ("appuser", USER, 0o700, false),
        (".cache", USER, 0o700, true),
        ("amlas-qwen-sft", USER, 0o700, true),
    ] {
        let fixed = CString::new(name).map_err(|_| "qwen_scratch_name")?;
        let mut fd = unsafe { openat(parent.as_raw_fd(), fixed.as_ptr(), DIRECTORY_FLAGS) };
        if fd < 0 && create && std::io::Error::last_os_error().raw_os_error() == Some(2) {
            if unsafe { mkdirat(parent.as_raw_fd(), fixed.as_ptr(), 0o700) } != 0 {
                return Err("qwen_scratch_create");
            }
            fd = unsafe { openat(parent.as_raw_fd(), fixed.as_ptr(), DIRECTORY_FLAGS) };
        }
        if fd < 0 {
            return Err("qwen_scratch_open");
        }
        let child = unsafe { File::from_raw_fd(fd) };
        let metadata = child.metadata().map_err(|_| "qwen_scratch_stat")?;
        if !metadata.is_dir()
            || metadata.uid() != owner
            || metadata.gid() != owner
            || metadata.mode() & 0o7777 != mode
        {
            return Err("qwen_scratch_metadata");
        }
        accepted.push((
            metadata.dev(),
            metadata.ino(),
            metadata.uid(),
            metadata.gid(),
            metadata.mode(),
        ));
        nodes.push(child);
        parent = nodes.last().ok_or("qwen_scratch_open")?;
    }
    let leaf = nodes.last().ok_or("qwen_scratch_open")?;
    let alias = format!("/proc/self/fd/{}", leaf.as_raw_fd());
    if fs::read_dir(alias)
        .map_err(|_| "qwen_scratch_inventory")?
        .next()
        .is_some()
    {
        return Err("qwen_scratch_nonempty");
    }
    #[cfg(test)]
    qwen_scratch_test_mutation(root, &nodes)?;
    let mut parent = root;
    for (index, name) in ["home", "appuser", ".cache", "amlas-qwen-sft"]
        .into_iter()
        .enumerate()
    {
        let named = open_at(parent, name, true).map_err(|_| "qwen_scratch_identity")?;
        let expected = nodes[index].metadata().map_err(|_| "qwen_scratch_stat")?;
        let actual = named.metadata().map_err(|_| "qwen_scratch_stat")?;
        let retained = (
            expected.dev(),
            expected.ino(),
            expected.uid(),
            expected.gid(),
            expected.mode(),
        );
        let current = (
            actual.dev(),
            actual.ino(),
            actual.uid(),
            actual.gid(),
            actual.mode(),
        );
        if retained != accepted[index] || current != accepted[index] {
            return Err("qwen_scratch_identity");
        }
        parent = &nodes[index];
    }
    Ok(())
}
#[cfg(all(test, feature = "sagemaker-native-v1"))]
fn qwen_scratch_test_mutation(root: &File, nodes: &[File]) -> Result<()> {
    let selected = root.metadata().map_err(|_| "test_mutation")?;
    let _ = selected;
    if std::env::var("AMLAS_QWEN_SCRATCH_MUTATION").ok().as_deref() == Some("mode") {
        if unsafe { fchmod(nodes[2].as_raw_fd(), 0o755) } != 0 {
            return Err("test_mutation");
        }
    } else if std::env::var("AMLAS_QWEN_SCRATCH_MUTATION").ok().as_deref() == Some("name") {
        let parent = format!("/proc/self/fd/{}", nodes[1].as_raw_fd());
        fs::rename(format!("{parent}/.cache"), format!("{parent}/.cache-old"))
            .map_err(|_| "test_mutation")?;
        fs::create_dir(format!("{parent}/.cache")).map_err(|_| "test_mutation")?;
    }
    Ok(())
}
#[cfg(all(test, feature = "sagemaker-native-v1"))]
#[path = "../tests/qwen_scratch/cases.rs"]
mod qwen_scratch_tests;

fn run() -> Result<()> {
    ids(0)?;
    let (profile, forwarded) = args()?;
    let env = child_environment(profile)?;
    let source_pair = forwarded
        .chunks_exact(2)
        .find(|pair| pair[0] == "--source-commit")
        .ok_or("source_binding")?;
    if env
        .get(&OsString::from("AMLAS_SOURCE_COMMIT"))
        .and_then(|v| v.to_str())
        != Some(source_pair[1].as_str())
    {
        return Err("source_binding");
    }
    let script = match profile {
        WorkerProfile::Acquisition => "/app/acquisition_worker.py",
        WorkerProfile::Inspection => "/app/inspect_dataset.py",
        WorkerProfile::Preparation => "/app/coding_preparation_worker.py",
        WorkerProfile::CaseFreeze => "/app/coding_case_worker.py",
        WorkerProfile::CodingPair => "/app/coding_pair_worker.py",
        WorkerProfile::ModelAcquisition => "/app/acquire_model.py",
        WorkerProfile::QwenSft => "/app/qwen_sft_train.py",
    };
    fixed_root_path("/usr/local/bin/amlas-processing-init", false)?;
    fixed_root_path("/usr/local/bin/python3.11", false)?;
    fixed_root_path("/usr/local/lib/python3.11", true)?;
    if profile == WorkerProfile::CodingPair {
        fixed_pair_interpreter()?;
        fixed_root_path("/opt/venv", true)?;
        fixed_root_path("/opt/venv/lib/python3.11/site-packages", true)?;
        fixed_root_path("/app/src", true)?;
    }
    fixed_root_path(script, false)?;
    match profile {
        WorkerProfile::Acquisition | WorkerProfile::ModelAcquisition => {
            fixed_root_path("/app/src", true)?;
        }
        WorkerProfile::Inspection => {
            fixed_root_path("/app/inspection_schema.py", false)?;
        }
        WorkerProfile::CodingPair => {}
        WorkerProfile::Preparation => {
            fixed_root_path("/app/coding_preparation_core.py", false)?;
            fixed_root_path("/app/preparation_profiles", true)?;
            fixed_root_path("/app/preparation_profiles/__init__.py", false)?;
            fixed_root_path("/app/preparation_profiles/coding_preparation.py", false)?;
            fixed_root_path("/app/preparation_profiles/open_swe_canonical.py", false)?;
            fixed_root_path("/app/preparation_profiles/open_swe_derived.py", false)?;
        }
        WorkerProfile::QwenSft => {
            fixed_root_path("/app/qwen_sft_core.py", false)?;
        }
        WorkerProfile::CaseFreeze => {
            for path in [
                "/app/case_worker_pkg",
                "/app/case_worker_pkg/backends",
                "/app/case_worker_pkg/data_eng",
                "/app/case_worker_pkg/sagemaker_entrypoints",
            ] {
                fixed_root_path(path, true)?;
            }
            for path in [
                "/app/case_worker_pkg/__init__.py",
                "/app/case_worker_pkg/backends/__init__.py",
                "/app/case_worker_pkg/data_eng/__init__.py",
                "/app/case_worker_pkg/sagemaker_entrypoints/__init__.py",
                "/app/case_worker_pkg/backends/types.py",
                "/app/case_worker_pkg/data_eng/coding_case_profile.py",
                "/app/case_worker_pkg/data_eng/coding_case_codec.py",
                "/app/case_worker_pkg/data_eng/coding_case_manifest.py",
                "/app/case_worker_pkg/data_eng/coding_case_managed.py",
                "/app/case_worker_pkg/data_eng/inspection_schema.py",
                "/app/case_worker_pkg/sagemaker_entrypoints/coding_case_freeze.py",
            ] {
                fixed_root_path(path, false)?;
            }
        }
    }
    #[cfg(feature = "sagemaker-native-v1")]
    let initial_bnd = {
        let last: u32 = bounded_os_text("/proc/sys/kernel/cap_last_cap", 16)?
            .trim()
            .parse()
            .map_err(|_| "capability_range")?;
        validate_native_initial(&native_state()?, last, |cap| unsafe {
            prctl(23, cap as usize, 0usize, 0usize, 0usize)
        })?
    };
    #[cfg(feature = "sagemaker-native-v1")]
    fixed_root_path("/app/processing_identity.py", false)?;
    #[cfg(feature = "sagemaker-native-v1")]
    let provider_inventory = if matches!(
        profile,
        WorkerProfile::ModelAcquisition | WorkerProfile::QwenSft
    ) {
        Some(capture_provider_inventory()?)
    } else {
        None
    };
    let output = if profile == WorkerProfile::QwenSft {
        initialize_model_output()?
    } else {
        initialize_output(profile)?
    };
    #[cfg(not(feature = "sagemaker-native-v1"))]
    drop_privileges()?;
    #[cfg(feature = "sagemaker-native-v1")]
    drop_native_privileges_for_profile(initial_bnd, profile)?;
    #[cfg(feature = "sagemaker-native-v1")]
    if profile == WorkerProfile::ModelAcquisition {
        let inventory = provider_inventory.ok_or("model_provider_inventory_missing")?;
        let descriptors = handoff_model_fds(output, inventory)?;
        native_startup_proof(initial_bnd)?;
        let proof = apply_model_file_boundary(descriptors.output())?;
        model_file_boundary_startup_proof(&proof)?;
        let tcp_proof = apply_model_tcp_boundary(&native_state()?)?;
        verify_model_credential_endpoint_denial()?;
        model_tcp_boundary_startup_proof(&tcp_proof)?;
        let _error = Command::new("/usr/local/bin/python3.11")
            .args(["-I", script])
            .args(forwarded)
            .env_clear()
            .envs(env)
            .current_dir("/app")
            .exec();
        return Err("fixed_model_worker_exec");
    }
    #[cfg(feature = "sagemaker-native-v1")]
    if profile == WorkerProfile::QwenSft {
        let inventory = provider_inventory.ok_or("qwen_provider_inventory_missing")?;
        let _descriptors = handoff_model_fds(output, inventory)?;
        native_startup_proof(initial_bnd)?;
        initialize_qwen_scratch()?;
        return qwen_tcp_startup_gate(
            || apply_qwen_tcp_boundary(&native_state()?),
            verify_qwen_tcp_endpoint_denial,
            qwen_tcp_boundary_startup_proof,
            || {
                let _error = Command::new("/usr/local/bin/python3.11")
                    .args(["-I", script])
                    .args(forwarded)
                    .env_clear()
                    .envs(env)
                    .current_dir("/app")
                    .exec();
                Err("fixed_qwen_worker_exec")
            },
        );
    }
    handoff_output(output)?;
    close_inherited_fds()?;
    let mut extra = 0;
    for entry in fs::read_dir("/proc/self/fd").map_err(|_| "fd_final_inventory")? {
        let name = entry.map_err(|_| "fd_final_inventory")?.file_name();
        let fd: i32 = name
            .to_str()
            .ok_or("fd_final_inventory")?
            .parse()
            .map_err(|_| "fd_final_inventory")?;
        if fd > 3 {
            extra += 1;
        }
    }
    if extra > 1 {
        return Err("remaining_inherited_fd");
    }
    #[cfg(feature = "sagemaker-native-v1")]
    native_startup_proof_for_profile(initial_bnd, profile)?;
    let interpreter = if profile == WorkerProfile::CodingPair {
        "/opt/venv/bin/python"
    } else {
        "/usr/local/bin/python3.11"
    };
    let _error = Command::new(interpreter)
        .args(["-I", script])
        .args(forwarded)
        .env_clear()
        .envs(env)
        .current_dir("/app")
        .exec();
    Err("fixed_worker_exec")
}
fn main() {
    if let Err(stage) = run() {
        eprintln!("processing initializer refused stage={stage}");
        std::process::exit(78);
    }
}

#[cfg(test)]
mod deferred_model_hint_tests {
    use super::*;

    #[test]
    fn only_fixed_native_model_relative_path_is_deferred() {
        assert_eq!(
            defer_model_relative_credential_hint(
                WorkerProfile::Inspection,
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/test"
            ),
            Ok(false)
        );
        assert_eq!(
            defer_model_relative_credential_hint(
                WorkerProfile::QwenSft,
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/test"
            ),
            Ok(false)
        );
        assert_eq!(
            defer_model_relative_credential_hint(
                WorkerProfile::ModelAcquisition,
                "AWS_ACCESS_KEY_ID",
                "inert-not-a-key"
            ),
            Ok(false)
        );
        #[cfg(feature = "sagemaker-native-v1")]
        assert_eq!(
            defer_model_relative_credential_hint(
                WorkerProfile::ModelAcquisition,
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/test"
            ),
            Ok(true)
        );
        #[cfg(not(feature = "sagemaker-native-v1"))]
        assert_eq!(
            defer_model_relative_credential_hint(
                WorkerProfile::ModelAcquisition,
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                "/v2/credentials/test"
            ),
            Ok(false)
        );
    }

    #[cfg(feature = "sagemaker-native-v1")]
    #[test]
    fn invalid_endpoint_shapes_are_refused_without_echo() {
        for value in [
            "",
            "http://example.invalid/",
            "//example.invalid/x",
            "/x?secret=value",
            "/x#fragment",
            "/x\nprivate",
            "/../x",
            "/x\\y",
        ] {
            assert_eq!(
                defer_model_relative_credential_hint(
                    WorkerProfile::ModelAcquisition,
                    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                    value
                ),
                Err("model_relative_credential_hint")
            );
        }
        assert_eq!(
            defer_model_relative_credential_hint(
                WorkerProfile::ModelAcquisition,
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                &"/".repeat(4097)
            ),
            Err("model_relative_credential_hint")
        );
    }
}

#[cfg(all(test, feature = "sagemaker-native-v1"))]
mod pair_environment_tests {
    use super::*;
    #[test]
    fn pair_retains_only_exact_source_launcher_and_native_relative_role_selector() {
        let executable = std::env::current_exe().unwrap();
        if std::env::var("AMLAS_PAIR_ENV_CHILD").ok().as_deref() != Some("1") {
            let result = Command::new(executable).args(["--exact", "pair_environment_tests::pair_retains_only_exact_source_launcher_and_native_relative_role_selector", "--test-threads=1"])
                .env_clear().env("AMLAS_PAIR_ENV_CHILD", "1").env("AMLAS_SOURCE_COMMIT", "c".repeat(40))
                .env("AMLAS_CODING_PAIR_SOURCE_COMMIT", "c".repeat(40)).env("AMLAS_CODING_PAIR_LAUNCHER_SHA256", "d".repeat(64))
                .env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/native-role").env("UNRELATED", "discarded").status().unwrap();
            if !result.success() {
                panic!("fixed pair environment child failed");
            }
            return;
        }
        let env = child_environment(WorkerProfile::CodingPair).unwrap();
        assert_eq!(
            env.get(&OsString::from("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"))
                .unwrap(),
            "/v2/credentials/native-role"
        );
        assert_eq!(
            env.get(&OsString::from("AMLAS_CODING_PAIR_SOURCE_COMMIT"))
                .unwrap(),
            &OsString::from("c".repeat(40))
        );
        assert_eq!(
            env.get(&OsString::from("AMLAS_CODING_PAIR_LAUNCHER_SHA256"))
                .unwrap(),
            &OsString::from("d".repeat(64))
        );
        if env.contains_key(&OsString::from("UNRELATED")) {
            panic!("unrelated provider environment retained");
        }
        assert_eq!(
            env.get(&OsString::from("TMPDIR")).unwrap(),
            "/opt/ml/processing/output"
        );
    }
}
