use super::*;
use std::os::unix::fs::PermissionsExt;

fn isolated(name: &str) -> bool {
    if std::env::var("AMLAS_CASE_PARENT_CHILD").ok().as_deref() == Some("1") {
        return true;
    }
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--test-threads=1"])
        .env("AMLAS_CASE_PARENT_CHILD", "1")
        .status()
        .unwrap();
    assert!(
        status.success(),
        "compiled original parent/output control failed"
    );
    false
}

fn fixture(parent_uid: u32) -> (File, (u64, u64), (u64, u64)) {
    fs::create_dir_all("/opt/ml/processing/output").unwrap();
    for path in ["/opt", "/opt/ml", "/opt/ml/processing"] {
        let fd = File::open(path).unwrap();
        let uid = if path == "/opt/ml/processing" {
            parent_uid
        } else {
            0
        };
        assert_eq!(unsafe { fchown(fd.as_raw_fd(), uid, uid) }, 0);
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let parent = File::open("/opt/ml/processing").unwrap();
    let output = File::open("/opt/ml/processing/output").unwrap();
    assert_eq!(unsafe { fchown(output.as_raw_fd(), 0, 0) }, 0);
    fs::set_permissions(
        "/opt/ml/processing/output",
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    assert_eq!(
        fs::read_dir("/opt/ml/processing/output").unwrap().count(),
        0
    );
    let p = parent.metadata().unwrap();
    let o = output.metadata().unwrap();
    (parent, (p.dev(), p.ino()), (o.dev(), o.ino()))
}

fn unchanged_parent(parent: &File, identity: (u64, u64), uid: u32) {
    let p = parent.metadata().unwrap();
    assert_eq!((p.dev(), p.ino()), identity);
    assert_eq!((p.uid(), p.gid(), p.mode() & 0o7777), (uid, uid, 0o755));
}

#[test]
fn case_root_parent_accepts_original_empty_output() {
    if !isolated("case_output_parent_tests::case_root_parent_accepts_original_empty_output") {
        return;
    }
    let (parent, p, original_output) = fixture(0);
    let output = case_parent_test_initialize(WorkerProfile::CaseFreeze)
        .expect("CaseFreeze root-owned input ancestor must reach original output guards");
    let o = output.metadata().unwrap();
    assert_eq!((o.dev(), o.ino()), original_output);
    assert_eq!((o.uid(), o.gid(), o.mode() & 0o7777), (USER, USER, 0o700));
    unchanged_parent(&parent, p, 0);
}

#[test]
fn case_worker_owned_parent_refuses_without_repair() {
    if !isolated("case_output_parent_tests::case_worker_owned_parent_refuses_without_repair") {
        return;
    }
    let (parent, p, output) = fixture(USER);
    assert_eq!(
        case_parent_test_initialize(WorkerProfile::CaseFreeze).err(),
        Some("foreign_parent")
    );
    unchanged_parent(&parent, p, USER);
    let named = fs::metadata("/opt/ml/processing/output").unwrap();
    assert_eq!((named.dev(), named.ino()), output);
    assert_eq!(
        (named.uid(), named.gid(), named.mode() & 0o7777),
        (0, 0, 0o755)
    );
}

#[test]
fn legacy_profiles_keep_worker_parent_and_original_output_fd() {
    if !isolated(
        "case_output_parent_tests::legacy_profiles_keep_worker_parent_and_original_output_fd",
    ) {
        return;
    }
    for profile in [
        WorkerProfile::Acquisition,
        WorkerProfile::Inspection,
        WorkerProfile::Preparation,
        WorkerProfile::ModelAcquisition,
    ] {
        let (parent, p, original_output) = fixture(USER);
        let output = case_parent_test_initialize(profile).unwrap();
        let o = output.metadata().unwrap();
        assert_eq!((o.dev(), o.ino()), original_output);
        assert_eq!((o.uid(), o.gid(), o.mode() & 0o7777), (USER, USER, 0o700));
        unchanged_parent(&parent, p, USER);
    }
}

#[test]
fn legacy_profiles_still_refuse_root_parent_without_repair() {
    if !isolated(
        "case_output_parent_tests::legacy_profiles_still_refuse_root_parent_without_repair",
    ) {
        return;
    }
    for profile in [
        WorkerProfile::Acquisition,
        WorkerProfile::Inspection,
        WorkerProfile::Preparation,
        WorkerProfile::ModelAcquisition,
    ] {
        let (parent, p, output) = fixture(0);
        assert_eq!(
            case_parent_test_initialize(profile).err(),
            Some("foreign_parent")
        );
        unchanged_parent(&parent, p, 0);
        let named = fs::metadata("/opt/ml/processing/output").unwrap();
        assert_eq!((named.dev(), named.ino()), output);
        assert_eq!(
            (named.uid(), named.gid(), named.mode() & 0o7777),
            (0, 0, 0o755)
        );
    }
}

#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn pair_root_parent_reuses_original_output_fd() {
    if !isolated("case_output_parent_tests::pair_root_parent_reuses_original_output_fd") {
        return;
    }
    let (parent, p, original_output) = fixture(0);
    let output = case_parent_test_initialize(WorkerProfile::CodingPair).unwrap();
    let o = output.metadata().unwrap();
    assert_eq!((o.dev(), o.ino()), original_output);
    assert_eq!((o.uid(), o.gid(), o.mode() & 0o7777), (USER, USER, 0o700));
    unchanged_parent(&parent, p, 0);
}
#[cfg(feature = "sagemaker-native-v1")]
#[test]
fn pair_worker_parent_refuses_without_repair() {
    if !isolated("case_output_parent_tests::pair_worker_parent_refuses_without_repair") {
        return;
    }
    let (parent, p, original_output) = fixture(USER);
    assert_eq!(
        case_parent_test_initialize(WorkerProfile::CodingPair).err(),
        Some("foreign_parent")
    );
    unchanged_parent(&parent, p, USER);
    let named = fs::metadata("/opt/ml/processing/output").unwrap();
    assert_eq!((named.dev(), named.ino()), original_output);
    assert_eq!(
        (named.uid(), named.gid(), named.mode() & 0o7777),
        (0, 0, 0o755)
    );
}
