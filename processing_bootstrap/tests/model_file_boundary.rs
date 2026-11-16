#[allow(dead_code)]
#[path = "../src/model_file_boundary.rs"]
mod boundary;
use boundary::test_support;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
extern "C" {
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    fn pwrite(fd: i32, data: *const u8, count: usize, offset: i64) -> isize;
    fn ftruncate(fd: i32, size: i64) -> i32;
    fn mmap(addr: *mut u8, size: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
}
fn fixture() -> (PathBuf, File) {
    let p = std::env::temp_dir().join(format!(
        "model-boundary-{}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&p).unwrap();
    fs::create_dir_all(p.join("home/appuser")).unwrap();
    fs::create_dir(p.join("root")).unwrap();
    let root = File::open(&p).unwrap();
    (p, root)
}
#[test]
fn fixed_providers_are_exact_and_no_dynamic_selector() {
    assert_eq!(
        boundary::PROVIDER_PATHS,
        [
            "/home/appuser/.aws/credentials",
            "/home/appuser/.aws/config",
            "/root/.aws/credentials",
            "/root/.aws/config"
        ]
    );
    assert_eq!(boundary::MAX_INVENTORY_BYTES, 16384);
}
#[test]
fn missing_ancestor_is_recorded_explicitly_not_as_inspected_leaf() {
    let (_, root) = fixture();
    let inv = test_support::inventory_at(&root).unwrap();
    assert_eq!(inv.records()[0].missing_component_index, 3);
    assert_eq!(inv.records()[1].missing_component_index, 3);
    assert_eq!(inv.records()[2].missing_component_index, 2);
    assert_eq!(inv.records()[3].missing_component_index, 2);
    assert_eq!(inv.records()[0].ancestors.len(), 3);
    assert_eq!(inv.records()[2].ancestors.len(), 2);
}
#[test]
fn missing_leaf_keeps_finite_original_ancestor_identities() {
    let (p, root) = fixture();
    fs::create_dir(p.join("home/appuser/.aws")).unwrap();
    fs::create_dir(p.join("root/.aws")).unwrap();
    let inv = test_support::inventory_at(&root).unwrap();
    assert_eq!(inv.records()[0].missing_component_index, 4);
    assert_eq!(inv.records()[2].missing_component_index, 3);
    for record in inv.records() {
        assert!(record.ancestors.len() <= 5);
        assert!(record.ancestors.iter().all(|a| a.inode != 0));
    }
    let mut text = String::new();
    inv.as_file()
        .try_clone()
        .unwrap()
        .read_to_string(&mut text)
        .unwrap();
    assert!(text.contains("raw_model_provider_absence_before_drop_v1"));
    assert!(text.contains("missing_component_index"));
    assert!(!text.contains("post_policy_absence"));
    assert!(inv.byte_count() <= boundary::MAX_INVENTORY_BYTES);
}
#[test]
fn existing_provider_file_refuses_without_reading_contents() {
    let (p, root) = fixture();
    fs::create_dir(p.join("root/.aws")).unwrap();
    fs::write(p.join("root/.aws/config"), b"DO-NOT-READ").unwrap();
    assert!(matches!(
        test_support::inventory_at(&root),
        Err("provider_presence")
    ));
}
#[test]
fn intermediate_symlink_is_not_hidden_by_resolution() {
    let (p, root) = fixture();
    fs::create_dir(p.join("elsewhere")).unwrap();
    symlink(p.join("elsewhere"), p.join("home/appuser/.aws")).unwrap();
    assert!(matches!(
        test_support::inventory_at(&root),
        Err("provider_symlink")
    ));
}
#[test]
fn non_directory_ancestor_is_unknown_not_absence() {
    let (p, root) = fixture();
    fs::write(p.join("root/.aws"), b"literal").unwrap();
    assert!(matches!(
        test_support::inventory_at(&root),
        Err("provider_ancestor_type")
    ));
}
#[test]
fn only_enoent_can_describe_observed_absence() {
    assert_eq!(test_support::errno_is_absence(2), Ok(true));
    for errno in [1, 5, 13, 20, 40, 0] {
        assert!(test_support::errno_is_absence(errno).is_err());
    }
}
#[test]
fn appearance_during_capture_refuses_before_sealed_return() {
    let (p, root) = fixture();
    let result = test_support::inventory_at_after(&root, || {
        fs::create_dir(p.join("root/.aws")).unwrap();
    });
    assert!(matches!(result, Err("provider_inventory_changed")));
}
#[test]
fn same_name_parent_substitution_refuses() {
    let (p, root) = fixture();
    let result = test_support::inventory_at_after(&root, || {
        fs::rename(p.join("home"), p.join("home-original")).unwrap();
        fs::create_dir_all(p.join("home/appuser")).unwrap();
    });
    assert!(matches!(result, Err("provider_inventory_changed")));
}
#[test]
fn ancestor_mode_drift_refuses() {
    let (p, root) = fixture();
    let result = test_support::inventory_at_after(&root, || {
        fs::set_permissions(p.join("root"), fs::Permissions::from_mode(0o700)).unwrap();
    });
    assert!(matches!(result, Err("provider_inventory_changed")));
}
#[test]
fn actual_memfd_has_four_irreversible_seals_and_readonly_ofd() {
    let literal = b"fixed producer metadata";
    let mut file = test_support::sealed_bytes(literal).unwrap();
    assert_eq!(unsafe { fcntl(file.as_raw_fd(), 1034) }, 15);
    assert_eq!(unsafe { fcntl(file.as_raw_fd(), 3) } & 3, 0);
    let mut got = Vec::new();
    file.read_to_end(&mut got).unwrap();
    assert_eq!(got, literal);
    file.seek(SeekFrom::Start(0)).unwrap();
    let writer = OpenOptions::new()
        .read(true)
        .write(true)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
        .unwrap();
    assert_eq!(
        unsafe { pwrite(writer.as_raw_fd(), b"X".as_ptr(), 1, 0) },
        -1
    );
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(1));
    for size in [0, literal.len() as i64 + 1] {
        assert_eq!(unsafe { ftruncate(writer.as_raw_fd(), size) }, -1);
        assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(1));
    }
    assert_eq!(
        unsafe { mmap(std::ptr::null_mut(), 4096, 3, 1, writer.as_raw_fd(), 0) } as isize,
        -1
    );
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(1));
    assert_eq!(unsafe { fcntl(writer.as_raw_fd(), 1033, 16) }, -1);
    let mut after = Vec::new();
    file.read_to_end(&mut after).unwrap();
    assert_eq!(after, literal);
}
#[test]
fn sealed_inventory_rejects_empty_or_oversized_payload() {
    assert!(matches!(
        test_support::sealed_bytes(b""),
        Err("inventory_size")
    ));
    assert!(matches!(
        test_support::sealed_bytes(&vec![0; boundary::MAX_INVENTORY_BYTES + 1]),
        Err("inventory_size")
    ));
}
#[test]
fn all_abi6_filesystem_actions_including_ioctl_are_handled() {
    assert_eq!(test_support::policy_mask(6), Ok(0xffff));
    assert_ne!(test_support::policy_mask(6).unwrap() & (1 << 15), 0);
    for abi in [-1, 0, 1, 2, 3, 4, 5, 7, 8, 99] {
        assert!(matches!(
            test_support::policy_mask(abi),
            Err("landlock_abi_unsupported")
        ));
    }
}

#[test]
fn policy_exact_syscalls_and_no_filter_downgrade() {
    assert_eq!(
        test_support::policy_attempts(6, 1, false, false, false).unwrap(),
        vec!["abi", "nnp", "create", "add", "restrict", "nnp"]
    );
    for abi in [-1, 0, 5, 7] {
        assert!(test_support::policy_attempts(abi, 1, false, false, false).is_err());
    }
    assert!(matches!(
        test_support::policy_attempts(6, 0, false, false, false),
        Err("landlock_nnp")
    ));
    assert!(matches!(
        test_support::policy_attempts(6, -1, false, false, false),
        Err("landlock_nnp")
    ));
}
#[test]
fn every_create_add_restrict_failure_refuses() {
    assert!(matches!(
        test_support::policy_attempts(6, 1, true, false, false),
        Err("landlock_create")
    ));
    assert!(matches!(
        test_support::policy_attempts(6, 1, false, true, false),
        Err("landlock_add")
    ));
    assert!(matches!(
        test_support::policy_attempts(6, 1, false, false, true),
        Err("landlock_restrict")
    ));
}
#[test]
fn output_write_mask_excludes_execute_devices_symlinks_and_ioctl() {
    assert_eq!(test_support::output_path_rights(), 0x61be);
    for bit in [0, 6, 9, 10, 11, 12, 15] {
        assert_eq!(test_support::output_path_rights() & (1 << bit), 0);
    }
}
#[test]
fn only_exact_source_owned_alias_target_is_admitted() {
    let (p, root) = fixture();
    fs::create_dir_all(p.join("usr/lib")).unwrap();
    symlink("usr/lib", p.join("lib")).unwrap();
    test_support::alias_at(&root, 0).unwrap();
    fs::rename(p.join("lib"), p.join("lib-original")).unwrap();
    symlink("elsewhere", p.join("lib")).unwrap();
    assert!(matches!(
        test_support::alias_at(&root, 0),
        Err("runtime_alias_target")
    ));
}
#[test]
fn alias_name_substitution_is_refused_before_policy() {
    let (p, root) = fixture();
    fs::create_dir_all(p.join("usr/lib")).unwrap();
    symlink("usr/lib", p.join("lib")).unwrap();
    let result = test_support::alias_at_after(&root, 0, || {
        fs::rename(p.join("lib"), p.join("lib-original")).unwrap();
        symlink("usr/lib", p.join("lib")).unwrap();
    });
    assert!(matches!(result, Err("runtime_alias_changed")));
}

#[test]
fn root_alias_parent_must_pass_immutable_mode_checks() {
    let (p, root) = fixture();
    fs::create_dir_all(p.join("usr/lib")).unwrap();
    symlink("usr/lib", p.join("lib")).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o777)).unwrap();
    assert!(matches!(
        test_support::alias_at(&root, 0),
        Err("runtime_writable")
    ));
}
#[test]
fn retained_alias_parent_mode_is_rechecked() {
    let (p, root) = fixture();
    fs::create_dir_all(p.join("usr/lib")).unwrap();
    symlink("usr/lib", p.join("lib")).unwrap();
    let result = test_support::alias_at_after(&root, 0, || {
        fs::set_permissions(&p, fs::Permissions::from_mode(0o700)).unwrap();
    });
    assert!(matches!(result, Err("runtime_alias_changed")));
}

#[test]
fn exact_production_twenty_eight_rules_reach_kernel_with_frozen_rights() {
    let expected = vec![
        ("/app", 12),
        ("/usr/local/lib/python3.11", 12),
        ("/usr/local/bin/python3.11", 5),
        ("/usr/local/lib/libpython3.11.so.1.0", 4),
        ("/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2", 5),
        ("/usr/lib/x86_64-linux-gnu/libbz2.so.1.0.4", 4),
        ("/usr/lib/x86_64-linux-gnu/libc.so.6", 4),
        ("/usr/lib/x86_64-linux-gnu/libcrypto.so.3", 4),
        ("/usr/lib/x86_64-linux-gnu/libdl.so.2", 4),
        ("/usr/lib/x86_64-linux-gnu/libffi.so.8.1.4", 4),
        ("/usr/lib/x86_64-linux-gnu/libgcc_s.so.1", 4),
        ("/usr/lib/x86_64-linux-gnu/liblzma.so.5.8.1", 4),
        ("/usr/lib/x86_64-linux-gnu/libm.so.6", 4),
        ("/usr/lib/x86_64-linux-gnu/libpthread.so.0", 4),
        ("/usr/lib/x86_64-linux-gnu/librt.so.1", 4),
        ("/usr/lib/x86_64-linux-gnu/libssl.so.3", 4),
        ("/usr/lib/x86_64-linux-gnu/libz.so.1.3.1", 4),
        ("/usr/lib/x86_64-linux-gnu/libzstd.so.1.5.7", 4),
        ("/etc/ssl/certs/ca-certificates.crt", 4),
        ("/etc/resolv.conf", 4),
        ("/etc/hosts", 4),
        ("/etc/nsswitch.conf", 4),
        ("/etc/gai.conf", 4),
        ("/etc/ld.so.cache", 4),
        ("/dev/null", 4),
        ("current_pid_status", 4),
        ("current_pid_fd", 8),
        ("/opt/ml/processing/output", 0x61be),
    ];
    assert_eq!(test_support::production_plan().unwrap(), expected);
}
#[test]
fn production_metadata_rejects_foreign_writable_and_wrong_type_runtime() {
    assert_eq!(
        test_support::metadata_case(0, 0, 0, 0o40755, 0, 1, true),
        Ok(())
    );
    for (uid, gid, mode) in [
        (10001, 0, 0o40755),
        (0, 10001, 0o40755),
        (0, 0, 0o40777),
        (0, 0, 0o100644),
        (0, 0, 0o120777),
    ] {
        assert!(test_support::metadata_case(0, uid, gid, mode, 0, 1, true).is_err());
    }
}
#[test]
fn production_metadata_limits_devnull_and_output_qualification() {
    assert_eq!(
        test_support::metadata_case(24, 0, 0, 0o20666, 0x103, 2, true),
        Ok(())
    );
    for (mode, rdev) in [(0o100644, 0x103), (0o20666, 0x105)] {
        assert!(test_support::metadata_case(24, 0, 0, mode, rdev, 2, true).is_err());
    }
    assert_eq!(
        test_support::metadata_case(27, 10001, 10001, 0o40700, 0, 4, true),
        Ok(())
    );
    for (uid, gid, mode) in [
        (0, 0, 0o40700),
        (10001, 10001, 0o40755),
        (10001, 10001, 0o100700),
    ] {
        assert!(test_support::metadata_case(27, uid, gid, mode, 0, 4, true).is_err());
    }
}
#[test]
fn output_rule_authority_retains_original_ofd_and_refuses_substitute_inode() {
    let (p, original) = fixture();
    let named = File::open(&p).unwrap();
    extern "C" {
        fn flock(fd: i32, op: i32) -> i32;
    }
    assert_eq!(unsafe { flock(original.as_raw_fd(), 2 | 4) }, 0);
    let authority = test_support::output_authority(&original, &named).unwrap();
    assert_eq!(unsafe { flock(authority.as_raw_fd(), 2 | 4) }, 0);
    fs::create_dir(p.join("other-output")).unwrap();
    let other = File::open(p.join("other-output")).unwrap();
    assert!(matches!(
        test_support::output_authority(&original, &other),
        Err("output_identity")
    ));
}

#[test]
fn measured_post_drop_proc_leaf_owner_zero_is_qualified_separately_from_pid_dir() {
    assert_eq!(
        test_support::proc_case(0, 25, 0, 0, 0o100444, 3, true),
        Ok(())
    );
    assert_eq!(
        test_support::proc_case(0, 26, 0, 0, 0o40500, 3, true),
        Ok(())
    );
    assert_eq!(
        test_support::proc_case(0, 25, 10001, 10001, 0o40555, 2, false),
        Ok(())
    );
    assert!(test_support::proc_case(0, 25, 10001, 10001, 0o100444, 3, true).is_err());
}
#[test]
fn dumpability_one_and_unknown_states_remain_distinct_and_closed() {
    assert_eq!(
        test_support::proc_case(1, 25, 10001, 10001, 0o100444, 3, true),
        Ok(())
    );
    assert_eq!(
        test_support::proc_case(1, 26, 10001, 10001, 0o40500, 3, true),
        Ok(())
    );
    for value in [-1, 2, 3, 99] {
        assert!(test_support::proc_case(value, 25, 10001, 10001, 0o100444, 3, true).is_err());
    }
    assert!(test_support::proc_case(1, 25, 0, 0, 0o100444, 3, true).is_err());
}

#[test]
fn all_five_production_aliases_have_exact_source_owned_target_checks() {
    let cases = [
        ("lib", "usr/lib"),
        ("lib64", "usr/lib64"),
        ("usr/lib/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"),
        ("usr/lib/ssl/certs", "/etc/ssl/certs"),
        (
            "usr/lib64/ld-linux-x86-64.so.2",
            "../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
        ),
    ];
    for (index, (name, target)) in cases.iter().enumerate() {
        let (p, root) = fixture();
        fs::create_dir_all(p.join("usr/lib/ssl")).unwrap();
        fs::create_dir_all(p.join("usr/lib64")).unwrap();
        symlink(target, p.join(name)).unwrap();
        test_support::alias_at(&root, index).unwrap();
        fs::rename(p.join(name), p.join(format!("saved-link-{index}"))).unwrap();
        symlink("wrong", p.join(name)).unwrap();
        assert!(matches!(
            test_support::alias_at(&root, index),
            Err("runtime_alias_target")
        ));
    }
}
#[test]
fn actual_provider_walk_eacces_is_unknown_not_missing() {
    const CHILD: &str = "AMLAS_PROVIDER_PERMISSION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "actual_provider_walk_eacces_is_unknown_not_missing",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(
            status.success(),
            "isolated permission child must verify real DAC denial"
        );
        return;
    }
    let (p, root) = fixture();
    let directory = p.join("root/.aws");
    fs::create_dir(&directory).unwrap();
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o755)).unwrap();
    #[repr(C)]
    struct Header {
        version: u32,
        pid: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Caps {
        effective: u32,
        permitted: u32,
        inheritable: u32,
    }
    extern "C" {
        fn syscall(number: i64, ...) -> i64;
    }
    let header = Header {
        version: 0x20080522,
        pid: 0,
    };
    let mut caps = [Caps {
        effective: 0,
        permitted: 0,
        inheritable: 0,
    }; 2];
    // Only remove privileges in this isolated test thread; never elevate or skip root.
    assert_eq!(
        unsafe { syscall(126, &header as *const Header, caps.as_ptr()) },
        0
    );
    assert_eq!(
        unsafe { syscall(125, &header as *const Header, caps.as_mut_ptr()) },
        0
    );
    assert!(caps
        .iter()
        .all(|c| c.effective == 0 && c.permitted == 0 && c.inheritable == 0));
    let missing = directory.join("credentials");
    assert_eq!(fs::metadata(&missing).unwrap_err().raw_os_error(), Some(2));
    assert!(
        test_support::inventory_at(&root).is_ok(),
        "positive traversal must observe true absence"
    );
    fs::set_permissions(&directory, fs::Permissions::from_mode(0)).unwrap();
    assert_eq!(
        fs::metadata(&missing).unwrap_err().raw_os_error(),
        Some(13),
        "fixture must actually yield EACCES"
    );
    assert!(matches!(
        test_support::inventory_at(&root),
        Err("provider_observation_unknown")
    ));
}
