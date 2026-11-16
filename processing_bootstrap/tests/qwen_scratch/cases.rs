use super::*;
use std::os::unix::fs::PermissionsExt;

#[test]
fn real_postdrop_scratch_enforces_original_components_and_private_leaf() {
    if std::env::var("AMLAS_QWEN_SCRATCH_CHILD").ok().as_deref() != Some("1") {
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "qwen_scratch_tests::real_postdrop_scratch_enforces_original_components_and_private_leaf", "--test-threads=1"])
            .env("AMLAS_QWEN_SCRATCH_CHILD", "1").status().unwrap();
        assert!(status.success());
        return;
    }
    let base = std::env::temp_dir().join(format!("qwen-scratch-{}", std::process::id()));
    fs::create_dir(&base).unwrap();
    fs::set_permissions(&base, fs::Permissions::from_mode(0o755)).unwrap();
    let mut roots = vec![];
    for case in [
        "absent",
        "valid",
        "nonempty",
        "symlink",
        "mode",
        "owner",
        "file",
        "cache_link",
        "mode_drift",
        "name_drift",
        "home_mode",
        "home_gid",
        "app_mode",
        "app_gid",
        "cache_mode",
        "cache_gid",
        "missing_home",
        "missing_app",
    ] {
        let root = base.join(case);
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
        let home = root.join("home");
        fs::create_dir(&home).unwrap();
        fs::set_permissions(&home, fs::Permissions::from_mode(0o755)).unwrap();
        let app = home.join("appuser");
        fs::create_dir(&app).unwrap();
        fs::set_permissions(&app, fs::Permissions::from_mode(0o700)).unwrap();
        let appfd = File::open(&app).unwrap();
        assert_eq!(unsafe { fchown(appfd.as_raw_fd(), 10001, 10001) }, 0);
        let cache = app.join(".cache");
        if case == "cache_link" {
            std::os::unix::fs::symlink(&app, &cache).unwrap();
        } else if case != "absent" {
            fs::create_dir(&cache).unwrap();
            fs::set_permissions(&cache, fs::Permissions::from_mode(0o700)).unwrap();
            let cachefd = File::open(&cache).unwrap();
            assert_eq!(unsafe { fchown(cachefd.as_raw_fd(), 10001, 10001) }, 0);
            let leaf = cache.join("amlas-qwen-sft");
            if case == "symlink" {
                std::os::unix::fs::symlink(&app, &leaf).unwrap();
            } else if case == "file" {
                fs::write(&leaf, b"synthetic").unwrap();
            } else {
                fs::create_dir(&leaf).unwrap();
                fs::set_permissions(
                    &leaf,
                    fs::Permissions::from_mode(if case == "mode" { 0o755 } else { 0o700 }),
                )
                .unwrap();
                let leaffd = File::open(&leaf).unwrap();
                if case != "owner" {
                    assert_eq!(unsafe { fchown(leaffd.as_raw_fd(), 10001, 10001) }, 0);
                }
                if case == "nonempty" {
                    fs::write(leaf.join("synthetic"), b"inert").unwrap();
                }
            }
        }
        match case {
            "home_mode" => fs::set_permissions(&home, fs::Permissions::from_mode(0o777)).unwrap(),
            "home_gid" => {
                let fd = File::open(&home).unwrap();
                assert_eq!(unsafe { fchown(fd.as_raw_fd(), 0, 10001) }, 0);
            }
            "app_mode" => fs::set_permissions(&app, fs::Permissions::from_mode(0o755)).unwrap(),
            "app_gid" => {
                let fd = File::open(&app).unwrap();
                assert_eq!(unsafe { fchown(fd.as_raw_fd(), 10001, 0) }, 0);
            }
            "cache_mode" => fs::set_permissions(&cache, fs::Permissions::from_mode(0o755)).unwrap(),
            "cache_gid" => {
                let fd = File::open(&cache).unwrap();
                assert_eq!(unsafe { fchown(fd.as_raw_fd(), 10001, 0) }, 0);
            }
            "missing_home" => fs::rename(&home, root.join("home-hidden")).unwrap(),
            "missing_app" => fs::rename(&app, home.join("app-hidden")).unwrap(),
            _ => {}
        }
        roots.push((case, File::open(&root).unwrap()));
    }
    assert_eq!(unsafe { setgroups(0, std::ptr::null()) }, 0);
    assert_eq!(unsafe { setresgid(10001, 10001, 10001) }, 0);
    assert_eq!(unsafe { setresuid(10001, 10001, 10001) }, 0);
    for (case, root) in roots {
        if case == "mode_drift" {
            std::env::set_var("AMLAS_QWEN_SCRATCH_MUTATION", "mode");
        } else if case == "name_drift" {
            std::env::set_var("AMLAS_QWEN_SCRATCH_MUTATION", "name");
        } else {
            std::env::remove_var("AMLAS_QWEN_SCRATCH_MUTATION");
        }
        let metadata_before = snapshot(&root);
        let before = fs::read_dir("/proc/self/fd").unwrap().count();
        let result = qwen_scratch_from_root(&root);
        let after = fs::read_dir("/proc/self/fd").unwrap().count();
        assert_eq!(before, after, "temporary scratch descriptors leaked");
        if !["absent", "mode_drift", "name_drift"].contains(&case) {
            assert_eq!(
                snapshot(&root),
                metadata_before,
                "invalid existing namespace was repaired"
            );
        }
        assert_eq!(
            result.is_ok(),
            case == "absent" || case == "valid",
            "case={case}, result={result:?}"
        );
    }
}

fn snapshot(root: &File) -> Vec<Option<(u64, u64, u32, u32, u32)>> {
    [
        "home",
        "home/appuser",
        "home/appuser/.cache",
        "home/appuser/.cache/amlas-qwen-sft",
    ]
    .into_iter()
    .map(|relative| {
        fs::symlink_metadata(format!("/proc/self/fd/{}/{relative}", root.as_raw_fd()))
            .ok()
            .map(|m| (m.dev(), m.ino(), m.uid(), m.gid(), m.mode()))
    })
    .collect()
}
