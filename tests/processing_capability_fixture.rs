//! Controlled Linux-only oracle. Never included in either production image.
use std::ffi::OsString;
use std::os::unix::process::CommandExt;
use std::process::Command;

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
    fn prctl(option: i32, ...) -> i32;
    fn capget(header: *mut CapHeader, data: *mut CapData) -> i32;
    fn capset(header: *const CapHeader, data: *const CapData) -> i32;
}
const NEEDED: u32 = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 6) | (1 << 7);

fn main() {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let mode = args
        .first()
        .and_then(|s| s.to_str())
        .expect("fixed fixture mode");
    assert!(matches!(
        mode,
        "empty" | "present" | "prepared-empty" | "prepared-present"
    ));
    if !mode.starts_with("prepared-") {
        // Add inheritable bits while they are still admitted by the bounding set.
        let mut header = CapHeader {
            version: 0x20080522,
            pid: 0,
        };
        let mut before = [CapData {
            effective: 0,
            permitted: 0,
            inheritable: 0,
        }; 2];
        assert_eq!(unsafe { capget(&mut header, before.as_mut_ptr()) }, 0);
        before[0].inheritable = NEEDED;
        before[1].inheritable = 0;
        assert_eq!(unsafe { capset(&header, before.as_ptr()) }, 0);
        let last: u32 = std::fs::read_to_string("/proc/sys/kernel/cap_last_cap")
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!(last <= 63);
        for cap in 0..=last {
            if mode == "empty" || cap >= 32 || NEEDED & (1 << cap) == 0 {
                assert_eq!(
                    unsafe { prctl(24, cap as usize, 0usize, 0usize, 0usize) },
                    0
                );
            }
        }
        let header = CapHeader {
            version: 0x20080522,
            pid: 0,
        };
        let data = [
            CapData {
                effective: NEEDED,
                permitted: NEEDED,
                inheritable: NEEDED,
            },
            CapData {
                effective: 0,
                permitted: 0,
                inheritable: 0,
            },
        ];
        assert_eq!(unsafe { capset(&header, data.as_ptr()) }, 0);
        let error = Command::new("/usr/local/bin/amlas-capability-fixture")
            .arg(format!("prepared-{mode}"))
            .args(&args[1..])
            .exec();
        panic!("fixed fixture exec failed: {error}");
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let get = |key: &str| {
        status
            .lines()
            .find_map(|s| s.strip_prefix(&format!("{key}:")))
            .unwrap()
            .trim()
    };
    let expected = format!("{NEEDED:016x}");
    for key in ["CapEff", "CapPrm", "CapInh"] {
        assert_eq!(get(key), expected);
    }
    assert_eq!(
        get("CapBnd"),
        if mode == "prepared-empty" {
            "0000000000000000"
        } else {
            expected.as_str()
        }
    );
    assert_eq!(get("CapAmb"), "0000000000000000");
    assert_eq!(get("NoNewPrivs"), "1");
    assert!(get("Uid").split_whitespace().all(|s| s == "0"));
    assert!(get("Gid").split_whitespace().all(|s| s == "0"));
    let read = unsafe { prctl(23, 0usize, 0usize, 0usize, 0usize) };
    assert_eq!(read, i32::from(mode == "prepared-present"));
    assert_eq!(unsafe { prctl(23, 8usize, 0usize, 0usize, 0usize) }, 0);
    // Native kernel evidence: absent CAP_SETPCAP does not make DROP idempotent.
    assert_eq!(unsafe { prctl(24, 8usize, 0usize, 0usize, 0usize) }, -1);
    let drop_errno = std::io::Error::last_os_error().raw_os_error().unwrap();
    assert_eq!(drop_errno, 1);
    println!("CAPABILITY-FIXTURE {{\"phase\":\"after_exec\",\"uid\":0,\"gid\":0,\"cap_eff\":\"{}\",\"cap_prm\":\"{}\",\"cap_inh\":\"{}\",\"cap_bnd\":\"{}\",\"cap_amb\":\"{}\",\"no_new_privs\":1,\"read_chown\":{},\"read_setpcap\":0,\"drop_absent_errno\":{}}}", get("CapEff"),get("CapPrm"),get("CapInh"),get("CapBnd"),get("CapAmb"),read,drop_errno);
    let error = Command::new("/usr/local/bin/amlas-processing-init")
        .arg("acquisition")
        .args(&args[1..])
        .exec();
    panic!("fixed initializer exec failed: {error}");
}
