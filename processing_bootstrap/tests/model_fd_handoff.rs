#[allow(dead_code)]
#[path = "../src/model_fd_handoff.rs"]
mod model_fd_handoff;
#[allow(dead_code)]
#[path = "../src/model_file_boundary.rs"]
mod model_file_boundary;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::process::Command;
extern "C" {
    fn fchown(fd: i32, uid: u32, gid: u32) -> i32;
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    fn flock(fd: i32, op: i32) -> i32;
    fn dup2(old: i32, new: i32) -> i32;
    fn close(fd: i32) -> i32;
}
#[test]
fn actual_fixed_handoff_cases_are_isolated_from_parent_fd_table() {
    for case in [
        "swapped",
        "fixed",
        "output4",
        "inventory3",
        "higher",
        "unsealed",
        "writable",
        "size",
        "owner",
        "mode",
        "unrelated",
        "typed",
        "stdio_source",
    ] {
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "child_fixed_handoff_case", "--nocapture"])
            .env("AMLAS_HANDOFF_TEST_CASE", case)
            .status()
            .unwrap();
        assert!(status.success(), "childcase {case}");
    }
}
#[test]
fn child_fixed_handoff_case() {
    let Ok(case) = std::env::var("AMLAS_HANDOFF_TEST_CASE") else {
        return;
    };
    let p = std::env::temp_dir().join(format!(
        "fd-handoff-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&p).unwrap();
    fs::create_dir_all(p.join("home/appuser")).unwrap();
    fs::create_dir(p.join("root")).unwrap();
    fs::create_dir(p.join("output")).unwrap();
    let fixture_root = File::open(&p).unwrap();
    let producer = model_file_boundary::test_support::inventory_at(&fixture_root).unwrap();
    let mut expected = Vec::new();
    producer
        .as_file()
        .try_clone()
        .unwrap()
        .read_to_end(&mut expected)
        .unwrap();
    let mut inventory = producer.as_file().try_clone().unwrap();
    inventory.seek(SeekFrom::Start(0)).unwrap();
    let bytes = producer.byte_count();
    drop(producer);
    drop(fixture_root);
    let output = File::open(p.join("output")).unwrap();
    assert_eq!(unsafe { fchown(output.as_raw_fd(), 10001, 10001) }, 0);
    fs::set_permissions(p.join("output"), fs::Permissions::from_mode(0o700)).unwrap();
    assert_eq!(unsafe { flock(output.as_raw_fd(), 2 | 4) }, 0);
    let out_id = (
        output.metadata().unwrap().dev(),
        output.metadata().unwrap().ino(),
    );
    let inv_id = (
        inventory.metadata().unwrap().dev(),
        inventory.metadata().unwrap().ino(),
    );
    let output = if case == "owner" {
        assert_eq!(unsafe { fchown(output.as_raw_fd(), 0, 0) }, 0);
        output
    } else if case == "mode" {
        fs::set_permissions(p.join("output"), fs::Permissions::from_mode(0o755)).unwrap();
        output
    } else {
        output
    };
    let inventory = if case == "unsealed" {
        let path = p.join("unsealed");
        fs::write(&path, b"literal").unwrap();
        drop(inventory);
        File::open(path).unwrap()
    } else if case == "writable" {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(format!("/proc/self/fd/{}", inventory.as_raw_fd()))
            .unwrap();
        drop(inventory);
        file
    } else {
        inventory
    };
    let expected_count = if case == "size" { bytes + 1 } else { bytes };
    if case == "typed" {
        let root = File::open(&p).unwrap();
        let typed = model_file_boundary::test_support::inventory_at(&root).unwrap();
        drop(root);
        drop(inventory);
        let fixed = model_fd_handoff::handoff_model_fds(output, typed).unwrap();
        assert_eq!(fixed.output().as_raw_fd(), 3);
        assert_eq!(fixed.inventory().as_raw_fd(), 4);
        assert_eq!(fixed.inventory_byte_count(), bytes);
        assert_eq!(unsafe { fcntl(4, 1034) }, 15);
        let mut data = Vec::new();
        fixed
            .inventory()
            .try_clone()
            .unwrap()
            .read_to_end(&mut data)
            .unwrap();
        assert_eq!(data, expected);
        println!("fixedcase typed PASS");
        std::process::exit(0);
    }
    // Transfer ownership to raw temporaries, then arrange sources without double-close.
    let out_temp = unsafe { fcntl(output.as_raw_fd(), 1030, 40) };
    let inv_temp = unsafe { fcntl(inventory.as_raw_fd(), 1030, 40) };
    assert!(out_temp >= 40 && inv_temp >= 40);
    drop(output);
    drop(inventory);
    let (out_source, inv_source) = match case.as_str() {
        "swapped" => (4, 3),
        "fixed" => (3, 4),
        "output4" => (4, 7),
        "inventory3" => (8, 3),
        "stdio_source" => (1, 9),
        _ => (8, 9),
    };
    assert_eq!(unsafe { dup2(out_temp, out_source) }, out_source);
    assert_eq!(unsafe { dup2(inv_temp, inv_source) }, inv_source);
    assert_eq!(unsafe { close(out_temp) }, 0);
    assert_eq!(unsafe { close(inv_temp) }, 0);
    let output = unsafe { File::from_raw_fd(out_source) };
    let inventory = unsafe { File::from_raw_fd(inv_source) };
    let low = unsafe { fcntl(0, 1030, 20) };
    let high = unsafe { fcntl(0, 1030, 4096) };
    assert!(low >= 20 && high >= 4096);
    let result = model_fd_handoff::test_support::handoff_files(output, inventory, expected_count);
    if [
        "unsealed",
        "writable",
        "size",
        "owner",
        "mode",
        "stdio_source",
    ]
    .contains(&case.as_str())
    {
        assert!(result.is_err(), "invalidcase {case}");
        std::process::exit(0);
    }
    let fixed = result.unwrap_or_else(|stage| {
        let errno = std::io::Error::last_os_error().raw_os_error();
        let kernel = fs::read_to_string("/proc/sys/kernel/osrelease")
            .unwrap_or_else(|_| "unavailable".to_string());
        panic!(
            "fixed handoff refused stage={stage} errno={errno:?} kernel={}",
            kernel.trim()
        );
    });
    assert_eq!(fixed.output().as_raw_fd(), 3);
    assert_eq!(fixed.inventory().as_raw_fd(), 4);
    assert_eq!(fixed.inventory_byte_count(), bytes);
    assert_eq!(
        (
            fixed.output().metadata().unwrap().dev(),
            fixed.output().metadata().unwrap().ino()
        ),
        out_id
    );
    assert_eq!(
        (
            fixed.inventory().metadata().unwrap().dev(),
            fixed.inventory().metadata().unwrap().ino()
        ),
        inv_id
    );
    assert_eq!(unsafe { fcntl(3, 1) }, 0);
    assert_eq!(unsafe { fcntl(4, 1) }, 0);
    assert_eq!(unsafe { fcntl(4, 3) } & 3, 0);
    assert_eq!(unsafe { fcntl(4, 1034) }, 15);
    let mut read = Vec::new();
    fixed
        .inventory()
        .try_clone()
        .unwrap()
        .read_to_end(&mut read)
        .unwrap();
    assert_eq!(read, expected);
    assert_eq!(unsafe { flock(3, 2 | 4) }, 0);
    let contender = File::open(p.join("output")).unwrap();
    assert_eq!(unsafe { flock(contender.as_raw_fd(), 2 | 4) }, -1);
    assert_eq!(unsafe { fcntl(low, 1) }, -1);
    assert_eq!(unsafe { fcntl(high, 1) }, -1);
    for fd in 0..3 {
        assert!(unsafe { fcntl(fd, 1) } >= 0);
    }
    println!("fixedcase {case} PASS");
    std::process::exit(0);
}
