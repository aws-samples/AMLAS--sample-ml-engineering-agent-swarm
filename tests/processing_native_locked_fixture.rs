//! Test-only locked-off keepcaps fixture. SETPCAP is removed before helper exec.
use std::os::unix::process::CommandExt;
use std::process::Command;
#[repr(C)]
struct Header {
    version: u32,
    pid: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Data {
    effective: u32,
    permitted: u32,
    inheritable: u32,
}
extern "C" {
    fn prctl(option: i32, ...) -> i32;
    fn capset(header: *const Header, data: *const Data) -> i32;
}
fn main() {
    assert_eq!(unsafe { prctl(28, 32usize, 0usize, 0usize, 0usize) }, 0);
    assert_eq!(unsafe { prctl(24, 8usize, 0usize, 0usize, 0usize) }, 0);
    let header = Header {
        version: 0x20080522,
        pid: 0,
    };
    let data = [
        Data {
            effective: 0x200404fb,
            permitted: 0x200404fb,
            inheritable: 0,
        },
        Data {
            effective: 0,
            permitted: 0,
            inheritable: 0,
        },
    ];
    assert_eq!(unsafe { capset(&header, data.as_ptr()) }, 0);
    assert_eq!(unsafe { prctl(27, 0usize, 0usize, 0usize, 0usize) }, 32);
    assert_eq!(unsafe { prctl(7, 0usize, 0usize, 0usize, 0usize) }, 0);
    assert_eq!(unsafe { prctl(23, 8usize, 0usize, 0usize, 0usize) }, 0);
    let error = Command::new("/usr/local/bin/amlas-processing-init")
        .arg("acquisition")
        .args(std::env::args_os().skip(1))
        .exec();
    panic!("fixed test helper exec failed: {error}");
}
