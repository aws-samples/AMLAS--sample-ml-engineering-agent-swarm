//! Fixed test-only exec metadata reporter for validated privilege fixtures.
use std::fs;
use std::os::unix::fs::MetadataExt;
extern "C" {
    fn prctl(option: i32, ...) -> i32;
}
fn main() {
    let s = fs::read_to_string("/proc/self/status").unwrap();
    let get = |key: &str| {
        s.lines()
            .find_map(|l| l.strip_prefix(&format!("{key}:")))
            .unwrap()
            .trim()
    };
    let uid: Vec<u32> = get("Uid")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let gid: Vec<u32> = get("Gid")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let groups: Vec<u32> = get("Groups")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let caps: Vec<u64> = ["CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd"]
        .iter()
        .map(|k| u64::from_str_radix(get(k), 16).unwrap())
        .collect();
    let nnp: u32 = get("NoNewPrivs").parse().unwrap();
    let seccomp: u32 = get("Seccomp").parse().unwrap();
    let securebits = unsafe { prctl(27, 0usize, 0usize, 0usize, 0usize) };
    let ns = fs::metadata("/proc/self/ns/user").unwrap().ino();
    println!("{{\"uid\":{uid:?},\"gid\":{gid:?},\"groups\":{groups:?},\"caps\":{caps:?},\"no_new_privs\":{nnp},\"seccomp\":{seccomp},\"securebits\":{securebits},\"userns_inode\":{ns}}}");
}
