//! FD/name identity check shared by the fixed initializer and real Linux tests.
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;

extern "C" {
    fn openat(fd: i32, path: *const i8, flags: i32, ...) -> i32;
}

/// Verify the compiled output name still denotes the original directory FD.
pub fn verify_named_directory(parent: &File, target: &File) -> Result<(), &'static str> {
    // Linux AMD64 O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC. No caller pathname.
    let fd = unsafe { openat(parent.as_raw_fd(), c"output".as_ptr(), 0o2600000) };
    if fd < 0 {
        return Err("target_name_open");
    }
    let named = unsafe { File::from_raw_fd(fd) };
    let a = target.metadata().map_err(|_| "target_stat")?;
    let b = named.metadata().map_err(|_| "target_name_stat")?;
    if !b.is_dir() || a.dev() != b.dev() || a.ino() != b.ino() {
        return Err("target_identity");
    }
    Ok(())
}

/// Remove only present bounding capabilities, refusing every unverifiable syscall.
/// The executable supplies fixed Linux PR_CAPBSET_READ/DROP operations.
pub fn drop_bounding_capabilities(
    last: u32,
    mut read: impl FnMut(u32) -> i32,
    mut drop: impl FnMut(u32) -> i32,
) -> Result<(), &'static str> {
    if last > 63 {
        return Err("capability_range");
    }
    for cap in 0..=last {
        match read(cap) {
            0 => continue,
            1 => {
                if drop(cap) != 0 {
                    return Err("bounding_capability_drop");
                }
                if read(cap) != 0 {
                    return Err("bounding_capability_readback");
                }
            }
            _ => return Err("bounding_capability_read"),
        }
    }
    Ok(())
}

/// Native profile metadata from fixed bounded kernel reads, not caller data.
#[derive(Clone, Debug)]
pub struct NativeState {
    pub uid: [u32; 4],
    pub gid: [u32; 4],
    pub groups: Vec<u32>,
    pub threads: u32,
    pub caps: [u64; 5],
    pub no_new_privs: u32,
    pub seccomp: u32,
    pub keep_caps: i32,
    pub securebits: i32,
}
/// Fixed observed native ceiling; no caller can widen it.
pub const NATIVE_CEILING: u64 = 0x200404fb;
fn native_securebits(state: &NativeState) -> bool {
    (0..=255).contains(&state.securebits) && state.securebits & 0x14 == 0 && state.keep_caps == 0
}
/// Validate fixed kernel metadata and individual bounding reads before mutation.
pub fn validate_native_initial(
    state: &NativeState,
    last: u32,
    mut read: impl FnMut(u32) -> i32,
) -> Result<u64, &'static str> {
    if last > 63
        || state.uid[..3] != [0; 3]
        || state.gid[..3] != [0; 3]
        || ![vec![], vec![0]].contains(&state.groups)
        || state.threads != 1
        || state.no_new_privs > 1
        || state.seccomp != 2
        || !native_securebits(state)
        || state.caps[2] != 0
        || state.caps[3] != 0
        || state.caps.iter().any(|cap| cap & !NATIVE_CEILING != 0)
    {
        return Err("native_initial_profile");
    }
    if last < 63 && state.caps[4] >> (last + 1) != 0 {
        return Err("native_bounding_range");
    }
    for cap in 0..=last {
        let actual = read(cap);
        if ![0, 1].contains(&actual) || actual != ((state.caps[4] >> cap) & 1) as i32 {
            return Err("native_bounding_read");
        }
    }
    Ok(state.caps[4])
}
/// Require exact retained initial bounding mask and every permanent-drop invariant.
pub fn validate_native_final(state: &NativeState, initial_bnd: u64) -> Result<(), &'static str> {
    if initial_bnd & !NATIVE_CEILING != 0
        || state.uid != [10001; 4]
        || state.gid != [10001; 4]
        || !state.groups.is_empty()
        || state.caps[..4] != [0; 4]
        || state.caps[4] != initial_bnd
        || state.no_new_privs != 1
        || state.seccomp != 2
        || !native_securebits(state)
    {
        return Err("native_final_profile");
    }
    Ok(())
}

/// Fixed OS file boundary for raw model acquisition only.
pub mod model_file_boundary;

/// Collision-safe fixed raw-model authority handoff.
pub mod model_fd_handoff;

/// Fixed raw-model TCP port restriction, separately stacked with the filesystem policy.
pub mod model_tcp_boundary;

/// Fixed Qwen-only TCP restriction, separate from downloader policy.
pub mod qwen_tcp_boundary;

/// Fixed unprivileged official-oracle child confinement, separately compiled.
#[cfg(feature = "oracle-child-v1")]
pub mod oracle_boundary;
