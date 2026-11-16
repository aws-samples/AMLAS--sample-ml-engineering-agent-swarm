//! Private fixed model descriptor handoff; no caller selects targets.
use crate::model_file_boundary::{SealedProviderInventory, MAX_INVENTORY_BYTES};
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
const MAX_FDS: u64 = 1048576;
extern "C" {
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    fn dup2(old: i32, new: i32) -> i32;
    fn flock(fd: i32, op: i32) -> i32;
    fn getrlimit(resource: i32, value: *mut Limits) -> i32;
    fn syscall(number: i64, ...) -> i64;
}
#[repr(C)]
struct Limits {
    current: u64,
    maximum: u64,
}
#[derive(PartialEq, Eq)]
struct Identity {
    dev: u64,
    inode: u64,
    size: u64,
    uid: u32,
    gid: u32,
    mode: u32,
}
fn identity(file: &File) -> Result<Identity, &'static str> {
    let m = file.metadata().map_err(|_| "handoff_stat")?;
    Ok(Identity {
        dev: m.dev(),
        inode: m.ino(),
        size: m.len(),
        uid: m.uid(),
        gid: m.gid(),
        mode: m.mode(),
    })
}
fn access(file: &File) -> Result<i32, &'static str> {
    let value = unsafe { fcntl(file.as_raw_fd(), 3) };
    if value < 0 {
        Err("handoff_access")
    } else {
        Ok(value)
    }
}
fn output_identity(file: &File) -> Result<Identity, &'static str> {
    let id = identity(file)?;
    if id.mode & 0o170000 != 0o040000 || (id.uid, id.gid, id.mode & 0o7777) != (10001, 10001, 0o700)
    {
        return Err("handoff_output_identity");
    }
    let flags = access(file)?;
    if flags & 3 != 0 || flags & 0x200000 != 0 {
        return Err("handoff_output_access");
    }
    if unsafe { flock(file.as_raw_fd(), 2 | 4) } != 0 {
        return Err("handoff_output_lease");
    }
    Ok(id)
}
fn inventory_identity(file: &File, count: usize) -> Result<Identity, &'static str> {
    let id = identity(file)?;
    let meta = file.metadata().map_err(|_| "handoff_stat")?;
    if count == 0
        || count > MAX_INVENTORY_BYTES
        || id.size != count as u64
        || id.mode & 0o170000 != 0o100000
        || meta.nlink() != 0
    {
        return Err("handoff_inventory_identity");
    }
    let flags = access(file)?;
    if flags & 3 != 0 || flags & 0x200000 != 0 {
        return Err("handoff_inventory_access");
    }
    if unsafe { fcntl(file.as_raw_fd(), 1034) } != 15 {
        return Err("handoff_inventory_seals");
    }
    Ok(id)
}
fn safe_duplicate(file: &File) -> Result<File, &'static str> {
    let fd = unsafe { fcntl(file.as_raw_fd(), 1030, 5) };
    if fd < 5 {
        return Err("handoff_safe_duplicate");
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}
pub struct FixedModelFds {
    output: File,
    inventory: File,
    byte_count: usize,
}
impl FixedModelFds {
    pub fn output(&self) -> &File {
        &self.output
    }
    pub fn inventory(&self) -> &File {
        &self.inventory
    }
    pub fn inventory_byte_count(&self) -> usize {
        self.byte_count
    }
}
fn handoff_files(
    output: File,
    inventory: File,
    count: usize,
) -> Result<FixedModelFds, &'static str> {
    if output.as_raw_fd() < 3 || inventory.as_raw_fd() < 3 {
        return Err("handoff_stdio_source");
    }
    for fd in 0..3 {
        if unsafe { fcntl(fd, 1) } < 0 {
            return Err("handoff_stdio_missing");
        }
    }
    let mut limit = Limits {
        current: 0,
        maximum: 0,
    };
    if unsafe { getrlimit(7, &mut limit) } != 0
        || limit.maximum > MAX_FDS
        || limit.maximum < 5
        || limit.current > limit.maximum
    {
        return Err("handoff_fd_bound");
    }
    let out_id = output_identity(&output)?;
    let inv_id = inventory_identity(&inventory, count)?;
    // Both sources remain live until both same-OFD safe copies exist.
    let out_temp = safe_duplicate(&output)?;
    let inv_temp = safe_duplicate(&inventory)?;
    if out_temp.as_raw_fd() == inv_temp.as_raw_fd() {
        return Err("handoff_duplicate_collision");
    }
    drop(output);
    drop(inventory);
    if unsafe { dup2(out_temp.as_raw_fd(), 3) } != 3 {
        return Err("handoff_output_dup");
    }
    let fixed_output = unsafe { File::from_raw_fd(3) };
    if unsafe { dup2(inv_temp.as_raw_fd(), 4) } != 4 {
        return Err("handoff_inventory_dup");
    }
    let fixed_inventory = unsafe { File::from_raw_fd(4) };
    if output_identity(&fixed_output)? != out_id
        || inventory_identity(&fixed_inventory, count)? != inv_id
    {
        return Err("handoff_identity_changed");
    }
    if unsafe { fcntl(3, 1) } != 0 || unsafe { fcntl(4, 1) } != 0 {
        return Err("handoff_inheritance");
    }
    drop(out_temp);
    drop(inv_temp);
    if unsafe { syscall(436, 5u32, u32::MAX, 0u32) } != 0 {
        return Err("handoff_close_range");
    }
    for fd in 5..limit.maximum as i32 {
        if unsafe { fcntl(fd, 1) } >= 0 {
            return Err("handoff_unrelated_fd");
        }
        if std::io::Error::last_os_error().raw_os_error() != Some(9) {
            return Err("handoff_fd_verification");
        }
    }
    if output_identity(&fixed_output)? != out_id
        || inventory_identity(&fixed_inventory, count)? != inv_id
    {
        return Err("handoff_identity_changed");
    }
    for fd in 0..3 {
        if unsafe { fcntl(fd, 1) } < 0 {
            return Err("handoff_stdio_missing");
        }
    }
    Ok(FixedModelFds {
        output: fixed_output,
        inventory: fixed_inventory,
        byte_count: count,
    })
}
/// Consuming private startup seam: targets are fixed, no numeric/path selector.
pub fn handoff_model_fds(
    output: File,
    inventory: SealedProviderInventory,
) -> Result<FixedModelFds, &'static str> {
    if output.as_raw_fd() < 3 || inventory.as_file().as_raw_fd() < 3 {
        return Err("handoff_stdio_source");
    }
    let count = inventory.byte_count();
    let file = inventory
        .as_file()
        .try_clone()
        .map_err(|_| "handoff_inventory_duplicate")?;
    drop(inventory);
    handoff_files(output, file, count)
}
#[cfg(test)]
pub mod test_support {
    use super::*;
    pub fn handoff_files(
        output: File,
        inventory: File,
        count: usize,
    ) -> Result<FixedModelFds, &'static str> {
        super::handoff_files(output, inventory, count)
    }
}
