//! Fixed raw-model filesystem boundary; no caller selects paths or grants.
use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
const O_PATH: i32 = 0x200000;
const O_NOFOLLOW: i32 = 0x20000;
const O_DIRECTORY: i32 = 0x10000;
const O_CLOEXEC: i32 = 0x80000;
const PATH_FLAGS: i32 = O_PATH | O_NOFOLLOW | O_CLOEXEC;
const ALL_SEALS: i32 = 15;
extern "C" {
    fn openat(fd: i32, path: *const i8, flags: i32, ...) -> i32;
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    fn syscall(number: i64, ...) -> i64;
    fn geteuid() -> u32;
    fn getpid() -> i32;
    fn readlinkat(fd: i32, name: *const i8, buf: *mut u8, length: usize) -> isize;
    fn prctl(option: i32, ...) -> i32;
    fn getresuid(a: *mut u32, b: *mut u32, c: *mut u32) -> i32;
    fn getresgid(a: *mut u32, b: *mut u32, c: *mut u32) -> i32;
    fn getgroups(size: i32, groups: *mut u32) -> i32;
    fn flock(fd: i32, operation: i32) -> i32;
}
pub const PROVIDER_PATHS: [&str; 4] = [
    "/home/appuser/.aws/credentials",
    "/home/appuser/.aws/config",
    "/root/.aws/credentials",
    "/root/.aws/config",
];
pub const MAX_INVENTORY_BYTES: usize = 16384;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AncestorIdentity {
    pub component_index: u8,
    pub device: u64,
    pub inode: u64,
    pub uid: u32,
    pub gid: u32,
    pub mode: u32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderAbsence {
    pub provider_index: u8,
    pub missing_component_index: u8,
    pub ancestors: Vec<AncestorIdentity>,
}
pub struct SealedProviderInventory {
    file: File,
    records: [ProviderAbsence; 4],
    byte_count: usize,
}
impl SealedProviderInventory {
    pub fn as_file(&self) -> &File {
        &self.file
    }
    pub fn records(&self) -> &[ProviderAbsence; 4] {
        &self.records
    }
    pub fn byte_count(&self) -> usize {
        self.byte_count
    }
}
#[derive(Debug)]
pub struct FileBoundaryProof {
    pub profile: &'static str,
    pub abi: u32,
    pub handled_access_fs: u64,
    pub original_output_device: u64,
    pub original_output_inode: u64,
    pub process_id: u32,
    pub process_dumpability: u32,
    pub rule_identities: Vec<AppliedRuleIdentity>,
}
fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}
fn open_component(parent: &File, name: &CString, flags: i32) -> Result<File, i32> {
    let fd = unsafe { openat(parent.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        Err(errno())
    } else {
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}
fn identity(file: &File, component_index: u8) -> Result<AncestorIdentity, &'static str> {
    let m = file.metadata().map_err(|_| "provider_metadata")?;
    if !m.is_dir() {
        return Err("provider_ancestor_type");
    }
    Ok(AncestorIdentity {
        component_index,
        device: m.dev(),
        inode: m.ino(),
        uid: m.uid(),
        gid: m.gid(),
        mode: m.mode() & 0o7777,
    })
}
struct HeldAncestor {
    parent: File,
    named: File,
    component: CString,
    identity: AncestorIdentity,
}
struct Walk {
    record: ProviderAbsence,
    root: File,
    held: Vec<HeldAncestor>,
    missing_parent: File,
    missing_component: CString,
}
fn observe(root: &File, provider: usize) -> Result<Walk, &'static str> {
    let parts: Vec<_> = PROVIDER_PATHS[provider]
        .split('/')
        .filter(|v| !v.is_empty())
        .collect();
    let mut ancestors = vec![identity(root, 0)?];
    let mut held = Vec::new();
    let mut parent = root.try_clone().map_err(|_| "provider_fd")?;
    for (index, part) in parts.iter().enumerate() {
        let component = CString::new(*part).map_err(|_| "provider_component")?;
        let child = match open_component(&parent, &component, PATH_FLAGS) {
            Ok(file) => file,
            Err(2) => {
                return Ok(Walk {
                    record: ProviderAbsence {
                        provider_index: provider as u8,
                        missing_component_index: index as u8 + 1,
                        ancestors,
                    },
                    root: root.try_clone().map_err(|_| "provider_fd")?,
                    held,
                    missing_parent: parent,
                    missing_component: component,
                })
            }
            Err(_) => return Err("provider_observation_unknown"),
        };
        let meta = child.metadata().map_err(|_| "provider_metadata")?;
        if meta.file_type().is_symlink() {
            return Err("provider_symlink");
        }
        if index + 1 == parts.len() {
            return Err("provider_presence");
        }
        let node = identity(&child, index as u8 + 1)?;
        ancestors.push(node.clone());
        let next = child.try_clone().map_err(|_| "provider_fd")?;
        held.push(HeldAncestor {
            parent,
            named: child,
            component,
            identity: node,
        });
        parent = next;
    }
    Err("provider_observation_unknown")
}
fn recheck(walk: &Walk) -> Result<(), &'static str> {
    if identity(&walk.root, 0).map_err(|_| "provider_inventory_changed")?
        != walk.record.ancestors[0]
    {
        return Err("provider_inventory_changed");
    }
    for item in &walk.held {
        let named = open_component(&item.parent, &item.component, PATH_FLAGS)
            .map_err(|_| "provider_inventory_changed")?;
        if identity(&named, item.identity.component_index)
            .map_err(|_| "provider_inventory_changed")?
            != item.identity
            || identity(&item.named, item.identity.component_index)
                .map_err(|_| "provider_inventory_changed")?
                != item.identity
        {
            return Err("provider_inventory_changed");
        }
    }
    match open_component(&walk.missing_parent, &walk.missing_component, PATH_FLAGS) {
        Err(2) => Ok(()),
        _ => Err("provider_inventory_changed"),
    }
}
fn encoded(records: &[ProviderAbsence; 4]) -> Result<Vec<u8>, &'static str> {
    let rows:Vec<String>=records.iter().map(|record| {
        let nodes:Vec<String>=record.ancestors.iter().map(|a|format!(r#"{{"component_index":{},"device":{},"inode":{},"uid":{},"gid":{},"mode":{}}}"#,a.component_index,a.device,a.inode,a.uid,a.gid,a.mode)).collect();
        format!(r#"{{"provider_index":{},"missing_component_index":{},"ancestors":[{}]}}"#,record.provider_index,record.missing_component_index,nodes.join(","))
    }).collect();
    let bytes=format!(r#"{{"schema_version":1,"inventory_kind":"raw_model_provider_absence_before_drop_v1","observation":"earlier_original_component_absence","providers":[{}]}}"#,rows.join(",")).into_bytes();
    if bytes.is_empty() || bytes.len() > MAX_INVENTORY_BYTES {
        Err("inventory_size")
    } else {
        Ok(bytes)
    }
}
fn proc_fd_directory() -> Result<File, &'static str> {
    let rootfd = unsafe { openat(-100, c"/proc".as_ptr(), PATH_FLAGS | O_DIRECTORY) };
    if rootfd < 0 {
        return Err("inventory_proc_open");
    }
    let proc = unsafe { File::from_raw_fd(rootfd) };
    let pid = CString::new(unsafe { getpid() }.to_string()).map_err(|_| "inventory_pid")?;
    let process =
        open_component(&proc, &pid, PATH_FLAGS | O_DIRECTORY).map_err(|_| "inventory_proc_open")?;
    open_component(
        &process,
        &CString::new("fd").map_err(|_| "inventory_fd_name")?,
        PATH_FLAGS | O_DIRECTORY,
    )
    .map_err(|_| "inventory_proc_open")
}
fn seal_bytes(bytes: &[u8]) -> Result<File, &'static str> {
    if bytes.is_empty() || bytes.len() > MAX_INVENTORY_BYTES {
        return Err("inventory_size");
    }
    let fd = unsafe { syscall(319, c"amlas-provider-inventory".as_ptr(), 3u32) };
    if fd < 0 || fd > i32::MAX as i64 {
        return Err("inventory_memfd_create");
    }
    let mut writer = unsafe { File::from_raw_fd(fd as i32) };
    writer.write_all(bytes).map_err(|_| "inventory_write")?;
    if unsafe { fcntl(writer.as_raw_fd(), 1033, ALL_SEALS) } != 0
        || unsafe { fcntl(writer.as_raw_fd(), 1034) } != ALL_SEALS
    {
        return Err("inventory_seal");
    }
    let before = writer.metadata().map_err(|_| "inventory_metadata")?;
    if !before.is_file() || before.len() != bytes.len() as u64 {
        return Err("inventory_metadata");
    }
    let directory = proc_fd_directory()?;
    let name = CString::new(writer.as_raw_fd().to_string()).map_err(|_| "inventory_fd_name")?;
    // The only followed link is the OS-generated proc entry for our private sealed memfd.
    let mut reader =
        open_component(&directory, &name, O_CLOEXEC).map_err(|_| "inventory_readonly_open")?;
    let after = reader.metadata().map_err(|_| "inventory_metadata")?;
    if (before.dev(), before.ino(), before.len()) != (after.dev(), after.ino(), after.len())
        || unsafe { fcntl(reader.as_raw_fd(), 3) } & 3 != 0
        || unsafe { fcntl(reader.as_raw_fd(), 1034) } != ALL_SEALS
    {
        return Err("inventory_readonly_identity");
    }
    let mut actual = Vec::new();
    Read::by_ref(&mut reader)
        .take(MAX_INVENTORY_BYTES as u64 + 1)
        .read_to_end(&mut actual)
        .map_err(|_| "inventory_readback")?;
    if actual != bytes {
        return Err("inventory_readback");
    }
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|_| "inventory_readback")?;
    drop(writer);
    Ok(reader)
}
fn capture_at(root: &File, hook: impl FnOnce()) -> Result<SealedProviderInventory, &'static str> {
    let mut walks = Vec::with_capacity(4);
    for index in 0..4 {
        walks.push(observe(root, index)?);
    }
    hook();
    for walk in &walks {
        recheck(walk)?;
    }
    let records: [ProviderAbsence; 4] = walks
        .iter()
        .map(|w| w.record.clone())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "inventory_count")?;
    let bytes = encoded(&records)?;
    let file = seal_bytes(&bytes)?;
    for walk in &walks {
        recheck(walk)?;
    }
    Ok(SealedProviderInventory {
        file,
        records,
        byte_count: bytes.len(),
    })
}
/// Capture only four pre-drop producer observations; never read provider contents.
pub fn capture_provider_inventory() -> Result<SealedProviderInventory, &'static str> {
    if unsafe { geteuid() } != 0 {
        return Err("inventory_initial_identity");
    }
    let fd = unsafe { openat(-100, c"/".as_ptr(), PATH_FLAGS | O_DIRECTORY) };
    if fd < 0 {
        return Err("provider_root_open");
    }
    let root = unsafe { File::from_raw_fd(fd) };
    capture_at(&root, || {})
}
fn policy_mask(abi: i64) -> Result<u64, &'static str> {
    if abi == 6 {
        Ok(0xffff)
    } else {
        Err("landlock_abi_unsupported")
    }
}
/// Finite actual inode metadata, never a permission grant or attestation.
#[derive(Clone, Debug)]
pub struct AppliedRuleIdentity {
    pub rule_index: u8,
    pub device: u64,
    pub inode: u64,
    pub uid: u32,
    pub gid: u32,
    pub mode: u32,
    pub allowed_access_fs: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct RawIdentity {
    device: u64,
    inode: u64,
    uid: u32,
    gid: u32,
    mode: u32,
    rdev: u64,
}
fn raw_identity(file: &File) -> Result<RawIdentity, &'static str> {
    let m = file.metadata().map_err(|_| "runtime_metadata")?;
    Ok(RawIdentity {
        device: m.dev(),
        inode: m.ino(),
        uid: m.uid(),
        gid: m.gid(),
        mode: m.mode(),
        rdev: m.rdev(),
    })
}
struct CheckedComponent {
    parent: File,
    parent_identity: RawIdentity,
    named: File,
    component: CString,
    identity: RawIdentity,
}
impl CheckedComponent {
    fn recheck(&self) -> Result<(), &'static str> {
        if raw_identity(&self.parent)? != self.parent_identity {
            return Err("runtime_identity_changed");
        }
        let named = open_component(&self.parent, &self.component, PATH_FLAGS)
            .map_err(|_| "runtime_identity_changed")?;
        if raw_identity(&named)? != self.identity || raw_identity(&self.named)? != self.identity {
            return Err("runtime_identity_changed");
        }
        Ok(())
    }
}
struct PinnedPath {
    file: File,
    root: File,
    root_identity: RawIdentity,
    components: Vec<CheckedComponent>,
}
impl PinnedPath {
    fn recheck(&self) -> Result<(), &'static str> {
        if raw_identity(&self.root)? != self.root_identity {
            return Err("runtime_identity_changed");
        }
        for node in &self.components {
            node.recheck()?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
enum Kind {
    File,
    Directory,
    Null,
    ProcFile,
    ProcDirectory,
    Output,
}
struct Rule {
    path: &'static str,
    kind: Kind,
    access: u64,
}
const OUTPUT_ACCESS: u64 = 0x61be;
const RULES: &[Rule] = &[
    Rule {
        path: "/app",
        kind: Kind::Directory,
        access: 12,
    },
    Rule {
        path: "/usr/local/lib/python3.11",
        kind: Kind::Directory,
        access: 12,
    },
    Rule {
        path: "/usr/local/bin/python3.11",
        kind: Kind::File,
        access: 5,
    },
    Rule {
        path: "/usr/local/lib/libpython3.11.so.1.0",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
        kind: Kind::File,
        access: 5,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libbz2.so.1.0.4",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libc.so.6",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libcrypto.so.3",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libdl.so.2",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libffi.so.8.1.4",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libgcc_s.so.1",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/liblzma.so.5.8.1",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libm.so.6",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libpthread.so.0",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/librt.so.1",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libssl.so.3",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libz.so.1.3.1",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/usr/lib/x86_64-linux-gnu/libzstd.so.1.5.7",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/etc/ssl/certs/ca-certificates.crt",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/etc/resolv.conf",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/etc/hosts",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/etc/nsswitch.conf",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/etc/gai.conf",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/etc/ld.so.cache",
        kind: Kind::File,
        access: 4,
    },
    Rule {
        path: "/dev/null",
        kind: Kind::Null,
        access: 4,
    },
    Rule {
        path: "current_pid_status",
        kind: Kind::ProcFile,
        access: 4,
    },
    Rule {
        path: "current_pid_fd",
        kind: Kind::ProcDirectory,
        access: 8,
    },
    Rule {
        path: "/opt/ml/processing/output",
        kind: Kind::Output,
        access: OUTPUT_ACCESS,
    },
];
const ALIASES: [(&str, &str); 5] = [
    ("/lib", "usr/lib"),
    ("/lib64", "usr/lib64"),
    (
        "/usr/lib/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ),
    ("/usr/lib/ssl/certs", "/etc/ssl/certs"),
    (
        "/usr/lib64/ld-linux-x86-64.so.2",
        "../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    ),
];
fn metadata_policy(
    node: &RawIdentity,
    kind: Kind,
    last: bool,
    index: usize,
    owner: (u32, u32),
    dumpability: u32,
) -> Result<(), &'static str> {
    let typ = node.mode & 0o170000;
    if typ == 0o120000 {
        return Err("runtime_symlink");
    }
    let proc = matches!(kind, Kind::ProcFile | Kind::ProcDirectory) && index >= 2;
    let output = matches!(kind, Kind::Output) && index >= 3;
    let expected = if proc && last {
        if dumpability == 0 {
            (0, 0)
        } else if dumpability == 1 {
            (10001, 10001)
        } else {
            return Err("boundary_dumpability");
        }
    } else if proc || output {
        (10001, 10001)
    } else {
        owner
    };
    if (node.uid, node.gid) != expected {
        return Err("runtime_owner");
    }
    if matches!(kind, Kind::Null) && last {
        if typ != 0o020000 || node.rdev != 0x103 {
            return Err("runtime_dev_null");
        }
        return Ok(());
    }
    if node.mode & 0o022 != 0 {
        return Err("runtime_writable");
    }
    if matches!(kind, Kind::Output) && index >= 3 {
        let mode = if last { 0o700 } else { 0o755 };
        if node.mode & 0o7777 != mode {
            return Err("output_mode");
        }
    }
    let directory = !last || matches!(kind, Kind::Directory | Kind::ProcDirectory | Kind::Output);
    if typ != if directory { 0o040000 } else { 0o100000 } {
        return Err("runtime_type");
    }
    Ok(())
}
fn pin_path(
    root: &File,
    path: &str,
    kind: Kind,
    owner: (u32, u32),
    dumpability: u32,
) -> Result<PinnedPath, &'static str> {
    let rootmeta = raw_identity(root)?;
    metadata_policy(&rootmeta, Kind::Directory, true, 0, owner, dumpability)?;
    let parts: Vec<_> = path.split('/').filter(|v| !v.is_empty()).collect();
    if parts.is_empty() || parts.len() > 8 {
        return Err("runtime_path");
    }
    let mut parent = root.try_clone().map_err(|_| "runtime_fd")?;
    let mut held = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let name = CString::new(*part).map_err(|_| "runtime_path")?;
        let file = open_component(&parent, &name, PATH_FLAGS).map_err(|_| "runtime_open")?;
        let meta = raw_identity(&file)?;
        metadata_policy(&meta, kind, i + 1 == parts.len(), i + 1, owner, dumpability)?;
        let next = file.try_clone().map_err(|_| "runtime_fd")?;
        let parent_identity = raw_identity(&parent)?;
        held.push(CheckedComponent {
            parent,
            parent_identity,
            named: file,
            component: name,
            identity: meta,
        });
        parent = next;
    }
    Ok(PinnedPath {
        file: parent,
        root: root.try_clone().map_err(|_| "runtime_fd")?,
        root_identity: rootmeta,
        components: held,
    })
}
struct PinnedAlias {
    path: PinnedPath,
    link: CheckedComponent,
    expected: &'static str,
}
fn alias_target(link: &CheckedComponent) -> Result<Vec<u8>, &'static str> {
    let mut bytes = [0u8; 256];
    let len = unsafe {
        readlinkat(
            link.parent.as_raw_fd(),
            link.component.as_ptr(),
            bytes.as_mut_ptr(),
            bytes.len(),
        )
    };
    if len < 0 || len as usize >= bytes.len() {
        return Err("runtime_alias_target");
    }
    Ok(bytes[..len as usize].to_vec())
}
fn pin_alias(root: &File, index: usize, owner: (u32, u32)) -> Result<PinnedAlias, &'static str> {
    let (path, target) = *ALIASES.get(index).ok_or("runtime_alias_index")?;
    let (parent, name) = path.rsplit_once('/').ok_or("runtime_alias_path")?;
    let parent = if parent.is_empty() {
        let meta = raw_identity(root)?;
        metadata_policy(&meta, Kind::Directory, true, 0, owner, 0)?;
        PinnedPath {
            file: root.try_clone().map_err(|_| "runtime_fd")?,
            root: root.try_clone().map_err(|_| "runtime_fd")?,
            root_identity: meta,
            components: Vec::new(),
        }
    } else {
        pin_path(root, parent, Kind::Directory, owner, 0)?
    };
    let component = CString::new(name).map_err(|_| "runtime_alias_path")?;
    let named =
        open_component(&parent.file, &component, PATH_FLAGS).map_err(|_| "runtime_alias_open")?;
    let identity = raw_identity(&named)?;
    if identity.mode & 0o170000 != 0o120000 || (identity.uid, identity.gid) != owner {
        return Err("runtime_alias_owner_type");
    }
    let link = CheckedComponent {
        parent_identity: raw_identity(&parent.file)?,
        parent: parent.file.try_clone().map_err(|_| "runtime_fd")?,
        named,
        component,
        identity,
    };
    if alias_target(&link)? != target.as_bytes() {
        return Err("runtime_alias_target");
    }
    let result = PinnedAlias {
        path: parent,
        link,
        expected: target,
    };
    result.recheck()?;
    Ok(result)
}
impl PinnedAlias {
    fn recheck(&self) -> Result<(), &'static str> {
        self.path.recheck().map_err(|_| "runtime_alias_changed")?;
        self.link.recheck().map_err(|_| "runtime_alias_changed")?;
        if alias_target(&self.link).map_err(|_| "runtime_alias_changed")?
            != self.expected.as_bytes()
        {
            return Err("runtime_alias_changed");
        }
        self.link.recheck().map_err(|_| "runtime_alias_changed")?;
        Ok(())
    }
}
trait Landlock {
    fn abi(&mut self) -> i64;
    fn nnp(&mut self) -> i32;
    fn create(&mut self, mask: u64) -> Result<File, &'static str>;
    fn add(&mut self, ruleset: &File, node: &File, access: u64) -> Result<(), &'static str>;
    fn restrict(&mut self, ruleset: &File) -> Result<(), &'static str>;
}
struct SystemLandlock;
#[repr(C, packed)]
struct Beneath {
    allowed_access: u64,
    parent_fd: i32,
}
impl Landlock for SystemLandlock {
    fn abi(&mut self) -> i64 {
        unsafe { syscall(444, std::ptr::null::<u8>(), 0usize, 1usize) }
    }
    fn nnp(&mut self) -> i32 {
        unsafe { prctl(39, 0usize, 0usize, 0usize, 0usize) }
    }
    fn create(&mut self, mask: u64) -> Result<File, &'static str> {
        let fd = unsafe { syscall(444, &mask as *const u64, 8usize, 0usize) };
        if fd < 0 || fd > i32::MAX as i64 {
            Err("landlock_create")
        } else {
            Ok(unsafe { File::from_raw_fd(fd as i32) })
        }
    }
    fn add(&mut self, ruleset: &File, node: &File, access: u64) -> Result<(), &'static str> {
        let rule = Beneath {
            allowed_access: access,
            parent_fd: node.as_raw_fd(),
        };
        if unsafe {
            syscall(
                445,
                ruleset.as_raw_fd(),
                1usize,
                &rule as *const Beneath,
                0usize,
            )
        } != 0
        {
            Err("landlock_add")
        } else {
            Ok(())
        }
    }
    fn restrict(&mut self, ruleset: &File) -> Result<(), &'static str> {
        if unsafe { syscall(446, ruleset.as_raw_fd(), 0usize) } != 0 {
            Err("landlock_restrict")
        } else {
            Ok(())
        }
    }
}
fn enforce(kernel: &mut impl Landlock, nodes: &[(File, u64)]) -> Result<(u32, u64), &'static str> {
    let abi = kernel.abi();
    if abi < 0 {
        return Err("landlock_abi_query");
    }
    let mask = policy_mask(abi)?;
    if kernel.nnp() != 1 {
        return Err("landlock_nnp");
    }
    let ruleset = kernel.create(mask)?;
    for (node, access) in nodes {
        kernel.add(&ruleset, node, *access)?;
    }
    kernel.restrict(&ruleset)?;
    if kernel.nnp() != 1 {
        return Err("landlock_nnp");
    }
    Ok((abi as u32, mask))
}
struct RulePlan {
    paths: Vec<PinnedPath>,
    nodes: Vec<(File, u64)>,
    proofs: Vec<AppliedRuleIdentity>,
}
fn rule_authority(kind: Kind, named: &File, original: &File) -> Result<File, &'static str> {
    if matches!(kind, Kind::Output) {
        if raw_identity(named)? != raw_identity(original)? {
            return Err("output_identity");
        }
        original.try_clone().map_err(|_| "output_fd")
    } else {
        named.try_clone().map_err(|_| "runtime_fd")
    }
}
fn assemble_fixed_rules(
    mut resolve: impl FnMut(usize, &Rule) -> Result<(PinnedPath, File, RawIdentity), &'static str>,
) -> Result<RulePlan, &'static str> {
    let mut plan = RulePlan {
        paths: Vec::with_capacity(RULES.len()),
        nodes: Vec::with_capacity(RULES.len()),
        proofs: Vec::with_capacity(RULES.len()),
    };
    for (index, rule) in RULES.iter().enumerate() {
        let (path, authority, meta) = resolve(index, rule)?;
        plan.nodes.push((authority, rule.access));
        plan.proofs.push(AppliedRuleIdentity {
            rule_index: index as u8,
            device: meta.device,
            inode: meta.inode,
            uid: meta.uid,
            gid: meta.gid,
            mode: meta.mode & 0o7777,
            allowed_access_fs: rule.access,
        });
        plan.paths.push(path);
    }
    Ok(plan)
}
fn validated_dumpability(value: i32) -> Result<u32, &'static str> {
    match value {
        0 | 1 => Ok(value as u32),
        _ => Err("boundary_dumpability"),
    }
}
fn process_dumpability() -> Result<u32, &'static str> {
    validated_dumpability(unsafe { prctl(3, 0usize, 0usize, 0usize, 0usize) })
}
fn child_identity() -> Result<(), &'static str> {
    let (mut a, mut b, mut c) = (0, 0, 0);
    if unsafe { getresuid(&mut a, &mut b, &mut c) } != 0 || [a, b, c] != [10001; 3] {
        return Err("boundary_identity");
    }
    if unsafe { getresgid(&mut a, &mut b, &mut c) } != 0
        || [a, b, c] != [10001; 3]
        || unsafe { getgroups(0, std::ptr::null_mut()) } != 0
    {
        return Err("boundary_identity");
    }
    if unsafe { prctl(39, 0usize, 0usize, 0usize, 0usize) } != 1 {
        return Err("landlock_nnp");
    }
    Ok(())
}
/// Additive fixed ABI6 filesystem policy for the raw model profile only.
pub fn apply_model_file_boundary(output: &File) -> Result<FileBoundaryProof, &'static str> {
    child_identity()?;
    let dumpability = process_dumpability()?;
    if output.as_raw_fd() != 3 {
        return Err("output_fd");
    }
    let outputmeta = raw_identity(output)?;
    metadata_policy(&outputmeta, Kind::Output, true, 4, (0, 0), dumpability)?;
    if unsafe { flock(output.as_raw_fd(), 2 | 4) } != 0 {
        return Err("output_lease");
    }
    let rootfd = unsafe { openat(-100, c"/".as_ptr(), PATH_FLAGS | O_DIRECTORY) };
    if rootfd < 0 {
        return Err("runtime_root");
    }
    let root = unsafe { File::from_raw_fd(rootfd) };
    let pid = unsafe { getpid() };
    if pid <= 0 {
        return Err("boundary_pid");
    }
    let mut aliases = Vec::with_capacity(5);
    for index in 0..ALIASES.len() {
        aliases.push(pin_alias(&root, index, (0, 0))?);
    }
    let plan = assemble_fixed_rules(|_index, rule| {
        let path = match rule.kind {
            Kind::ProcFile => format!("/proc/{pid}/status"),
            Kind::ProcDirectory => format!("/proc/{pid}/fd"),
            _ => rule.path.to_owned(),
        };
        let pinned = pin_path(&root, &path, rule.kind, (0, 0), dumpability)?;
        let meta = raw_identity(&pinned.file)?;
        let authority = rule_authority(rule.kind, &pinned.file, output)?;
        Ok((pinned, authority, meta))
    })?;
    for alias in &aliases {
        alias.recheck()?;
    }
    for path in &plan.paths {
        path.recheck()?;
    }
    if process_dumpability()? != dumpability {
        return Err("boundary_dumpability_changed");
    }
    let (abi, handled_access_fs) = enforce(&mut SystemLandlock, &plan.nodes)?;
    if process_dumpability()? != dumpability {
        return Err("boundary_dumpability_changed");
    }
    for alias in &aliases {
        alias.recheck()?;
    }
    for path in &plan.paths {
        path.recheck()?;
    }
    if raw_identity(output)? != outputmeta {
        return Err("output_identity");
    }
    Ok(FileBoundaryProof {
        profile: "raw_model_file_boundary_v1",
        abi,
        handled_access_fs,
        original_output_device: outputmeta.device,
        original_output_inode: outputmeta.inode,
        process_id: pid as u32,
        process_dumpability: dumpability,
        rule_identities: plan.proofs,
    })
}
#[cfg(test)]
pub mod test_support {
    pub fn proc_case(
        dumpability: i32,
        rule: usize,
        uid: u32,
        gid: u32,
        mode: u32,
        component: usize,
        last: bool,
    ) -> Result<(), &'static str> {
        let dumpability = validated_dumpability(dumpability)?;
        let rule = RULES.get(rule).ok_or("runtime_rule_index")?;
        metadata_policy(
            &RawIdentity {
                device: 1,
                inode: 1,
                uid,
                gid,
                mode,
                rdev: 0,
            },
            rule.kind,
            last,
            component,
            (0, 0),
            dumpability,
        )
    }
    pub fn production_plan() -> Result<Vec<(&'static str, u64)>, &'static str> {
        let root = File::open(std::env::temp_dir()).map_err(|_| "fixture")?;
        let plan = assemble_fixed_rules(|_index, rule| {
            let named = root.try_clone().map_err(|_| "fixture")?;
            let meta = raw_identity(&named)?;
            let authority = rule_authority(rule.kind, &named, &root)?;
            Ok((
                PinnedPath {
                    file: named,
                    root: root.try_clone().map_err(|_| "fixture")?,
                    root_identity: meta.clone(),
                    components: Vec::new(),
                },
                authority,
                meta,
            ))
        })?;
        struct Exact {
            expected: Vec<(i32, u64)>,
            seen: Vec<u64>,
        }
        impl Landlock for Exact {
            fn abi(&mut self) -> i64 {
                6
            }
            fn nnp(&mut self) -> i32 {
                1
            }
            fn create(&mut self, mask: u64) -> Result<File, &'static str> {
                assert_eq!(mask, 0xffff);
                File::open(std::env::temp_dir()).map_err(|_| "fixture")
            }
            fn add(&mut self, _set: &File, node: &File, access: u64) -> Result<(), &'static str> {
                assert_eq!((node.as_raw_fd(), access), self.expected[self.seen.len()]);
                self.seen.push(access);
                Ok(())
            }
            fn restrict(&mut self, _set: &File) -> Result<(), &'static str> {
                assert_eq!(self.seen.len(), 28);
                Ok(())
            }
        }
        let mut kernel = Exact {
            expected: plan
                .nodes
                .iter()
                .map(|(f, a)| (f.as_raw_fd(), *a))
                .collect(),
            seen: Vec::new(),
        };
        enforce(&mut kernel, &plan.nodes)?;
        Ok(RULES
            .iter()
            .zip(kernel.seen)
            .map(|(r, a)| (r.path, a))
            .collect())
    }
    pub fn metadata_case(
        rule: usize,
        uid: u32,
        gid: u32,
        mode: u32,
        rdev: u64,
        component: usize,
        last: bool,
    ) -> Result<(), &'static str> {
        let rule = RULES.get(rule).ok_or("runtime_rule_index")?;
        metadata_policy(
            &RawIdentity {
                device: 1,
                inode: 1,
                uid,
                gid,
                mode,
                rdev,
            },
            rule.kind,
            last,
            component,
            (0, 0),
            0,
        )
    }
    pub fn output_authority(original: &File, named: &File) -> Result<File, &'static str> {
        rule_authority(Kind::Output, named, original)
    }
    use super::*;
    pub fn inventory_at(root: &File) -> Result<SealedProviderInventory, &'static str> {
        capture_at(root, || {})
    }
    pub fn inventory_at_after(
        root: &File,
        hook: impl FnOnce(),
    ) -> Result<SealedProviderInventory, &'static str> {
        capture_at(root, hook)
    }
    pub fn sealed_bytes(bytes: &[u8]) -> Result<File, &'static str> {
        seal_bytes(bytes)
    }
    pub fn policy_mask(abi: i64) -> Result<u64, &'static str> {
        super::policy_mask(abi)
    }
    pub fn errno_is_absence(errno: i32) -> Result<bool, &'static str> {
        if errno == 2 {
            Ok(true)
        } else {
            Err("provider_observation_unknown")
        }
    }
    pub fn output_path_rights() -> u64 {
        OUTPUT_ACCESS
    }
    pub fn alias_at(root: &File, index: usize) -> Result<(), &'static str> {
        alias_at_after(root, index, || {})
    }
    pub fn alias_at_after(
        root: &File,
        index: usize,
        hook: impl FnOnce(),
    ) -> Result<(), &'static str> {
        let meta = root.metadata().unwrap();
        let alias = pin_alias(root, index, (meta.uid(), meta.gid()))?;
        hook();
        alias.recheck()
    }
    pub fn policy_attempts(
        abi: i64,
        nnp: i32,
        create_failure: bool,
        add_failure: bool,
        restrict_failure: bool,
    ) -> Result<Vec<&'static str>, &'static str> {
        struct Fake {
            abi: i64,
            nnp: i32,
            create_failure: bool,
            add_failure: bool,
            restrict_failure: bool,
            calls: Vec<&'static str>,
        }
        impl Landlock for Fake {
            fn abi(&mut self) -> i64 {
                self.calls.push("abi");
                self.abi
            }
            fn nnp(&mut self) -> i32 {
                self.calls.push("nnp");
                self.nnp
            }
            fn create(&mut self, mask: u64) -> Result<File, &'static str> {
                self.calls.push("create");
                assert_eq!(mask, 0xffff);
                if self.create_failure {
                    Err("landlock_create")
                } else {
                    File::open(std::env::temp_dir()).map_err(|_| "fixture")
                }
            }
            fn add(
                &mut self,
                _ruleset: &File,
                _node: &File,
                _access: u64,
            ) -> Result<(), &'static str> {
                self.calls.push("add");
                if self.add_failure {
                    Err("landlock_add")
                } else {
                    Ok(())
                }
            }
            fn restrict(&mut self, _ruleset: &File) -> Result<(), &'static str> {
                self.calls.push("restrict");
                if self.restrict_failure {
                    Err("landlock_restrict")
                } else {
                    Ok(())
                }
            }
        }
        let mut fake = Fake {
            abi,
            nnp,
            create_failure,
            add_failure,
            restrict_failure,
            calls: Vec::new(),
        };
        let node = File::open(std::env::temp_dir()).map_err(|_| "fixture")?;
        enforce(&mut fake, &[(node, 4)])?;
        Ok(fake.calls)
    }
}
