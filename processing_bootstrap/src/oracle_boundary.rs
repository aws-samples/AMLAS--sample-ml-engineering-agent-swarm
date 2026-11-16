//! Fixed unprivileged official-oracle boundary; no caller-selectable authority.
use std::ffi::{CString, OsString};
use std::fs::{self, File};
use std::io::Read;
use std::mem::ManuallyDrop;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::process::CommandExt;
use std::process::Command;

pub const LAUNCHER: &str = "/usr/local/bin/amlas-oracle-init";
pub const PYTHON: &str = "/usr/local/bin/python3.11";
const ELF_INTERPRETER: &str = "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2";
pub const CHILD: &str = "/app/oracle/coding_oracle_child.py";
pub const ORACLE_SOURCE_PATHS: [&str; 3] = [
    CHILD,
    "/app/oracle/coding_oracle.py",
    "/app/oracle/coding_oracle_catalog.py",
];
const SOURCE_BYTES: [&[u8]; 3] = [
    include_bytes!("../../src/amlas/evaluation/coding_oracle_child.py"),
    include_bytes!("../../src/amlas/evaluation/coding_oracle.py"),
    include_bytes!("../../src/amlas/evaluation/coding_oracle_catalog.py"),
];
const FS_MASK: u64 = 0xffff;
const SCOPE_MASK: u64 = 3;
const READ_FILE: u64 = 4;
const READ_DIR: u64 = 8;
const EXECUTE: u64 = 1;
const PATH_FLAGS: i32 = 0x2a0000;
const DIRECTORY: i32 = 0x10000;
// Immutable reviewed syscall allowlist, never grown from an observed trace.
// No sockets, process/thread creation, peer access, namespaces, io_uring or BPF.
const ALLOWED_SYSCALLS: &[u32] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 24, 25, 28, 39, 59,
    60, 63, 72, 79, 89, 96, 97, 98, 99, 102, 104, 107, 108, 110, 111, 115, 118, 120, 131, 157, 158,
    186, 202, 204, 217, 218, 219, 228, 229, 230, 231, 257, 262, 263, 267, 270, 271, 273, 292, 293,
    318, 332, 334, 436, 439,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleState {
    pub uid: [u32; 4],
    pub gid: [u32; 4],
    pub groups: Vec<u32>,
    pub threads: u32,
    pub caps: [u64; 5],
    pub no_new_privs: u32,
    pub seccomp: u32,
}
pub fn validate_argv(arguments: &[OsString]) -> Result<(), &'static str> {
    if arguments.len() != 1 {
        Err("oracle_arguments")
    } else {
        Ok(())
    }
}
pub fn validate_state(state: &OracleState) -> Result<(), &'static str> {
    if state.uid != [10001; 4]
        || state.gid != [10001; 4]
        || state.groups != [10001]
        || state.threads != 1
        || state.caps[..4] != [0; 4]
        || state.no_new_privs != 1
        || ![0, 2].contains(&state.seccomp)
    {
        Err("oracle_identity")
    } else {
        Ok(())
    }
}
fn environment(values: impl IntoIterator<Item = (OsString, OsString)>) -> Result<(), &'static str> {
    let mut found = Vec::new();
    for (name, value) in values {
        if !["PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"]
            .iter()
            .any(|n| name == *n)
            || value != "1"
            || found.contains(&name)
        {
            return Err("oracle_environment");
        }
        found.push(name);
    }
    if found.len() != 2 {
        return Err("oracle_environment");
    }
    Ok(())
}
fn parse_state(raw: &str) -> Result<OracleState, &'static str> {
    if raw.len() > 16384 {
        return Err("oracle_state_bound");
    }
    let field = |name: &str| -> Result<&str, &'static str> {
        let key = format!("{name}:");
        let mut matches = raw.lines().filter_map(|l| l.strip_prefix(&key));
        let value = matches.next().ok_or("oracle_state_field")?;
        if matches.next().is_some() {
            return Err("oracle_state_duplicate");
        }
        Ok(value.trim())
    };
    let ids = |name: &str| -> Result<[u32; 4], &'static str> {
        let v: Vec<u32> = field(name)?
            .split_whitespace()
            .map(|x| x.parse().map_err(|_| "oracle_ids"))
            .collect::<Result<_, _>>()?;
        v.try_into().map_err(|_| "oracle_ids")
    };
    let number = |name: &str| {
        field(name)?
            .parse::<u32>()
            .map_err(|_| "oracle_state_number")
    };
    let mut caps = [0; 5];
    for (i, name) in ["CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd"]
        .iter()
        .enumerate()
    {
        let value = field(name)?;
        if value.len() != 16 {
            return Err("oracle_capabilities");
        }
        caps[i] = u64::from_str_radix(value, 16).map_err(|_| "oracle_capabilities")?;
    }
    Ok(OracleState {
        uid: ids("Uid")?,
        gid: ids("Gid")?,
        groups: field("Groups")?
            .split_whitespace()
            .map(|x| x.parse().map_err(|_| "oracle_groups"))
            .collect::<Result<_, _>>()?,
        threads: number("Threads")?,
        caps,
        no_new_privs: number("NoNewPrivs")?,
        seccomp: number("Seccomp")?,
    })
}
extern "C" {
    fn openat(fd: i32, path: *const i8, flags: i32, ...) -> i32;
    fn prctl(option: i32, ...) -> i32;
    fn syscall(number: i64, ...) -> i64;
    fn fcntl(fd: i32, command: i32, ...) -> i32;
    fn readlinkat(fd: i32, path: *const i8, buffer: *mut u8, size: usize) -> isize;
}
fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}
fn state() -> Result<OracleState, &'static str> {
    let mut file = File::open("/proc/self/status").map_err(|_| "oracle_state_open")?;
    let mut raw = String::new();
    file.by_ref()
        .take(16385)
        .read_to_string(&mut raw)
        .map_err(|_| "oracle_state_read")?;
    parse_state(&raw)
}
fn stdio() -> Result<(), &'static str> {
    for fd in 0..3 {
        let file = ManuallyDrop::new(unsafe { File::from_raw_fd(fd) });
        let info = file.metadata().map_err(|_| "oracle_stdio")?;
        let flags = unsafe { fcntl(fd, 3) };
        if flags < 0 {
            return Err("oracle_stdio");
        }
        if fd < 2 {
            let target =
                fs::read_link(format!("/proc/self/fd/{fd}")).map_err(|_| "oracle_stdio_pipe")?;
            let name = target.to_str().ok_or("oracle_stdio_pipe")?;
            if !info.file_type().is_fifo()
                || flags & 3 != fd
                || !name.starts_with("pipe:[")
                || !name.ends_with(']')
            {
                return Err("oracle_stdio_pipe");
            }
        } else if !info.file_type().is_char_device()
            || info.rdev() != 259
            || ![1, 2].contains(&(flags & 3))
        {
            return Err("oracle_stdio_null");
        }
    }
    Ok(())
}
struct Pinned {
    path: &'static str,
    file: File,
    identity: (u64, u64, u32, u32, u32),
    rights: u64,
}
fn identity(file: &File) -> Result<(u64, u64, u32, u32, u32), &'static str> {
    let i = file.metadata().map_err(|_| "oracle_inode")?;
    Ok((i.dev(), i.ino(), i.uid(), i.gid(), i.mode()))
}
fn open_fixed(path: &str, directory: bool) -> Result<File, &'static str> {
    if !path.starts_with('/') || path.split('/').any(|p| p == "." || p == "..") {
        return Err("oracle_fixed_path");
    }
    let rootfd = unsafe { openat(-100, c"/".as_ptr(), PATH_FLAGS | DIRECTORY) };
    if rootfd < 0 {
        return Err("oracle_root");
    }
    let mut parent = unsafe { File::from_raw_fd(rootfd) };
    let rootmeta = parent.metadata().map_err(|_| "oracle_root")?;
    if !rootmeta.is_dir()
        || rootmeta.uid() != 0
        || rootmeta.gid() != 0
        || rootmeta.mode() & 0o022 != 0
    {
        return Err("oracle_root_owner");
    }
    let parts: Vec<_> = path.split('/').filter(|p| !p.is_empty()).collect();
    for (i, part) in parts.iter().enumerate() {
        let leaf = i + 1 == parts.len();
        let isdir = !leaf || directory;
        let name = CString::new(*part).map_err(|_| "oracle_fixed_path")?;
        let fd = unsafe {
            openat(
                parent.as_raw_fd(),
                name.as_ptr(),
                PATH_FLAGS | if isdir { DIRECTORY } else { 0 },
            )
        };
        if fd < 0 {
            return Err("oracle_fixed_open");
        }
        let next = unsafe { File::from_raw_fd(fd) };
        let meta = next.metadata().map_err(|_| "oracle_fixed_stat")?;
        if meta.uid() != 0
            || meta.gid() != 0
            || meta.mode() & 0o022 != 0
            || (isdir && !meta.is_dir())
            || (!isdir && (!meta.is_file() || meta.nlink() != 1))
        {
            return Err("oracle_fixed_owner");
        }
        parent = next;
    }
    Ok(parent)
}
fn source_bytes(file: &File, expected: &[u8]) -> Result<(), &'static str> {
    let mut reader = File::open(format!("/proc/self/fd/{}", file.as_raw_fd()))
        .map_err(|_| "oracle_source_open")?;
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take((expected.len() + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "oracle_source_read")?;
    if bytes != expected {
        return Err("oracle_source_bytes");
    }
    Ok(())
}
fn runtime_access(path: &str) -> u64 {
    READ_FILE
        | if path == PYTHON || path == ELF_INTERPRETER {
            EXECUTE
        } else {
            0
        }
}
const LOADER_ALIASES: [(&str, &str); 3] = [
    ("/lib", "usr/lib"),
    ("/lib64", "usr/lib64"),
    (
        "/usr/lib64/ld-linux-x86-64.so.2",
        "../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    ),
];
fn loader_alias_target_matches(path: &str, target: &[u8]) -> bool {
    LOADER_ALIASES
        .iter()
        .any(|(name, expected)| *name == path && expected.as_bytes() == target)
}
fn fixed_plan() -> Result<Vec<Pinned>, &'static str> {
    let mut plan = Vec::new();
    for (index, path) in ORACLE_SOURCE_PATHS.iter().enumerate() {
        let file = open_fixed(path, false)?;
        source_bytes(&file, SOURCE_BYTES[index])?;
        plan.push(Pinned {
            path,
            file: file.try_clone().map_err(|_| "oracle_rule_fd")?,
            identity: identity(&file)?,
            rights: READ_FILE,
        });
    }
    for path in RUNTIME_FILES {
        let file = open_fixed(path, false)?;
        plan.push(Pinned {
            path,
            file: file.try_clone().map_err(|_| "oracle_rule_fd")?,
            identity: identity(&file)?,
            rights: runtime_access(path),
        });
    }
    for path in IMPORT_DIRECTORIES {
        let file = open_fixed(path, true)?;
        plan.push(Pinned {
            path,
            file: file.try_clone().map_err(|_| "oracle_rule_fd")?,
            identity: identity(&file)?,
            rights: READ_DIR,
        });
    }
    Ok(plan)
}
fn recheck(plan: &[Pinned]) -> Result<(), &'static str> {
    for node in plan {
        let named = open_fixed(node.path, node.rights == READ_DIR)?;
        if identity(&named)? != node.identity || identity(&node.file)? != node.identity {
            return Err("oracle_runtime_changed");
        }
    }
    for (i, path) in ORACLE_SOURCE_PATHS.iter().enumerate() {
        let file = open_fixed(path, false)?;
        source_bytes(&file, SOURCE_BYTES[i])?;
    }
    Ok(())
}
struct LoaderDirectory {
    path: &'static str,
    original: File,
    identity: (u64, u64, u32, u32, u32),
}
struct LoaderAlias {
    path: &'static str,
    parent_path: &'static str,
    name: CString,
    parent: File,
    link: File,
    parent_identity: (u64, u64, u32, u32, u32),
    link_identity: (u64, u64, u32, u32, u32),
}
struct LoaderBinding {
    directories: Vec<LoaderDirectory>,
    aliases: Vec<LoaderAlias>,
    canonical: File,
    canonical_identity: (u64, u64, u32, u32, u32),
}
fn link_bytes(link: &File) -> Result<Vec<u8>, &'static str> {
    let mut buffer = [0u8; 4096];
    let count = unsafe {
        readlinkat(
            link.as_raw_fd(),
            c"".as_ptr(),
            buffer.as_mut_ptr(),
            buffer.len(),
        )
    };
    if count < 0 || count as usize >= buffer.len() {
        return Err("oracle_loader_link_read");
    }
    Ok(buffer[..count as usize].to_vec())
}
fn open_alias(parent: &File, name: &CString) -> Result<File, &'static str> {
    let fd = unsafe { openat(parent.as_raw_fd(), name.as_ptr(), PATH_FLAGS) };
    if fd < 0 {
        return Err("oracle_loader_link_open");
    }
    let link = unsafe { File::from_raw_fd(fd) };
    let info = link.metadata().map_err(|_| "oracle_loader_link_stat")?;
    // Symlink0777 is normal; original root-owned nonwritable parent controls it.
    if !info.file_type().is_symlink() || info.uid() != 0 || info.gid() != 0 {
        return Err("oracle_loader_link_owner");
    }
    Ok(link)
}
fn pin_loader(plan: &[Pinned]) -> Result<LoaderBinding, &'static str> {
    let loader = plan
        .iter()
        .find(|entry| entry.path == ELF_INTERPRETER)
        .ok_or("oracle_loader_rule")?;
    if loader.rights != READ_FILE | EXECUTE {
        return Err("oracle_loader_rule");
    }
    let mut binding = LoaderBinding {
        directories: Vec::new(),
        aliases: Vec::new(),
        canonical: loader.file.try_clone().map_err(|_| "oracle_loader_fd")?,
        canonical_identity: loader.identity,
    };
    for path in [
        "/",
        "/usr",
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        let original = open_fixed(path, true)?;
        binding.directories.push(LoaderDirectory {
            path,
            identity: identity(&original)?,
            original,
        });
    }
    for (path, _) in LOADER_ALIASES {
        let parent_path = if path == "/lib" || path == "/lib64" {
            "/"
        } else {
            "/usr/lib64"
        };
        let parent = open_fixed(parent_path, true)?;
        let name = CString::new(path.rsplit('/').next().ok_or("oracle_loader_name")?)
            .map_err(|_| "oracle_loader_name")?;
        let link = open_alias(&parent, &name)?;
        if !loader_alias_target_matches(path, &link_bytes(&link)?) {
            return Err("oracle_loader_link_target");
        }
        binding.aliases.push(LoaderAlias {
            path,
            parent_path,
            name,
            parent_identity: identity(&parent)?,
            link_identity: identity(&link)?,
            parent,
            link,
        });
    }
    recheck_loader(&binding)?;
    Ok(binding)
}
fn recheck_loader(binding: &LoaderBinding) -> Result<(), &'static str> {
    for directory in &binding.directories {
        let named = open_fixed(directory.path, true)?;
        if identity(&directory.original)? != directory.identity
            || identity(&named)? != directory.identity
        {
            return Err("oracle_loader_directory_changed");
        }
    }
    for alias in &binding.aliases {
        let named_parent = open_fixed(alias.parent_path, true)?;
        let named_link = open_alias(&named_parent, &alias.name)?;
        if identity(&alias.parent)? != alias.parent_identity
            || identity(&named_parent)? != alias.parent_identity
            || identity(&alias.link)? != alias.link_identity
            || identity(&named_link)? != alias.link_identity
            || !loader_alias_target_matches(alias.path, &link_bytes(&alias.link)?)
            || !loader_alias_target_matches(alias.path, &link_bytes(&named_link)?)
        {
            return Err("oracle_loader_alias_changed");
        }
    }
    let canonical = open_fixed(ELF_INTERPRETER, false)?;
    if identity(&binding.canonical)? != binding.canonical_identity
        || identity(&canonical)? != binding.canonical_identity
    {
        return Err("oracle_loader_inode_changed");
    }
    // Follow only the already pinned, exact byte/parent/link chain; both names
    // must denote the same captured inode receiving the sole loader RX grant.
    for path in [
        "/lib64/ld-linux-x86-64.so.2",
        "/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    ] {
        let resolved = File::open(path).map_err(|_| "oracle_loader_resolve")?;
        if identity(&resolved)? != binding.canonical_identity {
            return Err("oracle_loader_foreign_inode");
        }
    }
    Ok(())
}
#[repr(C)]
struct LandlockAttr {
    fs: u64,
    net: u64,
    scoped: u64,
}
#[repr(C, packed)]
struct Beneath {
    access: u64,
    fd: i32,
}
fn landlock(plan: &[Pinned]) -> Result<(), &'static str> {
    let abi = unsafe { syscall(444, std::ptr::null::<u8>(), 0usize, 1u32) };
    if abi < 6 {
        return Err("oracle_landlock_abi");
    }
    let attr = LandlockAttr {
        fs: FS_MASK,
        net: 3,
        scoped: SCOPE_MASK,
    };
    let fd = unsafe {
        syscall(
            444,
            &attr as *const LandlockAttr,
            std::mem::size_of::<LandlockAttr>(),
            0u32,
        )
    };
    if fd < 0 || fd > i32::MAX as i64 {
        return Err("oracle_landlock_create");
    }
    let set = unsafe { File::from_raw_fd(fd as i32) };
    for node in plan {
        let rule = Beneath {
            access: node.rights,
            fd: node.file.as_raw_fd(),
        };
        if unsafe { syscall(445, set.as_raw_fd(), 1u32, &rule as *const Beneath, 0u32) } != 0 {
            return Err("oracle_landlock_rule");
        }
    }
    if unsafe { syscall(446, set.as_raw_fd(), 0u32) } != 0 {
        return Err("oracle_landlock_restrict");
    }
    Ok(())
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Filter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}
#[repr(C)]
struct Program {
    length: u16,
    filters: *const Filter,
}
fn instruction(code: u16, jt: u8, jf: u8, k: u32) -> Filter {
    Filter { code, jt, jf, k }
}
fn filter() -> Vec<Filter> {
    let mut code = vec![
        instruction(0x20, 0, 0, 4),
        instruction(0x15, 1, 0, 0xc000003e),
        instruction(6, 0, 0, 0x80000000),
        instruction(0x20, 0, 0, 0),
        instruction(0x45, 0, 1, 0x40000000),
        instruction(6, 0, 0, 0x80000000),
    ];
    for nr in ALLOWED_SYSCALLS {
        code.push(instruction(0x15, 0, 1, *nr));
        code.push(instruction(6, 0, 0, 0x7fff0000));
    }
    // seccomp_data.args begins at byte16. AMD64 little-endian: inspect both
    // 32-bit halves of pid arg0 and new_limit arg2; never truncate pointers/PIDs.
    code.extend([
        instruction(0x15, 0, 9, 302),
        instruction(0x20, 0, 0, 16),
        instruction(0x15, 0, 7, 0),
        instruction(0x20, 0, 0, 20),
        instruction(0x15, 0, 5, 0),
        instruction(0x20, 0, 0, 32),
        instruction(0x15, 0, 3, 0),
        instruction(0x20, 0, 0, 36),
        instruction(0x15, 0, 1, 0),
        instruction(6, 0, 0, 0x7fff0000),
    ]);
    code.push(instruction(6, 0, 0, 0x00050001));
    code
}
fn seccomp() -> Result<(), &'static str> {
    let code = filter();
    let program = Program {
        length: u16::try_from(code.len()).map_err(|_| "oracle_filter_bound")?,
        filters: code.as_ptr(),
    };
    if unsafe { prctl(22, 2usize, &program as *const Program, 0usize, 0usize) } != 0
        || unsafe { prctl(21, 0usize, 0usize, 0usize, 0usize) } != 2
    {
        return Err("oracle_seccomp");
    }
    Ok(())
}
pub fn run() -> Result<(), &'static str> {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err("oracle_platform");
    }
    validate_argv(&std::env::args_os().collect::<Vec<_>>())?;
    environment(std::env::vars_os())?;
    let mut observed = state()?;
    if observed.no_new_privs > 1 {
        return Err("oracle_no_new_privs");
    }
    if unsafe { prctl(38, 1usize, 0usize, 0usize, 0usize) } != 0
        || unsafe { prctl(39, 0usize, 0usize, 0usize, 0usize) } != 1
    {
        return Err("oracle_no_new_privs");
    }
    observed.no_new_privs = 1;
    validate_state(&observed)?;
    stdio()?;
    let plan = fixed_plan()?;
    recheck(&plan)?;
    let loader = pin_loader(&plan)?;
    landlock(&plan)?;
    recheck_loader(&loader)?;
    seccomp()?;
    // All preparation/source/policy descriptors drop before final mandatory cleanup.
    drop(loader);
    drop(plan);
    if unsafe { syscall(436, 3u32, u32::MAX, 0u32) } != 0 {
        return Err("oracle_close_range");
    }
    // Kernel close_range success with one thread closes all >2; no later file open.
    if unsafe { fcntl(3, 1) } != -1 || errno() != 9 {
        return Err("oracle_fd_closure");
    }
    eprintln!("ORACLE-INIT-BOUNDING {:016x}", observed.caps[4]);
    let error = Command::new(PYTHON)
        .args(["-I", "-S", "-B", "-u", CHILD])
        .env_clear()
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("LC_ALL", "C.UTF-8")
        .exec();
    let _ = error;
    Err("oracle_exec")
}

// Reviewed leaf-only snapshot; inventory overapproximation is not native admission.
pub const RUNTIME_FILES: &[&str] = &[
    "/usr/lib/locale/C.utf8/LC_CTYPE",
    "/usr/lib/x86_64-linux-gnu/gconv/gconv-modules.cache",
    "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    "/usr/lib/x86_64-linux-gnu/libc.so.6",
    "/usr/lib/x86_64-linux-gnu/libcrypto.so.3",
    "/usr/lib/x86_64-linux-gnu/libm.so.6",
    "/usr/lib/x86_64-linux-gnu/libz.so.1.3.1",
    "/usr/lib/x86_64-linux-gnu/libzstd.so.1.5.7",
    "/usr/local/bin/python3.11",
    "/usr/local/lib/libpython3.11.so.1.0",
    "/usr/local/lib/python3.11/__future__.py",
    "/usr/local/lib/python3.11/_collections_abc.py",
    "/usr/local/lib/python3.11/_weakrefset.py",
    "/usr/local/lib/python3.11/abc.py",
    "/usr/local/lib/python3.11/bisect.py",
    "/usr/local/lib/python3.11/codecs.py",
    "/usr/local/lib/python3.11/collections/__init__.py",
    "/usr/local/lib/python3.11/collections/abc.py",
    "/usr/local/lib/python3.11/contextlib.py",
    "/usr/local/lib/python3.11/copy.py",
    "/usr/local/lib/python3.11/copyreg.py",
    "/usr/local/lib/python3.11/encodings/__init__.py",
    "/usr/local/lib/python3.11/encodings/aliases.py",
    "/usr/local/lib/python3.11/encodings/utf_8.py",
    "/usr/local/lib/python3.11/enum.py",
    "/usr/local/lib/python3.11/fnmatch.py",
    "/usr/local/lib/python3.11/functools.py",
    "/usr/local/lib/python3.11/genericpath.py",
    "/usr/local/lib/python3.11/hashlib.py",
    "/usr/local/lib/python3.11/importlib/__init__.py",
    "/usr/local/lib/python3.11/importlib/_abc.py",
    "/usr/local/lib/python3.11/importlib/_bootstrap.py",
    "/usr/local/lib/python3.11/importlib/_bootstrap_external.py",
    "/usr/local/lib/python3.11/importlib/util.py",
    "/usr/local/lib/python3.11/io.py",
    "/usr/local/lib/python3.11/ipaddress.py",
    "/usr/local/lib/python3.11/json/__init__.py",
    "/usr/local/lib/python3.11/json/decoder.py",
    "/usr/local/lib/python3.11/json/encoder.py",
    "/usr/local/lib/python3.11/json/scanner.py",
    "/usr/local/lib/python3.11/keyword.py",
    "/usr/local/lib/python3.11/lib-dynload/_bisect.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/_blake2.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/_hashlib.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/_json.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/_posixsubprocess.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/_random.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/_sha512.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/fcntl.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/math.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/lib-dynload/select.cpython-311-x86_64-linux-gnu.so",
    "/usr/local/lib/python3.11/locale.py",
    "/usr/local/lib/python3.11/ntpath.py",
    "/usr/local/lib/python3.11/operator.py",
    "/usr/local/lib/python3.11/os.py",
    "/usr/local/lib/python3.11/pathlib.py",
    "/usr/local/lib/python3.11/platform.py",
    "/usr/local/lib/python3.11/posixpath.py",
    "/usr/local/lib/python3.11/random.py",
    "/usr/local/lib/python3.11/re/__init__.py",
    "/usr/local/lib/python3.11/re/_casefix.py",
    "/usr/local/lib/python3.11/re/_compiler.py",
    "/usr/local/lib/python3.11/re/_constants.py",
    "/usr/local/lib/python3.11/re/_parser.py",
    "/usr/local/lib/python3.11/reprlib.py",
    "/usr/local/lib/python3.11/selectors.py",
    "/usr/local/lib/python3.11/signal.py",
    "/usr/local/lib/python3.11/stat.py",
    "/usr/local/lib/python3.11/string.py",
    "/usr/local/lib/python3.11/subprocess.py",
    "/usr/local/lib/python3.11/threading.py",
    "/usr/local/lib/python3.11/types.py",
    "/usr/local/lib/python3.11/typing.py",
    "/usr/local/lib/python3.11/urllib/__init__.py",
    "/usr/local/lib/python3.11/urllib/parse.py",
    "/usr/local/lib/python3.11/warnings.py",
    "/usr/local/lib/python3.11/weakref.py",
    "/usr/local/lib/python3.11/zipimport.py",
];
// READ_DIR enables FileFinder discovery and exposes names (including descendants),
// never READ_FILE authority for unknown siblings. No cache or writable grant.
pub const IMPORT_DIRECTORIES: &[&str] = &[
    "/app/oracle",
    "/usr/local/lib/python3.11",
    "/usr/local/lib/python3.11/collections",
    "/usr/local/lib/python3.11/encodings",
    "/usr/local/lib/python3.11/importlib",
    "/usr/local/lib/python3.11/json",
    "/usr/local/lib/python3.11/lib-dynload",
    "/usr/local/lib/python3.11/re",
    "/usr/local/lib/python3.11/urllib",
];

#[cfg(test)]
mod tests {
    use super::*;
    fn valid_state() -> OracleState {
        OracleState {
            uid: [10001; 4],
            gid: [10001; 4],
            groups: vec![10001],
            threads: 1,
            caps: [0, 0, 0, 0, 0x200404fb],
            no_new_privs: 1,
            seccomp: 2,
        }
    }
    #[test]
    fn fixed_no_argument_invocation_is_allowed_without_selector() {
        assert_eq!(validate_argv(&[OsString::from(LAUNCHER)]), Ok(()));
        assert!(validate_argv(&[]).is_err());
        assert!(validate_argv(&[OsString::from(LAUNCHER), OsString::from("neutral")]).is_err());
    }
    #[test]
    fn fixed_unprivileged_state_records_bounding_without_grant() {
        assert_eq!(validate_state(&valid_state()), Ok(()));
        let mut zero = valid_state();
        zero.caps[4] = 0;
        assert_eq!(validate_state(&zero), Ok(()));
    }
    #[test]
    fn identity_groups_active_caps_threads_and_final_nnp_are_closed() {
        for field in 0..4 {
            let mut state = valid_state();
            state.uid[field] = 0;
            assert!(validate_state(&state).is_err());
            let mut state = valid_state();
            state.gid[field] = 0;
            assert!(validate_state(&state).is_err());
            let mut state = valid_state();
            state.caps[field] = 1;
            assert!(validate_state(&state).is_err());
        }
        for groups in [vec![], vec![10001, 10002], vec![0]] {
            let mut state = valid_state();
            state.groups = groups;
            assert!(validate_state(&state).is_err());
        }
        for threads in [0, 2] {
            let mut state = valid_state();
            state.threads = threads;
            assert!(validate_state(&state).is_err());
        }
        for nnp in [0, 2] {
            let mut state = valid_state();
            state.no_new_privs = nnp;
            assert!(validate_state(&state).is_err());
        }
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    fn evaluate(nr: u32, arch: u32) -> u32 {
        evaluate_arguments(nr, arch, [0; 6])
    }
    fn evaluate_arguments(nr: u32, arch: u32, arguments: [u64; 6]) -> u32 {
        let code = filter();
        let mut pc = 0usize;
        let mut accumulator = 0u32;
        for _ in 0..code.len() + 1 {
            let instruction = code[pc];
            match instruction.code {
                0x20 => {
                    accumulator = match instruction.k {
                        0 => nr,
                        4 => arch,
                        offset if (16..64).contains(&offset) && offset % 4 == 0 => {
                            let index = ((offset - 16) / 8) as usize;
                            if (offset - 16) % 8 == 0 {
                                arguments[index] as u32
                            } else {
                                (arguments[index] >> 32) as u32
                            }
                        }
                        _ => panic!("unexpected seccomp_data mock offset"),
                    };
                    pc += 1;
                }
                0x15 => {
                    pc += 1 + if accumulator == instruction.k {
                        instruction.jt as usize
                    } else {
                        instruction.jf as usize
                    };
                }
                0x45 => {
                    pc += 1 + if accumulator & instruction.k != 0 {
                        instruction.jt as usize
                    } else {
                        instruction.jf as usize
                    };
                }
                6 => return instruction.k,
                _ => panic!("unsupported mock instruction"),
            }
        }
        panic!("mock filter did not terminate")
    }
    #[test]
    fn immutable_filter_denies_alternate_arch_x32_unknown_and_dangerous_numbers() {
        assert_eq!(evaluate(0, 0xc000003e), 0x7fff0000);
        assert_eq!(evaluate(0, 0x40000003), 0x80000000);
        assert_eq!(evaluate(0x40000000, 0xc000003e), 0x80000000);
        for nr in [
            41, 42, 53, 56, 57, 58, 62, 101, 105, 106, 116, 126, 165, 272, 308, 310, 311, 321, 323,
            322, 425, 426, 427, 434, 435, 438, 999,
        ] {
            assert_eq!(
                evaluate(nr, 0xc000003e),
                0x00050001,
                "mock syscall number {nr}"
            );
        }
    }
    #[test]
    fn reviewed_getdents64_is_allowed_without_legacy_directory_or_unknown_calls() {
        assert_eq!(evaluate(217, 0xc000003e), 0x7fff0000);
        assert_eq!(evaluate(78, 0xc000003e), 0x00050001);
        assert_eq!(evaluate(999, 0xc000003e), 0x00050001);
    }
    #[test]
    fn prlimit64_only_reads_self_with_full_width_null_new_limit() {
        let mut arguments = [0, 7, 0, 1, 0, 0];
        assert_eq!(evaluate_arguments(302, 0xc000003e, arguments), 0x7fff0000);
        for pid in [1, 1u64 << 32, u64::MAX] {
            arguments[0] = pid;
            assert_eq!(evaluate_arguments(302, 0xc000003e, arguments), 0x00050001);
        }
        arguments[0] = 0;
        for new_limit in [1, 1u64 << 32, u64::MAX] {
            arguments[2] = new_limit;
            assert_eq!(evaluate_arguments(302, 0xc000003e, arguments), 0x00050001);
        }
    }
    #[test]
    fn exact_loader_receives_execute_and_only_known_alias_bytes_are_allowed() {
        assert_eq!(runtime_access(ELF_INTERPRETER), READ_FILE | EXECUTE);
        assert_eq!(runtime_access(PYTHON), READ_FILE | EXECUTE);
        assert_eq!(
            runtime_access("/usr/local/lib/python3.11/json/decoder.py"),
            READ_FILE
        );
        assert!(loader_alias_target_matches("/lib", b"usr/lib"));
        assert!(loader_alias_target_matches("/lib64", b"usr/lib64"));
        assert!(loader_alias_target_matches(
            "/usr/lib64/ld-linux-x86-64.so.2",
            b"../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
        ));
        assert!(!loader_alias_target_matches("/lib64", b"neutral"));
        assert!(!loader_alias_target_matches("/neutral", b"usr/lib64"));
    }
    #[test]
    fn original_official_import_dependencies_are_read_only_fixed_leaves() {
        for path in [
            "/usr/local/lib/python3.11/typing.py",
            "/usr/local/lib/python3.11/copy.py",
            "/usr/local/lib/python3.11/weakref.py",
            "/usr/local/lib/python3.11/string.py",
        ] {
            assert!(
                RUNTIME_FILES.contains(&path),
                "missing reviewed import leaf {path}"
            );
            assert_eq!(runtime_access(path), READ_FILE);
            assert!(!IMPORT_DIRECTORIES.contains(&path));
        }
        assert!(!RUNTIME_FILES.contains(
            &"/usr/local/lib/python3.11/lib-dynload/_typing.cpython-311-x86_64-linux-gnu.so"
        ));
    }
    #[test]
    fn fixed_runtime_rules_cannot_grant_home_provider_proc_input_output_or_caches() {
        for path in RUNTIME_FILES {
            assert!(path.starts_with("/usr/"));
            assert!(!path.contains("__pycache__") && !path.contains("site-packages"));
            assert!(!["/usr", "/usr/local", "/usr/local/lib/python3.11"].contains(path));
        }
        assert_eq!(ORACLE_SOURCE_PATHS.len(), 3);
        assert!(ORACLE_SOURCE_PATHS
            .iter()
            .all(|path| path.starts_with("/app/oracle/")));
        assert_eq!(FS_MASK, 0xffff);
        assert_eq!(SCOPE_MASK, 3);
        assert_eq!(
            SOURCE_BYTES[0],
            include_bytes!("../../src/amlas/evaluation/coding_oracle_child.py")
        );
    }
    #[test]
    fn environment_has_no_provider_loader_policy_or_path_selector() {
        assert!(environment([
            (OsString::from("PYTHONUNBUFFERED"), OsString::from("1")),
            (
                OsString::from("PYTHONDONTWRITEBYTECODE"),
                OsString::from("1")
            )
        ])
        .is_ok());
        assert!(environment([(OsString::from("NEUTRAL"), OsString::from("1"))]).is_err());
    }
}
