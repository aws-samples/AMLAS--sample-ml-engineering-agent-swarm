
// Appended to an independently hash-verified unchanged policy prefix in AWS scratch.
// A normal product build never contains this suffix. rustc --cfg test (not --test)
// provides a single-thread standalone entry point and private parent-function access.
#[cfg(test)]
mod oracle_native_fixture {
    use super::*;
    use std::os::unix::io::IntoRawFd;

    const CANARY: &str = "/oracle-fixture/canary.txt";
    const UNKNOWN: &str = "/app/oracle/oracle_native_unknown.txt";
    const CONTENT: &[u8] = b"oracle-native-canary-v1\n";
    const MARKER: u64 = 0x6162_6364_6566_6768;
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Inet { family: u16, port: u16, address: u32, zero: [u8; 8] }
    #[repr(C)]
    struct Unix { family: u16, path: [u8; 108] }
    #[repr(C)]
    struct Limits { soft: u64, hard: u64 }
    #[repr(C)]
    struct IoVec { base: *mut u8, length: usize }
    extern "C" {
        fn fork() -> i32;
        fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        fn close(fd: i32) -> i32;
        fn dup2(old: i32, new: i32) -> i32;
        fn pipe2(fds: *mut i32, flags: i32) -> i32;
        fn alarm(seconds: u32) -> u32;
        fn socket(domain: i32, kind: i32, protocol: i32) -> i32;
        fn bind(fd: i32, address: *const u8, size: u32) -> i32;
        fn getsockname(fd: i32, address: *mut u8, size: *mut u32) -> i32;
        fn listen(fd: i32, backlog: i32) -> i32;
        fn connect(fd: i32, address: *const u8, size: u32) -> i32;
        fn sendto(fd: i32, bytes: *const u8, size: usize, flags: i32, address: *const u8, length: u32) -> isize;
        fn kill(pid: i32, signal: i32) -> i32;
    }
    fn mode() -> &'static str {
        if cfg!(oracle_fixture_complete) { "complete" }
        else if cfg!(oracle_fixture_landlock) { "landlock" }
        else { "baseline" }
    }
    fn check(name: &str, valid: bool, rc: i64, error: i32) -> Result<(), &'static str> {
        println!("CONTROL {} {name} {} rc={rc} errno={error}", mode(), if valid { "PASS" } else { "FAIL" });
        if valid { Ok(()) } else { Err("fixture_control") }
    }
    fn outcome(name: &str, rc: i64, denied: bool) -> Result<(), &'static str> {
        let error = if rc < 0 { errno() } else { 0 };
        let seccomp_denial = mode() == "complete" && !["canary_write_open", "home_create"].contains(&name);
        check(name, if denied { rc == -1 && if seccomp_denial { error == 1 } else { [1, 13].contains(&error) } } else { rc >= 0 }, rc, error)
    }
    fn inet_socket(kind: i32) -> Result<(i32, Inet), &'static str> {
        let fd = unsafe { socket(2, kind, 0) };
        if fd < 0 { return Err("fixture_socket"); }
        let mut address = Inet { family: 2, port: 0, address: u32::from_ne_bytes([127, 0, 0, 1]), zero: [0; 8] };
        if unsafe { bind(fd, &address as *const Inet as *const u8, 16) } != 0 { return Err("fixture_bind"); }
        let mut size = 16;
        if unsafe { getsockname(fd, &mut address as *mut Inet as *mut u8, &mut size) } != 0 || size != 16 { return Err("fixture_address"); }
        if kind == 1 && unsafe { listen(fd, 4) } != 0 { return Err("fixture_listen"); }
        Ok((fd, address))
    }
    fn unix_address() -> Unix {
        let mut address = Unix { family: 1, path: [0; 108] };
        address.path[1..22].copy_from_slice(b"oracle-native-control");
        address
    }
    fn preparation() -> Result<(Vec<Pinned>, LoaderBinding), &'static str> {
        validate_argv(&std::env::args_os().collect::<Vec<_>>())?;
        environment(std::env::vars_os())?;
        let mut observed = state()?;
        if observed.no_new_privs > 1 { return Err("oracle_no_new_privs"); }
        if unsafe { prctl(38, 1usize, 0usize, 0usize, 0usize) } != 0
            || unsafe { prctl(39, 0usize, 0usize, 0usize, 0usize) } != 1 { return Err("oracle_no_new_privs"); }
        observed.no_new_privs = 1;
        validate_state(&observed)?;
        stdio()?;
        let plan = fixed_plan()?;
        recheck(&plan)?;
        let loader = pin_loader(&plan)?;
        Ok((plan, loader))
    }
    fn read_control(path: &str, expected: Option<&[u8]>, denied: bool, name: &str) -> Result<(), &'static str> {
        let result = File::open(path).and_then(|mut file| {
            let mut bytes = Vec::new();
            file.by_ref().take(1_048_576).read_to_end(&mut bytes)?;
            Ok(bytes)
        });
        match result {
            Ok(bytes) => check(name, !denied && expected.map_or(!bytes.is_empty(), |value| value == bytes), bytes.len() as i64, 0),
            Err(error) => check(name, denied && [Some(1), Some(13)].contains(&error.raw_os_error()), -1, error.raw_os_error().unwrap_or(-1)),
        }
    }
    fn subject(peer: i32, marker: *mut u8, tcp: Inet, udp: Inet, inherited: &[i32]) -> Result<(), &'static str> {
        unsafe { alarm(15); }
        for fd in inherited { unsafe { close(*fd); } }
        let canary = File::open(CANARY).map_err(|_| "fixture_canary")?;
        let low = unsafe { fcntl(canary.as_raw_fd(), 1030, 20) };
        let high = unsafe { fcntl(canary.as_raw_fd(), 1030, 4096) };
        if low < 20 || high < 4096 { return Err("fixture_seed_fds"); }
        drop(canary);
        // These client FDs distinguish Landlock-only connection scope from seccomp.
        let tcp_client = unsafe { socket(2, 1, 0) };
        let unix_client = unsafe { socket(1, 1, 0) };
        if tcp_client < 0 || unix_client < 0 { return Err("fixture_clients"); }
        let (plan, loader) = preparation()?;
        let confined = mode() != "baseline";
        let complete = mode() == "complete";
        if confined {
            landlock(&plan)?;
            recheck_loader(&loader)?;
        }
        if complete { seccomp()?; }
        drop(loader);
        drop(plan);
        if complete {
            if unsafe { syscall(436, 3u32, u32::MAX, 0u32) } != 0 { return Err("oracle_close_range"); }
            if unsafe { fcntl(3, 1) } != -1 || errno() != 9 { return Err("oracle_fd_closure"); }
        }
        // Always establish actual granted operations before testing denied operations.
        for (index, path) in ORACLE_SOURCE_PATHS.iter().enumerate() {
            read_control(path, Some(SOURCE_BYTES[index]), false, &format!("source_read_{index}"))?;
        }
        read_control("/usr/local/lib/python3.11/typing.py", None, false, "runtime_read")?;
        let directory = File::open("/app/oracle").map_err(|_| "fixture_directory")?;
        let mut listing = [0u8; 4096];
        let count = unsafe { syscall(217, directory.as_raw_fd(), listing.as_mut_ptr(), listing.len()) };
        check("getdents64", count > 0, count, if count < 0 { errno() } else { 0 })?;
        drop(directory);
        check("unknown_sibling_metadata_exposed", fs::metadata(UNKNOWN).is_ok(), 0, 0)?;
        let mut limits = Limits { soft: 0, hard: 0 };
        let self_query = unsafe { syscall(302, 0usize, 7usize, std::ptr::null::<Limits>(), &mut limits as *mut Limits) };
        outcome("prlimit_self_read", self_query, false)?;
        println!("CONTROL {} stdout PASS rc=1 errno=0", mode());
        for (path, name) in [
            (CANARY, "ungranted_canary_read"),
            ("/home/appuser/oracle-native-canary.txt", "home_read"),
            ("/opt/ml/input/oracle-native-canary.txt", "input_read"),
            (UNKNOWN, "unknown_sibling_content"),
            ("/proc/self/status", "proc_self_read"),
        ] {
            read_control(path, if path.starts_with("/proc/") { None } else { Some(CONTENT) }, confined, name)?;
        }
        read_control(&format!("/proc/{peer}/status"), None, confined, "proc_peer_read")?;
        let write_path = CString::new(CANARY).unwrap();
        let writable = unsafe { openat(-100, write_path.as_ptr(), 1) };
        outcome("canary_write_open", writable as i64, confined)?;
        if writable >= 0 { unsafe { close(writable); } }
        let created = unsafe { openat(-100, c"/home/appuser/oracle-native-created.txt".as_ptr(), 1 | 64 | 128, 0o600u32) };
        outcome("home_create", created as i64, confined)?;
        if created >= 0 { unsafe { close(created); } }
        for (fd, name) in [(low, "low_fd_closed"), (high, "high_fd_closed")] {
            let rc = unsafe { fcntl(fd, 1) };
            let error = if rc < 0 { errno() } else { 0 };
            check(name, if complete { rc == -1 && error == 9 } else { rc >= 0 }, rc as i64, error)?;
            let _ = unsafe { syscall(8, fd, 0i64, 0usize) };
            let mut bytes = [0u8; 64];
            let read = unsafe { syscall(0, fd, bytes.as_mut_ptr(), bytes.len()) };
            let error = if read < 0 { errno() } else { 0 };
            check(&format!("{name}_read"), if complete { read == -1 && error == 9 } else { read == CONTENT.len() as i64 && &bytes[..CONTENT.len()] == CONTENT }, read, error)?;
        }
        for (family, kind, name) in [(2, 1, "tcp_socket"), (2, 2, "udp_socket"), (1, 1, "unix_socket")] {
            let fd = unsafe { socket(family, kind, 0) };
            outcome(name, fd as i64, complete)?;
            if fd >= 0 { unsafe { close(fd); } }
        }
        let rc = unsafe { connect(tcp_client, &tcp as *const Inet as *const u8, 16) };
        outcome("tcp_connect", rc as i64, confined)?;
        let address = unix_address();
        let rc = unsafe { connect(unix_client, &address as *const Unix as *const u8, 24) };
        outcome("cross_domain_abstract_connect", rc as i64, confined)?;
        let udp_client = unsafe { socket(2, 2, 0) };
        if !complete {
            if udp_client < 0 { return Err("fixture_udp"); }
            let sent = unsafe { sendto(udp_client, b"x".as_ptr(), 1, 0, &udp as *const Inet as *const u8, 16) };
            outcome("udp_send_unfiltered_without_seccomp", sent as i64, false)?;
            unsafe { close(udp_client); }
        }
        outcome("cross_domain_signal0", unsafe { kill(peer, 0) } as i64, confined)?;
        let peer_query = unsafe { syscall(302, peer as usize, 7usize, std::ptr::null::<Limits>(), &mut limits as *mut Limits) };
        outcome("prlimit_peer_read", peer_query, complete)?;
        outcome("prlimit_self_same_value_set", unsafe { syscall(302, 0usize, 7usize, &limits as *const Limits, std::ptr::null_mut::<Limits>()) }, complete)?;
        let mut local_marker = MARKER;
        let local = IoVec { base: &mut local_marker as *mut u64 as *mut u8, length: 8 };
        let remote = IoVec { base: marker, length: 8 };
        let read = unsafe { syscall(310, peer as usize, &local as *const IoVec, 1usize, &remote as *const IoVec, 1usize, 0usize) };
        outcome("peer_memory_read", read, confined)?;
        if !confined { check("peer_memory_value", local_marker == MARKER && read == 8, read, 0)?; }
        // Same value only: the positive fixture never changes the owned peer's data.
        let written = unsafe { syscall(311, peer as usize, &local as *const IoVec, 1usize, &remote as *const IoVec, 1usize, 0usize) };
        if confined { outcome("peer_memory_same_value_write", written, true)?; }
        else { check("peer_memory_same_value_write", written == 8, written, if written < 0 { errno() } else { 0 })?; }
        let child = unsafe { fork() };
        if child == 0 { unsafe { syscall(60, 0usize); } unreachable!(); }
        outcome("fork_process", child as i64, complete)?;
        if child > 0 {
            let mut status = 0;
            check("original_child_reaped", unsafe { waitpid(child, &mut status, 0) } == child && status == 0, status as i64, 0)?;
        }
        let unknown = unsafe { syscall(999, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize) };
        let error = errno();
        check("unknown_syscall", unknown == -1 && error == if complete { 1 } else { 38 }, unknown, error)?;
        Ok(())
    }
    pub fn main() {
        if !(cfg!(target_os = "linux") && cfg!(target_arch = "x86_64")) { std::process::exit(125); }
        let abi = unsafe { syscall(444, std::ptr::null::<u8>(), 0usize, 1u32) };
        if abi < 6 { println!("FIXTURE {} UNSUPPORTED abi={abi} errno={}", mode(), errno()); std::process::exit(125); }
        // Supervisor is outside the subject's ruleset and owns its original PID.
        // It performs no provider request and opens only fixed fixture paths/listeners.
        for (name, _) in std::env::vars_os() { std::env::remove_var(name); }
        std::env::set_var("PYTHONUNBUFFERED", "1");
        std::env::set_var("PYTHONDONTWRITEBYTECODE", "1");
        let null = fs::OpenOptions::new().write(true).open("/dev/null").unwrap().into_raw_fd();
        if unsafe { dup2(null, 2) } != 2 { std::process::exit(125); }
        if null != 2 { unsafe { close(null); } }
        let (tcp_fd, tcp) = inet_socket(1).unwrap();
        let (udp_fd, udp) = inet_socket(2).unwrap();
        let unix_fd = unsafe { socket(1, 1, 0) };
        let address = unix_address();
        if unix_fd < 0 || unsafe { bind(unix_fd, &address as *const Unix as *const u8, 24) } != 0
            || unsafe { listen(unix_fd, 4) } != 0 { std::process::exit(125); }
        let marker = MARKER;
        let marker_pointer = std::hint::black_box(&marker) as *const u64 as *mut u8;
        let peer = std::process::id() as i32;
        let mut gate = [-1; 2];
        if unsafe { pipe2(gate.as_mut_ptr(), 0x80000) } != 0 { std::process::exit(125); }
        let child = unsafe { fork() };
        if child < 0 { std::process::exit(125); }
        if child == 0 {
            unsafe { close(gate[1]); }
            let mut byte = 0u8;
            let ready = unsafe { syscall(0, gate[0], &mut byte as *mut u8, 1usize) };
            unsafe { close(gate[0]); }
            if ready != 1 || byte != 1 { std::process::exit(125); }
            let result = subject(peer, marker_pointer, tcp, udp, &[tcp_fd, udp_fd, unix_fd]);
            if let Err(stage) = result { println!("FIXTURE {} REFUSED {stage} errno={}", mode(), errno()); }
            std::process::exit(if result.is_ok() { 0 } else { 125 });
        }
        unsafe { close(gate[0]); }
        let trace = unsafe { prctl(0x59616d61, child as usize, 0usize, 0usize, 0usize) };
        let byte = if trace == 0 { 1u8 } else { 0u8 };
        unsafe { syscall(1, gate[1], &byte as *const u8, 1usize); close(gate[1]); }
        let mut status = 0;
        let reaped = unsafe { waitpid(child, &mut status, 0) };
        for fd in [tcp_fd, udp_fd, unix_fd] { unsafe { close(fd); } }
        println!("FIXTURE {} {} original_pid={child} wait_status={status} reaped={}", mode(), if trace == 0 && reaped == child && status == 0 { "PASS" } else { "REFUSED" }, reaped == child);
        std::process::exit(if trace == 0 && reaped == child && status == 0 { 0 } else { 125 });
    }
}
#[cfg(test)]
fn main() { oracle_native_fixture::main(); }
