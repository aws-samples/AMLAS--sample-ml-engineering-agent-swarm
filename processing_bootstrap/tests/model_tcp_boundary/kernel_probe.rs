//! Fixed controlled kernel probe, excluded from production images.
use amlas_processing_init::{
    model_tcp_boundary::apply_model_tcp_boundary, validate_native_final, validate_native_initial,
    NativeState,
};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::os::fd::AsRawFd;
use std::time::Duration;
extern "C" {
    fn syscall(number: i64, ...) -> i64;
    fn prctl(option: i32, ...) -> i32;
    fn setgroups(count: usize, groups: *const u32) -> i32;
    fn setresgid(a: u32, b: u32, c: u32) -> i32;
    fn setresuid(a: u32, b: u32, c: u32) -> i32;
    fn fork() -> i32;
    fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
    fn close(fd: i32) -> i32;
}
fn state() -> NativeState {
    let text = std::fs::read_to_string("/proc/self/status").unwrap();
    let field = |name: &str| {
        text.lines()
            .find(|line| line.starts_with(name))
            .unwrap()
            .split_once(':')
            .unwrap()
            .1
            .trim()
    };
    let ids = |name: &str| {
        field(name)
            .split_whitespace()
            .map(|s| s.parse::<u32>().unwrap())
            .collect::<Vec<_>>()
    };
    NativeState {
        uid: ids("Uid").try_into().unwrap(),
        gid: ids("Gid").try_into().unwrap(),
        groups: ids("Groups"),
        threads: field("Threads").parse().unwrap(),
        caps: ["CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd"]
            .map(|name| u64::from_str_radix(field(name), 16).unwrap()),
        no_new_privs: field("NoNewPrivs").parse().unwrap(),
        seccomp: field("Seccomp").parse().unwrap(),
        keep_caps: unsafe { prctl(7, 0usize, 0usize, 0usize, 0usize) },
        securebits: unsafe { prctl(27, 0usize, 0usize, 0usize, 0usize) },
    }
}
fn drop_identity() {
    assert_eq!(unsafe { prctl(38, 1usize, 0usize, 0usize, 0usize) }, 0);
    assert_eq!(unsafe { setgroups(0, std::ptr::null()) }, 0);
    assert_eq!(unsafe { setresgid(10001, 10001, 10001) }, 0);
    assert_eq!(unsafe { setresuid(10001, 10001, 10001) }, 0);
}
fn connect(port: u16) -> std::io::Result<TcpStream> {
    TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_secs(1),
    )
}
fn main() {
    let initial_state = state();
    let last: u32 = std::fs::read_to_string("/proc/sys/kernel/cap_last_cap")
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let original_bnd = validate_native_initial(&initial_state, last, |cap| unsafe {
        prctl(23, cap as usize, 0usize, 0usize, 0usize)
    })
    .unwrap();
    let abi = unsafe { syscall(444, std::ptr::null::<u8>(), 0usize, 1u32) };
    if abi < 4 {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
        drop_identity();
        let observed = state();
        assert_eq!(observed.threads, 1);
        validate_native_final(&observed, original_bnd).unwrap();
        let result = apply_model_tcp_boundary(&observed);
        assert!(
            result.is_err(),
            "unsupported ABI must never return installed proof"
        );
        println!("{{\"status\":\"UNSUPPORTED_REFUSED\",\"abi_query\":{abi},\"errno\":{errno},\"stage\":\"{}\",\"native_admission\":false}}",result.unwrap_err());
        return;
    }
    // All listeners are fixed loopback controls opened before the child policy.
    let listeners =
        [53, 80, 443].map(|port| TcpListener::bind(format!("127.0.0.1:{port}")).unwrap());
    // Independent identical dropped child without this ruleset establishes attribution.
    let baseline = unsafe { fork() };
    assert!(baseline >= 0);
    if baseline == 0 {
        for listener in &listeners {
            assert_eq!(unsafe { close(listener.as_raw_fd()) }, 0);
        }
        std::mem::forget(listeners);
        drop_identity();
        let observed = state();
        validate_native_final(&observed, original_bnd).unwrap();
        let unconfounded = [53, 80, 443].into_iter().all(|port| connect(port).is_ok())
            && TcpListener::bind("127.0.0.1:0").is_ok();
        std::process::exit(if unconfounded { 0 } else { 2 });
    }
    let mut baseline_status = 0;
    assert_eq!(
        unsafe { waitpid(baseline, &mut baseline_status, 0) },
        baseline
    );
    if baseline_status != 0 {
        println!("{{\"status\":\"INCONCLUSIVE_BASELINE\",\"native_admission\":false}}");
        std::process::exit(2);
    }
    let child = unsafe { fork() };
    assert!(child >= 0);
    if child == 0 {
        for listener in &listeners {
            assert_eq!(unsafe { close(listener.as_raw_fd()) }, 0);
        }
        std::mem::forget(listeners);
        drop_identity();
        let observed = state();
        assert_eq!(observed.threads, 1);
        validate_native_final(&observed, original_bnd).unwrap();
        let proof = match apply_model_tcp_boundary(&observed) {
            Ok(value) => value,
            Err(stage) => {
                println!("{{\"status\":\"KERNEL_REFUSED\",\"abi_query\":{abi},\"stage\":\"{stage}\",\"native_admission\":false}}");
                std::process::exit(2);
            }
        };
        assert!(connect(53).is_ok());
        assert!(connect(443).is_ok());
        let denied = connect(80).unwrap_err();
        assert!([Some(1), Some(13)].contains(&denied.raw_os_error()));
        let bind = TcpListener::bind("127.0.0.1:0").unwrap_err();
        assert!([Some(1), Some(13)].contains(&bind.raw_os_error()));
        // This module deliberately does not handle UDP, even on newer ABIs.
        let udp = UdpSocket::bind("127.0.0.1:0").unwrap();
        assert!(udp.send_to(b"x", "127.0.0.1:80").is_ok());
        println!("{{\"status\":\"CONTROLLED_TCP_PASS\",\"abi\":{},\"handled_access_net\":{},\"connect_ports\":[53,443],\"tcp80_denied\":true,\"explicit_bind_denied\":true,\"udp_unhandled\":true,\"native_admission\":false}}",proof.abi,proof.handled_access_net);
        std::process::exit(0);
    }
    let mut status = 0;
    assert_eq!(unsafe { waitpid(child, &mut status, 0) }, child);
    assert_eq!(
        status, 0,
        "kernel child failed/refused; not an installed positive"
    );
}
