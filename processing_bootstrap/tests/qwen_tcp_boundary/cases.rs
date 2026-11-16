use super::*;
use std::os::fd::AsRawFd;

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Nnp,
    Abi,
    Create(RulesetAttr, usize, u32),
    Restrict(u32),
}
struct Fake {
    abi: i64,
    nnp: i32,
    fail: Option<&'static str>,
    events: Vec<Event>,
    fd: Option<i32>,
}
impl Fake {
    fn new(abi: i64) -> Self {
        Self {
            abi,
            nnp: 1,
            fail: None,
            events: vec![],
            fd: None,
        }
    }
}
impl Kernel for Fake {
    fn nnp(&mut self) -> i32 {
        self.events.push(Event::Nnp);
        self.nnp
    }
    fn abi(&mut self) -> i64 {
        self.events.push(Event::Abi);
        self.abi
    }
    fn create(
        &mut self,
        attr: &RulesetAttr,
        size: usize,
        flags: u32,
    ) -> Result<File, &'static str> {
        self.events.push(Event::Create(*attr, size, flags));
        if self.fail == Some("create") {
            return Err("injected_create");
        }
        let file = File::open("/dev/null").unwrap();
        self.fd = Some(file.as_raw_fd());
        Ok(file)
    }
    fn restrict(&mut self, _set: &File, flags: u32) -> Result<(), &'static str> {
        self.events.push(Event::Restrict(flags));
        if self.fail == Some("restrict") {
            Err("injected_restrict")
        } else {
            Ok(())
        }
    }
}
fn state() -> NativeState {
    NativeState {
        uid: [10001; 4],
        gid: [10001; 4],
        groups: vec![],
        threads: 1,
        caps: [0, 0, 0, 0, crate::NATIVE_CEILING],
        no_new_privs: 1,
        seccomp: 2,
        keep_caps: 0,
        securebits: 0,
    }
}
#[test]
fn supported_policy_handles_tcp_without_any_port_grants() {
    for abi in [4, 6, 10] {
        let mut kernel = Fake::new(abi);
        let proof = enforce(&state(), &mut kernel).expect("fixed SFT TCP policy must install");
        assert_eq!(
            proof,
            QwenTcpBoundaryProof {
                profile: "qwen_sft_tcp_deny_v1",
                abi: abi as u32,
                handled_access_net: 3,
                allowed_connect_tcp_ports: [],
                allowed_bind_tcp_ports: []
            }
        );
        assert_eq!(
            kernel.events,
            vec![
                Event::Nnp,
                Event::Abi,
                Event::Create(
                    RulesetAttr {
                        handled_access_fs: 0,
                        handled_access_net: 3
                    },
                    16,
                    0
                ),
                Event::Restrict(0)
            ]
        );
    }
}
#[test]
fn unsupported_or_unbounded_abi_stops_before_create() {
    for abi in [-1, 0, 1, 3, i64::MAX] {
        let mut kernel = Fake::new(abi);
        assert_eq!(
            enforce(&state(), &mut kernel),
            Err("qwen_tcp_unsupported_abi")
        );
        assert_eq!(kernel.events, vec![Event::Nnp, Event::Abi]);
        assert!(kernel.fd.is_none());
    }
}
#[test]
fn independent_nnp_read_is_exact_before_mutation() {
    for nnp in [-1, 0, 2] {
        let mut kernel = Fake::new(6);
        kernel.nnp = nnp;
        assert_eq!(enforce(&state(), &mut kernel), Err("qwen_tcp_no_new_privs"));
        assert_eq!(kernel.events, vec![Event::Nnp]);
    }
}
#[test]
fn native_final_or_thread_drift_never_calls_kernel() {
    for case in 0..16 {
        let mut current = state();
        match case {
            0 => current.uid[0] = 0,
            1 => current.uid[3] = 0,
            2 => current.gid[0] = 0,
            3 => current.gid[3] = 0,
            4 => current.groups.push(0),
            5 => current.threads = 2,
            6 => current.threads = 0,
            7 => current.caps[0] = 1,
            8 => current.caps[1] = 1,
            9 => current.caps[2] = 1,
            10 => current.caps[3] = 1,
            11 => current.caps[4] |= 1 << 63,
            12 => current.no_new_privs = 0,
            13 => current.seccomp = 0,
            14 => current.keep_caps = 1,
            _ => current.securebits = 0x10,
        }
        let mut kernel = Fake::new(6);
        assert_eq!(enforce(&current, &mut kernel), Err("qwen_tcp_native_state"));
        assert!(kernel.events.is_empty());
    }
}
#[test]
fn create_and_restrict_failures_return_no_proof() {
    for (failure, expected, event_count) in [
        ("create", "qwen_tcp_ruleset_create", 3),
        ("restrict", "qwen_tcp_restrict_self", 4),
    ] {
        let mut kernel = Fake::new(6);
        kernel.fail = Some(failure);
        assert_eq!(enforce(&state(), &mut kernel), Err(expected));
        assert_eq!(kernel.events.len(), event_count);
    }
}
#[test]
fn ruleset_fd_is_closed_on_both_return_paths() {
    if std::env::var("AMLAS_QWEN_TCP_FD_CHILD").ok().as_deref() != Some("1") {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "qwen_tcp_boundary::tests::ruleset_fd_is_closed_on_both_return_paths",
                "--test-threads=1",
            ])
            .env("AMLAS_QWEN_TCP_FD_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    extern "C" {
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }
    for failure in [None, Some("restrict")] {
        let mut kernel = Fake::new(6);
        kernel.fail = failure;
        assert_eq!(enforce(&state(), &mut kernel).is_ok(), failure.is_none());
        let fd = kernel.fd.expect("owned ruleset FD existed");
        assert_eq!(unsafe { fcntl(fd, 1) }, -1);
        assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(9));
    }
}
#[test]
fn fixed_numeric_denials_require_only_permission_errors() {
    for number in [1, 13] {
        let mut targets = vec![];
        assert_eq!(
            verify_denials(|target| {
                targets.push(target);
                Err(std::io::Error::from_raw_os_error(number))
            }),
            Ok(())
        );
        assert_eq!(
            targets,
            vec![
                SocketAddr::from(([169, 254, 169, 254], 80)),
                SocketAddr::from(([169, 254, 170, 2], 80)),
                SocketAddr::from(([1, 1, 1, 1], 443))
            ]
        );
    }
}
#[test]
fn reachable_timeout_or_einval_never_passes_endpoint_checks() {
    for fail_at in 0..3 {
        for number in [0, 22, 110, 101, 24, 12] {
            let mut calls = 0;
            let result = verify_denials(|_| {
                let index = calls;
                calls += 1;
                if index != fail_at {
                    return Err(std::io::Error::from_raw_os_error(13));
                }
                if number == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::from_raw_os_error(number))
                }
            });
            assert_eq!(result, Err("qwen_tcp_endpoint_denial"));
            assert_eq!(calls, fail_at + 1);
        }
    }
}
