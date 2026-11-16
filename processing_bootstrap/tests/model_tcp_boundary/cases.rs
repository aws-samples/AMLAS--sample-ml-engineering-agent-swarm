use super::*;
use std::os::fd::AsRawFd;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Nnp,
    Abi,
    Create(RulesetAttr, usize, u32),
    Add(u32, PortAttr, u32),
    Restrict(u32),
}
struct Fake {
    version: i64,
    nnp: i32,
    fail: Option<&'static str>,
    events: Vec<Event>,
    owned_fd: Option<i32>,
}
impl Fake {
    fn new(version: i64) -> Self {
        Self {
            version,
            nnp: 1,
            fail: None,
            events: Vec::new(),
            owned_fd: None,
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
        self.version
    }
    fn create(
        &mut self,
        attr: &RulesetAttr,
        size: usize,
        flags: u32,
    ) -> Result<File, &'static str> {
        self.events.push(Event::Create(*attr, size, flags));
        if self.fail == Some("create") {
            return Err("fake_create");
        }
        let file = File::open("/dev/null").unwrap();
        self.owned_fd = Some(file.as_raw_fd());
        Ok(file)
    }
    fn add(
        &mut self,
        _set: &File,
        kind: u32,
        attr: &PortAttr,
        flags: u32,
    ) -> Result<(), &'static str> {
        self.events.push(Event::Add(kind, *attr, flags));
        if self.fail == Some(if attr.port == 53 { "add53" } else { "add443" }) {
            Err("fake_add")
        } else {
            Ok(())
        }
    }
    fn restrict(&mut self, _set: &File, flags: u32) -> Result<(), &'static str> {
        self.events.push(Event::Restrict(flags));
        if self.fail == Some("restrict") {
            Err("fake_restrict")
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
fn fixed_tcp_policy_is_exact_on_supported_abis() {
    for version in [4, 6, 10] {
        let mut kernel = Fake::new(version);
        let result = enforce(&state(), &mut kernel);
        assert!(
            result.is_ok(),
            "fixed TCP policy must succeed on admitted ABI: {result:?}"
        );
        let proof = result.unwrap();
        assert_eq!(
            proof,
            TcpBoundaryProof {
                profile: "raw_model_tcp_ports_v1",
                abi: version as u32,
                handled_access_net: 3,
                allowed_connect_tcp_ports: [53, 443],
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
                Event::Add(
                    2,
                    PortAttr {
                        allowed_access: 2,
                        port: 53
                    },
                    0
                ),
                Event::Add(
                    2,
                    PortAttr {
                        allowed_access: 2,
                        port: 443
                    },
                    0
                ),
                Event::Restrict(0)
            ]
        );
    }
}
#[test]
fn unsupported_or_unbounded_abi_never_mutates_policy() {
    for version in [-1, 0, 1, 3, i64::MAX] {
        let mut kernel = Fake::new(version);
        assert!(enforce(&state(), &mut kernel).is_err());
        assert_eq!(kernel.events, vec![Event::Nnp, Event::Abi]);
        assert!(kernel.owned_fd.is_none());
    }
}
#[test]
fn actual_nnp_read_is_required_before_abi_or_create() {
    for nnp in [-1, 0, 2] {
        let mut kernel = Fake::new(6);
        kernel.nnp = nnp;
        assert!(enforce(&state(), &mut kernel).is_err());
        assert_eq!(kernel.events, vec![Event::Nnp]);
    }
}
#[test]
fn native_state_drift_refuses_before_any_syscall() {
    for case in 0..11 {
        let mut observed = state();
        match case {
            0 => observed.uid[3] = 0,
            1 => observed.gid[0] = 0,
            2 => observed.groups.push(10001),
            3 => observed.threads = 2,
            4 => observed.caps[0] = 1,
            5 => observed.caps[1] = 1,
            6 => observed.caps[2] = 1,
            7 => observed.caps[3] = 1,
            8 => observed.caps[4] |= 1 << 63,
            9 => observed.no_new_privs = 0,
            _ => observed.seccomp = 0,
        }
        let mut kernel = Fake::new(6);
        assert!(enforce(&observed, &mut kernel).is_err());
        assert!(kernel.events.is_empty());
    }
}
#[test]
fn syscall_failures_never_return_proof_or_continue() {
    for (failure, last) in [("create", 2), ("add53", 3), ("add443", 4), ("restrict", 5)] {
        let mut kernel = Fake::new(6);
        kernel.fail = Some(failure);
        assert!(enforce(&state(), &mut kernel).is_err());
        assert_eq!(kernel.events.len(), last + 1);
        if failure == "create" {
            assert!(kernel.owned_fd.is_none());
        }
    }
}
#[test]
fn both_rules_use_host_u64_ports_and_fixed_layout() {
    assert_eq!(std::mem::size_of::<RulesetAttr>(), 16);
    assert_eq!(std::mem::size_of::<PortAttr>(), 16);
    assert_eq!(std::mem::align_of::<PortAttr>(), 8);
    let mut kernel = Fake::new(6);
    assert!(enforce(&state(), &mut kernel).is_ok());
    for event in kernel.events {
        if let Event::Add(kind, attr, flags) = event {
            assert_eq!((kind, flags, attr.allowed_access), (2, 0, 2));
            assert!(attr.port == 53 || attr.port == 443);
            let bytes =
                unsafe { std::slice::from_raw_parts((&attr as *const PortAttr).cast::<u8>(), 16) };
            assert_eq!(&bytes[8..], &attr.port.to_ne_bytes());
            assert_eq!(&bytes[..8], &2u64.to_ne_bytes());
        }
    }
}
#[test]
fn ruleset_fd_closes_on_failure_and_success() {
    if std::env::var("AMLAS_TCP_FD_CLOSURE_CHILD").ok().as_deref() != Some("1") {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "model_tcp_boundary::tests::ruleset_fd_closes_on_failure_and_success",
                "--test-threads=1",
            ])
            .env("AMLAS_TCP_FD_CLOSURE_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success(), "isolated closure child failed");
        return;
    }

    extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    for failure in [None, Some("add53"), Some("add443"), Some("restrict")] {
        let mut kernel = Fake::new(6);
        kernel.fail = failure;
        let result = enforce(&state(), &mut kernel);
        assert_eq!(result.is_ok(), failure.is_none());
        let fd = kernel.owned_fd.expect("ruleset FD must have existed");
        assert_eq!(
            unsafe { fcntl(fd, 1) },
            -1,
            "the owned ruleset must close before return"
        );
        assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(9));
    }
}

#[test]
fn descriptor_number_reuse_is_explicitly_ambiguous() {
    if std::env::var("AMLAS_TCP_FD_REUSE_CHILD").ok().as_deref() != Some("1") {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "model_tcp_boundary::tests::descriptor_number_reuse_is_explicitly_ambiguous",
                "--test-threads=1",
            ])
            .env("AMLAS_TCP_FD_REUSE_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    let original = File::open("/dev/null").unwrap();
    let fd = original.as_raw_fd();
    drop(original);
    let reused = File::open("/dev/null").unwrap();
    assert_eq!(fd, reused.as_raw_fd());
    assert!(
        unsafe { fcntl(fd, 1) } >= 0,
        "current descriptor says nothing about historical closure"
    );
}
#[test]
fn caller_original_bounding_drift_must_refuse() {
    let initial = crate::NATIVE_CEILING;
    let mut final_state = state();
    final_state.caps[4] &= !1;
    assert!(crate::validate_native_final(&final_state, initial).is_err());
    assert!(crate::validate_native_final(&state(), initial).is_ok());
}
