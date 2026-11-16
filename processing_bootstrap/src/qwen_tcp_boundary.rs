//! Fixed Qwen-only denial of new TCP bind/connect; no caller-selectable grants.
use crate::NativeState;
use std::fs::File;
use std::net::SocketAddr;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QwenTcpBoundaryProof {
    pub profile: &'static str,
    pub abi: u32,
    pub handled_access_net: u64,
    pub allowed_connect_tcp_ports: [u16; 0],
    pub allowed_bind_tcp_ports: [u16; 0],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RulesetAttr {
    handled_access_fs: u64,
    handled_access_net: u64,
}
trait Kernel {
    fn nnp(&mut self) -> i32;
    fn abi(&mut self) -> i64;
    fn create(&mut self, attr: &RulesetAttr, size: usize, flags: u32)
        -> Result<File, &'static str>;
    fn restrict(&mut self, set: &File, flags: u32) -> Result<(), &'static str>;
}
fn enforce(
    state: &NativeState,
    kernel: &mut impl Kernel,
) -> Result<QwenTcpBoundaryProof, &'static str> {
    crate::validate_native_final(state, state.caps[4]).map_err(|_| "qwen_tcp_native_state")?;
    if state.threads != 1 {
        return Err("qwen_tcp_native_state");
    }
    if kernel.nnp() != 1 {
        return Err("qwen_tcp_no_new_privs");
    }
    let abi = u32::try_from(kernel.abi()).map_err(|_| "qwen_tcp_unsupported_abi")?;
    if abi < 4 {
        return Err("qwen_tcp_unsupported_abi");
    }
    let ruleset = kernel
        .create(
            &RulesetAttr {
                handled_access_fs: 0,
                handled_access_net: 3,
            },
            std::mem::size_of::<RulesetAttr>(),
            0,
        )
        .map_err(|_| "qwen_tcp_ruleset_create")?;
    // No NET_PORT rules: all handled new TCP binds/connects remain denied.
    kernel
        .restrict(&ruleset, 0)
        .map_err(|_| "qwen_tcp_restrict_self")?;
    Ok(QwenTcpBoundaryProof {
        profile: "qwen_sft_tcp_deny_v1",
        abi,
        handled_access_net: 3,
        allowed_connect_tcp_ports: [],
        allowed_bind_tcp_ports: [],
    })
}
fn verify_denials(
    mut connect: impl FnMut(SocketAddr) -> std::io::Result<()>,
) -> Result<(), &'static str> {
    for target in [
        SocketAddr::from(([169, 254, 169, 254], 80)),
        SocketAddr::from(([169, 254, 170, 2], 80)),
        SocketAddr::from(([1, 1, 1, 1], 443)),
    ] {
        match connect(target) {
            Err(error) if matches!(error.raw_os_error(), Some(1 | 13)) => {}
            _ => return Err("qwen_tcp_endpoint_denial"),
        }
    }
    Ok(())
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
struct LinuxKernel;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
extern "C" {
    fn syscall(number: i64, ...) -> i64;
    fn prctl(option: i32, ...) -> i32;
}
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
impl Kernel for LinuxKernel {
    fn nnp(&mut self) -> i32 {
        unsafe { prctl(39, 0usize, 0usize, 0usize, 0usize) }
    }
    fn abi(&mut self) -> i64 {
        unsafe { syscall(444, std::ptr::null::<RulesetAttr>(), 0usize, 1u32) }
    }
    fn create(
        &mut self,
        attr: &RulesetAttr,
        size: usize,
        flags: u32,
    ) -> Result<File, &'static str> {
        use std::os::fd::FromRawFd;
        let fd = unsafe { syscall(444, attr as *const RulesetAttr, size, flags) };
        if fd < 0 || fd > i32::MAX as i64 {
            return Err("qwen_tcp_ruleset_create");
        }
        Ok(unsafe { File::from_raw_fd(fd as i32) })
    }
    fn restrict(&mut self, set: &File, flags: u32) -> Result<(), &'static str> {
        use std::os::fd::AsRawFd;
        if unsafe { syscall(446, set.as_raw_fd(), flags) } != 0 {
            return Err("qwen_tcp_restrict_self");
        }
        Ok(())
    }
}
/// Stack fixed new-TCP denial after the initializer's verified permanent drop.
/// Existing sockets, UDP, IP and other families are not revoked or filtered.
pub fn apply_qwen_tcp_boundary(state: &NativeState) -> Result<QwenTcpBoundaryProof, &'static str> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        enforce(state, &mut LinuxKernel)
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = state;
        Err("qwen_tcp_unsupported_platform")
    }
}
/// Fixed numeric permission-denial observations; errno alone does not attribute the LSM.
pub fn verify_qwen_tcp_endpoint_denial() -> Result<(), &'static str> {
    verify_denials(|target| {
        std::net::TcpStream::connect_timeout(&target, std::time::Duration::from_millis(500))
            .map(drop)
    })
}

#[cfg(test)]
#[path = "../tests/qwen_tcp_boundary/cases.rs"]
mod tests;
