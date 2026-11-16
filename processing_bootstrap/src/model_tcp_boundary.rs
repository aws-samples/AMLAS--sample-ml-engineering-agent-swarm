//! Fixed raw-model TCP port boundary; no caller-selectable network authority.
use crate::NativeState;
use std::fs::File;

const CONNECT_TCP: u64 = 2;
const HANDLED_NET: u64 = 3;
const CONNECT_PORTS: [u16; 2] = [53, 443];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcpBoundaryProof {
    pub profile: &'static str,
    pub abi: u32,
    pub handled_access_net: u64,
    pub allowed_connect_tcp_ports: [u16; 2],
    pub allowed_bind_tcp_ports: [u16; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RulesetAttr {
    handled_access_fs: u64,
    handled_access_net: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PortAttr {
    allowed_access: u64,
    port: u64,
}
trait Kernel {
    fn nnp(&mut self) -> i32;
    fn abi(&mut self) -> i64;
    fn create(&mut self, attr: &RulesetAttr, size: usize, flags: u32)
        -> Result<File, &'static str>;
    fn add(
        &mut self,
        set: &File,
        kind: u32,
        attr: &PortAttr,
        flags: u32,
    ) -> Result<(), &'static str>;
    fn restrict(&mut self, set: &File, flags: u32) -> Result<(), &'static str>;
}
fn enforce(
    state: &NativeState,
    kernel: &mut impl Kernel,
) -> Result<TcpBoundaryProof, &'static str> {
    crate::validate_native_final(state, state.caps[4]).map_err(|_| "tcp_native_state")?;
    if state.threads != 1 {
        return Err("tcp_native_state");
    }
    if kernel.nnp() != 1 {
        return Err("tcp_no_new_privs");
    }
    let version = kernel.abi();
    let abi = u32::try_from(version).map_err(|_| "tcp_unsupported_abi")?;
    if abi < 4 {
        return Err("tcp_unsupported_abi");
    }
    let ruleset = kernel
        .create(
            &RulesetAttr {
                handled_access_fs: 0,
                handled_access_net: HANDLED_NET,
            },
            std::mem::size_of::<RulesetAttr>(),
            0,
        )
        .map_err(|_| "tcp_ruleset_create")?;
    for port in CONNECT_PORTS {
        kernel
            .add(
                &ruleset,
                2,
                &PortAttr {
                    allowed_access: CONNECT_TCP,
                    port: u64::from(port),
                },
                0,
            )
            .map_err(|_| "tcp_port_rule")?;
    }
    kernel
        .restrict(&ruleset, 0)
        .map_err(|_| "tcp_restrict_self")?;
    // File owns the ruleset FD on all success/error paths. The restriction remains
    // enforced by the kernel after closing this handle; no authority FD escapes.
    Ok(TcpBoundaryProof {
        profile: "raw_model_tcp_ports_v1",
        abi,
        handled_access_net: HANDLED_NET,
        allowed_connect_tcp_ports: CONNECT_PORTS,
        allowed_bind_tcp_ports: [],
    })
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
            return Err("tcp_ruleset_create");
        }
        Ok(unsafe { File::from_raw_fd(fd as i32) })
    }
    fn add(
        &mut self,
        set: &File,
        kind: u32,
        attr: &PortAttr,
        flags: u32,
    ) -> Result<(), &'static str> {
        use std::os::fd::AsRawFd;
        if unsafe { syscall(445, set.as_raw_fd(), kind, attr as *const PortAttr, flags) } != 0 {
            return Err("tcp_port_rule");
        }
        Ok(())
    }
    fn restrict(&mut self, set: &File, flags: u32) -> Result<(), &'static str> {
        use std::os::fd::AsRawFd;
        if unsafe { syscall(446, set.as_raw_fd(), flags) } != 0 {
            return Err("tcp_restrict_self");
        }
        Ok(())
    }
}

/// Apply only the fixed TCP port policy after the initializer's permanent drop.
/// `state` is the caller's same freshly kernel-measured native-final state,
/// already checked against its verified initial bounding mask. This module
/// reuses that validator, requires one original thread, and independently reads
/// kernel NNP. No ports, addresses, profile, environment or syscall selectors exist.
/// TCP53 preserves DNS fallback; TCP443 preserves public TLS. Explicit binds
/// and all other new TCP connects are denied. IPs, UDP and existing connections
/// are not filtered or retroactively revoked by this separately stacked policy.
/// Returned metadata describes installed policy; it is not native attestation.
pub fn apply_model_tcp_boundary(state: &NativeState) -> Result<TcpBoundaryProof, &'static str> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        enforce(state, &mut LinuxKernel)
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = state;
        Err("tcp_unsupported_platform")
    }
}

#[cfg(test)]
#[path = "../tests/model_tcp_boundary/cases.rs"]
mod tests;
