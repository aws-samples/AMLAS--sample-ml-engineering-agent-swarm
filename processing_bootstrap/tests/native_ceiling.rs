use amlas_processing_init::{validate_native_final, validate_native_initial, NativeState};
const CEILING: u64 = 0x200404fb;
fn initial() -> NativeState {
    NativeState {
        uid: [0; 4],
        gid: [0; 4],
        groups: vec![0],
        threads: 1,
        caps: [CEILING, CEILING, 0, 0, CEILING],
        no_new_privs: 0,
        seccomp: 2,
        keep_caps: 0,
        securebits: 0,
    }
}
fn final_state() -> NativeState {
    NativeState {
        uid: [10001; 4],
        gid: [10001; 4],
        groups: vec![],
        threads: 1,
        caps: [0, 0, 0, 0, CEILING],
        no_new_privs: 1,
        seccomp: 2,
        keep_caps: 0,
        securebits: 0,
    }
}
fn accept(s: &NativeState) -> Result<u64, &'static str> {
    validate_native_initial(s, 40, |cap| ((s.caps[4] >> cap) & 1) as i32)
}
#[test]
fn measured_root_ceiling_requires_matching_kernel_reads() {
    let s = initial();
    assert_eq!(accept(&s), Ok(CEILING));
    assert!(validate_native_initial(&s, 40, |_| 0).is_err());
    assert!(validate_native_initial(&s, 40, |_| 2).is_err());
    assert!(validate_native_initial(&s, 40, |_| -1).is_err());
}
#[test]
fn reduced_subsets_and_empty_groups_are_admitted_without_ceiling_expansion() {
    let mut s = initial();
    s.groups.clear();
    s.caps[0] &= !1;
    s.caps[1] &= !1;
    assert_eq!(accept(&s), Ok(CEILING));
    s.no_new_privs = 1;
    assert_eq!(accept(&s), Ok(CEILING));
}
#[test]
fn initial_drift_refuses_before_kernel_or_output_work() {
    let mut bad = vec![];
    let mut s = initial();
    s.uid[0] = 10001;
    bad.push(s);
    let mut s = initial();
    s.gid[1] = 10001;
    bad.push(s);
    let mut s = initial();
    s.groups = vec![0, 10002];
    bad.push(s);
    let mut s = initial();
    s.threads = 2;
    bad.push(s);
    let mut s = initial();
    s.no_new_privs = 2;
    bad.push(s);
    let mut s = initial();
    s.seccomp = 0;
    bad.push(s);
    let mut s = initial();
    s.keep_caps = 1;
    bad.push(s);
    for field in 0..5 {
        let mut s = initial();
        s.caps[field] |= 1 << 8;
        bad.push(s);
    }
    for bits in [4, 16, 12, 48, 256, -1] {
        let mut s = initial();
        s.securebits = bits;
        bad.push(s);
    }
    for s in bad {
        assert!(validate_native_initial(&s, 40, |_| panic!(
            "invalid initial state reached kernel read"
        ))
        .is_err());
    }
    assert!(validate_native_initial(&initial(), 64, |_| panic!("range reached read")).is_err());
}
#[test]
fn locked_off_known_securebits_allowed_and_unknown_bits_refused() {
    for bits in [0, 3, 8, 32, 128] {
        let mut s = initial();
        s.securebits = bits;
        assert_eq!(accept(&s), Ok(CEILING));
        let mut f = final_state();
        f.securebits = bits;
        assert_eq!(validate_native_final(&f, CEILING), Ok(()));
    }
}
#[test]
fn final_exact_retained_ceiling_and_permanent_nonroot_slots() {
    assert_eq!(validate_native_final(&final_state(), CEILING), Ok(()));
    let mut s = final_state();
    s.caps[4] &= !1;
    assert!(validate_native_final(&s, CEILING).is_err());
}
#[test]
fn final_fs_ids_active_caps_nnp_keepcaps_and_seccomp_drift_refuse() {
    for index in 0..4 {
        let mut s = final_state();
        s.uid[index] = 0;
        assert!(validate_native_final(&s, CEILING).is_err());
        let mut s = final_state();
        s.gid[index] = 0;
        assert!(validate_native_final(&s, CEILING).is_err());
        let mut s = final_state();
        s.caps[index] = 1;
        assert!(validate_native_final(&s, CEILING).is_err());
    }
    let mut s = final_state();
    s.groups.push(0);
    assert!(validate_native_final(&s, CEILING).is_err());
    let mut s = final_state();
    s.no_new_privs = 0;
    assert!(validate_native_final(&s, CEILING).is_err());
    let mut s = final_state();
    s.keep_caps = 1;
    assert!(validate_native_final(&s, CEILING).is_err());
    let mut s = final_state();
    s.seccomp = 1;
    assert!(validate_native_final(&s, CEILING).is_err());
    let mut s = final_state();
    s.securebits = 4;
    assert!(validate_native_final(&s, CEILING).is_err());
}
