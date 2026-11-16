use amlas_processing_init::drop_bounding_capabilities;
use std::collections::VecDeque;

#[test]
fn absent_bits_do_not_request_drop() {
    let mut reads = Vec::new();
    assert_eq!(
        drop_bounding_capabilities(
            2,
            |cap| {
                reads.push(cap);
                0
            },
            |_| panic!("absent bit was dropped")
        ),
        Ok(())
    );
    assert_eq!(reads, [0, 1, 2]);
}

#[test]
fn present_bit_requires_successful_drop_and_zero_readback() {
    let mut answers = VecDeque::from([0, 1, 0, 0]);
    let mut reads = Vec::new();
    let mut drops = Vec::new();
    assert_eq!(
        drop_bounding_capabilities(
            2,
            |cap| {
                reads.push(cap);
                answers.pop_front().unwrap()
            },
            |cap| {
                drops.push(cap);
                0
            }
        ),
        Ok(())
    );
    assert_eq!(reads, [0, 1, 1, 2]);
    assert_eq!(drops, [1]);
    assert!(answers.is_empty());
}

#[test]
fn read_errors_and_non_boolean_results_refuse_without_drop() {
    for answer in [-1, 2] {
        assert_eq!(
            drop_bounding_capabilities(0, |_| answer, |_| panic!("failed read requested drop")),
            Err("bounding_capability_read")
        );
    }
}

#[test]
fn present_undroppable_bit_refuses() {
    assert_eq!(
        drop_bounding_capabilities(0, |_| 1, |_| -1),
        Err("bounding_capability_drop")
    );
}

#[test]
fn failed_or_nonzero_readback_refuses() {
    for answer in [-1, 1, 2] {
        let mut answers = VecDeque::from([1, answer]);
        assert_eq!(
            drop_bounding_capabilities(0, |_| answers.pop_front().unwrap(), |_| 0),
            Err("bounding_capability_readback")
        );
    }
}

#[test]
fn invalid_capability_range_refuses_without_syscalls() {
    assert_eq!(
        drop_bounding_capabilities(64, |_| panic!("invalid read"), |_| panic!("invalid drop")),
        Err("capability_range")
    );
}
