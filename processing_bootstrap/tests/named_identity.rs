use amlas_processing_init::verify_named_directory;
use std::fs::{self, File};

#[test]
fn named_directory_swap_is_rejected_with_original_fd_retained() {
    let path = std::env::temp_dir().join(format!(
        "init-identity-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    fs::create_dir(path.join("output")).unwrap();
    let parent = File::open(&path).unwrap();
    let target = File::open(path.join("output")).unwrap();
    assert!(verify_named_directory(&parent, &target).is_ok());
    fs::rename(path.join("output"), path.join("old-output")).unwrap();
    fs::create_dir(path.join("output")).unwrap();
    assert!(verify_named_directory(&parent, &target).is_err());
}

#[test]
fn named_symlink_to_retained_original_fd_is_rejected() {
    let path = std::env::temp_dir().join(format!(
        "init-symlink-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    fs::create_dir(path.join("output")).unwrap();
    let parent = File::open(&path).unwrap();
    let target = File::open(path.join("output")).unwrap();
    fs::rename(path.join("output"), path.join("old-output")).unwrap();
    std::os::unix::fs::symlink("old-output", path.join("output")).unwrap();
    assert!(verify_named_directory(&parent, &target).is_err());
}
