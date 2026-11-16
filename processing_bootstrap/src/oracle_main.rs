//! No caller path, policy, environment, source or privilege selector.
fn main() {
    if let Err(error) = amlas_processing_init::oracle_boundary::run() {
        eprintln!("ORACLE-INIT-REFUSED {error}");
        std::process::exit(125);
    }
}
