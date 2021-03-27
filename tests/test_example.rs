mod common;
use escargot::CargoBuild;

fn check_example(example: &str, test_name: &str, suffix: &str, flags: &[&str]) {
    let path = if cfg!(debug_assertions) {
        format!("target/debug/examples/{}", example)
    } else {
        format!("target/release/examples/{}", example)
    };

    let error = format!("failed to run: {} {:?}", path, flags);

    let actual_output = CargoBuild::new()
        .example(example)
        .current_release()
        .current_target()
        .run()
        .expect(&error)
        .command()
        .args(flags)
        .output()
        .expect(&error);

    if !actual_output.status.success() {
        panic!("{}", error); // NOT TESTED
    }
    let actual_bytes = actual_output.stdout.as_slice();

    common::impl_assert_output(
        &format!("example_{}", example),
        test_name,
        suffix,
        actual_bytes,
    );
}

#[test]
fn test_configurations() {
    check_example(
        "clients_server",
        test_name!(),
        "txt",
        &vec!["configurations"],
    );
}

#[test]
fn test_transitions() {
    check_example("clients_server", test_name!(), "txt", &vec!["transitions"]);
}

#[test]
fn test_client_states() {
    check_example(
        "clients_server",
        test_name!(),
        "dot",
        &vec!["states", "-c", "C(0)"],
    );
}

#[test]
fn test_server_states() {
    check_example(
        "clients_server",
        test_name!(),
        "dot",
        &vec!["states", "-c", "SRV"],
    );
}

#[test]
fn test_path() {
    check_example(
        "clients_server",
        test_name!(),
        "txt",
        &vec!["path", "INIT", "!INIT", "INIT"],
    );
}

#[test]
fn test_sequence() {
    check_example(
        "clients_server",
        test_name!(),
        "uml",
        &vec!["sequence", "INIT", "!INIT", "INIT"],
    );
}
