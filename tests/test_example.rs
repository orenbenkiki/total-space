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
fn test_conditions() {
    check_example("simple", test_name!(), "txt", &vec!["conditions"]);
}

#[test]
fn test_agents() {
    check_example("simple", test_name!(), "txt", &vec!["agents"]);
}

#[test]
fn test_legend() {
    check_example("simple", test_name!(), "txt", &vec!["legend"]);
}

#[test]
fn test_configurations() {
    check_example("simple", test_name!(), "txt", &vec!["configurations"]);
}

#[test]
fn test_transitions() {
    check_example("simple", test_name!(), "txt", &vec!["transitions"]);
}

#[test]
fn test_client_states() {
    check_example("simple", test_name!(), "dot", &vec!["states", "-c", "C(0)"]);
}

#[test]
fn test_server_states() {
    check_example("simple", test_name!(), "dot", &vec!["states", "-c", "S"]);
}

#[test]
fn test_1_1_path() {
    check_example(
        "simple",
        test_name!(),
        "txt",
        &vec!["path", "INIT", "!INIT", "INIT"],
    );
}

#[test]
fn test_1_1_sequence() {
    check_example(
        "simple",
        test_name!(),
        "uml",
        &vec!["sequence", "INIT", "!INIT", "INIT"],
    );
}

#[test]
fn test_2_1_path() {
    check_example(
        "simple",
        test_name!(),
        "txt",
        &vec!["-C", "2", "path", "INIT", "DEFERRED_TASK", "INIT"],
    );
}

#[test]
fn test_2_1_sequence() {
    check_example(
        "simple",
        test_name!(),
        "uml",
        &vec!["-C", "2", "sequence", "INIT", "DEFERRED_TASK", "INIT"],
    );
}
