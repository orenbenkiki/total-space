use std::fs::create_dir_all;
use std::fs::read;
use std::fs::write;

/// Find the name of the current test.
#[macro_export]
macro_rules! test_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        let prefix = &name[..name.len() - 3];
        let offset = prefix.rfind("::").unwrap();
        &prefix[offset + 2..]
    }};
}

/// A complete test case.
#[macro_export]
macro_rules! test_case {
    ($name:ident, $suffix:literal, $flags:expr) => {
        #[test]
        fn $name() {
            let app = add_clap(App::new(test_name!()));
            let arg_matches = app.get_matches_from($flags.iter());
            let mut model = test_model(&arg_matches);
            let mut stdout = Vec::new();
            model.do_clap(&arg_matches, &mut stdout);
            common::impl_assert_output(module_path!(), test_name!(), $suffix, &stdout);
        }
    };
}

/// Compare actual results to expected output in a file.
pub fn impl_assert_output(module_name: &str, test_name: &str, suffix: &str, actual_bytes: &[u8]) {
    let expected_dir = format!("tests/expected/{}", module_name);
    let actual_dir = format!("tests/actual/{}", module_name);

    create_dir_all(expected_dir.clone()).unwrap_or_else(|_| {
        panic!(
            "failed to create expected results directory {}",
            expected_dir
        )
    });
    create_dir_all(actual_dir.clone())
        .unwrap_or_else(|_| panic!("failed to create actual results directory {}", actual_dir));

    let expected_path = format!("{}/{}.{}", expected_dir, test_name, suffix);
    let actual_path = format!("{}/{}.{}", actual_dir, test_name, suffix);

    let expected_bytes = read(expected_path.clone()).unwrap_or_else(|_| {
        write(expected_path.clone(), actual_bytes).unwrap_or_else(|_| {
            panic!("failed to write expected results file {}", expected_path);
        });
        eprintln!(
            "WARNING: created expected results file {}, verify its contents",
            expected_path
        );
        actual_bytes.to_vec()
    });

    write(actual_path.clone(), actual_bytes)
        .unwrap_or_else(|_| panic!("failed to write actual results file {}", actual_path));

    assert!(
        &expected_bytes == actual_bytes,
        "The actual results file {} is different from the expected results file {}",
        expected_path,
        actual_path
    );
}
