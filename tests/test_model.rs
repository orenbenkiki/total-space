extern crate total_space;

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use total_space::*;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct Data {
    pub i: i32,
}

impl Display for Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.i)
    }
}

#[test]
fn test_memoize_value() {
    let mut memoize = Memoize::<Data, u8>::new(false, None);

    let first = Data { i: 17 };
    assert_eq!(first.i, 17);
    assert_eq!(memoize.store(first), 0);
    assert_eq!(first, *memoize.get(0));

    let second = Data { i: 11 };
    assert_eq!(second.i, 11);
    assert_eq!(memoize.store(second), 1);
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));

    assert_eq!(memoize.store(second), 1);
    assert_eq!(memoize.store(first), 0);
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));
}

#[test]
fn test_memoize_display() {
    let mut memoize = Memoize::<Data, u8>::new(true, None);

    let first = Data { i: 17 };
    assert_eq!(first.i, 17);
    assert_eq!(memoize.store(first), 0);
    assert_eq!(first, *memoize.get(0));
    assert_eq!("17", memoize.display(0));

    let second = Data { i: 11 };
    assert_eq!(second.i, 11);
    assert_eq!(memoize.store(second), 1);
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));
    assert_eq!("17", memoize.display(0));
    assert_eq!("11", memoize.display(1));

    assert_eq!(memoize.store(second), 1);
    assert_eq!(memoize.store(first), 0);
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));
    assert_eq!("17", memoize.display(0));
    assert_eq!("11", memoize.display(1));
}

#[test]
#[should_panic(expected = "too many memoized objects")]
fn test_memoize_limit() {
    let mut memoize = Memoize::<Data, u8>::new(false, Some(1));

    let first = Data { i: 17 };
    assert_eq!(first.i, 17);
    assert_eq!(memoize.store(first), 0);
    assert_eq!(first, *memoize.get(0));

    let second = Data { i: 11 };
    assert_eq!(second.i, 11);
    memoize.store(second);
}
