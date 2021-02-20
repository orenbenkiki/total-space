extern crate total_space;

use std::hash::Hash;
use total_space::*;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct Data {
    pub i: i32,
}

#[test]
fn test_memoize() {
    let mut memory = Memoize::<Data, u8>::new();

    let first = Data { i: 17 };
    assert_eq!(first.i, 17);
    assert_eq!(memory.store(first), 0);
    assert_eq!(first, memory.get(0));

    let second = Data { i: 11 };
    assert_eq!(second.i, 11);
    assert_eq!(memory.store(second), 1);
    assert_eq!(first, memory.get(0));
    assert_eq!(second, memory.get(1));

    assert_eq!(memory.store(second), 1);
    assert_eq!(memory.store(first), 0);
    assert_eq!(first, memory.get(0));
    assert_eq!(second, memory.get(1));
}
