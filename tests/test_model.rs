extern crate total_space;

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::mem::size_of;
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
    assert_eq!(memoize.insert(first), 0);
    assert_eq!(first, *memoize.get(0));

    let second = Data { i: 11 };
    assert_eq!(second.i, 11);
    assert_eq!(memoize.insert(second), 1);
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));

    assert_eq!(*memoize.lookup(&second).unwrap(), 1);
    assert_eq!(*memoize.lookup(&first).unwrap(), 0);
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));
}

#[test]
fn test_memoize_display() {
    let mut memoize = Memoize::<Data, u8>::new(true, None);

    let first = Data { i: 17 };
    assert_eq!(first.i, 17);
    assert_eq!(memoize.insert(first), 0);
    assert_eq!(first, *memoize.get(0));
    assert_eq!("17", memoize.display(0));

    let second = Data { i: 11 };
    assert_eq!(second.i, 11);
    assert_eq!(memoize.insert(second), 1);
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));
    assert_eq!("17", memoize.display(0));
    assert_eq!("11", memoize.display(1));

    assert_eq!(*memoize.lookup(&second).unwrap(), 1);
    assert_eq!(*memoize.lookup(&first).unwrap(), 0);
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
    assert_eq!(memoize.insert(first), 0);
    assert_eq!(first, *memoize.get(0));

    let second = Data { i: 11 };
    assert_eq!(second.i, 11);
    memoize.insert(second);
}

type TinyModel = Model<
    u8,  // AgentIndex
    u8,  // StateId
    u8,  // MessageId
    u8,  // InvalidId
    u32, // ConfigurationId
    u8,  // Payload
    u8,  // MessageOrder
    9,   // MAX_AGENTS
    18,  // MAX_MESSAGES
>;

type SmallModel = Model<
    u8,  // AgentIndex
    u8,  // StateId
    u8,  // MessageId
    u8,  // InvalidId
    u32, // ConfigurationId
    u8,  // Payload
    u8,  // MessageOrder
    19,  // MAX_AGENTS
    38,  // MAX_MESSAGES
>;

#[test]
fn test_sizes() {
    assert_eq!(
        32,
        size_of::<(
            <TinyModel as MetaModel>::Configuration,
            <TinyModel as MetaModel>::ConfigurationId
        )>()
    );
    assert_eq!(
        64,
        size_of::<(
            <SmallModel as MetaModel>::Configuration,
            <SmallModel as MetaModel>::ConfigurationId
        )>()
    );
}
