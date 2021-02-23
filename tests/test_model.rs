extern crate total_space;

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::hash::Hash;
use std::mem::size_of;
use total_space::*;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct Data {
    pub value: usize,
}

impl Display for Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[test]
fn test_memoize_value() {
    let mut memoize = Memoize::<Data, u8>::new(false);

    let first = Data { value: 17 };
    assert_eq!(first.value, 17);
    assert_eq!(
        memoize.store(first),
        Stored {
            id: 0,
            is_new: true
        }
    );
    assert_eq!(first, *memoize.get(0));

    let second = Data { value: 11 };
    assert_eq!(second.value, 11);
    assert_eq!(
        memoize.store(second),
        Stored {
            id: 1,
            is_new: true
        }
    );
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));

    assert_eq!(
        memoize.store(second),
        Stored {
            id: 1,
            is_new: false
        }
    );
    assert_eq!(
        memoize.store(first),
        Stored {
            id: 0,
            is_new: false
        }
    );
    assert_eq!(first, *memoize.get(0));
    assert_eq!(second, *memoize.get(1));
}

#[test]
fn test_memoize_display() {
    let mut memoize = Memoize::<Data, u8>::new(true);

    let first = Data { value: 17 };
    assert_eq!(first.value, 17);
    assert_eq!(memoize.store(first).id, 0);
    assert_eq!(first, *memoize.get(0));
    assert_eq!("17", memoize.display(0));

    let second = Data { value: 11 };
    assert_eq!(second.value, 11);
    assert_eq!(memoize.store(second).id, 1);
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
#[should_panic(expected = "too many (256) memoized objects")]
fn test_memoize_limit() {
    let mut memoize = Memoize::<Data, u8>::new(false);

    for value in 0..256 {
        let data = Data { value: value };
        assert_eq!(data.value, value);
        assert_eq!(memoize.store(data).id, u8::from_usize(value).unwrap());
        assert_eq!(data, *memoize.get(u8::from_usize(value).unwrap()));
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
struct Payload(u8);

impl Validated for Payload {}

impl Display for Payload {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "payload")
    }
}

// Example model which takes half a cache line in the configurations hash table.
type TinyModel = Model<
    u8,      // StateId
    u8,      // MessageId
    u8,      // InvalidId
    u32,     // ConfigurationId
    Payload, // Payload
    9,       // MAX_AGENTS
    17,      // MAX_MESSAGES
>;

// Example model which takes a single cache line in the configurations hash table.
type SmallModel = Model<
    u8,      // StateId
    u8,      // MessageId
    u8,      // InvalidId
    u32,     // ConfigurationId
    Payload, // Payload
    19,      // MAX_AGENTS
    37,      // MAX_MESSAGES
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
