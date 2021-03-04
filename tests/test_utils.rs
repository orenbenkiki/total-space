use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
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

index_type! { DataId, u8 }

#[test]
fn test_memoize_value() {
    let mut memoize = Memoize::<Data, DataId>::new(false);

    let first = Data { value: 17 };
    assert_eq!(first.value, 17);
    assert_eq!(
        memoize.store(first, None),
        Stored {
            id: DataId::from_usize(0),
            is_new: true
        }
    );
    assert_eq!(first, *memoize.get(DataId::from_usize(0)));

    let second = Data { value: 11 };
    assert_eq!(second.value, 11);
    assert_eq!(
        memoize.store(second, None),
        Stored {
            id: DataId::from_usize(1),
            is_new: true
        }
    );
    assert_eq!(first, *memoize.get(DataId::from_usize(0)));
    assert_eq!(second, *memoize.get(DataId::from_usize(1)));

    assert_eq!(
        memoize.store(second, None),
        Stored {
            id: DataId::from_usize(1),
            is_new: false
        }
    );
    assert_eq!(
        memoize.store(first, None),
        Stored {
            id: DataId::from_usize(0),
            is_new: false
        }
    );
    assert_eq!(first, *memoize.get(DataId::from_usize(0)));
    assert_eq!(second, *memoize.get(DataId::from_usize(1)));
}

#[test]
fn test_memoize_display() {
    let mut memoize = Memoize::<Data, DataId>::new(true);

    let first = Data { value: 17 };
    assert_eq!(first.value, 17);
    assert_eq!(
        memoize.store(first, Some("17!".to_string())).id,
        DataId::from_usize(0)
    );
    assert_eq!(first, *memoize.get(DataId::from_usize(0)));
    assert_eq!("17!", memoize.display(DataId::from_usize(0)));

    let second = Data { value: 11 };
    assert_eq!(second.value, 11);
    assert_eq!(
        memoize.store(second, Some("11!".to_string())).id,
        DataId::from_usize(1)
    );
    assert_eq!(first, *memoize.get(DataId::from_usize(0)));
    assert_eq!(second, *memoize.get(DataId::from_usize(1)));
    assert_eq!("17!", memoize.display(DataId::from_usize(0)));
    assert_eq!("11!", memoize.display(DataId::from_usize(1)));

    assert_eq!(*memoize.lookup(&second).unwrap(), DataId::from_usize(1));
    assert_eq!(*memoize.lookup(&first).unwrap(), DataId::from_usize(0));
    assert_eq!(first, *memoize.get(DataId::from_usize(0)));
    assert_eq!(second, *memoize.get(DataId::from_usize(1)));
    assert_eq!("17!", memoize.display(DataId::from_usize(0)));
    assert_eq!("11!", memoize.display(DataId::from_usize(1)));
}

#[test]
#[should_panic(expected = "too many (256) memoized objects")]
fn test_memoize_limit() {
    let mut memoize = Memoize::<Data, DataId>::new(false);

    for value in 0..256 {
        let data = Data { value: value };
        assert_eq!(data.value, value);
        assert_eq!(memoize.store(data, None).id, DataId::from_usize(value));
        assert_eq!(data, *memoize.get(DataId::from_usize(value)));
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct Payload(u8);

impl Validated for Payload {}

impl Name for Payload {
    fn name(&self) -> &'static str {
        "payload"
    }
}

impl Display for Payload {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "payload")
    }
}

index_type! { StateId, u8 }
index_type! { MessageId, u8 }
index_type! { InvalidId, u8 }
index_type! { ConfigurationId, u32 }

// Example model which takes half a cache line in the configurations hash table.
type TinyModel = Model<
    StateId,
    MessageId,
    InvalidId,
    ConfigurationId,
    Payload,
    6,  // MAX_AGENTS
    14, // MAX_MESSAGES
>;

// Example model which takes a single cache line in the configurations hash table.
type SmallModel = Model<
    StateId,
    MessageId,
    InvalidId,
    ConfigurationId,
    Payload, // Payload
    14,      // MAX_AGENTS
    30,      // MAX_MESSAGES
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
