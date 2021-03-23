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

impl_default_by_value! { Data = Data { value: usize::max_value() } }
impl_display_by_patched_debug! { Data }

index_type! { DataId, u8 }

#[test]
fn memoize_value() {
    let memoize = Memoize::<Data, DataId>::new(2, 2);

    let first = Data { value: 17 };
    assert_eq!(first.value, 17);
    assert_eq!(
        memoize.store(first),
        Stored {
            id: DataId::from_usize(0),
            is_new: true
        }
    );
    assert_eq!(first, memoize.get(DataId::from_usize(0)));

    let second = Data { value: 11 };
    assert_eq!(second.value, 11);
    assert_eq!(
        memoize.store(second),
        Stored {
            id: DataId::from_usize(1),
            is_new: true
        }
    );
    assert_eq!(first, memoize.get(DataId::from_usize(0)));
    assert_eq!(second, memoize.get(DataId::from_usize(1)));

    assert_eq!(
        memoize.store(second),
        Stored {
            id: DataId::from_usize(1),
            is_new: false
        }
    );
    assert_eq!(
        memoize.store(first),
        Stored {
            id: DataId::from_usize(0),
            is_new: false
        }
    );
    assert_eq!(first, memoize.get(DataId::from_usize(0)));
    assert_eq!(second, memoize.get(DataId::from_usize(1)));
}

#[test]
#[should_panic(expected = "too many (5) memoized objects")]
fn memoize_limit() {
    let memoize = Memoize::<Data, DataId>::new(2, 4);

    for value in 0..5 {
        let data = Data { value: value };
        assert_eq!(data.value, value);
        assert_eq!(memoize.store(data).id, DataId::from_usize(value));
        assert_eq!(data, memoize.get(DataId::from_usize(value)));
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
struct Payload(u8);

impl_default_by_value! { Payload = Payload(255u8) }
impl_display_by_patched_debug! { Payload }

impl Name for Payload {
    fn name(&self) -> String {
        "payload".to_string()
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
fn sizes() {
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
