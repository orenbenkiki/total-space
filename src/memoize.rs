use crate::utilities::*;

use hashbrown::HashMap;
use std::cmp::min;
use std::fmt::Debug;
use std::hash::Hash;

/// Result of a memoization store operation.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct Stored<I: IndexLike> {
    /// The short identifier the data is stored under.
    pub id: I,

    /// Whether this operation stored previously unseen data.
    pub is_new: bool,
}

/// Memoize values and, optionally, display strings.
///
/// This assigns each unique value a (short) integer identifier. This identifier can be later used
/// to retrieve the value.
///
/// This is used extensively by the library for performance.
///
/// This uses roughly twice the amount of memory it should, because the values are stored both as
/// keys in the HashMap and also as values in the vector. In principle, with clever use of
/// RawEntryBuilder it might be possible to replace the HashMap key size to the size of an index of
/// the vector.
pub struct Memoize<T: KeyLike, I: IndexLike> {
    /// Lookup the memoized identifier for a value.
    id_by_value: HashMap<T, I>,

    /// The maximal number of identifiers to generate.
    max_count: usize,

    /// Convert a memoized identifier to the value.
    value_by_id: Vec<T>,
}

impl<T: KeyLike + Default, I: IndexLike> Memoize<T, I> {
    /// Create a new memoization store.
    pub fn new(max_count: usize) -> Self {
        Self::with_capacity(max_count, max_count)
    }

    /// Create a new memoization store with some capacity.
    pub fn with_capacity(max_count: usize, capacity: usize) -> Self {
        let capacity = min(capacity, max_count);
        Self {
            max_count,
            id_by_value: HashMap::with_capacity(capacity),
            value_by_id: Vec::with_capacity(capacity),
        }
    }

    /// The number of allocated identifiers.
    pub fn len(&self) -> usize {
        self.value_by_id.len()
    }

    /// Whether we have no identifiers stored at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Given a value that may or may not exist in the memory, ensure it exists it and return its
    /// short identifier.
    pub fn store(&mut self, value: T) -> Stored<I> {
        let mut is_new = false;
        let next_id = self.value_by_id.len();
        let max_count = self.max_count;

        let id = *self.id_by_value.entry(value).or_insert_with(|| {
            is_new = true;
            assert!(
                next_id < max_count,
                "too many ({}) memoized objects",
                next_id + 1 // NOT TESTED
            );
            I::from_usize(next_id)
        });

        if is_new {
            self.value_by_id.push(value);
        }

        Stored { id, is_new }
    }

    /// Given a short identifier previously returned by `store`, return the full value.
    pub fn get(&self, id: I) -> T {
        debug_assert!(id.to_usize() < self.len());
        self.value_by_id[id.to_usize()]
    }
}
