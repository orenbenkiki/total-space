//! Odds and ends.

use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

/// A trait for data having a short name.
pub trait Name {
    fn name(&self) -> String;
}

/// A trait for anything we use as a key in a HashMap.
pub trait KeyLike: Eq + Hash + Copy + Debug + Sized {}

/// A trait for data we pass around in the model.
pub trait DataLike: KeyLike + Display + Name + Default {}

/// A trait for anything we use as a zero-based index.
pub trait IndexLike: KeyLike + PartialOrd + Ord {
    /// Convert a `usize` to the index.
    fn from_usize(value: usize) -> Self;

    /// Convert the index to a `usize`.
    fn to_usize(&self) -> usize;

    /// The invalid (maximal) value.
    fn invalid() -> Self;

    /// Decrement the value.
    fn decr(&mut self) {
        let value = self.to_usize();
        assert!(value > 0);
        *self = Self::from_usize(value - 1);
    }

    /// Increment the value.
    fn incr(&mut self) {
        assert!(self.is_valid());
        let value = self.to_usize();
        *self = Self::from_usize(value + 1);
        assert!(self.is_valid());
    }

    /// Is a valid value (not the maximal value).
    fn is_valid(&self) -> bool {
        *self != Self::invalid()
    }
}

pub(crate) const RIGHT_ARROW: &str = "&#8594;";

pub(crate) const RIGHT_DOUBLE_ARROW: &str = "&#8658;";

pub(crate) fn calculate_string_hash(string: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    string.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn calculate_strings_hash(first: &str, second: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    first.hash(&mut hasher);
    second.hash(&mut hasher);
    hasher.finish()
}
