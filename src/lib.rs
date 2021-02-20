// Copyright (C) 2017-2019 Oren Ben-Kiki. See the LICENSE.txt
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Explore the total space of states of communicating finite state machines.

use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

/// Memoize values and, optionally, display strings.
///
/// This assigns each unique value a (short) integer index. This index can be later used to retrieve
/// the value or the display string.
///
/// This is used extensively by the library for performance.
pub struct Memoize<T, I> {
    index_by_value: HashMap<T, I>,
    value_by_index: Vec<T>,
    display_by_index: Option<Vec<String>>,
}

impl<
        T: Eq + Hash + Copy + Clone + Sized + Display,
        I: FromPrimitive + ToPrimitive + Copy + Clone,
    > Memoize<T, I>
{
    /// Create a new memoization store.
    ///
    /// If `display`, will also memoize the display strings of the values.
    pub fn new(display: bool) -> Self {
        Memoize {
            index_by_value: HashMap::new(),
            value_by_index: Vec::new(),
            display_by_index: {
                if display {
                    Some(Vec::new())
                } else {
                    None
                }
            },
        }
    }

    /// Given a value, ensure it is stored in the memory and return its short index.
    pub fn store(&mut self, value: T) -> I {
        match self.index_by_value.get(&value) {
            Some(index) => *index,
            None => {
                let index = I::from_usize(self.index_by_value.len()).unwrap();
                self.index_by_value.insert(value, index);
                self.value_by_index.push(value);
                if let Some(display_by_index) = &mut self.display_by_index {
                    display_by_index.push(format!("{}", value));
                }
                index
            }
        }
    }

    /// Given a short index previously returned by `store`, return the full value.
    pub fn get(&self, index: I) -> &T {
        &self.value_by_index[index.to_usize().unwrap()]
    }

    /// Given a short index previously returned by `store`, return the display string (only if
    /// memoizing the display strings).
    pub fn display(&self, index: I) -> &str {
        &self.display_by_index.as_ref().unwrap()[index.to_usize().unwrap()]
    }
}
