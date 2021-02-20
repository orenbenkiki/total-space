use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use std::collections::HashMap;
use std::hash::Hash;
use std::vec::Vec;

pub struct Memoize<T, I> {
    by_value: HashMap<T, I>,
    by_index: Vec<T>,
}

impl<T: Eq + Hash + Copy + Clone, I: FromPrimitive + ToPrimitive + Copy> Memoize<T, I> {
    pub fn new() -> Self {
        Memoize {
            by_value: HashMap::new(),
            by_index: Vec::new(),
        }
    }

    pub fn store(&mut self, value: T) -> I {
        match self.by_value.get(&value) {
            Some(index) => *index,
            None => {
                let index = I::from_usize(self.by_value.len()).unwrap();
                self.by_value.insert(value, index);
                self.by_index.push(value);
                index
            }
        }
    }

    pub fn get(&self, index: I) -> T {
        self.by_index[index.to_usize().unwrap()]
    }
}

impl<T: Eq + Hash + Copy + Clone, I: FromPrimitive + ToPrimitive + Copy> Default for Memoize<T, I> {
    fn default() -> Self {
        Self::new()
    }
}
