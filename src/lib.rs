// total-space is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3, as published by the Free Software Foundation.
//
// total-space is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with total-space If not,
// see <http://www.gnu.org/licenses/>.

//! Explore the total space of states of communicating finite state machines.
//!
//! This is just the API reference documentation; see the
//! [README](https://github.com/orenbenkiki/total-space) for an overview and links to example code.

#![deny(missing_docs)]

mod agents;
mod claps;
mod configurations;
mod diagrams;
mod macros;
mod memoize;
mod messages;
mod models;
mod reactions;
mod utilities;

pub use crate::agents::*;
pub use crate::claps::*;
pub use crate::configurations::*;
pub use crate::messages::*;
pub use crate::models::*;
pub use crate::reactions::*;
pub use crate::utilities::*;
