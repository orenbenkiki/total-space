// Copyright (C) 2017-2019 Oren Ben-Kiki. See the LICENSE.txt
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Explore the total space of states of communicating finite state machines.

use hashbrown::HashMap;
use num_traits::Bounded;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::RwLock;

/// Memoize values and, optionally, display strings.
///
/// This assigns each unique value a (short) integer identifier. This identifier can be later used
/// to retrieve the value or the display string.
///
/// This is used extensively by the library for performance.
///
/// This uses roughly twice the amount of memory it should, because the values are stored both as
/// keys in the HashMap and also as values in the vector. In principle, with clever use of
/// RawEntryBuilder it might be possible to replace the HashMap key size to the size of an index of
/// the vector. This is probably not worth the effort though.
pub struct Memoize<T, I> {
    /// The maximal identifier allowed (restricts the number of memoized objects).
    max_id: usize,

    /// Lookup the memoized identifier for a value.
    id_by_value: HashMap<T, I>,

    /// Convert a memoized identifier to the value.
    value_by_id: Vec<T>,

    /// Optionally convert a memoized identifier to the display string.
    display_by_id: Option<Vec<String>>,
}

impl<
        T: Eq + Hash + Copy + Clone + Sized + Display,
        I: Bounded + FromPrimitive + ToPrimitive + Copy + Clone,
    > Memoize<T, I>
{
    /// Create a new memoization store.
    ///
    /// If `display`, will also memoize the display strings of the values.
    pub fn new(display: bool, stop_id: Option<I>) -> Self {
        Memoize {
            max_id: stop_id.map_or(I::to_usize(&I::max_value()).unwrap(), |id| {
                I::to_usize(&id).unwrap() - 1usize
            }),
            id_by_value: HashMap::new(),
            value_by_id: Vec::new(),
            display_by_id: {
                if display {
                    Some(Vec::new())
                } else {
                    None
                }
            },
        }
    }

    /// Given a value, ensure it is stored in the memory and return its short identifier.
    pub fn store(&mut self, value: T) -> I {
        match self.id_by_value.get(&value) {
            Some(id) => *id,
            None => {
                let size = self.id_by_value.len();
                if size > self.max_id {
                    panic!("too many memoized objects");
                }
                let id = I::from_usize(size).unwrap();
                self.id_by_value.insert(value, id);
                self.value_by_id.push(value);
                if let Some(display_by_id) = &mut self.display_by_id {
                    display_by_id.push(format!("{}", value));
                }
                id
            }
        }
    }

    /// Given a short identifier previously returned by `store`, return the full value.
    pub fn get(&self, id: I) -> &T {
        &self.value_by_id[id.to_usize().unwrap()]
    }

    /// Given a short identifier previously returned by `store`, return the display string (only if
    /// memoizing the display strings).
    pub fn display(&self, id: I) -> &str {
        &self.display_by_id.as_ref().unwrap()[id.to_usize().unwrap()]
    }
}

/// The type of the index for agent instances.
type AgentIndex = u8;

/// A value to use to indicate an invalid agent instance index.
const INVALID_AGENT_INDEX: AgentIndex = AgentIndex::max_value();

/// A message sent by an agent as part of an alternative action triggered by some event.
pub struct Emit<M> {
    /// The actual sent message.
    pub message: M,

    /// Where to send the message to.
    pub target: AgentIndex,

    /// Allow this message to replace another message from the same source to the same target, if
    /// this function returns ``true``.
    pub is_replacement: Option<fn(M) -> bool>,

    /// Whether this message needs to be delivered immediately, before any other.
    pub is_immediate: bool,

    /// Whether this message needs to be delivered only after all other ordered messages
    /// between the same source and target have been delivered.
    pub is_ordered: bool,
}

impl<M: Default> Default for Emit<M> {
    fn default() -> Self {
        Emit {
            message: Default::default(),
            target: INVALID_AGENT_INDEX,
            is_replacement: None,
            is_immediate: false,
            is_ordered: false,
        }
    }
}

/// All the messages sent by an agent as part of an alternative action triggered by some event.
pub enum Cast<M> {
    /// Do not send any messages.
    Silent,

    /// Send a single message.
    One(Emit<M>),

    /// Send two messages.
    Two(Emit<M>, Emit<M>),

    /// Send three messages.
    Three(Emit<M>, Emit<M>, Emit<M>),

    /// Send four messages.
    Four(Emit<M>, Emit<M>, Emit<M>, Emit<M>),
}

/// The type of the memoized identifier for agent states.
type StateId = u8;

/// A value to use to indicate no change in the agent state.
const SAME_STATE_ID: StateId = StateId::max_value();

/// A value to use to indicate the event is deferred.
const DEFER_STATE_ID: StateId = StateId::max_value() - 1;

/// An action that may be taken by an agent as a response to some event.
///
/// This specifies the next state the agent will be at (or one of the special values `SAME_STATE_ID`
/// and `DEFER_STATE_ID`), and the messages that the agent sends, if any.
pub struct Do<M>(StateId, Cast<M>);

impl<M> Do<M> {
    /// Return an action that indicates the agent silently ignores the event.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Response.Ignore` value.
    pub fn ignore() -> Self {
        Do(SAME_STATE_ID, Cast::Silent)
    }

    /// Return an action that indicates the agent defers handling the event.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Response.Defer` value.
    ///
    /// This is only allowed if the agent's `state_is_deferring`, waiting for
    /// specific message(s) to resume normal operations.
    pub fn defer() -> Self {
        Do(DEFER_STATE_ID, Cast::Silent)
    }
}

/// The response from an agent on some event.
pub enum Response<M> {
    /// Ignore the event.
    ///
    /// This has the same effect as `One(Do.ignore())`.
    Ignore,

    /// Defer handling the event.
    ///
    /// This has the same effect as `One(Do.defer())`.
    Defer,

    /// A single action (deterministic).
    One(Do<M>),

    /// One of two alternative actions (non-deterministic).
    Two(Do<M>, Do<M>),

    /// One of three alternative actions (non-deterministic).
    Three(Do<M>, Do<M>, Do<M>),

    /// One of four alternative actions (non-deterministic).
    Four(Do<M>, Do<M>, Do<M>, Do<M>),
}

/// The type of the memoized identifier for invalid conditions.
type InvalidId = u32;

/// A value to use to indicate there is no invalid condition.
const VALID_ID: InvalidId = InvalidId::max_value();

/// A trait describing a set of agents of some type.
pub trait AgentType<M> {
    /// Create a new instance of the agent type, sharing the memoized messages and invalid reasons.
    ///
    /// This should create the initial state of the agent with `StateId` zero.
    fn new(invalids: Arc<RwLock<Memoize<String, InvalidId>>>) -> Self;

    /// The name of the type of the agents.
    fn name(&self) -> &str;

    /// Return the actions that may be taken by an agent instance with some state when receiving a
    /// message.
    fn message_response(
        &self,
        agent_index: AgentIndex,
        state_id: StateId,
        message: &M,
    ) -> Response<M>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn time_response(&self, agent_index: AgentIndex, state_id: StateId) -> Response<M>;

    /// Whether any agent in the state is deferring messages.
    fn state_is_deferring(&self, state_id: StateId) -> bool;

    /// Display the state.
    ///
    /// The format of the display must be either `<state-name>` if the state is a simple enum, or
    /// `<state-name>(<state-data>)` if the state contains additional data. The `Debug` of the state
    /// might be acceptable as-is, but typically it is better to get rid or shorten the explicit
    /// field names, and/or format their values in a more compact form.
    fn state_display(&self, state_id: StateId) -> &str;

    /// Return the short state name.
    fn state_name(&self, state_id: StateId) -> &str;
}

/// Specify how a message is ordered relative to other messages between the same source and target
/// agents.
///
/// Special values are `UNORDERED_MESSAGE` and `IMMEDIATE_MESSAGE`.
type MessageOrder = u8;

/// Specify that a message is not ordered relative to the other messages between the same source and
/// target agents.
const UNORDERED_MESSAGE: MessageOrder = MessageOrder::max_value();

/// Specify that the message needs to be delivered immediately, bypassing any other message in the
/// system.
// const IMMEDIATE_MESSAGE: MessageOrder = MessageOrder::max_value() - 1;

/// A message in-flight between agents.
pub struct InFlightMessage<M> {
    /// The actual message, if any.
    pub message: M,

    /// The replaced message, if any.
    pub replaced_message: Option<M>,

    /// The target agent index.
    ///
    /// We know the source agent index by the index of the in-flight message in the
    /// `in_flight_messages` array.
    pub target_agent_index: AgentIndex,

    /// How the message is ordered relative to other messages between the same source and target
    /// agents.
    pub order: MessageOrder,
}

impl<M: Default> Default for InFlightMessage<M> {
    fn default() -> Self {
        InFlightMessage {
            message: Default::default(),
            replaced_message: None,
            target_agent_index: INVALID_AGENT_INDEX,
            order: UNORDERED_MESSAGE,
        }
    }
}

/// The type of the memoized identifier for in-flight messages.
pub type InFlightMessageId = u8;

/// A value to use to indicate an invalid in-flight message.
const INVALID_IN_FLIGHT_MESSAGE_ID: InFlightMessageId = InFlightMessageId::max_value();

/// The maximal number of agents in a system.
const MAX_CONFIGURATION_AGENTS: usize = 10;

/// The maximal number of in-flight messages from a single agent.
const MAX_AGENT_IN_FLIGHT_MESSAGES: usize = 4;

/// The maximal invalid conditions in a configuration.
const MAX_CONFIGURATION_INVALIDS: usize = 2;

/// The type of the memoized identifier for configurations.
pub type ConfigurationId = u32;

/// A value to use to indicate an invalid configuration identifier.
// const INVALID_CONFIGURATION_ID: ConfigurationId = ConfigurationId::max_value();

/// A complete system configuration.
///
/// We will have a *lot* of these, so keeping their size down and avoiding heap memory as much as
/// possible is critical. The maximal sizes were chosen so that the configuration plus its memoized
/// identifier will fit together inside exactly one cache lines, which should make this more
/// cache-friendly when placed inside a hash table.
pub struct Configuration {
    /// The messages sent by each agent.
    pub in_flight_message_ids:
        [[InFlightMessageId; MAX_AGENT_IN_FLIGHT_MESSAGES]; MAX_CONFIGURATION_AGENTS],

    /// The state of each agent.
    pub agent_state_ids: [StateId; MAX_CONFIGURATION_AGENTS],

    /// The invalid conditions, if any.
    pub invalids: [InvalidId; MAX_CONFIGURATION_INVALIDS],
}

impl Default for Configuration {
    fn default() -> Self {
        Configuration {
            agent_state_ids: [INVALID_AGENT_INDEX; MAX_CONFIGURATION_AGENTS],
            in_flight_message_ids: [[INVALID_IN_FLIGHT_MESSAGE_ID; MAX_AGENT_IN_FLIGHT_MESSAGES];
                MAX_CONFIGURATION_AGENTS],
            invalids: [VALID_ID; MAX_CONFIGURATION_INVALIDS],
        }
    }
}

// BNF for configuration display (`[ ... ]` stands for optional).
//
// CONFIGURATION := AGENT & AGENT & ...
//                  [ | IN_FLIGHT_MESSAGE & IN_FLIGHT_MESSAGE & ... ]
//                  [ ! INVALID & INVALID & ... ]
//
// AGENT := TYPE[-INDEX] # STATE_NAME[(STATE_DATA)]
//
// IN_FLIGHT_MESSAGE := SOURCE_TYPE[-INDEX} ->
//                      [REPLACED_NAME[(REPLACED_DATA) => ]
//                      [@INT or *] NAME(DATA) ->
//                      TARGET_TYPE[-INDEX]
//
// INVALID := KIND is INVALID because REASON

#[cfg(test)]
use std::mem::size_of;

#[test]
fn test_configuration_hash_entry_is_cache_line() {
    assert_eq!(64, size_of::<(Configuration, ConfigurationId)>());
}
