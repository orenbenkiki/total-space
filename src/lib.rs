// Copyright (C) 2021 Oren Ben-Kiki <oren@ben-kiki.org>
//
// This file is part of cargo-coverage-annotations.
//
// cargo-coverage-annotations is free software: you can redistribute it and/or
// modify it under the terms of the GNU General Public License, version 3, as
// published by the Free Software Foundation.
//
// cargo-coverage-annotations is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// cargo-coverage-annotations. If not, see <http://www.gnu.org/licenses/>.

//! Explore the total space of states of communicating finite state machines.

// TODO break vs. take_while

#![feature(trait_alias)]

use hashbrown::HashMap;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
// TODO use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cell::RefCell;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::RwLock;

/// A trait for anything we use as a key in a HashMap.
pub trait KeyLike = Eq + Hash + Copy + Debug + Sized + Send + Sync;

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

/// A trait for data having a short name.
pub trait Name {
    fn name(&self) -> &'static str;
}

/// A macro for implementing `Name` for enums annotated by `strum::IntoStaticStr`.
///
/// This should become unnecessary once `IntoStaticStr` allows converting a reference to a static
/// string, see `<https://github.com/Peternator7/strum/issues/142>`.
#[macro_export]
macro_rules! impl_name_for_into_static_str {
    ($name:ident) => {
        impl total_space::Name for $name {
            fn name(&self) -> &'static str {
                self.into()
            }
        }
    };
}

/// A trait for data that has a short name (via `AsRef<&'static str>`) and a full display name (via
/// `Display`).
pub trait Named = Display + Name;

// BEGIN MAYBE TESTED

/// Result of a memoization store operation.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct Stored<I> {
    /// The short identifier the data is stored under.
    pub id: I,

    /// Whether this operation stored previously unseen data.
    pub is_new: bool,
}

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
/// the vector.
#[derive(Debug)]
pub struct Memoize<T, I> {
    /// Lookup the memoized identifier for a value.
    id_by_value: HashMap<T, I>,

    /// Convert a memoized identifier to the value.
    value_by_id: Vec<T>,

    /// Optionally convert a memoized identifier to the display string.
    display_by_id: Option<Vec<String>>,
}

// END MAYBE TESTED

impl<T: KeyLike, I: IndexLike> Memoize<T, I> {
    /// Create a new memoization store.
    ///
    /// If `display`, will also memoize the display strings of the values.
    pub fn new(display: bool) -> Self {
        Memoize {
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

    /// The number of allocated identifiers.
    pub fn usize(&self) -> usize {
        self.id_by_value.len()
    }

    /// The number of allocated identifiers.
    pub fn size(&self) -> I {
        I::from_usize(self.usize())
    }

    /// Given a value, look it up in the memory.
    pub fn lookup(&self, value: &T) -> Option<&I> {
        self.id_by_value.get(value)
    }

    /// Given a value that may or may not exist in the memory, ensure it exists it and return its
    /// short identifier.
    pub fn store(&mut self, value: T, display: Option<String>) -> Stored<I> {
        match self.lookup(&value) {
            Some(id) => Stored {
                id: *id,
                is_new: false,
            },
            None => {
                if self.usize() >= I::invalid().to_usize() {
                    panic!("too many ({}) memoized objects", self.usize() + 1);
                }

                let id = self.size();
                self.id_by_value.insert(value, id);
                self.value_by_id.push(value);
                if let Some(ref mut display_by_id) = &mut self.display_by_id {
                    display_by_id.push(display.unwrap());
                } else {
                    debug_assert!(display.is_none());
                }

                Stored { id, is_new: true }
            }
        }
    }

    /// Given a short identifier previously returned by `store`, return the full value.
    pub fn get(&self, id: I) -> &T {
        &self.value_by_id[id.to_usize()]
    }

    /// Given a short identifier previously returned by `store`, return the display string (only if
    /// memoizing the display strings).
    pub fn display(&self, id: I) -> &str {
        &self.display_by_id.as_ref().unwrap()[id.to_usize()]
    }
}

// BEGIN MAYBE TESTED

/// A message sent by an agent as part of an alternative action triggered by some event.
#[derive(PartialEq, Eq)]
pub enum Emit<Payload> {
    /// Send a message that will be delivered immediately, before any other message is delivered.
    Immediate(Payload, usize),

    /// Send an unordered message, which will be delivered at any order relative to the other
    /// unordered messages.
    Unordered(Payload, usize),

    /// Send an ordered message, which will be delivered after delivering all previous ordered
    /// messages from this source agent to the same target agent.
    Ordered(Payload, usize),

    /// Send an immediate message that will replace the single in-flight message accepted by the
    /// callback, or be created as a new message is the callback accepts `None`.
    ImmediateReplacement(fn(Option<Payload>) -> bool, Payload, usize),

    /// Send an unordered message that will replace the single in-flight message accepted by the
    /// callback, or be created as a new message is the callback accepts `None`.
    UnorderedReplacement(fn(Option<Payload>) -> bool, Payload, usize),

    /// Send an ordered message that will replace the single in-flight message accepted by the
    /// callback, or be created as a new message is the callback accepts `None`.
    OrderedReplacement(fn(Option<Payload>) -> bool, Payload, usize),
}

/// Specify an action the agent may take when handling an event.
#[derive(PartialEq, Eq)]
pub enum Action<State, Payload> {
    /// Defer the event, keep the state the same, do not send any messages.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Reaction.Defer` value.
    ///
    /// This is only allowed if the agent's `state_is_deferring`, waiting for
    /// specific message(s) to resume normal operations.
    Defer,

    /// Consume (ignore) the event, keep the state the same, do not send any messages.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Reaction.Ignore` value.
    Ignore,

    /// Consume (handle) the event, change the agent state, do not send any messages.
    Change(State),

    /// Consume (handle) the event, keep the state the same, send a single message.
    Send1(Emit<Payload>),

    /// Consume (handle) the event, change the agent state, send a single message.
    ChangeAndSend1(State, Emit<Payload>),

    /// Consume (handle) the event, keep the state the same, send two messages.
    Send2(Emit<Payload>, Emit<Payload>),

    /// Consume (handle) the event, change the agent state, send two messages.
    ChangeAndSend2(State, Emit<Payload>, Emit<Payload>),

    /// Consume (handle) the event, keep the state the same, send three messages.
    Send3(Emit<Payload>, Emit<Payload>, Emit<Payload>),

    /// Consume (handle) the event, change the agent state, send three messages.
    ChangeAndSend3(State, Emit<Payload>, Emit<Payload>, Emit<Payload>),

    /// Consume (handle) the event, keep the state the same, send four messages.
    Send4(Emit<Payload>, Emit<Payload>, Emit<Payload>, Emit<Payload>),

    /// Consume (handle) the event, change the agent state, send four messages.
    ChangeAndSend4(
        State,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
    ),
}

/// The reaction of an agent to an event.
#[derive(PartialEq, Eq)]
pub enum Reaction<State, Payload> {
    /// Indicate an unexpected event.
    Unexpected,

    /// Defer handling the event.
    ///
    /// This has the same effect as `DoOne(Action.Defer)`.
    Defer,

    /// Ignore the event.
    ///
    /// This has the same effect as `DoOne(Action.Ignore)`.
    Ignore,

    /// A single action (deterministic).
    Do1(Action<State, Payload>),

    /// One of two alternative actions (non-deterministic).
    Do1Of2(Action<State, Payload>, Action<State, Payload>),

    /// One of three alternative actions (non-deterministic).
    Do1Of3(
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
    ),

    /// One of four alternative actions (non-deterministic).
    Do1Of4(
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
    ),
}

// END MAYBE TESTED

/// A trait describing a set of agents of some type.
pub trait AgentType<StateId, MessageId, Payload>: Name + Debug {
    /// Whether this type supports multiple instances.
    ///
    /// If true, the count will always be 1.
    fn is_indexed(&self) -> bool;

    /// The number of agents of this type that will be used in the system.
    fn count(&self) -> usize;

    /// Return the actions that may be taken by an agent instance with some state when receiving a
    /// message.
    fn receive_message(
        &self,
        instance: usize,
        state_id: StateId,
        payload: &Payload,
    ) -> Reaction<StateId, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn pass_time(&self, instance: usize, state_id: StateId) -> Reaction<StateId, Payload>;

    /// Whether any agent in the state is deferring messages.
    fn state_is_deferring(&self, state_id: StateId) -> bool;

    /// The maximal number of messages sent by an agent which may be in-flight when it is in the
    /// state.
    fn state_max_in_flight_messages(&self, state_id: StateId) -> Option<usize>;

    /// Display the state.
    ///
    /// The format of the display must be either `<state-name>` if the state is a simple enum, or
    /// `<state-name>(<state-data>)` if the state contains additional data. The `Debug` of the state
    /// might be acceptable as-is, but typically it is better to get rid or shorten the explicit
    /// field names, and/or format their values in a more compact form.
    ///
    /// Since we are caching the display string, we can't return it; instead, you'll need to provide
    /// a work function that will process it.
    fn state_display(&self, state_id: StateId, callback: Box<dyn FnOnce(&str)>);

    /// Return the short state name.
    fn state_name(&self, state_id: StateId) -> &'static str;
}

// BEGIN MAYBE TESTED

/// The data we need to implement an agent type.
///
/// This should be placed in a `Singleton` to allow the agent states to get services from it.
#[derive(Debug)]
pub struct AgentTypeData<State, StateId, Payload> {
    /// Memoization of the agent states.
    states: Arc<RwLock<Memoize<State, StateId>>>,

    /// The name of the agent type.
    name: &'static str,

    /// Whether this type supports multiple instances.
    is_indexed: bool,

    /// The number of instances of this type we'll be using in the system.
    count: usize,

    /// Trick the compiler into thinking we have a field of type Payload.
    _payload: PhantomData<Payload>,
}

// END MAYBE TESTED

impl<
        State: KeyLike + Validated + Named + Default,
        StateId: IndexLike,
        Payload: KeyLike + Validated + Named,
    > AgentTypeData<State, StateId, Payload>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(name: &'static str, is_indexed: bool, count: usize) -> Self {
        assert!(
            !is_indexed || count == 1,
            "specifying multiple instances for a non-indexed agent type"
        );
        let states = Arc::new(RwLock::new(Memoize::new(true)));
        let default_state: State = Default::default();
        states
            .write()
            .unwrap()
            .store(default_state, Some(format!("{}", default_state)));
        AgentTypeData {
            name,
            count,
            is_indexed,
            states,
            _payload: PhantomData,
        }
    }
}

// TODO: Allow container agent to access state of part agent

/// A trait for a single agent state.
pub trait AgentState<
    State: KeyLike + Validated + Named + Default,
    Payload: KeyLike + Validated + Named,
>
{
    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// message.
    fn receive_message(&self, instance: usize, payload: &Payload) -> Reaction<State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn pass_time(&self, instance: usize) -> Reaction<State, Payload>;

    // BEGIN NOT TESTED

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self) -> bool {
        false
    }

    // END NOT TESTED

    /// The maximal number of messages sent by this agent which may be in-flight when it is in this
    /// state.
    fn max_in_flight_messages(&self) -> Option<usize> {
        None
    }
}

pub trait Validated {
    /// If this object is invalid, return why.
    fn invalid(&self) -> Option<&'static str> {
        None
    }
}

impl<
        State: KeyLike + Validated + Named + Default + AgentState<State, Payload>,
        StateId: IndexLike,
        Payload: KeyLike + Validated + Named,
    > AgentTypeData<State, StateId, Payload>
{
    fn translate_reaction(&self, reaction: Reaction<State, Payload>) -> Reaction<StateId, Payload> {
        match reaction {
            Reaction::Unexpected => Reaction::Unexpected,
            Reaction::Ignore => Reaction::Ignore,
            Reaction::Defer => Reaction::Defer, // NOT TESTED
            Reaction::Do1(action) => Reaction::Do1(self.translate_action(action)),
            // BEGIN NOT TESTED
            Reaction::Do1Of2(action1, action2) => Reaction::Do1Of2(
                self.translate_action(action1),
                self.translate_action(action2),
            ),
            Reaction::Do1Of3(action1, action2, action3) => Reaction::Do1Of3(
                self.translate_action(action1),
                self.translate_action(action2),
                self.translate_action(action3),
            ),
            Reaction::Do1Of4(action1, action2, action3, action4) => Reaction::Do1Of4(
                self.translate_action(action1),
                self.translate_action(action2),
                self.translate_action(action3),
                self.translate_action(action4),
            ),
            // END NOT TESTED
        }
    }

    fn translate_action(&self, action: Action<State, Payload>) -> Action<StateId, Payload> {
        match action {
            Action::Defer => Action::Defer,

            Action::Ignore => Action::Ignore, // NOT TESTED
            Action::Change(state) => Action::Change(self.translate_state(state)),

            Action::Send1(emit) => Action::Send1(emit), // NOT TESTED
            Action::ChangeAndSend1(state, emit) => {
                Action::ChangeAndSend1(self.translate_state(state), emit)
            }

            // BEGIN NOT TESTED
            Action::Send2(emit1, emit2) => Action::Send2(emit1, emit2),
            Action::ChangeAndSend2(state, emit1, emit2) => {
                Action::ChangeAndSend2(self.translate_state(state), emit1, emit2)
            }

            Action::Send3(emit1, emit2, emit3) => Action::Send3(emit1, emit2, emit3),
            Action::ChangeAndSend3(state, emit1, emit2, emit3) => {
                Action::ChangeAndSend3(self.translate_state(state), emit1, emit2, emit3)
            }

            Action::Send4(emit1, emit2, emit3, emit4) => Action::Send4(emit1, emit2, emit3, emit4),
            Action::ChangeAndSend4(state, emit1, emit2, emit3, emit4) => {
                Action::ChangeAndSend4(self.translate_state(state), emit1, emit2, emit3, emit4)
            } // END NOT TESTED
        }
    }

    fn translate_state(&self, state: State) -> StateId {
        if let Some(state_id) = self.states.read().unwrap().lookup(&state) {
            return *state_id;
        }
        self.states
            .write()
            .unwrap()
            .store(state, Some(format!("{}", state)))
            .id
    }
}

impl<
        State: KeyLike + Validated + Named + Default + AgentState<State, Payload>,
        StateId: IndexLike,
        Payload: KeyLike + Validated + Named,
    > Name for AgentTypeData<State, StateId, Payload>
{
    fn name(&self) -> &'static str {
        &self.name
    }
}

impl<
        State: KeyLike + Validated + Named + Default + AgentState<State, Payload>,
        StateId: IndexLike,
        MessageId: IndexLike,
        Payload: KeyLike + Validated + Named,
    > AgentType<StateId, MessageId, Payload> for AgentTypeData<State, StateId, Payload>
{
    fn is_indexed(&self) -> bool {
        self.is_indexed
    }

    fn count(&self) -> usize {
        self.count
    }

    fn receive_message(
        &self,
        instance: usize,
        state_id: StateId,
        payload: &Payload,
    ) -> Reaction<StateId, Payload> {
        let reaction = self
            .states
            .read()
            .unwrap()
            .get(state_id)
            .receive_message(instance, payload);
        self.translate_reaction(reaction)
    }

    fn pass_time(&self, instance: usize, state_id: StateId) -> Reaction<StateId, Payload> {
        let reaction = self
            .states
            .read()
            .unwrap()
            .get(state_id)
            .pass_time(instance);
        self.translate_reaction(reaction)
    }

    // BEGIN NOT TESTED
    fn state_is_deferring(&self, state_id: StateId) -> bool {
        self.states.read().unwrap().get(state_id).is_deferring()
    }
    // END NOT TESTED

    fn state_max_in_flight_messages(&self, state_id: StateId) -> Option<usize> {
        self.states
            .read()
            .unwrap()
            .get(state_id)
            .max_in_flight_messages()
    }

    // BEGIN NOT TESTED
    fn state_display(&self, state_id: StateId, callback: Box<dyn FnOnce(&str)>) {
        (callback)(self.states.read().unwrap().display(state_id))
    }

    fn state_name(&self, state_id: StateId) -> &'static str {
        self.states.read().unwrap().get(state_id).name()
    }
    // END NOT TESTED
}

// BEGIN MAYBE TESTED

/// A macro for implementing some `IndexLike` type.
///
/// This should be concerted to a derive macro.
#[macro_export]
macro_rules! index_type {
    ($name:ident, $type:ident) => {
        #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
        pub struct $name($type);

        impl total_space::IndexLike for $name {
            fn from_usize(value: usize) -> Self {
                $name($type::from_usize(value).unwrap())
            }

            fn to_usize(&self) -> usize {
                let $name(value) = self;
                $type::to_usize(value).unwrap()
            }

            fn invalid() -> Self {
                $name($type::max_value())
            }
        }

        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
                write!(formatter, "{}", self.to_usize())
            }
        }
    };
}

/// The type of the index of a message in the configuration.
///
/// "A total of 256 in-flight messages should be enough for everybody" ;-)
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
pub struct MessageIndex(u8);

// END MAYBE TESTED

impl IndexLike for MessageIndex {
    fn from_usize(value: usize) -> MessageIndex {
        MessageIndex(u8::from_usize(value).unwrap())
    }

    fn to_usize(&self) -> usize {
        let MessageIndex(value) = self;
        u8::to_usize(value).unwrap()
    }

    // BEGIN MAYBE TESTED
    fn invalid() -> MessageIndex {
        MessageIndex(u8::max_value())
    }
    // END MAYBE TESTED
}

// BEGIN MAYBE TESTED

impl Display for MessageIndex {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "{}", self.to_usize())
    }
}

/// Possible way to order a message.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
pub enum MessageOrder {
    /// Deliver the message immediately, before any other message.
    Immediate,

    /// Deliver the message in any order relative to all other unordered messages.
    Unordered,

    /// Deliver the message in the specified order relative to all other ordered messages between
    /// the same source and target.
    Ordered(MessageIndex),
}

/// A message in-flight between agents.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
pub struct Message<Payload> {
    /// How the message is ordered.
    pub order: MessageOrder,

    /// The source agent index.
    pub source_index: usize,

    /// The target agent index.
    pub target_index: usize,

    /// The actual payload.
    pub payload: Payload,

    /// The replaced message, if any.
    pub replaced: Option<Payload>,
}

/// An indicator that something is invalid.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Invalid<MessageId> {
    Configuration(&'static str),
    Agent(usize, &'static str),
    Message(MessageId, &'static str),
}

/// A complete system configuration.
///
/// We will have a *lot* of these, so keeping their size down and avoiding heap memory as much as
/// possible is critical. The maximal sizes were chosen so that the configuration plus its memoized
/// identifier will fit together inside exactly one cache lines, which should make this more
/// cache-friendly when placed inside a hash table.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct Configuration<
    StateId,
    MessageId,
    InvalidId,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The state of each agent.
    pub state_ids: [StateId; MAX_AGENTS],

    /// The number of messages sent by each agent.
    pub message_counts: [MessageIndex; MAX_AGENTS],

    /// The messages sent by each agent.
    pub message_ids: [MessageId; MAX_MESSAGES],

    /// The invalid condition, if any.
    pub invalid_id: InvalidId,

    /// The index of the immediate message, or 255 if there is none.
    pub immediate_index: MessageIndex,
}

// END MAYBE TESTED

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>
{
    /// Remove a message from the configuration.
    fn remove_message(&mut self, source: usize, mut message_index: usize) {
        debug_assert!(self.message_counts[source].to_usize() > 0);
        debug_assert!(self.message_ids[message_index].is_valid());

        self.message_counts[source].decr();

        if self.immediate_index.is_valid() {
            // BEGIN NOT TESTED
            match self.immediate_index.to_usize() {
                immediate_index if immediate_index > message_index => {
                    self.immediate_index.decr();
                }
                immediate_index if immediate_index == message_index => {
                    self.immediate_index = MessageIndex::invalid();
                }
                _ => {}
            }
            // END NOT TESTED
        }

        loop {
            let next_message_index = message_index + 1;
            let next_message_id = if next_message_index < MAX_MESSAGES {
                self.message_ids[next_message_index]
            } else {
                MessageId::invalid() // NOT TESTED
            };
            self.message_ids[message_index] = next_message_id;
            if !next_message_id.is_valid() {
                return;
            }
            message_index = next_message_index; // NOT TESTED
        }
    }

    /// Add a message to the configuration.
    fn add_message(&mut self, source_index: usize, message_id: MessageId, is_immediate: bool) {
        debug_assert!(!is_immediate || !self.has_immediate());
        debug_assert!(source_index != usize::max_value());
        debug_assert!(self.message_counts[source_index] < MessageIndex::invalid());

        assert!(
            !self.message_ids[MAX_MESSAGES - 1].is_valid(),
            "too many in-flight messages, must be at most {}",
            MAX_MESSAGES
        );

        self.message_counts[source_index].incr();

        self.message_ids[MAX_MESSAGES - 1] = message_id;
        let immediate_index = if is_immediate {
            message_id // NOT TESTED
        } else {
            self.immediate_index().unwrap_or_else(MessageId::invalid)
        };

        self.message_ids.sort();

        if immediate_index.is_valid() {
            // BEGIN NOT TESTED
            let immediate_index = self
                .message_ids
                .iter()
                .position(|&message_id| message_id == immediate_index)
                .unwrap();
            self.immediate_index = MessageIndex::from_usize(immediate_index);
            // END NOT TESTED
        }
    }

    /// Return whether there is an immediate message.
    fn has_immediate(&self) -> bool {
        self.immediate_index.is_valid()
    }

    /// Return the immediate message identifier, if any.
    fn immediate_index(&self) -> Option<MessageId> {
        if self.has_immediate() {
            Some(self.message_ids[self.immediate_index.to_usize()]) // NOT TESTED
        } else {
            None
        }
    }

    /// Change the state of an agent in the configuration.
    fn change_state(&mut self, agent_index: usize, state_id: StateId) {
        self.state_ids[agent_index] = state_id;
    }
}

// BEGIN MAYBE TESTED

/// A transition from a given configuration.
#[derive(Copy, Clone, Debug)]
pub struct Outgoing<ConfigurationId> {
    /// The identifier of the target configuration.
    pub to: ConfigurationId,

    /// The index of the message of the source configuration that was delivered to its target agent
    /// to reach the target configuration.
    ///
    /// We use the configuration identifier type as this is guaranteed to be large enough, and
    /// anything smaller will not reduce the structure size, if we want fields to be properly
    /// aligned.
    pub message_index: ConfigurationId,
}

/// A transition to a given configuration.
#[derive(Copy, Clone, Debug)]
pub struct Incoming<ConfigurationId> {
    /// The identifier of the source configuration.
    pub from: ConfigurationId,

    /// The index of the message of the source configuration that was delivered to its target agent
    /// to reach the target configuration.
    ///
    /// We use the configuration identifier type as this is guaranteed to be large enough, and
    /// anything smaller will not reduce the structure size, if we want fields to be properly
    /// aligned.
    pub message_index: ConfigurationId,
}

/// A complete model.
pub struct Model<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: KeyLike + Validated + Named,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The agent types that will be used in the model.
    pub types: Vec<<Self as MetaModel>::AgentTypeArc>,

    /// The type of each agent.
    pub agent_types: Vec<<Self as MetaModel>::AgentTypeArc>,

    /// The label of each agent.
    pub agent_labels: Vec<Arc<String>>,

    /// The first instance of the same type of each agent.
    pub first_instances: Vec<usize>,

    /// Validation functions for the configuration.
    pub validators: Vec<<Self as MetaModel>::Validator>,

    /// Memoization of the configurations.
    pub configurations: RwLock<Memoize<<Self as MetaModel>::Configuration, ConfigurationId>>,

    /// Memoization of the in-flight messages.
    pub messages: Arc<RwLock<Memoize<Message<Payload>, MessageId>>>,

    /// Memoization of the invalid conditions.
    pub invalids: RwLock<Memoize<<Self as MetaModel>::Invalid, InvalidId>>,

    /// For each configuration, which configuration is reachable from it.
    pub outgoings: RwLock<Vec<RwLock<Vec<<Self as MetaModel>::Outgoing>>>>,

    /// For each configuration, which configuration can reach it.
    pub incomings: RwLock<Vec<RwLock<Vec<<Self as MetaModel>::Incoming>>>>,

    /// The maximal message string size we have seen so far.
    pub max_message_string_size: RwLock<usize>,

    /// The maximal invalid condition string size we have seen so far.
    pub max_invalid_string_size: RwLock<usize>,

    /// The maximal configuration string size we have seen so far.
    pub max_configuration_string_size: RwLock<usize>,
}

// END MAYBE TESTED

/// Allow querying the model's meta-parameters.
pub trait MetaModel {
    /// The type of state identifiers.
    type StateId;

    /// The type of message identifiers.
    type MessageId;

    /// The type of invalid condition identifiers.
    type InvalidId;

    /// The type of configuration identifiers.
    type ConfigurationId;

    /// The type of message payloads.
    type Payload;

    /// The maximal number of agents.
    const MAX_AGENTS: usize;

    /// The maximal number of in-flight messages.
    const MAX_MESSAGES: usize;

    /// The type of  boxed agent type.
    type AgentTypeArc;

    /// The type of in-flight messages.
    type Message;

    /// The type of a event handling by an agent.
    type Reaction;

    /// The type of an action from an agent.
    type Action;

    /// The type of an emitted messages.
    type Emit;

    /// The type of invalid conditions.
    type Invalid;

    /// The type of the included configurations.
    type Configuration;

    /// The type of a configuration validation function.
    type Validator;

    /// The type of the incoming transitions.
    type Incoming;

    /// The type of the outgoing transitions.
    type Outgoing;

    /// The context for processing event handling by an agent.
    type Context;
}

// BEGIN MAYBE TESTED

/// The context for processing event handling by an agent.
#[derive(Clone, Debug)]
pub struct Context<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: KeyLike + Validated + Named,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The identifier of the message that the agent received, or `None` if the agent received a
    /// time event.
    delivered_message_id: Option<MessageId>,

    /// Whether the delivered message was an immediate message.
    is_immediate: bool,

    /// The index of the agent that reacted to the event.
    agent_index: usize,

    /// The type of the agent that reacted to the event.
    agent_type: Arc<dyn AgentType<StateId, MessageId, Payload> + Send + Sync>,

    /// The index of the source agent in its type.
    agent_instance: usize,

    /// The identifier of the state of the agent when handling the event.
    agent_from_state_id: StateId,

    /// The incoming transition into the new configuration to be generated.
    incoming: Option<Incoming<ConfigurationId>>,

    /// Incrementally updated to become the target configuration.
    to_configuration: Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,
}

// END MAYBE TESTED

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: KeyLike + Validated + Named,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > MetaModel
    for Model<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>
{
    type StateId = StateId;
    type MessageId = MessageId;
    type InvalidId = InvalidId;
    type ConfigurationId = ConfigurationId;
    type Payload = Payload;
    const MAX_AGENTS: usize = MAX_AGENTS;
    const MAX_MESSAGES: usize = MAX_MESSAGES;

    type AgentTypeArc = Arc<dyn AgentType<StateId, MessageId, Payload> + Send + Sync>;
    type Message = Message<Payload>;
    type Reaction = Reaction<StateId, Payload>;
    type Action = Action<StateId, Payload>;
    type Emit = Emit<Payload>;
    type Invalid = Invalid<MessageId>;
    type Configuration = Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>;
    type Validator = fn(
        &Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,
    ) -> Option<&'static str>;
    type Incoming = Incoming<ConfigurationId>;
    type Outgoing = Outgoing<ConfigurationId>;
    type Context =
        Context<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>;
}

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: KeyLike + Validated + Named,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > Model<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>
{
    /// Create a new model.
    pub fn new(
        types: Vec<<Self as MetaModel>::AgentTypeArc>,
        validators: Vec<<Self as MetaModel>::Validator>,
        threads: usize,
    ) -> Self {
        assert!(
            MAX_MESSAGES < MessageIndex::invalid().to_usize(),
            "MAX_MESSAGES {} is too large, must be less than {}",
            MAX_MESSAGES,
            MessageIndex::invalid() // NOT TESTED
        );

        let mut agent_types: Vec<<Self as MetaModel>::AgentTypeArc> = vec![];
        let mut first_instances: Vec<usize> = vec![];
        let mut agent_labels: Vec<Arc<String>> = vec![];

        for agent_type in types.iter() {
            let count = agent_type.count();
            assert!(
                count > 0,
                "zero instances requested for the type {}",
                agent_type.name() // NOT TESTED
            );
            let first_instance = agent_types.len();
            for instance in 0..count {
                first_instances.push(first_instance);
                agent_types.push(agent_type.clone());
                let agent_label = if agent_type.is_indexed() {
                    format!("{}-{}", agent_type.name(), instance) // NOT TESTED
                } else {
                    debug_assert!(instance == 0);
                    agent_type.name().to_string()
                };
                agent_labels.push(Arc::new(agent_label));
            }
        }

        let dummy_agent_type = agent_types[0].clone();

        let model = Model {
            types,
            agent_types,
            agent_labels,
            first_instances,
            validators,
            configurations: RwLock::new(Memoize::new(false)),
            messages: Arc::new(RwLock::new(Memoize::new(true))),
            invalids: RwLock::new(Memoize::new(true)),
            outgoings: RwLock::new(Vec::new()),
            incomings: RwLock::new(Vec::new()),
            max_message_string_size: RwLock::new(0),
            max_invalid_string_size: RwLock::new(0),
            max_configuration_string_size: RwLock::new(0),
        };

        let initial_configuration = Configuration {
            state_ids: [StateId::from_usize(0); MAX_AGENTS],
            message_counts: [MessageIndex::from_usize(0); MAX_AGENTS],
            message_ids: [MessageId::invalid(); MAX_MESSAGES],
            invalid_id: InvalidId::invalid(),
            immediate_index: MessageIndex::invalid(),
        };

        let context = Context {
            delivered_message_id: None,
            is_immediate: false,
            agent_index: usize::max_value(),
            agent_type: dummy_agent_type,
            agent_instance: usize::max_value(),
            agent_from_state_id: StateId::invalid(),
            incoming: None,
            to_configuration: initial_configuration,
        };

        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| model.reach_configuration(context));

        model
    }

    fn reach_configuration(&self, mut context: <Self as MetaModel>::Context) {
        self.validate_configuration(&mut context);
        //      eprintln!(
        //          "reached: {}",
        //          self.display_configuration(&context.to_configuration)
        //      );

        let stored = self.fully_store_configuration(context.to_configuration, context.incoming);
        let configuration_id = stored.id;
        if !stored.is_new {
            if let Some(transition) = context.incoming {
                self.incomings.read().unwrap()[configuration_id.to_usize()]
                    .write()
                    .unwrap()
                    .push(transition);
            }
            return;
        }

        let messages_count = if context.to_configuration.has_immediate() {
            1 // NOT TESTED
        } else {
            context
                .to_configuration
                .message_ids
                .iter()
                .position(|&message_id| !message_id.is_valid())
                .unwrap_or(MAX_MESSAGES)
        };
        let events_count = self.agents_count() + messages_count;

        (0..events_count).into_iter().for_each(|event_index| {
            // TODO - par_iter
            if event_index < self.agents_count() {
                self.deliver_time_event(configuration_id, context.to_configuration, event_index);
            } else if context.to_configuration.has_immediate() {
                // BEGIN NOT TESTED
                debug_assert!(event_index == self.agents_count());
                self.deliver_message_event(
                    configuration_id,
                    context.to_configuration,
                    context.to_configuration.immediate_index.to_usize(),
                );
                // END NOT TESTED
            } else {
                self.deliver_message_event(
                    configuration_id,
                    context.to_configuration,
                    event_index - self.agents_count(),
                );
            }
        });
    }

    fn deliver_time_event(
        &self,
        from_configuration_id: ConfigurationId,
        from_configuration: <Self as MetaModel>::Configuration,
        agent_index: usize,
    ) {
        let agent_from_state_id = from_configuration.state_ids[agent_index];
        let agent_type = self.agent_types[agent_index].clone();
        let agent_instance = self.instance(agent_index);
        let reaction = agent_type.pass_time(agent_instance, agent_from_state_id);

        if reaction == Reaction::Ignore {
            return;
        }

        let incoming = Some(Incoming {
            from: from_configuration_id,
            message_index: ConfigurationId::invalid(),
        });

        let to_configuration = from_configuration;

        let context = Context {
            delivered_message_id: None,
            is_immediate: false,
            agent_index,
            agent_type,
            agent_instance,
            agent_from_state_id,
            incoming,
            to_configuration,
        };
        self.process_reaction(context, reaction);
    }

    fn deliver_message_event(
        &self,
        from_configuration_id: ConfigurationId,
        from_configuration: <Self as MetaModel>::Configuration,
        message_index: usize,
    ) {
        let message_id = from_configuration.message_ids[message_index];

        let (source_index, target_index, payload, is_immediate) = {
            let messages = self.messages.read().unwrap();
            let message = messages.get(message_id);
            if let MessageOrder::Ordered(order) = message.order {
                // BEGIN NOT TESTED
                if order.to_usize() > 0 {
                    return;
                }
                // END NOT TESTED
            }
            (
                message.source_index,
                message.target_index,
                message.payload,
                message.order == MessageOrder::Immediate,
            )
        };

        let target_instance = self.instance(target_index);
        let target_from_state_id = from_configuration.state_ids[target_index];
        let target_type = self.agent_types[target_index].clone();
        let reaction = target_type.receive_message(target_index, target_from_state_id, &payload);

        let incoming = Some(Incoming {
            from: from_configuration_id,
            message_index: ConfigurationId::from_usize(message_index),
        });

        let mut to_configuration = from_configuration;
        self.remove_message(&mut to_configuration, source_index, message_index);

        let context = Context {
            delivered_message_id: Some(message_id),
            is_immediate,
            agent_index: target_index,
            agent_type: target_type,
            agent_instance: target_instance,
            agent_from_state_id: target_from_state_id,
            incoming,
            to_configuration,
        };
        self.process_reaction(context, reaction);
    }

    fn process_reaction(
        &self,
        context: <Self as MetaModel>::Context,
        reaction: <Self as MetaModel>::Reaction,
    ) {
        match reaction {
            Reaction::Unexpected => self.unexpected_message(context), // MAYBE TESTED
            Reaction::Defer => self.is_deferring_message(context),    // NOT TESTED
            Reaction::Ignore => self.is_ignoring_message(context),    // NOT TESTED
            Reaction::Do1(action1) => self.perform_action(context, action1),

            // BEGIN NOT TESTED
            Reaction::Do1Of2(action1, action2) => {
                self.perform_action(context.clone(), action1);
                self.perform_action(context, action2);
            }

            Reaction::Do1Of3(action1, action2, action3) => {
                self.perform_action(context.clone(), action1);
                self.perform_action(context.clone(), action2);
                self.perform_action(context, action3);
            }

            Reaction::Do1Of4(action1, action2, action3, action4) => {
                self.perform_action(context.clone(), action1);
                self.perform_action(context.clone(), action2);
                self.perform_action(context.clone(), action3);
                self.perform_action(context, action4);
            } // END NOT TESTED
        }
    }

    fn perform_action(
        &self,
        mut context: <Self as MetaModel>::Context,
        action: <Self as MetaModel>::Action,
    ) {
        match action {
            Action::Defer => self.is_deferring_message(context),
            Action::Ignore => self.is_ignoring_message(context), // NOT TESTED

            Action::Change(target_to_state_id) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.reach_configuration(context);
            }
            // BEGIN NOT TESTED
            Action::Send1(emit1) => self.emit_transition(context, emit1),
            // END NOT TESTED
            Action::ChangeAndSend1(target_to_state_id, emit1) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(context, emit1);
            }

            // BEGIN NOT TESTED
            Action::ChangeAndSend2(target_to_state_id, emit1, emit2) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context, emit2);
            }

            Action::Send2(emit1, emit2) => {
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context, emit2);
            }

            Action::ChangeAndSend3(target_to_state_id, emit1, emit2, emit3) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context.clone(), emit2);
                self.emit_transition(context, emit3);
            }

            Action::Send3(emit1, emit2, emit3) => {
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context.clone(), emit2);
                self.emit_transition(context, emit3);
            }

            Action::ChangeAndSend4(target_to_state_id, emit1, emit2, emit3, emit4) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context.clone(), emit2);
                self.emit_transition(context.clone(), emit3);
                self.emit_transition(context, emit4);
            }

            Action::Send4(emit1, emit2, emit3, emit4) => {
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context.clone(), emit2);
                self.emit_transition(context.clone(), emit3);
                self.emit_transition(context, emit4);
            } // END NOT TESTED
        }
    }

    fn emit_transition(
        &self,
        mut context: <Self as MetaModel>::Context,
        emit: <Self as MetaModel>::Emit,
    ) {
        match emit {
            Emit::Immediate(payload, target_index) => {
                // BEGIN NOT TESTED
                let message = Message {
                    order: MessageOrder::Immediate,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced: None,
                };
                self.emit_message(context, message);
                // END NOT TESTED
            }

            Emit::Unordered(payload, target_index) => {
                let message = Message {
                    order: MessageOrder::Unordered,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced: None,
                };
                self.emit_message(context, message);
            }

            // BEGIN NOT TESTED
            Emit::Ordered(payload, target_index) => {
                let order = self.count_ordered(
                    &context.to_configuration,
                    context.agent_index,
                    target_index,
                ) + 1;
                let message = Message {
                    order: MessageOrder::Ordered(MessageIndex::from_usize(order)),
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced: None,
                };
                self.emit_message(context, message);
            }

            Emit::ImmediateReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(&mut context, callback, &payload, target_index);
                let message = Message {
                    order: MessageOrder::Immediate,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced,
                };
                self.emit_message(context, message);
            }

            Emit::UnorderedReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(&mut context, callback, &payload, target_index);
                let message = Message {
                    order: MessageOrder::Unordered,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced,
                };
                self.emit_message(context, message);
            }

            Emit::OrderedReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(&mut context, callback, &payload, target_index);
                let order = self.count_ordered(
                    &context.to_configuration,
                    context.agent_index,
                    target_index,
                ) + 1;
                let message = Message {
                    order: MessageOrder::Ordered(MessageIndex::from_usize(order)),
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced,
                };
                self.emit_message(context, message);
            } // END NOT TESTED
        }
    }

    // BEGIN NOT TESTED
    fn count_ordered(
        &self,
        configuration: &<Self as MetaModel>::Configuration,
        source_index: usize,
        target_index: usize,
    ) -> usize {
        let messages = self.messages.read().unwrap();
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .map(|message_id| messages.get(*message_id))
            .filter(|message| {
                message.source_index == source_index
                    && message.target_index == target_index
                    && matches!(message.order, MessageOrder::Ordered(_))
            })
            .fold(0, |count, _message| count + 1)
    }

    fn replace_message(
        &self,
        context: &mut <Self as MetaModel>::Context,
        callback: fn(Option<Payload>) -> bool,
        payload: &Payload,
        target_index: usize,
    ) -> Option<Payload> {
        let replaced = {
            let messages = self.messages.read().unwrap();
            context.to_configuration
                .message_ids
                .iter()
                .take_while(|message_id| message_id.is_valid())
                .enumerate()
                .map(|(message_index, message_id)| (message_index, messages.get(*message_id)))
                .filter(|(_, message)| {
                        message.source_index == context.agent_index
                            && message.target_index == target_index
                            && callback(Some(message.payload))
                })
                .fold(None, |replaced, (message_index, message)| {
                    match replaced {
                        None => Some((message_index, message)),
                        Some((_, ref conflict)) => {
                            let conflict_payload = format!("{}", conflict.payload);
                            let message_payload = format!("{}", message.payload);
                            let replacement_payload = format!("{}", payload);
                            let source_agent_label = self.agent_labels[context.agent_index].clone();
                            let target_agent_label = self.agent_labels[target_index].clone();
                            let event_label = self.event_label(context.delivered_message_id);
                            context.agent_type.state_display(context.agent_from_state_id, Box::new(move |from_state| {
                                panic!(
                                    "both the message {} and the message {} can be replaced by the ambiguous replacement message {} sent to the agent {} by the agent {} in the state {} when responding to the {}",
                                   conflict_payload,
                                   message_payload,
                                   replacement_payload,
                                   target_agent_label,
                                   source_agent_label,
                                   from_state,
                                   event_label
                               );
                            }));
                            None // NOT REACHED
                        }
                    }
                })
                .map(|(message_index, message)| Some((message_index, message.payload)))
                .unwrap_or_else(|| {
                    if !callback(None) {
                        let replacement_payload = format!("{}", payload);
                        let source_agent_label = self.agent_labels[context.agent_index].clone();
                        let target_agent_label = self.agent_labels[target_index].clone();
                        let event_label = self.event_label(context.delivered_message_id);
                        context.agent_type.state_display(context.agent_from_state_id, Box::new(move |from_state| {
                            panic!(
                                "nothing was replaced by the required replacement message {} sent to the agent {} by the agent {} in the state {} when responding to the {}",
                               replacement_payload,
                               target_agent_label,
                               source_agent_label,
                               from_state,
                               event_label
                           );
                        }));
                    }
                    None
                })
        };

        if let Some((replaced_index, replaced_payload)) = replaced {
            self.remove_message(
                &mut context.to_configuration,
                context.agent_index,
                replaced_index,
            );
            Some(replaced_payload)
        } else {
            None
        }
    }
    // END NOT TESTED

    fn remove_message(
        &self,
        configuration: &mut <Self as MetaModel>::Configuration,
        source: usize,
        message_index: usize,
    ) {
        let removed_message_id = configuration.message_ids[message_index];
        let (removed_source_index, removed_target_index, removed_order) = {
            let messages = self.messages.read().unwrap();
            let removed_message = messages.get(removed_message_id);
            if let MessageOrder::Ordered(removed_order) = removed_message.order {
                // BEGIN NOT TESTED
                (
                    removed_message.source_index,
                    removed_message.target_index,
                    Some(removed_order),
                )
                // END NOT TESTED
            } else {
                (
                    removed_message.source_index,
                    removed_message.target_index,
                    None,
                )
            }
        };

        configuration.remove_message(source, message_index);

        if let Some(removed_message_order) = removed_order {
            // BEGIN NOT TESTED
            let mut messages = self.messages.write().unwrap();
            let mut did_modify = false;
            for message_index in 0..MAX_MESSAGES {
                let message_id = configuration.message_ids[message_index];

                if !message_id.is_valid() {
                    break;
                }

                if message_id == removed_message_id {
                    continue;
                }

                let message = messages.get(message_id);
                if message.source_index != removed_source_index
                    || message.target_index != removed_target_index
                {
                    continue;
                }

                if let MessageOrder::Ordered(mut message_order) = message.order {
                    if message_order > removed_message_order {
                        let mut new_message = *message;
                        message_order.decr();
                        new_message.order = MessageOrder::Ordered(message_order);
                        configuration.message_ids[message_index] =
                            if let Some(new_message_id) = messages.lookup(&new_message) {
                                *new_message_id
                            } else {
                                messages
                                    .store(new_message, Some(self.display_message(&new_message)))
                                    .id
                            };
                        did_modify = true;
                    }
                }
            }

            if did_modify {
                assert!(configuration.immediate_index.is_valid());
                configuration.message_ids.sort();
            }
            // END NOT TESTED
        }
    }

    fn emit_message(
        &self,
        mut context: <Self as MetaModel>::Context,
        message: <Self as MetaModel>::Message,
    ) {
        let is_immediate = message.order == MessageOrder::Immediate;
        let message_id = self.store_message(message);
        context
            .to_configuration
            .add_message(context.agent_index, message_id, is_immediate);
        self.reach_configuration(context);
    }

    // BEGIN NOT TESTED
    fn unexpected_message(&self, context: <Self as MetaModel>::Context) {
        match context.delivered_message_id {
            None => {
                let agent_label = self.agent_labels[context.agent_index].clone();
                let event_label = self.event_label(None);
                context.agent_type.state_display(
                    context.agent_from_state_id,
                    Box::new(move |from_state| {
                        panic!(
                            "the agent {} does not expect the {} while in the state {}",
                            agent_label, event_label, from_state
                        );
                    }),
                );
            }

            Some(delivered_message_id) => {
                let agent_label = self.agent_labels[context.agent_index].clone();
                let event_label = self.event_label(Some(delivered_message_id));
                context.agent_type.state_display(
                    context.agent_from_state_id,
                    Box::new(move |from_state| {
                        panic!(
                            "the agent {} does not expect the {} while in the state {}",
                            agent_label, event_label, from_state
                        );
                    }),
                );
            }
        }
    }

    fn is_ignoring_message(&self, context: <Self as MetaModel>::Context) {
        if context.incoming.unwrap().message_index.is_valid() {
            self.reach_configuration(context);
        }
    }

    fn is_deferring_message(&self, context: <Self as MetaModel>::Context) {
        match context.delivered_message_id {
            None => {
                let agent_label = self.agent_labels[context.agent_index].clone();
                let event_label = self.event_label(None);
                context.agent_type.state_display(context.agent_from_state_id, Box::new(move |from_state| {
                    panic!("the agent {} is deferring (should be ignoring) the {} while in the state {}",
                           agent_label, event_label, from_state);
                }));
            }

            Some(delivered_message_id) => {
                if !context
                    .agent_type
                    .state_is_deferring(context.agent_from_state_id)
                {
                    let agent_label = self.agent_labels[context.agent_index].clone();
                    let event_label = self.event_label(Some(delivered_message_id));
                    context.agent_type.state_display(context.agent_from_state_id, Box::new(move |from_state| {
                        panic!("the agent {} is deferring (should be ignoring) the {} while in the state {}",
                               agent_label, event_label, from_state);
                    }));
                }

                if context.is_immediate {
                    let agent_label = self.agent_labels[context.agent_index].clone();
                    let event_label = self.event_label(Some(delivered_message_id));
                    context.agent_type.state_display(
                        context.agent_from_state_id,
                        Box::new(move |from_state| {
                            panic!(
                                "the agent {} is deferring the immediate {} while in the state {}",
                                agent_label, event_label, from_state
                            );
                        }),
                    );
                }
            }
        }
    }
    // END NOT TESTED

    fn validate_configuration(&self, context: &mut <Self as MetaModel>::Context) {
        if context.to_configuration.invalid_id.is_valid() {
            return;
        }

        if context.agent_index != usize::max_value() {
            let agent_state_id = context.to_configuration.state_ids[context.agent_index];
            if let Some(max_in_flight_messages) = context
                .agent_type
                .state_max_in_flight_messages(agent_state_id)
            {
                if context.to_configuration.message_counts[context.agent_index].to_usize()
                    > max_in_flight_messages
                {
                    // BEGIN NOT TESTED
                    let mut messages_string = String::new();
                    for message_id in context.to_configuration.message_ids.iter() {
                        if !message_id.is_valid() {
                            break;
                        }
                        messages_string.push_str("\n- ");
                        messages_string
                            .push_str(self.messages.read().unwrap().display(*message_id));
                    }

                    let agent_label = self.agent_labels[context.agent_index].clone();
                    let event_label = self.event_label(context.delivered_message_id);

                    context.agent_type.state_display(
                        context.agent_from_state_id,
                        Box::new(move |agent_from_state| {
                            panic!(
                                "the agent {} sends too many messages when reacting to the {} while in the state {}{}",
                                agent_label, event_label, agent_from_state, messages_string
                            );
                        }),
                    );
                    // END NOT TESTED
                }
            }
        }

        for validator in self.validators.iter() {
            // BEGIN NOT TESTED
            if let Some(reason) = validator(&context.to_configuration) {
                let invalid = <Self as MetaModel>::Invalid::Configuration(reason);
                context.to_configuration.invalid_id = self.store_invalid(invalid);
                return;
            }
            // END NOT TESTED
        }
    }

    fn store_message(&self, message: <Self as MetaModel>::Message) -> MessageId {
        if let Some(message_id) = self.messages.read().unwrap().lookup(&message) {
            return *message_id; // NOT TESTED
        }
        self.messages
            .write()
            .unwrap()
            .store(message, Some(self.display_message(&message)))
            .id
    }

    // BEGIN NOT TESTED
    fn store_invalid(&self, invalid: <Self as MetaModel>::Invalid) -> InvalidId {
        if let Some(invalid_id) = self.invalids.read().unwrap().lookup(&invalid) {
            return *invalid_id;
        }
        self.invalids
            .write()
            .unwrap()
            .store(invalid, Some(self.display_invalid(&invalid)))
            .id
    }
    // END NOT TESTED

    fn store_configuration(
        &self,
        configuration: <Self as MetaModel>::Configuration,
    ) -> Stored<ConfigurationId> {
        if let Some(configuration_id) = self.configurations.read().unwrap().lookup(&configuration) {
            return Stored {
                id: *configuration_id,
                is_new: false,
            };
        }
        self.configurations
            .write()
            .unwrap()
            .store(configuration, None)
    }

    fn fully_store_configuration(
        &self,
        configuration: <Self as MetaModel>::Configuration,
        incoming: Option<<Self as MetaModel>::Incoming>,
    ) -> Stored<ConfigurationId> {
        let stored = self.store_configuration(configuration);

        let is_new = stored.is_new;
        let to_configuration_id = stored.id;
        if is_new || incoming.is_some() {
            {
                let mut incoming_vector = self.incomings.write().unwrap();

                if is_new {
                    incoming_vector.push(RwLock::new(Vec::new()));
                }

                if let Some(transition) = incoming {
                    incoming_vector[to_configuration_id.to_usize()]
                        .write()
                        .unwrap()
                        .push(transition);
                }
            }

            {
                let mut outgoing_vector = self.outgoings.write().unwrap();

                if is_new {
                    outgoing_vector.push(RwLock::new(Vec::new()));
                }

                if let Some(transition) = incoming {
                    let from_configuration_id = transition.from;
                    let outgoing = Outgoing {
                        to: to_configuration_id,
                        message_index: transition.message_index,
                    };
                    outgoing_vector[from_configuration_id.to_usize()]
                        .write()
                        .unwrap()
                        .push(outgoing);
                }
            }
        }

        stored
    }

    /// Return the total number of agents.
    pub fn agents_count(&self) -> usize {
        self.agent_labels.len()
    }

    /// Return the index of the agent with the specified type name.
    ///
    /// If more than one agent of this type exist, also specify its index within its type.
    pub fn agent_index(&self, name: &'static str, instance: Option<usize>) -> usize {
        let mut agent_index: usize = 0;
        for agent_type in self.types.iter() {
            let count = agent_type.count();
            if agent_type.name() != name {
                agent_index += count;
            } else {
                match instance {
                    None => {
                        assert!(count == 1,
                                "no instance index specified when locating an agent of type {} which has {} instances",
                                name, count);
                    }
                    Some(index_in_type) => {
                        assert!(index_in_type < count,
                                "too large instance index {} specified when locating an agent of type {} which has {} instances",
                                index_in_type, name, count);
                        agent_index += index_in_type;
                    }
                }
                return agent_index;
            }
        }
        panic!("looking for an agent of an unknown type {}", name);
    }

    /// Return the index of the agent instance within its type.
    pub fn instance(&self, agent_index: usize) -> usize {
        agent_index - self.first_instances[agent_index]
    }

    /// Return whether all the reachable configurations are valid.
    pub fn is_valid(&self) -> bool {
        self.invalids.read().unwrap().usize() == 0
    }

    // BEGIN NOT TESTED
    fn event_label(&self, message_id: Option<MessageId>) -> String {
        let messages = self.messages.read().unwrap();
        match message_id {
            None => "time event".to_string(),
            Some(message_id) if !message_id.is_valid() => "time event".to_string(),
            Some(message_id) => "message ".to_string() + messages.display(message_id),
        }
    }
    // END NOT TESTED

    /// Display a message.
    pub fn display_message(&self, message: &<Self as MetaModel>::Message) -> String {
        let max_message_string_size = *self.max_message_string_size.read().unwrap();
        let mut string = String::with_capacity(max_message_string_size);

        string.push_str(&*self.agent_labels[message.source_index]);
        string.push_str(" -> ");

        match message.order // MAYBE TESTED
        {
            MessageOrder::Immediate => string.push_str("* "), // MAYBE TESTED
            MessageOrder::Unordered => {}
            MessageOrder::Ordered(order) => string.push_str(&format!("@{} ", order)), // NOT TESTED
        }

        if let Some(ref replaced) = message.replaced {
            // BEGIN NOT TESTED
            string.push_str(&format!("{}", replaced));
            string.push_str(" => ");
            // END NOT TESTED
        }

        string.push_str(&format!("{}", message.payload));

        string.push_str(" -> ");
        string.push_str(&*self.agent_labels[message.target_index]);

        string.shrink_to_fit();
        if string.len() > max_message_string_size {
            let mut max_message_string_size = self.max_message_string_size.write().unwrap();
            if string.len() > *max_message_string_size {
                *max_message_string_size = string.len();
            }
        }

        string
    }

    // BEGIN NOT TESTED

    /// Display an invalid condition.
    fn display_invalid(&self, invalid: &<Self as MetaModel>::Invalid) -> String {
        let max_invalid_string_size = *self.max_invalid_string_size.read().unwrap();
        let mut string = String::with_capacity(max_invalid_string_size);

        match invalid {
            Invalid::Configuration(reason) => {
                string.push_str("configuration is invalid because ");
                string.push_str(reason);
            }

            Invalid::Agent(agent_index, reason) => {
                string.push_str("agent ");
                string.push_str(&*self.agent_labels[*agent_index]);
                string.push_str(" because ");
                string.push_str(reason);
            }

            Invalid::Message(message_id, reason) => {
                string.push_str("message ");
                string.push_str(self.messages.read().unwrap().display(*message_id));
                string.push_str(" because ");
                string.push_str(reason);
            }
        }

        string.shrink_to_fit();
        if string.len() > max_invalid_string_size {
            let mut max_invalid_string_size = self.max_invalid_string_size.write().unwrap();
            if string.len() > *max_invalid_string_size {
                *max_invalid_string_size = string.len();
            }
        }

        string
    }

    // END NOT TESTED

    /// Display a configuration by its identifier.
    pub fn display_configuration_id(&self, configuration_id: ConfigurationId) -> String {
        let configurations = self.configurations.read().unwrap();
        self.display_configuration(configurations.get(configuration_id))
    }

    /// Display a configuration.
    fn display_configuration(&self, configuration: &<Self as MetaModel>::Configuration) -> String {
        let max_configuration_string_size = *self.max_configuration_string_size.read().unwrap();
        let string_rc = Rc::new(RefCell::new(String::with_capacity(
            max_configuration_string_size,
        )));

        let mut prefix = "";
        for agent_index in 0..self.agents_count() {
            let agent_type = &self.agent_types[agent_index];
            let agent_state_id = configuration.state_ids[agent_index];
            string_rc.borrow_mut().push_str(prefix);
            let cloned_rc = string_rc.clone();
            agent_type.state_display(
                agent_state_id,
                Box::new(move |agent_state| {
                    cloned_rc.borrow_mut().push_str(agent_state);
                }),
            );
            prefix = " & ";
        }

        let mut string = string_rc.take();

        prefix = " | ";
        for message_index in 0..MAX_MESSAGES {
            let message_id = configuration.message_ids[message_index];
            if !message_id.is_valid() {
                break;
            }
            string.push_str(prefix);
            string.push_str(self.messages.read().unwrap().display(message_id));
            prefix = " & ";
        }

        if configuration.invalid_id.is_valid() {
            string.push_str(" ! ");
            string.push_str(
                self.invalids
                    .read()
                    .unwrap()
                    .display(configuration.invalid_id),
            );
        }

        string.shrink_to_fit();
        if string.len() > max_configuration_string_size {
            let mut max_configuration_string_size =
                self.max_configuration_string_size.write().unwrap();
            if string.len() > *max_configuration_string_size {
                *max_configuration_string_size = string.len();
            }
        }

        string
    }
}

// BNF for configuration display (`[ ... ]` stands for optional).
//
// CONFIGURATION := AGENT & AGENT & ...
//                  [ | MESSAGE & MESSAGE & ... ]
//                  [ ! INVALID & INVALID & ... ]
//
// AGENT := TYPE[-INDEX] # STATE_NAME[(STATE_DATA)]
//
// MESSAGE := SOURCE_TYPE[-INDEX} ->
//            [REPLACED_NAME[(REPLACED_DATA) => ]
//            [@INT or *] NAME(DATA) ->
//            TARGET_TYPE[-INDEX]
//
// INVALID := KIND is INVALID because REASON
