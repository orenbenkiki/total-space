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

#![feature(trait_alias)]

use clap::App;
use clap::Arg;
use clap::ArgMatches;
use clap::SubCommand;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use rayon::scope;
use rayon::Scope as ParallelScope;
use rayon::ThreadPoolBuilder;
use scc::HashMap as SccHashMap;
use std::cmp::max;
use std::cmp::min;
use std::collections::hash_map::RandomState;
use std::collections::HashMap as StdHashMap;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::hash::Hash;
use std::io::Write;
use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/*
use std::thread::current as current_thread;

macro_rules! current_thread_name {
    () => { current_thread().name().unwrap_or("main") }
}
*/

/// The RwLock type to use.
///
/// Exporting this makes the `agent_index!` macro robust if we change implementations.
pub type RwLock<T> = parking_lot::RwLock<T>;

const GROWTH_FACTOR: usize = 4;

const RIGHT_ARROW: &str = "&#8594;";

const RIGHT_DOUBLE_ARROW: &str = "&#8658;";

/// A trait for anything we use as a key in a HashMap.
pub trait KeyLike = Eq + Hash + Copy + Debug + Sized + Send + Sync;

/// A trait for data we pass around in the model.
pub trait DataLike = KeyLike + Validated + Named + Default;

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
    fn name(&self) -> String;
}

// BEGIN MAYBE TESTED

/// A macro for implementing data-like (states, payload).
#[macro_export]
macro_rules! impl_data_like {
    ($name:ident = $value:expr $(, $from:literal => $to:literal)* $(,)?) => {
        impl_default_by_value! { $name = $value }
        impl_name_for_into_static_str! { $name $(, $from => $to)* }
        impl_display_by_patched_debug! { $name $(, $from => $to)* }
    };
}

/// A macro for implementing `Default` for types using a simple value.
#[macro_export]
macro_rules! impl_default_by_value {
    ($name:ident = $value:expr) => {
        impl Default for $name {
            fn default() -> Self {
                $value
            }
        }
    };
}

/// A macro for implementing `Name` for enums annotated by `strum::IntoStaticStr`.
///
/// This should become unnecessary once `IntoStaticStr` allows converting a reference to a static
/// string, see `<https://github.com/Peternator7/strum/issues/142>`.
#[macro_export]
macro_rules! impl_name_for_into_static_str {
    ($name:ident $(, $from:literal => $to:literal)* $(,)?) => {
        impl total_space::Name for $name {
            fn name(&self) -> String {
                let static_str: &'static str = self.into();
                let string = static_str.to_string();
                $(
                    let string = string.replace($from, $to);
                )*
                string
            }
        }
    };
}

/// A macro for implementing `Debug` for data which has `DisplayDebug`.
///
/// This should be concerted to a derive macro.
#[macro_export]
macro_rules! impl_display_by_patched_debug {
    ($name:ident $(, $from:literal => $to:literal)* $(,)?) => {
        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
                let string = format!("{:?}", self)
                    .replace(" ", "")
                    .replace(":", "=")
                    .replace("{", "(")
                    .replace("}", ")");
                $(
                    let string = string.replace($from, $to);
                )*
                write!(formatter, "{}", string)
            }
        }
    };
}

/// A macro for declaring a global variable containing agent indices.
#[macro_export]
macro_rules! declare_global_agent_indices {
    ($name:ident) => {
        use lazy_static::*;
        lazy_static! {
            static ref $name: total_space::RwLock<Vec<usize>> =
                total_space::RwLock::new(Vec::new());
        }
    };
}

/// A macro for declaring a global variable containing singleton agent index.
#[macro_export]
macro_rules! declare_global_agent_index {
    ($name:ident) => {
        use lazy_static::*;
        lazy_static! {
            static ref $name: total_space::RwLock<usize> =
                total_space::RwLock::new(usize::max_value());
        }
    };
}

/// A macro for initializing a global variable containing singleton agent index.
#[macro_export]
macro_rules! init_global_agent_indices {
    ($name:ident, $label:expr, $model:expr) => {{
        let mut indices = $name.write();
        indices.clear();
        let agent_type = $model.agent_type($label);
        for instance in 0..agent_type.instances_count() {
            indices.push($model.agent_index($label, Some(instance)));
        }
    }};
}

/// A macro for initializing a global variable containing singleton agent index.
#[macro_export]
macro_rules! init_global_agent_index {
    ($name:ident, $label:expr, $model:expr) => {
        let todox = *$name.write() = $model.agent_index($label, None);
        todox
    };
}

/// A macro for accessing a global variable containing agent index.
#[macro_export]
macro_rules! agent_index {
    ($name:ident) => {
        *$name.read()
    };
    ($name:ident[$index:expr]) => {
        $name.read()[$index]
    };
}

/// A macro for accessing the number of agent instances.
#[macro_export]
macro_rules! agents_count {
    ($name:ident) => {
        $name.read().len()
    };
}

/// A trait for data that has a short name (via `AsRef<&'static str>`) and a full display name (via
/// `Display`).
pub trait Named = Display + Name;

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
    id_by_value: SccHashMap<T, I, RandomState>,

    /// The maximal number of identifiers to generate.
    max_count: usize,

    /// The next unused identifier.
    next_id: AtomicUsize,

    /// Convert a memoized identifier to the value.
    value_by_id: RwLock<Vec<RwLock<T>>>,
}

// END MAYBE TESTED

impl<T: KeyLike + Default, I: IndexLike> Memoize<T, I> {
    /// Create a new memoization store.
    pub fn new(capacity: usize, max_count: usize) -> Self {
        let capacity = min(capacity, max_count);

        let mut value_by_id = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            value_by_id.push(RwLock::new(Default::default()));
        }

        Self {
            max_count,
            next_id: AtomicUsize::new(0),
            id_by_value: SccHashMap::new(capacity, RandomState::new()),
            value_by_id: RwLock::new(value_by_id),
        }
    }

    /// Reserve space for some amount of additional values.
    pub fn reserve(&self, additional: usize) {
        let mut value_by_id = self.value_by_id.write();
        for _ in 0..additional {
            value_by_id.push(RwLock::new(Default::default()));
        }

        // Not supported by scc::SccHashMap:
        // self.id_by_value.reserve(additional);
    }

    /// The number of allocated identifiers.
    pub fn len(&self) -> usize {
        self.next_id.load(Ordering::Relaxed)
    }

    /// The allowed number of allocated identifiers.
    pub fn capacity(&self) -> usize {
        self.value_by_id.read().len()
    }

    /// Whether we have no identifiers stored at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Given a value, look it up in the memory.
    pub fn lookup(&self, value: &T) -> Option<I> {
        self.id_by_value.read(value, |_value, id| *id)
    }

    /// Given a value that may or may not exist in the memory, ensure it exists it and return its
    /// short identifier.
    pub fn store(&self, value: T) -> Stored<I> {
        match self.id_by_value.insert(value, I::from_usize(0)) {
            Err(result) => Stored {
                id: *result.0.get().1,
                is_new: false,
            },
            Ok(result) => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                assert!(
                    id < self.max_count,
                    "too many ({}) memoized objects",
                    id + 1
                );

                debug_assert!(*result.get().1 == I::from_usize(0));
                *result.get().1 = I::from_usize(id);

                if id >= self.value_by_id.read().len() {
                    let mut value_by_id = self.value_by_id.write();
                    if id >= value_by_id.len() {
                        let additional = max(
                            value_by_id.len() / GROWTH_FACTOR,
                            1 + id - value_by_id.len(),
                        );
                        for _ in 0..additional {
                            value_by_id.push(RwLock::new(Default::default()));
                        }
                    }
                }

                let value_by_id = self.value_by_id.read();
                *value_by_id[id].write() = value;

                Stored {
                    id: I::from_usize(id),
                    is_new: true,
                }
            }
        }
    }

    /// Given a short identifier previously returned by `store`, return the full value.
    pub fn get(&self, id: I) -> T {
        debug_assert!(id.to_usize() < self.len());
        *self.value_by_id.read()[id.to_usize()].read()
    }
}

// BEGIN MAYBE TESTED

/// A message sent by an agent as part of an alternative action triggered by some event.
#[derive(PartialEq, Eq, Debug)]
pub enum Emit<Payload: DataLike> {
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
#[derive(PartialEq, Eq, Debug)]
pub enum Action<State: KeyLike, Payload: DataLike> {
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
#[derive(PartialEq, Eq, Debug)]
pub enum Reaction<State: KeyLike, Payload: DataLike> {
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

/// Specify the number of agent instances to use.
pub enum Instances {
    /// Always have just a single instance.
    Singleton,

    /// Use a specific number of instances.
    Count(usize),
}

// END MAYBE TESTED

/// A trait partially describing some agent instances of the same type.
pub trait AgentInstances<StateId: IndexLike, Payload: DataLike>: Name {
    /// Return the previous agent type in the chain, if any.
    fn prev_agent_type(&self) -> Option<Arc<dyn AgentType<StateId, Payload> + Send + Sync>>;

    /// The index of the first agent of this type.
    fn first_index(&self) -> usize;

    /// The next index after the last agent of this type.
    fn next_index(&self) -> usize;

    /// Whether this type only has a single instance.
    ///
    /// If true, the count will always be 1.
    fn is_singleton(&self) -> bool;

    /// The number of agents of this type that will be used in the system.
    fn instances_count(&self) -> usize;

    /// The order of the agent (for sequence diagrams).
    fn instance_order(&self, instance: usize) -> usize;

    /// Display the state.
    ///
    /// The format of the display must be either `<state-name>` if the state is a simple enum, or
    /// `<state-name>(<state-data>)` if the state contains additional data. The `Debug` of the state
    /// might be acceptable as-is, but typically it is better to get rid or shorten the explicit
    /// field names, and/or format their values in a more compact form.
    fn display_state(&self, state_id: StateId) -> String;

    /// Convert the full state identifier to the terse state identifier.
    fn terse_id(&self, state_id: StateId) -> StateId;

    /// Return the name of the terse state (just the state name).
    fn display_terse(&self, name_id: StateId) -> String;
}

/// A trait fully describing some agent instances of the same type.
pub trait AgentType<StateId: IndexLike, Payload: DataLike>:
    AgentInstances<StateId, Payload>
{
    /// Return the actions that may be taken by an agent instance with some state when receiving a
    /// message.
    fn receive_message(
        &self,
        instance: usize,
        state_ids: &[StateId],
        payload: &Payload,
    ) -> Reaction<StateId, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn pass_time(&self, instance: usize, state_ids: &[StateId]) -> Reaction<StateId, Payload>;

    /// Whether any agent in the state is deferring messages.
    fn state_is_deferring(&self, instance: usize, state_ids: &[StateId]) -> bool;

    /// The maximal number of messages sent by an agent which may be in-flight when it is in the
    /// state.
    fn state_max_in_flight_messages(&self, instance: usize, state_ids: &[StateId])
        -> Option<usize>;

    /// The total number of states seen so far.
    fn states_count(&self) -> usize;

    /// Compute mapping from full states to terse states (name only).
    fn compute_terse(&self);
}

/// Allow access to state of parts.
pub trait PartType<State: DataLike, StateId: IndexLike> {
    /// Access the part state by the state identifier.
    fn part_state_by_id(&self, state_id: StateId) -> State;

    /// The index of the first agent of this type.
    fn part_first_index(&self) -> usize;

    /// The number of agent instances of this type.
    fn parts_count(&self) -> usize;
}

// BEGIN MAYBE TESTED

/// The data we need to implement an agent type.
///
/// This should be placed in a `Singleton` to allow the agent states to get services from it.
pub struct AgentTypeData<State: DataLike, StateId: IndexLike, Payload: DataLike> {
    /// Memoization of the agent states.
    states: Memoize<State, StateId>,

    /// The index of the first agent of this type.
    first_index: usize,

    /// The name of the agent type.
    name: &'static str,

    /// Whether this type only has a single instance.
    is_singleton: bool,

    /// Convert a full state identifier to a terse state identifier.
    terse_of_state: RwLock<Vec<StateId>>,

    /// The names of the terse states (state names only).
    name_of_terse: RwLock<Vec<String>>,

    /// The order of each instance (for sequence diagrams).
    instance_orders: Vec<usize>,

    /// The previous agent type in the chain.
    prev_agent_type: Option<Arc<dyn AgentType<StateId, Payload> + Send + Sync>>,

    /// Trick the compiler into thinking we have a field of type Payload.
    _payload: PhantomData<Payload>,
}

/// The data we need to implement an agent type.
///
/// This should be placed in a `Singleton` to allow the agent states to get services from it.
pub struct ContainerTypeData<
    State: DataLike,
    Part: DataLike,
    StateId: IndexLike,
    Payload: DataLike,
    const MAX_PARTS: usize,
> {
    /// The basic agent type data.
    agent_type_data: AgentTypeData<State, StateId, Payload>,

    /// Access part states (for a container).
    part_type: Arc<dyn PartType<Part, StateId> + Send + Sync>,
}

// END MAYBE TESTED

impl<State: DataLike, StateId: IndexLike, Payload: DataLike>
    AgentTypeData<State, StateId, Payload>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(
        name: &'static str,
        instances: Instances,
        prev_agent_type: Option<Arc<dyn AgentType<StateId, Payload> + Send + Sync>>,
    ) -> Self {
        let (is_singleton, count) = match instances {
            Instances::Singleton => (true, 1),
            Instances::Count(amount) => {
                assert!(
                    amount > 0,
                    "zero instances specified for agent type {}",
                    name
                );
                (false, amount)
            }
        };

        let default_state: State = Default::default();
        let states = Memoize::new(StateId::invalid().to_usize(), StateId::invalid().to_usize());
        states.store(default_state);

        let instance_orders = vec![0; count];

        Self {
            name,
            instance_orders,
            is_singleton,
            terse_of_state: RwLock::new(vec![]),
            name_of_terse: RwLock::new(vec![]),
            states,
            first_index: prev_agent_type
                .clone()
                .map_or(0, |agent_type| agent_type.next_index()),
            prev_agent_type,
            _payload: PhantomData,
        }
    }

    /// Compute mapping between full and terse state identifiers.
    pub fn impl_compute_terse(&self) {
        let mut terse_of_state = self.terse_of_state.write();
        let mut name_of_terse = self.name_of_terse.write();

        assert!(terse_of_state.is_empty());
        assert!(name_of_terse.is_empty());

        terse_of_state.reserve(self.states.len());
        name_of_terse.reserve(self.states.len());

        for state_id in 0..self.states.len() {
            let state = self.states.get(StateId::from_usize(state_id));
            let state_name = state.name();
            if let Some(terse_id) = name_of_terse
                .iter()
                .position(|terse_name| terse_name == &state_name)
            {
                terse_of_state.push(StateId::from_usize(terse_id));
            } else {
                terse_of_state.push(StateId::from_usize(name_of_terse.len()));
                name_of_terse.push(state_name);
            }
        }

        name_of_terse.shrink_to_fit();
    }
}

impl<
        State: DataLike,
        Part: DataLike,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > ContainerTypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(
        name: &'static str,
        instances: Instances,
        part_type: Arc<dyn PartType<Part, StateId> + Send + Sync>,
        prev_type: Arc<dyn AgentType<StateId, Payload> + Send + Sync>,
    ) -> Self {
        Self {
            agent_type_data: AgentTypeData::new(name, instances, Some(prev_type)),
            part_type,
        }
    }
}

impl<State: DataLike, StateId: IndexLike, Payload: DataLike> PartType<State, StateId>
    for AgentTypeData<State, StateId, Payload>
{
    fn part_state_by_id(&self, state_id: StateId) -> State {
        self.states.get(state_id)
    }

    fn part_first_index(&self) -> usize {
        self.first_index
    }

    fn parts_count(&self) -> usize {
        self.instances_count()
    }
}

/// A trait for a single agent state.
pub trait AgentState<State: DataLike, Payload: DataLike> {
    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// message.
    fn receive_message(&self, instance: usize, payload: &Payload) -> Reaction<State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn pass_time(&self, instance: usize) -> Reaction<State, Payload>;

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self) -> bool {
        false
    }

    /// The maximal number of messages sent by this agent which may be in-flight when it is in this
    /// state.
    fn max_in_flight_messages(&self) -> Option<usize> {
        None
    }
}

/// A trait for a container agent state.
pub trait ContainerState<State: DataLike, Part: DataLike, Payload: DataLike> {
    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// message.
    fn receive_message(
        &self,
        instance: usize,
        payload: &Payload,
        parts: &[Part],
    ) -> Reaction<State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn pass_time(&self, instance: usize, parts: &[Part]) -> Reaction<State, Payload>;

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self, _parts: &[Part]) -> bool {
        false
    }

    /// The maximal number of messages sent by this agent which may be in-flight when it is in this
    /// state.
    fn max_in_flight_messages(&self, _parts: &[Part]) -> Option<usize> {
        None
    }
}

pub trait Validated {
    /// If this object is invalid, return why.
    fn invalid(&self) -> Option<&'static str> {
        None
    }
}

impl<State: DataLike, StateId: IndexLike, Payload: DataLike>
    AgentTypeData<State, StateId, Payload>
{
    fn translate_reaction(&self, reaction: Reaction<State, Payload>) -> Reaction<StateId, Payload> {
        match reaction {
            Reaction::Unexpected => Reaction::Unexpected,
            Reaction::Ignore => Reaction::Ignore,
            Reaction::Defer => Reaction::Defer,
            Reaction::Do1(action) => Reaction::Do1(self.translate_action(action)),
            Reaction::Do1Of2(action1, action2) => Reaction::Do1Of2(
                self.translate_action(action1),
                self.translate_action(action2),
            ),
            // BEGIN NOT TESTED
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

            Action::Send1(emit) => Action::Send1(emit),
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
        self.states.store(state).id
    }
}

impl<State: DataLike, StateId: IndexLike, Payload: DataLike> Name
    for AgentTypeData<State, StateId, Payload>
{
    fn name(&self) -> String {
        self.name.to_string()
    }
}

impl<
        State: DataLike,
        Part: DataLike,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > Name for ContainerTypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn name(&self) -> String {
        self.agent_type_data.name()
    }
}

impl<State: DataLike, StateId: IndexLike, Payload: DataLike> AgentInstances<StateId, Payload>
    for AgentTypeData<State, StateId, Payload>
{
    fn prev_agent_type(&self) -> Option<Arc<dyn AgentType<StateId, Payload> + Send + Sync>> {
        self.prev_agent_type.clone()
    }

    fn first_index(&self) -> usize {
        self.first_index
    }

    fn next_index(&self) -> usize {
        self.first_index + self.instances_count()
    }

    fn is_singleton(&self) -> bool {
        self.is_singleton
    }

    fn instances_count(&self) -> usize {
        self.instance_orders.len()
    }

    fn instance_order(&self, instance: usize) -> usize {
        self.instance_orders[instance]
    }

    fn display_state(&self, state_id: StateId) -> String {
        format!("{}", self.states.get(state_id))
    }

    fn terse_id(&self, state_id: StateId) -> StateId {
        self.terse_of_state.read()[state_id.to_usize()]
    }

    fn display_terse(&self, terse_id: StateId) -> String {
        self.name_of_terse.read()[terse_id.to_usize()].clone()
    }
}

impl<
        State: DataLike + ContainerState<State, Part, Payload>,
        Part: DataLike + AgentState<Part, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > AgentInstances<StateId, Payload>
    for ContainerTypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn prev_agent_type(&self) -> Option<Arc<dyn AgentType<StateId, Payload> + Send + Sync>> {
        self.agent_type_data.prev_agent_type.clone()
    }

    fn first_index(&self) -> usize {
        self.agent_type_data.first_index()
    }

    fn next_index(&self) -> usize {
        self.agent_type_data.next_index()
    }

    fn is_singleton(&self) -> bool {
        self.agent_type_data.is_singleton()
    }

    fn instances_count(&self) -> usize {
        self.agent_type_data.instances_count()
    }

    fn instance_order(&self, instance: usize) -> usize {
        self.agent_type_data.instance_order(instance)
    }

    fn display_state(&self, state_id: StateId) -> String {
        self.agent_type_data.display_state(state_id)
    }

    fn terse_id(&self, state_id: StateId) -> StateId {
        self.agent_type_data.terse_id(state_id)
    }

    fn display_terse(&self, terse_id: StateId) -> String {
        self.agent_type_data.display_terse(terse_id)
    }
}

impl<State: DataLike + AgentState<State, Payload>, StateId: IndexLike, Payload: DataLike>
    AgentType<StateId, Payload> for AgentTypeData<State, StateId, Payload>
{
    fn receive_message(
        &self,
        instance: usize,
        state_ids: &[StateId],
        payload: &Payload,
    ) -> Reaction<StateId, Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );
        let reaction = self
            .states
            .get(state_ids[self.first_index + instance])
            .receive_message(instance, payload);
        self.translate_reaction(reaction)
    }

    fn pass_time(&self, instance: usize, state_ids: &[StateId]) -> Reaction<StateId, Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );
        let reaction = self
            .states
            .get(state_ids[self.first_index + instance])
            .pass_time(instance);
        self.translate_reaction(reaction)
    }

    fn state_is_deferring(&self, instance: usize, state_ids: &[StateId]) -> bool {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );
        self.states
            .get(state_ids[self.first_index + instance])
            .is_deferring()
    }

    fn state_max_in_flight_messages(
        &self,
        instance: usize,
        state_ids: &[StateId],
    ) -> Option<usize> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );
        self.states
            .get(state_ids[self.first_index + instance])
            .max_in_flight_messages()
    }

    fn states_count(&self) -> usize {
        self.states.len()
    }

    fn compute_terse(&self) {
        self.impl_compute_terse();
    }
}

impl<
        State: DataLike + ContainerState<State, Part, Payload>,
        Part: DataLike + AgentState<Part, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > ContainerTypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn collect_parts(&self, state_ids: &[StateId], parts: &mut [Part; MAX_PARTS]) {
        let part_first_index = self.part_type.part_first_index();
        (0..self.part_type.parts_count()).for_each(|part_instance| {
            let state_id = state_ids[part_first_index + part_instance];
            parts[part_instance] = self.part_type.part_state_by_id(state_id);
        });
    }
}

impl<
        State: DataLike + ContainerState<State, Part, Payload>,
        Part: DataLike + AgentState<Part, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > AgentType<StateId, Payload> for ContainerTypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn receive_message(
        &self,
        instance: usize,
        state_ids: &[StateId],
        payload: &Payload,
    ) -> Reaction<StateId, Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );

        let mut parts_buffer = [Part::default(); MAX_PARTS];
        self.collect_parts(state_ids, &mut parts_buffer);
        let parts = &parts_buffer[0..self.part_type.parts_count()];

        let reaction = self
            .agent_type_data
            .states
            .get(state_ids[self.agent_type_data.first_index + instance])
            .receive_message(instance, payload, parts);
        self.agent_type_data.translate_reaction(reaction)
    }

    fn pass_time(&self, instance: usize, state_ids: &[StateId]) -> Reaction<StateId, Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );

        let mut parts_buffer = [Part::default(); MAX_PARTS];
        self.collect_parts(state_ids, &mut parts_buffer);
        let parts = &parts_buffer[0..self.part_type.parts_count()];

        let reaction = self
            .agent_type_data
            .states
            .get(state_ids[self.agent_type_data.first_index + instance])
            .pass_time(instance, parts);
        self.agent_type_data.translate_reaction(reaction)
    }

    fn state_is_deferring(&self, instance: usize, state_ids: &[StateId]) -> bool {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );

        let mut parts_buffer = [Part::default(); MAX_PARTS];
        self.collect_parts(state_ids, &mut parts_buffer);
        let parts = &parts_buffer[0..self.part_type.parts_count()];

        self.agent_type_data
            .states
            .get(state_ids[self.agent_type_data.first_index + instance])
            .is_deferring(parts)
    }

    fn state_max_in_flight_messages(
        &self,
        instance: usize,
        state_ids: &[StateId],
    ) -> Option<usize> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );

        let mut parts_buffer = [Part::default(); MAX_PARTS];
        self.collect_parts(state_ids, &mut parts_buffer);
        let parts = &parts_buffer[0..self.part_type.parts_count()];

        self.agent_type_data
            .states
            .get(state_ids[self.agent_type_data.first_index + instance])
            .max_in_flight_messages(parts)
    }

    // BEGIN NOT TESTED
    fn states_count(&self) -> usize {
        self.agent_type_data.states.len()
    }
    // END NOT TESTED

    fn compute_terse(&self) {
        self.agent_type_data.impl_compute_terse();
    }
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
pub struct Message<Payload: DataLike> {
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

/// A message in-flight between agents, considering only names.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub struct TerseMessage {
    /// Whether the message is immediate.
    pub is_immediate: bool,

    /// The source agent index.
    pub source_index: usize,

    /// The target agent index.
    pub target_index: usize,

    /// The actual payload (name only).
    pub payload: String,

    /// The replaced message, if any (name only).
    pub replaced: Option<String>,
}

impl<Payload: DataLike> Default for Message<Payload> {
    fn default() -> Self {
        Message {
            order: MessageOrder::Unordered,
            source_index: usize::max_value(),
            target_index: usize::max_value(),
            payload: Default::default(),
            replaced: None,
        }
    }
}

/// An indicator that something is invalid.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Invalid<MessageId: IndexLike> {
    Configuration(&'static str),
    Agent(usize, &'static str),
    Message(MessageId, &'static str),
}

impl<MessageId: IndexLike> Default for Invalid<MessageId> {
    fn default() -> Self {
        Invalid::Configuration("you should not be seeing this")
    }
}

/// A complete system configuration.
///
/// We will have a *lot* of these, so keeping their size down and avoiding heap memory as much as
/// possible is critical. The maximal sizes were chosen so that the configuration plus its memoized
/// identifier will fit together inside exactly one cache lines, which should make this more
/// cache-friendly when placed inside a hash table.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct Configuration<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
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
    > Default for Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>
{
    fn default() -> Self {
        Configuration {
            state_ids: [StateId::invalid(); MAX_AGENTS],
            message_counts: [MessageIndex::invalid(); MAX_AGENTS],
            message_ids: [MessageId::invalid(); MAX_MESSAGES],
            invalid_id: InvalidId::invalid(),
            immediate_index: MessageIndex::invalid(),
        }
    }
}

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
            match self.immediate_index.to_usize() {
                immediate_index if immediate_index > message_index => {
                    self.immediate_index.decr(); // NOT TESTED
                }
                immediate_index if immediate_index == message_index => {
                    self.immediate_index = MessageIndex::invalid();
                }
                _ => {}
            }
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
            message_index = next_message_index;
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
            message_id
        } else {
            self.immediate_index().unwrap_or_else(MessageId::invalid)
        };

        self.message_ids.sort();

        if immediate_index.is_valid() {
            let immediate_index = self
                .message_ids
                .iter()
                .position(|&message_id| message_id == immediate_index)
                .unwrap();
            self.immediate_index = MessageIndex::from_usize(immediate_index);
        }
    }

    /// Return whether there is an immediate message.
    fn has_immediate(&self) -> bool {
        self.immediate_index.is_valid()
    }

    /// Return the immediate message identifier, if any.
    fn immediate_index(&self) -> Option<MessageId> {
        if self.has_immediate() {
            Some(self.message_ids[self.immediate_index.to_usize()])
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
pub struct Outgoing<ConfigurationId: IndexLike> {
    /// The identifier of the target configuration.
    pub to_configuration_id: ConfigurationId,

    /// The index of the message of the source configuration that was delivered to its target agent
    /// to reach the target configuration.
    pub delivered_message_index: MessageIndex,
}

/// A transition to a given configuration.
#[derive(Copy, Clone, Debug)]
pub struct Incoming<ConfigurationId: IndexLike> {
    /// The identifier of the source configuration.
    pub from_configuration_id: ConfigurationId,

    /// The index of the message of the source configuration that was delivered to its target agent
    /// to reach the target configuration.
    pub delivered_message_index: MessageIndex,
}

/// Specify the number of threads to use.
pub enum Threads {
    /// Use all the logical processors.
    Logical,

    /// Use all the physical processors (ignore hyper-threading).
    Physical,

    /// Use a specific number of processors.
    Count(usize),
}

impl Threads {
    /// Get the actual number of threads to use.
    pub fn count(&self) -> usize {
        match self {
            Threads::Logical => num_cpus::get(),
            Threads::Physical => num_cpus::get_physical(),
            Threads::Count(count) => *count,
        }
    }
}

/// A complete model.
pub struct Model<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: DataLike,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The type of each agent.
    pub agent_types: Vec<<Self as MetaModel>::AgentTypeArc>,

    /// The label of each agent.
    pub agent_labels: Vec<Arc<String>>,

    /// The first index of the same type of each agent.
    pub first_indices: Vec<usize>,

    /// Validation functions for the configuration.
    pub validators: Vec<<Self as MetaModel>::Validator>,

    /// Memoization of the configurations.
    pub configurations: Memoize<<Self as MetaModel>::Configuration, ConfigurationId>,

    /// Memoization of the in-flight messages.
    pub messages: Memoize<Message<Payload>, MessageId>,

    /// Map the full message identifier to the terse message identifier.
    pub terse_of_message_id: RwLock<Vec<MessageId>>,

    /// Map the terse message identifier to the full message identifier.
    pub message_of_terse_id: RwLock<Vec<MessageId>>,

    /// Map ordered message identifiers to their smaller order.
    pub decr_order_messages: SccHashMap<MessageId, MessageId, RandomState>,

    /// Map ordered message identifiers to their larger order.
    pub incr_order_messages: SccHashMap<MessageId, MessageId, RandomState>,

    /// Memoization of the invalid conditions.
    pub invalids: Memoize<<Self as MetaModel>::Invalid, InvalidId>,

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

    /// Whether to print each new configuration as we reach it.
    pub print_progress_every: usize,

    /// Whether we'll be testing if the initial configuration is reachable from every configuration.
    pub ensure_init_is_reachable: bool,

    /// Whether to allow for invalid configurations.
    pub allow_invalid_configurations: bool,

    /// The number of threads to use for computing the model's configurations.
    ///
    /// If zero, uses all the available processors.
    pub threads: Threads,

    /// Named conditions on a configuration.
    pub conditions: SccHashMap<String, (<Self as MetaModel>::Condition, &'static str), RandomState>,
}

/// How a message relates to the previous configuration.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum PrevMessage {
    /// Not applicable.
    NotApplicable,

    /// The message is newly sent and did not exist in the previous configuration.
    NotThere,

    /// The message existed in the previous configuration in some index (possibly with a different order).
    Kept(usize),

    /// The message replaced another message in the previous configuration in some index.
    Replaced(usize),
}

/// How a message relates to the next configuration.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum NextMessage {
    /// Not applicable.
    NotApplicable,

    /// The message will be delivered to reach the next configuration.
    Delivered,

    /// The message will existed in the next configuration in some index (possibly with a different order).
    Kept(usize),

    /// The message will be replaced by another message in the next configuration in some index.
    Replaced(usize),
}

/// The additional control timelines associated with a specific agent.
#[derive(Clone, Debug)]
pub struct AgentTimelines {
    /// The indices of the control timelines to the left of the agent, ordered from closer to
    /// further.
    left: Vec<usize>,

    /// The indices of the control timelines to the right of the agent, ordered from closer to
    /// further.
    right: Vec<usize>,
}

/// The current state of a sequence diagram.
#[derive(Clone, Debug)]
pub struct SequenceState<MessageId: IndexLike, const MAX_AGENTS: usize, const MAX_MESSAGES: usize> {
    /// For each timeline, the index of the message it contains (in the current configuration).
    pub timelines: Vec<Option<usize>>,

    /// For each message in the current configuration, the timeline it is on.
    /// This may be `None` if the message was delivered in the very next transition
    /// (always true for immediate messages).
    pub message_timelines: StdHashMap<MessageId, usize>,

    /// The additional control timelines of each agent.
    pub agents_timelines: Vec<AgentTimelines>,
}

/// A transition between configurations along a path.
#[derive(Debug)]
pub struct PathTransition<
    MessageId: IndexLike,
    ConfigurationId: IndexLike,
    const MAX_MESSAGES: usize,
> {
    /// The source configuration identifier.
    pub from_configuration_id: ConfigurationId,

    /// How all the messages in the source configuration relate to the messages in the target
    /// configuration.
    pub from_next_messages: [NextMessage; MAX_MESSAGES],

    /// The index of the delivered message.
    pub delivered_message_index: Option<MessageIndex>,

    /// The identifier of the delivered message.
    pub delivered_message_id: Option<MessageId>,

    /// The agent that received the message.
    pub agent_index: usize,

    /// The target configuration identifier.
    pub to_configuration_id: ConfigurationId,

    /// How all the messages in the target configuration relate to the messages in the source
    /// configuration.
    pub to_prev_messages: [PrevMessage; MAX_MESSAGES],

    /// The name of the condition the target configuration satisfies.
    pub to_condition_name: Option<String>,
}

/// Control appearence of state graphs.
#[derive(Debug)]
pub struct Condense {
    /// Only use names, ignore details of state and payload.
    names_only: bool,

    /// Merge all agent instances.
    merge_instances: bool,

    /// Only consider the final value of a replaced message.
    final_replaced: bool,
}

impl Condense {
    /// Test whether we are condensing anything.
    pub fn is_active(&self) -> bool {
        self.names_only || self.merge_instances || self.final_replaced
    }
}

/// Identify related set of transition between agent states in the diagram.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
pub struct AgentStateTransitionContext<StateId: IndexLike> {
    /// The source configuration identifier.
    pub from_state_id: StateId,

    /// Whether the agent starting state was deferring.
    pub from_is_deferring: bool,

    /// The target configuration identifier.
    pub to_state_id: StateId,

    /// Whether the agent end state was deferring.
    pub to_is_deferring: bool,
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

    /// A condition on model configurations.
    type Condition;

    /// A transition along a path between configurations.
    type PathTransition;

    /// Identify a related set of transitions between agent states in the diagram.
    type AgentStateTransitionContext;

    /// The collection of all state transitions in the states diagram with the sent messages.
    type AgentStateTransitions;

    /// The sent messages indexed by the delivered messages for a transitions between two agent
    /// states.
    type AgentStateSentByDelivered;

    /// The state of a sequence diagram.
    type SequenceState;
}

// BEGIN MAYBE TESTED

/// The context for processing event handling by an agent.
#[derive(Clone)]
pub struct Context<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: DataLike,
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
    agent_type: Arc<dyn AgentType<StateId, Payload> + Send + Sync>,

    /// The index of the source agent in its type.
    agent_instance: usize,

    /// The identifier of the state of the agent when handling the event.
    agent_from_state_id: StateId,

    /// The incoming transition into the new configuration to be generated.
    incoming: Incoming<ConfigurationId>,

    /// The configuration when delivering the event.
    from_configuration: Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,

    /// Incrementally updated to become the target configuration.
    to_configuration: Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,
}

// END MAYBE TESTED

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: DataLike,
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

    type AgentTypeArc = Arc<dyn AgentType<StateId, Payload> + Send + Sync>;
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
    type Condition = fn(&Self, ConfigurationId) -> bool;
    type PathTransition = PathTransition<MessageId, ConfigurationId, MAX_MESSAGES>;
    type AgentStateTransitionContext = AgentStateTransitionContext<StateId>;
    type AgentStateTransitions = StdHashMap<
        AgentStateTransitionContext<StateId>,
        StdHashMap<Vec<MessageId>, Vec<MessageId>>,
    >;
    type AgentStateSentByDelivered = StdHashMap<Vec<MessageId>, Vec<Vec<MessageId>>>;
    type SequenceState = SequenceState<MessageId, MAX_AGENTS, MAX_MESSAGES>;
}

fn is_init<Model, ConfigurationId: IndexLike>(
    _model: &Model,
    configuration_id: ConfigurationId,
) -> bool {
    configuration_id.to_usize() == 0
}

// BEGIN NOT TESTED
fn is_valid<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: DataLike,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
>(
    model: &Model<
        StateId,
        MessageId,
        InvalidId,
        ConfigurationId,
        Payload,
        MAX_AGENTS,
        MAX_MESSAGES,
    >,
    configuration_id: ConfigurationId,
) -> bool {
    !model
        .configurations
        .get(configuration_id)
        .invalid_id
        .is_valid()
}

// END NOT TESTED

fn has_replacement<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: DataLike,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
>(
    model: &Model<
        StateId,
        MessageId,
        InvalidId,
        ConfigurationId,
        Payload,
        MAX_AGENTS,
        MAX_MESSAGES,
    >,
    configuration_id: ConfigurationId,
) -> bool {
    model
        .configurations
        .get(configuration_id)
        .message_ids
        .iter()
        .take_while(|message_id| message_id.is_valid())
        .map(|message_id| model.messages.get(*message_id))
        .any(|message| message.replaced.is_some())
}

fn has_messages_count<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: DataLike,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
    const MESSAGES_COUNT: usize,
>(
    model: &Model<
        StateId,
        MessageId,
        InvalidId,
        ConfigurationId,
        Payload,
        MAX_AGENTS,
        MAX_MESSAGES,
    >,
    configuration_id: ConfigurationId,
) -> bool {
    model
        .configurations
        .get(configuration_id)
        .message_ids
        .iter()
        .take_while(|message_id| message_id.is_valid())
        .count()
        == MESSAGES_COUNT
}

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: DataLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > Model<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>
{
    /// Create a new model, without computing anything yet.
    ///
    /// This allows querying the model for the `agent_index` of all the agents to use the results as
    /// a target for messages.
    pub fn new(
        size: usize,
        last_agent_type: Arc<dyn AgentType<StateId, Payload> + Send + Sync>,
        validators: Vec<<Self as MetaModel>::Validator>,
    ) -> Self {
        assert!(
            MAX_MESSAGES < MessageIndex::invalid().to_usize(),
            "MAX_MESSAGES {} is too large, must be less than {}",
            MAX_MESSAGES,
            MessageIndex::invalid() // NOT TESTED
        );

        let mut agent_types: Vec<<Self as MetaModel>::AgentTypeArc> = vec![];
        let mut first_indices: Vec<usize> = vec![];
        let mut agent_labels: Vec<Arc<String>> = vec![];

        Self::collect_agent_types(
            last_agent_type,
            &mut agent_types,
            &mut first_indices,
            &mut agent_labels,
        );

        let model = Self {
            agent_types,
            agent_labels,
            first_indices,
            validators,
            configurations: Memoize::new(size, usize::max_value()),
            messages: Memoize::new(
                MessageId::invalid().to_usize(),
                MessageId::invalid().to_usize(),
            ),
            terse_of_message_id: RwLock::new(vec![]),
            message_of_terse_id: RwLock::new(vec![]),
            decr_order_messages: SccHashMap::new(
                MessageId::invalid().to_usize(),
                RandomState::new(),
            ),
            incr_order_messages: SccHashMap::new(
                MessageId::invalid().to_usize(),
                RandomState::new(),
            ),
            invalids: Memoize::new(
                InvalidId::invalid().to_usize(),
                InvalidId::invalid().to_usize(),
            ),
            outgoings: RwLock::new(Vec::new()),
            incomings: RwLock::new(Vec::new()),
            max_message_string_size: RwLock::new(0),
            max_invalid_string_size: RwLock::new(0),
            max_configuration_string_size: RwLock::new(0),
            print_progress_every: 0,
            ensure_init_is_reachable: false,
            allow_invalid_configurations: false,
            threads: Threads::Physical,
            conditions: SccHashMap::new(128, RandomState::new()),
        };

        model.add_conditions();

        model
    }

    fn add_conditions(&self) {
        self.add_condition("INIT", is_init, "matches the initial configuration");
        self.add_condition(
            "VALID",
            is_valid,
            "matches any valid configuration (is typically negated)",
        );
        self.add_condition(
            "REPLACE",
            has_replacement,
            "matches a configuration with a replaced message",
        );
        self.add_condition(
            "0MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                0,
            >,
            "matches any configuration with no in-flight messages",
        );
        self.add_condition(
            "1MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                1,
            >,
            "matches any configuration with a single in-flight message",
        );
        self.add_condition(
            "2MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                2,
            >,
            "matches any configuration with 2 in-flight messages",
        );
        self.add_condition(
            "3MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                3,
            >,
            "matches any configuration with 3 in-flight messages",
        );
        self.add_condition(
            "4MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                4,
            >,
            "matches any configuration with 4 in-flight messages",
        );
        self.add_condition(
            "5MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                5,
            >,
            "matches any configuration with 5 in-flight messages",
        );
        self.add_condition(
            "6MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                6,
            >,
            "matches any configuration with 6 in-flight messages",
        );
        self.add_condition(
            "7MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                7,
            >,
            "matches any configuration with 7 in-flight messages",
        );
        self.add_condition(
            "8MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                8,
            >,
            "matches any configuration with 8 in-flight messages",
        );
        self.add_condition(
            "9MSG",
            has_messages_count::<
                StateId,
                MessageId,
                InvalidId,
                ConfigurationId,
                Payload,
                MAX_AGENTS,
                MAX_MESSAGES,
                9,
            >,
            "matches any configuration with 9 in-flight messages",
        );
    }

    pub fn add_condition(
        &self,
        name: &'static str,
        condition: <Self as MetaModel>::Condition,
        help: &'static str,
    ) {
        self.conditions.upsert(name.to_string(), (condition, help));
    }

    fn collect_agent_types(
        last_agent_type: Arc<dyn AgentType<StateId, Payload> + Send + Sync>,
        mut agent_types: &mut Vec<<Self as MetaModel>::AgentTypeArc>,
        mut first_indices: &mut Vec<usize>,
        mut agent_labels: &mut Vec<Arc<String>>,
    ) {
        if let Some(prev_agent_type) = last_agent_type.prev_agent_type() {
            Self::collect_agent_types(
                prev_agent_type,
                &mut agent_types,
                &mut first_indices,
                &mut agent_labels,
            );
        }

        let count = last_agent_type.instances_count();
        assert!(
            count > 0,
            "zero instances requested for the type {}",
            last_agent_type.name() // NOT TESTED
        );

        let first_index = first_indices.len();
        assert!(first_index == last_agent_type.first_index());

        for instance in 0..count {
            first_indices.push(first_index);
            agent_types.push(last_agent_type.clone());
            let agent_label = if last_agent_type.is_singleton() {
                debug_assert!(instance == 0);
                last_agent_type.name().to_string()
            } else {
                format!("{}({})", last_agent_type.name(), instance)
            };
            agent_labels.push(Arc::new(agent_label));
        }
    }

    /// Compute all the configurations of the model, if needed.
    pub fn compute(&self) {
        if !self.configurations.is_empty() {
            return;
        }

        let initial_configuration = Configuration {
            state_ids: [StateId::from_usize(0); MAX_AGENTS],
            message_counts: [MessageIndex::from_usize(0); MAX_AGENTS],
            message_ids: [MessageId::invalid(); MAX_MESSAGES],
            invalid_id: InvalidId::invalid(),
            immediate_index: MessageIndex::invalid(),
        };
        let stored = self.store_configuration(initial_configuration);

        ThreadPoolBuilder::new()
            .num_threads(self.threads.count())
            .build()
            .unwrap()
            .install(|| {
                scope(|parallel_scope| self.explore_configuration(parallel_scope, stored.id));
            });

        if self.ensure_init_is_reachable {
            self.assert_init_is_reachable();
        }
    }

    fn reach_configuration<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        mut context: <Self as MetaModel>::Context,
    ) {
        self.validate_configuration(&mut context);

        if !self.allow_invalid_configurations && context.to_configuration.invalid_id.is_valid() {
            // BEGIN NOT TESTED
            panic!(
                "reached an invalid configuration {}\n\
                   by the {}\n\
                   from the valid configuration {}",
                self.display_configuration_id(context.incoming.from_configuration_id),
                self.event_label(context.delivered_message_id),
                self.display_configuration(&context.to_configuration)
            );
            // END NOT TESTED
        }

        let stored = self.store_configuration(context.to_configuration);

        let to_configuration_id = stored.id;
        if self.ensure_init_is_reachable {
            self.incomings.read()[to_configuration_id.to_usize()]
                .write()
                .push(context.incoming);
        }

        let from_configuration_id = context.incoming.from_configuration_id;
        let outgoing = Outgoing {
            to_configuration_id,
            delivered_message_index: context.incoming.delivered_message_index,
        };

        self.outgoings.read()[from_configuration_id.to_usize()]
            .write()
            .push(outgoing);

        if stored.is_new {
            parallel_scope
                .spawn(move |same_scope| self.explore_configuration(same_scope, stored.id));
        }
    }

    fn explore_configuration<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        configuration_id: ConfigurationId,
    ) {
        let configuration = self.configurations.get(configuration_id);
        if self.print_progress_every > 0
            && configuration_id.to_usize() % self.print_progress_every == 0
        {
            eprintln!(
                "{} - {}",
                configuration_id.to_usize(),
                self.display_configuration(&configuration)
            );
        }

        let messages_count = if configuration.has_immediate() {
            1
        } else {
            configuration
                .message_ids
                .iter()
                .position(|&message_id| !message_id.is_valid())
                .unwrap_or(MAX_MESSAGES)
        };
        let events_count = self.agents_count() + messages_count;

        (0..events_count).into_par_iter().for_each(|event_index| {
            if event_index < self.agents_count() {
                self.deliver_time_event(
                    parallel_scope,
                    configuration_id,
                    configuration,
                    event_index,
                );
            } else if configuration.has_immediate() {
                debug_assert!(event_index == self.agents_count());
                self.deliver_message_event(
                    parallel_scope,
                    configuration_id,
                    configuration,
                    configuration.immediate_index.to_usize(),
                )
            } else {
                self.deliver_message_event(
                    parallel_scope,
                    configuration_id,
                    configuration,
                    event_index - self.agents_count(),
                );
            }
        });
    }

    fn deliver_time_event<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        from_configuration_id: ConfigurationId,
        from_configuration: <Self as MetaModel>::Configuration,
        agent_index: usize,
    ) {
        let agent_from_state_id = from_configuration.state_ids[agent_index];
        let agent_type = self.agent_types[agent_index].clone();
        let agent_instance = self.agent_instance(agent_index);
        let reaction = agent_type.pass_time(agent_instance, &from_configuration.state_ids);

        /*
        eprintln!("{} - FROM: {}", current_thread_name!(), self.display_configuration(&from_configuration));
        eprintln!("{} - BY: {}", current_thread_name!(), self.event_label(None));
        eprintln!("{} - REACTION: {:?}", current_thread_name!(), reaction);
        */

        if reaction == Reaction::Ignore {
            return;
        }

        let incoming = Incoming {
            from_configuration_id,
            delivered_message_index: MessageIndex::invalid(),
        };

        let context = Context {
            delivered_message_id: None,
            is_immediate: false,
            agent_index,
            agent_type,
            agent_instance,
            agent_from_state_id,
            incoming,
            from_configuration,
            to_configuration: from_configuration,
        };
        self.process_reaction(parallel_scope, context, reaction);
    }

    fn deliver_message_event<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        from_configuration_id: ConfigurationId,
        from_configuration: <Self as MetaModel>::Configuration,
        message_index: usize,
    ) {
        let message_id = from_configuration.message_ids[message_index];

        let (source_index, target_index, payload, is_immediate) = {
            let message = self.messages.get(message_id);
            if let MessageOrder::Ordered(order) = message.order {
                if order.to_usize() > 0 {
                    return;
                }
            }
            (
                message.source_index,
                message.target_index,
                message.payload,
                message.order == MessageOrder::Immediate,
            )
        };

        let target_instance = self.agent_instance(target_index);
        let target_from_state_id = from_configuration.state_ids[target_index];
        let target_type = self.agent_types[target_index].clone();
        let reaction =
            target_type.receive_message(target_instance, &from_configuration.state_ids, &payload);

        /*
        eprintln!("{} - FROM: {}", current_thread_name!(), self.display_configuration(&from_configuration));
        eprintln!("{} - BY: {}", current_thread_name!(), self.event_label(Some(message_id)));
        eprintln!("{} - REACTION: {:?}", current_thread_name!(), reaction);
        */

        let incoming = Incoming {
            from_configuration_id,
            delivered_message_index: MessageIndex::from_usize(message_index),
        };

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
            from_configuration,
            to_configuration,
        };
        self.process_reaction(parallel_scope, context, reaction);
    }

    fn process_reaction<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        context: <Self as MetaModel>::Context,
        reaction: <Self as MetaModel>::Reaction,
    ) {
        match reaction {
            Reaction::Unexpected => self.unexpected_message(context), // MAYBE TESTED
            Reaction::Defer => self.defer_message(context),
            Reaction::Ignore => self.ignore_message(parallel_scope, context),
            Reaction::Do1(action1) => self.perform_action(parallel_scope, context, action1),

            Reaction::Do1Of2(action1, action2) => {
                self.perform_action(parallel_scope, context.clone(), action1);
                self.perform_action(parallel_scope, context, action2);
            }

            // BEGIN NOT TESTED
            Reaction::Do1Of3(action1, action2, action3) => {
                self.perform_action(parallel_scope, context.clone(), action1);
                self.perform_action(parallel_scope, context.clone(), action2);
                self.perform_action(parallel_scope, context, action3);
            }

            Reaction::Do1Of4(action1, action2, action3, action4) => {
                self.perform_action(parallel_scope, context.clone(), action1);
                self.perform_action(parallel_scope, context.clone(), action2);
                self.perform_action(parallel_scope, context.clone(), action3);
                self.perform_action(parallel_scope, context, action4);
            } // END NOT TESTED
        }
    }

    fn perform_action<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        mut context: <Self as MetaModel>::Context,
        action: <Self as MetaModel>::Action,
    ) {
        match action {
            Action::Defer => self.defer_message(context),
            Action::Ignore => self.ignore_message(parallel_scope, context), // NOT TESTED

            Action::Change(target_to_state_id) => {
                if target_to_state_id == context.agent_from_state_id {
                    self.ignore_message(parallel_scope, context); // NOT TESTED
                } else {
                    context
                        .to_configuration
                        .change_state(context.agent_index, target_to_state_id);
                    self.reach_configuration(parallel_scope, context);
                }
            }
            Action::Send1(emit1) => self.emit_transition(parallel_scope, context, emit1),
            Action::ChangeAndSend1(target_to_state_id, emit1) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(parallel_scope, context, emit1);
            }

            // BEGIN NOT TESTED
            Action::ChangeAndSend2(target_to_state_id, emit1, emit2) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(parallel_scope, context.clone(), emit1);
                self.emit_transition(parallel_scope, context, emit2);
            }

            Action::Send2(emit1, emit2) => {
                self.emit_transition(parallel_scope, context.clone(), emit1);
                self.emit_transition(parallel_scope, context, emit2);
            }

            Action::ChangeAndSend3(target_to_state_id, emit1, emit2, emit3) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(parallel_scope, context.clone(), emit1);
                self.emit_transition(parallel_scope, context.clone(), emit2);
                self.emit_transition(parallel_scope, context, emit3);
            }

            Action::Send3(emit1, emit2, emit3) => {
                self.emit_transition(parallel_scope, context.clone(), emit1);
                self.emit_transition(parallel_scope, context.clone(), emit2);
                self.emit_transition(parallel_scope, context, emit3);
            }

            Action::ChangeAndSend4(target_to_state_id, emit1, emit2, emit3, emit4) => {
                context
                    .to_configuration
                    .change_state(context.agent_index, target_to_state_id);
                self.emit_transition(parallel_scope, context.clone(), emit1);
                self.emit_transition(parallel_scope, context.clone(), emit2);
                self.emit_transition(parallel_scope, context.clone(), emit3);
                self.emit_transition(parallel_scope, context, emit4);
            }

            Action::Send4(emit1, emit2, emit3, emit4) => {
                self.emit_transition(parallel_scope, context.clone(), emit1);
                self.emit_transition(parallel_scope, context.clone(), emit2);
                self.emit_transition(parallel_scope, context.clone(), emit3);
                self.emit_transition(parallel_scope, context, emit4);
            } // END NOT TESTED
        }
    }

    fn emit_transition<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        mut context: <Self as MetaModel>::Context,
        emit: <Self as MetaModel>::Emit,
    ) {
        match emit {
            Emit::Immediate(payload, target_index) => {
                let message = Message {
                    order: MessageOrder::Immediate,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced: None,
                };
                self.emit_message(parallel_scope, context, message);
            }

            Emit::Unordered(payload, target_index) => {
                let message = Message {
                    order: MessageOrder::Unordered,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced: None,
                };
                self.emit_message(parallel_scope, context, message);
            }

            Emit::Ordered(payload, target_index) => {
                let message = self.ordered_message(
                    &context.to_configuration,
                    context.agent_index,
                    target_index,
                    payload,
                    None,
                );
                self.emit_message(parallel_scope, context, message);
            }

            // BEGIN NOT TESTED
            Emit::ImmediateReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(&mut context, callback, &payload, target_index);
                let message = Message {
                    order: MessageOrder::Immediate,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced,
                };
                self.emit_message(parallel_scope, context, message);
            }
            // END NOT TESTED
            Emit::UnorderedReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(&mut context, callback, &payload, target_index);
                let message = Message {
                    order: MessageOrder::Unordered,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced,
                };
                self.emit_message(parallel_scope, context, message);
            }

            // BEGIN NOT TESTED
            Emit::OrderedReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(&mut context, callback, &payload, target_index);
                let message = self.ordered_message(
                    &context.to_configuration,
                    context.agent_index,
                    target_index,
                    payload,
                    replaced,
                );
                self.emit_message(parallel_scope, context, message);
            } // END NOT TESTED
        }
    }

    fn ordered_message(
        &self,
        to_configuration: &<Self as MetaModel>::Configuration,
        source_index: usize,
        target_index: usize,
        payload: Payload,
        replaced: Option<Payload>,
    ) -> <Self as MetaModel>::Message {
        let mut order = {
            to_configuration
                .message_ids
                .iter()
                .take_while(|message_id| message_id.is_valid())
                .map(|message_id| self.messages.get(*message_id))
                .filter(|message| {
                    message.source_index == source_index
                        && message.target_index == target_index
                        && matches!(message.order, MessageOrder::Ordered(_))
                })
                .fold(0, |count, _message| count + 1)
        };

        let message = Message {
            order: MessageOrder::Ordered(MessageIndex::from_usize(order)),
            source_index,
            target_index,
            payload,
            replaced,
        };

        let mut message = message;
        let mut message_id = self.messages.store(message).id;
        while order > 0 {
            order -= 1;
            match self
                .decr_order_messages
                .insert(message_id, MessageId::from_usize(0))
            {
                Ok(result) => {
                    message.order = MessageOrder::Ordered(MessageIndex::from_usize(order));
                    let decr_message_id = self.messages.store(message).id;
                    *result.get().1 = decr_message_id;
                    assert!(self
                        .incr_order_messages
                        .insert(decr_message_id, message_id)
                        .is_ok());
                    message_id = decr_message_id;
                }
                Err(_) => break,
            }
        }

        message
    }

    fn replace_message(
        &self,
        context: &mut <Self as MetaModel>::Context,
        callback: fn(Option<Payload>) -> bool,
        payload: &Payload,
        target_index: usize,
    ) -> Option<Payload> {
        let replaced = {
            context
                .to_configuration
                .message_ids
                .iter()
                .take_while(|message_id| message_id.is_valid())
                .enumerate()
                .map(|(message_index, message_id)| (message_index, self.messages.get(*message_id)))
                .filter(|(_, message)| {
                    message.source_index == context.agent_index
                        && message.target_index == target_index
                        && callback(Some(message.payload))
                })
                .fold(None, |replaced, (message_index, message)| {
                    match replaced // MAYBE TESTED
                    {
                        None => Some((message_index, message)),
                        // BEGIN NOT TESTED
                        Some((_, ref conflict)) => {
                            let conflict_payload = format!("{}", conflict.payload);
                            let message_payload = format!("{}", message.payload);
                            let replacement_payload = format!("{}", payload);
                            let source_label = self.agent_labels[context.agent_index].clone();
                            let target_label = self.agent_labels[target_index].clone();
                            let event_label = self.event_label(context.delivered_message_id);
                            let from_state = context
                                .agent_type
                                .display_state(context.agent_from_state_id);
                            panic!(
                                "both the message {} \
                                 and the message {} \
                                 can be replaced by the ambiguous replacement message {} \
                                 sent to the agent {} \
                                 by the agent {} \
                                 in the state {} \
                                 when responding to the {}",
                                conflict_payload,
                                message_payload,
                                replacement_payload,
                                target_label,
                                source_label,
                                from_state,
                                event_label
                            );
                        } // END NOT TESTED
                    }
                })
                .map(|(message_index, message)| Some((message_index, message.payload)))
                .unwrap_or_else(|| {
                    if !callback(None) {
                        // BEGIN NOT TESTED
                        let replacement_payload = format!("{}", payload);
                        let source_label = self.agent_labels[context.agent_index].clone();
                        let target_label = self.agent_labels[target_index].clone();
                        let event_label = self.event_label(context.delivered_message_id);
                        let from_state = context
                            .agent_type
                            .display_state(context.agent_from_state_id);
                        panic!(
                            "nothing was replaced by the required replacement message {} \
                             sent to the agent {} \
                             by the agent {} \
                             in the state {} \
                             when responding to the {}",
                            replacement_payload,
                            target_label,
                            source_label,
                            from_state,
                            event_label
                        );
                        // END NOT TESTED
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

    fn remove_message(
        &self,
        configuration: &mut <Self as MetaModel>::Configuration,
        source: usize,
        message_index: usize,
    ) {
        let removed_message_id = configuration.message_ids[message_index];
        let (removed_source_index, removed_target_index, removed_order) = {
            let removed_message = self.messages.get(removed_message_id);
            if let MessageOrder::Ordered(removed_order) = removed_message.order {
                (
                    removed_message.source_index,
                    removed_message.target_index,
                    Some(removed_order),
                )
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
            let mut did_modify = false;
            for message_index in 0..MAX_MESSAGES {
                let message_id = configuration.message_ids[message_index];
                if !message_id.is_valid() {
                    break;
                }

                if message_id == removed_message_id {
                    continue;
                }

                let message = self.messages.get(message_id);
                if message.source_index != removed_source_index
                    || message.target_index != removed_target_index
                {
                    continue;
                }

                if let MessageOrder::Ordered(message_order) = message.order {
                    if message_order > removed_message_order {
                        configuration.message_ids[message_index] =
                            self.decr_message_id(message_id).unwrap();
                        did_modify = true;
                    }
                }
            }

            if did_modify {
                assert!(!configuration.immediate_index.is_valid());
                configuration.message_ids.sort();
            }
        }
    }

    fn decr_message_id(&self, message_id: MessageId) -> Option<MessageId> {
        self.decr_order_messages
            .get(&message_id)
            .map(|result| *result.get().1)
    }

    fn incr_message_id(&self, message_id: MessageId) -> Option<MessageId> {
        self.incr_order_messages
            .get(&message_id)
            .map(|result| *result.get().1)
    }

    fn emit_message<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        mut context: <Self as MetaModel>::Context,
        message: <Self as MetaModel>::Message,
    ) {
        let is_immediate = message.order == MessageOrder::Immediate;
        if is_immediate && context.to_configuration.has_immediate() {
            // BEGIN NOT TESTED
            panic!(
                "sending a second immediate message {} while in the configuration {}",
                self.display_message(&message),
                self.display_configuration(&context.to_configuration)
            );
            // END NOT TESTED
        }
        context
            .to_configuration
            .message_ids
            .iter()
            .take_while(|to_message_id| to_message_id.is_valid())
            .map(|to_message_id| self.messages.get(*to_message_id))
            .filter(|to_message| {
                to_message.source_index == message.source_index
                    && to_message.target_index == message.target_index
                    && to_message.payload == message.payload
            })
            .for_each(|to_message| {
                let event_label = self.event_label(context.delivered_message_id);
                let source_label = self.agent_labels[to_message.source_index].clone();
                let target_label = self.agent_labels[to_message.target_index].clone();
                let from_configuration = self.display_configuration(&context.from_configuration);
                panic!(
                    "the agent {} \
                     sends a duplicate message {} \
                     to the agent {} \
                     when responding to the {} \
                     while in the configuration {}",
                    source_label, to_message.payload, target_label, event_label, from_configuration
                );
            });
        let message_id = self.messages.store(message).id;
        context
            .to_configuration
            .add_message(context.agent_index, message_id, is_immediate);
        self.reach_configuration(parallel_scope, context);
    }

    // BEGIN NOT TESTED
    fn unexpected_message(&self, context: <Self as MetaModel>::Context) {
        let event_label = self.event_label(context.delivered_message_id);
        let agent_label = self.agent_labels[context.agent_index].clone();
        let from_configuration = self.display_configuration(&context.from_configuration);
        panic!(
            "the agent {} does not expect the {} while in the configuration {}",
            agent_label, event_label, from_configuration
        );
    }
    // END NOT TESTED

    fn ignore_message<'a>(
        &'a self,
        parallel_scope: &ParallelScope<'a>,
        context: <Self as MetaModel>::Context,
    ) {
        if context.incoming.delivered_message_index.is_valid() {
            self.reach_configuration(parallel_scope, context);
        }
    }

    fn defer_message(&self, context: <Self as MetaModel>::Context) {
        match context.delivered_message_id // MAYBE TESTED
        {
            None => {
                // BEGIN NOT TESTED
                let agent_label = self.agent_labels[context.agent_index].clone();
                let event_label = self.event_label(None);
                let from_state = context
                    .agent_type
                    .display_state(context.agent_from_state_id);
                panic!(
                    "the agent {} is deferring (should be ignoring) the {} while in the state {}",
                    agent_label, event_label, from_state
                );
                // END NOT TESTED
            }

            Some(delivered_message_id) => {
                if !context.agent_type.state_is_deferring(
                    context.agent_instance,
                    &context.from_configuration.state_ids,
                ) {
                    // BEGIN NOT TESTED
                    let agent_label = self.agent_labels[context.agent_index].clone();
                    let event_label = self.event_label(Some(delivered_message_id));
                    let from_state = context
                        .agent_type
                        .display_state(context.agent_from_state_id);
                    panic!("the agent {} is deferring (should be ignoring) the {} while in the state {}",
                           agent_label, event_label, from_state);
                    // END NOT TESTED
                }

                if context.is_immediate {
                    // BEGIN NOT TESTED
                    let agent_label = self.agent_labels[context.agent_index].clone();
                    let event_label = self.event_label(Some(delivered_message_id));
                    let from_state = context
                        .agent_type
                        .display_state(context.agent_from_state_id);
                    panic!(
                        "the agent {} is deferring the immediate {} while in the state {}",
                        agent_label, event_label, from_state
                    );
                    // END NOT TESTED
                }
            }
        }
    }

    fn validate_configuration(&self, context: &mut <Self as MetaModel>::Context) {
        if context.to_configuration.invalid_id.is_valid() {
            return;
        }

        if context.agent_index != usize::max_value() {
            if let Some(max_in_flight_messages) = context.agent_type.state_max_in_flight_messages(
                context.agent_instance,
                &context.from_configuration.state_ids,
            ) {
                let in_flight_messages =
                    context.to_configuration.message_counts[context.agent_index].to_usize();
                if in_flight_messages > max_in_flight_messages {
                    // BEGIN NOT TESTED
                    let agent_label = self.agent_labels[context.agent_index].clone();
                    let event_label = self.event_label(context.delivered_message_id);
                    let to_configuration = self.display_configuration(&context.to_configuration);
                    let from_configuration =
                        self.display_configuration(&context.from_configuration);

                    panic!(
                        "the agent {} sends too more messages {} then allowed {} when reacting to the {} by moving\n\
                        from the configuration {}\n\
                        into the configuration {}",
                        agent_label, in_flight_messages, max_in_flight_messages, event_label, from_configuration, to_configuration
                    );
                    // END NOT TESTED
                }
            }
        }

        for validator in self.validators.iter() {
            // BEGIN NOT TESTED
            if let Some(reason) = validator(&context.to_configuration) {
                let invalid = <Self as MetaModel>::Invalid::Configuration(reason);
                context.to_configuration.invalid_id = self.invalids.store(invalid).id;
                return;
            }
            // END NOT TESTED
        }
    }

    fn store_configuration(
        &self,
        configuration: <Self as MetaModel>::Configuration,
    ) -> Stored<ConfigurationId> {
        let stored = self.configurations.store(configuration);

        if stored.is_new {
            let remaining = self.outgoings.read().len() - stored.id.to_usize();
            if remaining < self.threads.count() {
                let mut outgoings = self.outgoings.write();

                let remaining = outgoings.len() - stored.id.to_usize();
                if remaining < self.threads.count() {
                    let additional = max(outgoings.len() / GROWTH_FACTOR, self.threads.count());

                    eprintln!(
                        "increasing size from {} to {}",
                        outgoings.len(),
                        outgoings.len() + additional
                    );

                    self.configurations.reserve(additional);

                    outgoings.reserve(additional);
                    for _ in 0..additional {
                        outgoings.push(RwLock::new(vec![]));
                    }

                    if self.ensure_init_is_reachable {
                        let mut incomings = self.incomings.write();
                        incomings.reserve(additional);
                        for _ in 0..additional {
                            incomings.push(RwLock::new(vec![]));
                        }
                    }
                }
            }
        }

        stored
    }

    /// Return the total number of agents.
    pub fn agents_count(&self) -> usize {
        self.agent_labels.len()
    }

    /// Return the agent type for agents of some name.
    pub fn agent_type(&self, name: &'static str) -> &<Self as MetaModel>::AgentTypeArc {
        self.agent_types
            .iter()
            .find(|agent_type| agent_type.name() == name)
            .unwrap_or_else(
                || panic!("looking for an unknown agent type {}", name), // MAYBE TESTED
            )
    }

    /// Return the index of the agent with the specified type name.
    ///
    /// If more than one agent of this type exist, also specify its index within its type.
    pub fn agent_index(&self, name: &'static str, instance: Option<usize>) -> usize {
        let agent_type = self.agent_type(name);
        if let Some(lookup_instance) = instance {
            assert!(lookup_instance < agent_type.instances_count(),
                    "out of bounds instance {} specified when locating an agent of type {} which has {} instances",
                    lookup_instance, name, agent_type.instances_count()); // NOT TESTED
            self.first_indices[agent_type.first_index()] + lookup_instance
        } else {
            assert!(
                agent_type.is_singleton(),
                "no instance specified when locating a singleton agent of type {}",
                name
            );
            self.first_indices[agent_type.first_index()]
        }
    }

    /// Return the index of the agent instance within its type.
    pub fn agent_instance(&self, agent_index: usize) -> usize {
        agent_index - self.first_indices[agent_index]
    }

    /// Return whether all the reachable configurations are valid.
    pub fn is_valid(&self) -> bool {
        self.invalids.is_empty()
    }

    fn event_label(&self, message_id: Option<MessageId>) -> String {
        match message_id // MAYBE TESTED
        {
            None => "time event".to_string(),
            Some(message_id) if !message_id.is_valid() => "time event".to_string(),
            Some(message_id) => format!("message {}", self.display_message_id(message_id))
        }
    }

    /// Display a message by its identifier.
    pub fn display_message_id(&self, message_id: MessageId) -> String {
        self.display_message(&self.messages.get(message_id))
    }

    /// Display a message.
    pub fn display_message(&self, message: &<Self as MetaModel>::Message) -> String {
        let max_message_string_size = *self.max_message_string_size.read();
        let mut string = String::with_capacity(max_message_string_size);

        string.push_str(&*self.agent_labels[message.source_index]);
        string.push_str(" -> ");

        self.push_message_payload(message, false, false, &mut string);

        string.push_str(" -> ");
        string.push_str(&*self.agent_labels[message.target_index]);

        string.shrink_to_fit();
        if string.len() > max_message_string_size {
            let mut max_message_string_size = self.max_message_string_size.write();
            if string.len() > *max_message_string_size {
                *max_message_string_size = string.len();
            }
        }

        string
    }

    /// Display a message in the sequence diagram.
    pub fn display_sequence_message(
        &self,
        message: &<Self as MetaModel>::Message,
        is_final: bool,
    ) -> String {
        let max_message_string_size = *self.max_message_string_size.read();
        let mut string = String::with_capacity(max_message_string_size);
        self.push_message_payload(message, true, is_final, &mut string);
        string.shrink_to_fit();
        string
    }

    /// Display a message.
    pub fn push_message_payload(
        &self,
        message: &<Self as MetaModel>::Message,
        is_sequence: bool,
        is_final: bool,
        string: &mut String,
    ) {
        match message.order {
            MessageOrder::Immediate => {
                if !is_sequence {
                    string.push_str("* ");
                }
            }
            MessageOrder::Unordered => {}
            MessageOrder::Ordered(order) => string.push_str(&format!("@{} ", order)), //
        }

        if let Some(ref replaced) = message.replaced {
            if !is_sequence {
                string.push_str(&format!("{} => ", replaced));
            } else if !is_final {
                string.push_str(RIGHT_DOUBLE_ARROW);
                string.push(' ');
            }
        }

        string.push_str(&format!("{}", message.payload));
    }

    // BEGIN NOT TESTED
    /// Display an invalid condition by its identifier.
    fn display_invalid_id(&self, invalid_id: InvalidId) -> String {
        self.display_invalid(&self.invalids.get(invalid_id))
    }

    /// Display an invalid condition.
    fn display_invalid(&self, invalid: &<Self as MetaModel>::Invalid) -> String {
        let max_invalid_string_size = *self.max_invalid_string_size.read();
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
                string.push_str(&self.display_message_id(*message_id));
                string.push_str(" because ");
                string.push_str(reason);
            }
        }

        string.shrink_to_fit();
        if string.len() > max_invalid_string_size {
            let mut max_invalid_string_size = self.max_invalid_string_size.write();
            if string.len() > *max_invalid_string_size {
                *max_invalid_string_size = string.len();
            }
        }

        string
    }

    // END NOT TESTED

    /// Display a configuration by its identifier.
    pub fn display_configuration_id(&self, configuration_id: ConfigurationId) -> String {
        self.display_configuration(&self.configurations.get(configuration_id))
    }

    /// Display a configuration.
    fn display_configuration(&self, configuration: &<Self as MetaModel>::Configuration) -> String {
        let max_configuration_string_size = *self.max_configuration_string_size.read();
        let mut string = String::with_capacity(max_configuration_string_size);

        let mut prefix = "";
        (0..self.agents_count()).for_each(|agent_index| {
            let agent_type = &self.agent_types[agent_index];
            let agent_label = &self.agent_labels[agent_index];
            let agent_state_id = configuration.state_ids[agent_index];
            string.push_str(prefix);
            string.push_str(agent_label);
            string.push(':');
            string.push_str(&agent_type.display_state(agent_state_id));
            prefix = " & ";
        });

        prefix = " | ";
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .for_each(|message_id| {
                string.push_str(prefix);
                string.push_str(&self.display_message_id(*message_id));
                prefix = " & ";
            });

        if configuration.invalid_id.is_valid() {
            // BEGIN NOT TESTED
            string.push_str(" ! ");
            string.push_str(&self.display_invalid_id(configuration.invalid_id));
            // END NOT TESTED
        }

        string.shrink_to_fit();
        if string.len() > max_configuration_string_size {
            let mut max_configuration_string_size = self.max_configuration_string_size.write();
            if string.len() > *max_configuration_string_size {
                *max_configuration_string_size = string.len();
            }
        }

        string
    }

    fn do_compute(&mut self, arg_matches: &ArgMatches) {
        let threads = arg_matches.value_of("threads").unwrap();
        self.threads = if threads == "PHYSICAL" {
            Threads::Physical // NOT TESTED
        } else if threads == "LOGICAL" {
            Threads::Logical // NOT TESTED
        } else {
            Threads::Count(usize::from_str(threads).expect("invalid threads count"))
        };
        if self.threads.count() == 0 {
            self.threads = Threads::Count(1); // NOT TESTED
        }

        let threads = arg_matches.value_of("progress").unwrap();
        self.print_progress_every = usize::from_str(threads).expect("invalid progress rate");
        self.allow_invalid_configurations = arg_matches.is_present("invalid");

        self.ensure_init_is_reachable = arg_matches.is_present("reachable");
        if self.ensure_init_is_reachable {
            let mut incomings = self.incomings.write();
            let incomings_capacity = incomings.capacity();
            incomings.reserve(self.outgoings.read().capacity() - incomings_capacity);
        }

        let size = self.configurations.capacity();

        if self.ensure_init_is_reachable {
            let mut incomings = self.incomings.write();
            if incomings.len() < size {
                let additional = size - incomings.len();
                incomings.reserve(additional);
                for _ in 0..additional {
                    incomings.push(RwLock::new(vec![]));
                }
            }
        }

        {
            let mut outgoings = self.outgoings.write();
            if outgoings.len() < size {
                let additional = size - outgoings.len();
                outgoings.reserve(additional);
                for _ in 0..additional {
                    outgoings.push(RwLock::new(vec![]));
                }
            }
        }

        self.compute();
    }

    fn assert_init_is_reachable(&self) {
        let mut reached_configurations_mask = vec![false; self.configurations.len()];
        let mut pending_configuration_ids: VecDeque<usize> = VecDeque::new();
        pending_configuration_ids.push_back(0);

        let incomings = self.incomings.read();

        while let Some(next_configuration_id) = pending_configuration_ids.pop_front() {
            if reached_configurations_mask[next_configuration_id] {
                continue;
            }
            reached_configurations_mask[next_configuration_id] = true;
            incomings[next_configuration_id]
                .read()
                .iter()
                .for_each(|incoming| {
                    pending_configuration_ids.push_back(incoming.from_configuration_id.to_usize());
                });
        }

        let mut unreachable_count = 0;
        reached_configurations_mask
            .iter()
            .enumerate()
            .filter(|(_, is_reached)| !*is_reached)
            .for_each(|(configuration_id, _)| {
                // BEGIN NOT TESTED
                eprintln!(
                    "there is no path back to initial state from the configuration {}",
                    self.display_configuration_id(ConfigurationId::from_usize(configuration_id))
                );
                unreachable_count += 1;
                // END NOT TESTED
            });
        assert!(
            unreachable_count == 0,
            "there is no path back to initial state from {} configurations",
            unreachable_count
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn find_closest_configuration_id(
        &self,
        from_configuration_id: ConfigurationId,
        from_name: &str,
        to_condition: <Self as MetaModel>::Condition,
        to_negated: bool,
        to_name: &str,
        pending_configuration_ids: &mut VecDeque<ConfigurationId>,
        prev_configuration_ids: &mut [ConfigurationId],
    ) -> ConfigurationId {
        pending_configuration_ids.clear();
        pending_configuration_ids.push_back(from_configuration_id);

        prev_configuration_ids.fill(ConfigurationId::invalid());

        let outgoings = self.outgoings.read();
        while let Some(next_configuration_id) = pending_configuration_ids.pop_front() {
            for outgoing in outgoings[next_configuration_id.to_usize()].read().iter() {
                let to_configuration_id = outgoing.to_configuration_id;
                if prev_configuration_ids[to_configuration_id.to_usize()].is_valid() {
                    continue;
                }
                prev_configuration_ids[to_configuration_id.to_usize()] = next_configuration_id;

                let mut is_condition = false;
                if next_configuration_id != from_configuration_id
                    || to_configuration_id != from_configuration_id
                {
                    is_condition = to_condition(&self, to_configuration_id);
                    if to_negated {
                        is_condition = !is_condition; // NOT TESTED
                    }
                }

                if is_condition {
                    return to_configuration_id;
                }

                pending_configuration_ids.push_back(to_configuration_id);
            }
        }

        // BEGIN NOT TESTED
        panic!(
            "could not find a path from the condition {} to the condition {}\n\
            starting from the configuration {}",
            from_name,
            to_name,
            self.display_configuration_id(from_configuration_id)
        );
        // END NOT TESTED
    }

    fn collect_path(
        &self,
        subcommand_name: &str,
        matches: &ArgMatches,
    ) -> Vec<<Self as MetaModel>::PathTransition> {
        let mut steps: Vec<(<Self as MetaModel>::Condition, bool, String)> = matches
            .values_of("CONDITION")
            .unwrap_or_else(|| {
                // BEGIN NOT TESTED
                panic!(
                    "the {} command requires at least two configuration conditions, got none",
                    subcommand_name
                );
                // END NOT TESTED
            })
            .map(|name| {
                let (key, negated) = match name.strip_prefix("!") {
                    None => (name, false),
                    Some(suffix) => (suffix, true), // NOT TESTED
                };
                if let Some(result) = self.conditions.get(&key.to_string()) {
                    (result.get().1 .0, negated, name.to_string())
                } else {
                    panic!("unknown configuration condition {}", name); // NOT TESTED
                }
            })
            .collect();

        assert!(
            steps.len() > 1,
            "the {} command requires at least two configuration conditions, got only one",
            subcommand_name
        );

        let mut prev_configuration_ids =
            vec![ConfigurationId::invalid(); self.configurations.len()];

        let mut pending_configuration_ids: VecDeque<ConfigurationId> = VecDeque::new();

        let initial_configuration_id = ConfigurationId::from_usize(0);

        let (first_condition, first_negated, first_name) = steps[0].clone();
        let mut start_at_init = first_condition(self, initial_configuration_id);
        if first_negated {
            start_at_init = !start_at_init; // NOT TESTED
        }

        let mut current_configuration_id = initial_configuration_id;
        let mut current_name = first_name.to_string();

        if start_at_init {
            steps.remove(0);
        } else {
            current_configuration_id = self.find_closest_configuration_id(
                initial_configuration_id,
                "INIT",
                first_condition,
                first_negated,
                &first_name,
                &mut pending_configuration_ids,
                &mut prev_configuration_ids,
            );
        }

        let mut path = vec![PathTransition {
            from_configuration_id: current_configuration_id,
            from_next_messages: [NextMessage::NotApplicable; MAX_MESSAGES],
            delivered_message_index: None,
            delivered_message_id: None,
            agent_index: usize::max_value(),
            to_prev_messages: [PrevMessage::NotApplicable; MAX_MESSAGES],
            to_configuration_id: current_configuration_id,
            to_condition_name: Some(current_name.to_string()),
        }];

        steps
            .iter()
            .for_each(|(next_condition, next_negated, next_name)| {
                let next_configuration_id = self.find_closest_configuration_id(
                    current_configuration_id,
                    &current_name,
                    *next_condition,
                    *next_negated,
                    next_name,
                    &mut pending_configuration_ids,
                    &mut prev_configuration_ids,
                );
                self.collect_path_step(
                    current_configuration_id,
                    next_configuration_id,
                    Some(next_name),
                    &prev_configuration_ids,
                    &mut path,
                );
                current_configuration_id = next_configuration_id;
                current_name = next_name.to_string();
            });

        path
    }

    fn collect_path_step(
        &self,
        from_configuration_id: ConfigurationId,
        to_configuration_id: ConfigurationId,
        to_name: Option<&str>,
        prev_configuration_ids: &[ConfigurationId],
        mut path: &mut Vec<<Self as MetaModel>::PathTransition>,
    ) {
        let prev_configuration_id = prev_configuration_ids[to_configuration_id.to_usize()];
        assert!(prev_configuration_id.is_valid());

        if prev_configuration_id != from_configuration_id {
            self.collect_path_step(
                from_configuration_id,
                prev_configuration_id,
                None,
                &prev_configuration_ids,
                &mut path,
            );
        }
        let from_configuration_id = prev_configuration_id;

        let all_outgoings = self.outgoings.read();
        let from_outgoings = all_outgoings[from_configuration_id.to_usize()].read();
        let outgoing_index = from_outgoings
            .iter()
            .position(|outgoing| outgoing.to_configuration_id == to_configuration_id)
            .unwrap();
        let outgoing = from_outgoings[outgoing_index];

        let from_configuration = self.configurations.get(from_configuration_id);
        let to_configuration = self.configurations.get(to_configuration_id);

        let mut agent_index: Option<usize> = None;
        let mut delivered_message_id: Option<MessageId> = None;
        let mut from_next_messages = [NextMessage::NotApplicable; MAX_MESSAGES];
        from_configuration
            .message_ids
            .iter()
            .take_while(|from_message_id| from_message_id.is_valid())
            .enumerate()
            .for_each(|(from_message_index, from_message_id)| {
                from_next_messages[from_message_index] =
                    if from_message_index == outgoing.delivered_message_index.to_usize() {
                        let from_message = self.messages.get(*from_message_id);
                        agent_index = Some(from_message.target_index);
                        delivered_message_id = Some(*from_message_id);
                        NextMessage::Delivered
                    } else if let Some(to_message_index) =
                        self.from_message_kept_in_transition(*from_message_id, &to_configuration)
                    {
                        NextMessage::Kept(to_message_index)
                    } else {
                        let from_message = self.messages.get(*from_message_id);
                        let to_message_index = to_configuration
                            .message_ids
                            .iter()
                            .take_while(|to_message_id| to_message_id.is_valid())
                            .position(|to_message_id| {
                                let to_message = self.messages.get(*to_message_id);
                                to_message.source_index == from_message.source_index
                                    && to_message.target_index == from_message.target_index
                                    && to_message.replaced == Some(from_message.payload)
                            })
                            .unwrap();
                        NextMessage::Replaced(to_message_index)
                    }
            });

        let mut to_prev_messages = [PrevMessage::NotApplicable; MAX_MESSAGES];
        to_configuration
            .message_ids
            .iter()
            .take_while(|to_message_id| to_message_id.is_valid())
            .enumerate()
            .for_each(|(to_message_index, to_message_id)| {
                to_prev_messages[to_message_index] = if let Some(from_message_index) = self
                    .to_message_kept_in_transition(
                        *to_message_id,
                        &from_configuration,
                        outgoing.delivered_message_index,
                    ) {
                    PrevMessage::Kept(from_message_index)
                } else {
                    let to_message = self.messages.get(*to_message_id);
                    assert!(agent_index.is_none() || agent_index == Some(to_message.source_index));
                    agent_index = Some(to_message.source_index);

                    if let Some(replaced) = to_message.replaced {
                        let from_message_index = from_configuration
                            .message_ids
                            .iter()
                            .take_while(|from_message_id| from_message_id.is_valid())
                            .position(|from_message_id| {
                                let from_message = self.messages.get(*from_message_id);
                                from_message.source_index == to_message.source_index
                                    && from_message.target_index == to_message.target_index
                                    && from_message.payload == replaced
                            })
                            .unwrap();
                        PrevMessage::Replaced(from_message_index)
                    } else {
                        PrevMessage::NotThere
                    }
                };
            });

        if agent_index.is_none() {
            // BEGIN NOT TESTED
            agent_index = from_configuration
                .state_ids
                .iter()
                .take_while(|state_id| state_id.is_valid())
                .zip(to_configuration.state_ids.iter())
                .position(|(from_state_id, to_state_id)| from_state_id != to_state_id)
            // END NOT TESTED
        }
        assert!(
            agent_index.is_some(),
            "could not identify active agent transitioning from {} to {}",
            self.display_configuration(&from_configuration),
            self.display_configuration(&to_configuration)
        );

        let delivered_message_index = if !outgoing.delivered_message_index.is_valid() {
            None
        } else {
            Some(outgoing.delivered_message_index)
        };

        path.push(PathTransition {
            from_configuration_id,
            from_next_messages,
            delivered_message_index,
            delivered_message_id,
            agent_index: agent_index.unwrap(),
            to_prev_messages,
            to_configuration_id,
            to_condition_name: to_name.map(str::to_string),
        });
    }

    fn print_path(&self, path: &[<Self as MetaModel>::PathTransition], stdout: &mut dyn Write) {
        path.iter().for_each(|transition| {
            if transition.to_configuration_id != transition.from_configuration_id {
                writeln!(
                    stdout,
                    "BY {}",
                    self.event_label(transition.delivered_message_id)
                )
                .unwrap();
            }

            match &transition.to_condition_name {
                Some(condition_name) => writeln!(
                    stdout,
                    "{} {}",
                    condition_name,
                    self.display_configuration_id(transition.to_configuration_id)
                )
                .unwrap(),
                None => writeln!(
                    stdout,
                    "TO {}",
                    self.display_configuration_id(transition.to_configuration_id)
                )
                .unwrap(),
            }
        });
    }
}

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: DataLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > Model<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>
{
    fn compute_terse(&self, condense: &Condense) {
        let mut terse_of_message_id = self.terse_of_message_id.write();
        let mut message_of_terse_id = self.message_of_terse_id.write();
        assert!(terse_of_message_id.is_empty());
        assert!(message_of_terse_id.is_empty());

        if condense.names_only {
            for (agent_index, agent_type) in self.agent_types.iter().enumerate() {
                if agent_index == agent_type.first_index() {
                    agent_type.compute_terse();
                }
            }
        }

        let mut seen_messages: StdHashMap<TerseMessage, usize> = StdHashMap::new();
        seen_messages.reserve(self.messages.len());
        terse_of_message_id.reserve(self.messages.len());
        message_of_terse_id.reserve(self.messages.len());

        for message_id in 0..self.messages.len() {
            let message = self.messages.get(MessageId::from_usize(message_id));

            let source_index = if condense.merge_instances {
                self.agent_types[message.source_index].first_index()
            } else {
                message.source_index
            };

            let target_index = if condense.merge_instances {
                self.agent_types[message.target_index].first_index()
            } else {
                message.target_index
            };

            let payload = if condense.names_only {
                message.payload.name()
            } else {
                message.payload.to_string()
            };

            let replaced = if message.replaced.is_none() || condense.final_replaced {
                None
            } else if condense.names_only {
                Some(message.replaced.unwrap().name())
            } else {
                Some(message.replaced.unwrap().to_string())
            };

            let terse_message = TerseMessage {
                is_immediate: message.order == MessageOrder::Immediate,
                source_index,
                target_index,
                payload,
                replaced,
            };

            let todox = terse_message.clone();

            let terse_id = *seen_messages.entry(terse_message).or_insert_with(|| {
                let next_terse_id = message_of_terse_id.len();
                message_of_terse_id.push(MessageId::from_usize(message_id));
                next_terse_id
            });

            eprintln!(
                "TODOX TERSE OF {:?} MSG {} DATA {:?} IS {:?}",
                message_id,
                self.display_message(&message),
                todox,
                terse_id
            );
            terse_of_message_id.push(MessageId::from_usize(terse_id));
        }

        message_of_terse_id.shrink_to_fit();
    }

    pub fn print_agent_states_diagram(
        &self,
        condense: &Condense,
        agent_index: usize,
        stdout: &mut dyn Write,
    ) {
        let mut emitted_states = vec![false; self.agent_types[agent_index].states_count()];

        writeln!(stdout, "digraph {{").unwrap();
        writeln!(stdout, "color=white;").unwrap();
        writeln!(stdout, "graph [ fontname=\"sans-serif\" ];").unwrap();
        writeln!(stdout, "node [ fontname=\"sans-serif\" ];").unwrap();
        writeln!(stdout, "edge [ fontname=\"sans-serif\" ];").unwrap();

        let mut state_transition_index: usize = 0;

        if condense.is_active() {
            self.compute_terse(condense);
        }

        let state_transitions = self.collect_agent_state_transitions(condense, agent_index);
        let mut contexts: Vec<&<Self as MetaModel>::AgentStateTransitionContext> =
            state_transitions.keys().collect();
        contexts.sort();

        for context in contexts.iter() {
            let related_state_transitions = &state_transitions[context];
            let mut sent_keys: Vec<&Vec<MessageId>> = related_state_transitions.keys().collect();
            sent_keys.sort();

            let mut sent_by_delivered = <Self as MetaModel>::AgentStateSentByDelivered::new();

            for sent_message_ids_key in sent_keys.iter() {
                let sent_message_ids: &Vec<MessageId> = sent_message_ids_key;

                let delivered_message_ids: &Vec<MessageId> =
                    &related_state_transitions.get(sent_message_ids).unwrap();

                if !sent_by_delivered.contains_key(delivered_message_ids) {
                    sent_by_delivered.insert(delivered_message_ids.to_vec(), vec![]);
                }

                sent_by_delivered
                    .get_mut(delivered_message_ids)
                    .unwrap()
                    .push(sent_message_ids.to_vec());
            }

            let mut delivered_keys: Vec<&Vec<MessageId>> = sent_by_delivered.keys().collect();
            delivered_keys.sort();

            let mut intersecting_delivered_message_ids: Vec<Vec<MessageId>> = vec![];
            let mut distinct_delivered_message_ids: Vec<Vec<MessageId>> = vec![];
            for delivered_message_ids_key in delivered_keys.iter() {
                let delivered_message_ids: &Vec<MessageId> = delivered_message_ids_key;
                let delivered_sent_message_ids =
                    sent_by_delivered.get(delivered_message_ids).unwrap();

                assert!(!delivered_sent_message_ids.is_empty());
                if delivered_sent_message_ids.len() == 1 {
                    intersecting_delivered_message_ids.push(delivered_message_ids.to_vec());
                    continue;
                }

                if intersecting_delivered_message_ids
                    .iter()
                    .any(|message_ids| message_ids == delivered_message_ids)
                {
                    continue;
                }

                let mut is_intersecting: bool = false;
                for other_delivered_message_ids_key in delivered_keys.iter() {
                    let other_delivered_message_ids: &Vec<MessageId> =
                        other_delivered_message_ids_key;
                    if delivered_message_ids == other_delivered_message_ids {
                        continue;
                    }
                    // BEGIN NOT TESTED
                    for delivered_message_id in delivered_message_ids {
                        if other_delivered_message_ids
                            .iter()
                            .any(|message_id| message_id == delivered_message_id)
                        {
                            is_intersecting = true;
                            break;
                        }
                    }
                    if is_intersecting {
                        intersecting_delivered_message_ids
                            .push(other_delivered_message_ids.to_vec());
                        break;
                    }
                    // END NOT TESTED
                }

                if is_intersecting {
                    // BEGIN NOT TESTED
                    intersecting_delivered_message_ids.push(delivered_message_ids.to_vec());
                    // END NOT TESTED
                } else {
                    distinct_delivered_message_ids.push(delivered_message_ids.to_vec());
                }
            }

            for delivered_message_ids_key in intersecting_delivered_message_ids.iter() {
                let delivered_message_ids: &Vec<MessageId> = delivered_message_ids_key;
                let mut delivered_sent_keys = sent_by_delivered[delivered_message_ids].clone();
                delivered_sent_keys.sort();

                for sent_message_ids_key in delivered_sent_keys.iter() {
                    let sent_message_ids: &Vec<MessageId> = sent_message_ids_key;

                    self.print_agent_transition_cluster(
                        condense,
                        &mut emitted_states,
                        agent_index,
                        context,
                        delivered_message_ids,
                        state_transition_index,
                        false,
                        stdout,
                    );

                    self.print_agent_transition_sent_edges(
                        condense,
                        agent_index,
                        &sent_message_ids,
                        state_transition_index,
                        None,
                        stdout,
                    );

                    writeln!(stdout, "}}").unwrap();
                    state_transition_index += 1;
                }
            }

            for delivered_message_ids_key in distinct_delivered_message_ids.iter() {
                let mut delivered_sent_keys: Vec<&Vec<MessageId>> =
                    related_state_transitions.keys().collect();
                let delivered_message_ids: &Vec<MessageId> = delivered_message_ids_key;
                delivered_sent_keys.sort();

                self.print_agent_transition_cluster(
                    condense,
                    &mut emitted_states,
                    agent_index,
                    context,
                    delivered_message_ids,
                    state_transition_index,
                    true,
                    stdout,
                );

                for (alternative_index, sent_message_ids_key) in
                    delivered_sent_keys.iter().enumerate()
                {
                    let sent_message_ids: &Vec<MessageId> = sent_message_ids_key;

                    self.print_agent_transition_sent_edges(
                        condense,
                        agent_index,
                        &sent_message_ids,
                        state_transition_index,
                        Some(alternative_index),
                        stdout,
                    );
                }

                writeln!(stdout, "}}").unwrap();
                state_transition_index += 1;
            }
        }

        writeln!(stdout, "}}").unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    fn print_agent_transition_cluster(
        &self,
        condense: &Condense,
        emitted_states: &mut [bool],
        agent_index: usize,
        context: &<Self as MetaModel>::AgentStateTransitionContext,
        delivered_message_ids: &[MessageId],
        state_transition_index: usize,
        has_alternatives: bool,
        stdout: &mut dyn Write,
    ) {
        if !emitted_states[context.from_state_id.to_usize()] {
            self.print_agent_state_node(
                condense,
                agent_index,
                context.from_state_id,
                context.from_is_deferring,
                stdout,
            );
            emitted_states[context.from_state_id.to_usize()] = true;
        }

        if !emitted_states[context.to_state_id.to_usize()] {
            self.print_agent_state_node(
                condense,
                agent_index,
                context.to_state_id,
                context.to_is_deferring,
                stdout,
            );
            emitted_states[context.to_state_id.to_usize()] = true;
        }

        writeln!(stdout, "subgraph cluster_{} {{", state_transition_index).unwrap();
        Self::print_state_transition_node(state_transition_index, has_alternatives, stdout);

        Self::print_state_transition_edge(
            context.from_state_id,
            context.from_is_deferring,
            state_transition_index,
            stdout,
        );

        Self::print_transition_state_edge(
            state_transition_index,
            context.to_state_id,
            context.to_is_deferring,
            stdout,
        );

        for delivered_message_id in delivered_message_ids.iter() {
            self.print_message_node(
                condense,
                state_transition_index,
                None,
                agent_index,
                Some(*delivered_message_id),
                "D",
                stdout,
            );
            self.print_message_transition_edge(
                *delivered_message_id,
                state_transition_index,
                stdout,
            );
        }
    }

    fn print_agent_transition_sent_edges(
        &self,
        condense: &Condense,
        agent_index: usize,
        sent_message_ids: &[MessageId],
        state_transition_index: usize,
        mut alternative_index: Option<usize>,
        stdout: &mut dyn Write,
    ) {
        if let Some(alternative_index) = alternative_index {
            if sent_message_ids.len() > 1 {
                Self::print_state_alternative_node_and_edge(
                    state_transition_index,
                    alternative_index,
                    stdout,
                );
            }
        }

        if alternative_index.is_some() && sent_message_ids.is_empty() {
            self.print_message_node(
                condense,
                state_transition_index,
                None,
                agent_index,
                None,
                "S",
                stdout,
            );
            self.print_transition_message_edge(state_transition_index, None, None, stdout);
        }

        if sent_message_ids.len() < 2 {
            alternative_index = None;
        }

        for sent_message_id in sent_message_ids.iter() {
            self.print_message_node(
                condense,
                state_transition_index,
                alternative_index,
                agent_index,
                Some(*sent_message_id),
                "S",
                stdout,
            );
            self.print_transition_message_edge(
                state_transition_index,
                alternative_index,
                Some(*sent_message_id),
                stdout,
            );
        }
    }

    fn print_agent_state_node(
        &self,
        condense: &Condense,
        agent_index: usize,
        state_id: StateId,
        is_deferring: bool,
        stdout: &mut dyn Write,
    ) {
        let shape = if is_deferring { "octagon" } else { "ellipse" };
        let state = if condense.names_only {
            self.agent_types[agent_index].display_terse(state_id)
        } else {
            self.agent_types[agent_index].display_state(state_id)
        };

        writeln!(
            stdout,
            "A_{}_{} [ label=\"{}\", shape={} ];",
            state_id.to_usize(),
            is_deferring,
            state,
            shape
        )
        .unwrap();
    }

    fn print_state_transition_node(
        state_transition_index: usize,
        has_alternatives: bool,
        stdout: &mut dyn Write,
    ) {
        if has_alternatives {
            writeln!(
                stdout,
                "T_{}_{} [ shape=diamond, label=\"\", fontsize=0, \
                 width=0.15, height=0.15, style=filled, color=black ];",
                state_transition_index,
                usize::max_value()
            )
            .unwrap();
        } else {
            writeln!(
                stdout,
                "T_{}_{} [ shape=point, height=0.015, width=0.015 ];",
                state_transition_index,
                usize::max_value()
            )
            .unwrap();
        }
    }

    fn print_message_node(
        &self,
        condense: &Condense,
        state_transition_index: usize,
        alternative_index: Option<usize>,
        agent_index: usize,
        message_id: Option<MessageId>,
        prefix: &str,
        stdout: &mut dyn Write,
    ) {
        if message_id.is_none() {
            // BEGIN NOT TESTED
            writeln!(
                stdout,
                "{}_{}_{} [ label=\" \", shape=plain ];",
                prefix,
                state_transition_index,
                alternative_index.unwrap_or(usize::max_value()),
            )
            .unwrap();
            return;
            // END NOT TESTED
        }

        let mut message_id = message_id.unwrap();

        if !message_id.is_valid() {
            writeln!(
                stdout,
                "{}_{}_{}_{} [ label=\"Time\", shape=plain ];",
                prefix,
                state_transition_index,
                alternative_index.unwrap_or(usize::max_value()),
                message_id.to_usize()
            )
            .unwrap();
            return;
        }

        write!(
            stdout,
            "{}_{}_{}_{} [ label=\"",
            prefix,
            state_transition_index,
            alternative_index.unwrap_or(usize::max_value()),
            message_id.to_usize()
        )
        .unwrap();

        if condense.names_only {
            message_id = self.message_of_terse_id.read()[message_id.to_usize()];
        }

        let message = self.messages.get(message_id);
        if message.source_index != agent_index {
            let source = if condense.merge_instances {
                self.agent_types[message.source_index].name()
            } else {
                self.agent_labels[message.source_index].to_string()
            };
            write!(stdout, "{} {}\\n", source, RIGHT_ARROW).unwrap();
        }

        if !condense.final_replaced {
            if let Some(replaced) = message.replaced {
                if condense.names_only {
                    write!(stdout, "{} {}\\n", replaced.name(), RIGHT_DOUBLE_ARROW).unwrap();
                } else {
                    write!(stdout, "{} {}\\n", replaced, RIGHT_DOUBLE_ARROW).unwrap();
                }
            }
        }

        if condense.names_only {
            write!(stdout, "{}", message.payload.name()).unwrap();
        } else {
            write!(stdout, "{}", message.payload).unwrap();
        }

        if message.target_index != agent_index {
            let target = if condense.merge_instances {
                self.agent_types[message.target_index].name()
            } else {
                self.agent_labels[message.target_index].to_string()
            };
            write!(stdout, "\\n{} {}", RIGHT_ARROW, target).unwrap();
        }

        writeln!(stdout, "\", shape=plain ];").unwrap();
    }

    fn print_state_transition_edge(
        from_state_id: StateId,
        from_is_deferring: bool,
        to_state_transition_index: usize,
        stdout: &mut dyn Write,
    ) {
        writeln!(
            stdout,
            "A_{}_{} -> T_{}_{} [ arrowhead=none, direction=forward ];",
            from_state_id.to_usize(),
            from_is_deferring,
            to_state_transition_index,
            usize::max_value()
        )
        .unwrap();
    }

    fn print_transition_state_edge(
        from_state_transition_index: usize,
        to_state_id: StateId,
        to_is_deferring: bool,
        stdout: &mut dyn Write,
    ) {
        writeln!(
            stdout,
            "T_{}_{} -> A_{}_{};",
            from_state_transition_index,
            usize::max_value(),
            to_state_id.to_usize(),
            to_is_deferring,
        )
        .unwrap();
    }

    fn print_message_transition_edge(
        &self,
        from_message_id: MessageId,
        to_state_transition_index: usize,
        stdout: &mut dyn Write,
    ) {
        let is_immediate = from_message_id.is_valid()
            && self.messages.get(from_message_id).order == MessageOrder::Immediate;
        let arrowhead = if is_immediate {
            "normalnormal" // NOT TESTED
        } else {
            "normal"
        };

        writeln!(
            stdout,
            "D_{}_{}_{} -> T_{}_{} [ arrowhead={}, direction=forward, style=dashed ];",
            to_state_transition_index,
            usize::max_value(),
            from_message_id.to_usize(),
            to_state_transition_index,
            usize::max_value(),
            arrowhead
        )
        .unwrap();
    }

    // BEGIN NOT TESTED
    fn print_state_alternative_node_and_edge(
        state_transition_index: usize,
        alternative_index: usize,
        stdout: &mut dyn Write,
    ) {
        writeln!(
            stdout,
            "T_{}_{} [ shape=point, height=0.015, width=0.015, style=filled ];",
            state_transition_index, alternative_index,
        )
        .unwrap();

        writeln!(
            stdout,
            "T_{}_{} -> T_{}_{} [ arrowhead=none, direction=forward, style=dashed ];",
            state_transition_index,
            usize::max_value(),
            state_transition_index,
            alternative_index,
        )
        .unwrap();
    }
    // END NOT TESTED

    fn print_transition_message_edge(
        &self,
        from_state_transition_index: usize,
        from_alternative_index: Option<usize>,
        to_message_id: Option<MessageId>,
        stdout: &mut dyn Write,
    ) {
        if to_message_id.is_none() {
            // BEGIN NOT TESTED
            writeln!(
                stdout,
                "T_{}_{} -> S_{}_{} [ arrowhead=dot, direction=forward, style=dashed ];",
                from_state_transition_index,
                from_alternative_index.unwrap_or(usize::max_value()),
                from_state_transition_index,
                from_alternative_index.unwrap_or(usize::max_value()),
            )
            .unwrap();
            return;
            // END NOT TESTED
        }

        let to_message_id = to_message_id.unwrap();

        let is_immediate = to_message_id.is_valid()
            && self.messages.get(to_message_id).order == MessageOrder::Immediate;
        let arrowhead = if is_immediate {
            "normalnormal" // NOT TESTED
        } else {
            "normal"
        };

        writeln!(
            stdout,
            "T_{}_{} -> S_{}_{}_{} [ arrowhead={}, direction=forward, style=dashed ];",
            from_state_transition_index,
            from_alternative_index.unwrap_or(usize::max_value()),
            from_state_transition_index,
            from_alternative_index.unwrap_or(usize::max_value()),
            to_message_id.to_usize(),
            arrowhead
        )
        .unwrap();
    }

    fn print_sequence_diagram(
        &self,
        path: &[<Self as MetaModel>::PathTransition],
        stdout: &mut dyn Write,
    ) {
        writeln!(stdout, "@startuml").unwrap();
        writeln!(stdout, "autonumber \" <b>#</b> \"").unwrap();
        writeln!(stdout, "skinparam shadowing false").unwrap();
        writeln!(stdout, "skinparam sequence {{").unwrap();
        writeln!(stdout, "ArrowColor Black").unwrap();
        writeln!(stdout, "ActorBorderColor Black").unwrap();
        writeln!(stdout, "LifeLineBorderColor Black").unwrap();
        writeln!(stdout, "LifeLineBackgroundColor Black").unwrap();
        writeln!(stdout, "ParticipantBorderColor Black").unwrap();
        writeln!(stdout, "}}").unwrap();
        writeln!(stdout, "skinparam ControlBorderColor White").unwrap();
        writeln!(stdout, "skinparam ControlBackgroundColor White").unwrap();

        let mut sequence_state = SequenceState {
            timelines: vec![],
            message_timelines: StdHashMap::new(),
            agents_timelines: vec![
                AgentTimelines {
                    left: vec![],
                    right: vec![]
                };
                MAX_AGENTS
            ],
        };

        let first_configuration = self.configurations.get(path[0].from_configuration_id);

        self.print_sequence_participants(&first_configuration, stdout);
        self.print_first_timelines(&mut sequence_state, &first_configuration, stdout);
        self.print_sequence_first_notes(&sequence_state, &first_configuration, stdout);

        path.iter()
            .enumerate()
            .skip(1)
            .map(|(transition_index, transition)| {
                if transition_index + 1 < path.len() {
                    (transition, Some(&path[transition_index + 1]))
                } else {
                    (transition, None)
                }
            })
            .for_each(|(transition, next_transition)| {
                self.print_sequence_transition(
                    &mut sequence_state,
                    transition,
                    next_transition,
                    stdout,
                );
            });

        let last_configuration = self
            .configurations
            .get(path.last().unwrap().to_configuration_id);
        if last_configuration.invalid_id.is_valid() {}

        writeln!(stdout, "@enduml").unwrap();
    }

    fn print_sequence_participants(
        &self,
        first_configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) {
        self.agent_labels
            .iter()
            .enumerate()
            .for_each(|(agent_index, agent_label)| {
                let agent_type = &self.agent_types[agent_index];
                let agent_instance = self.agent_instance(agent_index);
                writeln!(
                    stdout,
                    "participant \"{}\" as A{} order {}",
                    agent_label,
                    agent_index,
                    self.agent_scaled_order(agent_index),
                )
                .unwrap();
                if agent_type.state_is_deferring(agent_instance, &first_configuration.state_ids) {
                    // BEGIN NOT TESTED
                    writeln!(stdout, "activate A{} #MediumPurple", agent_index).unwrap();
                    // END NOT TESTED
                } else {
                    writeln!(stdout, "activate A{} #CadetBlue", agent_index).unwrap();
                }
            });
    }

    fn print_first_timelines(
        &self,
        mut sequence_state: &mut <Self as MetaModel>::SequenceState,
        first_configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) {
        first_configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .enumerate()
            .for_each(|(message_index, message_id)| {
                let message = self.messages.get(*message_id);
                let timeline_index = self.add_sequence_timeline(
                    &mut sequence_state,
                    message_index,
                    *message_id,
                    &message,
                    stdout,
                );
                writeln!(stdout, "activate T{} #Silver", timeline_index).unwrap();
            });
    }

    fn agent_scaled_order(&self, agent_index: usize) -> usize {
        let agent_type = &self.agent_types[agent_index];
        let agent_instance = self.agent_instance(agent_index);
        let agent_order = agent_type.instance_order(agent_instance);
        ((agent_order + 1) * 100 + agent_index + 1) * 100
    }

    fn is_rightwards_message(&self, message: &<Self as MetaModel>::Message) -> bool {
        let source_scaled_order = self.agent_scaled_order(message.source_index);
        let target_scaled_order = self.agent_scaled_order(message.target_index);
        source_scaled_order < target_scaled_order
    }

    fn add_sequence_timeline(
        &self,
        sequence_state: &mut <Self as MetaModel>::SequenceState,
        message_index: usize,
        message_id: MessageId,
        message: &<Self as MetaModel>::Message,
        stdout: &mut dyn Write,
    ) -> usize {
        let timeline_index = sequence_state.timelines.len();
        sequence_state.timelines.push(Some(message_index));
        sequence_state
            .message_timelines
            .insert(message_id, timeline_index);

        let timeline_order = if self.is_rightwards_message(message) {
            sequence_state.agents_timelines[message.source_index]
                .right
                .push(timeline_index);
            self.agent_scaled_order(message.source_index)
                + sequence_state.agents_timelines[message.source_index]
                    .right
                    .len()
        } else {
            sequence_state.agents_timelines[message.source_index]
                .left
                .push(timeline_index);
            self.agent_scaled_order(message.source_index)
                - sequence_state.agents_timelines[message.source_index]
                    .left
                    .len()
        };

        writeln!(
            stdout,
            "control \" \" as T{} order {}",
            timeline_index, timeline_order
        )
        .unwrap();

        timeline_index
    }

    fn print_sequence_first_notes(
        &self,
        sequence_state: &<Self as MetaModel>::SequenceState,
        first_configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) {
        self.agent_types
            .iter()
            .enumerate()
            .map(|(agent_index, agent_type)| {
                (
                    agent_index,
                    agent_type.display_state(first_configuration.state_ids[agent_index]),
                )
            })
            .filter(|(_agent_index, agent_state)| !agent_state.is_empty())
            .enumerate()
            .for_each(|(note_index, (agent_index, agent_state))| {
                if note_index > 0 {
                    write!(stdout, "/ ").unwrap();
                }
                writeln!(stdout, "rnote over A{} : {}", agent_index, agent_state,).unwrap();
            });

        sequence_state
            .timelines
            .iter()
            .enumerate()
            .for_each(|(timeline_index, message_index)| {
                let message_id = first_configuration.message_ids[message_index.unwrap()];
                let message = self.messages.get(message_id);
                writeln!(
                    stdout,
                    "/ rnote over T{} : {}",
                    timeline_index,
                    self.display_sequence_message(&message, false)
                )
                .unwrap();
            });
    }

    fn print_sequence_transition(
        &self,
        mut sequence_state: &mut <Self as MetaModel>::SequenceState,
        transition: &<Self as MetaModel>::PathTransition,
        next_transition: Option<&<Self as MetaModel>::PathTransition>,
        stdout: &mut dyn Write,
    ) {
        let from_configuration = self.configurations.get(transition.from_configuration_id);
        let to_configuration = self.configurations.get(transition.to_configuration_id);

        let did_send_messages = self.print_sequence_send_messages(
            &mut sequence_state,
            transition,
            next_transition,
            &from_configuration,
            &to_configuration,
            stdout,
        );
        self.print_sequence_changed_notes(
            transition,
            did_send_messages,
            &from_configuration,
            &to_configuration,
            stdout,
        );
        self.print_sequence_activate_changed(&from_configuration, &to_configuration, stdout);
    }

    fn print_sequence_send_messages(
        &self,
        mut sequence_state: &mut <Self as MetaModel>::SequenceState,
        transition: &<Self as MetaModel>::PathTransition,
        next_transition: Option<&<Self as MetaModel>::PathTransition>,
        from_configuration: &<Self as MetaModel>::Configuration,
        to_configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) -> bool {
        if transition.delivered_message_index.is_some() {
            let delivered_message_id = transition.delivered_message_id.unwrap();
            let delivered_message = self.messages.get(delivered_message_id);
            if let Some(timeline_index) =
                sequence_state.message_timelines.get(&delivered_message_id)
            {
                writeln!(
                    stdout,
                    "T{} -> A{} : {}",
                    timeline_index,
                    delivered_message.target_index,
                    self.display_sequence_message(&delivered_message, true)
                )
                .unwrap();
                writeln!(stdout, "deactivate T{}", timeline_index).unwrap();
                sequence_state.timelines[*timeline_index] = None;
                if to_configuration.state_ids[transition.agent_index]
                    != from_configuration.state_ids[transition.agent_index]
                {
                    writeln!(stdout, "deactivate A{}", transition.agent_index).unwrap();
                }
            }
        } else {
            writeln!(stdout, "?o-> A{}", transition.agent_index).unwrap();
            if to_configuration.state_ids[transition.agent_index]
                != from_configuration.state_ids[transition.agent_index]
            {
                writeln!(stdout, "deactivate A{}", transition.agent_index).unwrap();
            }
        }

        let from_message_timelines = sequence_state.message_timelines.clone();
        sequence_state.message_timelines.clear();

        let mut did_send_messages: bool = false;

        transition
            .to_prev_messages
            .iter()
            .take_while(|prev_message| **prev_message != PrevMessage::NotApplicable)
            .enumerate()
            .for_each(|(to_message_index, prev_message)| match prev_message {
                PrevMessage::Kept(_) => {
                    let to_message_id = to_configuration.message_ids[to_message_index];
                    let timeline_index = from_message_timelines[&to_message_id];
                    sequence_state
                        .message_timelines
                        .insert(to_message_id, timeline_index);
                }
                PrevMessage::Replaced(from_message_index) => {
                    let from_message_id = from_configuration.message_ids[*from_message_index];
                    let timeline_index = from_message_timelines[&from_message_id];
                    let to_message_id = to_configuration.message_ids[to_message_index];
                    let to_message = self.messages.get(to_message_id);
                    sequence_state
                        .message_timelines
                        .insert(to_message_id, timeline_index);
                    writeln!(
                        stdout,
                        "A{} -> T{} : {}",
                        to_message.source_index,
                        timeline_index,
                        self.display_sequence_message(&to_message, false)
                    )
                    .unwrap();
                    did_send_messages = true;
                }
                _ => {}
            });

        transition
            .to_prev_messages
            .iter()
            .take_while(|prev_message| **prev_message != PrevMessage::NotApplicable)
            .enumerate()
            .filter(|(_, prev_message)| **prev_message == PrevMessage::NotThere)
            .for_each(|(to_message_index, _prev_message)| {
                let to_message_id = to_configuration.message_ids[to_message_index];
                let to_message = self.messages.get(to_message_id);
                debug_assert!(to_message.source_index == transition.agent_index);
                match next_transition {
                    Some(next_transition)
                        if next_transition.from_next_messages[to_message_index]
                            == NextMessage::Delivered =>
                    {
                        let color = if to_message.order == MessageOrder::Immediate {
                            "[#Crimson]" // NOT TESTED
                        } else {
                            ""
                        };
                        writeln!(
                            stdout,
                            "A{} -{}> A{} : {}",
                            to_message.source_index,
                            color,
                            to_message.target_index,
                            self.display_sequence_message(&to_message, false),
                        )
                        .unwrap();
                        let next_to_configuration =
                            self.configurations.get(next_transition.to_configuration_id);
                        if next_to_configuration.state_ids[to_message.target_index]
                            != from_configuration.state_ids[to_message.target_index]
                        {
                            writeln!(stdout, "deactivate A{}", to_message.target_index).unwrap();
                        }
                        did_send_messages = true;
                    }
                    _ => {
                        let agent_timelines =
                            &sequence_state.agents_timelines[to_message.source_index];
                        let side_timelines = if self.is_rightwards_message(&to_message) {
                            &agent_timelines.right
                        } else {
                            &agent_timelines.left
                        };
                        let empty_timeline_index = side_timelines
                            .iter()
                            .rev()
                            .position(|timeline_index| {
                                sequence_state.timelines[*timeline_index].is_none()
                            })
                            .map(|rev_index| side_timelines.len() - 1 - rev_index);

                        let timeline_index =
                            if let Some(existing_timeline_index) = empty_timeline_index {
                                // BEGIN NOT TESTED
                                debug_assert!(
                                    sequence_state.timelines[existing_timeline_index].is_none()
                                );
                                sequence_state
                                    .message_timelines
                                    .insert(to_message_id, existing_timeline_index);
                                existing_timeline_index
                                // END NOT TESTED
                            } else {
                                self.add_sequence_timeline(
                                    &mut sequence_state,
                                    to_message_index,
                                    to_message_id,
                                    &to_message,
                                    stdout,
                                )
                            };
                        writeln!(
                            stdout,
                            "A{} -> T{} : {}",
                            to_message.source_index,
                            timeline_index,
                            self.display_sequence_message(&to_message, false)
                        )
                        .unwrap();
                        writeln!(stdout, "activate T{} #Silver", timeline_index).unwrap();
                        did_send_messages = true;
                    }
                }
            });

        did_send_messages
    }

    fn print_sequence_changed_notes(
        &self,
        transition: &<Self as MetaModel>::PathTransition,
        did_send_messages: bool,
        from_configuration: &<Self as MetaModel>::Configuration,
        to_configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) {
        from_configuration
            .state_ids
            .iter()
            .take_while(|state_id| state_id.is_valid())
            .zip(to_configuration.state_ids.iter())
            .enumerate()
            .filter(|(_, (from_state_id, to_state_id))| from_state_id != to_state_id)
            .enumerate()
            .for_each(|(note_index, (agent_index, _state_ids))| {
                let agent_type = &self.agent_types[agent_index];
                if note_index > 0 {
                    write!(stdout, "/ ").unwrap(); // NOT TESTED
                }
                if agent_index != transition.agent_index || !did_send_messages {
                    writeln!(stdout, "autonumber stop").unwrap();
                    writeln!(stdout, "?-[#White]\\ A{}", agent_index).unwrap();
                    writeln!(stdout, "autonumber resume").unwrap();
                }
                let agent_state = agent_type.display_state(to_configuration.state_ids[agent_index]);
                if !agent_state.is_empty() {
                    writeln!(stdout, "rnote over A{} : {}", agent_index, agent_state).unwrap();
                }
            });
    }

    fn print_sequence_activate_changed(
        &self,
        from_configuration: &<Self as MetaModel>::Configuration,
        to_configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) {
        from_configuration
            .state_ids
            .iter()
            .take_while(|state_id| state_id.is_valid())
            .zip(to_configuration.state_ids.iter())
            .enumerate()
            .filter(|(_, (from_state_id, to_state_id))| from_state_id != to_state_id)
            .for_each(|(agent_index, _state_ids)| {
                let agent_type = &self.agent_types[agent_index];
                let agent_instance = self.agent_instance(agent_index);
                if agent_type.state_is_deferring(agent_instance, &to_configuration.state_ids) {
                    writeln!(stdout, "activate A{} #CadetBlue", agent_index).unwrap();
                } else {
                    writeln!(stdout, "activate A{} #MediumPurple", agent_index).unwrap();
                }
            });
    }

    fn collect_agent_state_transitions(
        &self,
        condense: &Condense,
        agent_index: usize,
    ) -> <Self as MetaModel>::AgentStateTransitions {
        let mut state_transitions = <Self as MetaModel>::AgentStateTransitions::default();
        self.outgoings
            .read()
            .iter()
            .take(self.configurations.len())
            .enumerate()
            .for_each(|(from_configuration_id, outgoings)| {
                let from_configuration = self
                    .configurations
                    .get(ConfigurationId::from_usize(from_configuration_id));
                outgoings.read().iter().for_each(|outgoing| {
                    let to_configuration = self.configurations.get(outgoing.to_configuration_id);
                    self.collect_agent_state_transition(
                        condense,
                        agent_index,
                        &from_configuration,
                        &to_configuration,
                        outgoing.delivered_message_index,
                        &mut state_transitions,
                    );
                });
            });
        state_transitions
    }

    fn collect_agent_state_transition(
        &self,
        condense: &Condense,
        agent_index: usize,
        from_configuration: &<Self as MetaModel>::Configuration,
        to_configuration: &<Self as MetaModel>::Configuration,
        delivered_message_index: MessageIndex,
        state_transitions: &mut <Self as MetaModel>::AgentStateTransitions,
    ) {
        let agent_type = &self.agent_types[agent_index];
        let agent_instance = self.agent_instance(agent_index);

        let mut context = AgentStateTransitionContext {
            from_state_id: from_configuration.state_ids[agent_index],
            from_is_deferring: agent_type
                .state_is_deferring(agent_instance, &from_configuration.state_ids),
            to_state_id: to_configuration.state_ids[agent_index],
            to_is_deferring: agent_type
                .state_is_deferring(agent_instance, &to_configuration.state_ids),
        };

        if condense.names_only {
            context.from_state_id = agent_type.terse_id(context.from_state_id);
            context.to_state_id = agent_type.terse_id(context.to_state_id);
        }

        let mut sent_message_ids: Vec<MessageId> = vec![];

        to_configuration
            .message_ids
            .iter()
            .take_while(|to_message_id| to_message_id.is_valid())
            .map(|to_message_id| (to_message_id, self.messages.get(*to_message_id)))
            .filter(|(_, to_message)| to_message.source_index == agent_index)
            .for_each(|(to_message_id, _)| {
                if self
                    .to_message_kept_in_transition(
                        *to_message_id,
                        &from_configuration,
                        delivered_message_index,
                    )
                    .is_none()
                {
                    let message_id = if condense.is_active() {
                        self.terse_of_message_id.read()[to_message_id.to_usize()]
                    } else {
                        *to_message_id
                    };
                    sent_message_ids.push(message_id);
                    sent_message_ids.sort();
                }
            });

        let mut delivered_message_id = MessageId::invalid();
        let mut is_delivered_to_us = false;
        if delivered_message_index.is_valid() {
            delivered_message_id =
                from_configuration.message_ids[delivered_message_index.to_usize()];
            let delivered_message = self.messages.get(delivered_message_id);
            if delivered_message.target_index == agent_index {
                is_delivered_to_us = true;
                if condense.is_active() {
                    delivered_message_id =
                        self.terse_of_message_id.read()[delivered_message_id.to_usize()];
                }
            }
        }

        if !is_delivered_to_us
            && context.from_state_id == context.to_state_id
            && sent_message_ids.is_empty()
        {
            return;
        }

        state_transitions
            .entry(context)
            .or_insert_with(StdHashMap::new);

        let state_delivered_message_ids: &mut StdHashMap<Vec<MessageId>, Vec<MessageId>> =
            state_transitions.get_mut(&context).unwrap();

        state_delivered_message_ids
            .entry(sent_message_ids.to_vec())
            .or_insert_with(Vec::new);

        let delivered_message_ids: &mut Vec<MessageId> = state_delivered_message_ids
            .get_mut(&sent_message_ids.to_vec())
            .unwrap();

        if !delivered_message_ids
            .iter()
            .any(|message_id| *message_id == delivered_message_id)
        {
            delivered_message_ids.push(delivered_message_id);
            delivered_message_ids.sort();
        }
    }

    fn from_message_kept_in_transition(
        &self,
        from_message_id: MessageId,
        to_configuration: &<Self as MetaModel>::Configuration,
    ) -> Option<usize> {
        let to_index =
            self.message_exists_in_configuration(from_message_id, to_configuration, None);
        if to_index.is_some() {
            return to_index;
        }

        match self.decr_message_id(from_message_id) {
            None => None,
            Some(decr_message_id) => {
                self.message_exists_in_configuration(decr_message_id, to_configuration, None)
            }
        }
    }

    fn to_message_kept_in_transition(
        &self,
        to_message_id: MessageId,
        from_configuration: &<Self as MetaModel>::Configuration,
        delivered_message_index: MessageIndex,
    ) -> Option<usize> {
        let from_index = self.message_exists_in_configuration(
            to_message_id,
            from_configuration,
            Some(delivered_message_index),
        );
        if from_index.is_some() {
            return from_index;
        }

        match self.incr_message_id(to_message_id) {
            None => None,
            Some(incr_message_id) => self.message_exists_in_configuration(
                incr_message_id,
                from_configuration,
                Some(delivered_message_index),
            ),
        }
    }

    fn message_exists_in_configuration(
        &self,
        message_id: MessageId,
        configuration: &<Self as MetaModel>::Configuration,
        exclude_message_index: Option<MessageIndex>,
    ) -> Option<usize> {
        configuration
            .message_ids
            .iter()
            .take_while(|configuration_message_id| configuration_message_id.is_valid())
            .enumerate()
            .filter(|(configuration_message_index, _)| {
                Some(MessageIndex::from_usize(*configuration_message_index))
                    != exclude_message_index
            })
            .position(|(_, configuration_message_id)| *configuration_message_id == message_id)
    }
}

/// Add clap commands and flags to a clap application.
pub fn add_clap<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("progress")
            .short("p")
            .long("progress-every")
            .default_value("1000")
            .help("print configurations as they are reached"),
    )
    .arg(
        Arg::with_name("threads")
            .short("t")
            .long("threads")
            .value_name("COUNT")
            .help("set the number of threads to use (may also specify PHYSICAL or LOGICAL)")
            .default_value("PHYSICAL"),
    )
    .arg(
        Arg::with_name("size")
            .short("s")
            .long("size")
            .value_name("COUNT")
            .help(
                "pre-allocate arrays to cover this number of configurations, for faster operation",
            )
            .default_value("AUTO"),
    )
    .arg(
        Arg::with_name("reachable")
            .short("r")
            .long("reachable")
            .help(
                "ensure that the initial configuration is reachable from all other configurations",
            ),
    )
    .arg(
        Arg::with_name("invalid")
            .short("i")
            .long("invalid")
            .help("allow for invalid configurations (but do not explore beyond them)"),
    )
    .subcommand(
        SubCommand::with_name("agents")
            .about("list the agents of the model (does not compute the model)"),
    )
    .subcommand(SubCommand::with_name("conditions").about(
        "list the conditions which can be used to identify configurations \
                   (does not compute the model)",
    ))
    .subcommand(
        SubCommand::with_name("configurations").about("list the configurations of the model"),
    )
    .subcommand(SubCommand::with_name("transitions").about("list the transitions of the model"))
    .subcommand(
        SubCommand::with_name("path")
            .about("list transitions for a path between configurations")
            .arg(Arg::with_name("CONDITION").multiple(true).help(
                "the name of at least two conditions identifying configurations along the path, \
                          which may be prefixed with ! to negate the condition",
            )),
    )
    .subcommand(
        SubCommand::with_name("sequence")
            .about("generate a PlantUML sequence diagram for a path between configurations")
            .arg(Arg::with_name("CONDITION").multiple(true).help(
                "the name of at least two conditions identifying configurations along the path, \
                          which may be prefixed with ! to negate the condition",
            )),
    )
    .subcommand(
        SubCommand::with_name("states")
            .about("generate a GraphViz dot diagrams for the states of a specific agent")
            .arg(
                Arg::with_name("AGENT")
                    .help("the name of the agent to generate a diagrams for the states of"),
            )
            .arg(
                Arg::with_name("names-only")
                    .short("n")
                    .long("names-only")
                    .help("condense graph nodes considering only the state & payload names"),
            )
            .arg(
                Arg::with_name("merge-instances")
                    .short("m")
                    .long("merge-instances")
                    .help("condense graph nodes considering only the agent type"),
            )
            .arg(
                Arg::with_name("final-replaced")
                    .short("f")
                    .long("final-replaced")
                    .help("condense graph nodes considering only the final (replaced) payload"),
            ),
    )
}

/// Execute operations on a model using clap commands.
pub trait ClapModel {
    /// Execute the chosen clap subcommand.
    ///
    /// Return whether a command was executed.
    fn do_clap(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) {
        let did_clap = self.do_clap_agents(arg_matches, stdout)
            || self.do_clap_conditions(arg_matches, stdout)
            || self.do_clap_configurations(arg_matches, stdout)
            || self.do_clap_transitions(arg_matches, stdout)
            || self.do_clap_path(arg_matches, stdout)
            || self.do_clap_sequence(arg_matches, stdout)
            || self.do_clap_states(arg_matches, stdout);
        assert!(did_clap);
    }

    /// Execute the `agents` clap subcommand, if requested to.
    ///
    /// This doesn't compute the model.
    fn do_clap_agents(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `conditions` clap subcommand, if requested to.
    ///
    /// This doesn't compute the model.
    fn do_clap_conditions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `configurations` clap subcommand, if requested to.
    ///
    /// This computes the model.
    fn do_clap_configurations(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `transitions` clap subcommand, if requested to.
    ///
    /// This computes the model.
    fn do_clap_transitions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `path` clap subcommand, if requested to.
    fn do_clap_path(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `sequence` clap subcommand, if requested to.
    fn do_clap_sequence(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `states` clap subcommand, if requested to.
    fn do_clap_states(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;
}

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: DataLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > ClapModel
    for Model<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>
{
    fn do_clap_agents(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("agents") {
            Some(_) => {
                self.agent_labels.iter().for_each(|agent_label| {
                    writeln!(stdout, "{}", agent_label).unwrap();
                });
                true
            }
            None => false,
        }
    }

    fn do_clap_conditions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("conditions") {
            Some(_) => {
                let mut names: Vec<&String> =
                    self.conditions.iter().map(|(key, _value)| key).collect();
                names.sort();
                names
                    .iter()
                    .map(|name| (*name, self.conditions.get(*name).unwrap().get().1 .1))
                    .for_each(|(name, about)| {
                        writeln!(stdout, "{}: {}", name, about).unwrap();
                    });
                true
            }
            None => false,
        }
    }

    fn do_clap_configurations(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("configurations") {
            Some(_) => {
                self.do_compute(arg_matches);

                (0..self.configurations.len())
                    .map(ConfigurationId::from_usize)
                    .for_each(|configuration_id| {
                        writeln!(
                            stdout,
                            "{}",
                            self.display_configuration_id(configuration_id)
                        )
                        .unwrap();
                    });
                true
            }
            None => false,
        }
    }

    fn do_clap_transitions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("transitions") {
            Some(_) => {
                self.do_compute(arg_matches);

                self.outgoings
                    .read()
                    .iter()
                    .enumerate()
                    .take(self.configurations.len())
                    .for_each(|(from_configuration_id, outgoings)| {
                        let from_configuration_id =
                            ConfigurationId::from_usize(from_configuration_id);
                        let from_configuration = self.configurations.get(from_configuration_id);

                        outgoings.read().iter().for_each(|outgoing| {
                            let event_label = if outgoing.delivered_message_index.is_valid() {
                                self.event_label(Some(
                                    from_configuration.message_ids
                                        [outgoing.delivered_message_index.to_usize()],
                                ))
                            } else {
                                self.event_label(None)
                            };

                            writeln!(
                                stdout,
                                "- BY {}\n  TO {}",
                                event_label,
                                self.display_configuration_id(outgoing.to_configuration_id)
                            )
                            .unwrap();
                        });
                    });
                true
            }
            None => false,
        }
    }

    fn do_clap_path(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("path") {
            Some(matches) => {
                self.do_compute(arg_matches);
                let path = self.collect_path("path", matches);
                self.print_path(&path, stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_sequence(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("sequence") {
            Some(matches) => {
                self.do_compute(arg_matches);
                let path = self.collect_path("path", matches);
                self.print_sequence_diagram(&path, stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_states(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("states") {
            Some(matches) => {
                let agent_label = matches
                    .value_of("AGENT")
                    .expect("the states command requires a single agent name, none were given");
                let condense = Condense {
                    names_only: matches.is_present("names-only"),
                    merge_instances: matches.is_present("merge-instances"),
                    final_replaced: matches.is_present("final-replaced"),
                };
                let agent_index = self
                    .agent_labels
                    .iter()
                    .position(|label| **label == agent_label)
                    .unwrap_or_else(|| panic!("unknown agent {}", agent_label));

                self.do_compute(arg_matches);
                self.print_agent_states_diagram(&condense, agent_index, stdout);

                true
            }
            None => false, // NOT TESTED
        }
    }
}

/// Parse the model size parameter.
pub fn model_size(arg_matches: &ArgMatches, auto: usize) -> usize {
    let size = arg_matches.value_of("size").unwrap();
    if size == "AUTO" {
        auto
    } else {
        usize::from_str(size).expect("invalid model size")
    }
}
