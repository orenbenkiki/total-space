// Copyright (C) 2017-2019 Oren Ben-Kiki. See the LICENSE.txt
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Explore the total space of states of communicating finite state machines.

#![feature(trait_alias)]

use hashbrown::HashMap;
use num_traits::Bounded;
use num_traits::FromPrimitive;
use num_traits::ToPrimitive;
use num_traits::Unsigned;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::RwLock;

/// A trait for anything we use as a key in a HashMap.
pub trait KeyLike = Eq + Hash + Copy + Sized + Send + Sync + Display;

/// A trait for anything we use as a zero-based index.
pub trait IndexLike = KeyLike + Bounded + FromPrimitive + ToPrimitive + Unsigned;

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

impl<T: KeyLike, I: IndexLike> Memoize<T, I> {
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

    /// The number of allocated identifiers.
    pub fn usize(&self) -> usize {
        self.id_by_value.len()
    }

    /// The number of allocated identifiers.
    pub fn size(&self) -> I {
        I::from_usize(self.usize()).unwrap()
    }

    /// Given a value, look it up in the memory.
    pub fn lookup(&self, value: &T) -> Option<&I> {
        self.id_by_value.get(value)
    }

    /// Given a value that may or may not exist in the memory, ensure it exists it and return its
    /// short identifier.
    pub fn store(&mut self, value: T) -> Stored<I> {
        match self.lookup(&value) {
            Some(id) => Stored {
                id: *id,
                is_new: false,
            },
            None => {
                if self.usize() > self.max_id {
                    panic!("too many memoized objects");
                }

                let id = self.size();
                self.id_by_value.insert(value, id);
                self.value_by_id.push(value);
                if let Some(display_by_id) = &mut self.display_by_id {
                    display_by_id.push(format!("{}", value));
                }

                Stored { id, is_new: true }
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

    /// Send an unordered message that will replace the single in-flight message accepted by the
    /// callback, or be created as a new message is the callback accepts `None`.
    UnorderedReplacement(fn(Option<Payload>) -> bool, Payload, usize),

    /// Send an ordered message that will replace the single in-flight message accepted by the
    /// callback, or be created as a new message is the callback accepts `None`.
    OrderedReplacement(fn(Option<Payload>) -> bool, Payload, usize),

    /// Send an immediate message that will replace the single in-flight message accepted by the
    /// callback, or be created as a new message is the callback accepts `None`.
    ImmediateReplacement(fn(Option<Payload>) -> bool, Payload, usize),
}

/// Specify an action the agent may take as a response to an event.
#[derive(PartialEq, Eq)]
pub enum Action<State, Payload> {
    /// Defer the event, keep the state the same, do not send any messages.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Response.Defer` value.
    ///
    /// This is only allowed if the agent's `state_is_deferring`, waiting for
    /// specific message(s) to resume normal operations.
    Defer,

    /// Consume (ignore) the event, keep the state the same, do not send any messages.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Response.Remain` value.
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

/// The response from an agent on some event.
#[derive(PartialEq, Eq)]
pub enum Response<State, Payload> {
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

/// A trait describing a set of agents of some type.
pub trait AgentType<StateId, MessageId, Payload> {
    /// The name of the type of the agents.
    fn name(&self) -> &'static str;

    /// Whether this type supports multiple instances.
    ///
    /// If true, the count will always be 1.
    fn is_indexed(&self) -> bool;

    /// The number of agents of this type that will be used in the system.
    fn count(&self) -> usize;

    /// Return the actions that may be taken by an agent instance with some state when receiving a
    /// message.
    fn message_response(
        &self,
        instance_index: usize,
        state_id: StateId,
        payload: &Payload,
    ) -> Response<StateId, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn time_response(&self, instance_index: usize, state_id: StateId)
        -> Response<StateId, Payload>;

    /// Whether any agent in the state is deferring messages.
    fn state_is_deferring(&self, state_id: StateId) -> bool;

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

/// The data we need to implement an agent type.
///
/// This should be placed in a `Singleton` to allow the agent states to get services from it.
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
    _phantom: PhantomData<Payload>,
}

impl<State: KeyLike + Validated, StateId: IndexLike, Payload: KeyLike + Validated>
    AgentTypeData<State, StateId, Payload>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(name: &'static str, is_indexed: bool, count: usize) -> Self {
        assert!(
            !is_indexed || count == 1,
            "specifying multiple instances for a non-indexed agent type"
        );
        AgentTypeData {
            name,
            count,
            is_indexed,
            states: Arc::new(RwLock::new(Memoize::new(true, Some(StateId::max_value())))),
            _phantom: PhantomData,
        }
    }
}

/// A trait for a single agent state.
pub trait AgentState<State, Payload> {
    /// Return the short state name.
    fn name(&self) -> &'static str;

    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// message.
    fn message_response(
        &self,
        instance_index: usize,
        payload: &Payload,
    ) -> Response<State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn time_response(&self, instance_index: usize) -> Response<State, Payload>;

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self) -> bool {
        false
    }
}

pub trait Validated {
    /// If this object is invalid, return why.
    fn invalid(&self) -> Option<&'static str> {
        None
    }
}

impl<
        State: KeyLike + Validated + AgentState<State, Payload>,
        StateId: IndexLike,
        Payload: KeyLike + Validated,
    > AgentTypeData<State, StateId, Payload>
{
    fn translate_response(&self, response: Response<State, Payload>) -> Response<StateId, Payload> {
        match response {
            Response::Ignore => Response::Ignore,
            Response::Defer => Response::Defer,
            Response::Do1(action) => Response::Do1(self.translate_action(action)),
            Response::Do1Of2(action1, action2) => Response::Do1Of2(
                self.translate_action(action1),
                self.translate_action(action2),
            ),
            Response::Do1Of3(action1, action2, action3) => Response::Do1Of3(
                self.translate_action(action1),
                self.translate_action(action2),
                self.translate_action(action3),
            ),
            Response::Do1Of4(action1, action2, action3, action4) => Response::Do1Of4(
                self.translate_action(action1),
                self.translate_action(action2),
                self.translate_action(action3),
                self.translate_action(action4),
            ),
        }
    }

    fn translate_action(&self, action: Action<State, Payload>) -> Action<StateId, Payload> {
        match action {
            Action::Defer => Action::Defer,

            Action::Ignore => Action::Ignore,
            Action::Change(state) => Action::Change(self.translate_state(state)),

            Action::Send1(emit) => Action::Send1(emit),
            Action::ChangeAndSend1(state, emit) => {
                Action::ChangeAndSend1(self.translate_state(state), emit)
            }

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
            }
        }
    }

    fn translate_state(&self, state: State) -> StateId {
        if let Some(state_id) = self.states.read().unwrap().lookup(&state) {
            *state_id
        } else {
            self.states.write().unwrap().store(state).id
        }
    }
}

impl<
        State: KeyLike + Validated + AgentState<State, Payload>,
        StateId: IndexLike,
        MessageId: IndexLike,
        Payload: KeyLike + Validated,
    > AgentType<StateId, MessageId, Payload> for AgentTypeData<State, StateId, Payload>
{
    fn name(&self) -> &'static str {
        &self.name
    }

    fn is_indexed(&self) -> bool {
        self.is_indexed
    }

    fn count(&self) -> usize {
        self.count
    }

    fn message_response(
        &self,
        instance_index: usize,
        state_id: StateId,
        payload: &Payload,
    ) -> Response<StateId, Payload> {
        self.translate_response(
            self.states
                .read()
                .unwrap()
                .get(state_id)
                .message_response(instance_index, payload),
        )
    }

    fn time_response(
        &self,
        instance_index: usize,
        state_id: StateId,
    ) -> Response<StateId, Payload> {
        self.translate_response(
            self.states
                .read()
                .unwrap()
                .get(state_id)
                .time_response(instance_index),
        )
    }

    fn state_is_deferring(&self, state_id: StateId) -> bool {
        self.states.read().unwrap().get(state_id).is_deferring()
    }

    fn state_display(&self, state_id: StateId, callback: Box<dyn FnOnce(&str)>) {
        (callback)(self.states.read().unwrap().display(state_id));
    }

    fn state_name(&self, state_id: StateId) -> &'static str {
        self.states.read().unwrap().get(state_id).name()
    }
}

/// Possible way to order a message.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub enum MessageOrder {
    /// Deliver the message immediately, before any other message.
    Immediate,

    /// Deliver the message in any order relative to all other unordered messages.
    Unordered,

    /// Deliver the message in the specified order relative to all other ordered messages between
    /// the same source and target.
    Ordered(u16),
}

/// A message in-flight between agents.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct Message<Payload> {
    /// How the message is ordered.
    pub order: MessageOrder,

    /// The source agent index.
    pub source: usize,

    /// The target agent index.
    pub target: usize,

    /// The actual payload.
    pub payload: Payload,

    /// The replaced message, if any.
    pub replaced: Option<Payload>,
}

impl<Payload> Display for Message<Payload> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "todo")
    }
}

/// An indicator that something is invalid.
#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub enum Invalid<Payload> {
    Configuration(&'static str),
    Agent(&'static str, Option<usize>, &'static str),
    Message(Message<Payload>, &'static str),
}

impl<Payload> Display for Invalid<Payload> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "todo")
    }
}

/// A complete system configuration.
///
/// We will have a *lot* of these, so keeping their size down and avoiding heap memory as much as
/// possible is critical. The maximal sizes were chosen so that the configuration plus its memoized
/// identifier will fit together inside exactly one cache lines, which should make this more
/// cache-friendly when placed inside a hash table.
#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub struct Configuration<
    StateId,
    MessageId,
    InvalidId,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The state of each agent.
    pub states: [StateId; MAX_AGENTS],

    /// The messages sent by each agent.
    pub messages: [MessageId; MAX_MESSAGES],

    /// The invalid condition, if any.
    pub invalid: InvalidId,

    /// The index of the immediate message, or 255 if there is none.
    pub immediate: u8,
}

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > Display for Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "todo")
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
    fn remove_message(&mut self, mut message_index: usize) {
        debug_assert!(self.messages[message_index] != MessageId::max_value());

        loop {
            let next_message_index = message_index + 1;
            let next_message_id = if next_message_index < MAX_MESSAGES {
                self.messages[next_message_index]
            } else {
                MessageId::max_value()
            };
            self.messages[message_index] = next_message_id;
            if next_message_id == MessageId::max_value() {
                return;
            }
            message_index = next_message_index;
        }
    }

    /// Change the state of an agent in the configuration.
    fn change_state(&mut self, agent_index: usize, state_id: StateId) {
        self.states[agent_index] = state_id;
    }
}

/// A transition from a given configuration.
#[derive(Copy, Clone)]
pub struct Outgoing<ConfigurationId> {
    /// The identifier of the target configuration.
    pub to: ConfigurationId,

    /// The index of the message of the source configuration that was delivered to its target agent
    /// to reach the target configuration.
    ///
    /// We use the configuration identifier type as this is guaranteed to be large enough, and
    /// anything smaller will not reduce the structure size, if we want fields to be properly
    /// aligned.
    pub message: ConfigurationId,
}

/// A transition to a given configuration.
#[derive(Copy, Clone)]
pub struct Incoming<ConfigurationId> {
    /// The identifier of the source configuration.
    pub from: ConfigurationId,

    /// The index of the message of the source configuration that was delivered to its target agent
    /// to reach the target configuration.
    ///
    /// We use the configuration identifier type as this is guaranteed to be large enough, and
    /// anything smaller will not reduce the structure size, if we want fields to be properly
    /// aligned.
    pub message: ConfigurationId,
}

/// A complete model.
pub struct Model<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: KeyLike + Validated,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The agent types that will be used in the model.
    pub types: Vec<<Self as MetaModel>::AgentTypeArc>,

    /// The type of each agent.
    pub agent_types: Vec<<Self as MetaModel>::AgentTypeArc>,

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
    pub outgoing: RwLock<Vec<RwLock<Vec<<Self as MetaModel>::Outgoing>>>>,

    /// For each configuration, which configuration can reach it.
    pub incoming: RwLock<Vec<RwLock<Vec<<Self as MetaModel>::Incoming>>>>,
}

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

    /// The type of a response from an agent.
    type Response;

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

    /// The context for processing responses.
    type Context;
}

/// The context for processing a response.
#[derive(Clone)]
pub struct Context<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: KeyLike + Validated,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The configuration the agent belonged to.
    configuration: Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,

    /// The index of the agent that generated the response.
    agent_index: usize,

    /// The type of the agent.
    agent_type: Arc<dyn AgentType<StateId, MessageId, Payload> + Send + Sync>,

    /// The index of the agent in its type.
    instance_index: usize,

    /// The identifier state of the state of the agent when generating the response.
    state_id: StateId,

    /// The identifier of the message that the agent received, or `max_value` if the agent received
    /// a time event.
    message_id: MessageId,

    /// The incoming transition into the new configuration to be generated.
    incoming: Incoming<ConfigurationId>,
}

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: KeyLike + Validated,
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
    type Response = Response<StateId, Payload>;
    type Action = Action<StateId, Payload>;
    type Emit = Emit<Payload>;
    type Invalid = Invalid<Payload>;
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
        Payload: KeyLike + Validated,
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
        let mut agent_types: Vec<<Self as MetaModel>::AgentTypeArc> = vec![];
        let mut first_instances: Vec<usize> = vec![];

        for agent_type in types.iter() {
            let count = agent_type.count();
            assert!(
                count > 0,
                "zero instances requested for the type {}",
                agent_type.name()
            );
            let first_instance = agent_types.len();
            for _ in 0..count {
                first_instances.push(first_instance);
                agent_types.push(agent_type.clone());
            }
        }

        let model = Model {
            types,
            agent_types,
            first_instances,
            validators,
            configurations: RwLock::new(Memoize::new(
                false,
                Some(Self::invalid_configuration_id()),
            )),
            messages: Arc::new(RwLock::new(Memoize::new(
                true,
                Some(MessageId::max_value()),
            ))),
            invalids: RwLock::new(Memoize::new(true, Some(Self::invalid_invalid_id()))),
            outgoing: RwLock::new(Vec::new()),
            incoming: RwLock::new(Vec::new()),
        };

        let initial_configuration = Configuration {
            states: [StateId::from_usize(0).unwrap(); MAX_AGENTS],
            messages: [MessageId::max_value(); MAX_MESSAGES],
            invalid: InvalidId::max_value(),
            immediate: u8::max_value(),
        };

        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| model.reach_configuration(initial_configuration, None));

        model
    }

    fn reach_configuration(
        &self,
        mut configuration: <Self as MetaModel>::Configuration,
        incoming: Option<<Self as MetaModel>::Incoming>,
    ) {
        self.validate_configuration(&mut configuration);

        let stored = self.store_configuration(configuration, incoming);
        let configuration_id = stored.id;
        if !stored.is_new {
            if let Some(transition) = incoming {
                self.incoming.read().unwrap()
                    [ConfigurationId::to_usize(&configuration_id).unwrap()]
                .write()
                .unwrap()
                .push(transition);
            }
            return;
        }

        let agents_count = self.agent_types.len();

        let messages_count = if configuration.immediate == u8::max_value() {
            configuration
                .messages
                .iter()
                .position(|&message_id| message_id == Self::invalid_message_id())
                .unwrap_or(MAX_MESSAGES)
        } else {
            1
        };
        let events_count = agents_count + messages_count;
        (0..events_count).into_par_iter().for_each(|event_index| {
            if event_index < agents_count {
                self.time_transition(configuration_id, configuration, event_index);
            } else if configuration.immediate == 1 {
                self.message_transition(
                    configuration_id,
                    configuration,
                    u8::to_usize(&configuration.immediate).unwrap(),
                );
            } else {
                self.message_transition(
                    configuration_id,
                    configuration,
                    event_index - agents_count,
                );
            }
        });
    }

    fn time_transition(
        &self,
        configuration_id: ConfigurationId,
        configuration: <Self as MetaModel>::Configuration,
        agent_index: usize,
    ) {
        let state_id = configuration.states[agent_index];
        let agent_type = self.agent_types[agent_index].clone();
        let instance_index = self.instance_index(agent_index);
        let response = agent_type.time_response(instance_index, state_id);

        if response == Response::Ignore {
            return;
        }

        let state_id = configuration.states[agent_index];
        let message_id = Self::invalid_message_id();
        let incoming = Incoming {
            from: configuration_id,
            message: ConfigurationId::max_value(),
        };

        let context = Context {
            configuration,
            agent_index,
            agent_type,
            instance_index,
            state_id,
            message_id,
            incoming,
        };
        self.response_transition(context, response);
    }

    fn message_transition(
        &self,
        configuration_id: ConfigurationId,
        mut configuration: <Self as MetaModel>::Configuration,
        message_index: usize,
    ) {
        let message_id = configuration.messages[message_index];

        let (agent_index, payload) = {
            let messages = self.messages.read().unwrap();
            let message = messages.get(message_id);
            (message.target, message.payload)
        };

        let instance_index = self.instance_index(agent_index);
        let state_id = configuration.states[agent_index];
        let agent_type = self.agent_types[agent_index].clone();
        let response = agent_type.message_response(instance_index, state_id, &payload);

        let incoming = Incoming {
            from: configuration_id,
            message: ConfigurationId::from_usize(message_index).unwrap(),
        };

        configuration.remove_message(message_index);

        let context = Context {
            configuration,
            agent_index,
            agent_type,
            instance_index,
            state_id,
            message_id,
            incoming,
        };
        self.response_transition(context, response);
    }

    fn response_transition(
        &self,
        context: <Self as MetaModel>::Context,
        response: <Self as MetaModel>::Response,
    ) {
        match response {
            Response::Defer => self.is_deferring_message(context),
            Response::Ignore => self.is_ignoring_message(context),

            Response::Do1(action) => self.action_transition(context, action),

            Response::Do1Of2(action1, action2) => {
                self.action_transition(context.clone(), action1);
                self.action_transition(context, action2);
            }

            Response::Do1Of3(action1, action2, action3) => {
                self.action_transition(context.clone(), action1);
                self.action_transition(context.clone(), action2);
                self.action_transition(context, action3);
            }

            Response::Do1Of4(action1, action2, action3, action4) => {
                self.action_transition(context.clone(), action1);
                self.action_transition(context.clone(), action2);
                self.action_transition(context.clone(), action3);
                self.action_transition(context, action4);
            }
        }
    }

    fn action_transition(
        &self,
        mut context: <Self as MetaModel>::Context,
        action: <Self as MetaModel>::Action,
    ) {
        match action {
            Action::Defer => self.is_deferring_message(context),

            Action::Ignore => self.is_ignoring_message(context),
            Action::Change(state_id) => {
                context
                    .configuration
                    .change_state(context.agent_index, state_id);
                self.reach_configuration(context.configuration, Some(context.incoming));
            }

            Action::Send1(emit) => self.emit_transition(context, emit),

            Action::ChangeAndSend2(state_id, emit1, emit2) => {
                context
                    .configuration
                    .change_state(context.agent_index, state_id);
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context, emit2);
            }

            Action::Send2(emit1, emit2) => {
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context, emit2);
            }

            Action::ChangeAndSend3(state_id, emit1, emit2, emit3) => {
                context
                    .configuration
                    .change_state(context.agent_index, state_id);
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context.clone(), emit2);
                self.emit_transition(context, emit3);
            }

            Action::Send3(emit1, emit2, emit3) => {
                self.emit_transition(context.clone(), emit1);
                self.emit_transition(context.clone(), emit2);
                self.emit_transition(context, emit3);
            }

            Action::ChangeAndSend4(state_id, emit1, emit2, emit3, emit4) => {
                context
                    .configuration
                    .change_state(context.agent_index, state_id);
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
            }

            _ => panic!("todo"),
        }
    }

    fn emit_transition(
        &self,
        _context: <Self as MetaModel>::Context,
        emit: <Self as MetaModel>::Emit,
    ) {
        match emit {
            // Emit::Immediate(payload, target) =>
            // Emit::Unordered(payload, target) =>
            // Emit::Ordered(payload, target) =>
            // Emit::UnorderedReplacement(callback, payload, target) =>
            // Emit::OrderedReplacement(callback, payload, target) =>
            // Emit::ImmediateReplacement(callback, payload, target) =>
            _ => panic!("todo"),
        }
    }

    fn is_ignoring_message(&self, context: <Self as MetaModel>::Context) {
        if context.incoming.message != Self::invalid_configuration_id() {
            self.reach_configuration(context.configuration, Some(context.incoming));
        }
    }

    fn is_deferring_message(&self, context: <Self as MetaModel>::Context) {
        if context.message_id == Self::invalid_message_id() {
            let name = context.agent_type.name();
            let is_indexed = context.agent_type.is_indexed();
            let instance_index = context.instance_index;
            context.agent_type.state_display(context.state_id, Box::new(move |state| {
                if is_indexed {
                    panic!("agent {}-{} is deferring (should be ignoring) the time event while in the state {}",
                           name, instance_index, state);
                } else {
                    panic!("agent {} is deferring (should be ignoring) the time event while in the state {}",
                           name, state);
                }
            }));
        } else if !context.agent_type.state_is_deferring(context.state_id) {
            let name = context.agent_type.name();
            let display = self
                .messages
                .read()
                .unwrap()
                .display(context.message_id)
                .to_string();
            let is_indexed = context.agent_type.is_indexed();
            let instance_index = context.instance_index;
            context.agent_type.state_display(context.state_id, Box::new(move |state| {
                if is_indexed {
                    panic!("agent {}-{} is deferring the message {} while in the non-deferring state {}",
                           name, instance_index, display, state);
                } else {
                    panic!("agent {} is deferring the message {} while in the non-deferring state {}",
                           name, display, state);
                }
            }));
        }
    }

    fn validate_configuration(&self, configuration: &mut <Self as MetaModel>::Configuration) {
        if configuration.invalid == Self::invalid_invalid_id() {
            for validator in self.validators.iter() {
                if let Some(reason) = validator(&configuration) {
                    let invalid = <Self as MetaModel>::Invalid::Configuration(reason);
                    if let Some(invalid_id) = self.invalids.read().unwrap().lookup(&invalid) {
                        configuration.invalid = *invalid_id;
                    } else {
                        configuration.invalid = self.invalids.write().unwrap().store(invalid).id;
                    }
                    break;
                }
            }
        }
    }

    fn store_configuration(
        &self,
        configuration: <Self as MetaModel>::Configuration,
        incoming: Option<<Self as MetaModel>::Incoming>,
    ) -> Stored<ConfigurationId> {
        let stored = self.configurations.write().unwrap().store(configuration);
        let is_new = stored.is_new;
        let to_configuration_id = stored.id;
        if is_new || incoming.is_some() {
            let mut incoming_vector = self.incoming.write().unwrap();

            if is_new {
                incoming_vector.push(RwLock::new(Vec::new()));
                self.outgoing.write().unwrap().push(RwLock::new(Vec::new()));
            }

            if let Some(transition) = incoming {
                incoming_vector[ConfigurationId::to_usize(&to_configuration_id).unwrap()]
                    .write()
                    .unwrap()
                    .push(transition);
            }
        }

        if let Some(transition) = incoming {
            let from_configuration_id = transition.from;
            let outgoing = Outgoing {
                to: to_configuration_id,
                message: transition.message,
            };
            self.outgoing.write().unwrap()
                [ConfigurationId::to_usize(&from_configuration_id).unwrap()]
            .write()
            .unwrap()
            .push(outgoing);
        }

        stored
    }

    /// Return the index of the agent with the specified type name.
    ///
    /// If more than one agent of this type exist, also specify its index within its type.
    pub fn agent_index(&self, name: &'static str, instance_index: Option<usize>) -> usize {
        let mut agent_index: usize = 0;
        for agent_type in self.types.iter() {
            let count = agent_type.count();
            if agent_type.name() != name {
                agent_index += count;
            } else {
                match instance_index {
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
    pub fn instance_index(&self, agent_index: usize) -> usize {
        agent_index - self.first_instances[agent_index]
    }

    /// Return whether all the reachable configurations are valid.
    pub fn is_valid(&self) -> bool {
        self.invalids.read().unwrap().usize() == 0
    }

    /// An invalid state identifier.
    pub fn invalid_state_id() -> StateId {
        StateId::max_value()
    }

    /// An invalid message identifier.
    pub fn invalid_message_id() -> MessageId {
        MessageId::max_value()
    }

    /// An invalid configuration identifier.
    pub fn invalid_configuration_id() -> ConfigurationId {
        ConfigurationId::max_value()
    }

    /// An invalid invalid condition identifier.
    pub fn invalid_invalid_id() -> InvalidId {
        InvalidId::max_value()
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
