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
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Sub;
use std::sync::Arc;
use std::sync::RwLock;

/// A trait for anything we use as a key in a HashMap.
pub trait KeyLike = Eq + Hash + Copy + Sized + Display;

/// A trait for anything we use as a zero-based index.
pub trait IndexLike = KeyLike + Bounded + FromPrimitive + ToPrimitive + Unsigned;

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

    /// Given a value that does not exist in the memory, insert it and return its short identifier.
    pub fn insert(&mut self, value: T) -> I {
        debug_assert!(
            self.lookup(&value).is_none(),
            "inserting an already existing value {}",
            value
        );
        if self.usize() > self.max_id {
            panic!("too many memoized objects");
        }
        let id = self.size();
        self.id_by_value.insert(value, id);
        self.value_by_id.push(value);
        if let Some(display_by_id) = &mut self.display_by_id {
            display_by_id.push(format!("{}", value));
        }
        id
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
pub struct Emit<AgentIndex, Payload> {
    /// The payload that will be delivered to the target agent.
    pub payload: Payload,

    /// Where to send the message to.
    pub target: AgentIndex,

    /// Allow this message to replace another message from the same source to the same target, if
    /// this function returns ``true``.
    pub is_replacement: Option<fn(Payload) -> bool>,

    /// Whether this message needs to be delivered immediately, before any other.
    pub is_immediate: bool,

    /// Whether this message needs to be delivered only after all other ordered messahes between the
    /// same source and target have been delivered.
    pub is_ordered: bool,
}

impl<AgentIndex: IndexLike, Payload: KeyLike + Default> Default for Emit<AgentIndex, Payload> {
    fn default() -> Self {
        Emit {
            payload: Default::default(),
            target: AgentIndex::max_value(),
            is_replacement: None,
            is_immediate: false,
            is_ordered: false,
        }
    }
}

/// Specify an action the agent may take as a response to an event.
pub enum Do<AgentIndex, State, Payload> {
    /// Consume (ignore) the event, keep the state the same, do not send any messages.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Response.Ignore` value.
    Ignore,

    /// Defer the event, keep the state the same, do not send any messages.
    ///
    /// This is only useful if it is needed to be listed as an alternative with other actions;
    /// Otherwise, use the `Response.Defer` value.
    ///
    /// This is only allowed if the agent's `state_is_deferring`, waiting for
    /// specific message(s) to resume normal operations.
    Defer,

    /// Consume (handle) the event, change the agent state, do not send any messages.
    SendNone(State),

    /// Consume (handle) the event, change the agent state, send a single message.
    SendOne(State, Emit<AgentIndex, Payload>),

    /// Consume (handle) the event, change the agent state, send two messages.
    SendTwo(State, Emit<AgentIndex, Payload>, Emit<AgentIndex, Payload>),

    /// Consume (handle) the event, change the agent state, send three messages.
    SendThree(
        State,
        Emit<AgentIndex, Payload>,
        Emit<AgentIndex, Payload>,
        Emit<AgentIndex, Payload>,
    ),

    /// Consume (handle) the event, change the agent state, send four messages.
    SendFour(
        State,
        Emit<AgentIndex, Payload>,
        Emit<AgentIndex, Payload>,
        Emit<AgentIndex, Payload>,
        Emit<AgentIndex, Payload>,
    ),
}

/// The response from an agent on some event.
pub enum Response<AgentIndex, State, Payload> {
    /// Ignore the event.
    ///
    /// This has the same effect as `One(Do.Ignore)`.
    Ignore,

    /// Defer handling the event.
    ///
    /// This has the same effect as `One(Do.Defer)`.
    Defer,

    /// A single action (deterministic).
    One(Do<AgentIndex, State, Payload>),

    /// One of two alternative actions (non-deterministic).
    Two(
        Do<AgentIndex, State, Payload>,
        Do<AgentIndex, State, Payload>,
    ),

    /// One of three alternative actions (non-deterministic).
    Three(
        Do<AgentIndex, State, Payload>,
        Do<AgentIndex, State, Payload>,
        Do<AgentIndex, State, Payload>,
    ),

    /// One of four alternative actions (non-deterministic).
    Four(
        Do<AgentIndex, State, Payload>,
        Do<AgentIndex, State, Payload>,
        Do<AgentIndex, State, Payload>,
        Do<AgentIndex, State, Payload>,
    ),
}

/// A trait describing a set of agents of some type.
pub trait AgentType<AgentIndex, StateId, Payload> {
    /// The name of the type of the agents.
    fn name(&self) -> &'static str;

    /// The number of agents of this type that will be used in the system.
    fn count(&self) -> AgentIndex;

    /// Return the actions that may be taken by an agent instance with some state when receiving a
    /// message.
    fn message_response(
        &self,
        agent_index: AgentIndex,
        state_id: StateId,
        payload: &Payload,
    ) -> Response<AgentIndex, StateId, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn time_response(
        &self,
        agent_index: AgentIndex,
        state_id: StateId,
    ) -> Response<AgentIndex, StateId, Payload>;

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
pub struct AgentTypeData<AgentIndex, StateId, State, Payload> {
    /// Memoization of the agent states.
    states: Arc<RwLock<Memoize<State, StateId>>>,

    /// The name of the agent type.
    name: &'static str,

    /// The number of instances of this type we'll be using in the system.
    count: AgentIndex,

    /// Trick the compiler into thinking we have a field of type Payload.
    _phantom: PhantomData<Payload>,
}

impl<AgentIndex: IndexLike, StateId: IndexLike, State: KeyLike, Payload: KeyLike>
    AgentTypeData<AgentIndex, StateId, State, Payload>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(name: &'static str, count: AgentIndex) -> Self {
        AgentTypeData {
            name,
            count,
            states: Arc::new(RwLock::new(Memoize::new(true, None))),
            _phantom: PhantomData,
        }
    }
}

/// A trait for a single agent state.
pub trait AgentState<AgentIndex, State, Payload> {
    /// Return the short state name.
    fn name(&self) -> &'static str;

    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// message.
    fn message_response(
        &self,
        agent_index: AgentIndex,
        payload: &Payload,
    ) -> Response<AgentIndex, State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn time_response(&self, agent_index: AgentIndex) -> Response<AgentIndex, State, Payload>;

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self) -> bool {
        false
    }
}

impl<
        AgentIndex: IndexLike,
        StateId: IndexLike,
        State: KeyLike + AgentState<AgentIndex, State, Payload>,
        Payload: KeyLike,
    > AgentTypeData<AgentIndex, StateId, State, Payload>
{
    fn translate_response(
        &self,
        response: Response<AgentIndex, State, Payload>,
    ) -> Response<AgentIndex, StateId, Payload> {
        match response {
            Response::Ignore => Response::Ignore,
            Response::Defer => Response::Defer,
            Response::One(action) => Response::One(self.translate_action(action)),
            Response::Two(action1, action2) => Response::Two(
                self.translate_action(action1),
                self.translate_action(action2),
            ),
            Response::Three(action1, action2, action3) => Response::Three(
                self.translate_action(action1),
                self.translate_action(action2),
                self.translate_action(action3),
            ),
            Response::Four(action1, action2, action3, action4) => Response::Four(
                self.translate_action(action1),
                self.translate_action(action2),
                self.translate_action(action3),
                self.translate_action(action4),
            ),
        }
    }

    fn translate_action(
        &self,
        action: Do<AgentIndex, State, Payload>,
    ) -> Do<AgentIndex, StateId, Payload> {
        match action {
            Do::Ignore => Do::Ignore,
            Do::Defer => Do::Defer,
            Do::SendNone(state) => Do::SendNone(self.translate_state(state)),
            Do::SendOne(state, emit) => Do::SendOne(self.translate_state(state), emit),
            Do::SendTwo(state, emit1, emit2) => {
                Do::SendTwo(self.translate_state(state), emit1, emit2)
            }
            Do::SendThree(state, emit1, emit2, emit3) => {
                Do::SendThree(self.translate_state(state), emit1, emit2, emit3)
            }
            Do::SendFour(state, emit1, emit2, emit3, emit4) => {
                Do::SendFour(self.translate_state(state), emit1, emit2, emit3, emit4)
            }
        }
    }

    fn translate_state(&self, state: State) -> StateId {
        if let Some(state_id) = self.states.read().unwrap().lookup(&state) {
            *state_id
        } else {
            self.states.write().unwrap().insert(state)
        }
    }
}

impl<
        AgentIndex: IndexLike,
        StateId: IndexLike,
        State: KeyLike + AgentState<AgentIndex, State, Payload>,
        Payload: KeyLike,
    > AgentType<AgentIndex, StateId, Payload>
    for AgentTypeData<AgentIndex, StateId, State, Payload>
{
    fn name(&self) -> &'static str {
        &self.name
    }

    fn count(&self) -> AgentIndex {
        self.count
    }

    fn message_response(
        &self,
        agent_index: AgentIndex,
        state_id: StateId,
        payload: &Payload,
    ) -> Response<AgentIndex, StateId, Payload> {
        self.translate_response(
            self.states
                .read()
                .unwrap()
                .get(state_id)
                .message_response(agent_index, payload),
        )
    }

    fn time_response(
        &self,
        agent_index: AgentIndex,
        state_id: StateId,
    ) -> Response<AgentIndex, StateId, Payload> {
        self.translate_response(
            self.states
                .read()
                .unwrap()
                .get(state_id)
                .time_response(agent_index),
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

/// A message in-flight between agents.
pub struct Message<AgentIndex, Payload, MessageOrder> {
    /// The source agent index.
    pub source: AgentIndex,

    /// The target agent index.
    pub target: AgentIndex,

    /// How the message is ordered.
    ///
    /// Zero means the message is the first to be delivered next, bypassing anything else. The
    /// maximal value means the message is unordered. Otherwise, the value is the order of the
    /// message relative to other ordered messages between the same source and target agents.
    pub order: MessageOrder,

    /// The actual payload.
    pub payload: Payload,

    /// The replaced message, if any.
    pub replaced: Option<Payload>,
}

impl<AgentIndex: IndexLike, Payload: KeyLike + Default, MessageOrder: IndexLike> Default
    for Message<AgentIndex, Payload, MessageOrder>
{
    fn default() -> Self {
        Message {
            source: AgentIndex::max_value(),
            target: AgentIndex::max_value(),
            order: MessageOrder::max_value(),
            payload: Default::default(),
            replaced: None,
        }
    }
}

/// An indicator that something is invalid.
pub enum Invalid<AgentIndex, Payload, MessageOrder> {
    Configuration(&'static str),
    Agent(&'static str, Option<AgentIndex>, &'static str),
    Message(Message<AgentIndex, Payload, MessageOrder>, &'static str),
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
}

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
            states: [StateId::max_value(); MAX_AGENTS],
            messages: [MessageId::max_value(); MAX_MESSAGES],
            invalid: InvalidId::max_value(),
        }
    }
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

/// A transition from a given configuration.
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
    AgentIndex,
    StateId,
    MessageId,
    InvalidId,
    ConfigurationId,
    Payload,
    MessageOrder,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The agent types that will be used in the model.
    pub types: Vec<Box<dyn AgentType<AgentIndex, StateId, Payload>>>,

    /// Memoization of the configurations.
    pub configurations: RwLock<
        Memoize<
            Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,
            ConfigurationId,
        >,
    >,

    /// For each configuration, which configuration is reachable from it.
    pub outgoing: RwLock<Vec<RwLock<Vec<Outgoing<ConfigurationId>>>>>,

    /// For each configuration, which configuration can reach it.
    pub incoming: RwLock<Vec<RwLock<Vec<Incoming<ConfigurationId>>>>>,

    /// Count of invalid configurations.
    pub invalids: RwLock<ConfigurationId>,

    /// Trick the compiler into thinking we have a field of type MessageOrder.
    _message_order: PhantomData<MessageOrder>,
}

/// Allow querying the model's meta-parameters.
pub trait MetaModel {
    /// The type of agent indices.
    type AgentIndex;

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

    /// The type of message orders.
    type MessageOrder;

    /// The maximal number of agents.
    const MAX_AGENTS: usize;

    /// The maximal number of in-flight messages.
    const MAX_MESSAGES: usize;

    /// The type of  boxed agent type.
    type AgentTypeBox;

    /// The type of in-flight messages.
    type Message;

    /// The type of invalid conditions.
    type Invalid;

    /// The type of the included configurations.
    type Configuration;

    /// The type of the incoming transitions.
    type Incoming;

    /// The type of the outgoing transitions.
    type Outgoing;
}

impl<
        AgentIndex: IndexLike,
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: KeyLike,
        MessageOrder: IndexLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > MetaModel
    for Model<
        AgentIndex,
        StateId,
        MessageId,
        InvalidId,
        ConfigurationId,
        Payload,
        MessageOrder,
        MAX_AGENTS,
        MAX_MESSAGES,
    >
{
    type AgentIndex = AgentIndex;
    type StateId = StateId;
    type MessageId = MessageId;
    type InvalidId = InvalidId;
    type ConfigurationId = ConfigurationId;
    type Payload = Payload;
    type MessageOrder = MessageOrder;
    const MAX_AGENTS: usize = MAX_AGENTS;
    const MAX_MESSAGES: usize = MAX_MESSAGES;

    type AgentTypeBox = Box<dyn AgentType<AgentIndex, StateId, Payload>>;
    type Message = Message<AgentIndex, Payload, MessageOrder>;
    type Invalid = Invalid<AgentIndex, Payload, MessageOrder>;
    type Configuration = Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>;
    type Incoming = Incoming<ConfigurationId>;
    type Outgoing = Outgoing<ConfigurationId>;
}

impl<
        AgentIndex: IndexLike,
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: KeyLike,
        MessageOrder: IndexLike + Sub<Output = MessageOrder>,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    >
    Model<
        AgentIndex,
        StateId,
        MessageId,
        InvalidId,
        ConfigurationId,
        Payload,
        MessageOrder,
        MAX_AGENTS,
        MAX_MESSAGES,
    >
{
    /// Create a new model.
    pub fn new(types: Vec<<Self as MetaModel>::AgentTypeBox>) -> Self {
        let mut model = Model {
            types,
            configurations: RwLock::new(Memoize::new(
                false,
                Some(Self::invalid_configuration_id()),
            )),
            outgoing: RwLock::new(Vec::new()),
            incoming: RwLock::new(Vec::new()),
            invalids: RwLock::new(ConfigurationId::from_usize(0).unwrap()),
            _message_order: Default::default(),
        };

        let initial_configuration = Configuration {
            states: [StateId::from_usize(0).unwrap(); MAX_AGENTS],
            ..Default::default()
        };
        model.insert_configuration(initial_configuration, None);

        model
    }

    fn insert_configuration(
        &mut self,
        configuration: <Self as MetaModel>::Configuration,
        incoming: Option<<Self as MetaModel>::Incoming>,
    ) {
        let configuration_id = self.configurations.write().unwrap().insert(configuration);
        self.outgoing.write().unwrap().push(RwLock::new(Vec::new()));
        let mut incoming_vector = self.incoming.write().unwrap();
        incoming_vector.push(RwLock::new(Vec::new()));

        if let Some(transition) = incoming {
            incoming_vector[ConfigurationId::to_usize(&configuration_id).unwrap()]
                .write()
                .unwrap()
                .push(transition);
        }
    }

    /// Return the index of the agent with the specified type name.
    ///
    /// If more than one agent of this type exist, also specify its index within its type.
    pub fn agent_index(&self, name: &'static str, index: Option<usize>) -> AgentIndex {
        let mut agent_index: usize = 0;
        for agent_type in self.types.iter() {
            if agent_type.name() != name {
                agent_index += AgentIndex::to_usize(&agent_type.count()).unwrap();
            } else {
                match index {
                    None => {
                        assert!(agent_type.count() == AgentIndex::from_usize(1).unwrap(),
                                "no index specified when locating an agent of type {} which has {} instances",
                                name, agent_type.count());
                    }
                    Some(index_in_type) => {
                        assert!(index_in_type < AgentIndex::to_usize(&agent_type.count()).unwrap(),
                                "too large index {} specified when locating an agent of type {} which has {} instances",
                                index_in_type, name, agent_type.count());
                        agent_index += index_in_type;
                    }
                }
                return AgentIndex::from_usize(agent_index).unwrap();
            }
        }
        panic!("looking for an agent of an unknown type {}", name);
    }

    /// An invalid agent index.
    pub fn invalid_agent_index() -> AgentIndex {
        AgentIndex::max_value()
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

    /// A message order for immediate messages.
    pub fn immediate_message_order() -> MessageOrder {
        MessageOrder::from_usize(0).unwrap()
    }

    /// A message order for unordered messages.
    pub fn unordered_message_order() -> MessageOrder {
        MessageOrder::max_value() - MessageOrder::from_usize(1).unwrap()
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
