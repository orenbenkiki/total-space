use crate::agents::*;
use crate::configurations::*;
use crate::diagrams::*;
use crate::memoize::*;
use crate::messages::*;
use crate::reactions::*;
use crate::utilities::*;

use clap::ArgMatches;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;
use std::cell::RefCell;
use std::cmp::max;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::io::stderr;
use std::io::Write;
use std::mem::swap;
use std::rc::Rc;
use std::str::FromStr;

// BEGIN MAYBE TESTED

thread_local! {
    /// The mast of configurations that can reach back to the initial state.
    static REACHABLE_CONFIGURATIONS_MASK: RefCell<Vec<bool>> = RefCell::new(vec![]);

    /// The error configuration.
    static ERROR_CONFIGURATION_ID: RefCell<usize> = RefCell::new(usize::max_value());
}

/// A transition from a given configuration.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Outgoing<MessageId: IndexLike, ConfigurationId: IndexLike> {
    /// The identifier of the target configuration.
    to_configuration_id: ConfigurationId,

    /// The identifier of the message that was delivered to its target agent to reach the target
    /// configuration.
    delivered_message_id: MessageId,
}

/// A transition to a given configuration.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Incoming<MessageId: IndexLike, ConfigurationId: IndexLike> {
    /// The identifier of the source configuration.
    from_configuration_id: ConfigurationId,

    /// The identifier of the message that was delivered to its target agent to reach the target
    /// configuration.
    delivered_message_id: MessageId,
}

// END MAYBE TESTED

/// A function to identify specific configurations.
pub(crate) enum ConditionFunction<Model: MetaModel> {
    /// Using just the configuration identifier.
    ById(fn(&Model, Model::ConfigurationId) -> bool),

    /// Using the configuration itself.
    ByConfig(fn(&Model, &Model::Configuration) -> bool),
}

impl<Model: MetaModel> Clone for ConditionFunction<Model> {
    /// Clone this (somehow can't be derived or declared as implemented).
    fn clone(&self) -> Self {
        match *self {
            ConditionFunction::ById(by_id) => ConditionFunction::ById(by_id),
            ConditionFunction::ByConfig(by_config) => ConditionFunction::ByConfig(by_config),
        }
    }
}

/// A path step (possibly negated named condition).
pub(crate) struct PathStep<Model: MetaModel> {
    /// The condition function.
    function: ConditionFunction<Model>,

    /// Whether to negate the condition.
    is_negated: bool,

    /// The name of the step.
    name: String,
}

impl<Model: MetaModel> PathStep<Model> {
    /// Clone this (can't be derived due to ConditionFunction).
    fn clone(&self) -> Self {
        PathStep {
            function: self.function.clone(),
            is_negated: self.is_negated,
            name: self.name.clone(),
        }
    }
}

/// A transition between configurations along a path.
#[derive(Debug)]
pub(crate) struct PathTransition<
    MessageId: IndexLike,
    ConfigurationId: IndexLike,
    const MAX_MESSAGES: usize,
> {
    /// The source configuration identifier.
    pub(crate) from_configuration_id: ConfigurationId,

    /// The identifier of the delivered message, if any.
    delivered_message_id: MessageId,

    /// The agent that received the message.
    agent_index: usize,

    /// The target configuration identifier.
    pub(crate) to_configuration_id: ConfigurationId,

    /// The name of the condition the target configuration satisfies.
    to_condition_name: Option<String>,
}

/// Identify related set of transition between agent states in the diagram.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
pub(crate) struct AgentStateTransitionContext<StateId: IndexLike> {
    /// The source configuration identifier.
    from_state_id: StateId,

    /// Whether the agent starting state was deferring.
    from_is_deferring: bool,

    /// The target configuration identifier.
    to_state_id: StateId,

    /// Whether the agent end state was deferring.
    to_is_deferring: bool,

    /// Whether the target state is "after" the source state.
    is_constraint: bool,
}

// BEGIN MAYBE TESTED

/// The context for processing event handling by an agent.
#[derive(Clone)]
pub(crate) struct Context<
    StateId: IndexLike,
    MessageId: IndexLike,
    InvalidId: IndexLike,
    ConfigurationId: IndexLike,
    Payload: DataLike,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// The index of the message of the source configuration that was delivered to its target agent
    /// to reach the target configuration.
    delivered_message_index: MessageIndex,

    /// The identifier of the message that the agent received, or `None` if the agent received an
    /// activity event.
    delivered_message_id: MessageId,

    /// Whether the delivered message was an immediate message.
    is_immediate: bool,

    /// The index of the agent that reacted to the event.
    agent_index: usize,

    /// The type of the agent that reacted to the event.
    agent_type: Rc<dyn AgentType<StateId, Payload>>,

    /// The index of the source agent in its type.
    agent_instance: usize,

    /// The identifier of the state of the agent when handling the event.
    agent_from_state_id: StateId,

    /// The incoming transition into the new configuration to be generated.
    incoming: Incoming<MessageId, ConfigurationId>,

    /// The configuration when delivering the event.
    from_configuration: Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,

    /// Incrementally updated to become the target configuration.
    to_configuration: Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,
}

// END MAYBE TESTED

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
    /// Optional validation for the model.
    validator: Option<Rc<dyn ModelValidator<Self>>>,

    /// The type of each agent.
    agent_types: Vec<<Self as MetaModel>::AgentTypeRc>,

    /// The label of each agent.
    agent_labels: Vec<Rc<String>>,

    /// The first index of the same type of each agent.
    first_indices: Vec<usize>,

    /// Memoization of the configurations.
    configurations: Memoize<<Self as MetaModel>::Configuration, ConfigurationId>,

    /// The total number of transitions.
    transitions_count: usize,

    /// Memoization of the in-flight messages.
    messages: Memoize<Message<Payload>, MessageId>,

    /// The label (display) of each message (to speed printing).
    label_of_message: Vec<String>,

    /// Map the full message identifier to the terse message identifier.
    terse_of_message_id: Vec<MessageId>,

    /// Map the terse message identifier to the full message identifier.
    message_of_terse_id: Vec<MessageId>,

    /// Map ordered message identifiers to their smaller order.
    decr_order_messages: HashMap<MessageId, MessageId>,

    /// Map ordered message identifiers to their larger order.
    incr_order_messages: HashMap<MessageId, MessageId>,

    /// Memoization of the invalid conditions.
    invalids: Memoize<<Self as MetaModel>::Invalid, InvalidId>,

    /// The label (display) of each invalid condition (to speed printing).
    label_of_invalid: Vec<String>,

    /// For each configuration, which configuration is reachable from it.
    outgoings: Vec<Vec<<Self as ModelTypes>::Outgoing>>,

    /// For each configuration, which configuration can reach it.
    incomings: Vec<Vec<<Self as ModelTypes>::Incoming>>,

    /// The maximal message string size we have seen so far.
    max_message_string_size: RefCell<usize>,

    /// The maximal invalid condition string size we have seen so far.
    max_invalid_string_size: RefCell<usize>,

    /// The maximal configuration string size we have seen so far.
    max_configuration_string_size: RefCell<usize>,

    /// Whether to print each new configuration as we reach it.
    pub(crate) print_progress_every: usize,

    /// Whether we'll be testing if the initial configuration is reachable from every configuration.
    pub(crate) ensure_init_is_reachable: bool,

    /// Whether to allow for invalid configurations.
    pub(crate) allow_invalid_configurations: bool,

    /// A step that, if reached, we can abort the computation.
    early_abort_step: Option<PathStep<Self>>,

    /// Whether we have reached the goal step and need to abort the computation.
    early_abort: bool,

    /// Named conditions on a configuration.
    conditions: HashMap<String, (ConditionFunction<Self>, &'static str)>,

    /// Mask of configurations that can reach back to the initial state.
    reachable_configurations_mask: Vec<bool>,

    /// Count of configurations that can reach back to the initial state.
    reachable_configurations_count: usize,

    /// The configurations we need to process.
    pending_configuration_ids: VecDeque<ConfigurationId>,
}

/// Allow querying the model's meta-parameters for public types.
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
    type AgentTypeRc;

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

    /// The type of a hash table entry for the configurations (should be cache-friendly).
    type ConfigurationHashEntry;

    /// A condition on model configurations.
    type Condition;
}

/// Validate parts of the model.
///
/// We are forced to wrap this in a separate object so that the application will be able to
/// implement the trait for it. It would have been nicer to implement the trait directly on the
/// model itself, but that isn't possible because it is defined in the total-space crate which is
/// different from the application crate.
pub trait ModelValidator<Model: MetaModel> {
    /// Return a reason that a configuration is invalid (unless it is valid).
    fn configuration_invalid_because(
        &self,
        _model: &Model,
        _configuration: &Model::Configuration,
    ) -> Option<&'static str> {
        None
    }

    /// Return a reason that a message is invalid (unless it is valid).
    fn message_invalid_because(
        &self,
        _model: &Model,
        _message: &Model::Message,
    ) -> Option<&'static str> {
        None
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

    type AgentTypeRc = Rc<dyn AgentType<StateId, Payload>>;
    type Message = Message<Payload>;
    type Reaction = Reaction<StateId, Payload>;
    type Action = Action<StateId, Payload>;
    type Emit = Emit<Payload>;
    type Invalid = Invalid<MessageId>;
    type Configuration = Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>;
    type ConfigurationHashEntry = (
        Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>,
        ConfigurationId,
    );
    type Condition =
        fn(&Self, &Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>) -> bool;
}

/// Allow querying the model's meta-parameters for private types.
pub(crate) trait ModelTypes: MetaModel {
    /// The type of the incoming transitions.
    type Incoming;

    /// The type of the outgoing transitions.
    type Outgoing;

    /// The context for processing event handling by an agent.
    type Context;

    /// A path step (possibly negated named condition).
    type PathStep;

    /// A transition along a path between configurations.
    type SequenceStep;

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

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: DataLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > ModelTypes
    for Model<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>
{
    type Incoming = Incoming<MessageId, ConfigurationId>;
    type Outgoing = Outgoing<MessageId, ConfigurationId>;
    type Context =
        Context<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>;
    type PathStep = PathStep<Self>;
    type SequenceStep = SequenceStep<StateId, MessageId>;
    type PathTransition = PathTransition<MessageId, ConfigurationId, MAX_MESSAGES>;
    type AgentStateTransitionContext = AgentStateTransitionContext<StateId>;
    type AgentStateTransitions =
        HashMap<AgentStateTransitionContext<StateId>, HashMap<Vec<MessageId>, Vec<MessageId>>>;
    type AgentStateSentByDelivered = HashMap<Vec<MessageId>, Vec<Vec<MessageId>>>;
    type SequenceState = SequenceState<MessageId, MAX_AGENTS, MAX_MESSAGES>;
}

// Model creation:

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
    /// Create a new model with validation.
    ///
    /// This allows querying the model for the `agent_index` of all the agents to use the results as
    /// a target for messages.
    pub fn with_validator(
        size: usize,
        last_agent_type: Rc<dyn AgentType<StateId, Payload>>,
        validator: Rc<dyn ModelValidator<Self>>,
    ) -> Self {
        Self::create(size, last_agent_type, Some(validator))
    }

    /// Create a new model.
    ///
    /// This allows querying the model for the `agent_index` of all the agents to use the results as
    /// a target for messages.
    pub fn new(size: usize, last_agent_type: Rc<dyn AgentType<StateId, Payload>>) -> Self {
        Self::create(size, last_agent_type, None)
    }

    fn create(
        size: usize,
        last_agent_type: Rc<dyn AgentType<StateId, Payload>>,
        validator: Option<Rc<dyn ModelValidator<Self>>>,
    ) -> Self {
        assert!(
            MAX_MESSAGES < MessageIndex::invalid().to_usize(),
            "MAX_MESSAGES {} is too large, must be less than {}",
            MAX_MESSAGES,
            MessageIndex::invalid() // NOT TESTED
        );

        let mut agent_types: Vec<<Self as MetaModel>::AgentTypeRc> = vec![];
        let mut first_indices: Vec<usize> = vec![];
        let mut agent_labels: Vec<Rc<String>> = vec![];

        Self::collect_agent_types(
            last_agent_type,
            &mut agent_types,
            &mut first_indices,
            &mut agent_labels,
        );

        let mut model = Self {
            validator,
            agent_types,
            agent_labels,
            first_indices,
            configurations: Memoize::with_capacity(usize::max_value(), size),
            transitions_count: 0,
            messages: Memoize::new(MessageId::invalid().to_usize()),
            label_of_message: vec![],
            terse_of_message_id: vec![],
            message_of_terse_id: vec![],
            decr_order_messages: HashMap::with_capacity(MessageId::invalid().to_usize()),
            incr_order_messages: HashMap::with_capacity(MessageId::invalid().to_usize()),
            invalids: Memoize::new(InvalidId::invalid().to_usize()),
            label_of_invalid: vec![],
            outgoings: vec![],
            incomings: vec![],
            max_message_string_size: RefCell::new(0),
            max_invalid_string_size: RefCell::new(0),
            max_configuration_string_size: RefCell::new(0),
            print_progress_every: 0,
            ensure_init_is_reachable: false,
            early_abort_step: None,
            early_abort: false,
            allow_invalid_configurations: false,
            conditions: HashMap::with_capacity(128),
            reachable_configurations_mask: Vec::new(),
            reachable_configurations_count: 0,
            pending_configuration_ids: VecDeque::new(),
        };

        model.add_standard_conditions();

        let mut initial_configuration = Configuration {
            state_ids: [StateId::invalid(); MAX_AGENTS],
            message_counts: [MessageIndex::from_usize(0); MAX_AGENTS],
            message_ids: [MessageId::invalid(); MAX_MESSAGES],
            invalid_id: InvalidId::invalid(),
        };

        assert!(model.agents_count() > 0);
        for agent_index in 0..model.agents_count() {
            initial_configuration.state_ids[agent_index] = StateId::from_usize(0);
        }

        let stored = model.store_configuration(initial_configuration);
        assert!(stored.is_new);
        assert!(stored.id.to_usize() == 0);

        model
    }

    fn collect_agent_types(
        last_agent_type: Rc<dyn AgentType<StateId, Payload>>,
        mut agent_types: &mut Vec<<Self as MetaModel>::AgentTypeRc>,
        mut first_indices: &mut Vec<usize>,
        mut agent_labels: &mut Vec<Rc<String>>,
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
            agent_labels.push(Rc::new(agent_label));
        }
    }

    fn add_standard_conditions(&mut self) {
        self.conditions.insert(
            "INIT".to_string(),
            (
                ConditionFunction::ById(Self::is_init_condition),
                "matches the initial configuration",
            ),
        );
        self.add_condition(
            "VALID",
            Self::is_valid_condition,
            "matches any valid configuration (is typically negated)",
        );
        self.add_condition(
            "IMMEDIATE_REPLACEMENT",
            Self::has_immediate_replacement,
            "matches a configuration with a message replaced by an immediate message",
        );
        self.add_condition(
            "UNORDERED_REPLACEMENT",
            Self::has_unordered_replacement,
            "matches a configuration with a message replaced by an unordered message",
        );
        self.add_condition(
            "ORDERED_REPLACEMENT",
            Self::has_ordered_replacement,
            "matches a configuration with a message replaced by an ordered message",
        );
        self.add_condition(
            "0MSG",
            Self::has_messages_count::<0>,
            "matches any configuration with no in-flight messages",
        );
        self.add_condition(
            "1MSG",
            Self::has_messages_count::<1>,
            "matches any configuration with a single in-flight message",
        );
        self.add_condition(
            "2MSG",
            Self::has_messages_count::<2>,
            "matches any configuration with 2 in-flight messages",
        );
        self.add_condition(
            "3MSG",
            Self::has_messages_count::<3>,
            "matches any configuration with 3 in-flight messages",
        );
        self.add_condition(
            "4MSG",
            Self::has_messages_count::<4>,
            "matches any configuration with 4 in-flight messages",
        );
        self.add_condition(
            "5MSG",
            Self::has_messages_count::<5>,
            "matches any configuration with 5 in-flight messages",
        );
        self.add_condition(
            "6MSG",
            Self::has_messages_count::<6>,
            "matches any configuration with 6 in-flight messages",
        );
        self.add_condition(
            "7MSG",
            Self::has_messages_count::<7>,
            "matches any configuration with 7 in-flight messages",
        );
        self.add_condition(
            "8MSG",
            Self::has_messages_count::<8>,
            "matches any configuration with 8 in-flight messages",
        );
        self.add_condition(
            "9MSG",
            Self::has_messages_count::<9>,
            "matches any configuration with 9 in-flight messages",
        );
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

// Conditions:

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
    fn is_init_condition(_model: &Self, configuration_id: ConfigurationId) -> bool {
        configuration_id.to_usize() == 0
    }

    // BEGIN NOT TESTED
    fn is_valid_condition(
        _model: &Self,
        configuration: &<Self as MetaModel>::Configuration,
    ) -> bool {
        configuration.invalid_id.is_valid()
    }
    // END NOT TESTED

    fn has_immediate_replacement(
        model: &Self,
        configuration: &<Self as MetaModel>::Configuration,
    ) -> bool {
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .map(|message_id| model.get_message(*message_id))
            .any(|message| message.replaced.is_some() && message.order == MessageOrder::Immediate)
    }

    fn has_unordered_replacement(
        model: &Self,
        configuration: &<Self as MetaModel>::Configuration,
    ) -> bool {
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .map(|message_id| model.get_message(*message_id))
            .any(|message| message.replaced.is_some() && message.order == MessageOrder::Unordered)
    }

    // BEGIN NOT TESTED
    fn has_ordered_replacement(
        model: &Self,
        configuration: &<Self as MetaModel>::Configuration,
    ) -> bool {
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .map(|message_id| model.get_message(*message_id))
            .any(|message| {
                message.replaced.is_some() && matches!(message.order, MessageOrder::Ordered(_))
            })
    }
    // END NOT TESTED

    fn has_messages_count<const MESSAGES_COUNT: usize>(
        _model: &Self,
        configuration: &<Self as MetaModel>::Configuration,
    ) -> bool {
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .count()
            == MESSAGES_COUNT
    }
}

// Model accessors:

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
    /// Add a named condition for defining paths through the configuration space.
    pub fn add_condition(
        &mut self,
        name: &'static str,
        condition: <Self as MetaModel>::Condition,
        help: &'static str,
    ) {
        self.conditions.insert(
            name.to_string(),
            (ConditionFunction::ByConfig(condition), help),
        );
    }

    /// Return the total number of agents.
    pub fn agents_count(&self) -> usize {
        self.agent_labels.len()
    }

    /// Return the index of the agent with the specified label.
    pub fn agent_label_index(&self, agent_label: &str) -> Option<usize> {
        self.agent_labels
            .iter()
            .position(|label| **label == agent_label)
    }

    /// Return the index of the agent instance within its type.
    pub fn agent_instance(&self, agent_index: usize) -> usize {
        agent_index - self.first_indices[agent_index]
    }

    /// Return the label (short name) of an agent.
    pub fn agent_label(&self, agent_index: usize) -> &str {
        &self.agent_labels[agent_index]
    }

    /// Return whether all the reachable configurations are valid.
    pub fn is_valid(&self) -> bool {
        self.invalids.is_empty()
    }

    /// Access a configuration by its identifier.
    pub fn get_configuration(
        &self,
        configuration_id: ConfigurationId,
    ) -> <Self as MetaModel>::Configuration {
        self.configurations.get(configuration_id)
    }

    /// Access a message by its identifier.
    pub fn get_message(&self, message_id: MessageId) -> <Self as MetaModel>::Message {
        self.messages.get(message_id)
    }
}

// Model computation:

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
    /// Compute all the configurations of the model.
    pub(crate) fn compute(&mut self) {
        assert!(self.configurations.len() == 1);

        if self.ensure_init_is_reachable {
            assert!(self.incomings.is_empty());
            self.incomings.reserve(self.outgoings.capacity());
            while self.incomings.len() < self.outgoings.len() {
                self.incomings.push(vec![]);
            }
        }

        assert!(self.pending_configuration_ids.is_empty());
        self.pending_configuration_ids
            .push_back(ConfigurationId::from_usize(0));

        while let Some(configuration_id) = self.pending_configuration_ids.pop_front() {
            self.explore_configuration(configuration_id);
        }

        if self.print_progress_every > 0 {
            eprintln!("total {} configurations", self.configurations.len());
        }
    }

    fn reach_configuration(&mut self, mut context: <Self as ModelTypes>::Context) {
        self.validate_configuration(&mut context);

        if !self.allow_invalid_configurations && context.to_configuration.invalid_id.is_valid() {
            // BEGIN NOT TESTED
            let configuration_label = self.display_configuration(&context.to_configuration);
            self.error(
                &context,
                &format!("reached an invalid configuration:{}", configuration_label),
            );
            // END NOT TESTED
        }

        let stored = self.store_configuration(context.to_configuration);
        let to_configuration_id = stored.id;

        if to_configuration_id == context.incoming.from_configuration_id {
            return;
        }

        if self.ensure_init_is_reachable {
            self.incomings[to_configuration_id.to_usize()].push(context.incoming);
        }

        let from_configuration_id = context.incoming.from_configuration_id;
        let outgoing = Outgoing {
            to_configuration_id,
            delivered_message_id: context.incoming.delivered_message_id,
        };

        self.outgoings[from_configuration_id.to_usize()].push(outgoing);
        self.transitions_count += 1;

        if stored.is_new {
            if !self.ensure_init_is_reachable {
                if let Some(ref step) = self.early_abort_step {
                    if self.step_matches_configuration(&step, to_configuration_id) {
                        eprintln!("reached {} - aborting further exploration", step.name);
                        self.pending_configuration_ids.clear();
                        self.early_abort = true;
                        return;
                    }
                }
            }

            self.pending_configuration_ids.push_back(stored.id);
        }
    }

    fn explore_configuration(&mut self, configuration_id: ConfigurationId) {
        let configuration = self.configurations.get(configuration_id);

        let mut immediate_message_index = usize::max_value();
        let mut immediate_message_id = MessageId::invalid();
        let mut immediate_message_target_index = usize::max_value();
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .enumerate()
            .for_each(|(message_index, message_id)| {
                let message = self.messages.get(*message_id);
                if message.order == MessageOrder::Immediate
                    && message.target_index < immediate_message_target_index
                {
                    immediate_message_index = message_index;
                    immediate_message_id = *message_id;
                    immediate_message_target_index = message.target_index;
                }
            });

        if immediate_message_id.is_valid() {
            self.message_event(
                configuration_id,
                configuration,
                MessageIndex::from_usize(immediate_message_index),
                immediate_message_id,
            );
        } else {
            for agent_index in 0..self.agents_count() {
                self.activity_event(configuration_id, configuration, agent_index);
            }
            configuration
                .message_ids
                .iter()
                .take_while(|message_id| message_id.is_valid())
                .enumerate()
                .for_each(|(message_index, message_id)| {
                    self.message_event(
                        configuration_id,
                        configuration,
                        MessageIndex::from_usize(message_index),
                        *message_id,
                    )
                })
        }
    }

    fn activity_event(
        &mut self,
        from_configuration_id: ConfigurationId,
        from_configuration: <Self as MetaModel>::Configuration,
        agent_index: usize,
    ) {
        let activity = {
            let agent_type = &self.agent_types[agent_index];
            let agent_instance = self.agent_instance(agent_index);
            agent_type.activity(agent_instance, &from_configuration.state_ids)
        };

        match activity {
            Activity::Passive => {}

            Activity::Process1(payload1) => {
                self.activity_message(
                    from_configuration_id,
                    from_configuration,
                    agent_index,
                    payload1,
                );
            }

            Activity::Process1Of(payloads) => {
                for payload in payloads.iter().flatten() {
                    self.activity_message(
                        from_configuration_id,
                        from_configuration,
                        agent_index,
                        *payload,
                    );
                }
            }
        }
    }

    fn activity_message(
        &mut self,
        from_configuration_id: ConfigurationId,
        from_configuration: <Self as MetaModel>::Configuration,
        agent_index: usize,
        payload: Payload,
    ) {
        let delivered_message = Message {
            order: MessageOrder::Unordered,
            source_index: usize::max_value(),
            target_index: agent_index,
            payload,
            replaced: None,
        };

        let delivered_message_id = self.store_message(delivered_message).id;

        self.message_event(
            from_configuration_id,
            from_configuration,
            MessageIndex::invalid(),
            delivered_message_id,
        );
    }

    fn message_event(
        &mut self,
        from_configuration_id: ConfigurationId,
        from_configuration: <Self as MetaModel>::Configuration,
        delivered_message_index: MessageIndex,
        delivered_message_id: MessageId,
    ) {
        let (source_index, target_index, payload, is_immediate) = {
            let message = self.messages.get(delivered_message_id);
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
            target_type.reaction(target_instance, &from_configuration.state_ids, &payload);

        let incoming = Incoming {
            from_configuration_id,
            delivered_message_id,
        };

        let mut to_configuration = from_configuration;
        if delivered_message_index.is_valid() {
            self.remove_message(
                &mut to_configuration,
                source_index,
                delivered_message_index.to_usize(),
            );
        }

        let context = Context {
            delivered_message_index,
            delivered_message_id,
            is_immediate,
            agent_index: target_index,
            agent_type: target_type,
            agent_instance: target_instance,
            agent_from_state_id: target_from_state_id,
            incoming,
            from_configuration,
            to_configuration,
        };
        self.process_reaction(context, &reaction);
    }

    fn process_reaction(
        &mut self,
        context: <Self as ModelTypes>::Context,
        reaction: &<Self as MetaModel>::Reaction,
    ) {
        match reaction // MAYBE TESTED
        {
            Reaction::Unexpected => self.error(&context, "unexpected message"), // MAYBE TESTED
            Reaction::Defer => self.defer_message(context),
            Reaction::Ignore => self.ignore_message(context),
            Reaction::Do1(action) => self.perform_action(context, action),

            // BEGIN NOT TESTED
            Reaction::Do1Of(actions) => {
                let mut did_action = false;
                for action in actions.iter().flatten() {
                    self.perform_action(context.clone(), action);
                    did_action = true;
                }
                if !did_action {
                    self.ignore_message(context);
                }
            }
            // END NOT TESTED
        }
    }

    fn perform_action(
        &mut self,
        mut context: <Self as ModelTypes>::Context,
        action: &<Self as MetaModel>::Action,
    ) {
        if self.early_abort {
            return;
        }

        match action {
            Action::Defer => self.defer_message(context),
            Action::Ignore => self.ignore_message(context), // NOT TESTED

            Action::Change(agent_to_state_id) => {
                if *agent_to_state_id == context.agent_from_state_id {
                    self.ignore_message(context); // NOT TESTED
                } else {
                    self.change_state(&mut context, *agent_to_state_id);
                    self.reach_configuration(context);
                }
            }
            Action::Send1(emit) => {
                self.collect_emit(&mut context, emit);
                self.reach_configuration(context);
            }
            Action::ChangeAndSend1(agent_to_state_id, emit) => {
                if *agent_to_state_id != context.agent_from_state_id {
                    self.change_state(&mut context, *agent_to_state_id);
                }
                self.collect_emit(&mut context, emit);
                self.reach_configuration(context);
            }

            // BEGIN NOT TESTED
            Action::Sends(emits) => {
                let mut did_emit = false;
                for emit in emits.iter().flatten() {
                    self.collect_emit(&mut context, emit);
                    did_emit = true;
                }
                if did_emit {
                    self.reach_configuration(context);
                } else {
                    self.ignore_message(context);
                }
            }
            // END NOT TESTED
            Action::ChangeAndSends(agent_to_state_id, emits) => {
                let mut did_react = false;

                if *agent_to_state_id != context.agent_from_state_id {
                    self.change_state(&mut context, *agent_to_state_id);
                    did_react = true;
                }

                for emit in emits.iter().flatten() {
                    self.collect_emit(&mut context, emit);
                    did_react = true;
                }

                if !did_react {
                    self.ignore_message(context); // NOT TESTED
                } else {
                    self.reach_configuration(context);
                }
            }
        }
    }

    fn change_state(
        &mut self,
        mut context: &mut <Self as ModelTypes>::Context,
        agent_to_state_id: StateId,
    ) {
        context
            .to_configuration
            .change_state(context.agent_index, agent_to_state_id);

        assert!(!context.to_configuration.invalid_id.is_valid());

        if let Some(reason) = context
            .agent_type
            .state_invalid_because(context.agent_instance, &context.to_configuration.state_ids)
        {
            // BEGIN NOT TESTED
            let invalid = Invalid::Agent(context.agent_index, reason);
            let invalid_id = self.store_invalid(invalid).id;
            context.to_configuration.invalid_id = invalid_id;
            // END NOT TESTED
        }
    }

    fn collect_emit(
        &mut self,
        mut context: &mut <Self as ModelTypes>::Context,
        emit: &<Self as MetaModel>::Emit,
    ) {
        match *emit {
            Emit::Immediate(payload, target_index) => {
                let message = Message {
                    order: MessageOrder::Immediate,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced: None,
                };
                self.emit_message(context, message);
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

            Emit::Ordered(payload, target_index) => {
                let message = self.ordered_message(
                    &context.to_configuration,
                    context.agent_index,
                    target_index,
                    payload,
                    None,
                );
                self.emit_message(context, message);
            }

            Emit::ImmediateReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(
                    &mut context,
                    callback,
                    MessageOrder::Immediate,
                    &payload,
                    target_index,
                );
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
                let replaced = self.replace_message(
                    &mut context,
                    callback,
                    MessageOrder::Unordered,
                    &payload,
                    target_index,
                );
                let message = Message {
                    order: MessageOrder::Unordered,
                    source_index: context.agent_index,
                    target_index,
                    payload,
                    replaced,
                };
                self.emit_message(context, message);
            }

            // BEGIN NOT TESTED
            Emit::OrderedReplacement(callback, payload, target_index) => {
                let replaced = self.replace_message(
                    &mut context,
                    callback,
                    MessageOrder::Ordered(MessageIndex::from_usize(0)),
                    &payload,
                    target_index,
                );
                let message = self.ordered_message(
                    &context.to_configuration,
                    context.agent_index,
                    target_index,
                    payload,
                    replaced,
                );
                self.emit_message(context, message);
            } // END NOT TESTED
        }
    }

    fn ordered_message(
        &mut self,
        to_configuration: &<Self as MetaModel>::Configuration,
        source_index: usize,
        target_index: usize,
        payload: Payload,
        replaced: Option<Payload>,
    ) -> <Self as MetaModel>::Message {
        let mut order = to_configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .map(|message_id| self.messages.get(*message_id))
            .filter(|message| {
                message.source_index == source_index
                    && message.target_index == target_index
                    && matches!(message.order, MessageOrder::Ordered(_))
            })
            .count();

        let message = Message {
            order: MessageOrder::Ordered(MessageIndex::from_usize(order)),
            source_index,
            target_index,
            payload,
            replaced,
        };

        let mut next_message = message;
        let mut next_message_id = self.store_message(next_message).id;
        while order > 0 {
            order -= 1;
            let entry = self.decr_order_messages.entry(next_message_id);
            let new_message = match entry // MAYBE TESTED
            {
                Entry::Occupied(_) => break,
                Entry::Vacant(entry) => {
                    next_message.order = MessageOrder::Ordered(MessageIndex::from_usize(order));
                    let stored = self.messages.store(next_message);
                    let decr_message_id = stored.id;
                    entry.insert(decr_message_id);
                    self.incr_order_messages
                        .insert(decr_message_id, next_message_id);
                    next_message_id = decr_message_id;
                    if stored.is_new {
                        Some(next_message)
                    } else {
                        None // NOT TESTED
                    }
                }
            };

            if let Some(message) = new_message {
                self.label_of_message.push(self.display_message(&message));
            }
        }

        message
    }

    fn replace_message(
        &mut self,
        context: &mut <Self as ModelTypes>::Context,
        callback: fn(Option<Payload>) -> bool,
        order: MessageOrder,
        payload: &Payload,
        target_index: usize,
    ) -> Option<Payload> {
        let mut replaced_message_index: Option<usize> = None;
        let mut replaced_message: Option<<Self as MetaModel>::Message> = None;

        let replacement_message = Message {
            order,
            source_index: context.agent_index,
            target_index,
            payload: *payload,
            replaced: None,
        };

        for (message_index, message_id) in context
            .to_configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .enumerate()
        {
            let message = self.messages.get(*message_id);
            if message.source_index == context.agent_index
                && message.target_index == target_index
                && callback(Some(message.payload))
            {
                if let Some(conflict_message) = replaced_message {
                    // BEGIN NOT TESTED
                    let conflict_label = self.display_message(&conflict_message);
                    let message_label = self.display_message(&message);
                    let replacement_label = self.display_message(&replacement_message);
                    self.error(
                        context,
                        &format!(
                            "both the message {}\n\
                             and the message {}\n\
                             can be replaced by the ambiguous replacement message {}",
                            conflict_label, message_label, replacement_label,
                        ),
                    );
                    // END NOT TESTED
                } else {
                    replaced_message_index = Some(message_index);
                    replaced_message = Some(message);
                }
            }
        }

        if let Some(message) = replaced_message {
            self.remove_message(
                &mut context.to_configuration,
                context.agent_index,
                replaced_message_index.unwrap(),
            );
            Some(message.payload)
        } else {
            if !callback(None) {
                // BEGIN NOT TESTED
                let replacement_label = self.display_message(&replacement_message);
                self.error(
                    context,
                    &format!(
                        "nothing was replaced by the required replacement message {}",
                        replacement_label
                    ),
                );
                // END NOT TESTED
            }
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
                configuration.message_ids.sort();
            }
        }
    }

    fn decr_message_id(&self, message_id: MessageId) -> Option<MessageId> {
        self.decr_order_messages.get(&message_id).copied()
    }

    fn minimal_message_id(&self, mut message_id: MessageId) -> MessageId {
        while let Some(decr_message_id) = self.decr_message_id(message_id) {
            message_id = decr_message_id;
        }
        message_id
    }

    fn incr_message_id(&self, message_id: MessageId) -> Option<MessageId> {
        self.incr_order_messages.get(&message_id).copied()
    }

    fn emit_message(
        &mut self,
        context: &mut <Self as ModelTypes>::Context,
        message: <Self as MetaModel>::Message,
    ) {
        assert!(!context.to_configuration.invalid_id.is_valid());

        for to_message_id in context
            .to_configuration
            .message_ids
            .iter()
            .take_while(|to_message_id| to_message_id.is_valid())
        {
            let to_message = self.messages.get(*to_message_id);
            if to_message.source_index == message.source_index
                && to_message.target_index == message.target_index
                && to_message.order == message.order
                && to_message.payload == message.payload
            {
                // BEGIN NOT TESTED
                let invalid =
                    <Self as MetaModel>::Invalid::Message(*to_message_id, "duplicate message");
                context.to_configuration.invalid_id = self.store_invalid(invalid).id;
                return;
                // END NOT TESTED
            }
        }

        let message_id = self.store_message(message).id;
        context
            .to_configuration
            .add_message(context.agent_index, message_id);
    }

    fn ignore_message(&mut self, context: <Self as ModelTypes>::Context) {
        self.reach_configuration(context);
    }

    fn defer_message(&mut self, context: <Self as ModelTypes>::Context) {
        if !context.delivered_message_index.is_valid() {
            // BEGIN NOT TESTED
            self.error(
                &context,
                &format!(
                    "the agent {} is deferring an activity",
                    self.agent_labels[context.agent_index]
                ),
            );
            // END NOT TESTED
        } else {
            if !context.agent_type.state_is_deferring(
                context.agent_instance,
                &context.from_configuration.state_ids,
            ) {
                // BEGIN NOT TESTED
                self.error(
                    &context,
                    &format!(
                        "the agent {} is deferring while in a non-deferring state",
                        self.agent_labels[context.agent_index]
                    ),
                );
                // END NOT TESTED
            }

            if context.is_immediate {
                // BEGIN NOT TESTED
                self.error(
                    &context,
                    &format!(
                        "the agent {} is deferring an immediate message",
                        self.agent_labels[context.agent_index]
                    ),
                );
                // END NOT TESTED
            }
        }
    }

    fn validate_configuration(&mut self, context: &mut <Self as ModelTypes>::Context) {
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
                    let invalid = <Self as MetaModel>::Invalid::Agent(
                        context.agent_index,
                        "it sends too many messages",
                    );
                    context.to_configuration.invalid_id = self.store_invalid(invalid).id;
                    // END NOT TESTED
                }
            }
        }

        if let Some(validator) = &self.validator {
            // BEGIN NOT TESTED
            for message_id in context
                .to_configuration
                .message_ids
                .iter()
                .take_while(|message_id| message_id.is_valid())
            {
                let message = self.messages.get(*message_id);
                if let Some(reason) = validator.message_invalid_because(self, &message) {
                    let invalid = <Self as MetaModel>::Invalid::Message(*message_id, reason);
                    context.to_configuration.invalid_id = self.store_invalid(invalid).id;
                    return;
                }
            }

            if let Some(reason) =
                validator.configuration_invalid_because(self, &context.to_configuration)
            {
                let invalid = <Self as MetaModel>::Invalid::Configuration(reason);
                context.to_configuration.invalid_id = self.store_invalid(invalid).id;
            }
            // END NOT TESTED
        }
    }

    // BEGIN NOT TESTED

    fn store_invalid(&mut self, invalid: <Self as MetaModel>::Invalid) -> Stored<InvalidId> {
        let stored = self.invalids.store(invalid);
        if stored.is_new {
            debug_assert!(self.label_of_invalid.len() == stored.id.to_usize());
            self.label_of_invalid.push(self.display_invalid(&invalid));
        }
        stored
    }

    // END NOT TESTED

    fn store_message(&mut self, message: <Self as MetaModel>::Message) -> Stored<MessageId> {
        let stored = self.messages.store(message);
        if stored.is_new {
            debug_assert!(self.label_of_message.len() == stored.id.to_usize());
            self.label_of_message.push(self.display_message(&message));
        }
        stored
    }

    fn store_configuration(
        &mut self,
        configuration: <Self as MetaModel>::Configuration,
    ) -> Stored<ConfigurationId> {
        let stored = self.configurations.store(configuration);

        if stored.is_new {
            if self.print_progress(stored.id.to_usize()) {
                eprintln!("computed {} configurations", stored.id.to_usize());
            }

            self.outgoings.push(vec![]);

            if self.ensure_init_is_reachable {
                self.incomings.push(vec![]);
            }
        }

        stored
    }
}

// Print output and progress:

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
    fn print_progress(&self, progress: usize) -> bool {
        self.print_progress_every == 1
            || (self.print_progress_every > 0
            // BEGIN NOT TESTED
                && progress > 0
                && progress % self.print_progress_every == 0)
        // END NOT TESTED
    }

    pub(crate) fn print_agents(&self, stdout: &mut dyn Write) {
        let mut names: Vec<String> = self
            .agent_labels
            .iter()
            .map(|agent_label| agent_label.to_string())
            .collect();
        names.sort();
        names.iter().for_each(|name| {
            writeln!(stdout, "{}", name).unwrap();
        });
    }

    pub(crate) fn print_conditions(&self, stdout: &mut dyn Write) {
        let mut names: Vec<&String> = self.conditions.iter().map(|(key, _value)| key).collect();
        names.sort();
        names
            .iter()
            .map(|name| (*name, self.conditions.get(*name).unwrap().1))
            .for_each(|(name, about)| {
                writeln!(stdout, "{}: {}", name, about).unwrap();
            });
    }

    pub(crate) fn print_stats(&self, stdout: &mut dyn Write) {
        let mut states_count = 0;
        for (agent_index, agent_type) in self.agent_types.iter().enumerate() {
            if agent_index == agent_type.first_index() {
                states_count += agent_type.states_count();
            }
        }

        writeln!(stdout, "agents: {}", self.agents_count()).unwrap();
        writeln!(stdout, "states: {}", states_count).unwrap();
        writeln!(stdout, "messages: {}", self.messages.len()).unwrap();
        writeln!(stdout, "configurations: {}", self.configurations.len()).unwrap();
        writeln!(stdout, "transitions: {}", self.transitions_count).unwrap();
    }

    pub(crate) fn print_configurations(&mut self, stdout: &mut dyn Write) {
        (0..self.configurations.len())
            .map(ConfigurationId::from_usize)
            .for_each(|configuration_id| {
                if self.print_progress(configuration_id.to_usize()) {
                    eprintln!(
                        "printed {} out of {} configurations ({}%)",
                        configuration_id.to_usize(),
                        self.configurations.len(),
                        (100 * configuration_id.to_usize()) / self.configurations.len(),
                    );
                }

                self.print_configuration_id(configuration_id, stdout);
                writeln!(stdout).unwrap();
            });
    }

    pub(crate) fn print_transitions(&self, stdout: &mut dyn Write) {
        self.outgoings
            .iter()
            .enumerate()
            .take(self.configurations.len())
            .for_each(|(from_configuration_id, outgoings)| {
                if self.print_progress(from_configuration_id) {
                    eprintln!(
                        "printed {} out of {} configurations ({}%)",
                        from_configuration_id,
                        self.configurations.len(),
                        (100 * from_configuration_id) / self.configurations.len(),
                    );
                }

                write!(stdout, "FROM:").unwrap();
                self.print_configuration_id(
                    ConfigurationId::from_usize(from_configuration_id),
                    stdout,
                );
                write!(stdout, "\n\n").unwrap();

                outgoings.iter().for_each(|outgoing| {
                    let delivered_label = self.display_message_id(outgoing.delivered_message_id);
                    write!(stdout, "BY: {}\nTO:", delivered_label).unwrap();
                    self.print_configuration_id(outgoing.to_configuration_id, stdout);
                    write!(stdout, "\n\n").unwrap();
                });
            });
    }

    pub(crate) fn print_path(
        &mut self,
        path: &[<Self as ModelTypes>::PathTransition],
        stdout: &mut dyn Write,
    ) {
        path.iter().for_each(|transition| {
            let is_first = transition.to_configuration_id == transition.from_configuration_id;
            if !is_first {
                let delivered_label = self.display_message_id(transition.delivered_message_id);
                writeln!(stdout, "BY: {}", delivered_label).unwrap();
            }

            if is_first {
                write!(stdout, "FROM").unwrap();
            } else {
                write!(stdout, "TO").unwrap();
            }

            if let Some(condition_name) = &transition.to_condition_name {
                write!(stdout, " {}", condition_name).unwrap();
            }

            write!(stdout, ":").unwrap();
            self.print_configuration_id(transition.to_configuration_id, stdout);
            write!(stdout, "\n\n").unwrap();
        });
    }
}

// Errors and related output:

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
    // BEGIN NOT TESTED
    fn error(&mut self, context: &<Self as ModelTypes>::Context, reason: &str) -> ! {
        let error_configuration_id = context.incoming.from_configuration_id;
        eprintln!(
            "ERROR: {}\n\
             when delivering the message: {}\n\
             in the configuration:{}\n\
             reached by path:\n",
            reason,
            self.display_message_id(context.delivered_message_id),
            self.display_configuration_id(error_configuration_id),
        );

        ERROR_CONFIGURATION_ID.with(|global_error_configuration_id| {
            *global_error_configuration_id.borrow_mut() = error_configuration_id.to_usize()
        });

        let is_error = move |_model: &Self, configuration_id: ConfigurationId| {
            ERROR_CONFIGURATION_ID.with(|global_error_configuration_id| {
                configuration_id
                    == ConfigurationId::from_usize(*global_error_configuration_id.borrow())
            })
        };

        let error_path_step = PathStep {
            function: ConditionFunction::ById(is_error),
            is_negated: false,
            name: "ERROR".to_string(),
        };

        self.error_path(error_path_step);
    }

    fn error_path(&mut self, error_path_step: PathStep<Self>) -> ! {
        let init_path_step = PathStep {
            function: ConditionFunction::ById(Self::is_init_condition),
            is_negated: false,
            name: "INIT".to_string(),
        };

        self.pending_configuration_ids.clear();
        let path = self.collect_path(vec![init_path_step, error_path_step]);
        self.print_path(&path, &mut stderr());

        panic!("ABORTING");
    }
    // END NOT TESTED
}

// Display data:

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
    /// Display a message by its identifier.
    pub fn display_message_id(&self, message_id: MessageId) -> &str {
        &self.label_of_message[message_id.to_usize()]
    }

    /// Display a message.
    pub fn display_message(&self, message: &<Self as MetaModel>::Message) -> String {
        let mut max_message_string_size = self.max_message_string_size.borrow_mut();
        let mut string = String::with_capacity(*max_message_string_size);

        if message.source_index != usize::max_value() {
            string.push_str(&*self.agent_labels[message.source_index]);
        } else {
            string.push_str("Activity");
        }
        string.push_str(" -> ");

        self.push_message_payload(message, false, false, &mut string);

        string.push_str(" -> ");
        string.push_str(&*self.agent_labels[message.target_index]);

        string.shrink_to_fit();
        *max_message_string_size = max(*max_message_string_size, string.len());

        string
    }

    /// Display a message in the sequence diagram.
    fn display_sequence_message(
        &self,
        message: &<Self as MetaModel>::Message,
        is_final: bool,
    ) -> String {
        let max_message_string_size = self.max_message_string_size.borrow();
        let mut string = String::with_capacity(*max_message_string_size);
        self.push_message_payload(message, true, is_final, &mut string);
        string.shrink_to_fit();
        string
    }

    /// Display a message.
    fn push_message_payload(
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
            MessageOrder::Ordered(order) => {
                if !is_sequence {
                    string.push_str(&format!("@{} ", order));
                }
            }
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
    fn display_invalid_id(&self, invalid_id: InvalidId) -> &str {
        &self.label_of_invalid[invalid_id.to_usize()]
    }

    /// Display an invalid condition.
    fn display_invalid(&self, invalid: &<Self as MetaModel>::Invalid) -> String {
        let mut max_invalid_string_size = self.max_invalid_string_size.borrow_mut();
        let mut string = String::with_capacity(*max_invalid_string_size);

        match invalid {
            Invalid::Configuration(reason) => {
                string.push_str("configuration is invalid because ");
                string.push_str(reason);
            }

            Invalid::Agent(agent_index, reason) => {
                string.push_str("agent ");
                string.push_str(&*self.agent_labels[*agent_index]);
                string.push_str(" is invalid because ");
                string.push_str(reason);
            }

            Invalid::Message(message_id, reason) => {
                string.push_str("message ");
                string.push_str(&self.display_message_id(*message_id));
                string.push_str(" is invalid because ");
                string.push_str(reason);
            }
        }

        string.shrink_to_fit();
        *max_invalid_string_size = max(string.len(), *max_invalid_string_size);

        string
    }

    /// Display a configuration by its identifier.
    pub fn display_configuration_id(&self, configuration_id: ConfigurationId) -> String {
        let configuration = self.configurations.get(configuration_id);
        self.display_configuration(&configuration)
    }

    /// Display a configuration.
    fn display_configuration(&self, configuration: &<Self as MetaModel>::Configuration) -> String {
        let mut max_configuration_string_size = self.max_configuration_string_size.borrow_mut();
        let mut buffer = Vec::<u8>::with_capacity(*max_configuration_string_size);
        self.print_configuration(configuration, &mut buffer);
        let mut string = String::from_utf8(buffer).unwrap();
        string.shrink_to_fit();
        *max_configuration_string_size = max(string.len(), *max_configuration_string_size);
        string
    }

    // END NOT TESTED

    /// Display a configuration.
    fn print_configuration_id(&self, configuration_id: ConfigurationId, stdout: &mut dyn Write) {
        let configuration = self.configurations.get(configuration_id);
        self.print_configuration(&configuration, stdout)
    }

    /// Display a configuration.
    fn print_configuration(
        &self,
        configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) {
        let mut hash: u64 = 0;

        let mut prefix = "\n- ";
        (0..self.agents_count()).for_each(|agent_index| {
            let agent_type = &self.agent_types[agent_index];
            let agent_label = &self.agent_labels[agent_index];
            let agent_state_id = configuration.state_ids[agent_index];
            let agent_state = agent_type.display_state(agent_state_id);

            hash ^= calculate_strings_hash(agent_label, &*agent_state);

            if !agent_state.is_empty() {
                write!(stdout, "{}{}:{}", prefix, agent_label, agent_state).unwrap();
                prefix = "\n& ";
            }
        });

        prefix = "\n| ";
        configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .for_each(|message_id| {
                let message_label = self.display_message_id(*message_id);
                hash ^= calculate_string_hash(&message_label);
                write!(stdout, "{}{}", prefix, message_label).unwrap();
                prefix = "\n& ";
            });

        if configuration.invalid_id.is_valid() {
            // BEGIN NOT TESTED
            let invalid_label = self.display_invalid_id(configuration.invalid_id);
            hash ^= calculate_string_hash(&invalid_label);
            write!(stdout, "\n! {}", invalid_label).unwrap();
            // END NOT TESTED
        }

        write!(stdout, "\n# {:016X}", hash).unwrap();
    }
}

// INIT reachability:

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
    pub(crate) fn assert_init_is_reachable(&mut self) {
        assert!(self.reachable_configurations_count == 0);
        assert!(self.reachable_configurations_mask.is_empty());

        self.reachable_configurations_mask
            .resize(self.configurations.len(), false);

        assert!(self.pending_configuration_ids.is_empty());
        self.pending_configuration_ids
            .push_back(ConfigurationId::from_usize(0));
        while let Some(configuration_id) = self.pending_configuration_ids.pop_front() {
            self.reachable_configuration(configuration_id);
        }

        REACHABLE_CONFIGURATIONS_MASK.with(|reachable_configurations_mask| {
            swap(
                &mut *reachable_configurations_mask.borrow_mut(),
                &mut self.reachable_configurations_mask,
            )
        });

        let unreachable_count = self.configurations.len() - self.reachable_configurations_count;
        if unreachable_count > 0 {
            // BEGIN NOT TESTED
            eprintln!(
                "ERROR: there is no path back to initial state from {} configurations\n",
                unreachable_count
            );

            let is_reachable = move |_model: &Self, configuration_id: ConfigurationId| {
                REACHABLE_CONFIGURATIONS_MASK.with(|reachable_configurations_mask| {
                    reachable_configurations_mask.borrow()[configuration_id.to_usize()]
                })
            };
            let error_path_step = PathStep {
                function: ConditionFunction::ById(is_reachable),
                is_negated: true,
                name: "DEADEND".to_string(),
            };

            self.error_path(error_path_step);
            // END NOT TESTED
        }
    }

    fn reachable_configuration(&mut self, configuration_id: ConfigurationId) {
        if self.reachable_configurations_mask[configuration_id.to_usize()] {
            return;
        }

        {
            let reachable_configuration =
                &mut self.reachable_configurations_mask[configuration_id.to_usize()];
            if *reachable_configuration {
                return;
            }
            *reachable_configuration = true;

            self.reachable_configurations_count += 1;

            if self.print_progress(self.reachable_configurations_count) {
                eprintln!(
                    "reached {} out of {} configurations ({}%)",
                    self.reachable_configurations_count,
                    self.configurations.len(),
                    (100 * self.reachable_configurations_count) / self.configurations.len(),
                );
            }
        }

        for next_configuration_id in self.incomings[configuration_id.to_usize()]
            .iter()
            .map(|incoming| incoming.from_configuration_id)
        {
            self.pending_configuration_ids
                .push_back(next_configuration_id);
        }
    }
}

// Path computation:

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
    fn step_matches_configuration(
        &self,
        step: &<Self as ModelTypes>::PathStep,
        configuration_id: ConfigurationId,
    ) -> bool {
        let mut is_match = match step.function // MAYBE TESTED
        {
            ConditionFunction::ById(by_id) => by_id(self, configuration_id),
            ConditionFunction::ByConfig(by_config) => {
                by_config(self, &self.get_configuration(configuration_id))
            }
        };

        if step.is_negated {
            is_match = !is_match
        }

        is_match
    }

    fn find_closest_configuration_id(
        &mut self,
        from_configuration_id: ConfigurationId,
        from_name: &str,
        to_step: &<Self as ModelTypes>::PathStep,
        prev_configuration_ids: &mut [ConfigurationId],
    ) -> ConfigurationId {
        assert!(self.pending_configuration_ids.is_empty());
        self.pending_configuration_ids
            .push_back(from_configuration_id);

        prev_configuration_ids.fill(ConfigurationId::invalid());

        while let Some(next_configuration_id) = self.pending_configuration_ids.pop_front() {
            for outgoing in self.outgoings[next_configuration_id.to_usize()].iter() {
                let to_configuration_id = outgoing.to_configuration_id;
                if prev_configuration_ids[to_configuration_id.to_usize()].is_valid() {
                    continue;
                }
                prev_configuration_ids[to_configuration_id.to_usize()] = next_configuration_id;

                let mut is_condition = false;

                if next_configuration_id != from_configuration_id
                    || to_configuration_id != from_configuration_id
                {
                    is_condition = self.step_matches_configuration(to_step, to_configuration_id);
                }

                if is_condition {
                    self.pending_configuration_ids.clear();
                    return to_configuration_id;
                }

                self.pending_configuration_ids
                    .push_back(to_configuration_id);
            }
        }

        // BEGIN NOT TESTED
        panic!(
            "could not find a path from the condition {} to the condition {}\n\
            starting from the configuration:{}",
            from_name,
            to_step.name,
            self.display_configuration_id(from_configuration_id)
        );
        // END NOT TESTED
    }

    pub(crate) fn collect_steps(
        &mut self,
        subcommand_name: &str,
        matches: &ArgMatches,
    ) -> Vec<<Self as ModelTypes>::PathStep> {
        let steps: Vec<<Self as ModelTypes>::PathStep> = matches
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
                let (key, is_negated) = match name.strip_prefix("!") {
                    None => (name, false),
                    Some(suffix) => (suffix, true),
                };
                if let Some(condition) = self.conditions.get(&key.to_string()) {
                    PathStep {
                        function: condition.0.clone(),
                        is_negated,
                        name: name.to_string(),
                    }
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

        self.early_abort = false;
        if steps.len() == 2
            && self.step_matches_configuration(&steps[0], ConfigurationId::from_usize(0))
        {
            self.early_abort_step = Some(steps[1].clone());
        } else {
            self.early_abort_step = None;
        }

        steps
    }

    pub(crate) fn collect_path(
        &mut self,
        mut steps: Vec<<Self as ModelTypes>::PathStep>,
    ) -> Vec<<Self as ModelTypes>::PathTransition> {
        let mut prev_configuration_ids =
            vec![ConfigurationId::invalid(); self.configurations.len()];

        let initial_configuration_id = ConfigurationId::from_usize(0);

        let start_at_init = self.step_matches_configuration(&steps[0], initial_configuration_id);

        let mut current_configuration_id = initial_configuration_id;
        let mut current_name = steps[0].name.to_string();

        if start_at_init {
            steps.remove(0);
        } else {
            current_configuration_id = self.find_closest_configuration_id(
                initial_configuration_id,
                "INIT",
                &steps[0],
                &mut prev_configuration_ids,
            );
        }

        let mut path = vec![PathTransition {
            from_configuration_id: current_configuration_id,
            delivered_message_id: MessageId::invalid(),
            agent_index: usize::max_value(),
            to_configuration_id: current_configuration_id,
            to_condition_name: Some(current_name.to_string()),
        }];

        steps.iter().for_each(|step| {
            let next_configuration_id = self.find_closest_configuration_id(
                current_configuration_id,
                &current_name,
                step,
                &mut prev_configuration_ids,
            );
            self.collect_path_step(
                current_configuration_id,
                next_configuration_id,
                Some(&step.name),
                &prev_configuration_ids,
                &mut path,
            );
            current_configuration_id = next_configuration_id;
            current_name = step.name.to_string();
        });

        path
    }

    fn collect_path_step(
        &mut self,
        from_configuration_id: ConfigurationId,
        to_configuration_id: ConfigurationId,
        to_name: Option<&str>,
        prev_configuration_ids: &[ConfigurationId],
        path: &mut Vec<<Self as ModelTypes>::PathTransition>,
    ) {
        let mut configuration_ids: Vec<ConfigurationId> = vec![to_configuration_id];

        let mut prev_configuration_id = to_configuration_id;
        loop {
            prev_configuration_id = prev_configuration_ids[prev_configuration_id.to_usize()];
            assert!(prev_configuration_id.is_valid());
            configuration_ids.push(prev_configuration_id);
            if prev_configuration_id == from_configuration_id {
                break;
            }
        }

        configuration_ids.reverse();

        for (prev_configuration_id, next_configuration_id) in configuration_ids
            [..configuration_ids.len() - 1]
            .iter()
            .zip(configuration_ids[1..].iter())
        {
            let next_name = if *next_configuration_id == to_configuration_id {
                to_name
            } else {
                None
            };

            self.collect_small_path_step(
                *prev_configuration_id,
                *next_configuration_id,
                next_name,
                path,
            );
        }
    }

    fn collect_small_path_step(
        &mut self,
        from_configuration_id: ConfigurationId,
        to_configuration_id: ConfigurationId,
        to_name: Option<&str>,
        path: &mut Vec<<Self as ModelTypes>::PathTransition>,
    ) {
        let from_outgoings = &self.outgoings[from_configuration_id.to_usize()];
        let outgoing_index = from_outgoings
            .iter()
            .position(|outgoing| outgoing.to_configuration_id == to_configuration_id)
            .unwrap();
        let outgoing = from_outgoings[outgoing_index];

        let agent_index = self
            .messages
            .get(outgoing.delivered_message_id)
            .target_index;
        let delivered_message_id = outgoing.delivered_message_id;

        path.push(PathTransition {
            from_configuration_id,
            delivered_message_id,
            agent_index,
            to_configuration_id,
            to_condition_name: to_name.map(str::to_string),
        });
    }
}

// Condensing diagrams:

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
    fn compute_terse(&mut self, condense: &Condense) {
        assert!(self.terse_of_message_id.is_empty());
        assert!(self.message_of_terse_id.is_empty());

        if condense.names_only {
            for (agent_index, agent_type) in self.agent_types.iter().enumerate() {
                if agent_index == agent_type.first_index() {
                    agent_type.compute_terse();
                }
            }
        }

        let mut seen_messages: HashMap<TerseMessage, usize> = HashMap::new();
        seen_messages.reserve(self.messages.len());
        self.terse_of_message_id.reserve(self.messages.len());
        self.message_of_terse_id.reserve(self.messages.len());

        for message_id in 0..self.messages.len() {
            let message = self.messages.get(MessageId::from_usize(message_id));

            let source_index =
                if message.source_index != usize::max_value() && condense.merge_instances {
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

            let order = if condense.final_replaced {
                MessageOrder::Unordered
            } else {
                match message.order {
                    MessageOrder::Ordered(_) => MessageOrder::Ordered(MessageIndex::invalid()),
                    order => order,
                }
            };

            let terse_message = TerseMessage {
                order,
                source_index,
                target_index,
                payload,
                replaced,
            };

            let terse_id = *seen_messages.entry(terse_message).or_insert_with(|| {
                let next_terse_id = self.message_of_terse_id.len();
                self.message_of_terse_id
                    .push(MessageId::from_usize(message_id));
                next_terse_id
            });

            self.terse_of_message_id
                .push(MessageId::from_usize(terse_id));
        }

        self.message_of_terse_id.shrink_to_fit();
    }
}

// States diagram:

// What to print.
//
// It is "most annoying" that when GraphViz sees an edge it creates the connected nodes there and
// then, even if they are explictly created inside a cluster later on. To combat this we need to
// first create all the nodes in their clusters and only then generate all the edges. This is rather
// inconvenient and this code is pretty fast so we just run the same code twice, once printing only
// the structure and once printing only the edges. YUCK.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum PrintFilter {
    // Only print the structure of the diagram.
    Structure,
    // Only print the edges.
    Edges,
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
    pub(crate) fn print_states_diagram(
        &mut self,
        condense: &Condense,
        agent_index: usize,
        stdout: &mut dyn Write,
    ) {
        writeln!(stdout, "digraph {{").unwrap();
        writeln!(stdout, "color=white;").unwrap();
        writeln!(stdout, "concentrate=true;").unwrap();
        writeln!(stdout, "graph [ fontname=\"sans-serif\" ];").unwrap();
        writeln!(stdout, "node [ fontname=\"sans-serif\" ];").unwrap();
        writeln!(stdout, "edge [ fontname=\"sans-serif\" ];").unwrap();

        self.compute_terse(condense);

        let state_transitions = self.collect_agent_state_transitions(condense, agent_index);
        let mut contexts: Vec<&<Self as ModelTypes>::AgentStateTransitionContext> =
            state_transitions.keys().collect();
        contexts.sort();

        let mut emitted_states = vec![false; self.agent_types[agent_index].states_count()];
        let mut is_first = true;

        let mut state_transition_index: usize = 0;
        for context in contexts.iter() {
            if !emitted_states[context.from_state_id.to_usize()] {
                if !is_first {
                    writeln!(stdout, "}}").unwrap();
                }
                is_first = false;
                writeln!(
                    stdout,
                    "subgraph cluster_{}_{} {{",
                    context.from_state_id.to_usize(),
                    context.from_is_deferring
                )
                .unwrap();

                self.print_agent_state_node(
                    condense,
                    agent_index,
                    context.from_state_id,
                    context.from_is_deferring,
                    stdout,
                );

                emitted_states[context.from_state_id.to_usize()] = true;
            }
            self.print_context_states_diagram(
                condense,
                context,
                &state_transitions,
                &mut state_transition_index,
                PrintFilter::Structure,
                stdout,
            );
        }

        if !is_first {
            writeln!(stdout, "}}").unwrap();
        }

        state_transition_index = 0;
        for context in contexts.iter() {
            self.print_context_states_diagram(
                condense,
                context,
                &state_transitions,
                &mut state_transition_index,
                PrintFilter::Edges,
                stdout,
            );
        }

        writeln!(stdout, "}}").unwrap();
    }

    fn print_context_states_diagram(
        &mut self,
        condense: &Condense,
        context: &<Self as ModelTypes>::AgentStateTransitionContext,
        state_transitions: &<Self as ModelTypes>::AgentStateTransitions,
        state_transition_index: &mut usize,
        print_filter: PrintFilter,
        stdout: &mut dyn Write,
    ) {
        let related_state_transitions = &state_transitions[context];

        let mut sent_keys: Vec<&Vec<MessageId>> = related_state_transitions.keys().collect();
        sent_keys.sort();

        let mut sent_by_delivered = <Self as ModelTypes>::AgentStateSentByDelivered::new();

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
            let delivered_sent_message_ids = sent_by_delivered.get(delivered_message_ids).unwrap();

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
                let other_delivered_message_ids: &Vec<MessageId> = other_delivered_message_ids_key;
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
                    intersecting_delivered_message_ids.push(other_delivered_message_ids.to_vec());
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
                    context,
                    delivered_message_ids,
                    *state_transition_index,
                    false,
                    stdout,
                    print_filter,
                );

                if print_filter == PrintFilter::Edges {
                    self.print_agent_transition_sent_edges(
                        condense,
                        context.from_state_id,
                        context.from_is_deferring,
                        &sent_message_ids,
                        *state_transition_index,
                        context.to_state_id,
                        context.to_is_deferring,
                        context.is_constraint,
                        None,
                        stdout,
                    );
                }

                if print_filter == PrintFilter::Structure {
                    writeln!(stdout, "}}").unwrap();
                }
                *state_transition_index += 1;
            }
        }

        for delivered_message_ids_key in distinct_delivered_message_ids.iter() {
            let mut delivered_sent_keys: Vec<&Vec<MessageId>> =
                related_state_transitions.keys().collect();
            let delivered_message_ids: &Vec<MessageId> = delivered_message_ids_key;
            delivered_sent_keys.sort();

            self.print_agent_transition_cluster(
                condense,
                context,
                delivered_message_ids,
                *state_transition_index,
                true,
                stdout,
                print_filter,
            );

            for (alternative_index, sent_message_ids_key) in delivered_sent_keys.iter().enumerate()
            {
                let sent_message_ids: &Vec<MessageId> = sent_message_ids_key;

                if print_filter == PrintFilter::Edges {
                    self.print_agent_transition_sent_edges(
                        condense,
                        context.from_state_id,
                        context.from_is_deferring,
                        &sent_message_ids,
                        *state_transition_index,
                        context.to_state_id,
                        context.to_is_deferring,
                        context.is_constraint,
                        Some(alternative_index),
                        stdout,
                    );
                }
            }

            if print_filter == PrintFilter::Structure {
                writeln!(stdout, "}}").unwrap();
            }
            *state_transition_index += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn print_agent_transition_cluster(
        &self,
        condense: &Condense,
        context: &<Self as ModelTypes>::AgentStateTransitionContext,
        delivered_message_ids: &[MessageId],
        state_transition_index: usize,
        has_alternatives: bool,
        stdout: &mut dyn Write,
        print_filter: PrintFilter,
    ) {
        match print_filter {
            PrintFilter::Structure => {
                writeln!(stdout, "subgraph cluster_{} {{", state_transition_index).unwrap();
                Self::print_state_transition_node(state_transition_index, has_alternatives, stdout);
            }

            PrintFilter::Edges => {
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
                    context.is_constraint,
                    stdout,
                );
            }
        }

        for delivered_message_id in delivered_message_ids.iter() {
            match print_filter {
                PrintFilter::Structure => {
                    self.print_message_node(
                        condense,
                        context.from_state_id,
                        context.from_is_deferring,
                        state_transition_index,
                        None,
                        Some(*delivered_message_id),
                        "D",
                        stdout,
                    );
                }

                PrintFilter::Edges => {
                    self.print_message_transition_edge(
                        *delivered_message_id,
                        state_transition_index,
                        stdout,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn print_agent_transition_sent_edges(
        &self,
        condense: &Condense,
        from_state_id: StateId,
        from_is_deferring: bool,
        sent_message_ids: &[MessageId],
        state_transition_index: usize,
        to_state_id: StateId,
        to_is_deferring: bool,
        is_constraint: bool,
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
            // BEGIN NOT TESTED
            self.print_message_node(
                condense,
                from_state_id,
                from_is_deferring,
                state_transition_index,
                None,
                None,
                "S",
                stdout,
            );
            self.print_transition_message_edge(state_transition_index, None, None, stdout);
            // END NOT TESTED
        }

        if sent_message_ids.len() < 2 {
            alternative_index = None;
        }

        for sent_message_id in sent_message_ids.iter() {
            self.print_message_node(
                condense,
                from_state_id,
                from_is_deferring,
                state_transition_index,
                alternative_index,
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

            if is_constraint {
                writeln!(
                    stdout,
                    "S_{}_{}_{} -> A_{}_{} [ style=invis ];",
                    state_transition_index,
                    alternative_index.unwrap_or(usize::max_value()),
                    sent_message_id.to_usize(),
                    to_state_id.to_usize(),
                    to_is_deferring,
                )
                .unwrap();
            }
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
        let color = if is_deferring {
            "Plum"
        } else {
            "PaleTurquoise"
        };
        let state = if condense.names_only {
            self.agent_types[agent_index].display_terse(state_id)
        } else {
            self.agent_types[agent_index]
                .display_state(state_id)
                .to_string()
        };

        writeln!(
            stdout,
            "A_{}_{} [ label=\"{}\", shape={}, style=filled, color={} ];",
            state_id.to_usize(),
            is_deferring,
            state,
            shape,
            color
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
                 width=0.2, height=0.2, style=filled, color=black ];",
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

    #[allow(clippy::too_many_arguments)]
    fn print_message_node(
        &self,
        condense: &Condense,
        from_state_id: StateId,
        from_is_deferring: bool,
        state_transition_index: usize,
        alternative_index: Option<usize>,
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

            if prefix == "D" {
                writeln!(
                    stdout,
                    "A_{}_{} -> {}_{}_{} [ style=invis ];",
                    from_state_id.to_usize(),
                    from_is_deferring,
                    prefix,
                    state_transition_index,
                    alternative_index.unwrap_or(usize::max_value()),
                )
                .unwrap();
            }

            return;
            // END NOT TESTED
        }

        let message_id = message_id.unwrap();
        write!(
            stdout,
            "{}_{}_{}_{} [ label=\"",
            prefix,
            state_transition_index,
            alternative_index.unwrap_or(usize::max_value()),
            message_id.to_usize()
        )
        .unwrap();

        let message = self
            .messages
            .get(self.message_of_terse_id[message_id.to_usize()]);

        let mut color = match message.order {
            MessageOrder::Unordered => "PaleGreen",
            MessageOrder::Ordered(_) => "Bisque",
            MessageOrder::Immediate => "LightSalmon",
        };

        if prefix == "D" {
            let source = if message.source_index == usize::max_value() {
                color = "SandyBrown";
                "Activity".to_string()
            } else if condense.merge_instances {
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

        if prefix == "S" {
            let target = if condense.merge_instances {
                self.agent_types[message.target_index].name()
            } else {
                self.agent_labels[message.target_index].to_string()
            };
            write!(stdout, "\\n{} {}", RIGHT_ARROW, target).unwrap();
        }

        writeln!(stdout, "\", shape=box, style=filled, color={} ];", color).unwrap();

        if prefix == "D" {
            writeln!(
                stdout,
                "A_{}_{} -> {}_{}_{}_{} [ style=invis ];",
                from_state_id.to_usize(),
                from_is_deferring,
                prefix,
                state_transition_index,
                alternative_index.unwrap_or(usize::max_value()),
                message_id.to_usize()
            )
            .unwrap();
        }
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
        is_constraint: bool,
        stdout: &mut dyn Write,
    ) {
        writeln!(
            stdout,
            "T_{}_{} -> A_{}_{} [ constraint={} ];",
            from_state_transition_index,
            usize::max_value(),
            to_state_id.to_usize(),
            to_is_deferring,
            is_constraint,
        )
        .unwrap();
    }

    fn print_message_transition_edge(
        &self,
        from_message_id: MessageId,
        to_state_transition_index: usize,
        stdout: &mut dyn Write,
    ) {
        writeln!(
            stdout,
            "D_{}_{}_{} -> T_{}_{} [ style=dashed ];",
            to_state_transition_index,
            usize::max_value(),
            from_message_id.to_usize(),
            to_state_transition_index,
            usize::max_value()
        )
        .unwrap();
    }

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
        writeln!(
            stdout,
            "T_{}_{} -> S_{}_{}_{} [ style=dashed ];",
            from_state_transition_index,
            from_alternative_index.unwrap_or(usize::max_value()),
            from_state_transition_index,
            from_alternative_index.unwrap_or(usize::max_value()),
            to_message_id.to_usize(),
        )
        .unwrap();
    }
}

// Sequence diagrams:

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
    pub(crate) fn collect_sequence_steps(
        &mut self,
        path: &[<Self as ModelTypes>::PathTransition],
    ) -> Vec<<Self as ModelTypes>::SequenceStep> {
        let mut sequence_steps: Vec<<Self as ModelTypes>::SequenceStep> = vec![];
        for path_transition in path {
            let agent_index = path_transition.agent_index;
            let from_configuration_id = path_transition.from_configuration_id;
            let from_configuration = self.configurations.get(from_configuration_id);
            let to_configuration_id = path_transition.to_configuration_id;
            let to_configuration = self.configurations.get(to_configuration_id);
            let did_change_state = to_configuration.state_ids[agent_index]
                != from_configuration.state_ids[agent_index];
            let from_outgoings = &self.outgoings[from_configuration_id.to_usize()];
            let outgoing_index = from_outgoings
                .iter()
                .position(|outgoing| outgoing.to_configuration_id == to_configuration_id)
                .unwrap();
            let outgoing = from_outgoings[outgoing_index];
            let delivered_message = self.messages.get(path_transition.delivered_message_id);
            let is_activity = delivered_message.source_index == usize::max_value();

            sequence_steps.push(SequenceStep::Received {
                source_index: delivered_message.source_index,
                target_index: delivered_message.target_index,
                did_change_state,
                is_activity,
                message_id: self.minimal_message_id(path_transition.delivered_message_id),
            });

            to_configuration
                .message_ids
                .iter()
                .take_while(|to_message_id| to_message_id.is_valid())
                .filter(|to_message_id| {
                    !self.to_message_kept_in_transition(
                        **to_message_id,
                        &from_configuration,
                        outgoing.delivered_message_id,
                    )
                })
                .for_each(|to_message_id| {
                    let to_message = self.messages.get(*to_message_id);
                    assert!(agent_index == to_message.source_index);

                    let replaced = to_message.replaced.map(|replaced_payload| {
                        *from_configuration
                            .message_ids
                            .iter()
                            .take_while(|from_message_id| from_message_id.is_valid())
                            .find(|from_message_id| {
                                let from_message = self.messages.get(**from_message_id);
                                from_message.source_index == to_message.source_index
                                    && from_message.target_index == to_message.target_index
                                    && from_message.payload == replaced_payload
                            })
                            .unwrap()
                    });
                    sequence_steps.push(SequenceStep::Emitted {
                        source_index: to_message.source_index,
                        target_index: to_message.target_index,
                        message_id: self.minimal_message_id(*to_message_id),
                        replaced,
                        is_immediate: to_message.order == MessageOrder::Immediate,
                    });
                });

            if did_change_state {
                let agent_type = &self.agent_types[agent_index];
                let agent_instance = self.agent_instance(agent_index);
                let state_id = to_configuration.state_ids[agent_index];
                let is_deferring =
                    agent_type.state_is_deferring(agent_instance, &to_configuration.state_ids);
                sequence_steps.push(SequenceStep::NewState {
                    agent_index,
                    state_id,
                    is_deferring,
                });
            }
        }

        sequence_steps
    }

    fn patch_sequence_steps(&self, sequence_steps: &mut [<Self as ModelTypes>::SequenceStep]) {
        self.eprint_sequence_steps("sequence_steps", sequence_steps);
        let mut did_merge_messages = true;
        while did_merge_messages {
            self.move_steps_up(sequence_steps);
            self.eprint_sequence_steps("moved_up", sequence_steps);
            did_merge_messages = self.merge_message_steps(sequence_steps);
            self.eprint_sequence_steps("merged_messages", sequence_steps);
        }
        self.merge_states(sequence_steps);
        self.eprint_sequence_steps("merged_states", sequence_steps);
    }

    fn eprint_sequence_steps(
        &self,
        label: &str,
        sequence_steps: &[<Self as ModelTypes>::SequenceStep],
    ) {
        if false {
            // BEGIN NOT TESTED
            eprintln!("{}:", label);
            for sequence_step in sequence_steps {
                self.eprint_sequence_step(sequence_step);
            }
            eprintln!("...");
            // END NOT TESTED
        }
    }

    // BEGIN NOT TESTED
    fn eprint_sequence_step(&self, sequence_step: &<Self as ModelTypes>::SequenceStep) {
        match sequence_step {
            SequenceStep::NoStep => {
                eprintln!("- NoStep");
            }

            SequenceStep::Received {
                message_id,
                is_activity,
                did_change_state,
                ..
            } => {
                let activity = if *is_activity { "activity" } else { "message" };
                let changed_state = if *did_change_state {
                    "changing"
                } else {
                    "unchanging"
                };
                eprintln!(
                    "- Received {:?} {} {} {}",
                    message_id,
                    changed_state,
                    activity,
                    self.display_message_id(*message_id)
                );
            }

            SequenceStep::Emitted {
                message_id,
                is_immediate,
                ..
            } => {
                let immediate = if *is_immediate { "immediate" } else { "fabric" };
                eprintln!(
                    "- Emitted {:?} {} {}",
                    message_id,
                    immediate,
                    self.display_message_id(*message_id)
                );
            }

            SequenceStep::Passed {
                message_id,
                target_did_change_state,
                ..
            } => {
                let changed_state = if *target_did_change_state {
                    "changing"
                } else {
                    "unchanging"
                };
                eprintln!(
                    "- Passed {:?} {} {}",
                    message_id,
                    changed_state,
                    self.display_message_id(*message_id)
                );
            }

            SequenceStep::NewState {
                agent_index,
                state_id,
                ..
            } => {
                let agent = &self.agent_labels[*agent_index];
                let agent_type = &self.agent_types[*agent_index];
                let state = agent_type.display_state(*state_id);
                eprintln!("- NewState {} = {}", agent, state);
            }

            SequenceStep::NewStates(new_states) => {
                eprintln!("- NewStates:");
                for new_state in new_states {
                    let agent = &self.agent_labels[new_state.agent_index];
                    let agent_type = &self.agent_types[new_state.agent_index];
                    let state = agent_type.display_state(new_state.state_id);
                    eprintln!("  - NewState {} = {}", agent, state);
                }
            }
        }
    }
    // END NOT TESTED

    fn move_steps_up(&self, mut sequence_steps: &mut [<Self as ModelTypes>::SequenceStep]) {
        let mut first_index: usize = 0;
        let mut main_agent_index =
            if let SequenceStep::Received { target_index, .. } = sequence_steps[first_index] {
                target_index
            } else {
                unreachable!();
            };

        while let Some((next_index, next_agent_index)) =
            self.move_remaining_steps_up(first_index, &mut sequence_steps, main_agent_index)
        {
            first_index = next_index;
            main_agent_index = next_agent_index;
        }
    }

    fn move_remaining_steps_up(
        &self,
        mut first_index: usize,
        sequence_steps: &mut [<Self as ModelTypes>::SequenceStep],
        main_agent_index: usize,
    ) -> Option<(usize, usize)> {
        let mut related_steps =
            self.compute_related_steps(first_index, sequence_steps, main_agent_index);

        let limit_index = first_index + 1;
        first_index += 1;
        while first_index < sequence_steps.len() - 1 {
            let second_index = first_index + 1;

            if !related_steps[second_index] {
                first_index = second_index;
                continue;
            }

            if !Self::can_swap(&sequence_steps[first_index], &sequence_steps[second_index]) {
                first_index = second_index;
                continue;
            }

            if !self.should_swap(first_index, second_index, &sequence_steps, &related_steps) {
                first_index = second_index;
                continue;
            }

            sequence_steps.swap(first_index, second_index);
            related_steps.swap(first_index, second_index);
            if first_index > limit_index {
                first_index -= 1;
            }
        }

        self.find_next_received_step(
            limit_index,
            sequence_steps,
            &related_steps,
            main_agent_index,
        )
    }

    fn compute_related_steps(
        &self,
        first_index: usize,
        sequence_steps: &mut [<Self as ModelTypes>::SequenceStep],
        main_agent_index: usize,
    ) -> Vec<bool> {
        let mut collect_emitted_messages = true;
        let mut related_messages: Vec<MessageId> = vec![];
        let mut affected_agents: [usize; MAX_AGENTS] = [0; MAX_AGENTS];
        let mut next_unrelated_message_id: Option<MessageId> = None;
        let mut related_steps = vec![false; sequence_steps.len()];

        affected_agents[main_agent_index] = 1;
        related_steps[first_index] = true;
        for next_index in (first_index + 1)..sequence_steps.len() {
            related_steps[next_index] = match &sequence_steps[next_index] {
                SequenceStep::Emitted {
                    message_id,
                    source_index,
                    ..
                } if collect_emitted_messages && *source_index == main_agent_index => {
                    related_messages.push(*message_id);
                    true
                }

                SequenceStep::Passed {
                    message_id,
                    source_index,
                    target_index,
                    ..
                } => {
                    if *target_index == main_agent_index {
                        collect_emitted_messages = false;
                        affected_agents[main_agent_index] = 0;
                    }

                    if collect_emitted_messages && *source_index == main_agent_index {
                        related_messages.push(*message_id);
                        affected_agents[*target_index] += 1;
                        true
                    } else {
                        if next_unrelated_message_id.is_none() {
                            next_unrelated_message_id = Some(*message_id);
                        }
                        false
                    }
                }

                SequenceStep::Received {
                    message_id,
                    target_index,
                    ..
                } => {
                    if *target_index == main_agent_index {
                        collect_emitted_messages = false;
                        affected_agents[main_agent_index] = 0;
                    }

                    if let Some(position) = related_messages
                        .iter()
                        .position(|related_message_id| *related_message_id == *message_id)
                    {
                        related_messages.remove(position);
                        affected_agents[*target_index] += 1;
                        true
                    } else {
                        if next_unrelated_message_id.is_none() {
                            next_unrelated_message_id = Some(*message_id);
                        }
                        false
                    }
                }

                SequenceStep::NewState { agent_index, .. } if affected_agents[*agent_index] > 0 => {
                    true
                }

                SequenceStep::NewStates(_) => {
                    unreachable!();
                }

                _ => false,
            };
        }

        related_steps
    }

    fn find_next_received_step(
        &self,
        mut next_index: usize,
        sequence_steps: &mut [<Self as ModelTypes>::SequenceStep],
        related_steps: &[bool],
        main_agent_index: usize,
    ) -> Option<(usize, usize)> {
        let mut next_agent_index: Option<usize> = None;
        let mut next_received_index: Option<usize> = None;
        while next_agent_index.is_none() || next_received_index.is_none() {
            if next_received_index.is_none() && !related_steps[next_index] {
                next_received_index = Some(next_index);
            }
            match sequence_steps[next_index] {
                SequenceStep::Received { target_index, .. }
                    if next_agent_index.is_none() && target_index != main_agent_index =>
                {
                    next_agent_index = Some(target_index);
                }

                SequenceStep::Passed { target_index, .. }
                    if next_agent_index.is_none() && target_index != main_agent_index =>
                {
                    next_agent_index = Some(target_index);
                }

                _ => {
                    next_index += 1;
                    if next_index == sequence_steps.len() {
                        return None;
                    }
                }
            }
        }

        Some((next_received_index.unwrap(), next_agent_index.unwrap()))
    }

    fn should_swap(
        &self,
        first_index: usize,
        second_index: usize,
        sequence_steps: &[<Self as ModelTypes>::SequenceStep],
        related_steps: &[bool],
    ) -> bool {
        if !related_steps[first_index] {
            return true;
        }

        if Self::swap_order(&sequence_steps[first_index])
            > Self::swap_order(&sequence_steps[second_index])
        {
            return true;
        }

        false
    }

    fn swap_order(sequence_step: &<Self as ModelTypes>::SequenceStep) -> usize {
        match sequence_step {
            SequenceStep::Received { .. } => 0,
            SequenceStep::Passed { .. } => 1,
            SequenceStep::Emitted { .. } => 2,
            SequenceStep::NewState { .. } => 3,
            SequenceStep::NoStep => 4,
            _ => {
                unreachable!();
            }
        }
    }

    fn can_swap(
        first_sequence_step: &<Self as ModelTypes>::SequenceStep,
        second_sequence_step: &<Self as ModelTypes>::SequenceStep,
    ) -> bool {
        if Self::swap_order(first_sequence_step) > Self::swap_order(second_sequence_step) {
            return Self::can_swap(second_sequence_step, first_sequence_step);
        }

        match (first_sequence_step, second_sequence_step) {
            (
                SequenceStep::Received {
                    target_index: first_target_index,
                    ..
                },
                SequenceStep::Received {
                    target_index: second_target_index,
                    ..
                },
            ) => second_target_index != first_target_index,

            (
                SequenceStep::Received {
                    target_index: first_target_index,
                    ..
                },
                SequenceStep::Passed {
                    source_index: second_source_index,
                    target_index: second_target_index,
                    ..
                },
            ) => {
                second_source_index != first_target_index
                    && second_target_index != first_target_index
            }

            (
                SequenceStep::Received {
                    message_id: first_message_id,
                    target_index: first_target_index,
                    ..
                },
                SequenceStep::Emitted {
                    message_id: second_message_id,
                    source_index: second_source_index,
                    ..
                },
            ) => second_message_id != first_message_id && second_source_index != first_target_index,

            (
                SequenceStep::Received {
                    target_index: first_target_index,
                    ..
                },
                SequenceStep::NewState {
                    agent_index: second_agent_index,
                    ..
                },
            ) => second_agent_index != first_target_index,

            (
                SequenceStep::Passed {
                    source_index: first_source_index,
                    target_index: first_target_index,
                    ..
                },
                SequenceStep::Passed {
                    source_index: second_source_index,
                    target_index: second_target_index,
                    ..
                },
            ) => {
                second_source_index != first_source_index
                    // BEGIN NOT TESTED
                    && second_source_index != first_target_index
                    && second_target_index != first_source_index
                    && second_target_index != first_target_index
                // END NOT TESTED
            }

            // BEGIN NOT TESTED
            (
                SequenceStep::Passed {
                    source_index: first_source_index,
                    target_index: first_target_index,
                    ..
                },
                SequenceStep::Emitted {
                    source_index: second_source_index,
                    ..
                },
            ) => {
                second_source_index != first_source_index
                    && second_source_index != first_target_index
            }
            // END NOT TESTED
            (
                SequenceStep::Passed {
                    source_index: first_source_index,
                    target_index: first_target_index,
                    ..
                },
                SequenceStep::NewState {
                    agent_index: second_agent_index,
                    ..
                },
            ) => {
                second_agent_index != first_source_index && second_agent_index != first_target_index
            }

            (
                SequenceStep::Emitted {
                    source_index: first_source_index,
                    ..
                },
                SequenceStep::Emitted {
                    source_index: second_source_index,
                    ..
                },
            ) => second_source_index != first_source_index,

            (
                SequenceStep::Emitted {
                    source_index: first_source_index,
                    ..
                },
                SequenceStep::NewState {
                    agent_index: second_agent_index,
                    ..
                },
            ) => second_agent_index != first_source_index,

            (
                SequenceStep::NewState {
                    agent_index: first_agent_index,
                    ..
                },
                SequenceStep::NewState {
                    agent_index: second_agent_index,
                    ..
                },
            ) => second_agent_index != first_agent_index,

            (_, SequenceStep::NoStep) => true,

            _ => {
                unreachable!();
            }
        }
    }

    fn merge_message_steps(
        &self,
        sequence_steps: &mut [<Self as ModelTypes>::SequenceStep],
    ) -> bool {
        let mut did_merge_messages = false;
        for first_index in 0..(sequence_steps.len() - 1) {
            if let SequenceStep::Emitted {
                source_index: first_source_index,
                target_index: first_target_index,
                message_id: first_message_id,
                replaced,
                is_immediate,
            } = sequence_steps[first_index]
            {
                for second_index in (first_index + 1)..sequence_steps.len() {
                    match sequence_steps[second_index] {
                        SequenceStep::NoStep => {
                            continue;
                        }

                        SequenceStep::Received {
                            message_id: second_message_id,
                            did_change_state,
                            is_activity: false,
                            ..
                        } => {
                            if second_message_id != first_message_id {
                                continue;
                            }

                            sequence_steps[first_index] = SequenceStep::Passed {
                                source_index: first_source_index,
                                target_index: first_target_index,
                                target_did_change_state: did_change_state,
                                message_id: first_message_id,
                                replaced,
                                is_immediate,
                            };
                            sequence_steps[second_index] = SequenceStep::NoStep;
                            did_merge_messages = true;
                            break;
                        }

                        SequenceStep::Emitted {
                            source_index: second_source_index,
                            ..
                        } if second_source_index != first_target_index => {
                            continue;
                        }

                        // BEGIN NOT TESTED
                        SequenceStep::Passed {
                            source_index: second_source_index,
                            ..
                        } if second_source_index != first_target_index => {
                            continue;
                        }
                        // END NOT TESTED
                        SequenceStep::NewState {
                            agent_index: state_agent_index,
                            ..
                        } if state_agent_index != first_target_index => {
                            continue;
                        }

                        _ => {
                            break;
                        }
                    }
                }
            }
        }
        did_merge_messages
    }

    fn merge_states(&self, sequence_steps: &mut [<Self as ModelTypes>::SequenceStep]) {
        for first_index in 0..(sequence_steps.len() - 1) {
            if let SequenceStep::NewState {
                agent_index,
                state_id,
                is_deferring,
            } = sequence_steps[first_index]
            {
                let mut merged_states: Vec<ChangeState<StateId>> = vec![ChangeState {
                    agent_index,
                    state_id,
                    is_deferring,
                }];

                let mut allowed_agent_indices = [true; MAX_AGENTS];
                allowed_agent_indices[agent_index] = false;

                for second_step in sequence_steps.iter_mut().skip(first_index + 1) {
                    match second_step {
                        SequenceStep::Emitted { source_index, .. } => {
                            allowed_agent_indices[*source_index] = false;
                        }

                        SequenceStep::Passed {
                            source_index,
                            target_index,
                            ..
                        } => {
                            allowed_agent_indices[*source_index] = false;
                            allowed_agent_indices[*target_index] = false;
                        }

                        SequenceStep::Received { target_index, .. } => {
                            allowed_agent_indices[*target_index] = false;
                        }

                        SequenceStep::NewState {
                            agent_index,
                            state_id,
                            is_deferring,
                        } if allowed_agent_indices[*agent_index] => {
                            allowed_agent_indices[*agent_index] = false;
                            merged_states.push(ChangeState {
                                agent_index: *agent_index,
                                state_id: *state_id,
                                is_deferring: *is_deferring,
                            });
                            *second_step = SequenceStep::NoStep;
                        }

                        SequenceStep::NewStates { .. } => {
                            unreachable!();
                        }

                        _ => {}
                    }
                }

                if merged_states.len() > 1 {
                    sequence_steps[first_index] = SequenceStep::NewStates(merged_states);
                }
            }
        }
    }

    pub(crate) fn print_sequence_diagram(
        &mut self,
        first_configuration_id: ConfigurationId,
        last_configuration_id: ConfigurationId,
        mut sequence_steps: &mut [<Self as ModelTypes>::SequenceStep],
        stdout: &mut dyn Write,
    ) {
        self.patch_sequence_steps(&mut sequence_steps);

        let first_configuration = self.configurations.get(first_configuration_id);
        let last_configuration = self.configurations.get(last_configuration_id);

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

        let agents_timelines = vec![
            AgentTimelines {
                left: vec![],
                right: vec![]
            };
            self.agents_count()
        ];

        let mut sequence_state = SequenceState {
            timelines: vec![],
            message_timelines: HashMap::new(),
            agents_timelines,
            has_reactivation_message: false,
        };

        self.print_sequence_participants(&first_configuration, stdout);
        self.print_first_timelines(&mut sequence_state, &first_configuration, stdout);
        self.print_sequence_first_notes(&sequence_state, &first_configuration, stdout);

        for sequence_step in sequence_steps.iter() {
            self.print_sequence_step(&mut sequence_state, sequence_step, stdout);
        }

        if last_configuration.invalid_id.is_valid() {
            // BEGIN NOT TESTED
            writeln!(
                stdout,
                "== {} ==",
                self.display_invalid_id(last_configuration.invalid_id)
            )
            .unwrap();
            // END NOT TESTED
        }

        self.print_sequence_final(&mut sequence_state, stdout);

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
        mut sequence_state: &mut <Self as ModelTypes>::SequenceState,
        first_configuration: &<Self as MetaModel>::Configuration,
        stdout: &mut dyn Write,
    ) {
        for message_id in first_configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
        {
            // BEGIN NOT TESTED
            self.reactivate(&mut sequence_state, stdout);
            let timeline_index =
                self.find_sequence_timeline(&mut sequence_state, *message_id, stdout);
            writeln!(stdout, "activate T{} #Silver", timeline_index).unwrap();
            // END NOT TESTED
        }
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

    fn find_sequence_timeline(
        &self,
        sequence_state: &mut <Self as ModelTypes>::SequenceState,
        message_id: MessageId,
        stdout: &mut dyn Write,
    ) -> usize {
        let message = self.messages.get(message_id);
        let is_rightwards_message = self.is_rightwards_message(&message);
        let empty_timeline_index = if is_rightwards_message {
            sequence_state.agents_timelines[message.source_index]
                .right
                .iter()
                .copied()
                .find(|timeline_index| sequence_state.timelines[*timeline_index].is_none())
        } else {
            // BEGIN NOT TESTED
            sequence_state.agents_timelines[message.source_index]
                .left
                .iter()
                .copied()
                .find(|timeline_index| sequence_state.timelines[*timeline_index].is_none())
            // END NOT TESTED
        };

        let timeline_index = empty_timeline_index.unwrap_or_else(|| sequence_state.timelines.len());

        let minimal_message_id = self.minimal_message_id(message_id);
        sequence_state
            .message_timelines
            .insert(minimal_message_id, timeline_index);

        if empty_timeline_index.is_some() {
            // BEGIN NOT TESTED
            sequence_state.timelines[timeline_index] = Some(minimal_message_id);
            return timeline_index;
            // END NOT TESTED
        }
        sequence_state.timelines.push(Some(minimal_message_id));

        let message = self.messages.get(message_id);
        let timeline_order = if is_rightwards_message {
            sequence_state.agents_timelines[message.source_index]
                .right
                .push(timeline_index);
            self.agent_scaled_order(message.source_index)
                + sequence_state.agents_timelines[message.source_index]
                    .right
                    .len()
        } else {
            // BEGIN NOT TESTED
            sequence_state.agents_timelines[message.source_index]
                .left
                .push(timeline_index);
            self.agent_scaled_order(message.source_index)
                - sequence_state.agents_timelines[message.source_index]
                    .left
                    .len()
            // END NOT TESTED
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
        sequence_state: &<Self as ModelTypes>::SequenceState,
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
            .filter(|(_timeline_index, message_id)| message_id.is_some()) // MAYBE TESTED
            .for_each(|(timeline_index, message_id)| {
                // BEGIN NOT TESTED
                let message = self.messages.get(message_id.unwrap());
                writeln!(
                    stdout,
                    "/ rnote over T{} : {}",
                    timeline_index,
                    self.display_sequence_message(&message, false)
                )
                .unwrap();
                // END NOT TESTED
            });
    }

    fn print_sequence_step(
        &self,
        mut sequence_state: &mut <Self as ModelTypes>::SequenceState,
        sequence_step: &<Self as ModelTypes>::SequenceStep,
        stdout: &mut dyn Write,
    ) {
        match sequence_step {
            SequenceStep::NoStep => {}

            SequenceStep::Received {
                target_index,
                did_change_state,
                is_activity: true,
                message_id,
                ..
            } => {
                let message = self.messages.get(*message_id);

                if *did_change_state {
                    sequence_state.has_reactivation_message = false;
                    self.reactivate(&mut sequence_state, stdout);
                    writeln!(stdout, "deactivate A{}", target_index).unwrap();
                    sequence_state.has_reactivation_message = false;
                }

                writeln!(
                    stdout,
                    "note over A{} : {}",
                    target_index,
                    self.display_sequence_message(&message, true)
                )
                .unwrap();
            }

            SequenceStep::Received {
                target_index,
                did_change_state,
                is_activity: false,
                message_id,
                ..
            } => {
                let message = self.messages.get(*message_id);
                let minimal_message_id = self.minimal_message_id(*message_id);
                let timeline_index = *sequence_state
                    .message_timelines
                    .get(&minimal_message_id)
                    .unwrap();

                let arrow = match message.order {
                    MessageOrder::Immediate => "-[#Crimson]>",
                    MessageOrder::Unordered => "->",
                    MessageOrder::Ordered(_) => "-[#Blue]>",
                };

                writeln!(
                    stdout,
                    "T{} {} A{} : {}",
                    timeline_index,
                    arrow,
                    message.target_index,
                    self.display_sequence_message(&message, true)
                )
                .unwrap();
                sequence_state.has_reactivation_message = true;

                writeln!(stdout, "deactivate T{}", timeline_index).unwrap();

                sequence_state.message_timelines.remove(&minimal_message_id);
                sequence_state.timelines[timeline_index] = None;
                if *did_change_state {
                    writeln!(stdout, "deactivate A{}", target_index).unwrap();
                    sequence_state.has_reactivation_message = false;
                }
            }

            SequenceStep::Emitted {
                source_index,
                message_id,
                replaced,
                ..
            } => {
                let timeline_index = match replaced // MAYBE TESTED
                {
                    Some(replaced_message_id) => {
                        let replaced_minimal_message_id = self.minimal_message_id(*replaced_message_id);
                        let timeline_index = *sequence_state
                            .message_timelines
                            .get(&replaced_minimal_message_id)
                            .unwrap();
                        sequence_state
                            .message_timelines
                            .remove(&replaced_minimal_message_id);
                        let minimal_message_id = self.minimal_message_id(*message_id);
                        sequence_state
                            .message_timelines
                            .insert(minimal_message_id, timeline_index);
                        sequence_state.timelines[timeline_index] = Some(minimal_message_id);
                        timeline_index
                    }
                    None => self.find_sequence_timeline(&mut sequence_state, *message_id, stdout),
                };
                let message = self.messages.get(*message_id);
                let arrow = match message.order {
                    MessageOrder::Immediate => "-[#Crimson]>",
                    MessageOrder::Unordered => "->",
                    MessageOrder::Ordered(_) => "-[#Blue]>",
                };
                writeln!(
                    stdout,
                    "A{} {} T{} : {}",
                    source_index,
                    arrow,
                    timeline_index,
                    self.display_sequence_message(&message, false)
                )
                .unwrap();
                if replaced.is_none() {
                    writeln!(stdout, "activate T{} #Silver", timeline_index).unwrap();
                }
                sequence_state.has_reactivation_message = true;
            }

            SequenceStep::Passed {
                source_index,
                target_index,
                target_did_change_state,
                message_id,
                replaced,
                ..
            } => {
                let replaced_timeline_index: Option<usize> = if let Some(replaced_message_id) =
                    replaced
                {
                    let replaced_minimal_message_id = self.minimal_message_id(*replaced_message_id);
                    sequence_state
                        .message_timelines
                        .get(&replaced_minimal_message_id)
                        .copied()
                        .map(|timeline_index| {
                            sequence_state
                                .message_timelines
                                .remove(&replaced_minimal_message_id);
                            sequence_state.timelines[timeline_index] = None;
                            timeline_index
                        })
                } else {
                    None
                };
                let message = self.messages.get(*message_id);
                let arrow = match message.order {
                    MessageOrder::Immediate => "-[#Crimson]>",
                    MessageOrder::Unordered => "->",
                    MessageOrder::Ordered(_) => "-[#Blue]>",
                };
                writeln!(
                    stdout,
                    "A{} {} A{} : {}",
                    source_index,
                    arrow,
                    target_index,
                    self.display_sequence_message(&message, false)
                )
                .unwrap();
                sequence_state.has_reactivation_message = true;

                if let Some(timeline_index) = replaced_timeline_index {
                    writeln!(stdout, "deactivate T{}", timeline_index).unwrap();
                }

                if *target_did_change_state {
                    writeln!(stdout, "deactivate A{}", target_index).unwrap();
                    sequence_state.has_reactivation_message = false;
                }
            }

            SequenceStep::NewState {
                agent_index,
                state_id,
                is_deferring,
            } => {
                self.reactivate(&mut sequence_state, stdout);
                if *is_deferring {
                    writeln!(stdout, "activate A{} #MediumPurple", agent_index).unwrap();
                } else {
                    writeln!(stdout, "activate A{} #CadetBlue", agent_index).unwrap();
                }
                let agent_type = &self.agent_types[*agent_index];
                let agent_state = agent_type.display_state(*state_id);
                writeln!(stdout, "rnote over A{} : {}", agent_index, agent_state).unwrap();
                sequence_state.has_reactivation_message = false;
            }

            SequenceStep::NewStates(new_states) => {
                self.reactivate(&mut sequence_state, stdout);

                for new_state in new_states.iter() {
                    let color = if new_state.is_deferring {
                        "#MediumPurple"
                    } else {
                        "#CadetBlue"
                    };
                    writeln!(stdout, "activate A{} {}", new_state.agent_index, color).unwrap();
                }

                for (state_index, new_state) in new_states.iter().enumerate() {
                    if state_index > 0 {
                        write!(stdout, "/ ").unwrap();
                    }
                    let agent_index = new_state.agent_index;
                    let agent_type = &self.agent_types[agent_index];
                    let agent_state = agent_type.display_state(new_state.state_id);
                    writeln!(stdout, "rnote over A{} : {}", agent_index, agent_state).unwrap();
                }
            }
        }
    }

    fn reactivate(
        &self,
        mut sequence_state: &mut <Self as ModelTypes>::SequenceState,
        stdout: &mut dyn Write,
    ) {
        if !sequence_state.has_reactivation_message {
            writeln!(stdout, "autonumber stop").unwrap();
            writeln!(stdout, "[<[#White]-- A0").unwrap();
            writeln!(stdout, "autonumber resume").unwrap();
            sequence_state.has_reactivation_message = true;
        }
    }

    fn print_sequence_final(
        &self,
        mut sequence_state: &mut <Self as ModelTypes>::SequenceState,
        stdout: &mut dyn Write,
    ) {
        sequence_state.has_reactivation_message = false;
        self.reactivate(&mut sequence_state, stdout);
        for agent_index in 0..self.agents_count() {
            writeln!(stdout, "deactivate A{}", agent_index).unwrap();
        }
        for (timeline_index, message_id) in sequence_state.timelines.iter().enumerate() {
            if message_id.is_some() {
                writeln!(stdout, "deactivate T{}", timeline_index).unwrap(); // NOT TESTED
            }
        }
    }

    fn collect_agent_state_transitions(
        &self,
        condense: &Condense,
        agent_index: usize,
    ) -> <Self as ModelTypes>::AgentStateTransitions {
        let mut state_transitions = <Self as ModelTypes>::AgentStateTransitions::default();
        self.outgoings
            .iter()
            .take(self.configurations.len())
            .enumerate()
            .for_each(|(from_configuration_id, outgoings)| {
                if self.print_progress(from_configuration_id) {
                    eprintln!(
                        "collected {} out of {} configurations ({}%)",
                        from_configuration_id,
                        self.configurations.len(),
                        (100 * from_configuration_id) / self.configurations.len(),
                    );
                }
                let from_configuration = self
                    .configurations
                    .get(ConfigurationId::from_usize(from_configuration_id));
                outgoings.iter().for_each(|outgoing| {
                    let to_configuration = self.configurations.get(outgoing.to_configuration_id);
                    self.collect_agent_state_transition(
                        condense,
                        agent_index,
                        &from_configuration,
                        &to_configuration,
                        outgoing.delivered_message_id,
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
        mut delivered_message_id: MessageId,
        state_transitions: &mut <Self as ModelTypes>::AgentStateTransitions,
    ) {
        let agent_type = &self.agent_types[agent_index];
        let agent_instance = self.agent_instance(agent_index);

        let mut from_state_id = from_configuration.state_ids[agent_index];
        let mut to_state_id = to_configuration.state_ids[agent_index];
        if condense.names_only {
            from_state_id = agent_type.terse_id(from_state_id);
            to_state_id = agent_type.terse_id(to_state_id);
        }

        let context = AgentStateTransitionContext {
            from_state_id,
            from_is_deferring: agent_type
                .state_is_deferring(agent_instance, &from_configuration.state_ids),
            to_state_id,
            to_is_deferring: agent_type
                .state_is_deferring(agent_instance, &to_configuration.state_ids),
            is_constraint: agent_type.is_state_before(from_state_id, to_state_id),
        };

        let mut sent_message_ids: Vec<MessageId> = vec![];

        to_configuration
            .message_ids
            .iter()
            .take_while(|to_message_id| to_message_id.is_valid())
            .map(|to_message_id| (to_message_id, self.messages.get(*to_message_id)))
            .filter(|(_, to_message)| to_message.source_index == agent_index)
            .for_each(|(to_message_id, _)| {
                if !self.to_message_kept_in_transition(
                    *to_message_id,
                    &from_configuration,
                    delivered_message_id,
                ) {
                    let message_id = self.terse_of_message_id[to_message_id.to_usize()];
                    sent_message_ids.push(message_id);
                    sent_message_ids.sort();
                }
            });

        let delivered_message = self.messages.get(delivered_message_id);
        let is_delivered_to_us = delivered_message.target_index == agent_index;
        if !is_delivered_to_us
            && context.from_state_id == context.to_state_id
            && sent_message_ids.is_empty()
        {
            return;
        }

        state_transitions
            .entry(context)
            .or_insert_with(HashMap::new);

        let state_delivered_message_ids: &mut HashMap<Vec<MessageId>, Vec<MessageId>> =
            state_transitions.get_mut(&context).unwrap();

        state_delivered_message_ids
            .entry(sent_message_ids.to_vec())
            .or_insert_with(Vec::new);

        let delivered_message_ids: &mut Vec<MessageId> = state_delivered_message_ids
            .get_mut(&sent_message_ids.to_vec())
            .unwrap();

        delivered_message_id = self.terse_of_message_id[delivered_message_id.to_usize()];

        if !delivered_message_ids
            .iter()
            .any(|message_id| *message_id == delivered_message_id)
        {
            delivered_message_ids.push(delivered_message_id);
            delivered_message_ids.sort();
        }
    }

    fn to_message_kept_in_transition(
        &self,
        to_message_id: MessageId,
        from_configuration: &<Self as MetaModel>::Configuration,
        delivered_message_id: MessageId,
    ) -> bool {
        if self.message_exists_in_configuration(
            to_message_id,
            from_configuration,
            Some(delivered_message_id),
        ) {
            return true;
        }

        match self.incr_message_id(to_message_id) {
            None => false,
            Some(incr_message_id) => self.message_exists_in_configuration(
                incr_message_id,
                from_configuration,
                Some(delivered_message_id),
            ),
        }
    }

    fn message_exists_in_configuration(
        &self,
        message_id: MessageId,
        configuration: &<Self as MetaModel>::Configuration,
        exclude_message_id: Option<MessageId>,
    ) -> bool {
        configuration
            .message_ids
            .iter()
            .take_while(|configuration_message_id| configuration_message_id.is_valid())
            .filter(|configuration_message_id| {
                Some(**configuration_message_id) != exclude_message_id
            })
            .any(|configuration_message_id| *configuration_message_id == message_id)
    }
}
