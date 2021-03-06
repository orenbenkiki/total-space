use crate::memoize::*;
use crate::reactions::*;
use crate::utilities::*;

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

/// Specify the number of agent instances to use.
pub enum Instances {
    /// Always have just a single instance.
    Singleton,

    /// Use a specific number of instances.
    Count(usize),
}

/// How to display an agent instance in sequence diagrams.
#[derive(Clone, Copy, Debug)]
pub struct InstanceAppearance {
    /// The relative order of the instance.
    pub order: usize,

    /// The optional group to put the instance in.
    pub group: Option<&'static str>,
}

impl Default for InstanceAppearance {
    fn default() -> Self {
        Self {
            order: 0,
            group: None,
        }
    }
}

/// A trait partially describing some agent instances of the same type.
pub trait AgentInstances<StateId: IndexLike, Payload: DataLike>: Name {
    /// Return the previous agent type in the chain, if any.
    fn prev_agent_type(&self) -> Option<Rc<dyn AgentType<StateId, Payload>>>;

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

    /// Control the appearance of each instance in a sequence diagram.
    fn instance_appearance(&self, instance: usize) -> InstanceAppearance;

    /// Display the state.
    ///
    /// The format of the display must be either `<state-name>` if the state is a simple enum, or
    /// `<state-name>(<state-data>)` if the state contains additional data. The `Debug` of the state
    /// might be acceptable as-is, but typically it is better to get rid or shorten the explicit
    /// field names, and/or format their values in a more compact form.
    fn display_state(&self, state_id: StateId) -> Rc<String>;

    /// Convert the full state identifier to the terse state identifier.
    fn terse_id(&self, state_id: StateId) -> StateId;

    /// Return the name of the terse state (just the state name).
    fn display_terse(&self, name_id: StateId) -> String;

    /// If true, the specified agent is a part of this container agent.
    fn does_contain(&self, agent_index: usize) -> bool;
}

/// A trait fully describing some agent instances of the same type.
pub trait AgentType<StateId: IndexLike, Payload: DataLike>:
    AgentInstances<StateId, Payload>
{
    /// Return the actions that may be taken by an agent instance with some state when receiving a
    /// payload.
    fn reaction(
        &self,
        instance: usize,
        state_ids: &[StateId],
        payload: &Payload,
    ) -> Reaction<StateId, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn activity(&self, instance: usize, state_ids: &[StateId]) -> Activity<Payload>;

    /// Whether any agent in the state is deferring messages.
    fn state_is_deferring(&self, instance: usize, state_ids: &[StateId]) -> bool;

    /// Return a reason that a state is invalid (unless it is valid).
    fn state_invalid_because(&self, instance: usize, state_ids: &[StateId])
        -> Option<&'static str>;

    /// The maximal number of messages sent by an agent which may be in-flight when it is in the
    /// state.
    fn state_max_in_flight_messages(&self, instance: usize, state_ids: &[StateId])
        -> Option<usize>;

    /// The total number of states seen so far.
    fn states_count(&self) -> usize;

    /// Compute mapping from full states to terse states (name only).
    fn compute_terse(&self);

    /// Is one state "before" another (for diagrams).
    fn is_state_before(&self, left_state_id: StateId, right_state_id: StateId) -> bool;
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
pub struct AgentTypeData<State: DataLike + PartialOrd, StateId: IndexLike, Payload: DataLike> {
    /// Memoization of the agent states.
    states: RefCell<Memoize<State, StateId>>,

    /// The index of the first agent of this type.
    first_index: usize,

    /// The name of the agent type.
    name: &'static str,

    /// Whether this type only has a single instance.
    is_singleton: bool,

    /// The display string of each state.
    label_of_state: RefCell<Vec<Rc<String>>>,

    /// Convert a full state identifier to a terse state identifier.
    terse_of_state: RefCell<Vec<StateId>>,

    /// The names of the terse states (state names only).
    name_of_terse: RefCell<Vec<String>>,

    /// The order of each instance (for sequence diagrams).
    appearance_of_instances: Vec<InstanceAppearance>,

    /// The previous agent type in the chain.
    prev_agent_type: Option<Rc<dyn AgentType<StateId, Payload>>>,

    /// Trick the compiler into thinking we have a field of type Payload.
    _payload: PhantomData<Payload>,
}

/// The data we need to implement an container agent type.
///
/// This should be placed in a `Singleton` to allow the agent states to get services from it.
pub struct ContainerOf1TypeData<
    State: DataLike + PartialOrd,
    Part: DataLike + PartialOrd,
    StateId: IndexLike,
    Payload: DataLike,
    const MAX_PARTS: usize,
> {
    /// The basic agent type data.
    agent_type_data: AgentTypeData<State, StateId, Payload>,

    /// Access part states (for a container).
    part_type: Rc<dyn PartType<Part, StateId>>,
}

/// The data we need to implement an container agent type.
///
/// This should be placed in a `Singleton` to allow the agent states to get services from it.
pub struct ContainerOf2TypeData<
    State: DataLike + PartialOrd,
    Part1: DataLike + PartialOrd,
    Part2: DataLike + PartialOrd,
    StateId: IndexLike,
    Payload: DataLike,
    const MAX_PARTS: usize,
> {
    /// The basic agent type data.
    agent_type_data: AgentTypeData<State, StateId, Payload>,

    /// Access first parts states (for a container).
    part1_type: Rc<dyn PartType<Part1, StateId>>,

    /// Access second parts states (for a container).
    part2_type: Rc<dyn PartType<Part2, StateId>>,
}

// END MAYBE TESTED

impl<State: DataLike + PartialOrd, StateId: IndexLike, Payload: DataLike>
    AgentTypeData<State, StateId, Payload>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(
        name: &'static str,
        instances: Instances,
        prev_agent_type: Option<Rc<dyn AgentType<StateId, Payload>>>,
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
        let mut states = Memoize::new(StateId::invalid().to_usize());
        states.store(default_state);
        let label_of_state = RefCell::new(vec![]);
        label_of_state
            .borrow_mut()
            .push(Rc::new(format!("{}", default_state)));

        let appearance_of_instances = vec![InstanceAppearance::default(); count];

        Self {
            name,
            appearance_of_instances,
            is_singleton,
            label_of_state,
            terse_of_state: RefCell::new(vec![]),
            name_of_terse: RefCell::new(vec![]),
            states: RefCell::new(states),
            first_index: prev_agent_type
                .clone()
                .map_or(0, |agent_type| agent_type.next_index()),
            prev_agent_type,
            _payload: PhantomData,
        }
    }

    // BEGIN NOT TESTED

    /// Set the appearance of an instance of the agent in a sequence diagram.
    pub fn set_appearance(&mut self, instance: usize, appearance: InstanceAppearance) {
        assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count()
        );
        self.appearance_of_instances[instance] = appearance;
    }

    // END NOT TESTED

    /// Compute mapping between full and terse state identifiers.
    fn impl_compute_terse(&self) {
        let mut terse_of_state = self.terse_of_state.borrow_mut();
        let mut name_of_terse = self.name_of_terse.borrow_mut();

        assert!(terse_of_state.is_empty());
        assert!(name_of_terse.is_empty());

        let states = self.states.borrow();
        terse_of_state.reserve(states.len());
        name_of_terse.reserve(states.len());

        for state_id in 0..states.len() {
            let state = states.get(StateId::from_usize(state_id));
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

    /// Is one state "before" another (for diagrams).
    fn impl_is_state_before(&self, left_state_id: StateId, right_state_id: StateId) -> bool {
        self.states.borrow().get(left_state_id) < self.states.borrow().get(right_state_id)
    }

    /// Access the actual state by its identifier.
    pub fn get_state(&self, state_id: StateId) -> State {
        self.states.borrow().get(state_id)
    }
}

impl<
        State: DataLike + PartialOrd,
        Part: DataLike + PartialOrd,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > ContainerOf1TypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(
        name: &'static str,
        instances: Instances,
        part_type: Rc<dyn PartType<Part, StateId>>,
        prev_type: Rc<dyn AgentType<StateId, Payload>>,
    ) -> Self {
        Self {
            agent_type_data: AgentTypeData::new(name, instances, Some(prev_type)),
            part_type,
        }
    }
}

// BEGIN NOT TESTED
impl<
        State: DataLike + PartialOrd,
        Part1: DataLike + PartialOrd,
        Part2: DataLike + PartialOrd,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > ContainerOf2TypeData<State, Part1, Part2, StateId, Payload, MAX_PARTS>
{
    /// Create new agent type data with the specified name and number of instances.
    pub fn new(
        name: &'static str,
        instances: Instances,
        part1_type: Rc<dyn PartType<Part1, StateId>>,
        part2_type: Rc<dyn PartType<Part2, StateId>>,
        prev_type: Rc<dyn AgentType<StateId, Payload>>,
    ) -> Self {
        Self {
            agent_type_data: AgentTypeData::new(name, instances, Some(prev_type)),
            part1_type,
            part2_type,
        }
    }
}
// END NOT TESTED

impl<State: DataLike + PartialOrd, StateId: IndexLike, Payload: DataLike> PartType<State, StateId>
    for AgentTypeData<State, StateId, Payload>
{
    fn part_state_by_id(&self, state_id: StateId) -> State {
        self.states.borrow().get(state_id)
    }

    fn part_first_index(&self) -> usize {
        self.first_index
    }

    fn parts_count(&self) -> usize {
        self.instances_count()
    }
}

/// A trait for a single agent state.
pub trait AgentState<State: DataLike + PartialOrd, Payload: DataLike> {
    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// payload.
    fn reaction(&self, instance: usize, payload: &Payload) -> Reaction<State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        Activity::Passive
    }

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self, _instance: usize) -> bool {
        false
    }

    /// If this object is invalid, return why.
    fn invalid_because(&self, _instance: usize) -> Option<&'static str> {
        None
    }

    /// The maximal number of messages sent by this agent which may be in-flight when it is in this
    /// state.
    fn max_in_flight_messages(&self, _instance: usize) -> Option<usize> {
        None
    }
}

/// A trait for a container agent state.
pub trait ContainerOf1State<
    State: DataLike + PartialOrd,
    Part: DataLike + PartialOrd,
    Payload: DataLike,
>
{
    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// payload.
    fn reaction(
        &self,
        instance: usize,
        payload: &Payload,
        parts: &[Part],
    ) -> Reaction<State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn activity(&self, _instance: usize, _parts: &[Part]) -> Activity<Payload> {
        Activity::Passive
    }

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self, _instance: usize, _parts: &[Part]) -> bool {
        false
    }

    // BEGIN NOT TESTED

    /// If this object is invalid, return why.
    fn invalid_because(&self, _instance: usize, _parts: &[Part]) -> Option<&'static str> {
        None
    }

    // END NOT TESTED

    /// The maximal number of messages sent by this agent which may be in-flight when it is in this
    /// state.
    fn max_in_flight_messages(&self, _instance: usize, _parts: &[Part]) -> Option<usize> {
        None
    }
}

// BEGIN NOT TESTED

/// A trait for a container agent state.
pub trait ContainerOf2State<
    State: DataLike + PartialOrd,
    Part1: DataLike + PartialOrd,
    Part2: DataLike + PartialOrd,
    Payload: DataLike,
>
{
    /// Return the actions that may be taken by an agent instance with this state when receiving a
    /// payload.
    fn reaction(
        &self,
        instance: usize,
        payload: &Payload,
        parts1: &[Part1],
        parts2: &[Part2],
    ) -> Reaction<State, Payload>;

    /// Return the actions that may be taken by an agent with some state when time passes.
    fn activity(
        &self,
        _instance: usize,
        _parts1: &[Part1],
        _parts2: &[Part2],
    ) -> Activity<Payload> {
        Activity::Passive
    }

    /// Whether any agent in this state is deferring messages.
    fn is_deferring(&self, _instance: usize, _parts1: &[Part1], _parts2: &[Part2]) -> bool {
        false
    }

    /// If this object is invalid, return why.
    fn invalid_because(
        &self,
        _instance: usize,
        _parts1: &[Part1],
        _parts2: &[Part2],
    ) -> Option<&'static str> {
        None
    }

    /// The maximal number of messages sent by this agent which may be in-flight when it is in this
    /// state.
    fn max_in_flight_messages(
        &self,
        _instance: usize,
        _parts1: &[Part1],
        _parts2: &[Part2],
    ) -> Option<usize> {
        None
    }
}

// END NOT TESTED

impl<State: DataLike + PartialOrd, StateId: IndexLike, Payload: DataLike>
    AgentTypeData<State, StateId, Payload>
{
    fn translate_reaction(
        &self,
        reaction: &Reaction<State, Payload>,
    ) -> Reaction<StateId, Payload> {
        match reaction {
            Reaction::Unexpected => Reaction::Unexpected,
            Reaction::Ignore => Reaction::Ignore, // MAYBE TESTED
            Reaction::Defer => Reaction::Defer,
            Reaction::Do1(action) => Reaction::Do1(self.translate_action(action)),
            Reaction::Do1Of(actions) => Reaction::Do1Of(self.translate_actions(&actions)), // NOT TESTED
        }
    }

    fn translate_action(&self, action: &Action<State, Payload>) -> Action<StateId, Payload> {
        match *action {
            Action::Defer => Action::Defer,

            Action::Ignore => Action::Ignore, // NOT TESTED
            Action::Change(state) => Action::Change(self.translate_state(state)),

            Action::Send1(emit) => Action::Send1(emit),
            Action::ChangeAndSend1(state, emit) => {
                Action::ChangeAndSend1(self.translate_state(state), emit)
            }

            Action::Sends(emits) => Action::Sends(emits), // NOT TESTED
            Action::ChangeAndSends(state, emits) => {
                Action::ChangeAndSends(self.translate_state(state), emits)
            }
        }
    }

    // BEGIN NOT TESTED

    fn translate_actions(
        &self,
        actions: &[Option<Action<State, Payload>>; MAX_COUNT],
    ) -> [Option<Action<StateId, Payload>>; MAX_COUNT] {
        let mut translated_actions: [Option<Action<StateId, Payload>>; MAX_COUNT] =
            [None; MAX_COUNT];
        for (maybe_action, maybe_translated) in actions.iter().zip(translated_actions.iter_mut()) {
            *maybe_translated = maybe_action
                .as_ref()
                .map(|action| self.translate_action(action));
        }
        translated_actions
    }

    // END NOT TESTED

    fn translate_state(&self, state: State) -> StateId {
        let stored = self.states.borrow_mut().store(state);
        if stored.is_new {
            debug_assert!(self.label_of_state.borrow().len() == stored.id.to_usize());
            self.label_of_state
                .borrow_mut()
                .push(Rc::new(format!("{}", state)));
        }
        stored.id
    }
}

impl<State: DataLike + PartialOrd, StateId: IndexLike, Payload: DataLike> Name
    for AgentTypeData<State, StateId, Payload>
{
    fn name(&self) -> String {
        self.name.to_string()
    }
}

impl<
        State: DataLike + PartialOrd,
        Part: DataLike + PartialOrd,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > Name for ContainerOf1TypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn name(&self) -> String {
        self.agent_type_data.name()
    }
}

// BEGIN NOT TESTED
impl<
        State: DataLike + PartialOrd,
        Part1: DataLike + PartialOrd,
        Part2: DataLike + PartialOrd,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > Name for ContainerOf2TypeData<State, Part1, Part2, StateId, Payload, MAX_PARTS>
{
    fn name(&self) -> String {
        self.agent_type_data.name()
    }
}
// END NOT TESTED

impl<State: DataLike + PartialOrd, StateId: IndexLike, Payload: DataLike>
    AgentInstances<StateId, Payload> for AgentTypeData<State, StateId, Payload>
{
    fn prev_agent_type(&self) -> Option<Rc<dyn AgentType<StateId, Payload>>> {
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
        self.appearance_of_instances.len()
    }

    fn instance_appearance(&self, instance: usize) -> InstanceAppearance {
        self.appearance_of_instances[instance]
    }

    fn display_state(&self, state_id: StateId) -> Rc<String> {
        self.label_of_state.borrow()[state_id.to_usize()].clone()
    }

    fn terse_id(&self, state_id: StateId) -> StateId {
        self.terse_of_state.borrow()[state_id.to_usize()]
    }

    fn display_terse(&self, terse_id: StateId) -> String {
        self.name_of_terse.borrow()[terse_id.to_usize()].clone()
    }

    fn does_contain(&self, _agent_index: usize) -> bool {
        false
    }
}

impl<
        State: DataLike + PartialOrd + ContainerOf1State<State, Part, Payload>,
        Part: DataLike + PartialOrd + PartialOrd + AgentState<Part, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > AgentInstances<StateId, Payload>
    for ContainerOf1TypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn prev_agent_type(&self) -> Option<Rc<dyn AgentType<StateId, Payload>>> {
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

    fn instance_appearance(&self, instance: usize) -> InstanceAppearance {
        self.agent_type_data.instance_appearance(instance)
    }

    fn display_state(&self, state_id: StateId) -> Rc<String> {
        self.agent_type_data.display_state(state_id)
    }

    // BEGIN NOT TESTED
    fn terse_id(&self, state_id: StateId) -> StateId {
        self.agent_type_data.terse_id(state_id)
    }

    fn display_terse(&self, terse_id: StateId) -> String {
        self.agent_type_data.display_terse(terse_id)
    }

    fn does_contain(&self, agent_index: usize) -> bool {
        let part_first_index = self.part_type.part_first_index();
        let parts_count = self.part_type.parts_count();
        part_first_index <= agent_index && agent_index < part_first_index + parts_count
    }
    // END NOT TESTED
}

// BEGIN NOT TESTED
impl<
        State: DataLike + PartialOrd + ContainerOf2State<State, Part1, Part2, Payload>,
        Part1: DataLike + PartialOrd + AgentState<Part1, Payload>,
        Part2: DataLike + PartialOrd + AgentState<Part2, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > AgentInstances<StateId, Payload>
    for ContainerOf2TypeData<State, Part1, Part2, StateId, Payload, MAX_PARTS>
{
    fn prev_agent_type(&self) -> Option<Rc<dyn AgentType<StateId, Payload>>> {
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

    fn instance_appearance(&self, instance: usize) -> InstanceAppearance {
        self.agent_type_data.instance_appearance(instance)
    }

    fn display_state(&self, state_id: StateId) -> Rc<String> {
        self.agent_type_data.display_state(state_id)
    }

    fn terse_id(&self, state_id: StateId) -> StateId {
        self.agent_type_data.terse_id(state_id)
    }

    fn display_terse(&self, terse_id: StateId) -> String {
        self.agent_type_data.display_terse(terse_id)
    }

    fn does_contain(&self, agent_index: usize) -> bool {
        let part1_first_index = self.part1_type.part_first_index();
        let parts1_count = self.part1_type.parts_count();
        let part2_first_index = self.part2_type.part_first_index();
        let parts2_count = self.part2_type.parts_count();
        (part1_first_index <= agent_index && agent_index < part1_first_index + parts1_count)
            || (part2_first_index <= agent_index && agent_index < part2_first_index + parts2_count)
    }
}
// END NOT TESTED

impl<
        State: DataLike + PartialOrd + AgentState<State, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
    > AgentType<StateId, Payload> for AgentTypeData<State, StateId, Payload>
{
    fn reaction(
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
        let state = self
            .states
            .borrow()
            .get(state_ids[self.first_index + instance]);
        let reaction = state.reaction(instance, payload);
        self.translate_reaction(&reaction)
    }

    fn activity(&self, instance: usize, state_ids: &[StateId]) -> Activity<Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );
        self.states
            .borrow()
            .get(state_ids[self.first_index + instance])
            .activity(instance)
    }

    fn state_is_deferring(&self, instance: usize, state_ids: &[StateId]) -> bool {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );
        self.states
            .borrow()
            .get(state_ids[self.first_index + instance])
            .is_deferring(instance)
    }

    fn state_invalid_because(
        &self,
        instance: usize,
        state_ids: &[StateId],
    ) -> Option<&'static str> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );
        self.states
            .borrow()
            .get(state_ids[self.first_index + instance])
            .invalid_because(instance)
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
            .borrow()
            .get(state_ids[self.first_index + instance])
            .max_in_flight_messages(instance)
    }

    fn states_count(&self) -> usize {
        self.states.borrow().len()
    }

    fn compute_terse(&self) {
        self.impl_compute_terse();
    }

    fn is_state_before(&self, left_state_id: StateId, right_state_id: StateId) -> bool {
        self.impl_is_state_before(left_state_id, right_state_id)
    }
}

impl<
        State: DataLike + PartialOrd + ContainerOf1State<State, Part, Payload>,
        Part: DataLike + PartialOrd + AgentState<Part, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > ContainerOf1TypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn collect_parts(&self, state_ids: &[StateId]) -> [Part; MAX_PARTS] {
        let mut parts = [Part::default(); MAX_PARTS];
        let part_first_index = self.part_type.part_first_index();
        (0..self.part_type.parts_count()).for_each(|part_instance| {
            let state_id = state_ids[part_first_index + part_instance];
            parts[part_instance] = self.part_type.part_state_by_id(state_id);
        });

        parts
    }

    // BEGIN NOT TESTED

    /// Set the appearance of an instance of the agent in a sequence diagram.
    pub fn set_appearance(&mut self, instance: usize, appearance: InstanceAppearance) {
        self.agent_type_data.set_appearance(instance, appearance);
    }

    // END NOT TESTED
}

// BEGIN NOT TESTED
impl<
        State: DataLike + PartialOrd + ContainerOf2State<State, Part1, Part2, Payload>,
        Part1: DataLike + PartialOrd + AgentState<Part1, Payload>,
        Part2: DataLike + PartialOrd + AgentState<Part2, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > ContainerOf2TypeData<State, Part1, Part2, StateId, Payload, MAX_PARTS>
{
    fn collect_parts(&self, state_ids: &[StateId]) -> ([Part1; MAX_PARTS], [Part2; MAX_PARTS]) {
        let mut parts1 = [Part1::default(); MAX_PARTS];
        let part1_first_index = self.part1_type.part_first_index();
        (0..self.part1_type.parts_count()).for_each(|part1_instance| {
            let state_id = state_ids[part1_first_index + part1_instance];
            parts1[part1_instance] = self.part1_type.part_state_by_id(state_id);
        });

        let mut parts2 = [Part2::default(); MAX_PARTS];
        let part2_first_index = self.part2_type.part_first_index();
        (0..self.part2_type.parts_count()).for_each(|part2_instance| {
            let state_id = state_ids[part2_first_index + part2_instance];
            parts2[part2_instance] = self.part2_type.part_state_by_id(state_id);
        });

        (parts1, parts2)
    }

    /// Set the appearance of an instance of the agent in a sequence diagram.
    pub fn set_appearance(&mut self, instance: usize, appearance: InstanceAppearance) {
        self.agent_type_data.set_appearance(instance, appearance);
    }
}
// END NOT TESTED

impl<
        State: DataLike + PartialOrd + ContainerOf1State<State, Part, Payload>,
        Part: DataLike + PartialOrd + AgentState<Part, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > AgentType<StateId, Payload>
    for ContainerOf1TypeData<State, Part, StateId, Payload, MAX_PARTS>
{
    fn reaction(
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

        let all_parts = self.collect_parts(state_ids);
        let parts = &all_parts[..self.part_type.parts_count()];

        let reaction = self
            .agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .reaction(instance, payload, parts);
        self.agent_type_data.translate_reaction(&reaction)
    }

    fn activity(&self, instance: usize, state_ids: &[StateId]) -> Activity<Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );

        let all_parts = self.collect_parts(state_ids);
        let parts = &all_parts[..self.part_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .activity(instance, parts)
    }

    fn state_is_deferring(&self, instance: usize, state_ids: &[StateId]) -> bool {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count() // NOT TESTED
        );

        let all_parts = self.collect_parts(state_ids);
        let parts = &all_parts[..self.part_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .is_deferring(instance, parts)
    }

    // BEGIN NOT TESTED
    fn state_invalid_because(
        &self,
        instance: usize,
        state_ids: &[StateId],
    ) -> Option<&'static str> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count()
        );

        let all_parts = self.collect_parts(state_ids);
        let parts = &all_parts[..self.part_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .invalid_because(instance, parts)
    }
    // END NOT TESTED

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

        let all_parts = self.collect_parts(state_ids);
        let parts = &all_parts[..self.part_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .max_in_flight_messages(instance, parts)
    }

    // BEGIN NOT TESTED
    fn states_count(&self) -> usize {
        self.agent_type_data.states.borrow().len()
    }
    // END NOT TESTED

    fn compute_terse(&self) {
        self.agent_type_data.impl_compute_terse();
    }

    // BEGIN NOT TESTED
    fn is_state_before(&self, left_state_id: StateId, right_state_id: StateId) -> bool {
        self.agent_type_data
            .impl_is_state_before(left_state_id, right_state_id)
    }
    // END NOT TESTED
}

// BEGIN NOT TESTED
impl<
        State: DataLike + PartialOrd + ContainerOf2State<State, Part1, Part2, Payload>,
        Part1: DataLike + PartialOrd + AgentState<Part1, Payload>,
        Part2: DataLike + PartialOrd + AgentState<Part2, Payload>,
        StateId: IndexLike,
        Payload: DataLike,
        const MAX_PARTS: usize,
    > AgentType<StateId, Payload>
    for ContainerOf2TypeData<State, Part1, Part2, StateId, Payload, MAX_PARTS>
{
    fn reaction(
        &self,
        instance: usize,
        state_ids: &[StateId],
        payload: &Payload,
    ) -> Reaction<StateId, Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count()
        );

        let (all_parts1, all_parts2) = self.collect_parts(state_ids);
        let parts1 = &all_parts1[..self.part1_type.parts_count()];
        let parts2 = &all_parts2[..self.part2_type.parts_count()];

        let reaction = self
            .agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .reaction(instance, payload, parts1, parts2);
        self.agent_type_data.translate_reaction(&reaction)
    }

    fn activity(&self, instance: usize, state_ids: &[StateId]) -> Activity<Payload> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count()
        );

        let (all_parts1, all_parts2) = self.collect_parts(state_ids);
        let parts1 = &all_parts1[..self.part1_type.parts_count()];
        let parts2 = &all_parts2[..self.part2_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .activity(instance, parts1, parts2)
    }

    fn state_is_deferring(&self, instance: usize, state_ids: &[StateId]) -> bool {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count()
        );

        let (all_parts1, all_parts2) = self.collect_parts(state_ids);
        let parts1 = &all_parts1[..self.part1_type.parts_count()];
        let parts2 = &all_parts2[..self.part2_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .is_deferring(instance, parts1, parts2)
    }

    fn state_invalid_because(
        &self,
        instance: usize,
        state_ids: &[StateId],
    ) -> Option<&'static str> {
        debug_assert!(
            instance < self.instances_count(),
            "instance: {} count: {}",
            instance,
            self.instances_count()
        );

        let (all_parts1, all_parts2) = self.collect_parts(state_ids);
        let parts1 = &all_parts1[..self.part1_type.parts_count()];
        let parts2 = &all_parts2[..self.part2_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .invalid_because(instance, parts1, parts2)
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
            self.instances_count()
        );

        let (all_parts1, all_parts2) = self.collect_parts(state_ids);
        let parts1 = &all_parts1[..self.part1_type.parts_count()];
        let parts2 = &all_parts2[..self.part2_type.parts_count()];

        self.agent_type_data
            .states
            .borrow()
            .get(state_ids[self.agent_type_data.first_index + instance])
            .max_in_flight_messages(instance, parts1, parts2)
    }

    fn states_count(&self) -> usize {
        self.agent_type_data.states.borrow().len()
    }

    fn compute_terse(&self) {
        self.agent_type_data.impl_compute_terse();
    }

    fn is_state_before(&self, left_state_id: StateId, right_state_id: StateId) -> bool {
        self.agent_type_data
            .impl_is_state_before(left_state_id, right_state_id)
    }
}
// END NOT TESTED

// BEGIN MAYBE TESTED

/// A macro for implementing some `IndexLike` type.
///
/// This should be concerted to a derive macro.
#[macro_export]
macro_rules! index_type {
    ($name:ident, $type:ident) => {
        #[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
        pub struct $name($type);

        impl KeyLike for $name {}

        impl total_space::IndexLike for $name {
            fn from_usize(value: usize) -> Self {
                $name($type::from_usize(value).unwrap())
            }

            fn to_usize(self) -> usize {
                let $name(value) = self;
                $type::to_usize(&value).unwrap()
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
