use crate::messages::*;
use crate::utilities::*;

use hashbrown::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Control appearance of state graphs.
#[derive(Debug)]
pub(crate) struct Condense {
    /// Only use names, ignore details of state and payload.
    pub(crate) names_only: bool,

    /// Merge all agent instances.
    pub(crate) merge_instances: bool,

    /// Only consider the final value of a replaced message.
    pub(crate) final_replaced: bool,
}

/// A message in-flight between agents, considering only names.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub(crate) struct TerseMessage {
    /// The terse message order.
    pub(crate) order: MessageOrder,

    /// The source agent index.
    pub(crate) source_index: usize,

    /// The target agent index.
    pub(crate) target_index: usize,

    /// The actual payload (name only).
    pub(crate) payload: String,

    /// The replaced message, if any (name only).
    pub(crate) replaced: Option<String>,
}

// BEGIN MAYBE TESTED

/// The additional control timelines associated with a specific agent.
#[derive(Clone, Debug)]
pub(crate) struct AgentTimelines {
    /// The indices of the control timelines to the left of the agent, ordered from closer to
    /// further.
    pub(crate) left: Vec<usize>,

    /// The indices of the control timelines to the right of the agent, ordered from closer to
    /// further.
    pub(crate) right: Vec<usize>,
}

/// The current state of a sequence diagram.
#[derive(Clone, Debug)]
pub(crate) struct SequenceState<
    MessageId: IndexLike,
    const MAX_AGENTS: usize,
    const MAX_MESSAGES: usize,
> {
    /// For each timeline, the message it contains.
    pub(crate) timelines: Vec<Option<MessageId>>,

    /// For each message in the current configuration, the timeline it is on, if any.
    pub(crate) message_timelines: HashMap<MessageId, usize>,

    /// The additional control timelines of each agent.
    pub(crate) agents_timelines: Vec<AgentTimelines>,

    /// Whether we have any received messages since the last deactivation.
    pub(crate) has_reactivation_message: bool,
}

/// A single step in a sequence diagram.
#[derive(Copy, Clone, Debug)]
pub(crate) enum SequenceStep<StateId: IndexLike, MessageId: IndexLike> {
    /// No step (created when merging steps).
    NoStep,

    /// A message received by an agent, possibly changing its state.
    Received {
        source_index: usize,
        target_index: usize,
        is_activity: bool,
        did_change_state: bool,
        message_id: MessageId,
    },

    /// A message was emitted by an agent, possibly changing its state, possibly replacing an
    /// exiting message.
    Emitted {
        source_index: usize,
        target_index: usize,
        message_id: MessageId,
        replaced: Option<MessageId>,
        is_immediate: bool,
    },

    /// A message passed from one agent to another (e.g., immediate), possibly changing their
    /// states, possibly replacing an existing message.
    Passed {
        source_index: usize,
        target_index: usize,
        target_did_change_state: bool,
        message_id: MessageId,
        replaced: Option<MessageId>,
    },

    /// Update the state of a single agent, which might be deferring.
    NewState {
        agent_index: usize,
        state_id: StateId,
        is_deferring: bool,
    },

    /// Update the state of two agents, which might be deferring.
    NewStates {
        first_agent_index: usize,
        first_state_id: StateId,
        first_is_deferring: bool,
        second_agent_index: usize,
        second_state_id: StateId,
        second_is_deferring: bool,
    },
}
