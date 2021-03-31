use crate::messages::*;
use crate::utilities::*;

// BEGIN MAYBE TESTED

/// An indicator that something is invalid.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Invalid<MessageId: IndexLike> {
    /// An agent failed its internal validation check.
    Agent(usize, &'static str),

    /// A message failed its validation check (in the context of some configuration).
    Message(MessageId, &'static str),

    /// A configuration failed the global invalidation check.
    Configuration(&'static str),
}

impl<MessageId: IndexLike> KeyLike for Invalid<MessageId> {}

// END MAYBE TESTED

impl<MessageId: IndexLike> Default for Invalid<MessageId> {
    fn default() -> Self {
        Invalid::Configuration("you should not be seeing this")
    }
}

// BEGIN MAYBE TESTED

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
}

// END MAYBE TESTED

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > KeyLike for Configuration<StateId, MessageId, InvalidId, MAX_AGENTS, MAX_MESSAGES>
{
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
            state_ids: [StateId::invalid(); MAX_AGENTS],
            message_counts: [MessageIndex::invalid(); MAX_AGENTS],
            message_ids: [MessageId::invalid(); MAX_MESSAGES],
            invalid_id: InvalidId::invalid(),
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
    pub(crate) fn remove_message(&mut self, source: usize, mut message_index: usize) {
        debug_assert!(self.message_counts[source].to_usize() > 0);
        debug_assert!(self.message_ids[message_index].is_valid());

        self.message_counts[source].decr();

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
    pub(crate) fn add_message(&mut self, source_index: usize, message_id: MessageId) {
        debug_assert!(source_index != usize::max_value());
        debug_assert!(self.message_counts[source_index] < MessageIndex::invalid());

        assert!(
            !self.message_ids[MAX_MESSAGES - 1].is_valid(),
            "too many in-flight messages, must be at most {}",
            MAX_MESSAGES
        );

        self.message_counts[source_index].incr();
        self.message_ids[MAX_MESSAGES - 1] = message_id;
        self.message_ids.sort();
    }

    /// Change the state of an agent in the configuration.
    pub(crate) fn change_state(&mut self, agent_index: usize, state_id: StateId) {
        self.state_ids[agent_index] = state_id;
    }
}
