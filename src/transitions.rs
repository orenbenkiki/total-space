use crate::utilities::*;

use std::fmt::Debug;

/// The type of an index of a transition for a specific configuration.
///
/// "A total of 256 transitions should be enough for everybody" ;-)
pub(crate) type TransitionIndex = u8;

/// Transitions to/from some configuration.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Transitions<
    MessageId: IndexLike,
    ConfigurationId: IndexLike,
    const MAX_TRANSITIONS: usize,
> {
    /// The reachable configuration identifiers.
    pub(crate) configuration_ids: [ConfigurationId; MAX_TRANSITIONS],

    /// The identifier of the message that was delivered to its target in the transition.
    pub(crate) delivered_message_ids: [MessageId; MAX_TRANSITIONS],

    /// The number of outgoing transitions (starting at position 0).
    pub(crate) outgoing_count: TransitionIndex,

    /// The number of incoming transitions (ending at MAX_TRANSITIONS).
    pub(crate) incoming_count: TransitionIndex,
}

impl<MessageId: IndexLike, ConfigurationId: IndexLike, const MAX_TRANSITIONS: usize> Default
    for Transitions<MessageId, ConfigurationId, MAX_TRANSITIONS>
{
    fn default() -> Self {
        Transitions {
            configuration_ids: [ConfigurationId::invalid(); MAX_TRANSITIONS],
            delivered_message_ids: [MessageId::invalid(); MAX_TRANSITIONS],
            outgoing_count: 0,
            incoming_count: 0,
        }
    }
}

/// A transition between configurations.
///
/// This is used for both outgoing and incoming configurations.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Transition<MessageId: IndexLike, ConfigurationId: IndexLike> {
    /// The identifier of the other configuration.
    pub(crate) other_configuration_id: ConfigurationId,

    /// The identifier of the message that was delivered to its target agent to reach the target
    /// configuration.
    pub(crate) delivered_message_id: MessageId,
}

/// An iterator on the transitions.
pub(crate) struct TransitionsIterator<
    'a,
    MessageId: IndexLike,
    ConfigurationId: IndexLike,
    const MAX_TRANSITIONS: usize,
> {
    /// The transitions we are iterating on.
    transitions: &'a Transitions<MessageId, ConfigurationId, MAX_TRANSITIONS>,

    /// The position of the next transition.
    next: TransitionIndex,

    /// The stop position.
    stop: TransitionIndex,
}

impl<'a, MessageId: IndexLike, ConfigurationId: IndexLike, const MAX_TRANSITIONS: usize> Iterator
    for TransitionsIterator<'a, MessageId, ConfigurationId, MAX_TRANSITIONS>
{
    type Item = Transition<MessageId, ConfigurationId>;

    fn next(&mut self) -> Option<Transition<MessageId, ConfigurationId>> {
        if self.next == self.stop {
            return None;
        }

        let result = Some(Transition {
            other_configuration_id: self.transitions.configuration_ids[self.next as usize],
            delivered_message_id: self.transitions.delivered_message_ids[self.next as usize],
        });

        self.next += 1;

        result
    }
}

impl<MessageId: IndexLike, ConfigurationId: IndexLike, const MAX_TRANSITIONS: usize>
    Transitions<MessageId, ConfigurationId, MAX_TRANSITIONS>
{
    /// Add an outgoing transition.
    pub(crate) fn add_outgoing(
        &mut self,
        to_configuration_id: ConfigurationId,
        delivered_message_id: MessageId,
    ) {
        assert!((self.outgoing_count as usize + self.incoming_count as usize) < MAX_TRANSITIONS);
        self.configuration_ids[self.outgoing_count as usize] = to_configuration_id;
        self.delivered_message_ids[self.outgoing_count as usize] = delivered_message_id;
        self.outgoing_count += 1;
    }

    /// Add an incoming transition.
    pub(crate) fn add_incoming(
        &mut self,
        from_configuration_id: ConfigurationId,
        delivered_message_id: MessageId,
    ) {
        assert!((self.outgoing_count as usize + self.incoming_count as usize) < MAX_TRANSITIONS);
        self.incoming_count += 1;
        let index = MAX_TRANSITIONS - self.incoming_count as usize;
        self.configuration_ids[index] = from_configuration_id;
        self.delivered_message_ids[index] = delivered_message_id;
    }

    /// Iterate on all the outgoing transitions.
    pub(crate) fn iter_outgoing(
        &self,
    ) -> TransitionsIterator<MessageId, ConfigurationId, MAX_TRANSITIONS> {
        TransitionsIterator {
            transitions: self,
            next: 0,
            stop: self.outgoing_count,
        }
    }

    /// Iterate on all the incoming transitions.
    pub(crate) fn iter_incoming(
        &self,
    ) -> TransitionsIterator<MessageId, ConfigurationId, MAX_TRANSITIONS> {
        TransitionsIterator {
            transitions: self,
            next: MAX_TRANSITIONS as TransitionIndex - self.incoming_count,
            stop: MAX_TRANSITIONS as TransitionIndex,
        }
    }
}
