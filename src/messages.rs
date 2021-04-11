use crate::utilities::*;

/// The type of the index of a message in the configuration.
///
/// "A total of 256 in-flight messages should be enough for everybody" ;-)
pub type MessageIndex = u8;

// BEGIN MAYBE TESTED

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

impl<Payload: DataLike> KeyLike for Message<Payload> {}

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
