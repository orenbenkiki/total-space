// FILE MAYBE TESTED

use crate::utilities::*;

/// The maximal number of alternative actions in a reaction, payloads in an activity or emitted
/// messages within an action.
///
/// Making this static allows us to avoid dynamic memory allocation when computing reactions which
/// speeds things up a lot. As a result, the size of a reaction is pretty large, but we allocate it
/// on the stack so that seems like an acceptable trade-off.
pub const MAX_COUNT: usize = 6;

/// A message sent by an agent as part of an alternative action triggered by some event.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
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
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
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

    /// Consume (handle) the event, keep the state the same, send multiple messages.
    Sends([Option<Emit<Payload>>; MAX_COUNT]),

    /// Consume (handle) the event, change the agent state, send multiple messages.
    ChangeAndSends(State, [Option<Emit<Payload>>; MAX_COUNT]),
}

/// The reaction of an agent to time passing.
#[derive(PartialEq, Eq, Debug)]
pub enum Activity<Payload: DataLike> {
    /// The agent is passive, will only respond to a message.
    Passive,

    /// The agent activity generates a message, to be delivered to it for processing.
    Process1(Payload),

    /// The agent activity generates one of several messages, to be delivered to it for processing.
    Process1Of([Option<Payload>; MAX_COUNT]),
}

/// The reaction of an agent to receiving a message.
#[derive(PartialEq, Eq, Debug)]
pub enum Reaction<State: KeyLike, Payload: DataLike> {
    /// Indicate an unexpected event.
    Unexpected,

    /// Defer handling the event.
    ///
    /// This has the same effect as `Do1(Action.Defer)`.
    Defer,

    /// Ignore the event.
    ///
    /// This has the same effect as `Do1(Action.Ignore)`.
    Ignore,

    /// A single action (deterministic).
    Do1(Action<State, Payload>),

    /// One of several alternative actions (non-deterministic).
    Do1Of([Option<Action<State, Payload>>; MAX_COUNT]),
}
