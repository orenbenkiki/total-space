// FILE MAYBE TESTED

use crate::utilities::*;

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

    /// Consume (handle) the event, keep the state the same, send five messages.
    Send5(
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
    ),

    /// Consume (handle) the event, change the agent state, send five messages.
    ChangeAndSend5(
        State,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
    ),

    /// Consume (handle) the event, keep the state the same, send six messages.
    Send6(
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
    ),

    /// Consume (handle) the event, change the agent state, send six messages.
    ChangeAndSend6(
        State,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
        Emit<Payload>,
    ),
}

/// The reaction of an agent to time passing.
#[derive(PartialEq, Eq, Debug)]
pub enum Activity<Payload: DataLike> {
    /// The agent is passive, will only respond to a message.
    Passive,

    /// The agent activity generates a message, to be delivered to it for processing.
    Process1(Payload),

    /// The agent activity generates one of two messages, to be delivered to it for processing.
    Process1Of2(Payload, Payload),

    /// The agent activity generates one of three messages, to be delivered to it for processing.
    Process1Of3(Payload, Payload, Payload),

    /// The agent activity generates one of four messages, to be delivered to it for processing.
    Process1Of4(Payload, Payload, Payload, Payload),

    /// The agent activity generates one of five messages, to be delivered to it for processing.
    Process1Of5(Payload, Payload, Payload, Payload, Payload),

    /// The agent activity generates one of six messages, to be delivered to it for processing.
    Process1Of6(Payload, Payload, Payload, Payload, Payload, Payload),
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

    /// One of five alternative actions (non-deterministic).
    Do1Of5(
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
    ),

    /// One of four alternative actions (non-deterministic).
    Do1Of6(
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
        Action<State, Payload>,
    ),
}
