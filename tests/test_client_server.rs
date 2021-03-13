mod common;

use clap::App;
use clap::ArgMatches;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::str;
use std::sync::Arc;
use strum::IntoStaticStr;
use total_space::*;

declare_global_agent_index! {CLIENT}
declare_global_agent_index! {SERVER}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Ping,
    Request,
    Response,
}
impl_data_like! {
    Payload = Self::Ping,
    "Ping" => "PNG",
    "Request" => "REQ",
    "Response" => "RSP"
}
// END MAYBE TESTED

impl Validated for Payload {}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait,
}
impl_data_like! {
    ClientState = Self::Idle,
    "Idle" => "IDL",
    "Wait" => "WAT"
}
// END MAYBE TESTED

impl Validated for ClientState {}

fn is_maybe_ping(payload: Option<Payload>) -> bool {
    matches!(payload, None | Some(Payload::Ping))
}

impl AgentState<ClientState, Payload> for ClientState {
    fn activity(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1Of2(
                Action::ChangeAndSend2(
                    Self::Wait,
                    Emit::Unordered(Payload::Request, agent_index!(SERVER)),
                    Emit::ImmediateReplacement(is_maybe_ping, Payload::Ping, agent_index!(SERVER)),
                ),
                Action::Send1(Emit::UnorderedReplacement(
                    is_maybe_ping,
                    Payload::Ping,
                    agent_index!(SERVER),
                )),
            ),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Wait, Payload::Response) => Reaction::Do1(Action::Change(Self::Idle)),
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn max_in_flight_messages(&self) -> Option<usize> {
        Some(2)
    }
}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    Listen,
    Work,
}
impl_data_like! { ServerState = Self::Listen, "Listen" => "LST", "Work" => "WRK" }
// END MAYBE TESTED

impl Validated for ServerState {}

impl AgentState<ServerState, Payload> for ServerState {
    fn activity(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Listen => Reaction::Ignore,
            Self::Work => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, agent_index!(CLIENT)),
            )),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (_, Payload::Ping) => Reaction::Ignore,
            (Self::Listen, Payload::Request) => Reaction::Do1(Action::Change(Self::Work)),
            (Self::Work, Payload::Request) => Reaction::Defer, // NOT TESTED
            _ => Reaction::Unexpected,                         // NOT TESTED
        }
    }

    fn is_deferring(&self) -> bool {
        self == &Self::Work
    }

    fn max_in_flight_messages(&self) -> Option<usize> {
        Some(1)
    }
}

index_type! { StateId, u8 }
index_type! { MessageId, u8 }
index_type! { InvalidId, u8 }
index_type! { ConfigurationId, u32 }

type TestModel = Model<
    StateId,
    MessageId,
    InvalidId,
    ConfigurationId,
    Payload,
    6,  // MAX_AGENTS
    14, // MAX_MESSAGES
>;

fn test_model(arg_matches: &ArgMatches) -> TestModel {
    let client_type = Arc::new(AgentTypeData::<
        ClientState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new("C", Instances::Singleton, None));

    let server_type = Arc::new(AgentTypeData::<
        ServerState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new(
        "S", Instances::Singleton, Some(client_type.clone())
    ));

    let model = TestModel::new(model_size(arg_matches, 1), server_type, vec![]);
    init_global_agent_index!(CLIENT, "C", model);
    init_global_agent_index!(SERVER, "S", model);
    model
}

test_case! { conditions, "txt", vec!["test", "conditions"] }
test_case! { agents, "txt", vec!["test", "agents"] }
test_case! { configurations, "txt", vec!["test", "-r", "-p", "1", "-t", "1", "configurations"] }
test_case! { transitions, "txt", vec!["test", "-p", "1", "-t", "1", "transitions"] }
test_case! { abort, "txt", vec!["test", "-p", "1", "-t", "1", "path", "INIT", "!INIT"] }
test_case! { path, "txt", vec!["test", "-p", "1", "-t", "1", "path", "INIT", "2MSG", "INIT", "UNORDERED_REPLACEMENT", "IMMEDIATE_REPLACEMENT", "INIT"] }
test_case! { sequence, "uml", vec!["test", "-p", "1", "-t", "1", "sequence", "INIT", "2MSG", "INIT", "UNORDERED_REPLACEMENT", "IMMEDIATE_REPLACEMENT", "INIT"] }
test_case! { client_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C"] }
test_case! { client_states_names, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C", "-n"] }
test_case! { client_states_final, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C", "-f"] }
test_case! { client_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C", "-c"] }
test_case! { server_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "S"] }
test_case! { server_states_names, "dot", vec!["test", "-p", "1", "-t", "1", "states", "S", "-n"] }
test_case! { server_states_final, "dot", vec!["test", "-p", "1", "-t", "1", "states", "S", "-f"] }
test_case! { server_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "S", "-c"] }
