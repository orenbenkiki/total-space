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

declare_global_agent_indices! {CLIENTS}
declare_global_agent_index! {SERVER}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Request { client: usize },
    Response,
}
impl_data_like! {
    Payload = Self::Response,
    "client" => "C",
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

impl AgentState<ClientState, Payload> for ClientState {
    fn activity(&self, instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait,
                Emit::Unordered(Payload::Request { client: instance }, agent_index!(SERVER)),
            )),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Wait, Payload::Response) => Reaction::Do1(Action::Change(Self::Idle)),
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn max_in_flight_messages(&self) -> Option<usize> {
        Some(1)
    }
}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    Listen,
    Work { client: usize },
}
impl_data_like! {
    ServerState = Self::Listen,
    "client" => "C",
    "Listen" => "LST",
    "Work" => "WRK"
}
// END MAYBE TESTED

impl Validated for ServerState {}

impl AgentState<ServerState, Payload> for ServerState {
    fn activity(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Listen => Reaction::Ignore,
            Self::Work { client } => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, agent_index!(CLIENTS[*client])),
            )),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Listen, Payload::Request { client }) => {
                Reaction::Do1(Action::Change(Self::Work { client: *client }))
            }
            (Self::Work { client: _ }, Payload::Request { client: _ }) => Reaction::Defer,
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn is_deferring(&self) -> bool {
        matches!(self, &Self::Work { client: _ })
    }

    fn max_in_flight_messages(&self) -> Option<usize> {
        Some(agents_count!(CLIENTS))
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
    >::new("C", Instances::Count(2), None));
    let server_type = Arc::new(AgentTypeData::<
        ServerState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new(
        "SRV", Instances::Singleton, Some(client_type.clone())
    ));
    let model = TestModel::new(model_size(arg_matches, 1), server_type, vec![]);
    init_global_agent_indices!(CLIENTS, "C", model);
    init_global_agent_index!(SERVER, "SRV", model);
    model
}

test_case! { agents, "txt", vec!["test", "agents"] }
test_case! { configurations, "txt", vec!["test", "-p", "1", "-s", "1", "-t", "1", "configurations"] }
test_case! { transitions, "txt", vec!["test", "-p", "1", "-t", "1", "transitions"] }
test_case! { path, "txt", vec!["test", "-r", "-p", "1", "-t", "1", "path", "1MSG", "2MSG", "INIT"] }
test_case! { sequence, "uml", vec!["test", "-p", "1", "-t", "1", "sequence", "INIT", "1MSG", "2MSG", "INIT"] }
test_case! { client_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C(0)"] }
test_case! { client_states_names, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C(0)", "-n"] }
test_case! { client_states_merge, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C(0)", "-m"] }
test_case! { client_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C(0)", "-c"] }
test_case! { server_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "SRV"] }
test_case! { server_states_names, "dot", vec!["test", "-p", "1", "-t", "1", "states", "SRV", "-n"] }
test_case! { server_states_merge, "dot", vec!["test", "-p", "1", "-t", "1", "states", "SRV", "-m"] }
test_case! { server_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "SRV", "-c"] }
