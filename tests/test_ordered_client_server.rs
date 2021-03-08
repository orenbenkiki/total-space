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
    Request(usize),
    Response(usize),
}
impl_data_like! { Payload = Self::Request(1) }
// END MAYBE TESTED

impl Validated for Payload {}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait(usize),
}
impl_data_like! { ClientState = Self::Idle }
// END MAYBE TESTED

impl Validated for ClientState {}

impl AgentState<ClientState, Payload> for ClientState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Idle => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait(1),
                Emit::Ordered(Payload::Request(1), agent_index!(SERVER)),
            )),
            Self::Wait(1) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait(3),
                Emit::Ordered(Payload::Request(2), agent_index!(SERVER)),
            )),
            _ => Reaction::Ignore,
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Wait(mask), Payload::Response(index)) if mask == index => {
                Reaction::Do1(Action::Change(Self::Idle))
            }
            (Self::Wait(mask), Payload::Response(index)) if mask & index != 0 => {
                Reaction::Do1(Action::Change(Self::Wait(mask & !index)))
            }
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
    Work(usize),
}
impl_data_like! { ServerState = Self::Listen }
// END MAYBE TESTED

impl Validated for ServerState {}

impl AgentState<ServerState, Payload> for ServerState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Listen => Reaction::Ignore,
            Self::Work(index) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Ordered(Payload::Response(*index), agent_index!(CLIENT)),
            )),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Listen, Payload::Request(index)) => {
                Reaction::Do1(Action::Change(Self::Work(*index)))
            }
            (Self::Work(_), Payload::Request(_)) => Reaction::Defer,
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn is_deferring(&self) -> bool {
        matches!(self, Self::Work(_))
    }

    fn max_in_flight_messages(&self) -> Option<usize> {
        Some(2)
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
    >::new("Client", Instances::Singleton, None));
    let server_type = Arc::new(AgentTypeData::<
        ServerState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new(
        "Server", Instances::Singleton, Some(client_type.clone())
    ));

    let model = TestModel::new(model_size(arg_matches, 1), server_type, vec![]);
    init_global_agent_index!(CLIENT, "Client", model);
    init_global_agent_index!(SERVER, "Server", model);
    model
}

test_case! { agents, "txt", vec!["test", "agents"] }
test_case! { configurations, "txt", vec!["test", "-r", "-p", "1", "-t", "1", "configurations"] }
test_case! { transitions, "txt", vec!["test", "-p", "1", "-t", "1", "transitions"] }
test_case! { path, "txt", vec!["test", "-p", "1", "-t", "1", "path", "INIT", "2MSG", "INIT"] }
test_case! { sequence, "uml", vec!["test", "-p", "1", "-t", "1", "sequence", "INIT", "2MSG", "INIT"] }
test_case! { client_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "Client"] }
test_case! { client_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "Client", "-c"] }
test_case! { server_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "Server"] }
test_case! { server_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "Server", "-c"] }
