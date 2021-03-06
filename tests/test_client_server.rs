mod common;

use clap::App;
use clap::ArgMatches;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::rc::Rc;
use std::str;
use strum::IntoStaticStr;
use total_space::*;

declare_agent_index! {CLIENT}
declare_agent_index! {SERVER}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Need,
    Worry,
    Completed,
    Ping,
    Request,
    Response,
}
impl_enum_data! {
    Payload = Self::Ping,
    "Ping" => "PNG",
    "Request" => "REQ",
    "Response" => "RSP"
}
// END MAYBE TESTED

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait,
}
impl_enum_data! {
    ClientState = Self::Idle,
    "Idle" => "IDL",
    "Wait" => "WAT"
}
// END MAYBE TESTED

fn is_maybe_ping(payload: Option<Payload>) -> bool {
    matches!(payload, None | Some(Payload::Ping))
}

impl AgentState<ClientState, Payload> for ClientState {
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            Self::Wait => Activity::Passive,
            Self::Idle => activity_alternatives!(Payload::Need, Payload::Worry),
        }
    }

    fn reaction(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Idle, Payload::Need) => Reaction::Do1(action_change_and_sends!(
                Self::Wait,
                Emit::Unordered(Payload::Request, agent_index!(SERVER)),
                Emit::ImmediateReplacement(is_maybe_ping, Payload::Ping, agent_index!(SERVER)),
            )),

            (Self::Idle, Payload::Worry) => Reaction::Do1(Action::Send1(
                Emit::UnorderedReplacement(is_maybe_ping, Payload::Ping, agent_index!(SERVER)),
            )),

            (Self::Wait, Payload::Response) => Reaction::Do1(Action::Change(Self::Idle)),

            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn max_in_flight_messages(&self, _instance: usize) -> Option<usize> {
        Some(2)
    }
}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    Listen,
    Work,
}
impl_enum_data! { ServerState = Self::Listen, "Listen" => "LST", "Work" => "WRK" }
// END MAYBE TESTED

impl AgentState<ServerState, Payload> for ServerState {
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            Self::Listen => Activity::Passive,
            Self::Work => Activity::Process1(Payload::Completed),
        }
    }

    fn reaction(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (_, Payload::Ping) => Reaction::Ignore,
            (Self::Listen, Payload::Request) => Reaction::Do1(Action::Change(Self::Work)),
            (Self::Work, Payload::Completed) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, agent_index!(CLIENT)),
            )),
            (Self::Work, Payload::Request) => Reaction::Defer, // NOT TESTED
            _ => Reaction::Unexpected,                         // NOT TESTED
        }
    }

    fn is_deferring(&self, _instance: usize) -> bool {
        self == &Self::Work
    }

    fn max_in_flight_messages(&self, _instance: usize) -> Option<usize> {
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
    let client_type = Rc::new(AgentTypeData::<
        ClientState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new("C", Instances::Singleton, None));

    let server_type = Rc::new(AgentTypeData::<
        ServerState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new(
        "S", Instances::Singleton, Some(client_type.clone())
    ));

    init_agent_index!(CLIENT, client_type);
    init_agent_index!(SERVER, server_type);

    TestModel::new(model_size(arg_matches, 1), server_type)
}

test_case! { conditions, "txt", vec!["test", "conditions"] }
test_case! { legend, "txt", vec!["test", "legend"] }
test_case! { agents, "txt", vec!["test", "agents"] }
test_case! { compute, "txt", vec!["test", "-r", "compute"] }
test_case! { configurations, "txt", vec!["test", "-r", "-p", "1", "configurations"] }
test_case! { transitions, "txt", vec!["test", "transitions"] }
test_case! { abort, "txt", vec!["test", "-p", "1", "path", "INIT", "!INIT"] }
test_case! { path, "txt", vec!["test", "-p", "1", "path", "INIT", "2MSG", "INIT", "UNORDERED_REPLACEMENT", "IMMEDIATE_REPLACEMENT", "INIT"] }
test_case! { sequence, "uml", vec!["test", "-p", "1", "sequence", "-h", "INIT", "2MSG", "INIT", "UNORDERED_REPLACEMENT", "IMMEDIATE_REPLACEMENT", "INIT"] }
test_case! { client_states, "dot", vec!["test", "-p", "1", "states", "C"] }
test_case! { client_states_names, "dot", vec!["test", "-p", "1", "states", "C", "-n"] }
test_case! { client_states_final, "dot", vec!["test", "-p", "1", "states", "C", "-f"] }
test_case! { client_states_condensed, "dot", vec!["test", "-p", "1", "states", "C", "-c"] }
test_case! { server_states, "dot", vec!["test", "-p", "1", "states", "S"] }
test_case! { server_states_names, "dot", vec!["test", "-p", "1", "states", "S", "-n"] }
test_case! { server_states_final, "dot", vec!["test", "-p", "1", "states", "S", "-f"] }
test_case! { server_states_condensed, "dot", vec!["test", "-p", "1", "states", "S", "-c"] }
