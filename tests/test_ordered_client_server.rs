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
    Completed,
    Request(usize),
    Response(usize),
}
impl_enum_data! { Payload = Self::Request(1) }
// END MAYBE TESTED

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait(usize),
}
impl_enum_data! { ClientState = Self::Idle }
// END MAYBE TESTED

impl AgentState<ClientState, Payload> for ClientState {
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            Self::Idle => Activity::Process1(Payload::Need),
            Self::Wait(1) => Activity::Process1(Payload::Need),
            _ => Activity::Passive,
        }
    }

    fn reaction(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Idle, Payload::Need) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait(1),
                Emit::Ordered(Payload::Request(1), agent_index!(SERVER)),
            )),
            (Self::Wait(1), Payload::Need) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait(3),
                Emit::Ordered(Payload::Request(2), agent_index!(SERVER)),
            )),
            (Self::Wait(mask), Payload::Response(index)) if mask == index => {
                Reaction::Do1(Action::Change(Self::Idle))
            }
            (Self::Wait(mask), Payload::Response(index)) if mask & index != 0 => {
                Reaction::Do1(Action::Change(Self::Wait(mask & !index)))
            }
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn max_in_flight_messages(&self, _instance: usize) -> Option<usize> {
        Some(2)
    }
}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    Listen,
    Work(usize),
}
impl_enum_data! { ServerState = Self::Listen }
// END MAYBE TESTED

impl AgentState<ServerState, Payload> for ServerState {
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            Self::Listen => Activity::Passive,
            Self::Work(_index) => Activity::Process1(Payload::Completed),
        }
    }

    fn reaction(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Listen, Payload::Request(index)) => {
                Reaction::Do1(Action::Change(Self::Work(*index)))
            }
            (Self::Work(index), Payload::Completed) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Ordered(Payload::Response(*index), agent_index!(CLIENT)),
            )),
            (Self::Work(_), Payload::Request(_)) => Reaction::Defer,
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn is_deferring(&self, _instance: usize) -> bool {
        matches!(self, Self::Work(_))
    }

    fn max_in_flight_messages(&self, _instance: usize) -> Option<usize> {
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
    38, // MAX_TRANSITIONS
>;

fn test_model(arg_matches: &ArgMatches) -> TestModel {
    let client_type = Rc::new(AgentTypeData::<
        ClientState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new("Client", Instances::Singleton, None));
    let server_type = Rc::new(AgentTypeData::<
        ServerState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new(
        "Server", Instances::Singleton, Some(client_type.clone())
    ));

    init_agent_index!(CLIENT, client_type);
    init_agent_index!(SERVER, server_type);

    TestModel::new(model_size(arg_matches, 1), server_type)
}

test_case! { agents, "txt", vec!["test", "agents"] }
test_case! { configurations, "txt", vec!["test", "-r", "-p", "1", "configurations"] }
test_case! { transitions, "txt", vec!["test", "-p", "1", "transitions"] }
test_case! { path, "txt", vec!["test", "-p", "1", "path", "INIT", "2MSG", "INIT"] }
test_case! { sequence, "uml", vec!["test", "-p", "1", "sequence", "INIT", "2MSG", "INIT"] }
test_case! { client_states, "dot", vec!["test", "-p", "1", "states", "Client"] }
test_case! { client_states_condensed, "dot", vec!["test", "-p", "1", "states", "Client", "-c"] }
test_case! { server_states, "dot", vec!["test", "-p", "1", "states", "Server"] }
test_case! { server_states_condensed, "dot", vec!["test", "-p", "1", "states", "Server", "-c"] }
