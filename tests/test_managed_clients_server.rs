mod common;

use clap::App;
use clap::ArgMatches;
use lazy_static::*;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::str;
use std::sync::Arc;
use strum::IntoStaticStr;
use total_space::*;

declare_global_agent_indices! { CLIENTS }
declare_global_agent_index! { MANAGER }
declare_global_agent_index! { SERVER }

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Need,
    Worry,
    Completed,
    Check { client: usize },
    Confirm,
    Request { client: usize },
    Response,
}
impl_data_like! {
    Payload = Self::Confirm,
    "Check" => "CHK",
    "Confirm" => "CNF",
    "Request" => "REQ",
    "Response" => "RSP",
    "client" => "C"
}
// END MAYBE TESTED

impl Validated for Payload {}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ManagerState {
    Fixed,
}
impl_data_like! {
    ManagerState = Self::Fixed,
    "Fixed" => ""
}
// END MAYBE TESTED

impl Validated for ManagerState {}

impl ContainerOf1State<ManagerState, ClientState, Payload> for ManagerState {
    fn receive_message(
        &self,
        _instance: usize,
        payload: &Payload,
        clients: &[ClientState],
    ) -> Reaction<Self, Payload> {
        match payload // MAYBE TESTED
        {
            Payload::Check { client } if clients[*client] == ClientState::Check => {
                Reaction::Do1(Action::Send1(Emit::Immediate(
                    Payload::Confirm,
                    agent_index!(CLIENTS[*client]),
                )))
            }
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn max_in_flight_messages(&self, clients: &[ClientState]) -> Option<usize> {
        Some(clients.len())
    }
}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Check,
    Wait,
}
impl_data_like! {
    ClientState = Self::Idle,
    "Idle" => "IDL",
    "Wait" => "WAT",
    "Check" => "CHK"
}
// END MAYBE TESTED

impl Validated for ClientState {}

impl AgentState<ClientState, Payload> for ClientState {
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            Self::Idle => Activity::Process1Of2(Payload::Need, Payload::Worry),
            _ => Activity::Passive,
        }
    }

    fn receive_message(&self, instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Idle, Payload::Need) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait,
                Emit::Unordered(Payload::Request { client: instance }, agent_index!(SERVER)),
            )),
            (Self::Idle, Payload::Worry) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Check,
                Emit::Unordered(Payload::Check { client: instance }, agent_index!(MANAGER)),
            )),
            (Self::Wait, Payload::Response) => Reaction::Do1(Action::Change(Self::Idle)),
            (Self::Check, Payload::Confirm) => Reaction::Do1(Action::Change(Self::Idle)),
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
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            Self::Listen => Activity::Passive,
            Self::Work { .. } => Activity::Process1(Payload::Completed),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Listen, Payload::Request { client }) => {
                Reaction::Do1(Action::Change(Self::Work { client: *client }))
            }
            (Self::Work { client }, Payload::Completed) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, agent_index!(CLIENTS[*client])),
            )),
            (Self::Work { .. }, Payload::Request { .. }) => Reaction::Defer,
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn is_deferring(&self) -> bool {
        matches!(self, &Self::Work { .. })
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
    const CLIENTS_COUNT: usize = 2;
    let client_type = Arc::new(AgentTypeData::<ClientState, StateId, Payload>::new(
        "C",
        Instances::Count(CLIENTS_COUNT),
        None,
    ));
    let manager_type = Arc::new(ContainerOf1TypeData::<
        ManagerState,
        ClientState,
        StateId,
        Payload,
        CLIENTS_COUNT,
    >::new(
        "MGR",
        Instances::Singleton,
        client_type.clone(),
        client_type.clone(),
    ));
    let server_type = Arc::new(AgentTypeData::<ServerState, StateId, Payload>::new(
        "SRV",
        Instances::Singleton,
        Some(manager_type.clone()),
    ));
    let model = TestModel::new(model_size(arg_matches, 1), server_type, vec![]);
    init_global_agent_indices!(CLIENTS, "C", model);
    init_global_agent_index!(MANAGER, "MGR", model);
    init_global_agent_index!(SERVER, "SRV", model);
    model
}

test_case! { agents, "txt", vec!["test", "agents"] }
test_case! { configurations, "txt", vec!["test", "-r", "-p", "1", "-t", "1", "configurations"] }
test_case! { transitions, "txt", vec!["test", "-p", "1", "-t", "1", "transitions"] }
test_case! { path, "txt", vec!["test", "-p", "1", "-t", "1", "path", "INIT", "2MSG", "INIT"] }
test_case! { sequence, "uml", vec!["test", "-p", "1", "-t", "1", "sequence", "INIT", "2MSG", "INIT"] }
test_case! { client_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C(0)"] }
test_case! { client_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "C(0)", "-c"] }
test_case! { server_states, "dot", vec!["test", "-p", "1", "-t", "1", "states", "SRV"] }
test_case! { server_states_condensed, "dot", vec!["test", "-p", "1", "-t", "1", "states", "SRV", "-c"] }
