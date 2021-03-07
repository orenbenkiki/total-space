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

#[test]
fn test_agents() {
    let app = add_clap(App::new(test_name!()));
    let arg_matches = app.get_matches_from(vec!["test", "agents"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_configurations() {
    let app = add_clap(App::new(test_name!()));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "configurations"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_transitions() {
    let app = add_clap(App::new(test_name!()));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "transitions"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_states() {
    let app = add_clap(App::new(test_name!()));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "states", "Client"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "dot");
}

#[test]
fn test_states_names() {
    let app = add_clap(App::new(test_name!()));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "states", "-n", "Client"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "dot");
}
