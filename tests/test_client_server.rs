extern crate total_space;

use clap::App;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::str;
use std::sync::Arc;
use strum::IntoStaticStr;
use total_space::*;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Ping,
    Request,
    Response,
}

impl_name_for_into_static_str! {Payload}

impl Display for Payload {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "{}", self.name())
    }
}

impl Validated for Payload {}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait,
}

impl_name_for_into_static_str! {ClientState}

impl Display for ClientState {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "{}", self.name())
    }
}

impl Validated for ClientState {}

impl Default for ClientState {
    fn default() -> Self // NOT TESTED
    {
        Self::Idle
    }
}

fn is_maybe_ping(payload: Option<Payload>) -> bool {
    match payload {
        None => true,
        Some(Payload::Ping) => true,
        _ => false, // NOT TESTED
    }
}

impl AgentState<ClientState, Payload> for ClientState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1Of2(
                Action::ChangeAndSend1(Self::Wait, Emit::Unordered(Payload::Request, 1)),
                Action::Send1(Emit::ImmediateReplacement(is_maybe_ping, Payload::Ping, 1)),
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

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    Listen,
    Work,
}

impl_name_for_into_static_str! {ServerState}

impl Display for ServerState {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
        write!(formatter, "{}", self.name())
    }
}

impl Validated for ServerState {}

impl Default for ServerState {
    fn default() -> Self // NOT TESTED
    {
        Self::Listen
    }
}

impl AgentState<ServerState, Payload> for ServerState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Listen => Reaction::Ignore,
            Self::Work => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, 0),
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

    fn is_deferring(&self) -> bool // NOT TESTED
    {
        self == &Self::Work // NOT TESTED
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

fn test_model() -> TestModel {
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

    TestModel::new(server_type, vec![])
}

#[test]
fn test_agents() {
    let mut model = test_model();
    let app = add_clap(App::new("test_agents"));
    let arg_matches = app.get_matches_from(vec!["test", "conditions"].iter());
    let mut stdout_bytes = Vec::new();
    assert!(model.do_clap(&arg_matches, &mut stdout_bytes));
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(stdout, "init: matches the initial configuration\n");
}

#[test]
fn test_conditions() {
    let mut model = test_model();
    let app = add_clap(App::new("test_conditions"));
    let arg_matches = app.get_matches_from(vec!["test", "agents"].iter());
    let mut stdout_bytes = Vec::new();
    assert!(model.do_clap(&arg_matches, &mut stdout_bytes));
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        Client\n\
        Server\n\
        "
    );
}

#[test]
fn test_configurations() {
    let mut model = test_model();
    let app = add_clap(App::new("test_configurations"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "configurations"].iter());
    let mut stdout_bytes = Vec::new();
    assert!(model.do_clap(&arg_matches, &mut stdout_bytes));
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
            stdout,
            "Client:Idle & Server:Listen\n\
            Client:Wait & Server:Listen | Client -> Request -> Server\n\
            Client:Wait & Server:Work\n\
            Client:Wait & Server:Listen | Server -> Response -> Client\n\
            Client:Idle & Server:Listen | Client -> * Ping -> Server\n\
            Client:Wait & Server:Listen | Client -> Request -> Server & Client -> * Ping -> Server\n\
            Client:Idle & Server:Listen | Client -> * Ping => Ping -> Server\n\
            Client:Wait & Server:Listen | Client -> Request -> Server & Client -> * Ping => Ping -> Server\n\
            "
        );
}

#[test]
fn test_transitions() {
    let mut model = test_model();
    let app = add_clap(App::new("test_transitions"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "transitions"].iter());
    let mut stdout_bytes = Vec::new();
    assert!(model.do_clap(&arg_matches, &mut stdout_bytes));
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
            stdout,
            "FROM Client:Idle & Server:Listen\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server\n\
            - BY time event\n  \
              TO Client:Idle & Server:Listen | Client -> * Ping -> Server\n\
            FROM Client:Wait & Server:Listen | Client -> Request -> Server\n\
            - BY message Client -> Request -> Server\n  \
              TO Client:Wait & Server:Work\n\
            FROM Client:Wait & Server:Work\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Server -> Response -> Client\n\
            FROM Client:Wait & Server:Listen | Server -> Response -> Client\n\
            - BY message Server -> Response -> Client\n  \
              TO Client:Idle & Server:Listen\n\
            FROM Client:Idle & Server:Listen | Client -> * Ping -> Server\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server & Client -> * Ping -> Server\n\
            - BY time event\n  \
              TO Client:Idle & Server:Listen | Client -> * Ping => Ping -> Server\n\
            - BY message Client -> * Ping -> Server\n  \
              TO Client:Idle & Server:Listen\n\
            FROM Client:Wait & Server:Listen | Client -> Request -> Server & Client -> * Ping -> Server\n\
            - BY message Client -> * Ping -> Server\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server\n\
            FROM Client:Idle & Server:Listen | Client -> * Ping => Ping -> Server\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server & Client -> * Ping => Ping -> Server\n\
            - BY time event\n  \
              TO Client:Idle & Server:Listen | Client -> * Ping => Ping -> Server\n\
            - BY message Client -> * Ping => Ping -> Server\n  \
              TO Client:Idle & Server:Listen\n\
            FROM Client:Wait & Server:Listen | Client -> Request -> Server & Client -> * Ping => Ping -> Server\n\
            - BY message Client -> * Ping => Ping -> Server\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server\n\
            "
        );
}
