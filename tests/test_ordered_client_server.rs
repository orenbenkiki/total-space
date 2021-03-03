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
    Wait1,
    Wait2,
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

impl AgentState<ClientState, Payload> for ClientState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Idle => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait1,
                Emit::Ordered(Payload::Request, 1),
            )),
            Self::Wait1 => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait2,
                Emit::Ordered(Payload::Request, 1),
            )),
            _ => Reaction::Ignore,
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Wait2, Payload::Response) => Reaction::Do1(Action::Change(Self::Wait1)),
            (Self::Wait1, Payload::Response) => Reaction::Do1(Action::Change(Self::Idle)),
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
                Emit::Ordered(Payload::Response, 0),
            )),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Listen, Payload::Request) => Reaction::Do1(Action::Change(Self::Work)),
            (Self::Work, Payload::Request) => Reaction::Defer,
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn is_deferring(&self) -> bool {
        self == &Self::Work
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
    let app = add_clap(App::new("agents"));
    let arg_matches = app.get_matches_from(vec!["test", "agents"].iter());
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
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
    let app = add_clap(App::new("configurations"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "configurations"].iter());
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
            stdout,
            "\
            Client:Idle & Server:Listen\n\
            Client:Wait1 & Server:Listen | Client -> @0 Request -> Server\n\
            Client:Wait2 & Server:Listen | Client -> @0 Request -> Server & Client -> @1 Request -> Server\n\
            Client:Wait2 & Server:Work | Client -> @0 Request -> Server\n\
            Client:Wait2 & Server:Listen | Client -> @0 Request -> Server & Server -> @0 Response -> Client\n\
            Client:Wait2 & Server:Work | Server -> @0 Response -> Client\n\
            Client:Wait2 & Server:Listen | Server -> @0 Response -> Client & Server -> @1 Response -> Client\n\
            Client:Wait1 & Server:Work\n\
            Client:Wait1 & Server:Listen | Server -> @0 Response -> Client\n\
            "
        );
}

#[test]
fn test_transitions() {
    let mut model = test_model();
    let app = add_clap(App::new("transitions"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "transitions"].iter());
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
            stdout,
            "\
            FROM Client:Idle & Server:Listen\n\
            - BY time event\n  \
              TO Client:Wait1 & Server:Listen | Client -> @0 Request -> Server\n\
            FROM Client:Wait1 & Server:Listen | Client -> @0 Request -> Server\n\
            - BY time event\n  \
              TO Client:Wait2 & Server:Listen | Client -> @0 Request -> Server & Client -> @1 Request -> Server\n\
            - BY message Client -> @0 Request -> Server\n  \
              TO Client:Wait1 & Server:Work\n\
            FROM Client:Wait2 & Server:Listen | Client -> @0 Request -> Server & Client -> @1 Request -> Server\n\
            - BY message Client -> @0 Request -> Server\n  \
              TO Client:Wait2 & Server:Work | Client -> @0 Request -> Server\n\
            FROM Client:Wait2 & Server:Work | Client -> @0 Request -> Server\n\
            - BY time event\n  \
              TO Client:Wait2 & Server:Listen | Client -> @0 Request -> Server & Server -> @0 Response -> Client\n\
            FROM Client:Wait2 & Server:Listen | Client -> @0 Request -> Server & Server -> @0 Response -> Client\n\
            - BY message Client -> @0 Request -> Server\n  \
              TO Client:Wait2 & Server:Work | Server -> @0 Response -> Client\n\
            - BY message Server -> @0 Response -> Client\n  \
              TO Client:Wait1 & Server:Listen | Client -> @0 Request -> Server\n\
            FROM Client:Wait2 & Server:Work | Server -> @0 Response -> Client\n\
            - BY time event\n  \
              TO Client:Wait2 & Server:Listen | Server -> @0 Response -> Client & Server -> @1 Response -> Client\n\
            - BY message Server -> @0 Response -> Client\n  \
              TO Client:Wait1 & Server:Work\n\
            FROM Client:Wait2 & Server:Listen | Server -> @0 Response -> Client & Server -> @1 Response -> Client\n\
            - BY message Server -> @0 Response -> Client\n  \
              TO Client:Wait1 & Server:Listen | Server -> @0 Response -> Client\n\
            FROM Client:Wait1 & Server:Work\n\
            - BY time event\n  \
              TO Client:Wait2 & Server:Work | Client -> @0 Request -> Server\n\
            - BY time event\n  \
              TO Client:Wait1 & Server:Listen | Server -> @0 Response -> Client\n\
            FROM Client:Wait1 & Server:Listen | Server -> @0 Response -> Client\n\
            - BY time event\n  \
              TO Client:Wait2 & Server:Listen | Client -> @0 Request -> Server & Server -> @0 Response -> Client\n\
            - BY message Server -> @0 Response -> Client\n  \
              TO Client:Idle & Server:Listen\n\
            ",
        );
}
