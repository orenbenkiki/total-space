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

impl AgentState<ClientState, Payload> for ClientState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait,
                Emit::Unordered(Payload::Request, 1),
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
        match self {
            Self::Wait => Some(1),
            Self::Idle => Some(0),
        }
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
        match self {
            Self::Listen => Some(1),
            Self::Work => Some(0),
        }
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

#[test]
fn test_model() {
    let client_type = AgentTypeData::<ClientState, <TestModel as MetaModel>::StateId, Payload>::new(
        "Client", false, 1,
    );
    let server_type = AgentTypeData::<ServerState, <TestModel as MetaModel>::StateId, Payload>::new(
        "Server", false, 1,
    );
    let types: Vec<<TestModel as MetaModel>::AgentTypeArc> =
        vec![Arc::new(client_type), Arc::new(server_type)];
    let mut model = TestModel::new(types, vec![]);
    model.eprint_progress = true;
    model.threads = Threads::Count(1);

    {
        let app = add_clap_subcommands(App::new("test_client_server_model"));
        let mut arg_matches = app.get_matches_from(vec!["test", "agents"].iter());
        let mut stdout_bytes = Vec::new();
        assert!(model.do_clap_subcommand(&mut arg_matches, &mut stdout_bytes));
        let stdout = str::from_utf8(&stdout_bytes).unwrap();
        assert_eq!(
            stdout,
            "\
            Client\n\
            Server\n\
            "
        );
    }

    {
        let app = add_clap_subcommands(App::new("test_client_server_model"));
        let mut arg_matches = app.get_matches_from(vec!["test", "configurations"].iter());
        let mut stdout_bytes = Vec::new();
        assert!(model.do_clap_subcommand(&mut arg_matches, &mut stdout_bytes));
        let stdout = str::from_utf8(&stdout_bytes).unwrap();
        assert_eq!(
            stdout,
            "\
            Client:Idle & Server:Listen\n\
            Client:Wait & Server:Listen | Client -> Request -> Server\n\
            Client:Wait & Server:Work\n\
            Client:Wait & Server:Listen | Server -> Response -> Client\n\
            "
        );
    }
}
