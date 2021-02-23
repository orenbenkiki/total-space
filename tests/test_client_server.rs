extern crate total_space;

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
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
    fn default() -> Self {
        Self::Idle
    }
}

impl AgentState<ClientState, Payload> for ClientState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait,
                Emit::Unordered(Payload::Request, 0),
            )),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Wait, Payload::Response) => Reaction::Do1(Action::Change(Self::Idle)),
            _ => panic!("unexpected"),
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
    fn default() -> Self {
        Self::Listen
    }
}

impl AgentState<ServerState, Payload> for ServerState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Listen => Reaction::Ignore,
            Self::Work => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, 1),
            )),
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
            (Self::Listen, Payload::Request) => Reaction::Do1(Action::Change(Self::Work)),
            (Self::Work, Payload::Request) => Reaction::Defer,
            _ => panic!("unexpected"),
        }
    }

    fn is_deferring(&self) -> bool {
        self == &Self::Work
    }
}

type TestModel = Model<
    u8,      // StateId
    u8,      // MessageId
    u8,      // InvalidId
    u32,     // ConfigurationId
    Payload, // Payload
    19,      // MAX_AGENTS
    37,      // MAX_MESSAGES
>;

#[test]
fn test_configurations() {
    let client_type = AgentTypeData::<ClientState, <TestModel as MetaModel>::StateId, Payload>::new(
        "Client", false, 1,
    );
    let server_type = AgentTypeData::<ServerState, <TestModel as MetaModel>::StateId, Payload>::new(
        "Client", false, 1,
    );
    let types: Vec<<TestModel as MetaModel>::AgentTypeArc> =
        vec![Arc::new(client_type), Arc::new(server_type)];
    let _model = TestModel::new(types, vec![], 1);
}
