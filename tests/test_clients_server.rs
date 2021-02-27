extern crate total_space;

use lazy_static::*;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::sync::Arc;
use std::sync::RwLock;
use strum::IntoStaticStr;
use total_space::*;

index_type! { StateId, u8 }
index_type! { MessageId, u8 }
index_type! { InvalidId, u8 }
index_type! { ConfigurationId, u32 }

lazy_static! {
    static ref CLIENTS: RwLock<Vec<usize>> = RwLock::new(Vec::new());
    static ref SERVER: RwLock<usize> = RwLock::new(0);
}

// BEGIN MAYBE TESTED

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Request { client: usize },
    Response,
}

// END MAYBE TESTED

impl_name_for_into_static_str! {Payload}

impl_display_by_patched_debug! {Payload}
impl PatchDebug for Payload {
    fn patch_debug(string: String) -> String {
        string
            .replace("client", "C")
            .replace("Request", "REQ")
            .replace("Response", "RSP")
    }
}

impl Validated for Payload {}

// BEGIN MAYBE TESTED

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait,
}

// END MAYBE TESTED

impl_name_for_into_static_str! {ClientState}

impl_display_by_patched_debug! {ClientState}
impl PatchDebug for ClientState {
    fn patch_debug(string: String) -> String {
        string.replace("Idle", "IDL").replace("Wait", "WAT")
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
    fn pass_time(&self, instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait,
                Emit::Unordered(
                    Payload::Request { client: instance },
                    *SERVER.read().unwrap(),
                ),
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

// BEGIN MAYBE TESTED

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    Listen,
    Work { client: usize },
}

// END MAYBE TESTED

impl_name_for_into_static_str! {ServerState}

impl_display_by_patched_debug! {ServerState}
impl PatchDebug for ServerState {
    fn patch_debug(string: String) -> String {
        string
            .replace("client", "C")
            .replace("Listen", "LST")
            .replace("Work", "WRK")
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
            Self::Work { client } => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, CLIENTS.read().unwrap()[*client]),
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
        Some(CLIENTS.read().unwrap().len())
    }
}

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
fn test_configurations() {
    let client_type =
        AgentTypeData::<ClientState, <TestModel as MetaModel>::StateId, Payload>::new("C", true, 2);
    let server_type = AgentTypeData::<ServerState, <TestModel as MetaModel>::StateId, Payload>::new(
        "SRV", false, 1,
    );
    let types: Vec<<TestModel as MetaModel>::AgentTypeArc> =
        vec![Arc::new(client_type), Arc::new(server_type)];
    let mut model = TestModel::new(types, vec![]);
    model.eprint_progress = true;
    {
        let mut clients = CLIENTS.write().unwrap();
        clients.push(model.agent_index("C", Some(0)));
        clients.push(model.agent_index("C", Some(1)));
    }
    *SERVER.write().unwrap() = model.agent_index("SRV", None);
    model.compute(1);
}