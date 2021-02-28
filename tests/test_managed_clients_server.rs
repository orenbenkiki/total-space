extern crate total_space;

use clap::App;
use lazy_static::*;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::str;
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
    static ref MANAGER: RwLock<usize> = RwLock::new(usize::max_value());
    static ref SERVER: RwLock<usize> = RwLock::new(usize::max_value());
}

// BEGIN MAYBE TESTED

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Check { client: usize },
    Confirm,
    Request { client: usize },
    Response,
}

// END MAYBE TESTED

impl_name_for_into_static_str! {Payload}

impl_display_by_patched_debug! {Payload}
impl PatchDebug for Payload {
    fn patch_debug(string: String) -> String {
        string
            .replace("Check", "CHK")
            .replace("Confirm", "CNF")
            .replace("Request", "REQ")
            .replace("Response", "RSP")
            .replace("client", "C")
    }
}

impl Validated for Payload {}

// BEGIN MAYBE TESTED

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ManagerState {
    Fixed,
}

// END MAYBE TESTED

impl ContainerState<ManagerState, ClientState, Payload> for ManagerState {
    fn pass_time(&self, _instance: usize, _clients: &[ClientState]) -> Reaction<Self, Payload> {
        Reaction::Ignore
    }

    fn receive_message(
        &self,
        _instance: usize,
        payload: &Payload,
        clients: &[ClientState],
    ) -> Reaction<Self, Payload> {
        match payload {
            Payload::Check { client } if clients[*client] == ClientState::Check => {
                Reaction::Do1(Action::Send1(Emit::Unordered(
                    Payload::Confirm,
                    CLIENTS.read().unwrap()[*client],
                )))
            }
            _ => Reaction::Unexpected, // NOT TESTED
        }
    }

    fn max_in_flight_messages(&self, clients: &[ClientState]) -> Option<usize> {
        Some(clients.len())
    }
}

impl_name_for_into_static_str! {ManagerState}

impl_display_by_patched_debug! {ManagerState}
impl PatchDebug for ManagerState {
    fn patch_debug(string: String) -> String {
        string.replace("Idle", "IDL").replace("Wait", "WAT")
    }
}

impl Validated for ManagerState {}

impl Default for ManagerState {
    fn default() -> Self // NOT TESTED
    {
        Self::Fixed
    }
}

// BEGIN MAYBE TESTED

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Check,
    Wait,
}

// END MAYBE TESTED

impl_name_for_into_static_str! {ClientState}

impl_display_by_patched_debug! {ClientState}
impl PatchDebug for ClientState {
    fn patch_debug(string: String) -> String {
        string
            .replace("Idle", "IDL")
            .replace("Wait", "WAT")
            .replace("Check", "CHK")
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
            Self::Idle => Reaction::Do1Of2(
                Action::ChangeAndSend1(
                    Self::Wait,
                    Emit::Unordered(
                        Payload::Request { client: instance },
                        *SERVER.read().unwrap(),
                    ),
                ),
                Action::ChangeAndSend1(
                    Self::Check,
                    Emit::Unordered(
                        Payload::Check { client: instance },
                        *MANAGER.read().unwrap(),
                    ),
                ),
            ),
            _ => Reaction::Ignore,
        }
    }

    fn receive_message(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        match (self, payload) {
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
fn test_model() {
    const CLIENTS_COUNT: usize = 2;
    let client_type = Arc::new(AgentTypeData::<ClientState, StateId, Payload>::new(
        "C",
        Instances::Count(CLIENTS_COUNT),
        None,
    ));
    let manager_type = Arc::new(ContainerTypeData::<
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
    let mut model = TestModel::new(server_type, vec![]);
    {
        let mut clients = CLIENTS.write().unwrap();
        clients.push(model.agent_index("C", Some(0)));
        clients.push(model.agent_index("C", Some(1)));
    }
    *MANAGER.write().unwrap() = model.agent_index("MGR", None);
    *SERVER.write().unwrap() = model.agent_index("SRV", None);

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
            C(0)\n\
            C(1)\n\
            MGR\n\
            SRV\n\
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
            "C(0):IDL & C(1):IDL & MGR:Fixed & SRV:LST\n\
            C(0):WAT & C(1):IDL & MGR:Fixed & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:WRK(C=0) | C(1) -> REQ(C=1) -> SRV\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:LST | SRV -> RSP -> C(0) & SRV -> RSP -> C(1)\n\
            C(0):IDL & C(1):WAT & MGR:Fixed & SRV:LST | SRV -> RSP -> C(1)\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:WRK(C=0) | SRV -> RSP -> C(1)\n\
            C(0):WAT & C(1):IDL & MGR:Fixed & SRV:WRK(C=0)\n\
            C(0):WAT & C(1):CHK & MGR:Fixed & SRV:WRK(C=0) | C(1) -> CHK(C=1) -> MGR\n\
            C(0):WAT & C(1):CHK & MGR:Fixed & SRV:LST | SRV -> RSP -> C(0) & C(1) -> CHK(C=1) -> MGR\n\
            C(0):IDL & C(1):CHK & MGR:Fixed & SRV:LST | C(1) -> CHK(C=1) -> MGR\n\
            C(0):WAT & C(1):CHK & MGR:Fixed & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> CHK(C=1) -> MGR\n\
            C(0):WAT & C(1):CHK & MGR:Fixed & SRV:LST | C(0) -> REQ(C=0) -> SRV & MGR -> CNF -> C(1)\n\
            C(0):WAT & C(1):CHK & MGR:Fixed & SRV:WRK(C=0) | MGR -> CNF -> C(1)\n\
            C(0):WAT & C(1):CHK & MGR:Fixed & SRV:LST | SRV -> RSP -> C(0) & MGR -> CNF -> C(1)\n\
            C(0):IDL & C(1):CHK & MGR:Fixed & SRV:LST | MGR -> CNF -> C(1)\n\
            C(0):CHK & C(1):CHK & MGR:Fixed & SRV:LST | MGR -> CNF -> C(1) & C(0) -> CHK(C=0) -> MGR\n\
            C(0):CHK & C(1):IDL & MGR:Fixed & SRV:LST | C(0) -> CHK(C=0) -> MGR\n\
            C(0):CHK & C(1):WAT & MGR:Fixed & SRV:LST | C(1) -> REQ(C=1) -> SRV & C(0) -> CHK(C=0) -> MGR\n\
            C(0):CHK & C(1):WAT & MGR:Fixed & SRV:WRK(C=1) | C(0) -> CHK(C=0) -> MGR\n\
            C(0):CHK & C(1):WAT & MGR:Fixed & SRV:LST | SRV -> RSP -> C(1) & C(0) -> CHK(C=0) -> MGR\n\
            C(0):CHK & C(1):WAT & MGR:Fixed & SRV:LST | SRV -> RSP -> C(1) & MGR -> CNF -> C(0)\n\
            C(0):CHK & C(1):IDL & MGR:Fixed & SRV:LST | MGR -> CNF -> C(0)\n\
            C(0):CHK & C(1):WAT & MGR:Fixed & SRV:LST | C(1) -> REQ(C=1) -> SRV & MGR -> CNF -> C(0)\n\
            C(0):CHK & C(1):WAT & MGR:Fixed & SRV:WRK(C=1) | MGR -> CNF -> C(0)\n\
            C(0):IDL & C(1):WAT & MGR:Fixed & SRV:WRK(C=1)\n\
            C(0):WAT & C(1):WAT & MGR:Fixed & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
            C(0):IDL & C(1):WAT & MGR:Fixed & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
            C(0):CHK & C(1):CHK & MGR:Fixed & SRV:LST | C(1) -> CHK(C=1) -> MGR & MGR -> CNF -> C(0)\n\
            C(0):CHK & C(1):CHK & MGR:Fixed & SRV:LST | MGR -> CNF -> C(1) & MGR -> CNF -> C(0)\n\
            C(0):CHK & C(1):CHK & MGR:Fixed & SRV:LST | C(1) -> CHK(C=1) -> MGR & C(0) -> CHK(C=0) -> MGR\n\
            C(0):WAT & C(1):IDL & MGR:Fixed & SRV:LST | SRV -> RSP -> C(0)\n\
            "
        );
    }
}
