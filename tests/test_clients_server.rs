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

declare_global_agent_indices! {CLIENTS}
declare_global_agent_index! {SERVER}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Request { client: usize },
    Response,
}
impl_data_like! {
    Payload = Self::Response,
    "client" => "C",
    "Request" => "REQ",
    "Response" => "RSP"
}
// END MAYBE TESTED

impl Validated for Payload {}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait,
}
impl_data_like! {
    ClientState = Self::Idle,
    "Idle" => "IDL",
    "Wait" => "WAT"
}
// END MAYBE TESTED

impl Validated for ClientState {}

impl AgentState<ClientState, Payload> for ClientState {
    fn pass_time(&self, instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait,
                Emit::Unordered(Payload::Request { client: instance }, agent_index!(SERVER)),
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
        Some(1)
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
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Listen => Reaction::Ignore,
            Self::Work { client } => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, agent_index!(CLIENTS[*client])),
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
    let client_type = Arc::new(AgentTypeData::<
        ClientState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new("C", Instances::Count(2), None));
    let server_type = Arc::new(AgentTypeData::<
        ServerState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new(
        "SRV", Instances::Singleton, Some(client_type.clone())
    ));
    let model = TestModel::new(model_size(arg_matches, 1), server_type, vec![]);
    init_global_agent_indices!(CLIENTS, "C", model);
    init_global_agent_index!(SERVER, "SRV", model);
    model
}

#[test]
fn test_agents() {
    let app = add_clap(App::new("agents"));
    let arg_matches = app.get_matches_from(vec!["test", "agents"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        C(0)\n\
        C(1)\n\
        SRV\n\
        "
    );
}

#[test]
fn test_configurations() {
    let app = add_clap(App::new("configurations"));
    let arg_matches = app
        .get_matches_from(vec!["test", "-r", "-p", "-s", "1", "-t", "1", "configurations"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        C(0):IDL & C(1):IDL & SRV:LST\n\
        C(0):WAT & C(1):IDL & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
        C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=0) | C(1) -> REQ(C=1) -> SRV\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
        C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=0) | SRV -> RSP -> C(1)\n\
        C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(1) & SRV -> RSP -> C(0)\n\
        C(0):WAT & C(1):IDL & SRV:WRK(C=0)\n\
        C(0):WAT & C(1):IDL & SRV:LST | SRV -> RSP -> C(0)\n\
        C(0):WAT & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
        C(0):IDL & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
        C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
        C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
        "
    );
}

#[test]
fn test_transitions() {
    let app = add_clap(App::new("transitions"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-s", "1", "-t", "1", "transitions"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
            stdout,
            "\
            FROM C(0):IDL & C(1):IDL & SRV:LST\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):IDL & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
            - BY time event\n  \
              TO C(0):IDL & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
            FROM C(0):WAT & C(1):IDL & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
            - BY message C(0) -> REQ(C=0) -> SRV\n  \
              TO C(0):WAT & C(1):IDL & SRV:WRK(C=0)\n\
            FROM C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
            - BY message C(0) -> REQ(C=0) -> SRV\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=0) | C(1) -> REQ(C=1) -> SRV\n\
            - BY message C(1) -> REQ(C=1) -> SRV\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
            FROM C(0):WAT & C(1):WAT & SRV:WRK(C=0) | C(1) -> REQ(C=1) -> SRV\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
            FROM C(0):WAT & C(1):WAT & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
            FROM C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
            - BY message C(0) -> REQ(C=0) -> SRV\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=0) | SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):WAT & C(1):IDL & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
            FROM C(0):WAT & C(1):WAT & SRV:WRK(C=0) | SRV -> RSP -> C(1)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(1) & SRV -> RSP -> C(0)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):WAT & C(1):IDL & SRV:WRK(C=0)\n\
            FROM C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(1) & SRV -> RSP -> C(0)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):WAT & C(1):IDL & SRV:LST | SRV -> RSP -> C(0)\n\
            - BY message SRV -> RSP -> C(0)\n  \
              TO C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
            FROM C(0):WAT & C(1):IDL & SRV:WRK(C=0)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=0) | C(1) -> REQ(C=1) -> SRV\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):IDL & SRV:LST | SRV -> RSP -> C(0)\n\
            FROM C(0):WAT & C(1):IDL & SRV:LST | SRV -> RSP -> C(0)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
            - BY message SRV -> RSP -> C(0)\n  \
              TO C(0):IDL & C(1):IDL & SRV:LST\n\
            FROM C(0):WAT & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
            - BY message C(1) -> REQ(C=1) -> SRV\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
            - BY message SRV -> RSP -> C(0)\n  \
              TO C(0):IDL & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
            FROM C(0):WAT & C(1):WAT & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(1) & SRV -> RSP -> C(0)\n\
            - BY message SRV -> RSP -> C(0)\n  \
              TO C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
            FROM C(0):IDL & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
            - BY message C(1) -> REQ(C=1) -> SRV\n  \
              TO C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
            FROM C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
            - BY time event\n  \
              TO C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
            FROM C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):IDL & C(1):IDL & SRV:LST\n\
            "
        );
}

#[test]
fn test_path() {
    let app = add_clap(App::new("path"));
    let arg_matches = app.get_matches_from(
        vec![
            "test", "-r", "-p", "-s", "1", "-t", "1", "path", "1MSG", "2MSG", "INIT",
        ]
        .iter(),
    );
    let mut model = test_model(&arg_matches);
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        1MSG C(0):WAT & C(1):IDL & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
        BY time event\n\
        TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
        BY message C(0) -> REQ(C=0) -> SRV\n\
        1MSG C(0):WAT & C(1):WAT & SRV:WRK(C=0) | C(1) -> REQ(C=1) -> SRV\n\
        BY time event\n\
        2MSG C(0):WAT & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
        BY message C(1) -> REQ(C=1) -> SRV\n\
        TO C(0):WAT & C(1):WAT & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
        BY time event\n\
        TO C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(1) & SRV -> RSP -> C(0)\n\
        BY message SRV -> RSP -> C(1)\n\
        TO C(0):WAT & C(1):IDL & SRV:LST | SRV -> RSP -> C(0)\n\
        BY message SRV -> RSP -> C(0)\n\
        INIT C(0):IDL & C(1):IDL & SRV:LST\n\
        "
    );
}

#[test]
fn test_sequence() {
    let app = add_clap(App::new("sequence"));
    let arg_matches = app.get_matches_from(
        vec![
            "test", "-r", "-p", "-s", "1", "-t", "1", "sequence", "1MSG", "2MSG", "0MSG",
        ]
        .iter(),
    );
    let mut model = test_model(&arg_matches);
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        @startuml\n\
        autonumber \" <b>#</b> \"\n\
        skinparam shadowing false\n\
        skinparam sequence {\n\
        ArrowColor Black\n\
        ActorBorderColor Black\n\
        LifeLineBorderColor Black\n\
        LifeLineBackgroundColor Black\n\
        ParticipantBorderColor Black\n\
        }\n\
        skinparam ControlBorderColor White\n\
        skinparam ControlBackgroundColor White\n\
        participant \"C(0)\" as A0 order 10100\n\
        activate A0 #CadetBlue\n\
        participant \"C(1)\" as A1 order 10200\n\
        activate A1 #CadetBlue\n\
        participant \"SRV\" as A2 order 10300\n\
        activate A2 #CadetBlue\n\
        control \" \" as T0 order 10101\n\
        activate T0 #Silver\n\
        rnote over A0 : WAT\n\
        / rnote over A1 : IDL\n\
        / rnote over A2 : LST\n\
        / rnote over T0 : REQ(C=0)\n\
        ?o-> A1\n\
        deactivate A1\n\
        control \" \" as T1 order 10201\n\
        A1 -> T1 : REQ(C=1)\n\
        activate T1 #Silver\n\
        rnote over A1 : WAT\n\
        activate A1 #MediumPurple\n\
        T0 -> A2 : REQ(C=0)\n\
        deactivate T0\n\
        deactivate A2\n\
        autonumber stop\n\
        ?-[#White]\\ A2\n\
        autonumber resume\n\
        rnote over A2 : WRK(C=0)\n\
        activate A2 #CadetBlue\n\
        ?o-> A2\n\
        deactivate A2\n\
        control \" \" as T2 order 10299\n\
        A2 -> T2 : RSP\n\
        activate T2 #Silver\n\
        rnote over A2 : LST\n\
        activate A2 #MediumPurple\n\
        T1 -> A2 : REQ(C=1)\n\
        deactivate T1\n\
        deactivate A2\n\
        autonumber stop\n\
        ?-[#White]\\ A2\n\
        autonumber resume\n\
        rnote over A2 : WRK(C=1)\n\
        activate A2 #CadetBlue\n\
        T2 -> A0 : RSP\n\
        deactivate T2\n\
        deactivate A0\n\
        autonumber stop\n\
        ?-[#White]\\ A0\n\
        autonumber resume\n\
        rnote over A0 : IDL\n\
        activate A0 #MediumPurple\n\
        @enduml\n\
        "
    );
}
