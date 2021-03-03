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
        Some(1)
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

fn test_model() -> TestModel {
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
    let model = TestModel::new(server_type, vec![]);
    if CLIENTS.read().unwrap().len() == 0 {
        let mut clients = CLIENTS.write().unwrap();
        clients.push(model.agent_index("C", Some(0)));
        clients.push(model.agent_index("C", Some(1)));
        *SERVER.write().unwrap() = model.agent_index("SRV", None);
    }
    model
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
        C(0)\n\
        C(1)\n\
        SRV\n\
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
        C(0):IDL & C(1):IDL & SRV:LST\n\
        C(0):WAT & C(1):IDL & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
        C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=0) | C(1) -> REQ(C=1) -> SRV\n\
        C(0):WAT & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
        C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(0) & SRV -> RSP -> C(1)\n\
        C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
        C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=0) | SRV -> RSP -> C(1)\n\
        C(0):WAT & C(1):IDL & SRV:WRK(C=0)\n\
        C(0):WAT & C(1):IDL & SRV:LST | SRV -> RSP -> C(0)\n\
        C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
        C(0):WAT & C(1):WAT & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
        C(0):IDL & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
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
            FROM C(0):WAT & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV & SRV -> RSP -> C(0)\n\
            - BY message C(1) -> REQ(C=1) -> SRV\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
            - BY message SRV -> RSP -> C(0)\n  \
              TO C(0):IDL & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
            FROM C(0):WAT & C(1):WAT & SRV:WRK(C=1) | SRV -> RSP -> C(0)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(0) & SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(0)\n  \
              TO C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
            FROM C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(0) & SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(0)\n  \
              TO C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):WAT & C(1):IDL & SRV:LST | SRV -> RSP -> C(0)\n\
            FROM C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):IDL & C(1):IDL & SRV:LST\n\
            FROM C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
            - BY message C(0) -> REQ(C=0) -> SRV\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=0) | SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):WAT & C(1):IDL & SRV:LST | C(0) -> REQ(C=0) -> SRV\n\
            FROM C(0):WAT & C(1):WAT & SRV:WRK(C=0) | SRV -> RSP -> C(1)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(0) & SRV -> RSP -> C(1)\n\
            - BY message SRV -> RSP -> C(1)\n  \
              TO C(0):WAT & C(1):IDL & SRV:WRK(C=0)\n\
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
            FROM C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
            - BY time event\n  \
              TO C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
            FROM C(0):WAT & C(1):WAT & SRV:WRK(C=1) | C(0) -> REQ(C=0) -> SRV\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & SRV -> RSP -> C(1)\n\
            FROM C(0):IDL & C(1):WAT & SRV:LST | C(1) -> REQ(C=1) -> SRV\n\
            - BY time event\n  \
              TO C(0):WAT & C(1):WAT & SRV:LST | C(0) -> REQ(C=0) -> SRV & C(1) -> REQ(C=1) -> SRV\n\
            - BY message C(1) -> REQ(C=1) -> SRV\n  \
              TO C(0):IDL & C(1):WAT & SRV:WRK(C=1)\n\
            "
        );
}

#[test]
fn test_states() {
    let mut model = test_model();
    let app = add_clap(App::new("states"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "states", "SRV"].iter());
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        digraph {\n\
        color=white;\n\
        graph [ fontname=\"sans-serif\" ];\n\
        node [ fontname=\"sans-serif\" ];\n\
        edge [ fontname=\"sans-serif\" ];\n\
        A_0_false [ label=\"LST\", shape=ellipse ];\n\
        A_1_true [ label=\"WRK(C=0)\", shape=octagon ];\n\
        subgraph cluster_0 {\n\
        T_0 [ shape=point, height=0.015, width=0.015 ];\n\
        M_0_0 [ label=\"C(0) &#8594;\\nREQ(C=0)\", shape=plain ];\n\
        M_0_0 -> T_0 [ arrowhead=normal, direction=forward, style=dashed ];\n\
        }\n\
        A_0_false -> T_0 [ arrowhead=none, direction=forward ];\n\
        T_0 -> A_1_true;\n\
        A_2_true [ label=\"WRK(C=1)\", shape=octagon ];\n\
        subgraph cluster_1 {\n\
        T_1 [ shape=point, height=0.015, width=0.015 ];\n\
        M_1_1 [ label=\"C(1) &#8594;\\nREQ(C=1)\", shape=plain ];\n\
        M_1_1 -> T_1 [ arrowhead=normal, direction=forward, style=dashed ];\n\
        }\n\
        A_0_false -> T_1 [ arrowhead=none, direction=forward ];\n\
        T_1 -> A_2_true;\n\
        subgraph cluster_2 {\n\
        T_2 [ shape=point, height=0.015, width=0.015 ];\n\
        M_2_2 [ label=\"RSP\\n&#8594; C(0)\", shape=plain ];\n\
        T_2 -> M_2_2 [ arrowhead=normal, direction=forward, style=dashed ];\n\
        M_2_255 [ label=\"Time\", shape=plain ];\n\
        M_2_255 -> T_2 [ arrowhead=normal, direction=forward, style=dashed ];\n\
        }\n\
        A_1_true -> T_2 [ arrowhead=none, direction=forward ];\n\
        T_2 -> A_0_false;\n\
        subgraph cluster_3 {\n\
        T_3 [ shape=point, height=0.015, width=0.015 ];\n\
        M_3_3 [ label=\"RSP\\n&#8594; C(1)\", shape=plain ];\n\
        T_3 -> M_3_3 [ arrowhead=normal, direction=forward, style=dashed ];\n\
        M_3_255 [ label=\"Time\", shape=plain ];\n\
        M_3_255 -> T_3 [ arrowhead=normal, direction=forward, style=dashed ];\n\
        }\n\
        A_2_true -> T_3 [ arrowhead=none, direction=forward ];\n\
        T_3 -> A_0_false;\n\
        }\n\
        "
    );
}

#[test]
fn test_path() {
    let mut model = test_model();
    let app = add_clap(App::new("path"));
    let arg_matches = app.get_matches_from(
        vec![
            "test", "-r", "-p", "-t", "1", "path", "1MSG", "2MSG", "INIT",
        ]
        .iter(),
    );
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
        TO C(0):WAT & C(1):WAT & SRV:LST | SRV -> RSP -> C(0) & SRV -> RSP -> C(1)\n\
        BY message SRV -> RSP -> C(0)\n\
        TO C(0):IDL & C(1):WAT & SRV:LST | SRV -> RSP -> C(1)\n\
        BY message SRV -> RSP -> C(1)\n\
        INIT C(0):IDL & C(1):IDL & SRV:LST\n\
        "
    );
}

#[test]
fn test_sequence() {
    let mut model = test_model();
    let app = add_clap(App::new("sequence"));
    let arg_matches = app.get_matches_from(
        vec![
            "test", "-r", "-p", "-t", "1", "sequence", "1MSG", "2MSG", "0MSG",
        ]
        .iter(),
    );
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
