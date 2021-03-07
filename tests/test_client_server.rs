mod common;

use clap::App;
use clap::ArgMatches;
use common::*;
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
    Ping,
    Request,
    Response,
}
impl_data_like! {
    Payload = Self::Ping,
    "Ping" => "PNG",
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

fn is_maybe_ping(payload: Option<Payload>) -> bool {
    matches!(payload, None | Some(Payload::Ping))
}

impl AgentState<ClientState, Payload> for ClientState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Wait => Reaction::Ignore,
            Self::Idle => Reaction::Do1Of2(
                Action::ChangeAndSend1(
                    Self::Wait,
                    Emit::Unordered(Payload::Request, agent_index!(SERVER)),
                ),
                Action::Send1(Emit::UnorderedReplacement(
                    is_maybe_ping,
                    Payload::Ping,
                    agent_index!(SERVER),
                )),
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

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    Listen,
    Work,
}
impl_data_like! { ServerState = Self::Listen, "Listen" => "LST", "Work" => "WRK" }
// END MAYBE TESTED

impl Validated for ServerState {}

impl AgentState<ServerState, Payload> for ServerState {
    fn pass_time(&self, _instance: usize) -> Reaction<Self, Payload> {
        match self {
            Self::Listen => Reaction::Ignore,
            Self::Work => Reaction::Do1(Action::ChangeAndSend1(
                Self::Listen,
                Emit::Unordered(Payload::Response, agent_index!(CLIENT)),
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

    fn is_deferring(&self) -> bool {
        self == &Self::Work
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

fn test_model(arg_matches: &ArgMatches) -> TestModel {
    let client_type = Arc::new(AgentTypeData::<
        ClientState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new("C", Instances::Singleton, None));

    let server_type = Arc::new(AgentTypeData::<
        ServerState,
        <TestModel as MetaModel>::StateId,
        Payload,
    >::new(
        "S", Instances::Singleton, Some(client_type.clone())
    ));

    let model = TestModel::new(model_size(arg_matches, 1), server_type, vec![]);
    init_global_agent_index!(CLIENT, "C", model);
    init_global_agent_index!(SERVER, "S", model);
    model
}

#[test]
fn test_conditions() {
    let app = add_clap(App::new(test_name!()));
    let arg_matches = app.get_matches_from(vec!["test", "conditions"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_agents() {
    let app = add_clap(App::new("test_conditions"));
    let arg_matches = app.get_matches_from(vec!["test", "agents"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_configurations() {
    let app = add_clap(App::new("test_configurations"));
    let arg_matches = app.get_matches_from(vec!["test", "-p", "-t", "1", "configurations"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_transitions() {
    let app = add_clap(App::new("test_transitions"));
    let arg_matches = app.get_matches_from(vec!["test", "-p", "-t", "1", "transitions"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_path() {
    let app = add_clap(App::new("test_path"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-p", "-t", "1", "path", "INIT", "INIT"].iter());
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "txt");
}

#[test]
fn test_sequence() {
    let app = add_clap(App::new("sequence"));
    let arg_matches = app.get_matches_from(
        vec![
            "test", "-p", "-t", "1", "sequence", "INIT", "REPLACE", "2MSG", "INIT",
        ]
        .iter(),
    );
    let mut model = test_model(&arg_matches);
    let mut stdout = Vec::new();
    model.do_clap(&arg_matches, &mut stdout);
    assert_output!(stdout, "uml");
}

#[test]
fn test_client_states() {
    let app = add_clap(App::new("states"));
    let arg_matches = app.get_matches_from(vec!["test", "-p", "-t", "1", "states", "C"].iter());
    let mut model = test_model(&arg_matches);
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
            A_0_false [ label=\"IDL\", shape=ellipse ];\n\
            subgraph cluster_0 {\n\
                T_0_18446744073709551615 [ shape=diamond, label=\"\", fontsize=0, width=0.15, height=0.15, style=filled, color=black ];\n\
                A_0_false -> T_0_18446744073709551615 [ arrowhead=none, direction=forward ];\n\
                T_0_18446744073709551615 -> A_0_false;\n\
                M_0_18446744073709551615_255 [ label=\"Time\", shape=plain ];\n\
                M_0_18446744073709551615_255 -> T_0_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                M_0_18446744073709551615_1 [ label=\"PNG\\n&#8594; S\", shape=plain ];\n\
                T_0_18446744073709551615 -> M_0_18446744073709551615_1 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                M_0_18446744073709551615_2 [ label=\"PNG &#8658;\\nPNG\\n&#8594; S\", shape=plain ];\n\
                T_0_18446744073709551615 -> M_0_18446744073709551615_2 [ arrowhead=normal, direction=forward, style=dashed ];\n\
            }\n\
            A_1_false [ label=\"WAT\", shape=ellipse ];\n\
            subgraph cluster_1 {\n\
                T_1_18446744073709551615 [ shape=point, height=0.015, width=0.015 ];\n\
                A_0_false -> T_1_18446744073709551615 [ arrowhead=none, direction=forward ];\n\
                T_1_18446744073709551615 -> A_1_false;\n\
                M_1_18446744073709551615_255 [ label=\"Time\", shape=plain ];\n\
                M_1_18446744073709551615_255 -> T_1_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                M_1_18446744073709551615_0 [ label=\"REQ\\n&#8594; S\", shape=plain ];\n\
                T_1_18446744073709551615 -> M_1_18446744073709551615_0 [ arrowhead=normal, direction=forward, style=dashed ];\n\
            }\n\
            subgraph cluster_2 {\n\
                T_2_18446744073709551615 [ shape=point, height=0.015, width=0.015 ];\n\
                A_1_false -> T_2_18446744073709551615 [ arrowhead=none, direction=forward ];\n\
                T_2_18446744073709551615 -> A_0_false;\n\
                M_2_18446744073709551615_3 [ label=\"S &#8594;\\nRSP\", shape=plain ];\n\
                M_2_18446744073709551615_3 -> T_2_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
            }\n\
        }\n\
        "
    );
}

#[test]
fn test_server_states() {
    let app = add_clap(App::new("states"));
    let arg_matches = app.get_matches_from(vec!["test", "-p", "-t", "1", "states", "S"].iter());
    let mut model = test_model(&arg_matches);
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
                subgraph cluster_0 {\n\
                    T_0_18446744073709551615 [ shape=point, height=0.015, width=0.015 ];\n\
                        A_0_false -> T_0_18446744073709551615 [ arrowhead=none, direction=forward ];\n\
                        T_0_18446744073709551615 -> A_0_false;\n\
                        M_0_18446744073709551615_1 [ label=\"C &#8594;\\nPNG\", shape=plain ];\n\
                        M_0_18446744073709551615_1 -> T_0_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                        M_0_18446744073709551615_2 [ label=\"C &#8594;\\nPNG &#8658;\\nPNG\", shape=plain ];\n\
                        M_0_18446744073709551615_2 -> T_0_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                }\n\
            A_1_true [ label=\"WRK\", shape=octagon ];\n\
                subgraph cluster_1 {\n\
                    T_1_18446744073709551615 [ shape=point, height=0.015, width=0.015 ];\n\
                        A_0_false -> T_1_18446744073709551615 [ arrowhead=none, direction=forward ];\n\
                        T_1_18446744073709551615 -> A_1_true;\n\
                        M_1_18446744073709551615_0 [ label=\"C &#8594;\\nREQ\", shape=plain ];\n\
                        M_1_18446744073709551615_0 -> T_1_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                }\n\
            subgraph cluster_2 {\n\
                T_2_18446744073709551615 [ shape=point, height=0.015, width=0.015 ];\n\
                    A_1_true -> T_2_18446744073709551615 [ arrowhead=none, direction=forward ];\n\
                    T_2_18446744073709551615 -> A_0_false;\n\
                    M_2_18446744073709551615_255 [ label=\"Time\", shape=plain ];\n\
                    M_2_18446744073709551615_255 -> T_2_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                    M_2_18446744073709551615_3 [ label=\"RSP\\n&#8594; C\", shape=plain ];\n\
                    T_2_18446744073709551615 -> M_2_18446744073709551615_3 [ arrowhead=normal, direction=forward, style=dashed ];\n\
            }\n\
            subgraph cluster_3 {\n\
                T_3_18446744073709551615 [ shape=point, height=0.015, width=0.015 ];\n\
                    A_1_true -> T_3_18446744073709551615 [ arrowhead=none, direction=forward ];\n\
                    T_3_18446744073709551615 -> A_1_true;\n\
                    M_3_18446744073709551615_1 [ label=\"C &#8594;\\nPNG\", shape=plain ];\n\
                    M_3_18446744073709551615_1 -> T_3_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
                    M_3_18446744073709551615_2 [ label=\"C &#8594;\\nPNG &#8658;\\nPNG\", shape=plain ];\n\
                    M_3_18446744073709551615_2 -> T_3_18446744073709551615 [ arrowhead=normal, direction=forward, style=dashed ];\n\
            }\n\
        }\n\
        "
    );
}
