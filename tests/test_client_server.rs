extern crate total_space;

use clap::App;
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

declare_global_agent_index! {CLIENT}
declare_global_agent_index! {SERVER}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    Ping,
    Request,
    Response,
}
impl_message_payload! { Payload }
// END MAYBE TESTED

impl Validated for Payload {}

// BEGIN MAYBE TESTED
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    Idle,
    Wait,
}
impl_agent_state! { ClientState = Self::Idle }
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
impl_agent_state! { ServerState = Self::Listen }
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

    let model = TestModel::new(server_type, vec![]);
    init_global_agent_index!(CLIENT, "Client", model);
    init_global_agent_index!(SERVER, "Server", model);
    model
}

#[test]
fn test_conditions() {
    let mut model = test_model();
    let app = add_clap(App::new("test_agents"));
    let arg_matches = app.get_matches_from(vec!["test", "conditions"].iter());
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        0MSG: matches any configuration with no in-flight messages\n\
        1MSG: matches any configuration with a single in-flight message\n\
        2MSG: matches any configuration with 2 in-flight messages\n\
        3MSG: matches any configuration with 3 in-flight messages\n\
        4MSG: matches any configuration with 4 in-flight messages\n\
        5MSG: matches any configuration with 5 in-flight messages\n\
        6MSG: matches any configuration with 6 in-flight messages\n\
        7MSG: matches any configuration with 7 in-flight messages\n\
        8MSG: matches any configuration with 8 in-flight messages\n\
        9MSG: matches any configuration with 9 in-flight messages\n\
        INIT: matches the initial configuration\n\
        REPLACE: matches a configuration with a replaced message\n\
        VALID: matches any valid configuration (is typically negated)\n\
        "
    );
}

#[test]
fn test_agents() {
    let mut model = test_model();
    let app = add_clap(App::new("test_conditions"));
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
    let app = add_clap(App::new("test_configurations"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "configurations"].iter());
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
            stdout,
            "\
            Client:Idle & Server:Listen\n\
            Client:Wait & Server:Listen | Client -> Request -> Server\n\
            Client:Idle & Server:Listen | Client -> Ping -> Server\n\
            Client:Wait & Server:Listen | Client -> Request -> Server & Client -> Ping -> Server\n\
            Client:Idle & Server:Listen | Client -> Ping => Ping -> Server\n\
            Client:Wait & Server:Listen | Client -> Request -> Server & Client -> Ping => Ping -> Server\n\
            Client:Wait & Server:Work | Client -> Ping => Ping -> Server\n\
            Client:Wait & Server:Listen | Client -> Ping => Ping -> Server & Server -> Response -> Client\n\
            Client:Wait & Server:Work\n\
            Client:Wait & Server:Listen | Server -> Response -> Client\n\
            Client:Wait & Server:Work | Client -> Ping -> Server\n\
            Client:Wait & Server:Listen | Client -> Ping -> Server & Server -> Response -> Client\n\
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
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
            stdout,
            "\
            FROM Client:Idle & Server:Listen\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server\n\
            - BY time event\n  \
              TO Client:Idle & Server:Listen | Client -> Ping -> Server\n\
            FROM Client:Wait & Server:Listen | Client -> Request -> Server\n\
            - BY message Client -> Request -> Server\n  \
              TO Client:Wait & Server:Work\n\
            FROM Client:Idle & Server:Listen | Client -> Ping -> Server\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server & Client -> Ping -> Server\n\
            - BY time event\n  \
              TO Client:Idle & Server:Listen | Client -> Ping => Ping -> Server\n\
            - BY message Client -> Ping -> Server\n  \
              TO Client:Idle & Server:Listen\n\
            FROM Client:Wait & Server:Listen | Client -> Request -> Server & Client -> Ping -> Server\n\
            - BY message Client -> Request -> Server\n  \
              TO Client:Wait & Server:Work | Client -> Ping -> Server\n\
            - BY message Client -> Ping -> Server\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server\n\
            FROM Client:Idle & Server:Listen | Client -> Ping => Ping -> Server\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server & Client -> Ping => Ping -> Server\n\
            - BY time event\n  \
              TO Client:Idle & Server:Listen | Client -> Ping => Ping -> Server\n\
            - BY message Client -> Ping => Ping -> Server\n  \
              TO Client:Idle & Server:Listen\n\
            FROM Client:Wait & Server:Listen | Client -> Request -> Server & Client -> Ping => Ping -> Server\n\
            - BY message Client -> Request -> Server\n  \
              TO Client:Wait & Server:Work | Client -> Ping => Ping -> Server\n\
            - BY message Client -> Ping => Ping -> Server\n  \
              TO Client:Wait & Server:Listen | Client -> Request -> Server\n\
            FROM Client:Wait & Server:Work | Client -> Ping => Ping -> Server\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Ping => Ping -> Server & Server -> Response -> Client\n\
            - BY message Client -> Ping => Ping -> Server\n  \
              TO Client:Wait & Server:Work\n\
            FROM Client:Wait & Server:Listen | Client -> Ping => Ping -> Server & Server -> Response -> Client\n\
            - BY message Client -> Ping => Ping -> Server\n  \
              TO Client:Wait & Server:Listen | Server -> Response -> Client\n\
            - BY message Server -> Response -> Client\n  \
              TO Client:Idle & Server:Listen | Client -> Ping => Ping -> Server\n\
            FROM Client:Wait & Server:Work\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Server -> Response -> Client\n\
            FROM Client:Wait & Server:Listen | Server -> Response -> Client\n\
            - BY message Server -> Response -> Client\n  \
              TO Client:Idle & Server:Listen\n\
            FROM Client:Wait & Server:Work | Client -> Ping -> Server\n\
            - BY time event\n  \
              TO Client:Wait & Server:Listen | Client -> Ping -> Server & Server -> Response -> Client\n\
            - BY message Client -> Ping -> Server\n  \
              TO Client:Wait & Server:Work\n\
            FROM Client:Wait & Server:Listen | Client -> Ping -> Server & Server -> Response -> Client\n\
            - BY message Client -> Ping -> Server\n  \
              TO Client:Wait & Server:Listen | Server -> Response -> Client\n\
            - BY message Server -> Response -> Client\n  \
              TO Client:Idle & Server:Listen | Client -> Ping -> Server\n\
            "
        );
}

#[test]
fn test_path() {
    let mut model = test_model();
    let app = add_clap(App::new("test_path"));
    let arg_matches =
        app.get_matches_from(vec!["test", "-r", "-p", "-t", "1", "path", "INIT", "INIT"].iter());
    let mut stdout_bytes = Vec::new();
    model.do_clap(&arg_matches, &mut stdout_bytes);
    let stdout = str::from_utf8(&stdout_bytes).unwrap();
    assert_eq!(
        stdout,
        "\
        INIT Client:Idle & Server:Listen\n\
        BY time event\n\
        TO Client:Idle & Server:Listen | Client -> Ping -> Server\n\
        BY message Client -> Ping -> Server\n\
        INIT Client:Idle & Server:Listen\n\
        "
    );
}

#[test]
fn test_sequence() {
    let mut model = test_model();
    let app = add_clap(App::new("sequence"));
    let arg_matches = app.get_matches_from(
        vec![
            "test", "-r", "-p", "-t", "1", "sequence", "INIT", "REPLACE", "2MSG", "INIT",
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
        participant \"Client\" as A0 order 10100\n\
        activate A0 #CadetBlue\n\
        participant \"Server\" as A1 order 10200\n\
        activate A1 #CadetBlue\n\
        rnote over A0 : Idle\n\
        / rnote over A1 : Listen\n\
        ?o-> A0\n\
        control \" \" as T0 order 10101\n\
        A0 -> T0 : Ping\n\
        activate T0 #Silver\n\
        ?o-> A0\n\
        A0 -> T0 : &#8658; Ping\n\
        ?o-> A0\n\
        deactivate A0\n\
        A0 -> A1 : Request\n\
        deactivate A1\n\
        rnote over A0 : Wait\n\
        activate A0 #MediumPurple\n\
        autonumber stop\n\
        ?-[#White]\\ A1\n\
        autonumber resume\n\
        rnote over A1 : Work\n\
        activate A1 #CadetBlue\n\
        ?o-> A1\n\
        deactivate A1\n\
        control \" \" as T1 order 10199\n\
        A1 -> T1 : Response\n\
        activate T1 #Silver\n\
        rnote over A1 : Listen\n\
        activate A1 #MediumPurple\n\
        T0 -> A1 : Ping\n\
        deactivate T0\n\
        T1 -> A0 : Response\n\
        deactivate T1\n\
        deactivate A0\n\
        autonumber stop\n\
        ?-[#White]\\ A0\n\
        autonumber resume\n\
        rnote over A0 : Idle\n\
        activate A0 #MediumPurple\n\
        @enduml\n\
        "
    );
}
