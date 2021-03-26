// FILE NOT TESTED

use clap::App;
use clap::Arg;
use clap::ArgMatches;
use core::mem::size_of;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::io::stdout;
use std::io::BufWriter;
use std::rc::Rc;
use std::str;
use std::str::FromStr;
use strum::IntoStaticStr;
use total_space::*;

// We will have multiple clients, and to send messages to them, we'll need to know their indices in
// the system. This will automate placing these indices in a global variable for us.
declare_agent_indices! {CLIENTS}

// We will have a single server, and to send messages to it, we'll need to know its index in the
// system. This will automate placing this index in a global variable for us.
declare_agent_index! {SERVER}

// The payload of the messages (and activities).
//
// It would have been nice to have a separate type for payloads of messages exchanged between each
// pair of agent types, and for each agent type activities, but doing that in Rust's type system is
// extremely difficult.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum Payload {
    // A client develops a need for something (activity payload).
    Need,

    // A client sends a request to the server.
    //
    // Note that while the framework knows the source of each message, the target agent is not given
    // this information. If it needs to know where the message came from, then this must be
    // explicitly included in the message, as we do here.
    Request {
        client: usize,
    },

    // The server completed some work (activity payload).
    Completed,

    /// The server sends a response to a client.
    Response,
}

// The framework requires all sort of boilerplate from the enums (or structs) used for payload and
// states. This macro (and the similar `impl_struct_data`) macro automates setting all this up.
impl_enum_data! {
    // The first thing is giving the name of the enum, and its default value. The default value of
    // the payload has no special significance but is needed due to limitations of the framework.
    // Just pick any of the values.
    Payload = Self::Response,

    // Displaying the data is done by first showing it using the `Debug` trait. The framework then
    // patches it into something like `Request(client=0)` or `Response`. However, in diagrams and in
    // general we often want to use shorthand names. Therefore the macro supports a list of
    // additional patches, "from" => "to". In this example `Request(client=0)` will be converted to
    // `REQ(C=0)`.
    "client" => "C",
    "Request" => "REQ",
    "Response" => "RSP"
}

// The state of a client.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    // The client is idle, not doing anything.
    Idle,

    // The client is waiting for a response from the server.
    Wait,
}

// Implement all the boilerplate for the client state.
impl_enum_data! {
    ClientState = Self::Idle,
    "Idle" => "IDL",
    "Wait" => "WAT"
}

// Actual model logic - behavior of the client agents.
impl AgentState<ClientState, Payload> for ClientState {
    // A client can only send one message at a time.
    fn max_in_flight_messages(&self) -> Option<usize> {
        Some(1)
    }

    // What the clients do "on their own", regardless of messages.
    //
    // We are provided the agent instance within its type, but typically it has no effect on
    // behavior.
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            // A waiting client does nothing.
            Self::Wait => Activity::Passive,

            // An idle client may develop a need for a response from the server. This will be
            // delivered as a payload to the ``reaction`` function, as if it was sent from another
            // agent. In theory we can even use the same payloads for both messages and activities.
            Self::Idle => Activity::Process1(Payload::Need),
        }
    }

    // How the client reacts to receiving a payload (from either an activity or from another agent).
    fn reaction(&self, instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        // We have Rust's pattern matching at our disposal. Here we match on both our state (self)
        // and the payload we received. For complex agents, it may make more sense to match first on
        // one, then invoke another function that will switch on the other, or arrange the code in
        // any way that makes sense. Having a huge match clause that lists all options in one
        // unreadable mess is possible but is not readable/maintainable.
        match (self, payload) {
            // If we are idle, and we developed a need for a response, change the state to waiting
            // and send a request to the server. Note we use `agent_index!` to access the global
            // SERVER index variable declared above.
            (Self::Idle, Payload::Need) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Wait,
                // Here we use our instance number so the server will know how to send the response
                // back to us. Note we send the instance number and not our index.
                Emit::Unordered(Payload::Request { client: instance }, agent_index!(SERVER)),
            )),

            // If we are waiting, and got a response, go back to the idle state, sending no
            // messages.
            (Self::Wait, Payload::Response) => Reaction::Do1(Action::Change(Self::Idle)),

            // Anything else will mark the configuration as invalid.
            _ => Reaction::Unexpected,
        }
    }
}

// The state of a server.
#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    // The server is listening for a request.
    Listen,

    // The server got a request from a client and is working on it. We need to remember which client
    // this is so we can send the response to it.
    Work { client: usize },
}

// The usual implement-boilerplate macro and patching of the display of the state.
impl_enum_data! {
    ServerState = Self::Listen,
    "client" => "C",
    "Listen" => "LST",
    "Work" => "WRK"
}

// Actual model logic - behavior of the server agent.
impl AgentState<ServerState, Payload> for ServerState {
    // A server can have an in-flight response message for each client.
    fn max_in_flight_messages(&self) -> Option<usize> {
        Some(agents_count!(CLIENTS))
    }

    // The server may defer messages while it is working.
    fn is_deferring(&self) -> bool {
        matches!(self, &Self::Work { .. })
    }

    // What the server does "on its own", regardless of messages.
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            // The server does nothing while listening.
            Self::Listen => Activity::Passive,

            // The server may complete the task when working.
            Self::Work { .. } => Activity::Process1(Payload::Completed),
        }
    }

    // How the server reacts to receiving a payload (from either an activity or from another agent).
    fn reaction(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        // The usual ``match``
        match (*self, *payload) {
            // Accept a request while listening for one.
            (Self::Listen, Payload::Request { client }) => {
                // Change state to working (remember the client).
                Reaction::Do1(Action::Change(Self::Work { client: client }))
            }

            // Completed the task while working on one.
            (Self::Work { client }, Payload::Completed) => Reaction::Do1(Action::ChangeAndSend1(
                // Change the state back to listening,
                Self::Listen,
                // Send the response to the client. Here we see how to access the agent index of one
                // of several agents of the same type, using their instance.
                Emit::Unordered(Payload::Response, agent_index!(CLIENTS[client])),
            )),

            // Additional request(s) may arrive while we are working.
            // Defer them until after we have completed the task.
            (Self::Work { .. }, Payload::Request { .. }) => Reaction::Defer,

            // Anything else is unexpected.
            _ => Reaction::Unexpected,
        }
    }
}

// We need to define the data types used in the model.
// For a tiny model like this, it doesn't matter much.
// However, for larger models (X * 100 * M configurations), every byte counts.

// The type for identifying agent states.
//
// Each agent type has its own "namespace" and it is unlikely that a single agent type will have
// more than 255 possible internal states (if it does, it should probably become a container of
// simpler agents).
index_type! { StateId, u8 }

// The type for identifying messages.
//
// Note that this identifies both the payload, the source, the target, and any
// immediate/ordering/replacement information. Here we can use 8 bits for simple models but larger
// ones may need 16 bits.
index_type! { MessageId, u8 }

// The type for identifying invalid conditions.
//
// Unless one goes berserk with defining validation functions, 8 bits should be plenty.
index_type! { InvalidId, u8 }

// The type for identifying configurations.
//
// In this trivial model, 8 bits would be too much. Realistic models can easily overflow 16 bits. 32
// bits should be plenty though (you;d run out of memory way before running out of identifiers).
index_type! { ConfigurationId, u32 }

// The full model is parameterized by the above.
//
// This is fixed at compilation time and establishes limits on what the binary can do.
type ExampleModel = Model<
    StateId,
    MessageId,
    InvalidId,
    ConfigurationId,
    Payload,
    6,  // MAX_AGENTS - the maximal number of agents in the model.
    14, // MAX_MESSAGES - the maximal number of messages in-flight.
>;

// Create a model based on the command-line arguments.
fn example_model(arg_matches: &ArgMatches) -> ExampleModel {
    let clients_arg = arg_matches.value_of("Clients").unwrap();
    let clients_count = usize::from_str(clients_arg).expect("invalid progress rate");

    // Create an agent data type for the client(s).
    // The number of clients comes from
    let client_type = Rc::new(AgentTypeData::<ClientState, StateId, Payload>::new(
        "C",                             // The name of the agent type ("C" for short).
        Instances::Count(clients_count), // The number of instances from the command line.
        None,                            // No previous agent type.
    ));

    // Create an agent data type for the server.
    let server_type = Rc::new(AgentTypeData::<ServerState, StateId, Payload>::new(
        "SRV",                     // The name of the agent type ("SRV" for short).
        Instances::Singleton,      // Only one instance.
        Some(client_type.clone()), // Create a linked list of all the agent types.
    ));

    // Estimate the number of configurations, allowing override from the command line. This doesn't
    // have to be exact. It just pre-reserves space for this number of configurations, which make
    // for more efficient execution (less dynamic resizing of growing vectors and the like).
    let size = model_size(
        arg_matches, // Allow override by command line,
        (1 + clients_count) // State counts of server
        * (1 << clients_count)  // State counts of clients
        * (1 << (clients_count - 1)), // Messages sent from/to clients (estimate).
    );

    // Create the model, giving it the estimated size, the linked list of agent types, and a vector
    // of configuration validation functions (empty here).
    let model = ExampleModel::new(size, server_type, vec![]);

    // Initialize the global agent indices variables.
    init_agent_indices!(CLIENTS, "C", model);
    init_agent_index!(SERVER, "SRV", model);

    // Return the result.
    model
}

fn main() {
    // The largest data structure we keep is a Swiss hash table for the configurations. It helps
    // efficiency if the size of the entries of this hash table is half of a cache line or a full
    // cache line. The size is a linear combination of the parameters to the Model struct. The
    // theory is that it is better to "waste" a few bytes in the configuration to ensure that we
    // never need to hit two cache lines to fetch one - the larger the model, the more painful
    // having to fetch two cache lines instead of one becomes.
    assert_eq!(
        32,
        size_of::<<ExampleModel as MetaModel>::ConfigurationHashEntry>()
    );

    // Define and parse the command line options.
    let arg_matches = add_clap(
        App::new("clients-server").arg(
            // Argument for number of clients.
            // By convention, arguments that control the model itself are in upper case.
            // This prevents them from being confused with generic total space arguments.
            Arg::with_name("Clients")
                .long("Clients")
                .short("C")
                .help("set the number of clients")
                .default_value("2"),
        ),
    )
    .get_matches();

    // Build the model using the command line options.
    let mut model = example_model(&arg_matches);

    // Buffering stdout is a really good idea, especially for large models.
    let mut output = BufWriter::new(stdout());

    // Perform the operation(s) specified in the command line options.
    model.do_clap(&arg_matches, &mut output);
}
