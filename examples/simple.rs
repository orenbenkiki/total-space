// FILE NOT TESTED

use clap::App;
use clap::Arg;
use clap::ArgMatches;
// use core::mem::size_of;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FormatterResult;
use std::rc::Rc;
use std::str;
use std::str::FromStr;
use strum::IntoStaticStr;
use total_space::*;

// The maximal number of workers we allow.
const MAX_WORKERS: usize = 4;

// We will have multiple clients, and multiple workers, and to send messages to them, we'll need to
// know their indices in the system. This will automate placing these indices in global variables
// for us.
declare_agent_indices! {WORKERS}
declare_agent_indices! {CLIENTS}

// We will need to access the workers and clients in our conditions. This will place a reference to
// the worker type in a global variable so we can do so.
declare_agent_type_data! { WORKER_TYPE, WorkerState, SimpleModel }
declare_agent_type_data! { CLIENT_TYPE, ClientState, SimpleModel }

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

    // A task that a client needs the result for.
    //
    // Note that while the framework knows the source of each message, the target agent is not given
    // this information. If it needs to know where the message came from, then this must be
    // explicitly included in the message, as we do here.
    Task { client: usize },

    // A worker completed some work (activity payload).
    Completed,

    // The result of executing the task.
    Result,
}

// The state of a worker.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum WorkerState {
    // The worker is not doing anything.
    Idle,

    // The worker is working on some task.
    Working { client: usize },
}

// The framework requires all sort of boilerplate from the enums (or structs) used for payload and
// states. This macro (and the similar `impl_struct_data`) macro automates setting all this up.
impl_enum_data! {
    // The first thing is giving the name of the enum, and its default value. The default value of
    // the payload has no special significance but is needed due to limitations of the framework.
    // Just pick any of the values.
    WorkerState = Self::Idle,

    // Displaying the data is done by first showing it using the `Debug` trait. The framework then
    // patches it into something like `Working(client=0)`. However, in diagrams and in general we often
    // want to use shorthand names. Therefore the macro supports a list of additional patches,
    // "from" => "to". In this example `Working(client=0)` will be converted to `WRK(C=0)`.
    "client" => "C",
    "Idle" => "IDL",
    "Working" => "WRK"
}

// Sometimes it is easier to write the code using a struct instead of an enum for the agent's state.
// This may make
// In such cases, you could write something like:

#[cfg(struct_instead_of_enum)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum WorkerStateName {
    Idle,
    Working,
}

#[cfg(struct_instead_of_enum)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug)]
struct WorkerState {
    // The short name of the state.
    name: WorkerStateName,

    // Additional fields.
    client: usize,
}

#[cfg(struct_instead_of_enum)]
impl AgentState<WorkerState, Payload> for WorkerState {
    // Arbitrary validation for the agent's state. By default this returns `None` meaning the state
    // is valid. Here we do so explicitly for the example's sake. If there is something wrong, the
    // function should return a reason string. This will be reported together with the actual state
    // of the agent, so there's no need to include specific details in the message itself. In theory
    // the validation might depend on the instance, typically it does not.
    //
    // When using a `struct` for the state, it typically allows for invalid combinations, so it is
    // useful to verify they never occur in practice. Using an `enum` allows for ensuring that
    // invalid states can't be represented in the 1st place. However, using a `struct` allows for
    // easier access to common fields that occur in many enum variants, and allows for `match`
    // statements to test such fields regardless of the value of the `name`. The best approach
    // "depends".
    fn invalid_because(&self, _instance: usize) -> Option<&'static str> {
        if self.name == WorkerStateName::Idle && self.client != 0 {
            Some("non-zero client when in the Idle state")
        } else {
            None
        }
    }
}

// Implement the boilerplate.
#[cfg(struct_instead_of_enum)]
impl_struct_data! {
    WorkerState = WorkerState { name: Idle, client: 0 },
    // "from" => "to", ...
}

// Actual model logic - behavior of the worker agent.
impl AgentState<WorkerState, Payload> for WorkerState {
    // The maximal number of in-flight messages a worker can generate. By default this is not
    // restricted. In theory the maximal number of in-flight messages might depend on the instance
    // and the state, but typically it is a constant.
    fn max_in_flight_messages(&self, _instance: usize) -> Option<usize> {
        // In theory we might have completed a task for each client and sent the result to them,
        // and all of these might be in-flight at the same time.
        Some(agents_count!(CLIENTS))
    }

    // What the worker does "on its own", regardless of messages. By default it is passive. We are
    // provided the agent instance within its type, but typically it has no effect on behavior.
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            // The worker does nothing while listening.
            Self::Idle => Activity::Passive,

            // The worker may complete the task when working.
            Self::Working { .. } => Activity::Process1(Payload::Completed),
        }
    }

    // How the worker reacts to receiving a payload (from either an activity or from another agent).
    // Again we are provided the agent instance within its type, but typically it has no effect on
    // behavior.
    fn reaction(&self, _instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        // We have Rust's pattern matching at our disposal. Here we match on both our state (self)
        // and the payload we received. For complex agents, it may make more sense to match first on
        // one, then invoke another function that will switch on the other, or arrange the code in
        // any way that makes sense. Having a huge match clause that lists all options in one
        // unreadable mess is possible but is not readable/maintainable.
        match (*self, *payload) {
            // Accept a task while idle.
            (Self::Idle, Payload::Task { client }) => {
                // Change state to working (remember the client), not sending any messages.
                Reaction::Do1(Action::Change(Self::Working { client: client }))
            }

            // Completed the task we are working on.
            (Self::Working { client }, Payload::Completed) => {
                Reaction::Do1(Action::ChangeAndSend1(
                    // Change the state back to idle.
                    Self::Idle,
                    // Send the result to the client. Here we see how to access the agent index of one
                    // of several agents of the same type, using their instance.
                    Emit::Unordered(Payload::Result, agent_index!(CLIENTS[client])),
                ))
            }

            // Anything else is unexpected.
            _ => Reaction::Unexpected,
        }
    }
}

// The state of a server. We'll make this a container of workers, so it doesn't have an interesting
// state of its own.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ServerState {
    // The server is active (the only state).
    Active,
}

// The usual implement-boilerplate macro and patching of the display of the state.
impl_enum_data! {
    ServerState = Self::Active,
    "Active" => "" // Since we only have one state there's no point in naming it in the diagrams.
}

// Actual model logic - behavior of the server agent.
impl ContainerOf1State<ServerState, WorkerState, Payload> for ServerState {
    // A server can have one in-flight messages (because it only sends a single immediate message
    // each time). In theory the maximal number of in-flight messages might also depend on the state
    // of the parts, but typically it is a constant.
    fn max_in_flight_messages(&self, _instance: usize, _workers: &[WorkerState]) -> Option<usize> {
        Some(1)
    }

    // The server may defer messages if both workers are busy.
    fn is_deferring(&self, _instance: usize, _workers: &[WorkerState]) -> bool {
        true
    }

    // How the server reacts to receiving a payload (from either an activity or from another agent).
    // Here because the server is a container it has read-only access to the state of its worker
    // parts.
    fn reaction(
        &self,
        _instance: usize,
        payload: &Payload,
        workers: &[WorkerState],
    ) -> Reaction<Self, Payload> {
        // Match only on the payload (as we only have one state).
        match *payload {
            // A task was sent to us, presumably by a client.
            Payload::Task { client } => {
                // It is useful to factor out complex logic into functions to keep the match clauses
                // clean.
                Self::task_reaction(client, workers)
            }

            // Anything else is unexpected.
            _ => Reaction::Unexpected,
        }
    }
}

// Support functions for the server agent.
impl ServerState {
    // We got a task from a client, return a reaction.
    fn task_reaction(client: usize, workers: &[WorkerState]) -> Reaction<Self, Payload> {
        let mut actions: [Option<Action<Self, Payload>>; MAX_COUNT] = [None; MAX_COUNT];

        let mut has_idle_worker = false;
        for (worker, worker_state) in workers.iter().enumerate() {
            match *worker_state {
                WorkerState::Working {
                    client: worker_client,
                } => {
                    // The worker is busy serving someone. If this someone is the client that sent
                    // the task, we have a client that sends two tasks in parallel, which is
                    // unexpected in this simple model.
                    if worker_client == client {
                        return Reaction::Unexpected;
                    }
                }

                WorkerState::Idle => {
                    // We found an idle worker. We can perform the action of forwarding the task to
                    // it. This is "immediate" since we consider the worker to be a part of the
                    // server, that is, this isn't as much a communication as it is a direct
                    // invocation, so it can't be (meaningfully) reordered with anything else which
                    // might be in-flight.
                    //
                    // In a more complex model, we'd just be exchanging messages with the worker,
                    // and all sort of interesting race conditions would need to be dealt with -
                    // possibly by having both a server "shadow worker" part that represents its
                    // internal view of the worker state, and a separate independent "real worker"
                    // agent which does the actual work. Messages to the shadow worker would be
                    // immediate but messages to the real worker would be normal communication which
                    // might be reordered with other messages, causing all sort of interesting
                    // scenarios.
                    actions[worker] = Some(Action::Send1(Emit::Immediate(
                        Payload::Task { client },
                        agent_index!(WORKERS[worker]),
                    )));
                    has_idle_worker = true;
                }
            }
        }

        if !has_idle_worker {
            // All workers are busy. In this simple model, the server simply defers receiving the
            // task (this is equivalent to having a queue of messages that the server reads from).
            return Reaction::Defer;
        }

        // We have at least one and possible several alternative workers who are idle. In this
        // simple model, any one of them might be used to perform the task.
        Reaction::Do1Of(actions)
    }
}

// The usual implement-boilerplate macro and patching of the display of the state.
impl_enum_data! { Payload = Self::Result }

// The state of a client.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone, Debug, IntoStaticStr)]
enum ClientState {
    // The client is running, not bothering the server.
    Running,

    // The client is blocked, waiting for a response from the server.
    Blocked,
}

// Implement all the boilerplate for the client state.
impl_enum_data! {
    ClientState = Self::Running,
    "Running" => "RUN",
    "Blocked" => "BLK"
}

// Actual model logic - behavior of the client agents.
impl AgentState<ClientState, Payload> for ClientState {
    // A client can only send one message at a time.
    fn max_in_flight_messages(&self, _instance: usize) -> Option<usize> {
        Some(1)
    }

    // What the clients do "on their own", regardless of messages.
    fn activity(&self, _instance: usize) -> Activity<Payload> {
        match self {
            // A running client may develop a need for a response from the server. This will be
            // delivered as a payload to the ``reaction`` function, as if it was sent from another
            // agent. In theory we can even use the same payloads for both messages and activities.
            Self::Running => Activity::Process1(Payload::Need),

            // A waiting client does nothing.
            Self::Blocked => Activity::Passive,
        }
    }

    // How the client reacts to receiving a payload (from either an activity or from another agent).
    fn reaction(&self, instance: usize, payload: &Payload) -> Reaction<Self, Payload> {
        // The usual match.
        match (self, payload) {
            // If we are running, and we developed a need for a response, change the state to
            // blocked and send a request to the server. Note we use `agent_index!` to access the
            // global SERVER index variable declared above.
            (Self::Running, Payload::Need) => Reaction::Do1(Action::ChangeAndSend1(
                Self::Blocked,
                // Here we use our instance number so the server will know how to send the response
                // back to us. Note we send our instance number, not our index.
                Emit::Unordered(Payload::Task { client: instance }, agent_index!(SERVER)),
            )),

            // If we are blocked, and got a response, go back to the running state, sending no
            // messages.
            (Self::Blocked, Payload::Result) => Reaction::Do1(Action::Change(Self::Running)),

            // Anything else will mark the configuration as invalid.
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
type SimpleModel = Model<
    StateId,
    MessageId,
    InvalidId,
    ConfigurationId,
    Payload,
    6,  // MAX_AGENTS - the maximal number of agents in the model.
    14, // MAX_MESSAGES - the maximal number of messages in-flight.
>;

// Validate the model (in addition to independent validation of each agent).
struct SimpleValidator;
impl ModelValidator<SimpleModel> for SimpleValidator {
    // Functions to validate a message or the whole configuration. In this simple model we have
    // none, so we could have just omitted defining `SimpleValidator` in the 1st place by using
    // `Model::new` instead of `Model::with_validator` below.
}

// Create a model based on the command-line arguments.
fn example_model(arg_matches: &ArgMatches) -> SimpleModel {
    let workers_arg = arg_matches.value_of("Workers").unwrap();
    let workers_count = usize::from_str(workers_arg).expect("invalid number of workers");
    assert!(
        workers_count > 0,
        "the number of workers must be at least 1"
    );
    assert!(
        workers_count <= MAX_WORKERS,
        "the number of workers can be at most {}",
        MAX_WORKERS
    );

    let clients_arg = arg_matches.value_of("Clients").unwrap();
    let clients_count = usize::from_str(clients_arg).expect("invalid number of clients");
    assert!(
        clients_count > 0,
        "the number of clients must be at least 1"
    );

    assert!(
        clients_count >= workers_count,
        "the number of workers can be at most the number of clients"
    );

    // Create an agent data type for the client(s).
    let worker_type = Rc::new(AgentTypeData::<WorkerState, StateId, Payload>::new(
        "W",                             // The name of the agent type ("W" for short).
        Instances::Count(workers_count), // The number of instances from the command line.
        None,                            // No previous agent type.
    ));

    // Create an agent data type for the server.
    let mut server_type = Rc::new(ContainerOf1TypeData::<
        ServerState,
        WorkerState,
        StateId,
        Payload,
        MAX_WORKERS,
    >::new(
        "S",                  // The name of the agent type ("S" for short).
        Instances::Singleton, // Only one instance.
        worker_type.clone(),  // The type of the agents which are parts of this container.
        worker_type.clone(),  // Create a linked list of all the agent types.
    ));

    // By default, the order of the agents in the sequence diagram is the same as the order they are
    // added to the linked list above. It is possible to override this for a singleton agent,
    // immediately after it is created. This isn't necessary in this simple case, but for the
    // example's sake:
    Rc::get_mut(&mut server_type).unwrap().set_appearance(
        0,
        InstanceAppearance {
            order: 10,
            ..Default::default()
        },
    );

    // Create an agent data type for the client(s).
    let mut client_type = Rc::new(AgentTypeData::<ClientState, StateId, Payload>::new(
        "C",                             // The name of the agent type ("C" for short).
        Instances::Count(clients_count), // The number of instances from the command line.
        Some(server_type.clone()),       // Create a linked list of all the agent types.
    ));

    // Example of overriding the order of multiple agent instances (not really needed in this simple
    // example). Here we also create a group (a box around) for all the instances.
    for client in 0..clients_count {
        Rc::get_mut(&mut client_type).unwrap().set_appearance(
            client,
            InstanceAppearance {
                order: 20 + client,
                group: Some("clients"),
            },
        );
    }

    // Initialize the global reference to the worker and client agent type.
    init_agent_type_data!(WORKER_TYPE, worker_type);
    init_agent_type_data!(CLIENT_TYPE, client_type);

    // Estimate the number of configurations, allowing override from the command line. This doesn't
    // have to be exact. It just pre-reserves space for this number of configurations, which make
    // for more efficient execution. In this case we just pre-allocate 100 states which should be
    // enough for reasonable model parameters.
    let size = model_size(
        arg_matches, // Allow override by command line,
        100,         // By default pre-allocate 100 configurations.
    );

    // Initialize the global agent indices variables.
    init_agent_indices!(WORKERS, worker_type);
    init_agent_index!(SERVER, server_type);
    init_agent_indices!(CLIENTS, client_type);

    // Create the model, giving it the estimated size and the linked list of agent types. If there
    // are no validation functions, the simpler `Model::new` skips the 3rd argument. It is included
    // here for the example's sake.
    let mut model = SimpleModel::with_validator(size, client_type, Rc::new(SimpleValidator));

    // Add some interesting conditions.
    model.add_condition(
        "ALL_WORKERS_ARE_BUSY",
        all_clients_are_blocked,
        "All the workers are in the busy state",
    );
    model.add_condition(
        "ALL_CLIENTS_ARE_BLOCKED",
        all_workers_are_busy,
        "All the clients are in the blocked blocked",
    );
    model.add_condition(
        "DEFERRED_TASK",
        has_deferred_task,
        "There exists a task request message that is deferred by the server",
    );

    // Return the result.
    model
}

// A condition on the overall configuration - in this case, that all workers are busy. This
// demonstrates accessing complex data types associated with the model via the `MetaModel` trait,
// and the `agent_states!` macro that iterates on the states of all agents of some type.
fn all_workers_are_busy(
    _model: &SimpleModel,
    configuration: &<SimpleModel as MetaModel>::Configuration,
) -> bool {
    agent_states_iter!(
        configuration,
        WORKER_TYPE,
        all(|worker_state| worker_state == WorkerState::Idle)
    )
}

// A similar condition on the clients.
fn all_clients_are_blocked(
    _model: &SimpleModel,
    configuration: &<SimpleModel as MetaModel>::Configuration,
) -> bool {
    agent_states_iter!(
        configuration,
        CLIENT_TYPE,
        all(|client_state| client_state == ClientState::Blocked)
    )
}

// A condition for deferred messages. This is tricky to detect because the framework doesn't track
// whether messages were deferred, instead we figure that if the number of in-flight task messages
// is larger than the number of idle workers, then one of these messages will need to be deferred.
fn has_deferred_task(
    model: &SimpleModel,
    configuration: &<SimpleModel as MetaModel>::Configuration,
) -> bool {
    let task_messages_count = messages_iter!(
        model,
        configuration,
        filter(|message| matches!(message.payload, Payload::Task { .. })).count()
    );

    let idle_workers_count = agent_states_iter!(
        configuration,
        WORKER_TYPE,
        filter(|worker_state| *worker_state == WorkerState::Idle).count()
    );

    task_messages_count > idle_workers_count
}

// The largest data structure we keep is a Swiss hash table for the configurations. It helps
// efficiency if the size of the entries of this hash table is half of a cache line or a full
// cache line. The size is a linear combination of the parameters to the Model struct. The
// theory is that it is better to "waste" a few bytes in the configuration to ensure that we
// never need to hit two cache lines to fetch one - the larger the model, the more painful
// having to fetch two cache lines instead of one becomes.
assert_configuration_hash_entry_size!(SimpleModel, 32);

fn main() {
    // Define and parse the command line options.
    let mut app = add_clap(
        App::new("clients-server")
            .arg(
                // Argument for number of workers.
                // By convention, arguments that control the model itself are in upper case.
                // This prevents them from being confused with generic total space arguments.
                Arg::with_name("Workers")
                    .long("Workers")
                    .short("W")
                    .help("set the number of workers")
                    .default_value("1"),
            )
            .arg(
                // Argument for number of clients.
                Arg::with_name("Clients")
                    .long("Clients")
                    .short("C")
                    .help("set the number of clients")
                    .default_value("1"),
            ),
    );

    // Build the model using the global flags.
    // Technically this also includes the flags of the 1st command, which don't affect the model.
    let mut model = example_model(&get_model_arg_matches(&mut app));

    // Perform the commands(s) specified in the command line options.
    // This will loop on each command (separated by '-o') and execute it.
    model.do_clap(&mut app);
}
