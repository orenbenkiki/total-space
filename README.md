# total-space [![Build Status](https://api.travis-ci.org/orenbenkiki/total-space.svg?branch=master)](https://travis-ci.org/orenbenkiki/total-space) [![codecov](https://codecov.io/gh/orenbenkiki/total-space/branch/master/graph/badge.svg)](https://codecov.io/gh/orenbenkiki/total-space) [![Docs](https://docs.rs/total-space/badge.svg)](https://docs.rs/crate/total-space)

Investigate the total state space of communicating finite state machines. Specifically, given a
model of a system comprising of multiple agents, where each agent is a non-deterministic state
machine, which responds to either time or receiving a message with one of some possible actions,
where each such action can change the agent state and/or send messages to other agents; Then the
code here will generate the total possible configurations space of the overall system, validate the
model for completeness, validate each system configuration for additional arbitrary correctness
criteria, and visualize the model in various ways.

## Installing

To install:

```
cargo install total-space
```

## Using

To use this, you need to create your own main program:

```rust
use total_space;

fn main() {
    let arg_ = total_space::add_clap(
        clap::App::new("...")
        ... add your own command line arguments to control the model ...
    );
    let arg_matches = &app.get_matches();
    let mut model = ... Use your command line arguments to create the model ...;
    let mut output = BufWriter::new(stdout());
    model.do_clap(arg_matches, &mut output);
}
```

The trick, of course, is in creating the model. Having done so, you get an executable program which
allows computing the model (that is, collecting all the possible configurations that can be reached
by the model) and generating all sorts of outputs from it (statistics, lists, and diagrams).

## Example

Documentation lies, only the code speaks the truth: See
[examples/clients_server.rs](examples/clients_server.rs) for a complete simple model example. See
[tests/expected/example_clients_server](tests/expected/example_clients_server) for the expected
results of running various commands. The `svg` files were generated from the `dot` and `uml` files
using [GraphViz](https://www.graphviz.org/) and [PlantUML](https://plantuml.com/starting).

That said, to understand the code, the following will help.

## Concepts

* We have a finite collection of *Agents*.

* Each *Agent* is an *Instance* of some *Type*. The *Instance* is a small unsigned
  integer (0..).

* Each *Agent* has an *Index* in the overall system. The *Index* is a small unsigned
  integer (0..).

* An *Agent* may be a *Container* of some *Part* *Agents* (there's also support for an agent
  containing two types of *Parts*).

* Each *Agent* has a *State*. The type of the *State* is the same for all *Instances* of the same
  *Type* of *Agent*.

* *Agents* can send and receive *Messages* between them.

* Each *Message* carries a *Payload* from a *Source* *Agent* to a *Target* *Agent*.

* *Messages* are delivered as follows:

  * *Immediate* - are delivered before any non-*Immediate* messages. This is typically used to model
    a *Container* modifying the state of a *Part* by sending a *Message* to it.

  * *Unordered* - may be delivered in any order.

  * *Ordered* - may be delivered only after delivering all previously sent messages from the same
    *Source* to the same *Target*.

* A *Message* may *Replace* (overwrite) an existing *Message* from the same *Source* to the same
  *Target*. For example, this can be used to model writing to a "mailbox" memory address, where
  writing a new message may overwrite the previous one before it has been delivered.

* An *Agent* can also trigger an *Activity* regardless of *Messages*.

* An *Activity* carries a *Payload*, as if it was a *Message* from the agent to itself.

* An *Activity* is delivered before any of the messages.

* When an *Agent* is delivered a *Payload*, from either a *Message* or an *Activity*, it computes a
  *Reaction*. Computing such *Reactions* is the focus of the model.

* An *Agent* can only consider its own *State* when computing a *Reaction*, except for *Containers*
  which can also consider the *State* of their *Parts*.

* A *Reaction* may be:

  * *Ignore* - the *Payload* is discarded and no other changes are made to the system.

  * *Defer* - the *Agent* indicates the *Message* should only be delivered only after it changes its
    *State*. Only *Messages* can be *Deferred*. An *Agent* needs to explicitly identify its *State*
    as "deferring" to allow it to *Defer* while in that *State*. This is used to highlight such
    *States* in diagrams.

  * One or more *Actions*. If more than one *Action* is given, these are possible alternatives,
    where each one leads to a different possible flow.

* An *Action* may:

  * Change the *State* or the *Agent*. This is the only way to modify the *State* of an *Agent*.
    Note that *Containers* may not directly change the state of their *Parts*. Instead, they need to
    send an appropriate (typically *Immediate*) *Message* to them that will cause them to change
    their *State*.

  * Emit zero or more *Messages* to other *Agent(s)*. The total number of *Messages* sent by each
    *Agent* may be bounded by some limit which may be different for each *Instance*.

* A system *Configuration* is the collection of all the *States* of all the *Agents*, all the
  in-flight *Messages*, and optionally an indication that some *Invalid* condition occurred.

* An *Invalid* condition may occur due to a specific invalid *Agent* *State*, *Message* or
  *Activity* *Payload*, or as a result of some combinations of these in some *Configuration*.

* An *Action* results in a *Transition* between *Configurations*.

* The code computes the total space (graph) of *Configurations* and the *Transitions* that connect
  them.

## Validation

The code allows applying a validation function to each *State* and overall *Configuration*. Thus,
simply computing the total *Configurations* space can ensure arbitrary validation conditions.

The *Reaction* logic in each agent will typically `match` some combinations of its *State* and the
*Payload*, which will include a catch-all `Reaction::Unexpected` clause. Thus computing the model
verifies that no unexpected message is received while in a state that does not expect to see it.

The code also verifies the number of sent in-flight *Messages* from each *Agent* does not exceed the
specified threshold. This ensures that no *Agent* is sending an unbounded number of *Messages*.

In addition, the code allows verifying that there is a path from every *Configurations* back to the
initial *Configuration*, which ensures there are no deadlock *Configurations* or a cycle of livelock
*Configurations*.

Finally, the code allows for generating *Transition* paths leading to *Configurations* that satisfy
arbitrary conditions, which can be used to ensure that any *Configuration* of interest is actually
reachable from the initial *Configuration*.

Another way to ensure the model is covered is to collect coverage information for the model (see
[grcov](https://marco-c.github.io/2020/11/24/rust-source-based-code-coverage.html)) and looking at
the output to ensure that all the code was reached. Uncovered `match` clauses will indicate that
that specific flow is not reachable from the initial *Configuration*, which could indicate a bug in
the model or that a simpler model may suffice.

## Output

The code allows simply listing all *Configurations* and *Transitions*. This has limited usefulness,
mainly for debugging.

In addition, the code can generate two types of diagrams:

* A GraphViz diagram of all the *States* of some *Agent* and the *Actions* that moved the *Agent*
  between these *States*. This is used to visualize the behavior of each *Agent* in isolation.

* A UML sequence diagram of all the *Transitions* between *Configurations* along the shortest path
  between *Configurations* that satisfy arbitrary conditions. This is used to visualize complete
  system scenarios.

## License

`total-space` is distributed under the GNU General Public License (Version 3.0). See the
[LICENSE](LICENSE.txt) for details.
