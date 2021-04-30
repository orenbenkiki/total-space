# total-space
[![license](https://img.shields.io/crates/l/total-space.svg)](https://github.com/orenbenkiki/total-space/blob/master/LICENSE.txt)
[![crates.io](https://img.shields.io/crates/v/total-space.svg)](https://crates.io/crates/total-space)
[![Build Status](https://api.travis-ci.com/orenbenkiki/total-space.svg?branch=master)](https://travis-ci.com/orenbenkiki/total-space)
[![codecov](https://codecov.io/gh/orenbenkiki/total-space/branch/master/graph/badge.svg)](https://codecov.io/gh/orenbenkiki/total-space)
[![Docs](https://docs.rs/total-space/badge.svg)](https://docs.rs/crate/total-space)

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
    let arg_matches = total_space::add_clap(
        clap::App::new("...")
        ... add your own command line arguments to control the model ...
    ).get_matches();
    let mut model = ... Use your command line arguments to create the model ...;
    let mut output = BufWriter::new(stdout());
    model.do_clap(&arg_matches, &mut output);
}
```

The trick, of course, is in creating the model. Having done so, you get an executable program which
allows computing the model (that is, collecting all the possible configurations that can be reached
by the model) and generating all sorts of outputs from it (statistics, lists, and diagrams).

## Example

You can look at the [API reference](https://docs.rs/total-space/) for details about a specific
function. That is of no use at all to get started, though. For that, see
[examples/simple.rs](examples/simple.rs) for a complete simple model example. See
[tests/expected/example_simple](tests/expected/example_simple) for the expected results of running
various commands. The `svg` files were generated from the `dot` and `uml` files using
[GraphViz](https://www.graphviz.org/) and [PlantUML](https://plantuml.com/starting).

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

  * *Ordered* - may be delivered only after delivering all previously sent *Ordered* messages from
    the same *Source* to the same *Target*. That is, *Unordered* messages can be delivered in any
    order relative to *Ordered* messages.

* A *Message* may *Replace* (overwrite) an existing *Message* from the same *Source* to the same
  *Target*. For example, this can be used to model writing to a "mailbox" memory address, where
  writing a new message may overwrite the previous one before it has been read by the target agent.

* An *Agent* can also trigger an *Activity* regardless of *Messages*. This models things like
  external interrupts or the *Agent* performing some some internal computation such as completing a
  task.

* An *Activity* carries a *Payload*, as if it was a *Message* from the agent to itself.

* An *Activity* is delivered before any of the messages.

* When an *Agent* is delivered a *Payload*, from either a *Message* or an *Activity*, it computes a
  *Reaction*. Computing such *Reactions* is the focus of the model.

* An *Agent* can only consider its own *State* when computing a *Reaction*, except for *Containers*
  which can also consider the *State* of their *Parts*.

* A *Reaction* may be:

  * *Unexpected* - the model does not expect this *Payload* to be delivered to this *Agent* while
    it is at this *State*. This indicates that the model is either under-specified (some scenario is
    possible, but was not planned for) or that the model has a bug (some scenario is possible in the
    model, but should not be). By default, if this happens, the framework will emit the shortest
    path of *Transitions* from the initial *Configuration* to the *Unexpected* case, which makes it
    easy to debug such cases.

  * *Ignore* - the *Payload* is discarded and no other changes are made to the system.

  * *Defer* - the *Agent* indicates the *Message* should only be delivered only after it changes its
    *State*. Only *Messages* can be *Deferred*. An *Agent* needs to explicitly identify its *State*
    as "deferring" to allow it to *Defer* while in that *State*. This is used to highlight such
    *States* in diagrams. *Deferring* is roughly equivalent to queueing the *Message* for later
    processing.

  * One or more *Actions*. If more than one *Action* is given, these are possible alternatives,
    where each one leads to a different possible flow.

* An *Action* may:

  * Change the *State* of the *Agent*. This is the only way to modify the *State* of an *Agent*.
    Note that *Containers* may not directly change the state of their *Parts*. Instead, they need to
    send an appropriate (typically *Immediate*) *Message* to them that will cause them to change
    their *State*.

  * Emit zero or more *Messages* to other *Agent(s)*. The total number of *Messages* sent by each
    *Agent* may be bounded by some limit, which may be different for each *Instance* (for example,
    encountering an *Unexpected* reaction).

* A system *Configuration* is the collection of all the *States* of all the *Agents*, all the
  in-flight *Messages*, and optionally an indication that some *Invalid* condition occurred.

* An *Invalid* condition may occur due to a specific invalid *Agent* *State*, *Message* or
  *Activity* *Payload*, or as a result of some combinations of these in some *Configuration*.
  Similarly to the *Unexpected* case, by default, if an *Invalid* condition is seen, the framework
  will emit the shortest path of *Transitions* from the initial *Configuration* to the *Invalid*
  one, which makes it easy to debug such cases.

* An *Action* results in a *Transition* between different *Configurations*. Note that an *Ignore*
  *Reaction* consumes the *Message* so its target *Configuration* would have one less in-flight
  *Message*. An *Ignore* *Reaction* to an *Activity* can have the same source and target
  *Configuration*, and is simply ignored; that is, there are no self-edges in the *Transitions*
  graph.

* The code computes the total space (graph) of *Configurations* and the *Transitions* that connect
  them. The framework allows computing *Paths* on this graph, starting at the initial
  *Configuration* and then looking for the shortest minimal number of *Transitions* leading to a
  *Configuration* that satisfies some *Condition*. Multiple *Conditions* may be specified to define
  a complex *Path*.

## Validation

The code allows applying a validation function to each *State* and overall *Configuration*. Thus,
simply computing the total *Configurations* space can ensure arbitrary validation conditions,
and the framework makes it easy to debug *Invalid* conditions by providing the shortest *Path*
that leads to them.

The *Reaction* logic in each agent will typically `match` some combinations of its *State* and the
*Payload*, which will include a catch-all clause returning *Unexpected*. Thus computing the model
verifies that no unexpected message is received while in a state that does not expect to see it.
Again the framework makes it easy to debug *Unexpected* scenarios by providing the shortest *Path*
leading to them.

The code also verifies the number of sent in-flight *Messages* from each *Agent* does not exceed the
specified threshold. This ensures that no *Agent* is sending an unbounded number of *Messages*.

In addition, the code allows verifying that there is a path from every *Configurations* back to the
initial *Configuration*, which ensures there are no deadlock *Configurations* or a cycle of livelock
*Configurations*. If there are such deadlock/livelock *Configurations*, the framework will provide
the shortest path from the initial *Configuration* to the deadlock/livelock. Debugging these cases
is harder and typically requires looking at the *Configurations* that are reachable from the
deadlock/livelock by consulting the complete list of *Transitions*. Possibly the framework should be
enhanced to dump just the relevant *Transitions* reachable from the deadlock/livelock
*Configurations* in such a case to allow debugging in a large model.

Finally, the code allows for generating *Transition* paths leading to *Configurations* that satisfy
arbitrary *Conditions*, which can be used to ensure that any *Configuration* of interest is actually
reachable from the initial *Configuration*.

Another way to ensure the model is covered is to collect coverage information for the model (see
[grcov](https://marco-c.github.io/2020/11/24/rust-source-based-code-coverage.html)) and looking at
the output to ensure that all the code was reached. Uncovered `match` clauses will indicate that
that specific flow is not reachable from the initial *Configuration*, which could indicate a bug in
the model or that a simpler model may suffice.

## Output

The code allows simply listing all *Agents*, *Configurations* and *Transitions*. This has limited
usefulness, mainly for debugging.

In addition, the code can generate two types of diagrams:

* A GraphViz diagram of all the *States* of some *Agent* *Instance* and the *Actions* that moved
  this *Agent* between these *States*. This is used to visualize the behavior of each *Agent* in
  isolation. This graph can be *Condensed* (ignore internal *State* details and *Agent* *Instance*
  indices), which makes for very usable diagrams visualizing the *Agent* logic.

* A UML sequence diagram of all the *Transitions* between *Configurations* along the shortest *Path*
  between *Configurations* that satisfy arbitrary *Conditions*. This is used to visualize complete
  system scenarios as opposed of just a single *Agent's* logic.

Both types of diagrams use the same colors [legend](legend.html).

## License

`total-space` is distributed under the GNU AFFERO General Public License (Version 3.0). See the
[LICENSE](LICENSE.txt) for details.
