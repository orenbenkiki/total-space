# total-space [![Build
# Status](https://api.travis-ci.org/orenbenkiki/total-space.svg?branch=master)](https://travis-ci.org/orenbenkiki/total-space)
# [![codecov](https://codecov.io/gh/orenbenkiki/total-space/branch/master/graph/badge.svg)](https://codecov.io/gh/orenbenkiki/total-space)
# [![Docs](https://docs.rs/total-space/badge.svg)](https://docs.rs/crate/total-space)

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
    let app = clap::App::new("...");
    ... add your own command line arguments to control the model ...
    let arg_matches = total_space::add_clap(app);
    ... initialize the agent types, possibly using command line flags ...
    let mut model = TestModel::new(last_agent_type, validators);
    model.do_clap(&arg_matches);
}
```

The trick, of course, is in creating the model. You can see examples of simple models in the tests
of this crate.

## TODO

* Document how to create a model.
* Creating a model requires a lot of boiler plate code, which could be automated by macros.
* Some macros should be converted to procedural derive macros.
* Generating diagrams is not implemented yet.

## License

`total-space` is distributed under the GNU General Public License (Version 3.0). See the
[LICENSE](LICENSE.txt) for details.
