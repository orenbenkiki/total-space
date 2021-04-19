use crate::diagrams::*;
use crate::models::*;
use crate::utilities::*;

use clap::App;
use clap::Arg;
use clap::ArgMatches;
use clap::SubCommand;
use std::fs::File;
use std::io::stdout;
use std::io::BufWriter;
use std::io::Write;
use std::process::exit;
use std::str::FromStr;

/// Add clap commands and flags to a clap application.
pub fn add_clap<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("progress")
            .short("p")
            .long("progress-every")
            .default_value("0")
            .help("print configurations as they are reached"),
    )
    .arg(
        Arg::with_name("size")
            .short("s")
            .long("size")
            .value_name("COUNT")
            .help(
                "pre-allocate arrays to cover this number of configurations, for faster operation",
            )
            .default_value("AUTO"),
    )
    .arg(
        Arg::with_name("reachable")
            .short("r")
            .long("reachable")
            .help(
                "ensure that the initial configuration is reachable from all other configurations",
            ),
    )
    .arg(
        Arg::with_name("invalid")
            .short("i")
            .long("invalid")
            .help("allow for invalid configurations (but do not explore beyond them)"),
    )
    .arg(
        Arg::with_name("output")
            .short("o")
            .long("output")
            .value_name("FILE")
            .help(
                "redirect stdout to a file (or to '-' for the default); \
                 NOTE: to repeat commands, prefix each one with this, \
                 e.g. `main -p 100 -o - agents -o conditions.txt conditional`."
            )
    )
    .subcommand(
        SubCommand::with_name("agents")
            .about("list the agents of the model (does not compute the model)"),
    )
    .subcommand(SubCommand::with_name("conditions").about(
        "list the conditions which can be used to identify configurations \
                   (does not compute the model)",
    ))
    .subcommand(
        SubCommand::with_name("compute").about("only compute the model and print basic statistics"),
    )
    .subcommand(
        SubCommand::with_name("configurations").about("list the configurations of the model"),
    )
    .subcommand(SubCommand::with_name("transitions").about("list the transitions of the model"))
    .subcommand(
        SubCommand::with_name("path")
            .about("list transitions for a path between configurations")
            .arg(Arg::with_name("CONDITION").multiple(true).help(
                "the name of at least two conditions identifying configurations along the path, \
                          which may be prefixed with ! to negate the condition",
            )),
    )
    .subcommand(
        SubCommand::with_name("sequence")
            .about("generate a PlantUML sequence diagram for a path between configurations")
            .arg(Arg::with_name("CONDITION").multiple(true).help(
                "the name of at least two conditions identifying configurations along the path, \
                          which may be prefixed with ! to negate the condition",
            )),
    )
    .subcommand(
        SubCommand::with_name("states")
            .about("generate a GraphViz dot diagrams for the states of a specific agent")
            .arg(
                Arg::with_name("AGENT")
                    .help("the name of the agent to generate a diagrams for the states of"),
            )
            .arg(
                Arg::with_name("names-only")
                    .short("n")
                    .long("names-only")
                    .help("condense graph nodes considering only the state & payload names"),
            )
            .arg(
                Arg::with_name("merge-instances")
                    .short("m")
                    .long("merge-instances")
                    .help("condense graph nodes considering only the agent type"),
            )
            .arg(
                Arg::with_name("final-replaced")
                    .short("f")
                    .long("final-replaced")
                    .help("condense graph nodes considering only the final (replaced) payload"),
            )
            .arg(
                Arg::with_name("condensed")
                    .short("c")
                    .long("condensed")
                    .help("most condensed graph (implies --names-only, --merge-instances and --final-replaced)"),
            ),
    )
}

fn get_arg_matches<'a>(app: &'a mut App, args: &[String]) -> ArgMatches<'a> {
    match app.get_matches_from_safe_borrow(args.iter()) {
        Ok(arg_matches) => arg_matches,
        Err(reason) => {
            eprintln!("{}", reason);
            exit(1);
        }
    }
}

fn next_output_flag_position(base: usize, args: &[String]) -> Option<usize> {
    args.iter()
        .position(|arg| *arg == "-o" || *arg == "--output")
        .map(|position| base + position)
}

fn first_sub_command_position(args: &[String]) -> Option<usize> {
    match next_output_flag_position(0, args) {
        None => None,
        Some(first_position) => {
            match next_output_flag_position(first_position + 1, &args[first_position + 1..]) {
                None => None,
                Some(_) => Some(first_position),
            }
        }
    }
}

/// Return arg matches for building the model.
///
/// This strips away the repeated commands, if any.
pub fn get_model_arg_matches<'a>(app: &'a mut App) -> ArgMatches<'a> {
    let args: Vec<String> = std::env::args().collect();
    match first_sub_command_position(&args) {
        None => get_arg_matches(app, &args),
        Some(first_position) => get_arg_matches(app, &args[..first_position]),
    }
}

/// Execute operations on a model using clap commands.
pub trait ClapModel {
    /// Execute a full clap system command line (possibly with multiple commands).
    fn do_clap(&mut self, app: &mut App) {
        let args: Vec<String> = std::env::args().collect();
        match first_sub_command_position(&args) {
            None => {
                self.do_clap_command(&get_arg_matches(app, &args), &mut BufWriter::new(stdout()));
            }
            Some(very_first_position) => {
                let mut prefix: Vec<String> = args[..very_first_position].to_vec();
                let mut first_position = very_first_position;
                loop {
                    match next_output_flag_position(first_position + 1, &args[first_position + 1..])
                    {
                        None => {
                            prefix.extend_from_slice(&args[first_position..]);
                            self.do_clap_command(
                                &get_arg_matches(app, &prefix),
                                &mut BufWriter::new(stdout()),
                            );
                            return;
                        }
                        Some(next_position) => {
                            prefix.extend_from_slice(&args[first_position..next_position]);
                            self.do_clap_command(
                                &get_arg_matches(app, &prefix),
                                &mut BufWriter::new(stdout()),
                            );
                            prefix.truncate(very_first_position);
                            first_position = next_position;
                        }
                    }
                }
            }
        }
    }

    /// Execute a single clap subcommand (including the output redirection flag).
    fn do_clap_command(&mut self, arg_matches: &ArgMatches, default_stdout: &mut dyn Write) {
        if let Some(output) = arg_matches.value_of("output") {
            // BEGIN NOT TESTED
            if output != "-" {
                self.dispatch_clap_command(arg_matches, &mut File::create(output).unwrap());
                return;
            }
            // END NOT TESTED
        }
        self.dispatch_clap_command(arg_matches, default_stdout);
    }

    /// Dispatch a single clap subcommand (after processing the output redirection flag).
    fn dispatch_clap_command(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) {
        if !self.do_clap_agents(arg_matches, stdout)
            && !self.do_clap_conditions(arg_matches, stdout)
            && !self.do_clap_compute(arg_matches, stdout)
            && !self.do_clap_configurations(arg_matches, stdout)
            && !self.do_clap_transitions(arg_matches, stdout)
            && !self.do_clap_path(arg_matches, stdout)
            && !self.do_clap_sequence(arg_matches, stdout)
            && !self.do_clap_states(arg_matches, stdout)
        {
            // BEGIN NOT TESTED
            eprintln!("no command specified; use --help to list the commands");
            exit(1);
            // END NOT TESTED
        }
    }

    /// Compute the model.
    fn do_compute(&mut self, arg_matches: &ArgMatches);

    /// Execute the `agents` clap subcommand, if requested to.
    ///
    /// This doesn't compute the model.
    fn do_clap_agents(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `conditions` clap subcommand, if requested to.
    ///
    /// This doesn't compute the model.
    fn do_clap_conditions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Only compute the model (no output).
    fn do_clap_compute(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `configurations` clap subcommand, if requested to.
    ///
    /// This computes the model.
    fn do_clap_configurations(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `transitions` clap subcommand, if requested to.
    ///
    /// This computes the model.
    fn do_clap_transitions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `path` clap subcommand, if requested to.
    fn do_clap_path(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `sequence` clap subcommand, if requested to.
    fn do_clap_sequence(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;

    /// Execute the `states` clap subcommand, if requested to.
    fn do_clap_states(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool;
}

impl<
        StateId: IndexLike,
        MessageId: IndexLike,
        InvalidId: IndexLike,
        ConfigurationId: IndexLike,
        Payload: DataLike,
        const MAX_AGENTS: usize,
        const MAX_MESSAGES: usize,
    > ClapModel
    for Model<StateId, MessageId, InvalidId, ConfigurationId, Payload, MAX_AGENTS, MAX_MESSAGES>
{
    fn do_compute(&mut self, arg_matches: &ArgMatches) {
        let progress_every = arg_matches.value_of("progress").unwrap();
        self.print_progress_every = usize::from_str(progress_every).expect("invalid progress rate");
        self.allow_invalid_configurations = arg_matches.is_present("invalid");
        self.ensure_init_is_reachable = arg_matches.is_present("reachable");

        self.compute();

        if self.ensure_init_is_reachable {
            self.assert_init_is_reachable();
        }
    }

    fn do_clap_agents(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("agents") {
            Some(_) => {
                self.print_agents(stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_conditions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("conditions") {
            Some(_) => {
                self.print_conditions(stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_compute(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("compute") {
            Some(_) => {
                self.do_compute(arg_matches);
                self.print_stats(stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_configurations(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("configurations") {
            Some(_) => {
                self.do_compute(arg_matches);
                self.print_configurations(stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_transitions(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("transitions") {
            Some(_) => {
                self.do_compute(arg_matches);
                self.print_transitions(stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_path(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("path") {
            Some(matches) => {
                let steps = self.collect_steps("path", matches);
                self.do_compute(arg_matches);
                let path = self.collect_path(steps);
                self.print_path(&path, stdout);
                true
            }
            None => false,
        }
    }

    fn do_clap_sequence(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("sequence") {
            Some(matches) => {
                let steps = self.collect_steps("sequence", matches);
                self.do_compute(arg_matches);
                let path = self.collect_path(steps);
                let mut sequence_steps = self.collect_sequence_steps(&path[1..]);
                let first_configuration_id = path[1].from_configuration_id;
                let last_configuration_id = path.last().unwrap().to_configuration_id;
                self.print_sequence_diagram(
                    first_configuration_id,
                    last_configuration_id,
                    &mut sequence_steps,
                    stdout,
                );
                true
            }
            None => false,
        }
    }

    fn do_clap_states(&mut self, arg_matches: &ArgMatches, stdout: &mut dyn Write) -> bool {
        match arg_matches.subcommand_matches("states") {
            Some(matches) => {
                let agent_label = matches
                    .value_of("AGENT")
                    .expect("the states command requires a single agent name, none were given");
                let condense = Condense {
                    names_only: matches.is_present("names-only") || matches.is_present("condensed"),
                    merge_instances: matches.is_present("merge-instances")
                        || matches.is_present("condensed"),
                    final_replaced: matches.is_present("final-replaced")
                        || matches.is_present("condensed"),
                };
                let agent_index = self
                    .agent_label_index(agent_label)
                    .unwrap_or_else(|| panic!("unknown agent {}", agent_label));

                self.do_compute(arg_matches);
                self.print_states_diagram(&condense, agent_index, stdout);

                true
            }
            None => false, // NOT TESTED
        }
    }
}
