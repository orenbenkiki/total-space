// FILE MAYBE TESTED

/// A macro for implementing data-like (states, payload) for a `struct` with a name field.
///
/// Usage is `impl_struct_data! { StructName = default value, "from" => "to", ... }`
/// to implement all the boilerplate for a `struct` named `StructName`, providing it with
/// a default value and displaying it using the results of `Debug` patched by replacing
/// each of the (optional) `"from"` string(s) with its matching `"to"` string.
///
/// The `struct` type needs to `#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]`. It needs to
/// have a `name` field which is a simple `enum` using The `struct` type needs to
/// `#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, strum::IntoStaticStr)]`, which is used as
/// the short name for the state (e.g. in condensed diagrams). Additional fields would be used to
/// carry detailed state data.
#[macro_export]
macro_rules! impl_struct_data {
    ($name:ident = $value:expr $(, $from:literal => $to:literal)* $(,)?) => {
        impl KeyLike for $name {}
        impl DataLike for $name {}
        impl_name_by_member! { $name }
        impl_default_by_value! { $name = $value }
        impl_display_by_patched_debug! { $name $(, $from => $to)* }
    };
}

/// A macro for extracting static string name from a struct.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_name_by_member {
    ($name:ident) => {
        impl total_space::Name for $name {
            fn name(&self) -> String {
                let name: &'static str = std::convert::From::from(self.name);
                name.to_string()
            }
        }
    };
}

/// A macro for implementing data-like (states, payload) for an `enum`.
///
/// Usage is `impl_struct_data! { EnumName = default value, "from" => "to", ... }`
/// to implement all the boilerplate for an `enum` named `EnumName`, providing it with
/// a default value and displaying it using the results of `Debug` patched by replacing
/// each of the (optional) `"from"` string(s) with its matching `"to"` string.
///
/// The `enum` type needs to `#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug,
/// strum::IntoStaticStr)]`. The variant name will be used as the short name for the state (e.g. in
/// condensed diagrams). Additional fields would be used to carry detailed state data.
#[macro_export]
macro_rules! impl_enum_data {
    ($name:ident = $value:expr $(, $from:literal => $to:literal)* $(,)?) => {
        impl KeyLike for $name {}
        impl DataLike for $name {}
        impl_default_by_value! { $name = $value }
        impl_name_for_into_static_str! { $name $(, $from => $to)* }
        impl_display_by_patched_debug! { $name, "name=" => "" $(, $from => $to)* }
    };
}

/// A macro for implementing `Default` for types using a simple value.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_default_by_value {
    ($name:ident = $value:expr) => {
        impl Default for $name {
            fn default() -> Self {
                $value
            }
        }
    };
}

/// A macro for implementing `Name` for enums annotated by `strum::IntoStaticStr`.
///
/// This should become unnecessary once `IntoStaticStr` allows converting a reference to a static
/// string, see `<https://github.com/Peternator7/strum/issues/142>`.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_name_for_into_static_str {
    ($name:ident $(, $from:literal => $to:literal)* $(,)?) => {
        impl total_space::Name for $name {
            fn name(&self) -> String {
                let static_str: &'static str = self.into();
                let string = static_str.to_string();
                $(
                    let string = string.replace($from, $to);
                )*
                string
            }
        }
    };
}

/// A macro for implementing `Debug` for data which has `DisplayDebug`.
///
/// This should be concerted to a derive macro.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_display_by_patched_debug {
    ($name:ident $(, $from:literal => $to:literal)* $(,)?) => {
        impl Display for $name {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatterResult {
                let string = format!("{:?}", self)
                    .replace(" ", "")
                    .replace(":", "=")
                    .replace("{", "(")
                    .replace("}", ")");
                $(
                    let string = string.replace($from, $to);
                )*
                write!(formatter, "{}", string)
            }
        }
    };
}

/// A macro for declaring a global variable containing an agent type.
///
/// Usage is `declare_agent_type_data! { TYPE_NAME, StateTypeName, ModelTypeName }`. It declares a
/// global variable named `TYPE_NAME` which contains
/// `std::cell::RefCell<Option<std::rc::Rc<AgentTypeData::<StateTypeName, <ModelTypeName as
/// MetaModel>::StateId, <ModelTypeName as MetaModel>::Payload >>>`, the details of which shouldn't
/// interest you much since other macros access this in a friendly way.
#[macro_export]
macro_rules! declare_agent_type_data {
    ($name:ident, $agent:ident, $model:ident) => {
        std::thread_local! {
            static $name: std::cell::RefCell<
                Option<
                   std::rc::Rc<
                        AgentTypeData::<
                            $agent,
                            <$model as MetaModel>::StateId,
                            <$model as MetaModel>::Payload,
                        >
                    >
                >
            > = std::cell::RefCell::new(None);
        }
    };
}

/// A macro for initializing a global variable containing an agent type.
///
/// This initializes the variable declared by `declare_agent_type_data`. Usage is
/// `init_agent_type_data! { TYPE_NAME, agent_type }` passing it the the
/// `agent_type` created to initialize the model.
#[macro_export]
macro_rules! init_agent_type_data {
    ($name:ident, $agent_type:expr) => {
        $name.with(|data| *data.borrow_mut() = Some($agent_type.clone()))
    };
}

/// A macro for declaring a global variable containing agent indices.
///
/// Usage is `declare_agent_indices! { NAME }`. It declares a global variable named `NAME` to hold
/// the agent indices for instances of some (non-singleton) agent type.
#[macro_export]
macro_rules! declare_agent_indices {
    ($name:ident) => {
        thread_local! {
            static $name: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::new());
        }
    };
}

/// A macro for declaring a global variable containing singleton agent index.
///
/// Usage is `declare_agent_index! { NAME }`. It declares a global variable named `NAME` to hold the
/// agent index of the singleton instance of some agent type.
#[macro_export]
macro_rules! declare_agent_index {
    ($name:ident) => {
        thread_local! {
            static $name: std::cell::RefCell<usize> = std::cell::RefCell::new(usize::max_value());
        }
    };
}

/// A macro for initializing a global variable containing singleton agent index.
///
/// Usage is `init_agent_indices! { NAME, agent_type }`. It will initialize the global variable
/// created by `declare_agent_indices` using the (non-singleton) `agent_type` created to initialize
/// the model.
#[macro_export]
macro_rules! init_agent_indices {
    ($name:ident, $agent_type:expr) => {{
        assert!(!$agent_type.is_singleton());
        $name.with(|refcell| {
            let mut indices = refcell.borrow_mut();
            indices.clear();
            for instance in 0..$agent_type.instances_count() {
                indices.push($agent_type.first_index() + instance);
            }
        });
    }};
}

/// A macro for initializing a global variable containing singleton agent index.
///
/// Usage is `init_agent_index! { NAME, agent_type }`. It will initialize the global variable
/// created by `declare_agent_index` using the (singleton) `agent_type` created to initialize the
/// model.
#[macro_export]
macro_rules! init_agent_index {
    ($name:ident, $agent_type:expr) => {
        assert!($agent_type.is_singleton());
        $name.with(|refcell| *refcell.borrow_mut() = $agent_type.first_index());
    };
}

/// A macro for accessing a global variable containing agent index.
///
/// Usage is `agent_index!(NAME)` for singleton agent types or `agent_index!(NAME[instance])` non
/// for singleton agent types, for accessing the global variable declared by `declare_agent_index`
/// or `declare_agent_indices`.
#[macro_export]
macro_rules! agent_index {
    ($name:ident) => {{
        $name.with(|refcell| *refcell.borrow())
    }};
    ($name:ident[$index:expr]) => {
        $name.with(|refcell| refcell.borrow()[$index])
    };
}

/// A macro for accessing the number of agent instances.
///
/// Usage is `agents_count!(NAME)`. It will access the global variable declared by
/// `declare_agent_indices` to access the number of instances of agents of some (non-singleton)
/// type.
#[macro_export]
macro_rules! agents_count {
    ($name:ident) => {
        $name.with(|refcell| refcell.borrow().len())
    };
}

/// A macro for activity processing one out several payloads.
///
/// Usage is `activity_alternatives!(payload, payload, ...)` to specify an activity that causes the
/// agent to process one of several alternative payloads.
#[macro_export]
macro_rules! activity_alternatives {
    ($payload1:expr, $payload2:expr $(,)?) => {
        total_space::Activity::Process1Of([
            Some($payload1),
            Some($payload2),
            None,
            None,
            None,
            None,
        ])
    };
    ($payload1:expr, $payload2:expr $(,)?) => {
        total_space::Activity::Process1Of([
            Some($payload1),
            Some($payload2),
            None,
            None,
            None,
            None,
        ])
    };
    ($payload1:expr, $payload2:expr, $payload3:expr $(,)?) => {
        total_space::Activity::Process1Of([
            Some($payload1),
            Some($payload2),
            Some($payload3),
            None,
            None,
            None,
        ])
    };
    ($payload1:expr, $payload2:expr, $payload3:expr, $payload4:expr $(,)?) => {
        total_space::Activity::Process1Of([
            Some($payload1),
            Some($payload2),
            Some($payload3),
            Some($payload4),
            None,
            None,
        ])
    };
    ($payload1:expr, $payload2:expr, $payload3:expr, $payload4:expr, $payload5:expr $(,)?) => {
        total_space::Activity::Process1Of([
            Some($payload1),
            Some($payload2),
            Some($payload3),
            Some($payload4),
            Some($payload5),
            None,
        ])
    };
    ($payload1:expr, $payload2:expr, $payload3:expr, $payload4:expr, $payload5:expr, $payload6:expr $(,)?) => {
        total_space::Activity::Process1Of([
            Some($payload1),
            Some($payload2),
            Some($payload3),
            Some($payload4),
            Some($payload5),
            Some($payload6),
        ])
    };
    ($_:tt) => {
        compile_error!("expected 2 to 6 payloads");
    };
}

/// A macro for one of several alternatives reaction.
///
/// Usage is `reaction_alternatives!(action, action, ...)` to specify that the agent may take one of
/// several alternative actions when reacting to a payload.
#[macro_export]
macro_rules! reaction_alternatives {
    ($action1:expr, $action2:expr $(,)?) => {
        total_space::Reaction::Do1Of([Some(action1), Some(action2), None, None, None, None])
    };
    ($action1:expr, $action2:expr, $action3:expr $(,)?) => {
        total_space::Reaction::Do1Of([
            Some(action1),
            Some(action2),
            Some(action3),
            None,
            None,
            None,
        ])
    };
    ($action1:expr, $action2:expr, $action3:expr, $action4:expr $(,)?) => {
        total_space::Reaction::Do1Of([
            Some(action1),
            Some(action2),
            Some(action3),
            Some(action4),
            None,
            None,
        ])
    };
    ($action1:expr, $action2:expr, $action3:expr, $action4:expr, $action5:expr $(,)?) => {
        total_space::Reaction::Do1Of([
            Some(action1),
            Some(action2),
            Some(action3),
            Some(action4),
            Some(action5),
            None,
        ])
    };
    ($action1:expr, $action2:expr, $action3:expr, $action4:expr, $action5:expr, $action6:expr $(,)?) => {
        total_space::Reaction::Do1Of([
            Some(action1),
            Some(action2),
            Some(action3),
            Some(action4),
            Some(action5),
            Some(action6),
        ])
    };
    ($_:tt) => {
        compile_error!("expected 2 to 6 actions");
    };
}

/// A macro for an action sending several messages.
///
/// Usage is `action_sends!(emit, emit, ...)` to create an action that emits multiple messages.
#[macro_export]
macro_rules! action_sends {
    ($emit1:expr, $emit2:expr $(,)?) => {
        total_space::Action::Sends([Some($emit1), Some($emit2), None, None, None, None])
    };
    ($emit1:expr, $emit2:expr, $emit3:expr $(,)?) => {
        total_space::Action::Sends([Some($emit1), Some($emit2), Some($emit3), None, None, None])
    };
    ($emit1:expr, $emit2:expr, $emit3:expr, $emit4:expr $(,)?) => {
        total_space::Action::Sends([
            Some($emit1),
            Some($emit2),
            Some($emit3),
            Some($emit4),
            None,
            None,
        ])
    };
    ($emit1:expr, $emit2:expr, $emit3:expr, $emit4:expr, $emit5:expr $(,)?) => {
        total_space::Action::Sends([
            Some($emit1),
            Some($emit2),
            Some($emit3),
            Some($emit4),
            Some($emit5),
            None,
        ])
    };
    ($emit1:expr, $emit2:expr, $emit3:expr, $emit4:expr, $emit5:expr, $emit6:expr $(,)?) => {
        total_space::Action::Sends([
            Some($emit1),
            Some($emit2),
            Some($emit3),
            Some($emit4),
            Some($emit5),
            Some($emit6),
        ])
    };
    ($_:tt) => {
        compile_error!("expected 2 to 6 emits");
    };
}

/// A macro for an action changing the state and sending several messages.
///
/// Usage is `action_sends!(state, emit, emit, ...)` to create an action that changes the agent's
/// state and emits multiple messages.
#[macro_export]
macro_rules! action_change_and_sends {
    ($state:expr, $emit1:expr, $emit2:expr $(,)?) => {
        total_space::Action::ChangeAndSends(
            $state,
            [Some($emit1), Some($emit2), None, None, None, None],
        )
    };
    ($state:expr, $emit1:expr, $emit2:expr, $emit3:expr $(,)?) => {
        total_space::Action::ChangeAndSends(
            $state,
            [Some($emit1), Some($emit2), Some($emit3), None, None, None],
        )
    };
    ($state:expr, $emit1:expr, $emit2:expr, $emit3:expr, $emit4:expr $(,)?) => {
        total_space::Action::ChangeAndSends(
            $state,
            [
                Some($emit1),
                Some($emit2),
                Some($emit3),
                Some($emit4),
                None,
                None,
            ],
        )
    };
    ($state:expr, $emit1:expr, $emit2:expr, $emit3:expr, $emit4:expr, $emit5:expr $(,)?) => {
        total_space::Action::ChangeAndSends(
            $state,
            [
                Some($emit1),
                Some($emit2),
                Some($emit3),
                Some($emit4),
                Some($emit5),
                None,
            ],
        )
    };
    ($state:expr, $emit1:expr, $emit2:expr, $emit3:expr, $emit4:expr, $emit5:expr, $emit6:expr $(,)?) => {
        total_space::Action::ChangeAndSends(
            $state,
            [
                Some($emit1),
                Some($emit2),
                Some($emit3),
                Some($emit4),
                Some($emit5),
                Some($emit6),
            ],
        )
    };
    ($_:tt) => {
        compile_error!("expected state and 2 to 6 emits");
    };
}

/// A macro for static assertion on the size of the configuration hash entry.
///
/// Usage is `assert_configuration_hash_entry_size!(ModelType, size)` to statically assert that the
/// size of a configuration hash entry is as expected, to ensure this size is cache-friendly.
#[macro_export]
macro_rules! assert_configuration_hash_entry_size {
    ($model:ident, $size:literal) => {
        const _: usize = 0
            - (std::mem::size_of::<<$model as MetaModel>::ConfigurationHashEntry>() != $size)
                as usize;
    };
}

/// A macro for iterating on all the agents of some type in a configuration.
///
/// Usage is `agent_states_iter!(configuration, TYPE_NAME, iterations)` to iterate on the states of
/// the instances of the configuration's agents of the global `TYPE_NAME` variable declared by
/// `declare_agent_type_data`. Iteration is some iterator method, as if we wrote
/// `states.iter().iteration` (e.g., `iteration` could be `for_each(|state| ...)`.
#[macro_export]
macro_rules! agent_states_iter {
    ($configuration:expr, $name:ident, $($iter:tt)*) => {
        $name.with(|refcell| {
            if let Some(agent_type) = refcell.borrow().as_ref() {
                (0..agent_type.instances_count())
                    .map(|agent_instance| {
                        let agent_index = agent_type.first_index() + agent_instance;
                        let state_id = $configuration.state_ids[agent_index];
                        agent_type.get_state(state_id)
                    })
                    .$($iter)*
            } else {
                unreachable!()
            }
        })
    };
}

/// A macro for iterating on all the in-flight messages configuration.
///
/// Usage is `messages_iter!(model, configuration, iterations)` to iterate on the in-flight messages
/// of the configuration. Iteration is some iterator method, as if we wrote
/// `messages.iter().iteration` (e.g., `iteration` could be `for_each(|message| ...)`.
#[macro_export]
macro_rules! messages_iter {
    ($model:expr, $configuration:expr, $($iter:tt)*) => {
        $configuration
            .message_ids
            .iter()
            .take_while(|message_id| message_id.is_valid())
            .map(|message_id| $model.get_message(*message_id))
            .$($iter)*
    };
}
