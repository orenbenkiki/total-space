/// A macro for implementing data-like (states, payload) for a struct withj a name field.
#[macro_export]
macro_rules! impl_struct_data {
    ($name:ident = $value:expr $(, $from:literal => $to:literal)* $(,)?) => {
        impl_name_by_member! { $name }
        impl_default_by_value! { $name = $value }
        impl_display_by_patched_debug! { $name $(, $from => $to)* }
    };
}

/// A macro for extracting static string name from a struct.
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

/// A macro for implementing data-like (states, payload).
#[macro_export]
macro_rules! impl_enum_data {
    ($name:ident = $value:expr $(, $from:literal => $to:literal)* $(,)?) => {
        impl_default_by_value! { $name = $value }
        impl_name_for_into_static_str! { $name $(, $from => $to)* }
        impl_display_by_patched_debug! { $name $(, $from => $to)* }
    };
}

/// A macro for implementing `Default` for types using a simple value.
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
#[macro_export]
macro_rules! init_agent_type_data {
    ($name:ident, $data:expr) => {
        $name.with(|data| *data.borrow_mut() = Some($data))
    };
}

/// A macro for declaring a global variable containing agent indices.
#[macro_export]
macro_rules! declare_agent_indices {
    ($name:ident) => {
        thread_local! {
            static $name: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::new());
        }
    };
}

/// A macro for declaring a global variable containing singleton agent index.
#[macro_export]
macro_rules! declare_agent_index {
    ($name:ident) => {
        thread_local! {
            static $name: std::cell::RefCell<usize> = std::cell::RefCell::new(usize::max_value());
        }
    };
}

/// A macro for initializing a global variable containing singleton agent index.
#[macro_export]
macro_rules! init_agent_indices {
    ($name:ident, $label:expr, $model:expr) => {{
        $name.with(|refcell| {
            let mut indices = refcell.borrow_mut();
            indices.clear();
            let agent_type = $model.agent_type($label);
            for instance in 0..agent_type.instances_count() {
                indices.push($model.agent_instance_index($label, Some(instance)));
            }
        });
    }};
}

/// A macro for initializing a global variable containing singleton agent index.
#[macro_export]
macro_rules! init_agent_index {
    ($name:ident, $label:expr, $model:expr) => {
        $name.with(|refcell| *refcell.borrow_mut() = $model.agent_instance_index($label, None));
    };
}

/// A macro for accessing a global variable containing agent index.
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
#[macro_export]
macro_rules! agents_count {
    ($name:ident) => {
        $name.with(|refcell| refcell.borrow().len())
    };
}
