//! Module containing environment types

use crate::{
    error::{status_to_result, OrtError, Result},
    g_ort,
    onnxruntime::custom_logger,
    session::SessionBuilder,
    LoggingLevel,
};
use onnxruntime_sys as sys;
use std::ffi::CString;
use tracing::{debug, error, warn};

/// An [`Environment`](session/struct.Environment.html) is the main entry point of the ONNX Runtime.
///
///
/// Once an environment is created, a [`Session`](../session/struct.Session.html)
/// can be obtained from it.
///
/// **NOTE**: While the [`Environment`](environment/struct.Environment.html) constructor takes a `name` parameter
/// to name the environment, only the first name will be considered if many environments
/// are created.
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use onnxruntime::{environment::Environment, LoggingLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder()
///     .with_name("test")
///     .with_log_level(LoggingLevel::Verbose)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Environment {
    name: String,
    env: *mut sys::OrtEnv,
}

impl Environment {
    /// Create a new environment builder using default values
    /// (name: `default`, log level: [`LoggingLevel::Warning`](../enum.LoggingLevel.html#variant.Warning))
    #[must_use]
    pub fn builder() -> EnvBuilder {
        EnvBuilder {
            name: "default".into(),
            log_level: LoggingLevel::Warning,
        }
    }

    /// Return the name of the current environment
    #[must_use]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub(crate) fn env_ptr(&self) -> *const sys::OrtEnv {
        self.env
    }

    #[tracing::instrument]
    fn new(name: String, log_level: LoggingLevel) -> Result<Environment> {
        debug!("Environment not yet initialized, creating a new one.");

        let mut env_ptr: *mut sys::OrtEnv = std::ptr::null_mut();

        let logging_function: sys::OrtLoggingFunction = Some(custom_logger);
        // FIXME: What should go here?
        let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();

        let cname = CString::new(name.clone()).unwrap();

        let create_env_with_custom_logger = g_ort().CreateEnvWithCustomLogger.unwrap();
        let status = {
            unsafe {
                create_env_with_custom_logger(
                    logging_function,
                    logger_param,
                    log_level.into(),
                    cname.as_ptr(),
                    &mut env_ptr,
                )
            }
        };

        status_to_result(status).map_err(OrtError::Environment)?;

        debug!(
            env_ptr = format!("{:?}", env_ptr).as_str(),
            "Environment created."
        );

        Ok(Environment { env: env_ptr, name })
    }

    /// Create a new [`SessionBuilder`](../session/struct.SessionBuilder.html)
    /// used to create a new ONNX session.
    pub fn new_session_builder(&self) -> Result<SessionBuilder> {
        SessionBuilder::new(self)
    }
}

impl Drop for Environment {
    #[tracing::instrument]
    fn drop(&mut self) {
        let release_env = g_ort().ReleaseEnv.unwrap();
        if self.env.is_null() {
            error!("OrtEnv pointer is null, not dropping.");
        } else {
            unsafe { release_env(self.env) };
        }
    }
}

/// Struct used to build an environment [`Environment`](environment/struct.Environment.html)
///
/// This is the crate's main entry point. An environment _must_ be created
/// as the first step. An [`Environment`](environment/struct.Environment.html) can only be built
/// using `EnvBuilder` to configure it.
///
/// **NOTE**: If the same configuration method (for example [`with_name()`](struct.EnvBuilder.html#method.with_name))
/// is called multiple times, the last value will have precedence.
pub struct EnvBuilder {
    name: String,
    log_level: LoggingLevel,
}

impl EnvBuilder {
    /// Configure the environment with a given name
    ///
    /// **NOTE**: Since ONNX can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    pub fn with_name<S>(mut self, name: S) -> EnvBuilder
    where
        S: Into<String>,
    {
        self.name = name.into();
        self
    }

    /// Configure the environment with a given log level
    ///
    /// **NOTE**: Since ONNX can only define one environment per process,
    /// creating multiple environments using multiple `EnvBuilder` will
    /// end up re-using the same environment internally; a new one will _not_
    /// be created. New parameters will be ignored.
    #[must_use]
    pub fn with_log_level(mut self, log_level: LoggingLevel) -> EnvBuilder {
        self.log_level = log_level;
        self
    }

    /// Commit the configuration to a new [`Environment`](environment/struct.Environment.html)
    pub fn build(self) -> Result<Environment> {
        Environment::new(self.name, self.log_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_log::test;

    #[test]
    fn env_is_initialized() {
        let _env = Environment::builder()
            .with_name("env_is_initialized")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();

        let _env2 = Environment::new("test_2".into(), LoggingLevel::Warning).unwrap();
    }

    #[test]
    fn concurrent_environment_creations() {
        let children: Vec<_> = (0..10)
            .map(|t| {
                std::thread::spawn(move || {
                    let name = format!("concurrent_environment_creation: {}", t);
                    let _env = Environment::builder()
                        .with_name(name)
                        .with_log_level(LoggingLevel::Warning)
                        .build()
                        .unwrap();
                })
            })
            .collect();

        let res: Vec<std::thread::Result<_>> = children
            .into_iter()
            .map(std::thread::JoinHandle::join)
            .collect();
        assert!(res.into_iter().all(|r| std::result::Result::is_ok(&r)));
    }
}
