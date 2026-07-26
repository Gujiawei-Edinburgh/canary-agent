use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    #[error("logging error: {0}")]
    Initialization(String),
}

pub type Result<T> = std::result::Result<T, LoggingError>;

pub struct LoggingGuard {
    _writer: DurableFileWriter,
}

#[derive(Clone)]
struct DurableFileWriter {
    file: Arc<Mutex<File>>,
}

struct DurableFileWriterGuard {
    file: Arc<Mutex<File>>,
}

impl Write for DurableFileWriterGuard {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.file
            .lock()
            .map_err(|_| io::Error::other("log file lock is poisoned"))?
            .write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut file = self
            .file
            .lock()
            .map_err(|_| io::Error::other("log file lock is poisoned"))?;
        file.flush()?;
        file.sync_data()
    }
}

impl Drop for DurableFileWriterGuard {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl<'a> MakeWriter<'a> for DurableFileWriter {
    type Writer = DurableFileWriterGuard;

    fn make_writer(&'a self) -> Self::Writer {
        DurableFileWriterGuard {
            file: self.file.clone(),
        }
    }
}

pub fn init_file_logging(state_dir: impl Into<PathBuf>) -> Result<LoggingGuard> {
    let state_dir = state_dir.into();
    create_dir_all(&state_dir).map_err(|error| LoggingError::Initialization(error.to_string()))?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(state_dir.join("lite-agent.log"))
        .map_err(|error| LoggingError::Initialization(error.to_string()))?;
    let writer = DurableFileWriter {
        file: Arc::new(Mutex::new(file)),
    };
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::DEBUG.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .json()
        .with_ansi(false)
        .with_env_filter(filter)
        .with_writer(writer.clone())
        .try_init()
        .map_err(|error| LoggingError::Initialization(error.to_string()))?;

    Ok(LoggingGuard { _writer: writer })
}
