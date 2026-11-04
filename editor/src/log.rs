//! Messages shown in the Output dock.

const MAX_ENTRIES: usize = 5000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Level {
    Info,
    Warning,
    Error,
}

pub struct Entry {
    pub level: Level,
    pub text: String,
}

#[derive(Default)]
pub struct Log {
    pub entries: Vec<Entry>,
}

impl Log {
    pub fn push(&mut self, level: Level, text: impl Into<String>) {
        if self.entries.len() >= MAX_ENTRIES {
            self.entries.drain(..MAX_ENTRIES / 10);
        }
        self.entries.push(Entry { level, text: text.into() });
    }

    pub fn info(&mut self, text: impl Into<String>) {
        self.push(Level::Info, text);
    }

    pub fn error(&mut self, text: impl Into<String>) {
        self.push(Level::Error, text);
    }
}
