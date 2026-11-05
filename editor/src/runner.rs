//! Runs the game as a separate process, the way Godot's Play button does, so it runs
//! the real game code and a crash can't take the editor down.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::log::{Level, Log};

pub struct GameRunner {
    child: Option<Child>,
    sender: Sender<(Level, String)>,
    receiver: Receiver<(Level, String)>,
}

impl GameRunner {
    pub fn new() -> Self {
        let (sender, receiver) = channel();
        Self {
            child: None,
            sender,
            receiver,
        }
    }

    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }

    pub fn start(&mut self, scene: &Path, log: &mut Log) {
        self.stop(log);
        log.info(format!("Running {} (building the game if needed)…", scene.display()));
        let spawned = Command::new("cargo")
            // JSON keeps compiler output apart from the game's, so warnings stay out of
            // the Output dock and only build errors are shown.
            .args(["run", "--release", "--quiet", "--message-format=json", "-p", "vulkano-game", "--"])
            .arg(scene)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        match spawned {
            Ok(mut child) => {
                if let Some(out) = child.stdout.take() {
                    self.forward(out, Level::Info);
                }
                if let Some(err) = child.stderr.take() {
                    self.forward(err, Level::Warning);
                }
                self.child = Some(child);
            }
            Err(e) => log.error(format!("Could not start cargo: {e}")),
        }
    }

    fn forward(&self, stream: impl Read + Send + 'static, level: Level) {
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                let (level, line) = match classify(&line) {
                    Line::Game => (level, line),
                    Line::BuildError(text) => (Level::Error, text),
                    Line::BuildNoise => continue,
                };
                if sender.send((level, line)).is_err() {
                    break;
                }
            }
        });
    }

    pub fn stop(&mut self, log: &mut Log) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
            log.info("Game stopped.");
        }
    }

    /// Forward the game's output to the log and notice when it exits.
    pub fn poll(&mut self, log: &mut Log) {
        while let Ok((level, line)) = self.receiver.try_recv() {
            log.push(level, format!("[game] {line}"));
        }
        if let Some(child) = &mut self.child {
            if let Ok(Some(status)) = child.try_wait() {
                self.child = None;
                if status.success() {
                    log.info("Game exited.");
                } else {
                    log.error(format!("Game exited with {status}."));
                }
            }
        }
    }
}

enum Line {
    Game,
    BuildError(String),
    BuildNoise,
}

/// Cargo's JSON messages are objects with a "reason"; everything else is from the game.
fn classify(line: &str) -> Line {
    if !line.starts_with('{') {
        return Line::Game;
    }
    let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
        return Line::Game;
    };
    let Some(reason) = message.get("reason").and_then(|r| r.as_str()) else {
        return Line::Game;
    };
    let diagnostic = message.get("message");
    let is_error = reason == "compiler-message"
        && diagnostic.and_then(|d| d.get("level")).and_then(|l| l.as_str()) == Some("error");
    match diagnostic.and_then(|d| d.get("rendered")).and_then(|r| r.as_str()) {
        Some(text) if is_error => Line::BuildError(text.trim_end().to_string()),
        _ => Line::BuildNoise,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_output_passes_through() {
        assert!(matches!(classify("[script] hello"), Line::Game));
        assert!(matches!(classify("{not json"), Line::Game));
        assert!(matches!(classify(r#"{"some":"game json"}"#), Line::Game));
    }

    #[test]
    fn build_warnings_are_dropped_and_errors_kept() {
        let warning = r#"{"reason":"compiler-message","message":{"level":"warning","rendered":"unused"}}"#;
        assert!(matches!(classify(warning), Line::BuildNoise));
        assert!(matches!(classify(r#"{"reason":"compiler-artifact"}"#), Line::BuildNoise));
        let error = r#"{"reason":"compiler-message","message":{"level":"error","rendered":"bad code\n"}}"#;
        assert!(matches!(classify(error), Line::BuildError(text) if text == "bad code"));
    }
}

impl Drop for GameRunner {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
