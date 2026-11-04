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
            .args(["run", "--release", "--quiet", "-p", "vulkano-game", "--"])
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

impl Drop for GameRunner {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
