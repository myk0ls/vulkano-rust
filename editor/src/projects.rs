//! What the editor remembers across projects (recent projects, where Blender is), and
//! talking to the operating system: switching projects and showing files.

use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_RECENT: usize = 10;

/// Kept per user, not per project, in the system's config folder.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlobalConfig {
    pub recent: Vec<PathBuf>,
    /// The Blender used to import .blend files. Empty means find it.
    pub blender: String,
}

fn config_file() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".config")))
    }?;
    Some(base.join("vulkano-editor").join("config.txt"))
}

impl GlobalConfig {
    pub fn load() -> Self {
        let text = config_file().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
        Self::parse(&text)
    }

    fn parse(text: &str) -> Self {
        let mut config = Self::default();
        for line in text.lines() {
            match line.split_once('=') {
                Some(("recent", path)) if !path.trim().is_empty() => config.recent.push(PathBuf::from(path.trim())),
                Some(("blender", path)) => config.blender = path.trim().to_string(),
                _ => {}
            }
        }
        config
    }

    fn format(&self) -> String {
        let mut text = format!("blender={}\n", self.blender);
        for path in &self.recent {
            text += &format!("recent={}\n", path.display());
        }
        text
    }

    pub fn save(&self) {
        if let Some(path) = config_file() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(path, self.format());
        }
    }

    /// Put a project at the top of the recent list.
    pub fn remember(&mut self, dir: &Path) {
        let dir = absolute(dir);
        self.recent.retain(|p| p != &dir);
        self.recent.insert(0, dir);
        self.recent.truncate(MAX_RECENT);
    }

    pub fn blender_path(&self) -> Option<PathBuf> {
        (!self.blender.trim().is_empty()).then(|| PathBuf::from(self.blender.trim()))
    }
}

/// A path made absolute against the current folder, without Windows' `\\?\` prefix.
pub fn absolute(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    // Tidy "a/./b" so paths compare equal and look right.
    joined.components().filter(|c| !matches!(c, std::path::Component::CurDir)).collect()
}

/// Start a new editor in another project. The caller quits this one.
pub fn relaunch(project: &Path) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    Command::new(exe)
        .arg("--project")
        .arg(project)
        .current_dir(project)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("could not start the editor for {}: {e}", project.display()))
}

/// Show a file or folder in the system's file manager, with a file selected.
pub fn reveal(path: &Path) -> Result<(), String> {
    let path = absolute(path);
    #[cfg(windows)]
    let result = {
        use std::os::windows::process::CommandExt;
        let shown = path.to_string_lossy().replace('/', "\\");
        if path.is_dir() {
            Command::new("explorer").arg(&shown).spawn()
        } else {
            // explorer wants "/select," and the quoted path as one argument.
            Command::new("explorer").raw_arg(format!("/select,\"{shown}\"")).spawn()
        }
    };
    #[cfg(not(windows))]
    let result = if cfg!(target_os = "macos") {
        if path.is_dir() { Command::new("open").arg(&path).spawn() } else { Command::new("open").arg("-R").arg(&path).spawn() }
    } else {
        let folder = if path.is_dir() { path.clone() } else { path.parent().map(Path::to_path_buf).unwrap_or_default() };
        Command::new("xdg-open").arg(folder).spawn()
    };
    result.map(|_| ()).map_err(|e| format!("could not open the file manager: {e}"))
}

/// Open a file with the program the system uses for it.
pub fn open_with_system(path: &Path) -> Result<(), String> {
    let path = absolute(path);
    let result = if cfg!(windows) {
        Command::new("explorer").arg(path.to_string_lossy().replace('/', "\\")).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(&path).spawn()
    } else {
        Command::new("xdg-open").arg(&path).spawn()
    };
    result.map(|_| ()).map_err(|e| format!("could not open {}: {e}", path.display()))
}

/// Where new projects go unless the user picks somewhere else.
pub fn default_projects_folder() -> PathBuf {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from).unwrap_or_default();
    home.join("Documents").join("Vulkano Projects")
}

/// A project name turned into a folder name.
pub fn folder_name(name: &str) -> String {
    name.trim()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '_' })
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips() {
        let config = GlobalConfig {
            recent: vec![PathBuf::from("C:\\games\\one"), PathBuf::from("/home/me/two")],
            blender: "C:\\Blender\\blender.exe".into(),
        };
        assert_eq!(GlobalConfig::parse(&config.format()), config);
    }

    #[test]
    fn remembering_moves_a_project_to_the_top() {
        let mut config = GlobalConfig::default();
        config.remember(Path::new("/a"));
        config.remember(Path::new("/b"));
        config.remember(Path::new("/a"));
        assert_eq!(config.recent.len(), 2);
        assert!(config.recent[0].ends_with("a"));
    }

    #[test]
    fn folder_names_are_safe() {
        assert_eq!(folder_name(" My Game: 2 "), "My Game_ 2");
    }
}
