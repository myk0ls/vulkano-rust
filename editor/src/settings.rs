//! Editor preferences, kept in `.vulkano-editor` in the project root.

const SETTINGS_FILE: &str = ".vulkano-editor";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Settings {
    pub theme: String,
    /// The command that opens a script. Empty means detect one.
    pub script_editor: String,
}

impl Settings {
    pub fn load() -> Self {
        Self::parse(&std::fs::read_to_string(SETTINGS_FILE).unwrap_or_default())
    }

    fn parse(text: &str) -> Self {
        let mut settings = Self::default();
        for line in text.lines() {
            match line.split_once('=') {
                Some(("theme", value)) => settings.theme = value.trim().to_string(),
                Some(("script_editor", value)) => settings.script_editor = value.trim().to_string(),
                // Older versions stored only the theme's name.
                None if !line.trim().is_empty() => settings.theme = line.trim().to_string(),
                _ => {}
            }
        }
        settings
    }

    fn format(&self) -> String {
        format!("theme={}\nscript_editor={}\n", self.theme, self.script_editor)
    }

    pub fn save(&self) {
        let _ = std::fs::write(SETTINGS_FILE, self.format());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip() {
        let settings = Settings {
            theme: "Gruvbox".into(),
            script_editor: "flatpak run dev.zed.Zed {project} {file}".into(),
        };
        assert_eq!(Settings::parse(&settings.format()), settings);
    }

    #[test]
    fn the_old_theme_only_format_still_loads() {
        assert_eq!(Settings::parse("Godot Dark").theme, "Godot Dark");
    }

    #[test]
    fn a_missing_file_gives_defaults() {
        assert_eq!(Settings::parse(""), Settings::default());
    }
}
