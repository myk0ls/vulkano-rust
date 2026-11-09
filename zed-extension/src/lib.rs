//! Starts `vulkano-rhai-lsp` for Rhai files. The server is found, in order:
//! 1. at `lsp.vulkano-rhai-lsp.binary.path` in Zed's settings,
//! 2. as `vulkano-rhai-lsp` on PATH,
//! 3. in the project's own build, `target/release/vulkano-rhai-lsp` (`.exe` on Windows).

use zed_extension_api::{self as zed, settings::LspSettings, LanguageServerId, Result};

struct VulkanoRhai;

impl zed::Extension for VulkanoRhai {
    fn new() -> Self {
        VulkanoRhai
    }

    fn language_server_command(&mut self, id: &LanguageServerId, worktree: &zed::Worktree) -> Result<zed::Command> {
        let binary = LspSettings::for_worktree(id.as_ref(), worktree).ok().and_then(|s| s.binary);
        let args = binary.as_ref().and_then(|b| b.arguments.clone()).unwrap_or_default();
        let env = worktree.shell_env();

        if let Some(path) = binary.and_then(|b| b.path) {
            return Ok(zed::Command { command: path, args, env });
        }
        if let Some(path) = worktree.which("vulkano-rhai-lsp") {
            return Ok(zed::Command { command: path, args, env });
        }
        let (os, _) = zed::current_platform();
        let exe = if os == zed::Os::Windows { "vulkano-rhai-lsp.exe" } else { "vulkano-rhai-lsp" };
        let root = worktree.root_path();
        let separator = if os == zed::Os::Windows { "\\" } else { "/" };
        let local = [root.trim_end_matches(['/', '\\']), "target", "release", exe].join(separator);
        Ok(zed::Command { command: local, args, env })
    }
}

zed::register_extension!(VulkanoRhai);
