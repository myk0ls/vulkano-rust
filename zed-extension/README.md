# Vulkano Rhai for Zed

Completion, hover docs, signature help, and error checking for the engine's `.rhai`
scripts, from `vulkano-rhai-lsp` (the `rhai-lsp` folder of this repo).

## Setup (once)

1. Install the **rhai** extension from Zed's extension list (it gives Rhai files syntax
   highlighting; this extension adds the language server to that language).
2. Build the server from the repo root:

   ```
   cargo build --release -p vulkano-rhai-lsp
   ```

3. In Zed, run **zed: install dev extension** from the command palette and pick this
   `zed-extension` folder. Zed needs Rust installed through rustup to build it.

That's it. Open any `.rhai` file in the project and the server starts.

## Finding the server

The extension looks for the server in this order:

1. `binary.path` in your Zed settings, for example:

   ```json
   "lsp": {
     "vulkano-rhai-lsp": {
       "binary": { "path": "C:\\dev\\vulkano-rust\\target\\release\\vulkano-rhai-lsp.exe" }
     }
   }
   ```

2. `vulkano-rhai-lsp` on your PATH (for example after
   `cargo install --path rhai-lsp`).
3. The project's own build: `target/release/vulkano-rhai-lsp` (`.exe` on Windows), which
   works with no setup when the project you open in Zed is this repo.

## After changing the scripting API

The server lists functions by building the engine's real scripting engine, and takes its
descriptions from `engine/src/scripting/docs.rs` (the same help the editor shows). After
adding or changing script functions, rebuild it with the command above, then run
**editor: restart language server** in Zed.

## Checking scripts from a terminal

```
target/release/vulkano-rhai-lsp --check scripts/*.rhai
```

prints the same errors and warnings Zed shows, and exits with 1 if there are any.
