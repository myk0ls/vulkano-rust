//! A language server for the engine's Rhai scripts.
//!
//! It knows the exact scripting API because it builds the same scripting engine the game
//! runs, so completion and checks stay in step with the engine. Editors start it and talk
//! to it over stdin/stdout. See `zed-extension/README.md` for using it from Zed.
//!
//! Features: completion (API functions, `e.` methods, `this.` values, callbacks after `fn`,
//! component/field/key/entity/scene names inside strings), hover docs, signature help, and
//! diagnostics (syntax errors, calls to functions that don't exist).

mod analysis;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use analysis::{Api, Kind, Project};
use lsp_server::{Connection, Message, Notification, Request, Response};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionResponse, Diagnostic, DiagnosticSeverity, Documentation,
    Hover, HoverContents, InsertTextFormat, MarkupContent, MarkupKind, ParameterInformation, ParameterLabel, Position,
    PublishDiagnosticsParams, Range, SignatureHelp, SignatureInformation, Uri,
};
use serde_json::{Value, json};
use vulkano_engine::scripting::{ScriptRuntime, docs::API_HELP};

fn main() {
    if std::env::args().any(|a| a == "--version") {
        println!("vulkano-rhai-lsp {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    // Startup checks of the API print panics from components that need a running game;
    // those are expected, so keep stderr quiet while building it.
    let (runtime, api) = build_api();
    if std::env::args().any(|a| a == "--list") {
        for (name, function) in &api.functions {
            println!("{name}: {} overloads, {} docs", function.overloads.len(), function.docs.len());
        }
        println!("{} components, {} std functions", api.components.len(), api.std_functions.len());
        return;
    }

    // `--check a.rhai b.rhai`: print each file's problems, like the editor would show them.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--check") {
        let mut count = 0;
        for path in &args[2..] {
            let Ok(text) = std::fs::read_to_string(path) else {
                eprintln!("{path}: can't read");
                continue;
            };
            for problem in analysis::diagnostics(&api, runtime.engine(), &text) {
                let (line, column) = analysis::position_at(&text, problem.start);
                let level = if problem.error { "error" } else { "warning" };
                println!("{path}:{}:{}: {level}: {}", line + 1, column + 1, problem.message);
                count += 1;
            }
        }
        std::process::exit(if count == 0 { 0 } else { 1 });
    }

    let (connection, io_threads) = Connection::stdio();
    let capabilities = json!({
        "textDocumentSync": { "openClose": true, "change": 1 },
        "completionProvider": { "triggerCharacters": [".", "\"", "("] },
        "hoverProvider": true,
        "signatureHelpProvider": { "triggerCharacters": ["(", ","], "retriggerCharacters": [","] },
    });
    let init = match connection.initialize(capabilities) {
        Ok(init) => init,
        Err(e) => {
            eprintln!("vulkano-rhai-lsp: {e}");
            return;
        }
    };
    let mut server = Server {
        api,
        runtime,
        root: project_root(&init),
        project: Project::default(),
        scanned: None,
        documents: HashMap::new(),
    };
    server.run(&connection);
    drop(connection);
    let _ = io_threads.join();
}

fn build_api() -> (ScriptRuntime, Api) {
    let registry = vulkano_game::component_registry();
    let runtime = ScriptRuntime::new(vulkano_game::component_registry());
    let engine_signatures = runtime.engine().gen_fn_signatures(false);
    let all_signatures = runtime.engine().gen_fn_signatures(true);

    // Each component's fields, read from a fresh one on a scratch entity. Some need parts
    // of a running game and panic; they're listed without fields.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut components = BTreeMap::new();
    for def in registry.defs() {
        let fields = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut world = shipyard::World::new();
            let id = world.add_entity(());
            def.add(&mut world, id).ok()?;
            def.fields(&world, id).map(|fields| fields.into_iter().map(|f| f.name.to_string()).collect::<Vec<_>>())
        }))
        .ok()
        .flatten()
        .unwrap_or_default();
        components.insert(def.name.to_string(), fields);
    }
    std::panic::set_hook(previous_hook);

    let api = Api::new(&engine_signatures, &all_signatures, API_HELP, components);
    (runtime, api)
}

/// The project folder: the workspace root, from the initialize request.
fn project_root(init: &Value) -> Option<PathBuf> {
    let uri = init
        .get("workspaceFolders")
        .and_then(|f| f.get(0))
        .and_then(|f| f.get("uri"))
        .or_else(|| init.get("rootUri"))
        .and_then(Value::as_str)?;
    uri_to_path(uri)
}

/// `file:///home/me/game` or `file:///C:/Users/me/game` to a path, undoing %-escapes.
fn uri_to_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    let bytes = rest.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&rest[i + 1..i + 3], 16) {
                decoded.push(byte);
                i += 3;
                continue;
            }
        }
        decoded.push(bytes[i]);
        i += 1;
    }
    let mut path = String::from_utf8(decoded).ok()?;
    // Windows: "/C:/Users" -> "C:/Users"
    if path.len() > 2 && path.as_bytes()[0] == b'/' && path.as_bytes()[2] == b':' {
        path.remove(0);
    }
    Some(PathBuf::from(path))
}

struct Server {
    api: Api,
    runtime: ScriptRuntime,
    root: Option<PathBuf>,
    project: Project,
    scanned: Option<Instant>,
    documents: HashMap<String, String>,
}

impl Server {
    fn run(&mut self, connection: &Connection) {
        for message in &connection.receiver {
            match message {
                Message::Request(request) => {
                    if connection.handle_shutdown(&request).unwrap_or(true) {
                        return;
                    }
                    let response = self.request(&request);
                    let _ = connection.sender.send(Message::Response(response));
                }
                Message::Notification(notification) => {
                    if let Some(published) = self.notification(&notification) {
                        let _ = connection.sender.send(Message::Notification(published));
                    }
                }
                Message::Response(_) => {}
            }
        }
    }

    fn request(&mut self, request: &Request) -> Response {
        let result = match request.method.as_str() {
            "textDocument/completion" => self.completion(&request.params),
            "textDocument/hover" => self.hover(&request.params),
            "textDocument/signatureHelp" => self.signature_help(&request.params),
            _ => {
                return Response::new_err(request.id.clone(), -32601, format!("{} isn't supported", request.method));
            }
        };
        Response::new_ok(request.id.clone(), result.unwrap_or(Value::Null))
    }

    /// The document and byte offset a request is about.
    fn target(&self, params: &Value) -> Option<(&str, usize)> {
        let uri = params.pointer("/textDocument/uri")?.as_str()?;
        let text = self.documents.get(uri)?;
        let line = params.pointer("/position/line")?.as_u64()? as u32;
        let character = params.pointer("/position/character")?.as_u64()? as u32;
        Some((text, analysis::offset_at(text, line, character)))
    }

    fn completion(&mut self, params: &Value) -> Option<Value> {
        // Entity and scene names come from the project's files; look again now and then.
        if self.scanned.is_none_or(|t| t.elapsed() > Duration::from_secs(10)) {
            if let Some(root) = &self.root {
                self.project = Project::scan(root);
            }
            self.scanned = Some(Instant::now());
        }
        let (text, offset) = self.target(params)?;
        let items: Vec<CompletionItem> = analysis::complete(&self.api, &self.project, text, offset)
            .into_iter()
            .map(|c| CompletionItem {
                kind: Some(match c.kind {
                    Kind::Function => CompletionItemKind::FUNCTION,
                    Kind::Method => CompletionItemKind::METHOD,
                    Kind::Property => CompletionItemKind::FIELD,
                    Kind::Variable => CompletionItemKind::VARIABLE,
                    Kind::Keyword => CompletionItemKind::KEYWORD,
                    Kind::Snippet => CompletionItemKind::SNIPPET,
                    Kind::Value => CompletionItemKind::VALUE,
                }),
                detail: c.detail,
                documentation: c.documentation.map(|value| {
                    Documentation::MarkupContent(MarkupContent { kind: MarkupKind::Markdown, value })
                }),
                insert_text_format: c.insert.as_ref().map(|_| InsertTextFormat::SNIPPET),
                insert_text: c.insert,
                sort_text: Some(format!("{}{}", c.rank, c.label)),
                label: c.label,
                ..Default::default()
            })
            .collect();
        serde_json::to_value(CompletionResponse::Array(items)).ok()
    }

    fn hover(&self, params: &Value) -> Option<Value> {
        let (text, offset) = self.target(params)?;
        let value = analysis::hover(&self.api, text, offset)?;
        serde_json::to_value(Hover {
            contents: HoverContents::Markup(MarkupContent { kind: MarkupKind::Markdown, value }),
            range: None,
        })
        .ok()
    }

    fn signature_help(&self, params: &Value) -> Option<Value> {
        let (text, offset) = self.target(params)?;
        let help = analysis::signature_help(&self.api, text, offset)?;
        let signatures = help
            .signatures
            .into_iter()
            .map(|(label, ranges, docs)| {
                // LSP wants parameter offsets in UTF-16 units; labels are ASCII.
                let parameters = ranges
                    .into_iter()
                    .map(|(start, end)| ParameterInformation {
                        label: ParameterLabel::LabelOffsets([start as u32, end as u32]),
                        documentation: None,
                    })
                    .collect();
                SignatureInformation {
                    label,
                    documentation: docs.map(|value| {
                        Documentation::MarkupContent(MarkupContent { kind: MarkupKind::Markdown, value })
                    }),
                    parameters: Some(parameters),
                    active_parameter: None,
                }
            })
            .collect();
        serde_json::to_value(SignatureHelp {
            signatures,
            active_signature: Some(help.active_signature as u32),
            active_parameter: Some(help.active_parameter as u32),
        })
        .ok()
    }

    fn notification(&mut self, notification: &Notification) -> Option<Notification> {
        let params = &notification.params;
        let uri = params.pointer("/textDocument/uri")?.as_str()?.to_string();
        match notification.method.as_str() {
            "textDocument/didOpen" => {
                let text = params.pointer("/textDocument/text")?.as_str()?.to_string();
                self.documents.insert(uri.clone(), text);
            }
            "textDocument/didChange" => {
                // Full sync: the last change holds the whole text.
                let changes = params.get("contentChanges")?.as_array()?;
                let text = changes.last()?.get("text")?.as_str()?.to_string();
                self.documents.insert(uri.clone(), text);
            }
            "textDocument/didClose" => {
                self.documents.remove(&uri);
                return Some(publish(&uri, Vec::new()));
            }
            _ => return None,
        }
        let text = &self.documents[&uri];
        let diagnostics = analysis::diagnostics(&self.api, self.runtime.engine(), text)
            .into_iter()
            .map(|problem| {
                let (sl, sc) = analysis::position_at(text, problem.start);
                let (el, ec) = analysis::position_at(text, problem.end);
                Diagnostic {
                    range: Range { start: Position::new(sl, sc), end: Position::new(el, ec) },
                    severity: Some(if problem.error { DiagnosticSeverity::ERROR } else { DiagnosticSeverity::WARNING }),
                    source: Some("vulkano".into()),
                    message: problem.message,
                    ..Default::default()
                }
            })
            .collect();
        Some(publish(&uri, diagnostics))
    }
}

fn publish(uri: &str, diagnostics: Vec<Diagnostic>) -> Notification {
    let uri: Uri = uri.parse().expect("the editor sent this URI");
    Notification::new(
        "textDocument/publishDiagnostics".into(),
        PublishDiagnosticsParams { uri, diagnostics, version: None },
    )
}
