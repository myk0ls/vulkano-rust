//! What the language server knows about the scripting API and works out from a script's
//! text. Kept apart from the protocol code so it can be tested on plain strings.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

// ── The API ─────────────────────────────────────────────────────────────────

/// Everything scripts can call, from the real scripting engine.
pub struct Api {
    /// Engine functions by name.
    pub functions: BTreeMap<String, Function>,
    /// Property names (like a Vec3's `x`), readable as `value.name`.
    pub properties: BTreeSet<String>,
    /// Functions from Rhai's standard library (`len`, `push`, `to_string`, ...).
    pub std_functions: BTreeSet<String>,
    /// Functions the engine calls on scripts, like `on_update`.
    pub callbacks: Vec<Callback>,
    /// Component names and their field names, for `get`, `set`, `add`, `has`, `remove`.
    pub components: BTreeMap<String, Vec<String>>,
}

#[derive(Default, Debug)]
pub struct Function {
    pub overloads: Vec<Overload>,
    pub docs: Vec<Doc>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Overload {
    pub params: Vec<String>,
    pub ret: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Doc {
    pub example: String,
    pub description: String,
}

#[derive(Clone, Debug)]
pub struct Callback {
    pub name: String,
    pub signature: String,
    pub description: String,
}

/// Types a function may be called on with method syntax, `value.name(...)`.
const RECEIVERS: &[&str] = &["Entity", "Vec3"];

/// Words that look like calls but are syntax or built into Rhai itself.
const BUILTIN_CALLS: &[&str] = &[
    "if", "while", "for", "switch", "return", "throw", "loop", "do", "until", "catch", "try", "in", "fn", "Fn", "call",
    "curry", "print", "debug", "type_of", "eval", "is_def_var", "is_def_fn", "is_shared", "this", "global",
];

pub const KEYWORDS: &[&str] = &[
    "let", "const", "fn", "if", "else", "while", "loop", "for", "in", "do", "until", "break", "continue", "return",
    "throw", "try", "catch", "switch", "true", "false", "this", "export", "import", "as", "private", "global",
];

/// Key names `key_down` and friends understand (SDL's names; case doesn't matter).
pub const KEY_NAMES: &[&str] = &[
    "W", "A", "S", "D", "Q", "E", "R", "F", "G", "C", "V", "X", "Z", "Space", "Return", "Escape", "Tab", "Backspace",
    "Left Shift", "Right Shift", "Left Ctrl", "Right Ctrl", "Left Alt", "Right Alt", "Up", "Down", "Left", "Right",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
    "F11", "F12", "B", "H", "I", "J", "K", "L", "M", "N", "O", "P", "T", "U", "Y",
];

impl Overload {
    /// Parse one of Rhai's signatures, like `translate(_: &mut Entity, _: f64) -> i64`.
    pub fn parse(signature: &str) -> Option<(String, Overload)> {
        let open = signature.find('(')?;
        let name = signature[..open].trim().to_string();
        let close = matching_paren(signature, open)?;
        let inner = &signature[open + 1..close];
        let params = split_top_level(inner)
            .into_iter()
            .filter(|p| !p.trim().is_empty())
            .map(|p| {
                let p = p.trim();
                match p.split_once(':') {
                    Some((name, ty)) if name.trim() != "_" => format!("{}: {}", name.trim(), clean_type(ty)),
                    Some((_, ty)) => clean_type(ty),
                    None => clean_type(p),
                }
            })
            .collect();
        let ret = signature[close + 1..].trim().strip_prefix("->").map(clean_type);
        Some((name, Overload { params, ret }))
    }

    /// Whether this can be called as `receiver.name(...)`.
    pub fn is_method(&self) -> bool {
        self.params.first().is_some_and(|p| RECEIVERS.contains(&param_type(p)))
    }

    /// The parameters as written in a call: without the receiver for a method call.
    pub fn call_params(&self, as_method: bool) -> &[String] {
        if as_method && self.is_method() { &self.params[1..] } else { &self.params }
    }

    pub fn label(&self, name: &str, as_method: bool) -> String {
        let mut label = format!("{name}({})", self.call_params(as_method).join(", "));
        if let Some(ret) = &self.ret {
            label.push_str(&format!(" -> {ret}"));
        }
        label
    }
}

fn param_type(param: &str) -> &str {
    param.rsplit(':').next().unwrap_or(param).trim()
}

/// Rust type names to the names scripts know: `types::dynamic::Dynamic` is "any",
/// `i64` is "int", and paths are dropped.
fn clean_type(ty: &str) -> String {
    let ty = ty.trim().trim_start_matches("&mut ").trim_start_matches('&').trim();
    let last = ty.rsplit("::").next().unwrap_or(ty);
    match last {
        "Dynamic" => "any",
        "i64" | "INT" => "int",
        "f64" | "f32" | "FLOAT" => "float",
        "ImmutableString" | "str" | "String" => "string",
        "bool" => "bool",
        other => other,
    }
    .to_string()
}

fn matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in text[open..].char_indices() {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, c) in text.char_indices() {
        match c {
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

impl Api {
    /// Build from Rhai's signature lists and the engine's help text.
    pub fn new(
        engine_signatures: &[String],
        all_signatures: &[String],
        help: &[(&str, &[(&str, &str)])],
        components: BTreeMap<String, Vec<String>>,
    ) -> Self {
        let mut functions: BTreeMap<String, Function> = BTreeMap::new();
        let mut properties = BTreeSet::new();
        for signature in engine_signatures {
            let Some((name, overload)) = Overload::parse(signature) else { continue };
            if let Some(property) = name.strip_prefix("get$").or_else(|| name.strip_prefix("set$")) {
                properties.insert(property.to_string());
                continue;
            }
            if !is_identifier(&name) {
                continue; // operators
            }
            let function = functions.entry(name).or_default();
            if !function.overloads.contains(&overload) {
                function.overloads.push(overload);
            }
        }
        for function in functions.values_mut() {
            function.overloads.sort_by_key(|o| o.params.len());
        }

        let std_functions = all_signatures
            .iter()
            .filter_map(|s| Overload::parse(s))
            .map(|(name, _)| name)
            .filter(|name| is_identifier(name) && !functions.contains_key(name))
            .collect();

        let mut callbacks = Vec::new();
        for (section, entries) in help {
            for (example, description) in entries.iter() {
                if section.starts_with("Called by the engine") {
                    // "fn on_update(dt)" becomes a callback; "on_<signal>" is a pattern.
                    if let Some(rest) = example.strip_prefix("fn ")
                        && let Some(open) = rest.find('(')
                        && is_identifier(&rest[..open])
                    {
                        callbacks.push(Callback {
                            name: rest[..open].to_string(),
                            signature: example.to_string(),
                            description: description.to_string(),
                        });
                    }
                    continue;
                }
                for name in called_names(example) {
                    if let Some(function) = functions.get_mut(&name) {
                        function.docs.push(Doc { example: example.to_string(), description: description.to_string() });
                    }
                }
            }
        }
        // Functions no example shows may still be named in a description, as calls
        // (`pick(array)`, `.length()`) or listed after "Also". Use those for them only, so a
        // passing mention doesn't add to a function's own entry.
        for (_, entries) in help {
            for (example, description) in entries.iter() {
                for name in mentioned_names(description) {
                    if let Some(function) = functions.get_mut(&name)
                        && function.docs.iter().all(|d| d.example.as_str() != *example)
                        && function.docs.iter().all(|d| !called_names(&d.example).contains(&name))
                    {
                        function.docs.push(Doc { example: example.to_string(), description: description.to_string() });
                    }
                }
            }
        }

        Api { functions, properties, std_functions, callbacks, components }
    }

    fn knows_function(&self, name: &str) -> bool {
        self.functions.contains_key(name) || self.std_functions.contains(name) || BUILTIN_CALLS.contains(&name)
    }

    fn markdown(&self, name: &str, as_method: bool) -> Option<String> {
        let function = self.functions.get(name)?;
        let mut text = String::from("```rhai\n");
        for overload in &function.overloads {
            if as_method && !overload.is_method() {
                continue;
            }
            text.push_str(&overload.label(name, as_method));
            text.push('\n');
        }
        text.push_str("```\n");
        for doc in &function.docs {
            text.push_str(&format!("\n`{}`  \n{}\n", doc.example, doc.description));
        }
        Some(text)
    }
}

/// The names called at the top level of a help example: `time(), frame()` gives both,
/// `set_text(label, fps())` gives only `set_text`.
fn called_names(example: &str) -> Vec<String> {
    let bytes = example.as_bytes();
    let mut names = Vec::new();
    let mut depth = 0;
    let mut i = 0;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            if c == b'\\' {
                i += 1;
            } else if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' | b'`' | b'\'' => quote = Some(c),
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ if is_ident_byte(c) && (i == 0 || !is_ident_byte(bytes[i - 1])) => {
                let start = i;
                while i < bytes.len() && is_ident_byte(bytes[i]) {
                    i += 1;
                }
                if depth == 0 && bytes.get(i) == Some(&b'(') {
                    names.push(example[start..i].to_string());
                }
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    names
}

/// Function names a help description mentions: written as calls, or listed after "Also".
fn mentioned_names(description: &str) -> Vec<String> {
    let mut names = Vec::new();
    let bytes = description.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if is_ident_byte(bytes[i]) && (i == 0 || !is_ident_byte(bytes[i - 1])) {
            let start = i;
            while i < bytes.len() && is_ident_byte(bytes[i]) {
                i += 1;
            }
            if bytes.get(i) == Some(&b'(') {
                names.push(description[start..i].to_string());
            }
            continue;
        }
        i += 1;
    }
    for sentence in description.split('.').filter_map(|s| s.trim().strip_prefix("Also ")) {
        for word in sentence.split([',', ';', ' ']) {
            let word = word.trim();
            if is_identifier(word) && word != "and" && word != "takes" {
                names.push(word.to_string());
            }
        }
    }
    names
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn is_identifier(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(is_ident_byte)
        && !name.as_bytes()[0].is_ascii_digit()
}

// ── The project ─────────────────────────────────────────────────────────────

/// Things in the game's folder that scripts refer to by name.
#[derive(Default, Debug)]
pub struct Project {
    /// Entity names in scenes and prefabs, for `find("...")`.
    pub entity_names: BTreeSet<String>,
    /// Scene and prefab files, relative to the project, for `spawn` and `change_scene`.
    pub scene_files: BTreeSet<String>,
}

impl Project {
    pub fn scan(root: &Path) -> Self {
        let mut project = Project::default();
        // Breadth first, so scene folders near the top are found before deep asset folders.
        let mut queue = std::collections::VecDeque::from([root.to_path_buf()]);
        let mut seen = 0;
        while let Some(dir) = queue.pop_front() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                seen += 1;
                if seen > 20_000 {
                    return project; // a huge folder; give up rather than stall
                }
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if path.is_dir() {
                    if !name.starts_with('.') && !matches!(name.as_str(), "target" | "node_modules" | "data") {
                        queue.push_back(path);
                    }
                } else if name.ends_with(".vscene") {
                    if let Ok(relative) = path.strip_prefix(root) {
                        project.scene_files.insert(relative.to_string_lossy().replace('\\', "/"));
                    }
                    if let Ok(text) = std::fs::read_to_string(&path) {
                        project.entity_names.extend(entity_names(&text));
                    }
                }
            }
        }
        project
    }
}

/// Entity names in a scene file: the `Text("...")` after each `component: "Name"`, on the
/// same line or the following ones (scenes saved by the editor are spread over lines).
fn entity_names(scene: &str) -> Vec<String> {
    let mut names = Vec::new();
    for (start, _) in scene.match_indices("\"Name\"") {
        let rest = &scene[start..];
        let window = &rest[..rest.len().min(300)];
        if let Some(text) = window.find("Text(\"") {
            let value = &window[text + 6..];
            if let Some(end) = value.find("\")") {
                names.push(value[..end].to_string());
            }
        }
    }
    names
}

#[cfg(test)]
#[test]
fn entity_names_are_found_in_both_scene_layouts() {
    let one_line = r#"(component: "Name", fields: {"name": Text("Coin")}),"#;
    let spread = "(\n    component: \"Name\",\n    fields: {\n        \"name\": Text(\"Player\"),\n    },\n),";
    assert_eq!(entity_names(one_line), ["Coin"]);
    assert_eq!(entity_names(spread), ["Player"]);
}

// ── Reading a script ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Region {
    Code,
    String,
    Comment,
}

/// For each byte of the script, whether it's code, inside a string, or in a comment.
/// Handles `${...}` interpolation in backtick strings and nested block comments.
pub fn regions(text: &str) -> Vec<Region> {
    let bytes = text.as_bytes();
    let mut out = vec![Region::Code; bytes.len()];
    // What we're inside: a string with this quote, or code inside `${` (counting braces).
    enum Frame {
        Str(u8),
        Interp(u32),
    }
    let mut stack: Vec<Frame> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match stack.last_mut() {
            Some(Frame::Str(q)) => {
                let q = *q;
                out[i] = Region::String;
                if c == b'\\' && i + 1 < bytes.len() {
                    out[i + 1] = Region::String;
                    i += 2;
                    continue;
                }
                if c == q {
                    stack.pop();
                } else if q == b'`' && c == b'$' && bytes.get(i + 1) == Some(&b'{') {
                    out[i + 1] = Region::String;
                    stack.push(Frame::Interp(0));
                    i += 2;
                    continue;
                }
                i += 1;
            }
            frame => {
                // Code, at the top level or inside an interpolation.
                if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
                    while i < bytes.len() && bytes[i] != b'\n' {
                        out[i] = Region::Comment;
                        i += 1;
                    }
                    continue;
                }
                if c == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    let mut depth = 0;
                    while i < bytes.len() {
                        if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                            depth += 1;
                            out[i] = Region::Comment;
                            out[i + 1] = Region::Comment;
                            i += 2;
                        } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                            depth -= 1;
                            out[i] = Region::Comment;
                            out[i + 1] = Region::Comment;
                            i += 2;
                            if depth == 0 {
                                break;
                            }
                        } else {
                            out[i] = Region::Comment;
                            i += 1;
                        }
                    }
                    continue;
                }
                if let Some(Frame::Interp(depth)) = frame {
                    if c == b'{' {
                        *depth += 1;
                    } else if c == b'}' {
                        if *depth == 0 {
                            out[i] = Region::String;
                            stack.pop();
                            i += 1;
                            continue;
                        }
                        *depth -= 1;
                    }
                }
                if c == b'"' || c == b'`' || c == b'\'' {
                    out[i] = Region::String;
                    stack.push(Frame::Str(c));
                }
                i += 1;
            }
        }
    }
    out
}

/// Byte offset of an LSP position (line, UTF-16 column).
pub fn offset_at(text: &str, line: u32, character: u32) -> usize {
    let mut offset = 0;
    for (index, l) in text.split_inclusive('\n').enumerate() {
        if index as u32 == line {
            let mut units = 0;
            for (i, c) in l.char_indices() {
                if units >= character || c == '\n' {
                    return offset + i;
                }
                units += c.len_utf16() as u32;
            }
            return offset + l.trim_end_matches('\n').len();
        }
        offset += l.len();
    }
    text.len()
}

/// LSP position (line, UTF-16 column) of a byte offset.
pub fn position_at(text: &str, offset: usize) -> (u32, u32) {
    let before = &text[..offset.min(text.len())];
    let line = before.matches('\n').count() as u32;
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    let column = before[line_start..].chars().map(|c| c.len_utf16() as u32).sum();
    (line, column)
}

fn word_start(text: &str, offset: usize) -> usize {
    let bytes = text.as_bytes();
    let mut start = offset;
    while start > 0 && is_ident_byte(bytes[start - 1]) {
        start -= 1;
    }
    start
}

fn word_end(text: &str, offset: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = offset;
    while end < bytes.len() && is_ident_byte(bytes[end]) {
        end += 1;
    }
    end
}

/// The identifier just before `offset`, skipping spaces.
fn identifier_before(text: &str, offset: usize) -> Option<(usize, &str)> {
    let end = text[..offset].trim_end().len();
    let start = word_start(text, end);
    (start < end).then(|| (start, &text[start..end]))
}

/// What the script itself defines.
#[derive(Default, Debug)]
pub struct ScriptSymbols {
    /// Script functions: name, parameters, and the comment lines above them.
    pub functions: Vec<(String, Vec<String>, String)>,
    /// Names from `let`, `const`, and function parameters.
    pub variables: BTreeSet<String>,
    /// Names used as `this.name`, including `export let` values.
    pub this_fields: BTreeSet<String>,
}

pub fn script_symbols(text: &str, regions: &[Region]) -> ScriptSymbols {
    let mut symbols = ScriptSymbols::default();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if regions[i] != Region::Code || !is_ident_byte(bytes[i]) || (i > 0 && is_ident_byte(bytes[i - 1])) {
            i += 1;
            continue;
        }
        let end = word_end(text, i);
        let word = &text[i..end];
        let after = text[end..].trim_start();
        let next_word = |s: &str| {
            let e = s.bytes().position(|c| !is_ident_byte(c)).unwrap_or(s.len());
            s[..e].to_string()
        };
        match word {
            "fn" => {
                let name = next_word(after);
                if is_identifier(&name) {
                    let rest = after[name.len()..].trim_start();
                    let params: Vec<String> = rest
                        .strip_prefix('(')
                        .and_then(|r| r.split_once(')'))
                        .map(|(inside, _)| {
                            inside.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
                        })
                        .unwrap_or_default();
                    symbols.variables.extend(params.iter().cloned());
                    symbols.functions.push((name, params, comment_above(text, i)));
                }
            }
            "let" | "const" => {
                let name = next_word(after);
                if is_identifier(&name) {
                    symbols.variables.insert(name);
                }
            }
            "export" => {
                let rest = after.strip_prefix("let").or_else(|| after.strip_prefix("const"));
                if let Some(rest) = rest {
                    let name = next_word(rest.trim_start());
                    if is_identifier(&name) {
                        symbols.this_fields.insert(name);
                    }
                }
            }
            "this" if text[end..].starts_with('.') => {
                let name = next_word(&text[end + 1..]);
                if is_identifier(&name) {
                    symbols.this_fields.insert(name);
                }
            }
            _ => {}
        }
        i = end;
    }
    // Closure parameters: |a, b|
    for (start, _) in text.match_indices('|') {
        if regions.get(start) != Some(&Region::Code) {
            continue;
        }
        if let Some(close) = text[start + 1..].find('|') {
            let inside = &text[start + 1..start + 1 + close];
            if inside.split(',').all(|p| is_identifier(p.trim()) || p.trim().is_empty()) {
                symbols.variables.extend(inside.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()));
            }
        }
    }
    symbols
}

fn comment_above(text: &str, offset: usize) -> String {
    let line_start = text[..offset].rfind('\n').map_or(0, |i| i + 1);
    let mut lines = Vec::new();
    for line in text[..line_start].lines().rev() {
        match line.trim().strip_prefix("//") {
            Some(comment) => lines.push(comment.trim_start_matches('/').trim().to_string()),
            None => break,
        }
    }
    lines.reverse();
    lines.join("\n")
}

/// The call the cursor is inside: its name, whether it's a method call, which argument
/// the cursor is on, and where its `(` is.
pub struct CallContext {
    pub name: String,
    pub method: bool,
    pub argument: usize,
    pub open: usize,
}

pub fn enclosing_call(text: &str, regions: &[Region], offset: usize) -> Option<CallContext> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut argument = 0;
    let mut i = offset.min(bytes.len());
    while i > 0 {
        i -= 1;
        if regions[i] != Region::Code {
            continue;
        }
        match bytes[i] {
            b')' | b']' | b'}' => depth += 1,
            b'[' | b'{' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            b'(' => {
                if depth == 0 {
                    let (start, name) = identifier_before(text, i)?;
                    if !is_identifier(name) || name == "fn" {
                        return None;
                    }
                    let method = text[..start].trim_end().ends_with('.');
                    return Some(CallContext { name: name.to_string(), method, argument, open: i });
                }
                depth -= 1;
            }
            b',' if depth == 0 => argument += 1,
            b';' if depth == 0 => return None,
            _ => {}
        }
    }
    None
}

// ── Answers ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Function,
    Method,
    Property,
    Variable,
    Keyword,
    Snippet,
    Value,
}

#[derive(Clone, Debug)]
pub struct Completion {
    pub label: String,
    pub kind: Kind,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    /// Snippet text, with `$1`/`$0` tab stops.
    pub insert: Option<String>,
    /// Lower sorts first.
    pub rank: u8,
}

impl Completion {
    fn plain(label: impl Into<String>, kind: Kind, rank: u8) -> Self {
        Completion { label: label.into(), kind, detail: None, documentation: None, insert: None, rank }
    }
}

pub fn complete(api: &Api, project: &Project, text: &str, offset: usize) -> Vec<Completion> {
    let regions = regions(text);
    let offset = offset.min(text.len());
    let region = if offset == 0 { Region::Code } else { regions[offset - 1] };
    match region {
        Region::Comment => Vec::new(),
        Region::String => complete_in_string(api, project, text, &regions, offset),
        Region::Code => complete_code(api, text, &regions, offset),
    }
}

fn complete_in_string(api: &Api, project: &Project, text: &str, regions: &[Region], offset: usize) -> Vec<Completion> {
    // Find the opening quote of this string.
    let mut quote = offset;
    while quote > 0 && regions[quote - 1] == Region::String {
        quote -= 1;
    }
    if quote == offset {
        return Vec::new();
    }
    let Some(call) = enclosing_call(text, regions, quote) else { return Vec::new() };
    // As a function, the entity is the first argument: get(e, "Transform", ...).
    let entity_first = !call.method && matches!(call.name.as_str(), "get" | "set" | "add" | "has" | "remove");
    let argument = call.argument - usize::from(entity_first && call.argument > 0);
    let values = |items: Vec<String>, kind: Kind| -> Vec<Completion> {
        items.into_iter().map(|item| Completion::plain(item, kind, 0)).collect()
    };
    match (call.name.as_str(), argument) {
        ("get" | "set" | "add" | "has" | "remove", 0) => values(api.components.keys().cloned().collect(), Kind::Value),
        ("get" | "set", 1) => {
            // The component is the first string in this call.
            let args = &text[call.open + 1..quote];
            let component = args.split('"').nth(1).unwrap_or("");
            values(api.components.get(component).cloned().unwrap_or_default(), Kind::Property)
        }
        ("key_down" | "key_pressed" | "key_released", 0) => {
            values(KEY_NAMES.iter().map(|k| k.to_string()).collect(), Kind::Value)
        }
        ("find" | "find_all", 0) => values(project.entity_names.iter().cloned().collect(), Kind::Value),
        ("spawn" | "change_scene", 0) => values(project.scene_files.iter().cloned().collect(), Kind::Value),
        ("after" | "every", 1) => {
            let symbols = script_symbols(text, regions);
            values(symbols.functions.into_iter().map(|f| f.0).collect(), Kind::Function)
        }
        _ => Vec::new(),
    }
}

fn function_completion(api: &Api, name: &str, function: &Function, method: bool) -> Completion {
    let overloads: Vec<&Overload> =
        function.overloads.iter().filter(|o| !method || o.is_method()).collect();
    let takes_arguments = overloads.iter().any(|o| !o.call_params(method).is_empty());
    let detail = overloads.first().map(|o| {
        let mut label = o.label(name, method);
        if overloads.len() > 1 {
            label.push_str(&format!("  (+{} more)", overloads.len() - 1));
        }
        label
    });
    Completion {
        label: name.to_string(),
        kind: if method { Kind::Method } else { Kind::Function },
        detail,
        documentation: api.markdown(name, method),
        insert: Some(if takes_arguments { format!("{name}($1)$0") } else { format!("{name}()$0") }),
        rank: 0,
    }
}

fn complete_code(api: &Api, text: &str, regions: &[Region], offset: usize) -> Vec<Completion> {
    let start = word_start(text, offset);
    let before = text[..start].trim_end_matches([' ', '\t']);
    let symbols = script_symbols(text, regions);
    let mut out = Vec::new();

    // After `fn `: the engine's callbacks, written out.
    if before.ends_with("fn") && !before.ends_with("_fn") {
        for callback in &api.callbacks {
            let params = callback.signature.split_once('(').map_or("", |(_, p)| p.trim_end_matches(')'));
            out.push(Completion {
                label: callback.name.clone(),
                kind: Kind::Snippet,
                detail: Some(callback.signature.clone()),
                documentation: Some(callback.description.clone()),
                insert: Some(format!("{}({params}) {{\n\t$0\n}}", callback.name)),
                rank: 0,
            });
        }
        return out;
    }

    // After `.`: methods and properties, or this script's own values after `this.`.
    if let Some(receiver_end) = before.strip_suffix('.').map(str::len) {
        let receiver = identifier_before(text, receiver_end).map(|(_, w)| w).unwrap_or("");
        if receiver == "this" {
            for field in &symbols.this_fields {
                out.push(Completion::plain(field.clone(), Kind::Property, 0));
            }
            return out;
        }
        for (name, function) in &api.functions {
            if function.overloads.iter().any(Overload::is_method) {
                out.push(function_completion(api, name, function, true));
            }
        }
        for property in &api.properties {
            out.push(Completion::plain(property.clone(), Kind::Property, 0));
        }
        for (name, params, docs) in &symbols.functions {
            let mut item = Completion::plain(name.clone(), Kind::Method, 1);
            item.detail = Some(format!("fn {name}({})", params.join(", ")));
            item.documentation = (!docs.is_empty()).then(|| docs.clone());
            item.insert = Some(format!("{name}($1)$0"));
            out.push(item);
        }
        for name in &api.std_functions {
            let mut item = Completion::plain(name.clone(), Kind::Method, 3);
            item.insert = Some(format!("{name}($1)$0"));
            out.push(item);
        }
        return out;
    }

    for (name, function) in &api.functions {
        out.push(function_completion(api, name, function, false));
    }
    for (name, params, docs) in &symbols.functions {
        let mut item = Completion::plain(name.clone(), Kind::Function, 1);
        item.detail = Some(format!("fn {name}({})", params.join(", ")));
        item.documentation = (!docs.is_empty()).then(|| docs.clone());
        item.insert = Some(if params.is_empty() { format!("{name}()$0") } else { format!("{name}($1)$0") });
        out.push(item);
    }
    for variable in &symbols.variables {
        out.push(Completion::plain(variable.clone(), Kind::Variable, 1));
    }
    for keyword in KEYWORDS {
        out.push(Completion::plain(*keyword, Kind::Keyword, 2));
    }
    for name in &api.std_functions {
        let mut item = Completion::plain(name.clone(), Kind::Function, 3);
        item.insert = Some(format!("{name}($1)$0"));
        out.push(item);
    }
    out
}

/// Markdown for whatever is under the cursor.
pub fn hover(api: &Api, text: &str, offset: usize) -> Option<String> {
    let regions = regions(text);
    if regions.get(offset).is_some_and(|r| *r != Region::Code) {
        return None;
    }
    let start = word_start(text, offset);
    let end = word_end(text, offset);
    if start == end {
        return None;
    }
    let word = &text[start..end];
    let method = text[..start].trim_end().ends_with('.');
    let defining = text[..start].trim_end().ends_with("fn");

    let symbols = script_symbols(text, &regions);
    if defining {
        if let Some(callback) = api.callbacks.iter().find(|c| c.name == word) {
            return Some(format!("```rhai\n{}\n```\nCalled by the engine. {}", callback.signature, callback.description));
        }
    }
    if let Some((name, params, docs)) = symbols.functions.iter().find(|f| f.0 == word) {
        let mut text = format!("```rhai\nfn {name}({})\n```", params.join(", "));
        if !docs.is_empty() {
            text.push_str(&format!("\n{docs}"));
        }
        return Some(text);
    }
    if let Some(markdown) = api.markdown(word, method) {
        return Some(markdown);
    }
    api.callbacks
        .iter()
        .find(|c| c.name == word)
        .map(|c| format!("```rhai\n{}\n```\nCalled by the engine. {}", c.signature, c.description))
}

pub struct Signatures {
    /// Each overload: its label, the byte ranges of its parameters in the label, docs.
    pub signatures: Vec<(String, Vec<(usize, usize)>, Option<String>)>,
    pub active_signature: usize,
    pub active_parameter: usize,
}

pub fn signature_help(api: &Api, text: &str, offset: usize) -> Option<Signatures> {
    let regions = regions(text);
    let call = enclosing_call(text, &regions, offset.min(text.len()))?;
    let mut signatures = Vec::new();
    let mut arities = Vec::new();

    let label_with = |name: &str, params: &[String], ret: Option<&String>| {
        let mut label = format!("{name}(");
        let mut ranges = Vec::new();
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                label.push_str(", ");
            }
            ranges.push((label.len(), label.len() + param.len()));
            label.push_str(param);
        }
        label.push(')');
        if let Some(ret) = ret {
            label.push_str(&format!(" -> {ret}"));
        }
        (label, ranges)
    };

    if let Some(function) = api.functions.get(&call.name) {
        let docs = api.markdown(&call.name, call.method).map(|m| {
            // The overloads are already listed; keep the descriptions.
            m.split_once("```\n").map(|(_, rest)| rest.trim().to_string()).unwrap_or(m)
        });
        // A plain call lists the plain overloads first; method-style ones also work as
        // plain calls, with the receiver as the first argument.
        let mut overloads: Vec<&Overload> =
            function.overloads.iter().filter(|o| !call.method || o.is_method()).collect();
        overloads.sort_by_key(|o| (!call.method && o.is_method(), o.params.len()));
        for overload in overloads {
            let params = overload.call_params(call.method);
            let (label, ranges) = label_with(&call.name, params, overload.ret.as_ref());
            arities.push(params.len());
            signatures.push((label, ranges, docs.clone().filter(|d| !d.is_empty())));
        }
    } else if let Some((name, params, docs)) =
        script_symbols(text, &regions).functions.into_iter().find(|f| f.0 == call.name)
    {
        let (label, ranges) = label_with(&name, &params, None);
        arities.push(params.len());
        signatures.push((label, ranges, (!docs.is_empty()).then_some(docs)));
    }
    if signatures.is_empty() {
        return None;
    }
    let active_signature = arities.iter().position(|&n| n > call.argument).unwrap_or(arities.len() - 1);
    Some(Signatures { signatures, active_signature, active_parameter: call.argument })
}

#[derive(Debug, PartialEq)]
pub struct Problem {
    pub start: usize,
    pub end: usize,
    pub message: String,
    pub error: bool,
}

/// Syntax errors from Rhai itself, and calls to functions that don't exist.
pub fn diagnostics(api: &Api, engine: &rhai::Engine, text: &str) -> Vec<Problem> {
    let mut problems = unknown_calls(api, text);
    if let Err(error) = engine.compile(text) {
        let position = error.position();
        let start = match (position.line(), position.position()) {
            (Some(line), column) => {
                let line_start: usize = text.split_inclusive('\n').take(line - 1).map(str::len).sum();
                let line_text = text[line_start..].lines().next().unwrap_or("");
                let column = column.unwrap_or(1).saturating_sub(1);
                line_start + line_text.char_indices().nth(column).map_or(line_text.len(), |(i, _)| i)
            }
            _ => 0,
        };
        let end = word_end(text, start).max((start + 1).min(text.len()));
        problems.insert(0, Problem { start, end, message: error.0.to_string(), error: true });
    }
    problems
}

/// Calls to functions that are neither in the API nor defined in the script.
fn unknown_calls(api: &Api, text: &str) -> Vec<Problem> {
    let regions = regions(text);
    let symbols = script_symbols(text, &regions);
    let script_fns: BTreeSet<&str> = symbols.functions.iter().map(|f| f.0.as_str()).collect();
    let bytes = text.as_bytes();
    let mut problems = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if regions[i] != Region::Code || !is_ident_byte(bytes[i]) || (i > 0 && is_ident_byte(bytes[i - 1])) {
            i += 1;
            continue;
        }
        let end = word_end(text, i);
        let word = &text[i..end];
        let calls = text[end..].trim_start_matches([' ', '\t']).starts_with('(');
        let before = text[..i].trim_end();
        let defining = before.ends_with("fn") && !before.ends_with("_fn");
        if calls && !defining && !bytes[i].is_ascii_digit() {
            let known = api.knows_function(word)
                || script_fns.contains(word)
                || symbols.variables.contains(word)
                || api.callbacks.iter().any(|c| c.name == word);
            if !known {
                let method = before.ends_with('.');
                problems.push(Problem {
                    start: i,
                    end,
                    message: if method {
                        format!("No method named `{word}` in the scripting API or this script.")
                    } else {
                        format!("No function named `{word}` in the scripting API or this script.")
                    },
                    error: false,
                });
            }
        }
        i = end;
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api() -> Api {
        let engine = vec![
            "translate(_: types::dynamic::Dynamic, _: types::dynamic::Dynamic, _: types::dynamic::Dynamic)".to_string(),
            "translate(_: &mut Entity, _: Vec3)".to_string(),
            "set_text(_: &mut Entity, _: types::dynamic::Dynamic)".to_string(),
            "find(_: string) -> types::dynamic::Dynamic".to_string(),
            "fps() -> i64".to_string(),
            "get$x(_: &mut Vec3) -> f64".to_string(),
            "+(_: Vec3, _: Vec3) -> Vec3".to_string(),
            "get(_: &mut Entity, _: &str, _: &str) -> types::dynamic::Dynamic".to_string(),
            "key_down(_: &str) -> bool".to_string(),
        ];
        let all = vec!["len(_: &mut Array) -> i64".to_string(), "push(_: &mut Array, _: Dynamic)".to_string()];
        let help: &[(&str, &[(&str, &str)])] = &[
            ("Called by the engine", &[("fn on_update(dt)", "Runs every frame."), ("fn on_<signal>(...)", "x")]),
            ("Text", &[("e.set_text(\"Score: 3\")", "On a UiElement."), ("fps(), frame_ms()", "Frames per second.")]),
        ];
        let mut components = BTreeMap::new();
        components.insert("Transform".to_string(), vec!["position".to_string(), "scale".to_string()]);
        Api::new(&engine, &all, help, components)
    }

    fn at(text: &str) -> (String, usize) {
        let offset = text.find('|').unwrap();
        (text.replacen('|', "", 1), offset)
    }

    fn labels(items: &[Completion]) -> Vec<&str> {
        items.iter().map(|c| c.label.as_str()).collect()
    }

    #[test]
    fn signatures_are_parsed_into_script_types() {
        let (name, o) = Overload::parse("translate(_: &mut Entity, _: types::dynamic::Dynamic) -> i64").unwrap();
        assert_eq!(name, "translate");
        assert_eq!(o.params, ["Entity", "any"]);
        assert_eq!(o.ret.as_deref(), Some("int"));
        assert!(o.is_method());
        assert_eq!(o.label("translate", true), "translate(any) -> int");
    }

    #[test]
    fn functions_get_docs_and_operators_and_properties_are_separated() {
        let api = api();
        assert!(api.functions.contains_key("set_text"));
        assert!(!api.functions.contains_key("+"));
        assert!(api.properties.contains("x"));
        assert_eq!(api.functions["set_text"].docs[0].description, "On a UiElement.");
        assert_eq!(api.functions["fps"].docs[0].description, "Frames per second.");
        assert_eq!(api.callbacks.len(), 1);
    }

    #[test]
    fn global_completion_offers_api_script_functions_and_variables() {
        let (text, offset) = at("fn helper(a) {}\nfn on_update(dt) { let speed = 1; tr| }");
        let items = complete(&api(), &Project::default(), &text, offset);
        let names = labels(&items);
        for expected in ["translate", "fps", "helper", "speed", "dt", "let", "len"] {
            assert!(names.contains(&expected), "{expected} missing from {names:?}");
        }
        let fps = items.iter().find(|c| c.label == "fps").unwrap();
        assert_eq!(fps.insert.as_deref(), Some("fps()$0"));
    }

    #[test]
    fn member_completion_offers_methods_without_the_receiver() {
        let (text, offset) = at("fn on_update(dt) { find(\"A\").| }");
        let items = complete(&api(), &Project::default(), &text, offset);
        let names = labels(&items);
        assert!(names.contains(&"set_text") && names.contains(&"x"), "{names:?}");
        assert!(!names.contains(&"fps") && !names.contains(&"find"), "{names:?}");
        let set_text = items.iter().find(|c| c.label == "set_text").unwrap();
        assert_eq!(set_text.detail.as_deref(), Some("set_text(any)"));
    }

    #[test]
    fn this_completion_offers_the_scripts_own_values() {
        let (text, offset) = at("export let speed = 1.0;\nfn on_start() { this.hp = 3; }\nfn on_update(dt) { this.| }");
        let names: Vec<_> = complete(&api(), &Project::default(), &text, offset).into_iter().map(|c| c.label).collect();
        assert_eq!(names, ["hp", "speed"]);
    }

    #[test]
    fn after_fn_the_callbacks_are_offered_as_snippets() {
        let (text, offset) = at("fn on_|");
        let items = complete(&api(), &Project::default(), &text, offset);
        assert_eq!(labels(&items), ["on_update"]);
        assert_eq!(items[0].insert.as_deref(), Some("on_update(dt) {\n\t$0\n}"));
    }

    #[test]
    fn strings_complete_components_fields_keys_and_entities() {
        let api = api();
        let mut project = Project::default();
        project.entity_names.insert("Player".into());
        let cases = [
            ("e.get(\"|\")", "Transform"),
            ("e.get(\"Transform\", \"|\")", "position"),
            ("get(e, \"Transform\", \"|\")", "position"),
            ("key_down(\"|\")", "Space"),
            ("find(\"P|\")", "Player"),
        ];
        for (source, expected) in cases {
            let (text, offset) = at(source);
            let items = complete(&api, &project, &text, offset);
            assert!(labels(&items).contains(&expected), "{source}: {:?}", labels(&items));
        }
        let (text, offset) = at("print(\"hello |\")");
        assert!(complete(&api, &project, &text, offset).is_empty());
        let (text, offset) = at("// tra|");
        assert!(complete(&api, &project, &text, offset).is_empty());
    }

    #[test]
    fn interpolated_code_is_code() {
        let text = "let s = `fps ${fps()} now`;";
        let r = regions(text);
        assert_eq!(r[text.find("fps()").unwrap()], Region::Code);
        assert_eq!(r[text.find("now").unwrap()], Region::String);
    }

    #[test]
    fn signature_help_tracks_the_argument() {
        let (text, offset) = at("translate(1.0, |");
        let help = signature_help(&api(), &text, offset).unwrap();
        assert_eq!(help.active_parameter, 1);
        let (label, ranges, _) = &help.signatures[help.active_signature];
        assert_eq!(label, "translate(any, any, any)");
        assert_eq!(&label[ranges[1].0..ranges[1].1], "any");

        let (text, offset) = at("e.translate(|");
        let help = signature_help(&api(), &text, offset).unwrap();
        assert_eq!(help.signatures.len(), 1);
        assert_eq!(help.signatures[0].0, "translate(Vec3)");
    }

    #[test]
    fn hover_shows_overloads_and_docs() {
        let (text, offset) = at("e.set_te|xt(\"a\")");
        let markdown = hover(&api(), &text, offset).unwrap();
        assert!(markdown.contains("set_text(any)") && markdown.contains("On a UiElement."), "{markdown}");
        let (text, offset) = at("// Heals.\nfn heal(n) {}\nfn on_start() { he|al(2); }");
        assert!(hover(&api(), &text, offset).unwrap().contains("Heals."));
    }

    #[test]
    fn unknown_functions_are_flagged_and_known_ones_are_not() {
        let engine = rhai::Engine::new();
        let text = "fn helper() {}\nfn on_update(dt) {\n  helper(); fps(); [1].len(); let f = |x| x; f(2);\n  set_txt(\"a\");\n  print(`${fps()}`);\n}";
        let problems = diagnostics(&api(), &engine, text);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert_eq!(&text[problems[0].start..problems[0].end], "set_txt");
        assert!(!problems[0].error);
    }

    #[test]
    fn syntax_errors_point_at_their_line() {
        let engine = rhai::Engine::new();
        let text = "fn on_start() {\n  let x = ;\n}";
        let problems = diagnostics(&api(), &engine, text);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].error);
        assert_eq!(position_at(text, problems[0].start).0, 1);
    }

    #[test]
    fn windows_line_endings_work() {
        let engine = rhai::Engine::new();
        let text = "fn on_update(dt) {\r\n    set_txt(1);\r\n    fp\r\n}\r\n";
        let problems = diagnostics(&api(), &engine, text);
        let unknown = problems.iter().find(|p| !p.error).unwrap();
        assert_eq!(&text[unknown.start..unknown.end], "set_txt");
        assert_eq!(position_at(text, unknown.start), (1, 4));
        let offset = offset_at(text, 2, 6);
        assert_eq!(&text[offset - 2..offset], "fp");
        assert!(labels(&complete(&api(), &Project::default(), text, offset)).contains(&"fps"));
    }

    #[test]
    fn positions_round_trip_with_utf16_columns() {
        let text = "let a = \"ž\";\nfoo(1)";
        let offset = offset_at(text, 1, 4);
        assert_eq!(&text[offset..offset + 1], "1");
        assert_eq!(position_at(text, offset), (1, 4));
        let offset = offset_at(text, 0, 10);
        assert_eq!(&text[offset..offset + 1], "\"");
    }
}
