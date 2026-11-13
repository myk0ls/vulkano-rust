//! `.blend` files, loaded by having Blender export them to glTF, as Godot does.
//!
//! Blender's own file format changes between versions and keeps materials as node graphs,
//! so the engine doesn't read it. Instead Blender runs in the background and exports a
//! `.glb`: meshes with modifiers applied, armatures and every action as an animation clip,
//! and Principled BSDF materials (base colour, metallic, roughness, normal map, emission,
//! alpha) with their textures. The result is cached in `.import/` next to the project and
//! only redone when the `.blend` is newer.
//!
//! Blender is found from `set_blender_path`, the `VULKANO_BLENDER` environment variable,
//! `blender` on PATH, or the usual install folders (including Steam libraries).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// Where converted files go, relative to the project root.
pub const IMPORT_DIR: &str = ".import";

const EXPORT_SCRIPT: &str = r#"
import sys, bpy
out = sys.argv[sys.argv.index("--") + 1]
options = dict(
    filepath=out,
    export_format='GLB',
    export_apply=True,               # modifiers
    export_animations=True,
    export_animation_mode='ACTIONS', # every action becomes a clip
    export_materials='EXPORT',
    export_yup=True,
)
try:
    bpy.ops.export_scene.gltf(**options)
except TypeError as error:
    # An option this Blender version doesn't know; use its defaults instead.
    print("vulkano: retrying the export with default options:", error)
    bpy.ops.export_scene.gltf(filepath=out, export_format='GLB')
"#;

static BLENDER: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn is_blend(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".blend")
}

/// Use this Blender executable, for example one chosen in the editor's settings.
pub fn set_blender_path(path: Option<PathBuf>) {
    *BLENDER.lock().unwrap() = path.filter(|p| !p.as_os_str().is_empty());
}

/// The Blender executable to convert with, if there is one.
pub fn find_blender() -> Option<PathBuf> {
    if let Some(chosen) = BLENDER.lock().unwrap().clone() {
        return Some(chosen);
    }
    if let Some(path) = std::env::var_os("VULKANO_BLENDER").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(path);
    }
    let exe = if cfg!(windows) { "blender.exe" } else { "blender" };
    if let Some(path) = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|dir| dir.join(exe))
        .find(|p| p.is_file())
    {
        return Some(path);
    }
    install_candidates().into_iter().find(|p| p.is_file())
}

fn install_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if cfg!(windows) {
        // The official installer: "Blender Foundation/Blender 4.2/blender.exe", newest first.
        for root in ["C:\\Program Files\\Blender Foundation", "C:\\Program Files (x86)\\Blender Foundation"] {
            let mut versions: Vec<PathBuf> = std::fs::read_dir(root)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path().join("blender.exe"))
                .collect();
            versions.sort();
            candidates.extend(versions.into_iter().rev());
        }
        // Steam, in its own folder and every library it knows about.
        for steam in ["C:\\Program Files (x86)\\Steam", "C:\\Program Files\\Steam", "C:\\Steam"] {
            candidates.push(Path::new(steam).join("steamapps\\common\\Blender\\blender.exe"));
            let libraries = std::fs::read_to_string(Path::new(steam).join("steamapps\\libraryfolders.vdf"));
            for line in libraries.unwrap_or_default().lines() {
                // "path"		"D:\\SteamLibrary"
                let mut parts = line.split('"').filter(|p| !p.trim().is_empty());
                if parts.next() == Some("path") {
                    if let Some(library) = parts.next() {
                        let library = library.replace("\\\\", "\\");
                        candidates.push(Path::new(&library).join("steamapps\\common\\Blender\\blender.exe"));
                    }
                }
            }
        }
    } else if cfg!(target_os = "macos") {
        candidates.push("/Applications/Blender.app/Contents/MacOS/Blender".into());
    } else {
        candidates.push("/usr/bin/blender".into());
        candidates.push("/snap/bin/blender".into());
        if let Some(home) = std::env::var_os("HOME") {
            candidates.push(Path::new(&home).join(".steam/steam/steamapps/common/Blender/blender"));
        }
    }
    candidates
}

/// Where the converted copy of a `.blend` lives. The name keeps the folder it came from
/// apart, so two `player.blend`s in different folders don't collide.
pub fn cached_glb(blend: &str) -> PathBuf {
    let normalized = blend.replace('\\', "/");
    let stem = Path::new(&normalized).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    // FNV-1a: stable between runs, unlike the standard library's hasher.
    let hash = normalized.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    Path::new(IMPORT_DIR).join(format!("{stem}-{:08x}.glb", hash as u32))
}

fn is_fresh(blend: &Path, glb: &Path) -> bool {
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    matches!((modified(blend), modified(glb)), (Some(b), Some(g)) if g >= b)
}

/// The `.glb` for a `.blend`, exporting it with Blender if the cached copy is missing or
/// older than the `.blend`. This can take several seconds.
pub fn convert(blend: &str) -> Result<PathBuf, String> {
    let source = Path::new(blend);
    if !source.is_file() {
        return Err(format!("Model not found: {blend}"));
    }
    let glb = cached_glb(blend);
    if is_fresh(source, &glb) {
        return Ok(glb);
    }
    let blender = find_blender().ok_or_else(|| {
        format!(
            "{blend}: Blender is needed to import .blend files and wasn't found. Install it, \
             or set its path in the editor (Project > Project Settings) or in VULKANO_BLENDER."
        )
    })?;
    std::fs::create_dir_all(IMPORT_DIR).map_err(|e| format!("could not create {IMPORT_DIR}: {e}"))?;
    let script = Path::new(IMPORT_DIR).join("export_gltf.py");
    std::fs::write(&script, EXPORT_SCRIPT).map_err(|e| format!("could not write {}: {e}", script.display()))?;

    let absolute = |p: &Path| std::env::current_dir().map(|d| d.join(p)).unwrap_or_else(|_| p.to_path_buf());
    let output = Command::new(&blender)
        .arg("--background")
        .arg("--factory-startup")
        .arg(absolute(source))
        .arg("--python-exit-code")
        .arg("1")
        .arg("--python")
        .arg(absolute(&script))
        .arg("--")
        .arg(absolute(&glb))
        .output()
        .map_err(|e| format!("could not run Blender ({}): {e}", blender.display()))?;
    if !output.status.success() || !glb.is_file() {
        let log = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        let tail: Vec<&str> = log.lines().rev().filter(|l| !l.trim().is_empty()).take(6).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        return Err(format!("Blender could not export {blend}:\n{}", tail.join("\n")));
    }
    Ok(glb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_names_differ_by_folder() {
        assert_ne!(cached_glb("a/player.blend"), cached_glb("b/player.blend"));
        assert_eq!(cached_glb("a/player.blend"), cached_glb("a\\player.blend"));
        assert!(cached_glb("a/player.blend").to_string_lossy().contains("player-"));
    }

    #[test]
    fn blend_paths_are_recognised() {
        assert!(is_blend("models/Hero.BLEND"));
        assert!(!is_blend("models/hero.glb"));
    }
}
