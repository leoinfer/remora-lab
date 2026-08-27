//! Static boundary gate for the HAR production subtree.
//!
//! This program is deliberately dependency-free. It complements dependency
//! metadata and executable tracing by checking that the checked-in production
//! boundary contains Rust host code only, with shader files as the explicit
//! GPU exception.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PRODUCTION_ROOTS: &[&str] = &["crates"];
const FORBIDDEN_NATIVE_EXTENSIONS: &[&str] = &["c", "cc", "cpp", "h", "hh", "hpp"];

fn main() {
    let root = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut violations = Vec::new();
    let mut files_scanned = 0usize;

    for relative in PRODUCTION_ROOTS {
        let path = root.join(relative);
        if path.exists() {
            scan_tree(&root, &path, &mut violations, &mut files_scanned);
        }
    }
    check_manifest(&root.join("Cargo.toml"), &mut violations);
    let crates = root.join("crates");
    if crates.exists() {
        scan_manifests(&crates, &mut violations);
    }
    for name in ["CMakeLists.txt", "build.rs"] {
        if root.join(name).exists() {
            violations.push(format!("{}: obsolete native build boundary exists", root.join(name).display()));
        }
    }

    if violations.is_empty() {
        println!("RUST_ONLY_RUNTIME PASS: scanned {files_scanned} production files");
        return;
    }
    eprintln!("RUST_ONLY_RUNTIME FAIL: {} violation(s)", violations.len());
    for violation in violations {
        eprintln!("  {violation}");
    }
    std::process::exit(1);
}

fn scan_tree(root: &Path, path: &Path, violations: &mut Vec<String>, files_scanned: &mut usize) {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) => {
            violations.push(format!("{}: cannot read: {error}", display(root, path)));
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().is_some_and(|name| name == "target") {
            continue;
        }
        if path.is_dir() {
            scan_tree(root, &path, violations, files_scanned);
            continue;
        }
        *files_scanned += 1;
        let relative = display(root, &path);
        let extension = path.extension().and_then(|value| value.to_str()).unwrap_or("");
        let file_name = path.file_name().and_then(|value| value.to_str()).unwrap_or("");
        if file_name == "build.rs" || FORBIDDEN_NATIVE_EXTENSIONS.contains(&extension) {
            violations.push(format!("{relative}: non-Rust host source"));
        }
        if extension == "py" {
            violations.push(format!("{relative}: Python is outside the production runtime"));
        }
        if extension == "rs" {
            check_rust_source(&path, &relative, violations);
        }
    }
}

fn check_rust_source(path: &Path, relative: &str, violations: &mut Vec<String>) {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            violations.push(format!("{relative}: cannot read Rust source: {error}"));
            return;
        }
    };
    let lower = source.to_ascii_lowercase();
    for marker in [
        "std::process::command",
        "command::new(",
        "extern \"c\"",
        "#[link",
        "libloading",
        "llama_cpp",
        "llama.cpp",
        "python3",
    ] {
        if lower.contains(marker) {
            violations.push(format!("{relative}: forbidden runtime marker `{marker}`"));
        }
    }
}

fn scan_manifests(path: &Path, violations: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(path) else { return };
    for entry in entries.flatten() {
        let child = entry.path();
        if child.is_dir() {
            scan_manifests(&child, violations);
        } else if child.file_name().is_some_and(|name| name == "Cargo.toml") {
            check_manifest(&child, violations);
        }
    }
}

fn check_manifest(path: &Path, violations: &mut Vec<String>) {
    let Ok(source) = fs::read_to_string(path) else { return };
    for line in source.lines() {
        let lower = line.to_ascii_lowercase();
        let trimmed = lower.trim();
        if trimmed.starts_with("cc ")
            || trimmed.starts_with("cc=")
            || trimmed.starts_with("cmake ")
            || trimmed.starts_with("cmake=")
            || trimmed.starts_with("bindgen ")
            || trimmed.starts_with("bindgen=")
            || trimmed.starts_with("ggml ")
            || trimmed.starts_with("ggml=")
            || trimmed.contains("llama_cpp")
        {
            violations.push(format!("{}: forbidden dependency marker: {line}", path.display()));
        }
    }
}

fn display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).display().to_string()
}
