//! Small dependency-free audit for the clean-room publication tree.
//!
//! This is a release gate, not part of HAR's model-serving runtime. It checks
//! the tree as published: paths, file classes, model-payload exclusions,
//! private-path leakage, obvious credentials, and forbidden production
//! implementation surfaces.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024;

#[derive(Default)]
struct Audit {
    files: usize,
    bytes: u64,
    findings: Vec<String>,
}

impl Audit {
    fn finding(&mut self, path: &Path, message: impl Into<String>) {
        self.findings
            .push(format!("{}: {}", path.display(), message.into()));
    }
}

fn main() {
    let root = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().expect("current directory"));
    let root = root
        .canonicalize()
        .unwrap_or_else(|error| panic!("cannot resolve {}: {error}", root.display()));

    let mut audit = Audit::default();
    walk(&root, &root, &mut audit);

    if audit.findings.is_empty() {
        println!(
            "PUBLICATION_AUDIT PASS: {} files, {} bytes, no release-gate findings",
            audit.files, audit.bytes
        );
        return;
    }

    eprintln!(
        "PUBLICATION_AUDIT FAIL: {} files, {} bytes, {} finding(s)",
        audit.files,
        audit.bytes,
        audit.findings.len()
    );
    for finding in audit.findings {
        eprintln!("- {finding}");
    }
    std::process::exit(1);
}

fn walk(root: &Path, path: &Path, audit: &mut Audit) {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) => {
            audit.finding(path, format!("cannot read directory: {error}"));
            return;
        }
    };

    let mut entries: Vec<_> = entries.filter_map(Result::ok).collect();
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let relative_text = relative.to_string_lossy();
        if relative.components().any(|component| {
            matches!(
                component.as_os_str().to_string_lossy().as_ref(),
                ".git" | "target"
            )
        }) {
            continue;
        }

        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => {
                audit.finding(relative, format!("cannot stat: {error}"));
                continue;
            }
        };
        if metadata.file_type().is_symlink() {
            audit.finding(
                relative,
                "symbolic links are not allowed in the publication tree",
            );
            continue;
        }
        if metadata.is_dir() {
            walk(root, &path, audit);
            continue;
        }
        if !metadata.is_file() {
            audit.finding(relative, "special filesystem entries are not allowed");
            continue;
        }

        audit.files += 1;
        audit.bytes = audit.bytes.saturating_add(metadata.len());
        check_path(relative, &relative_text, metadata.len(), audit);

        if metadata.len() > MAX_FILE_BYTES {
            audit.finding(
                relative,
                format!(
                    "file is larger than the {} MiB source limit",
                    MAX_FILE_BYTES / 1024 / 1024
                ),
            );
        }

        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                audit.finding(relative, format!("cannot read: {error}"));
                continue;
            }
        };
        if bytes.contains(&0) {
            check_binary(relative, metadata.len(), audit);
        } else if let Ok(text) = std::str::from_utf8(&bytes) {
            check_text(relative, &relative_text, text, audit);
        }
    }
}

fn check_path(path: &Path, text: &str, size: u64, audit: &mut Audit) {
    let lower = text.to_ascii_lowercase();
    let production = lower == "har" || lower.starts_with("har/");
    let denied_suffixes = [
        ".gguf",
        ".safetensors",
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".npz",
        ".npy",
        ".parquet",
        ".sqlite",
        ".db",
        ".zip",
        ".tar",
        ".gz",
        ".xz",
        ".zst",
        ".7z",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".mp4",
        ".wav",
    ];
    if denied_suffixes.iter().any(|suffix| lower.ends_with(suffix)) {
        audit.finding(
            path,
            "model, archive, media, or experiment payload is not publishable",
        );
    }

    if production {
        let forbidden_source_suffixes = [".py", ".pyc", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"];
        if forbidden_source_suffixes
            .iter()
            .any(|suffix| lower.ends_with(suffix))
            || lower.ends_with("/cmakelists.txt")
        {
            audit.finding(path, "non-Rust production implementation file");
        }
        for marker in ["llama", "ggml", "reference_llama"] {
            if lower.contains(marker) {
                audit.finding(path, format!("forbidden production path marker: {marker}"));
            }
        }
    }

    if size == 0 {
        audit.finding(path, "empty files are not allowed in the publication tree");
    }
}

fn check_binary(path: &Path, size: u64, audit: &mut Audit) {
    let text = path.to_string_lossy();
    let allowed_fixture = (text.starts_with("har/fixtures/")
        || (text.starts_with("har/shaders/") && text.ends_with(".spv")))
        && size <= 1024 * 1024;
    if !allowed_fixture {
        audit.finding(
            path,
            "binary payload is outside the reviewed fixture allowlist",
        );
    }
}

fn check_text(path: &Path, relative: &str, text: &str, audit: &mut Audit) {
    // This source file contains the audit vocabulary by design. The denylist
    // is also a policy document that names examples. Both are reviewed as
    // policy rather than treated as leaked data.
    if relative == "tools/publication-audit/src/main.rs" || relative == "PUBLICATION_DENYLIST.md" {
        return;
    }

    let lower = text.to_ascii_lowercase();
    let sensitive_markers = [
        "/home/",
        "/users/",
        "c:\\users\\",
        "/mnt/",
        "$home",
        "cachyos",
        "ghp_",
        "github_pat_",
        "glpat-",
        "xoxb-",
        "begin private key",
        "authorization: bearer",
        "api_key=",
        "api-key=",
        "secret_key=",
        "access_token=",
        "github.com/leoinfer/",
        "swarm/",
    ];
    for marker in sensitive_markers {
        if lower.contains(marker) {
            audit.finding(path, format!("sensitive or private marker: {marker}"));
        }
    }

    let production = relative == "har" || relative.starts_with("har/");
    if production {
        let code_like = [".rs", ".comp", ".glsl", ".har"]
            .iter()
            .any(|suffix| lower.ends_with(suffix));
        if code_like {
            for marker in [
                "command::new",
                "std::process::command",
                "extern \"c\"",
                "llama.cpp",
                "llama_cpp",
                "llama-cpp",
            ] {
                if lower.contains(marker) {
                    audit.finding(
                        path,
                        format!("forbidden production execution marker: {marker}"),
                    );
                }
            }
        }
    }
}
