//! Small dependency-free audit for the clean-room publication tree.
//!
//! This is a release gate, not part of HAR's model-serving runtime. It checks
//! the tree as published: paths, file classes, model-payload exclusions,
//! private-path leakage, obvious credentials, and forbidden production
//! implementation surfaces.

use std::env;
use std::fs;
use std::net::Ipv6Addr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

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

fn push_unique(findings: &mut Vec<&'static str>, finding: &'static str) {
    if !findings.contains(&finding) {
        findings.push(finding);
    }
}

fn trim_candidate(value: &str) -> &str {
    value.trim_matches(|character: char| {
        matches!(
            character,
            '`' | '"'
                | '\''
                | ','
                | ';'
                | '!'
                | '?'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
        )
    })
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn assignment_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let lower = line.to_ascii_lowercase();
    let mut search_from = 0;

    while let Some(relative) = lower.get(search_from..)?.find(key) {
        let index = search_from + relative;
        let end = index + key.len();
        let at_word_start = index == 0 || !is_word_byte(lower.as_bytes()[index - 1]);
        let at_word_end = end == lower.len() || !is_word_byte(lower.as_bytes()[end]);
        if at_word_start && at_word_end {
            let rest = line.get(end..)?.trim_start();
            if matches!(rest.as_bytes().first(), Some(b'=' | b':')) {
                return Some(rest[1..].trim());
            }
        }
        search_from = end;
    }

    None
}

fn assignment_value_without_wrappers(value: &str) -> &str {
    value.trim_matches(|character: char| {
        matches!(character, '`' | '"' | '\'' | ',' | ';' | ')' | ']' | '}')
    })
}

fn looks_placeholder(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        "false",
        "true",
        "null",
        "none",
        "unknown",
        "omitted",
        "not set",
        "not published",
        "redacted",
        "placeholder",
        "example",
        "dummy",
        "fake",
        "changeme",
        "replace",
        "your_",
        "your-",
        "<",
        ">",
        "...",
        "token id",
        "token_id",
        "token sequence",
        "input_token",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn looks_like_credential_assignment(value: &str, key: &str) -> bool {
    let wrapped = value.trim();
    let value = assignment_value_without_wrappers(value);
    if value.len() < 8 || looks_placeholder(value) {
        return false;
    }

    if key == "token" {
        let explicitly_quoted = matches!(wrapped.as_bytes().first(), Some(b'"' | b'\'' | b'`'));
        if (!explicitly_quoted && !looks_like_jwt(value) && !looks_like_prefixed_credential(value))
            || value.bytes().all(|byte| byte.is_ascii_digit())
            || value.split_whitespace().count() > 1
            || value.to_ascii_lowercase().contains("token")
        {
            return false;
        }
    }

    true
}

fn looks_like_ipv4(candidate: &str) -> bool {
    let parts: Vec<_> = candidate.split('.').collect();
    parts.len() == 4
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.len() <= 3
                && part.bytes().all(|byte| byte.is_ascii_digit())
                && part.parse::<u8>().is_ok()
        })
}

fn ipv4_host(candidate: &str) -> &str {
    let candidate = candidate.trim_matches(|character| matches!(character, '[' | ']'));
    if let Some((host, port)) = candidate.rsplit_once(':') {
        if !port.is_empty() && port.bytes().all(|byte| byte.is_ascii_digit()) {
            return host.trim_matches(|character| matches!(character, '[' | ']'));
        }
    }
    candidate
}

fn looks_like_ipv6(candidate: &str) -> bool {
    candidate.matches(':').count() >= 2 && Ipv6Addr::from_str(candidate).is_ok()
}

fn looks_like_mac(candidate: &str) -> bool {
    let separator = if candidate.contains(':') { ':' } else { '-' };
    let parts: Vec<_> = candidate.split(separator).collect();
    parts.len() == 6
        && parts
            .iter()
            .all(|part| part.len() == 2 && part.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn looks_like_uuid(candidate: &str) -> bool {
    let parts: Vec<_> = candidate.split('-').collect();
    parts.len() == 5
        && [8, 4, 4, 4, 12]
            .iter()
            .zip(parts.iter())
            .all(|(length, part)| {
                part.len() == *length && part.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
}

fn looks_like_jwt(candidate: &str) -> bool {
    let parts: Vec<_> = candidate.split('.').collect();
    parts.len() == 3
        && parts[0].starts_with("eyJ")
        && parts.iter().all(|part| {
            part.len() >= 8
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        })
}

fn looks_like_prefixed_credential(candidate: &str) -> bool {
    const PREFIXES: &[(&str, usize)] = &[
        ("ghp_", 8),
        ("gho_", 8),
        ("ghu_", 8),
        ("ghs_", 8),
        ("ghr_", 8),
        ("github_pat_", 8),
        ("glpat-", 8),
        ("sk-", 12),
        ("hf_", 8),
        ("AKIA", 16),
        ("ASIA", 16),
        ("AIza", 16),
        ("xoxa-", 8),
        ("xoxb-", 8),
        ("xoxp-", 8),
        ("xoxr-", 8),
        ("xoxs-", 8),
    ];

    PREFIXES.iter().any(|(prefix, minimum_suffix)| {
        candidate.starts_with(prefix)
            && candidate[prefix.len()..].len() >= *minimum_suffix
            && candidate[prefix.len()..]
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    })
}

fn looks_like_email(candidate: &str) -> bool {
    let mut parts = candidate.split('@');
    let local = parts.next().unwrap_or_default();
    let domain = parts.next().unwrap_or_default();
    if local.is_empty() || domain.is_empty() || parts.next().is_some() || domain.contains("..") {
        return false;
    }

    let labels: Vec<_> = domain.split('.').collect();
    labels.len() >= 2
        && labels.iter().all(|label| {
            !label.is_empty()
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                && !label.starts_with('-')
                && !label.ends_with('-')
        })
        && labels
            .last()
            .is_some_and(|label| label.bytes().all(|byte| byte.is_ascii_alphabetic()))
        && local
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._%+-".contains(&byte))
}

fn looks_like_software_version(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let version_context = ["version", "vulkan", "spirv", "sdk", "loader", "conformance"];
    let network_context = ["address", "bind", "host", "ip", "server", "socket", "http"];
    version_context.iter().any(|marker| lower.contains(marker))
        && !network_context.iter().any(|marker| lower.contains(marker))
}

fn looks_like_private_tmp_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let Some(suffix) = lower.strip_prefix("/tmp/") else {
        return false;
    };
    ["user", "home", "private", "secret", "hostname", "machine"]
        .iter()
        .any(|marker| suffix.contains(marker))
        || (suffix.contains("-gpu") && suffix.contains("lock"))
}

fn scan_private_tmp_paths(text: &str, findings: &mut Vec<&'static str>) {
    let mut remaining = text;
    while let Some(index) = remaining.find("/tmp/") {
        let path = &remaining[index..];
        let end = path
            .find(|character: char| {
                character.is_whitespace()
                    || matches!(
                        character,
                        '`' | '"' | '\'' | '<' | '>' | ',' | ';' | ')' | ']'
                    )
            })
            .unwrap_or(path.len());
        if looks_like_private_tmp_path(trim_candidate(&path[..end])) {
            push_unique(findings, "private temporary path");
        }
        remaining = &path[end..];
        if remaining.is_empty() {
            break;
        }
        remaining = &remaining[1..];
    }
}

fn generic_privacy_findings(text: &str) -> Vec<&'static str> {
    let mut findings = Vec::new();

    for line in text.lines() {
        for raw in line.split(|character: char| {
            character.is_whitespace()
                || matches!(
                    character,
                    '/' | '\\' | '[' | ']' | '(' | ')' | '{' | '}' | '<' | '>'
                )
        }) {
            let candidate = trim_candidate(raw);
            if candidate.is_empty() {
                continue;
            }

            if looks_like_email(candidate) {
                push_unique(&mut findings, "email address");
            }

            let host = ipv4_host(candidate);
            if looks_like_ipv4(host)
                && !host.starts_with("127.")
                && !looks_like_software_version(line)
            {
                push_unique(&mut findings, "IPv4 address");
            }
            if looks_like_ipv6(candidate) {
                push_unique(&mut findings, "IPv6 address");
            }
            if looks_like_mac(candidate) {
                push_unique(&mut findings, "MAC address");
            }
            if looks_like_uuid(candidate) {
                push_unique(&mut findings, "UUID");
            }
            if looks_like_jwt(candidate) || looks_like_prefixed_credential(candidate) {
                push_unique(&mut findings, "credential token");
            }
        }
    }

    let lower = text.to_ascii_lowercase();
    if lower.contains("-----begin ") && lower.contains("private key-----") {
        push_unique(&mut findings, "private key header");
    }

    let credential_assignment_found = text.lines().any(|line| {
        [
            "password",
            "passwd",
            "api_key",
            "apikey",
            "api-key",
            "secret",
            "secret_key",
            "client_secret",
            "access_token",
            "refresh_token",
            "authorization",
            "bearer",
            "token",
        ]
        .iter()
        .any(|key| {
            assignment_value(line, key)
                .is_some_and(|value| looks_like_credential_assignment(value, key))
        })
    });
    if credential_assignment_found {
        push_unique(&mut findings, "credential assignment");
    }

    for line in text.lines() {
        for key in [
            "serial",
            "serial_number",
            "machine_id",
            "machine-id",
            "disk_uuid",
            "filesystem_uuid",
        ] {
            if assignment_value(line, key).is_some_and(|value| {
                let value = assignment_value_without_wrappers(value);
                !value.is_empty() && !looks_placeholder(value)
            }) {
                push_unique(&mut findings, "serial or machine identifier");
            }
        }
    }

    scan_private_tmp_paths(text, &mut findings);
    findings.sort_unstable();
    findings
}

fn runtime_identity_markers() -> Vec<String> {
    ["USER", "USERNAME", "HOSTNAME", "COMPUTERNAME"]
        .iter()
        .filter_map(|variable| env::var(variable).ok())
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| {
            value.len() >= 3
                && value != "root"
                && value != "user"
                && value != "localhost"
                && value != "unknown"
                && value != "leoinfer"
        })
        .collect()
}

fn contains_identity_marker(text: &str, marker: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    if marker
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return lower
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .any(|token| token == marker);
    }

    lower
        .split(|character: char| {
            !character.is_ascii_alphanumeric() && character != '-' && character != '_'
        })
        .any(|token| token == marker)
}

fn check_text(path: &Path, relative: &str, text: &str, audit: &mut Audit) {
    // This source file contains the audit vocabulary by design. The denylist
    // is also a policy document that names examples. Both are reviewed as
    // policy rather than treated as leaked data.
    if relative == "tools/publication-audit/src/main.rs" || relative == "PUBLICATION_DENYLIST.md" {
        return;
    }

    let lower = text.to_ascii_lowercase();
    // Public operating-system names are not private identifiers. Private
    // paths, hostnames, and credentials remain denylisted below.
    let sensitive_markers = [
        "/home/",
        "/users/",
        "c:\\users\\",
        "/mnt/",
        "$home",
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

    for finding in generic_privacy_findings(text) {
        audit.finding(path, format!("privacy/credential marker: {finding}"));
    }
    for marker in runtime_identity_markers() {
        if contains_identity_marker(text, &marker) {
            audit.finding(path, "local username or hostname marker");
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

#[cfg(test)]
mod tests {
    use super::generic_privacy_findings;

    #[test]
    fn synthetic_private_values_are_rejected() {
        let password = ["correct", "-horse", "-battery", "-staple"].concat();
        let serial = ["BOARD", "-", "123456"].concat();
        let temporary_path = ["user", "-gpu0.lock"].concat();
        let fixture = format!(
            "{}{} {}.{}.{}.{} {}:{}::{} {}:{}:{}:{}:{}:{} {}-{}-{}-{}-{} {}{} {}{} {}{} {}{} {}{} {}{} {}.{}.{} {}{} password = \"{}\" serial = \"{}\" /tmp/{}",
            "owner",
            "@example.com",
            203,
            0,
            113,
            42,
            "2001",
            "db8",
            42,
            "aa",
            "bb",
            "cc",
            "dd",
            "ee",
            "ff",
            "550e8400",
            "e29b",
            "41d4",
            "a716",
            "446655440000",
            "ghp_",
            "FAKEVALUE123456",
            "sk-proj-",
            "FAKEVALUE123456",
            "hf_",
            "FAKEVALUE123456",
            "AKIA",
            "1234567890ABCD",
            "AIza",
            "1234567890ABCDEF",
            "xoxb-",
            "FAKEVALUE123456",
            "eyJhbGciOiJIUzI1NiJ9",
            "eyJzdWIiOiIxMjMifQ",
            "signature",
            "-----BEGIN ",
            "PRIVATE KEY-----",
            password,
            serial,
            temporary_path
        );
        let findings = generic_privacy_findings(&fixture);

        for expected in [
            "email address",
            "IPv4 address",
            "IPv6 address",
            "MAC address",
            "UUID",
            "credential token",
            "private key header",
            "credential assignment",
            "serial or machine identifier",
            "private temporary path",
        ] {
            assert!(findings.contains(&expected), "missing finding: {expected}");
        }
    }

    #[test]
    fn ordinary_model_token_terminology_is_allowed() {
        let ordinary = concat!(
            "token IDs and token sequence length are model terminology; ",
            "token = 7; token: 12345; model version 1.4.5.3; ",
            "Vulkan loader 1.4.357.0; PCI product 1002:7590"
        );

        assert!(generic_privacy_findings(ordinary).is_empty());
    }
}
