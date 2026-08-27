use har_kv::StableDigest;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PolicyError {
    MissingBlock,
    UnknownDirective(String),
    InvalidValue(String),
    DuplicateDirective(String),
    MissingDirective(&'static str),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBlock => {
                formatter.write_str("context policy must contain a context block")
            }
            Self::UnknownDirective(value) => {
                write!(formatter, "unknown context directive: {value}")
            }
            Self::InvalidValue(value) => write!(formatter, "invalid context policy value: {value}"),
            Self::DuplicateDirective(value) => {
                write!(formatter, "duplicate context directive: {value}")
            }
            Self::MissingDirective(value) => {
                write!(formatter, "missing required context directive: {value}")
            }
        }
    }
}
impl std::error::Error for PolicyError {}

/// Immutable compiled policy.  It is data consumed by the Rust controller;
/// there is no per-token interpreted HAR language execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledContextPolicy {
    pub policy_id: StableDigest,
    pub source_digest: StableDigest,
    pub hot_window_tokens: u32,
    pub exact_hot_format: String,
    pub cold_codec: String,
    pub fallback: String,
    pub require_tail_probe_pass: bool,
    pub tail_probe_pass: bool,
}

impl CompiledContextPolicy {
    pub fn with_tail_probe_result(&self, pass: bool) -> Self {
        let mut copy = self.clone();
        copy.tail_probe_pass = pass;
        copy.policy_id = StableDigest::from_parts(&[
            self.source_digest.as_str().as_bytes(),
            if pass { b"tail-pass" } else { b"tail-fail" },
        ]);
        copy
    }

    pub fn admitted(&self) -> bool {
        !self.require_tail_probe_pass || self.tail_probe_pass
    }
}

pub fn compile_policy(source: &str) -> Result<CompiledContextPolicy, PolicyError> {
    let trimmed = source.trim();
    if !trimmed.starts_with("context {") || !trimmed.ends_with('}') {
        return Err(PolicyError::MissingBlock);
    }
    let body = &trimmed["context {".len()..trimmed.len() - 1];
    let mut hot_window_tokens = None;
    let mut exact_hot_format = None;
    let mut cold_codec = None;
    let mut fallback = None;
    let mut require_tail_probe_pass = None;
    for raw in body.split(';') {
        let statement = raw.trim();
        if statement.is_empty() {
            continue;
        }
        let words: Vec<&str> = statement.split_whitespace().collect();
        match words.as_slice() {
            ["hot", format, "window", tokens] => {
                if hot_window_tokens.is_some() {
                    return Err(PolicyError::DuplicateDirective("hot".to_string()));
                }
                if *format != "exact_q8" {
                    return Err(PolicyError::InvalidValue((*format).to_string()));
                }
                hot_window_tokens = Some(
                    tokens
                        .parse::<u32>()
                        .map_err(|_| PolicyError::InvalidValue((*tokens).to_string()))?,
                );
                exact_hot_format = Some((*format).to_string());
            }
            ["cold", "contextfold", "codec", codec] => {
                if cold_codec.is_some() {
                    return Err(PolicyError::DuplicateDirective("cold".to_string()));
                }
                if codec.is_empty() {
                    return Err(PolicyError::InvalidValue("empty codec".to_string()));
                }
                cold_codec = Some((*codec).to_string());
            }
            ["fallback", value] => {
                if fallback.is_some() {
                    return Err(PolicyError::DuplicateDirective("fallback".to_string()));
                }
                if *value != "token_archive" && *value != "full_reference" {
                    return Err(PolicyError::InvalidValue((*value).to_string()));
                }
                fallback = Some((*value).to_string());
            }
            ["require", "tail_probe_pass"] => {
                if require_tail_probe_pass.is_some() {
                    return Err(PolicyError::DuplicateDirective(
                        "require tail_probe_pass".to_string(),
                    ));
                }
                require_tail_probe_pass = Some(true);
            }
            _ => return Err(PolicyError::UnknownDirective(statement.to_string())),
        }
    }
    let hot_window_tokens =
        hot_window_tokens.ok_or(PolicyError::MissingDirective("hot exact_q8 window"))?;
    if hot_window_tokens == 0 {
        return Err(PolicyError::InvalidValue(
            "hot window must be positive".to_string(),
        ));
    }
    let exact_hot_format =
        exact_hot_format.ok_or(PolicyError::MissingDirective("hot exact_q8 window"))?;
    let cold_codec = cold_codec.ok_or(PolicyError::MissingDirective("cold contextfold codec"))?;
    let fallback = fallback.ok_or(PolicyError::MissingDirective("fallback"))?;
    let require_tail_probe_pass = require_tail_probe_pass.unwrap_or(false);
    let source_digest = StableDigest::from_text(trimmed);
    let policy_id = StableDigest::from_parts(&[
        source_digest.as_str().as_bytes(),
        b"compiled-context-policy-v1",
    ]);
    Ok(CompiledContextPolicy {
        policy_id,
        source_digest,
        hot_window_tokens,
        exact_hot_format,
        cold_codec,
        fallback,
        require_tail_probe_pass,
        tail_probe_pass: false,
    })
}
