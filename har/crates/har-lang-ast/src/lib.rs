//! Syntax-only AST.  It intentionally carries no executable callbacks or
//! runtime state; all values are lowered before decode starts.

use har_lang_diagnostics::Span;
use har_lang_lexer::TokenKind;

#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub source_name: String,
    pub declarations: Vec<Block>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub kind: String,
    pub name: String,
    pub fields: Vec<Field>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub key: String,
    pub value: Value,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Ident(String),
    String(String),
    Number(String),
    Quantity { number: String, unit: String },
    Range { start: String, end: String },
    List(Vec<Value>),
}

impl Value {
    pub fn as_atom(&self) -> Option<&str> {
        match self {
            Self::Ident(value) | Self::String(value) | Self::Number(value) => Some(value),
            Self::Quantity { .. } | Self::Range { .. } | Self::List(_) => None,
        }
    }

    pub fn as_ident(&self) -> Option<&str> {
        match self {
            Self::Ident(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }
}

/// Kept as a tiny helper so downstream crates can identify punctuation without
/// depending on parser implementation details.
pub fn is_identifier(token: &TokenKind) -> bool {
    matches!(token, TokenKind::Ident(_))
}
