//! Diagnostics shared by every offline HAR language stage.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

impl Span {
    pub const fn point(start: usize, line: usize, column: usize) -> Self {
        Self {
            start,
            end: start,
            line,
            column,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn warning(
        code: impl Into<String>,
        message: impl Into<String>,
        span: Option<Span>,
    ) -> Self {
        Self {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn note(code: impl Into<String>, message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            severity: Severity::Note,
            code: code.into(),
            message: message.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn render(&self, source_name: &str) -> String {
        let severity = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        };
        let location = self.span.map_or_else(
            || source_name.to_string(),
            |span| format!("{}:{}:{}", source_name, span.line, span.column),
        );
        let mut rendered = format!("{location}: {severity}[{}]: {}", self.code, self.message);
        for note in &self.notes {
            rendered.push_str(&format!("\n  note: {note}"));
        }
        rendered
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Diagnostics {
    pub items: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.items.push(diagnostic);
    }
    pub fn extend<I: IntoIterator<Item = Diagnostic>>(&mut self, diagnostics: I) {
        self.items.extend(diagnostics);
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn has_errors(&self) -> bool {
        self.items
            .iter()
            .any(|item| item.severity == Severity::Error)
    }
    pub fn into_result<T>(self, value: T) -> Result<T, Vec<Diagnostic>> {
        if self.has_errors() {
            Err(self.items)
        } else {
            Ok(value)
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.render("<har>"))
    }
}
