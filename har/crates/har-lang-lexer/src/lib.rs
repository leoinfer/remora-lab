//! Allocation is confined to source compilation.  The token loop never calls this crate.

use har_lang_diagnostics::{Diagnostic, Span};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Symbol {
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Semicolon,
    Comma,
    Colon,
    DotDot,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    Ident(String),
    String(String),
    Number(String),
    Symbol(Symbol),
    Eof,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

fn symbol(byte: u8, next: Option<u8>) -> Option<(Symbol, usize)> {
    let value = match byte {
        b'{' => (Symbol::LBrace, 1),
        b'}' => (Symbol::RBrace, 1),
        b'[' => (Symbol::LBracket, 1),
        b']' => (Symbol::RBracket, 1),
        b'(' => (Symbol::LParen, 1),
        b')' => (Symbol::RParen, 1),
        b';' => (Symbol::Semicolon, 1),
        b',' => (Symbol::Comma, 1),
        b':' => (Symbol::Colon, 1),
        b'.' if next == Some(b'.') => (Symbol::DotDot, 2),
        _ => return None,
    };
    Some(value)
}

pub fn lex(source: &str) -> Result<Vec<Token>, Vec<Diagnostic>> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    let mut index = 0usize;
    let mut line = 1usize;
    let mut column = 1usize;

    let advance = |index: &mut usize, line: &mut usize, column: &mut usize, count: usize| {
        for byte in &bytes[*index..(*index + count).min(bytes.len())] {
            if *byte == b'\n' {
                *line += 1;
                *column = 1;
            } else {
                *column += 1;
            }
        }
        *index = (*index + count).min(bytes.len());
    };

    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_whitespace() {
            advance(&mut index, &mut line, &mut column, 1);
            continue;
        }
        if byte == b'/' && bytes.get(index + 1) == Some(&b'/') {
            while index < bytes.len() && bytes[index] != b'\n' {
                advance(&mut index, &mut line, &mut column, 1);
            }
            continue;
        }
        let start = index;
        let start_line = line;
        let start_column = column;
        if let Some((kind, width)) = symbol(byte, bytes.get(index + 1).copied()) {
            advance(&mut index, &mut line, &mut column, width);
            tokens.push(Token {
                kind: TokenKind::Symbol(kind),
                span: Span {
                    start,
                    end: index,
                    line: start_line,
                    column: start_column,
                },
            });
            continue;
        }
        if byte == b'"' {
            advance(&mut index, &mut line, &mut column, 1);
            let mut value = String::new();
            let mut terminated = false;
            while index < bytes.len() {
                match bytes[index] {
                    b'"' => {
                        advance(&mut index, &mut line, &mut column, 1);
                        terminated = true;
                        break;
                    }
                    b'\\' if index + 1 < bytes.len() => {
                        advance(&mut index, &mut line, &mut column, 1);
                        let escaped = match bytes[index] {
                            b'n' => '\n',
                            b'r' => '\r',
                            b't' => '\t',
                            other => other as char,
                        };
                        value.push(escaped);
                        advance(&mut index, &mut line, &mut column, 1);
                    }
                    b'\n' => {
                        errors.push(Diagnostic::error(
                            "L0003",
                            "newline is not allowed inside a string",
                            Some(Span {
                                start,
                                end: index + 1,
                                line: start_line,
                                column: start_column,
                            }),
                        ));
                        advance(&mut index, &mut line, &mut column, 1);
                    }
                    other => {
                        value.push(other as char);
                        advance(&mut index, &mut line, &mut column, 1);
                    }
                }
            }
            if !terminated {
                errors.push(Diagnostic::error(
                    "L0004",
                    "unterminated string literal",
                    Some(Span {
                        start,
                        end: index,
                        line: start_line,
                        column: start_column,
                    }),
                ));
            } else {
                tokens.push(Token {
                    kind: TokenKind::String(value),
                    span: Span {
                        start,
                        end: index,
                        line: start_line,
                        column: start_column,
                    },
                });
            }
            continue;
        }
        if byte.is_ascii_digit()
            || (byte == b'.' && bytes.get(index + 1).is_some_and(u8::is_ascii_digit))
        {
            advance(&mut index, &mut line, &mut column, 1);
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                advance(&mut index, &mut line, &mut column, 1);
            }
            if bytes.get(index) == Some(&b'.') && bytes.get(index + 1) != Some(&b'.') {
                advance(&mut index, &mut line, &mut column, 1);
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    advance(&mut index, &mut line, &mut column, 1);
                }
            }
            if matches!(bytes.get(index), Some(b'e' | b'E')) {
                advance(&mut index, &mut line, &mut column, 1);
                if matches!(bytes.get(index), Some(b'+' | b'-')) {
                    advance(&mut index, &mut line, &mut column, 1);
                }
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    advance(&mut index, &mut line, &mut column, 1);
                }
            }
            tokens.push(Token {
                kind: TokenKind::Number(source[start..index].to_string()),
                span: Span {
                    start,
                    end: index,
                    line: start_line,
                    column: start_column,
                },
            });
            continue;
        }
        if byte.is_ascii_alphabetic() || byte == b'_' {
            advance(&mut index, &mut line, &mut column, 1);
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'_' | b'-'))
            {
                advance(&mut index, &mut line, &mut column, 1);
            }
            tokens.push(Token {
                kind: TokenKind::Ident(source[start..index].to_string()),
                span: Span {
                    start,
                    end: index,
                    line: start_line,
                    column: start_column,
                },
            });
            continue;
        }
        errors.push(Diagnostic::error(
            "L0001",
            format!("unexpected character `{}`", byte as char),
            Some(Span {
                start,
                end: start + 1,
                line: start_line,
                column: start_column,
            }),
        ));
        advance(&mut index, &mut line, &mut column, 1);
    }
    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::point(index, line, column),
    });
    if errors.is_empty() {
        Ok(tokens)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_quantity_and_range() {
        let tokens = lex("vram_budget 15.9 GiB; horizon 0..3;").expect("lex");
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Symbol(Symbol::DotDot)));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Number("15.9".into())));
    }
}
