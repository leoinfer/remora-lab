//! Recursive-descent parser for the deliberately small HAR configuration DSL.

use har_lang_ast::{Block, Field, Program, Value};
use har_lang_diagnostics::{Diagnostic, Span};
use har_lang_lexer::{Symbol, Token, TokenKind};

pub fn parse(tokens: &[Token], source_name: impl Into<String>) -> Result<Program, Vec<Diagnostic>> {
    Parser::new(tokens, source_name.into()).parse_program()
}

struct Parser<'a> {
    tokens: &'a [Token],
    index: usize,
    source_name: String,
    errors: Vec<Diagnostic>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token], source_name: String) -> Self {
        Self {
            tokens,
            index: 0,
            source_name,
            errors: Vec::new(),
        }
    }

    fn current(&self) -> &'a Token {
        &self.tokens[self.index.min(self.tokens.len().saturating_sub(1))]
    }

    fn advance(&mut self) -> &'a Token {
        let token = self.current();
        if !matches!(token.kind, TokenKind::Eof) {
            self.index += 1;
        }
        token
    }

    fn is_symbol(&self, symbol: Symbol) -> bool {
        matches!(&self.current().kind, TokenKind::Symbol(current) if *current == symbol)
    }

    fn take_symbol(&mut self, symbol: Symbol) -> bool {
        if self.is_symbol(symbol) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_program(mut self) -> Result<Program, Vec<Diagnostic>> {
        let mut declarations = Vec::new();
        while !matches!(self.current().kind, TokenKind::Eof) {
            match self.parse_block() {
                Some(block) => declarations.push(block),
                None => self.recover_top_level(),
            }
        }
        if self.errors.is_empty() {
            Ok(Program {
                source_name: self.source_name,
                declarations,
            })
        } else {
            Err(self.errors)
        }
    }

    fn parse_block(&mut self) -> Option<Block> {
        let start = self.current().span;
        let kind = self.expect_ident("P0001", "expected a declaration kind")?;
        let name = self.expect_ident("P0002", "expected a declaration name")?;
        if !self.take_symbol(Symbol::LBrace) {
            self.error(
                "P0003",
                "expected `{` after declaration name",
                self.current().span,
            );
            return None;
        }
        let mut fields = Vec::new();
        while !self.is_symbol(Symbol::RBrace) && !matches!(self.current().kind, TokenKind::Eof) {
            let field_start = self.current().span;
            let key = match self.expect_ident("P0004", "expected a field name") {
                Some(key) => key,
                None => {
                    self.recover_field();
                    continue;
                }
            };
            let value = match self.parse_value() {
                Some(value) => value,
                None => {
                    self.recover_field();
                    continue;
                }
            };
            if !self.take_symbol(Symbol::Semicolon) {
                self.error("P0005", "expected `;` after field", self.current().span);
                self.recover_field();
            }
            fields.push(Field {
                key,
                value,
                span: Span {
                    start: field_start.start,
                    end: self.current().span.end,
                    line: field_start.line,
                    column: field_start.column,
                },
            });
        }
        if !self.take_symbol(Symbol::RBrace) {
            self.error(
                "P0006",
                "unterminated declaration; expected `}`",
                self.current().span,
            );
        }
        Some(Block {
            kind,
            name,
            fields,
            span: Span {
                start: start.start,
                end: self.current().span.end,
                line: start.line,
                column: start.column,
            },
        })
    }

    fn parse_value(&mut self) -> Option<Value> {
        match &self.current().kind {
            TokenKind::String(value) => {
                let value = value.clone();
                self.advance();
                Some(Value::String(value))
            }
            TokenKind::Number(number) => {
                let number = number.clone();
                self.advance();
                if self.take_symbol(Symbol::DotDot) {
                    let end = self.expect_number("P0010", "expected range end")?;
                    Some(Value::Range { start: number, end })
                } else if let TokenKind::Ident(unit) = &self.current().kind {
                    let unit = unit.clone();
                    self.advance();
                    Some(Value::Quantity { number, unit })
                } else {
                    Some(Value::Number(number))
                }
            }
            TokenKind::Ident(value) => {
                let value = value.clone();
                self.advance();
                if self.take_symbol(Symbol::DotDot) {
                    let end = self.expect_number("P0011", "expected range end")?;
                    Some(Value::Range { start: value, end })
                } else {
                    Some(Value::Ident(value))
                }
            }
            TokenKind::Symbol(Symbol::LBracket) => self.parse_list(),
            _ => {
                self.error(
                    "P0007",
                    "expected a string, identifier, number, range, or list",
                    self.current().span,
                );
                None
            }
        }
    }

    fn parse_list(&mut self) -> Option<Value> {
        self.advance();
        let mut values = Vec::new();
        while !self.is_symbol(Symbol::RBracket) && !matches!(self.current().kind, TokenKind::Eof) {
            let value = match &self.current().kind {
                TokenKind::Number(number) => {
                    let value = Value::Number(number.clone());
                    self.advance();
                    value
                }
                TokenKind::Ident(value) => {
                    let value = Value::Ident(value.clone());
                    self.advance();
                    value
                }
                TokenKind::String(value) => {
                    let value = Value::String(value.clone());
                    self.advance();
                    value
                }
                _ => {
                    self.error(
                        "P0008",
                        "list elements must be scalar values",
                        self.current().span,
                    );
                    self.recover_list();
                    continue;
                }
            };
            values.push(value);
            if !self.take_symbol(Symbol::Comma) && !self.is_symbol(Symbol::RBracket) {
                self.error("P0009", "expected `,` or `]` in list", self.current().span);
                self.recover_list();
            }
        }
        if !self.take_symbol(Symbol::RBracket) {
            self.error(
                "P0012",
                "unterminated list; expected `]`",
                self.current().span,
            );
        }
        Some(Value::List(values))
    }

    fn expect_ident(&mut self, code: &str, message: &str) -> Option<String> {
        match &self.current().kind {
            TokenKind::Ident(value) => {
                let value = value.clone();
                self.advance();
                Some(value)
            }
            _ => {
                self.error(code, message, self.current().span);
                None
            }
        }
    }

    fn expect_number(&mut self, code: &str, message: &str) -> Option<String> {
        match &self.current().kind {
            TokenKind::Number(value) => {
                let value = value.clone();
                self.advance();
                Some(value)
            }
            _ => {
                self.error(code, message, self.current().span);
                None
            }
        }
    }

    fn error(&mut self, code: &str, message: &str, span: Span) {
        self.errors
            .push(Diagnostic::error(code, message, Some(span)));
    }

    fn recover_top_level(&mut self) {
        while !matches!(self.current().kind, TokenKind::Eof) && !self.is_symbol(Symbol::RBrace) {
            self.advance();
        }
        self.take_symbol(Symbol::RBrace);
    }

    fn recover_field(&mut self) {
        while !matches!(self.current().kind, TokenKind::Eof)
            && !self.is_symbol(Symbol::Semicolon)
            && !self.is_symbol(Symbol::RBrace)
        {
            self.advance();
        }
        self.take_symbol(Symbol::Semicolon);
    }

    fn recover_list(&mut self) {
        while !matches!(self.current().kind, TokenKind::Eof)
            && !self.is_symbol(Symbol::Comma)
            && !self.is_symbol(Symbol::RBracket)
        {
            self.advance();
        }
        self.take_symbol(Symbol::Comma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use har_lang_lexer::lex;

    #[test]
    fn parses_named_blocks() {
        let source = "target local { gpu \"RX\"; wave 32; }";
        let tokens = lex(source).expect("lex");
        let program = parse(&tokens, "test.har").expect("parse");
        assert_eq!(program.declarations[0].name, "local");
        assert_eq!(program.declarations[0].fields.len(), 2);
    }
}
