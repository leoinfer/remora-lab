//! Stable formatter for reviewable HAR source.  It is not used by runtime execution.

use har_lang_ast::{Block, Program, Value};

pub fn format_program(program: &Program) -> String {
    let mut output = String::new();
    for block in &program.declarations {
        format_block(block, &mut output);
        output.push('\n');
    }
    output
}

fn format_block(block: &Block, output: &mut String) {
    output.push_str(&format!("{} {} {{\n", block.kind, block.name));
    for field in &block.fields {
        output.push_str("    ");
        output.push_str(&field.key);
        output.push(' ');
        format_value(&field.value, output);
        output.push_str(";\n");
    }
    output.push_str("}\n");
}

fn format_value(value: &Value, output: &mut String) {
    match value {
        Value::Ident(value) | Value::Number(value) => output.push_str(value),
        Value::String(value) => {
            output.push('"');
            for character in value.chars() {
                match character {
                    '"' => output.push_str("\\\""),
                    '\n' => output.push_str("\\n"),
                    '\\' => output.push_str("\\\\"),
                    other => output.push(other),
                }
            }
            output.push('"');
        }
        Value::Quantity { number, unit } => output.push_str(&format!("{number} {unit}")),
        Value::Range { start, end } => output.push_str(&format!("{start}..{end}")),
        Value::List(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                format_value(value, output);
            }
            output.push(']');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use har_lang_lexer::lex;
    use har_lang_parser::parse;

    #[test]
    fn formatting_is_parseable() {
        let source = "target local { gpu \"RX\"; horizon 0..3; }";
        let program = parse(&lex(source).unwrap(), "test.har").unwrap();
        let formatted = format_program(&program);
        assert!(formatted.contains("target local"));
        assert!(parse(&lex(&formatted).unwrap(), "formatted.har").is_ok());
    }
}
