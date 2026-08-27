use har_lang_compiler::compile;
use std::env;
use std::fs;

fn main() -> std::process::ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(value) => value,
        None => {
            eprintln!("usage: har-compile SOURCE.har [OUTPUT.json]");
            return std::process::ExitCode::from(2);
        }
    };
    let source = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("har-compile: {error}");
            return std::process::ExitCode::from(2);
        }
    };
    match compile(&path, &source) {
        Ok(program) => {
            let bytes = match har_core::canonical_json(&program) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("har-compile: {error}");
                    return std::process::ExitCode::from(1);
                }
            };
            if let Some(output) = args.next() {
                if let Err(error) = fs::write(output, bytes) {
                    eprintln!("har-compile: {error}");
                    return std::process::ExitCode::from(1);
                }
            } else {
                println!("{}", String::from_utf8_lossy(&bytes));
            }
            std::process::ExitCode::SUCCESS
        }
        Err(errors) => {
            for error in errors {
                eprintln!("{error}");
            }
            std::process::ExitCode::from(1)
        }
    }
}
