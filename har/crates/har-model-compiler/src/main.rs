use har_model_compiler::{
    compile_phenotype, CalibrationEvidenceSet, CalibrationPlan, GgufInspector, HardwarePolicy,
    ModelPhenotype,
};
use har_model_package::{PackageReader, SidecarReader};
use std::env;
use std::fs;

fn usage() -> ! {
    eprintln!(
        "har-modelc inspect <model.gguf> [--json-out PATH]\n\
         har-modelc compile <model.gguf|phenotype.json> --out PATH [--budget BYTES] [--captures PATH]\n\
         har-modelc verify <model.harpkg>\n\
         har-modelc verify-sidecar <model.harx> [--source-sha256 HASH]\n\
         har-modelc calibration-plan [--out PATH]"
    );
    std::process::exit(2)
}

fn write_json<T: serde::Serialize>(
    value: &T,
    path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = serde_json::to_string_pretty(value)? + "\n";
    if let Some(path) = path {
        fs::write(path, text)?;
    } else {
        print!("{text}");
    }
    Ok(())
}

fn load_phenotype(path: &str) -> Result<ModelPhenotype, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() >= 4 && &bytes[..4] == b"GGUF" {
        Ok(GgufInspector::inspect(path)?.phenotype)
    } else {
        Ok(serde_json::from_slice(&bytes)?)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| usage());
    match command.as_str() {
        "inspect" => {
            let input = args.next().unwrap_or_else(|| usage());
            let mut json_out = None;
            while let Some(arg) = args.next() {
                if arg == "--json-out" {
                    json_out = args.next();
                } else {
                    usage();
                }
            }
            let inspection = GgufInspector::inspect(&input)?;
            write_json(&inspection, json_out.as_deref())?;
        }
        "compile" => {
            let input = args.next().unwrap_or_else(|| usage());
            let mut output = None;
            let mut budget = None;
            let mut captures = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" | "-o" => output = args.next(),
                    "--budget" | "--budget-bytes" => {
                        budget = args.next().and_then(|value| value.parse::<u64>().ok())
                    }
                    "--captures" => captures = args.next(),
                    _ => usage(),
                }
            }
            let output = output.unwrap_or_else(|| {
                eprintln!("compile requires --out");
                usage()
            });
            let phenotype = load_phenotype(&input)?;
            let mut policy = HardwarePolicy::rdna4_default();
            policy.total_model_bytes_budget = budget;
            let evidence = match captures.as_deref() {
                Some(path) if path.ends_with(".csv") => CalibrationEvidenceSet::from_csv(path)?,
                Some(path) if path.ends_with(".jsonl") => CalibrationEvidenceSet::from_jsonl(path)?,
                Some(path) => CalibrationEvidenceSet::from_json(path)?,
                None => CalibrationEvidenceSet::empty(),
            };
            let compiled =
                compile_phenotype(phenotype, policy, evidence, "har-model-compiler/0.1")?;
            let manifest =
                har_model_package::PackageWriter::write(&output, &compiled.manifest, &[])?;
            eprintln!(
                "wrote {} tensors, {} bytes planned, package={}",
                manifest.tensors.len(),
                compiled.allocation.expected_total_bytes,
                output
            );
            write_json(
                &serde_json::json!({
                    "package": output,
                    "schema": manifest.schema,
                    "source_model_root_sha256": manifest.source_model_root_sha256,
                    "tensor_count": manifest.tensors.len(),
                    "expected_total_bytes": compiled.allocation.expected_total_bytes,
                    "routed_bytes_per_active_token": compiled.allocation.routed_bytes_per_active_token,
                    "required_kernels": manifest.required_kernels,
                    "calibration_status": compiled.allocation.calibration_status,
                    "quality_claim": manifest.claims.get("quality_claim"),
                    "unresolved_risks": compiled.allocation.unresolved_risks,
                }),
                None,
            )?;
        }
        "verify" => {
            let input = args.next().unwrap_or_else(|| usage());
            if args.next().is_some() {
                usage();
            }
            let verified = PackageReader::verify(&input)?;
            write_json(
                &serde_json::json!({
                    "valid": true,
                    "path": input,
                    "schema": verified.manifest.schema,
                    "source_model_root_sha256": verified.manifest.source_model_root_sha256,
                    "manifest_sha256": verified.manifest_sha256,
                    "package_root_sha256": verified.package_root_sha256,
                    "tensor_count": verified.manifest.tensors.len(),
                    "payload_count": verified.manifest.payload_locations.len(),
                    "quality_claim": verified.manifest.claims.get("quality_claim"),
                }),
                None,
            )?;
        }
        "verify-sidecar" => {
            let input = args.next().unwrap_or_else(|| usage());
            let mut source = None;
            while let Some(arg) = args.next() {
                if arg == "--source-sha256" {
                    source = args.next();
                } else {
                    usage();
                }
            }
            let verified = SidecarReader::verify(&input, source.as_deref())?;
            write_json(
                &serde_json::json!({
                    "valid": true,
                    "path": input,
                    "schema": verified.manifest.schema,
                    "index_sha256": verified.index_sha256,
                    "entry_count": verified.manifest.entries.len(),
                    "direct_index": verified.manifest.direct_index,
                    "runtime_tensor_scanning": verified.manifest.runtime_tensor_scanning,
                }),
                None,
            )?;
        }
        "calibration-plan" => {
            let mut output = None;
            while let Some(arg) = args.next() {
                if arg == "--out" || arg == "-o" {
                    output = args.next();
                } else {
                    usage();
                }
            }
            write_json(&CalibrationPlan::default(), output.as_deref())?;
        }
        _ => usage(),
    }
    Ok(())
}
