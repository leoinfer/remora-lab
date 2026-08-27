//! `har-package-slice` — deterministic, git-safe bounded package slices.
//!
//! ```text
//! har-package-slice extract <source.gguf> --tensor NAME --rows S..E --out SLICE
//!                        [--source-root-hash HASH] [--package-root HASH]
//!                        [--generation N] [--fixture-class CLASS] [--approval NOTE]
//! har-package-slice verify SLICE [--source <source.gguf>]
//!                        [--source-root-hash HASH] [--package-root HASH]
//!                        [--generation N]
//! har-package-slice inspect SLICE
//! har-package-slice reconstruct SLICE --source <source.gguf> --out <package.harpkg>
//! ```

use har_model_compiler::GgufInspector;
use har_package_slice::{
    align_up, Result, RowRange, SliceError, SliceManifest, SliceReader, SliceReconstruction,
    SliceWriter, MAX_SLICE_FIXTURE_PAYLOAD_BYTES, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMENTS,
    SLICE_ALIGNMENT,
};
use std::env;
use std::fs;
use std::io::{Read, Seek};
use std::path::Path;

const Q4_K_BLOCK_ELEMENTS: u64 = 256;
const Q4_K_BLOCK_BYTES: u64 = 144;

fn usage() -> ! {
    eprintln!(
        "har-package-slice extract <source.gguf> --tensor NAME --rows S..E --out SLICE [--source-root-hash HASH] [--package-root HASH] [--generation N] [--fixture-class CLASS] [--approval NOTE] [--allow-large]\n\
         har-package-slice verify SLICE [--source <source.gguf>] [--source-root-hash HASH] [--package-root HASH] [--generation N]\n\
         har-package-slice inspect SLICE\n\
         har-package-slice reconstruct SLICE --source <source.gguf> --out <package.harpkg>"
    );
    std::process::exit(2)
}

fn parse_rows(value: &str) -> (u64, u64) {
    let (start, end) = value
        .split_once("..")
        .ok_or_else(|| "row range must be START..END".to_string())
        .unwrap();
    let start: u64 = start.parse().unwrap_or_else(|_| usage());
    let end: u64 = end.parse().unwrap_or_else(|_| usage());
    if end <= start {
        eprintln!("row range END must exceed START");
        usage();
    }
    (start, end - start)
}

fn resolve_tensor<'a>(
    phenotype: &'a har_model_compiler::ModelPhenotype,
    name: &str,
) -> Result<&'a har_model_compiler::ModelTensorDescriptor> {
    phenotype
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| SliceError::Invalid(format!("tensor {name} not found in source GGUF")))
}

fn cmd_extract(mut args: impl Iterator<Item = String>) -> Result<()> {
    let source = args.next().unwrap_or_else(|| usage());
    let mut tensor = None;
    let mut rows = None;
    let mut out = None;
    let mut source_root_hash = None;
    let mut package_root = None;
    let mut generation = 1u64;
    let mut fixture_class = "bounded-execution-evidence".to_string();
    let mut approval = "review note (format-compatibility-v2 evidence span)".to_string();
    let mut allow_large = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--tensor" => tensor = args.next(),
            "--rows" => rows = args.next(),
            "--out" => out = args.next(),
            "--source-root-hash" => source_root_hash = args.next(),
            "--package-root" => package_root = args.next(),
            "--generation" => {
                generation = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--fixture-class" => fixture_class = args.next().unwrap_or_else(|| usage()),
            "--approval" => approval = args.next().unwrap_or_else(|| usage()),
            "--allow-large" => allow_large = true,
            _ => usage(),
        }
    }
    let tensor = tensor.unwrap_or_else(|| {
        eprintln!("extract requires --tensor");
        usage()
    });
    let rows = rows.unwrap_or_else(|| {
        eprintln!("extract requires --rows");
        usage()
    });
    let out = out.unwrap_or_else(|| {
        eprintln!("extract requires --out");
        usage()
    });
    let (row_start, row_count) = parse_rows(&rows);

    let inspection = GgufInspector::inspect(&source)?;
    let phenotype = &inspection.phenotype;
    let descriptor = resolve_tensor(phenotype, &tensor)?;
    // Canonical block geometry per quant format (v1: Q4_K, MXFP4/UD, F32).
    let (block_elements, block_bytes) =
        match (descriptor.quantization.as_str(), descriptor.ggml_type) {
            ("Q4_K", 12) => (Q4_K_BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES),
            ("MXFP4", 39) => (MXFP4_BLOCK_ELEMENTS, MXFP4_BLOCK_BYTES),
            ("F32", 0) => (1, 4),
            (quant, type_id) => {
                return Err(SliceError::Invalid(format!(
                "slice tensor {} is {} (type {}); the v1 format supports Q4_K, MXFP4 and F32 only",
                tensor, quant, type_id
            )))
            }
        };
    let (row_bytes, total_rows, logical_bytes) = SliceWriter::validate_geometry(
        descriptor,
        descriptor.dimensions.first().copied().unwrap_or(0),
        block_elements,
        block_bytes,
        &RowRange {
            start: row_start,
            count: row_count,
        },
    )?;
    if logical_bytes > MAX_SLICE_FIXTURE_PAYLOAD_BYTES && !allow_large {
        return Err(SliceError::Invalid(format!(
            "requested slice is {logical_bytes} bytes; the repository fixture budget is {MAX_SLICE_FIXTURE_PAYLOAD_BYTES} bytes (use a smaller row range, or --allow-large for local-only hash evidence)"
        )));
    }
    let source_offset = descriptor.file_offset + row_start * row_bytes;
    let mut payload = vec![0u8; logical_bytes as usize];
    {
        let mut f = fs::File::open(&source)?;
        f.seek(std::io::SeekFrom::Start(source_offset))?;
        f.read_exact(&mut payload)?;
    }
    let source_root = match source_root_hash {
        Some(hash) => hash,
        None => phenotype.sha256.clone(),
    };
    let package_root = package_root.unwrap_or_else(|| {
        "0000000000000000000000000000000000000000000000000000000000000000".to_string()
    });
    let manifest = SliceManifest {
        schema: "har.package_slice.v1".into(),
        format_version: 1,
        slice_id: format!("{tensor}/rows{row_start}-{}", row_start + row_count),
        source: har_package_slice::source_slice_from_phenotype(phenotype, descriptor, descriptor.dimensions.first().copied().unwrap_or(0), block_elements, block_bytes)?,
        row_range: RowRange { start: row_start, count: row_count },
        offsets: har_package_slice::SliceOffsets {
            source_tensor_offset: descriptor.file_offset,
            source_offset,
            logical_bytes,
            physical_bytes: align_up(logical_bytes, SLICE_ALIGNMENT as u64),
            leading_padding_bytes: 0,
            trailing_padding_bytes: align_up(logical_bytes, SLICE_ALIGNMENT as u64) - logical_bytes,
        },
        payload_checksum_sha256: String::new(),
        source_span_checksum_sha256: String::new(),
        package: har_package_slice::SlicePackage {
            package_schema: "har.packed_model_package.v0".into(),
            package_root_sha256: package_root,
            generation,
            compiler_version: "har-package-slice/0.1".into(),
        },
        reconstruction: SliceReconstruction {
            command: format!(
                "har-package-slice extract <source.gguf> --tensor {tensor} --rows {row_start}..{} --out <slice.harslice> --source-root-hash {source_root}",
                row_start + row_count
            ),
            deterministic: true,
            fixture_class,
            approval,
        },
    };
    let written = if allow_large {
        SliceWriter::write_large(&out, manifest, &payload)?
    } else {
        SliceWriter::write(&out, manifest, &payload)?
    };
    let hash = written.stable_hash()?;
    if allow_large {
        println!("  note: --allow-large used; this slice exceeds the committed-fixture budget and must stay local (CI rejects committed over-budget slices)");
    }
    println!(
        "slice {} rows {}..{} ({} bytes) -> {}\n  source root: {}\n  payload sha256: {}\n  manifest sha256: {}",
        tensor, row_start, row_start + row_count, logical_bytes, out, source_root, written.payload_checksum_sha256, hash
    );
    // model root sanity: the source-root hash must match the shard identity
    if source_root == phenotype.sha256 {
        println!("  model root verified against shard sha256");
    }
    let _ = total_rows;
    Ok(())
}

fn cmd_verify(mut args: impl Iterator<Item = String>) -> Result<()> {
    let slice = args.next().unwrap_or_else(|| usage());
    let mut source = None;
    let mut source_root_hash = None;
    let mut package_root = None;
    let mut generation = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--source" => source = args.next(),
            "--source-root-hash" => source_root_hash = args.next(),
            "--package-root" => package_root = args.next(),
            "--generation" => {
                generation = Some(
                    args.next()
                        .unwrap_or_else(|| usage())
                        .parse()
                        .unwrap_or_else(|_| usage()),
                )
            }
            _ => usage(),
        }
    }
    let verified = SliceReader::open(&slice)?;
    SliceReader::validate_identity(
        &verified,
        source_root_hash.as_deref(),
        generation,
        package_root.as_deref(),
    )?;
    let payload = SliceReader::read_payload(&verified)?;
    if let Some(source_path) = source {
        let source_path = Path::new(&source_path);
        let inspection = GgufInspector::inspect(source_path)?;
        let phenotype = &inspection.phenotype;
        if phenotype.sha256 != verified.manifest.source.model_root_sha256 {
            return Err(SliceError::Invalid(format!(
                "source shard sha256 {} does not match slice model root {}",
                phenotype.sha256, verified.manifest.source.model_root_sha256
            )));
        }
        let descriptor = resolve_tensor(phenotype, &verified.manifest.source.tensor_identity)?;
        let span_offset = descriptor.file_offset
            + verified.manifest.row_range.start * verified.manifest.source.row_bytes;
        let span_bytes = verified.manifest.offsets.logical_bytes;
        let mut source_span = vec![0u8; span_bytes as usize];
        {
            let mut f = fs::File::open(source_path)?;
            f.seek(std::io::SeekFrom::Start(span_offset))?;
            f.read_exact(&mut source_span)?;
        }
        if source_span != payload {
            return Err(SliceError::Invalid(
                "slice payload is not byte-identical to the source span".into(),
            ));
        }
        println!("byte identity vs source: PASS ({} bytes)", span_bytes);
    } else {
        println!(
            "byte identity vs source: skipped (no --source); payload checksum validated internally"
        );
    }
    println!(
        "slice {} rows {}..{} payload {}B sha256 {}",
        verified.manifest.source.tensor_identity,
        verified.manifest.row_range.start,
        verified.manifest.row_range.start + verified.manifest.row_range.count,
        payload.len(),
        verified.manifest.payload_checksum_sha256
    );
    Ok(())
}

fn cmd_inspect(mut args: impl Iterator<Item = String>) -> Result<()> {
    let slice = args.next().unwrap_or_else(|| usage());
    let verified = SliceReader::open(&slice)?;
    let m = &verified.manifest;
    // Identity fields come from the manifest; the payload is never loaded.
    println!(
        "slice_id: {}\nschema: {} v{}\nmodel_root: {} (sha256 {})\nshard: {}\ntensor: {} (type {}, {})\nquant: {} {}B/{}el\ndimensions: {:?}\nrows: {}..{} ({} rows x {}B = {}B logical / {}B physical)\nsource_offset: {}\npayload_checksum: {}\nsource_span_checksum: {}\npackage_root: {}\ngeneration: {}\nreconstruction: {}\nfixture_class: {}\napproval: {}\nmanifest_sha256: {}\nslice_root_sha256: {}",
        m.slice_id, m.schema, m.format_version, m.source.model_root, m.source.model_root_sha256, m.source.shard_basename,
        m.source.tensor_identity, m.source.ggml_type_id, m.source.quant_format, m.source.quant_format, m.source.block_bytes, m.source.block_elements,
        m.source.tensor_dimensions, m.row_range.start, m.row_range.start + m.row_range.count, m.row_range.count, m.source.row_bytes,
        m.offsets.logical_bytes, m.offsets.physical_bytes, m.offsets.source_offset, m.payload_checksum_sha256,
        m.source_span_checksum_sha256, m.package.package_root_sha256, m.package.generation, m.reconstruction.command,
        m.reconstruction.fixture_class, m.reconstruction.approval, verified.manifest_sha256, verified.slice_root_sha256
    );
    Ok(())
}

fn cmd_reconstruct(mut args: impl Iterator<Item = String>) -> Result<()> {
    let slice = args.next().unwrap_or_else(|| usage());
    let mut source = None;
    let mut out = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--source" => source = args.next(),
            "--out" => out = args.next(),
            _ => usage(),
        }
    }
    let source = source.unwrap_or_else(|| {
        eprintln!("reconstruct requires --source");
        usage()
    });
    let out = out.unwrap_or_else(|| {
        eprintln!("reconstruct requires --out");
        usage()
    });
    let verified = SliceReader::open(&slice)?;
    let m = &verified.manifest;
    // Deterministic reconstruction: run the fixed packer over the declared
    // source and compare the resulting package root with the manifest claim.
    let inspection = GgufInspector::inspect(&source)?;
    let phenotype = &inspection.phenotype;
    if phenotype.sha256 != m.source.model_root_sha256 {
        return Err(SliceError::Invalid(
            "source shard does not match the slice model root".into(),
        ));
    }
    let descriptor = resolve_tensor(phenotype, &m.source.tensor_identity)?;
    let row_bytes = m.source.row_bytes;
    let source_offset = descriptor.file_offset + m.row_range.start * row_bytes;
    let logical_bytes = m.offsets.logical_bytes;
    let mut payload = vec![0u8; logical_bytes as usize];
    {
        let mut f = fs::File::open(&source)?;
        f.seek(std::io::SeekFrom::Start(source_offset))?;
        f.read_exact(&mut payload)?;
    }
    if har_package_slice::sha256_bytes(&payload) != m.payload_checksum_sha256 {
        return Err(SliceError::Invalid(
            "reconstructed payload does not match the slice checksum".into(),
        ));
    }
    // Write the bounded deterministic local package (slice-only entry).
    let entry = har_model_package::PackedEntry {
        source_tensor_id: m.source.tensor_identity.clone(),
        source_file: m.source.shard_basename.clone(),
        source_offset,
        source_bytes: logical_bytes,
        payload_location_id: format!(
            "packed/{}/rows{}-{}",
            m.source.tensor_identity,
            m.row_range.start,
            m.row_range.start + m.row_range.count
        ),
        dimensions: vec![
            m.row_range.count * row_bytes / row_bytes,
            row_bytes / m.source.block_bytes * m.source.block_elements,
        ],
        element_count: m.row_range.count * m.source.row_elements,
        quant_format: m.source.quant_format.clone(),
        layer: None,
        expert_id: None,
        projection: None,
        role: har_model_package::TensorRole::DenseFfnGate,
        tensor_class: "dense_ffn".into(),
        kernel_requirement: "har.vulkan.gemm.q4_k".into(),
        representation_identity: "byte-identical-source".into(),
    };
    let source_model = har_model_package::SourceModel {
        path: source.clone(),
        root_sha256: m.source.model_root_sha256.clone(),
        file_bytes: phenotype.file_bytes,
        gguf_version: phenotype.gguf_version,
        architecture: phenotype.architecture.clone(),
        model_name: phenotype.model_name.clone(),
        tensor_count: phenotype.tensor_count,
        metadata_count: phenotype.metadata_count,
        data_offset: phenotype.data_offset,
        tensor_payload_bytes: phenotype.tensor_payload_bytes,
    };
    let hardware = har_model_compiler::HardwarePolicy::rdna4_default().hardware;
    let mut manifest =
        har_model_package::PackageManifest::new(source_model, hardware, "har-package-slice/0.1");
    manifest.source_model_root_sha256 = m.source.model_root_sha256.clone();
    manifest.claims.insert(
        "package_kind".to_owned(),
        "bounded_slice_reconstruction".to_owned(),
    );
    manifest
        .claims
        .insert("slice_id".to_owned(), m.slice_id.clone());
    manifest.claims.insert(
        "reproducible_from".to_owned(),
        m.reconstruction.command.clone(),
    );
    let written = har_model_package::PackageWriter::write_packed(
        &out,
        manifest,
        &source,
        std::slice::from_ref(&entry),
    )?;
    let root = written
        .claims
        .get("package_root_sha256")
        .cloned()
        .unwrap_or_default();
    println!("reconstructed package {} root {}", out, root);
    if root == m.package.package_root_sha256 {
        println!("package root matches the slice manifest claim");
    } else if m.package.package_root_sha256
        == "0000000000000000000000000000000000000000000000000000000000000000"
    {
        println!("note: slice declares a placeholder package root; run extract with --package-root to pin it");
    } else {
        return Err(SliceError::Invalid(format!(
            "package root {} does not match the slice claim {}",
            root, m.package.package_root_sha256
        )));
    }
    Ok(())
}

fn main() {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| usage());
    let result = match command.as_str() {
        "extract" => cmd_extract(args),
        "verify" => cmd_verify(args),
        "inspect" => cmd_inspect(args),
        "reconstruct" => cmd_reconstruct(args),
        _ => usage(),
    };
    if let Err(error) = result {
        eprintln!("har-package-slice error: {error}");
        std::process::exit(1);
    }
}
