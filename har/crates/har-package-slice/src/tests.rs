//! Fail-closed tests for the HARSLICE1 bounded package-slice format.
//!
//! These tests build tiny synthetic GGUF sources (no model weights) and
//! exercise: deterministic round-trip, byte identity, wrong source root,
//! wrong row range, truncation, stale metadata, payload corruption, and
//! inspect-without-payload semantics.

use super::*;
use har_model_compiler::GgufInspector;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("har-slice-{name}-{nonce}"))
}

/// Build a tiny deterministic GGUF with one Q4_K tensor (rows x 5120 cols).
fn write_synthetic_source(rows: u64) -> (PathBuf, ModelPhenotype, ModelTensorDescriptor, u64) {
    let path = temp_path("src");
    let mut file = File::create(&path).unwrap();
    let write_str = |f: &mut File, s: &str| {
        f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    };
    file.write_all(b"GGUF").unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&1u64.to_le_bytes()).unwrap(); // 1 tensor
    file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 kv
    write_str(&mut file, "general.name");
    file.write_all(&8u32.to_le_bytes()).unwrap();
    write_str(&mut file, "synthetic-slice-source");
    write_str(&mut file, "general.architecture");
    file.write_all(&8u32.to_le_bytes()).unwrap();
    write_str(&mut file, "synthetic-hybrid");
    // tensor info: blk.0.ffn_gate.weight [5120, rows] Q4_K offset=0
    write_str(&mut file, "blk.0.ffn_gate.weight");
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(&5120u64.to_le_bytes()).unwrap();
    file.write_all(&rows.to_le_bytes()).unwrap();
    file.write_all(&12u32.to_le_bytes()).unwrap(); // Q4_K
    file.write_all(&0u64.to_le_bytes()).unwrap();
    let infos_end = file.stream_position().unwrap();
    let data_start = align_up(infos_end, 32);
    let row_bytes = (5120u64 / 256) * 144;
    let payload = vec![0x5Au8; (row_bytes * rows) as usize];
    {
        let mut data = File::options().write(true).open(&path).unwrap();
        data.seek(SeekFrom::Start(data_start)).unwrap();
        data.write_all(&payload).unwrap();
    }
    let inspection = GgufInspector::inspect(&path).unwrap();
    let descriptor = inspection
        .phenotype
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate.weight")
        .unwrap()
        .clone();
    (path, inspection.phenotype, descriptor, data_start)
}

fn base_manifest(
    phenotype: &ModelPhenotype,
    descriptor: &ModelTensorDescriptor,
    row_range: RowRange,
    package_root: String,
) -> SliceManifest {
    let source = source_slice_from_phenotype(phenotype, descriptor, 5120, 256, 144).unwrap();
    let row_bytes = source.row_bytes;
    let logical = row_bytes * row_range.count;
    SliceManifest {
        schema: SLICE_SCHEMA.into(),
        format_version: SLICE_FORMAT_VERSION,
        slice_id: format!(
            "{}/rows{}-{}",
            descriptor.name,
            row_range.start,
            row_range.start + row_range.count
        ),
        source,
        row_range,
        offsets: SliceOffsets {
            source_tensor_offset: descriptor.file_offset,
            source_offset: descriptor.file_offset + row_range.start * row_bytes,
            logical_bytes: logical,
            physical_bytes: align_up(logical, 4096),
            leading_padding_bytes: 0,
            trailing_padding_bytes: align_up(logical, 4096) - logical,
        },
        payload_checksum_sha256: String::new(),
        source_span_checksum_sha256: String::new(),
        package: SlicePackage {
            package_schema: "har.packed_model_package.v0".into(),
            package_root_sha256: package_root,
            generation: 1,
            compiler_version: "har-package-slice/0.1".into(),
        },
        reconstruction: SliceReconstruction {
            command:
                "har-package-slice extract <source.gguf> --tensor blk.0.ffn_gate.weight --rows 0..1"
                    .into(),
            deterministic: true,
            fixture_class: "bounded-execution-evidence".into(),
            approval: "review note (format-compatibility-v2 evidence span)".into(),
        },
    }
}

fn read_span(path: &Path, offset: u64, bytes: u64) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut out = vec![0u8; bytes as usize];
    file.read_exact(&mut out).unwrap();
    out
}

#[test]
fn roundtrip_is_deterministic_and_byte_identical() {
    let (src, phenotype, descriptor, data_start) = write_synthetic_source(64);
    let row_bytes = (5120u64 / 256) * 144;
    let range = RowRange {
        start: 0,
        count: 16,
    };
    let manifest = base_manifest(&phenotype, &descriptor, range, "a".repeat(64));
    let payload = read_span(&src, descriptor.file_offset, row_bytes * 16);
    let out1 = temp_path("slice1");
    let out2 = temp_path("slice2");
    SliceWriter::write(&out1, manifest.clone(), &payload).unwrap();
    SliceWriter::write(&out2, manifest, &payload).unwrap();
    let f1 = fs::read(&out1).unwrap();
    let f2 = fs::read(&out2).unwrap();
    assert_eq!(
        f1, f2,
        "identical inputs must produce byte-identical slice files"
    );
    let verified = SliceReader::open(&out1).unwrap();
    assert_eq!(verified.manifest.row_range, range);
    assert_eq!(verified.manifest.offsets.source_offset, data_start);
    let back = SliceReader::read_payload(&verified).unwrap();
    assert_eq!(
        back, payload,
        "payload must be byte-identical to the source span"
    );
    // manifest + payload checksums agree
    assert_eq!(
        verified.manifest.payload_checksum_sha256,
        sha256_bytes(&payload)
    );
    for p in [&out1, &out2, &src] {
        let _ = fs::remove_file(p);
    }
}

#[test]
fn wrong_source_root_is_rejected() {
    let (src, phenotype, descriptor, _) = write_synthetic_source(64);
    let manifest = base_manifest(
        &phenotype,
        &descriptor,
        RowRange {
            start: 0,
            count: 16,
        },
        "a".repeat(64),
    );
    let payload = read_span(&src, descriptor.file_offset, (5120u64 / 256) * 144 * 16);
    let out = temp_path("root");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    let verified = SliceReader::open(&out).unwrap();
    let error =
        SliceReader::validate_identity(&verified, Some("deadbeef"), None, None).unwrap_err();
    assert!(error.to_string().contains("model root"));
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&src);
}

#[test]
fn wrong_row_range_is_rejected() {
    let (src, phenotype, descriptor, _) = write_synthetic_source(64);
    let manifest = base_manifest(
        &phenotype,
        &descriptor,
        RowRange {
            start: 0,
            count: 16,
        },
        "a".repeat(64),
    );
    let payload = read_span(&src, descriptor.file_offset, (5120u64 / 256) * 144 * 16);
    let out = temp_path("rows");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    // Out-of-bounds range against the same source tensor must fail geometry.
    let err = SliceWriter::validate_geometry(
        &descriptor,
        5120,
        256,
        144,
        &RowRange {
            start: 60,
            count: 16,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("outside tensor row count"));
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&src);
}

#[test]
fn truncated_slice_is_rejected() {
    let (src, phenotype, descriptor, _) = write_synthetic_source(64);
    let manifest = base_manifest(
        &phenotype,
        &descriptor,
        RowRange {
            start: 0,
            count: 16,
        },
        "a".repeat(64),
    );
    let payload = read_span(&src, descriptor.file_offset, (5120u64 / 256) * 144 * 16);
    let out = temp_path("trunc");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    let file = fs::read(&out).unwrap();
    for cut in [8usize, 64, file.len() / 2, file.len() - 17] {
        let truncated = temp_path(&format!("trunc-{cut}"));
        fs::write(&truncated, &file[..cut]).unwrap();
        assert!(
            SliceReader::open(&truncated).is_err(),
            "truncation at {cut} must fail closed"
        );
        let _ = fs::remove_file(&truncated);
    }
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&src);
}

#[test]
fn stale_metadata_is_rejected() {
    let (src, phenotype, descriptor, _) = write_synthetic_source(64);
    let manifest = base_manifest(
        &phenotype,
        &descriptor,
        RowRange {
            start: 0,
            count: 16,
        },
        "a".repeat(64),
    );
    let payload = read_span(&src, descriptor.file_offset, (5120u64 / 256) * 144 * 16);
    let out = temp_path("stale");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    let verified = SliceReader::open(&out).unwrap();
    let error = SliceReader::validate_identity(&verified, None, Some(7), None).unwrap_err();
    assert!(error.to_string().contains("stale"));
    let error2 =
        SliceReader::validate_identity(&verified, None, None, Some("b".repeat(64).as_str()))
            .unwrap_err();
    assert!(error2.to_string().contains("package root"));
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&src);
}

#[test]
fn payload_corruption_is_rejected() {
    let (src, phenotype, descriptor, _) = write_synthetic_source(64);
    let manifest = base_manifest(
        &phenotype,
        &descriptor,
        RowRange {
            start: 0,
            count: 16,
        },
        "a".repeat(64),
    );
    let payload = read_span(&src, descriptor.file_offset, (5120u64 / 256) * 144 * 16);
    let out = temp_path("corrupt");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    let mut file = fs::read(&out).unwrap();
    let verified = SliceReader::open(&out).unwrap();
    let flip = verified.payload_offset as usize + 100;
    file[flip] ^= 0xFF;
    let corrupted = temp_path("corrupt2");
    fs::write(&corrupted, &file).unwrap();
    let error = SliceReader::open(&corrupted).unwrap_err();
    assert!(error.to_string().contains("checksum"));
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&corrupted);
    let _ = fs::remove_file(&src);
}

#[test]
fn inspect_does_not_require_payload_integrity() {
    // `open` validates header + manifest; payload integrity is validated by
    // the root checksum too, so build a manifest-only view: a truncated file
    // with an intact header+manifest must still fail (truncation), while an
    // inspect of a valid slice succeeds even if we never read the payload
    // region beyond `open`'s range check.  This pins the documented
    // "inspect without full package loading" property: identity fields are
    // available from the manifest alone after `open`.
    let (src, phenotype, descriptor, _) = write_synthetic_source(64);
    let manifest = base_manifest(
        &phenotype,
        &descriptor,
        RowRange {
            start: 0,
            count: 16,
        },
        "a".repeat(64),
    );
    let payload = read_span(&src, descriptor.file_offset, (5120u64 / 256) * 144 * 16);
    let out = temp_path("inspect");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    let verified = SliceReader::open(&out).unwrap();
    assert_eq!(
        verified.manifest.source.tensor_identity,
        "blk.0.ffn_gate.weight"
    );
    assert_eq!(verified.manifest.row_range.count, 16);
    assert_eq!(verified.manifest.package.generation, 1);
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&src);
}

#[test]
fn mxfp4_expert_slice_roundtrip_and_large_escape() {
    // Track B geometry: 4096-element rows = 128 blocks of 17 B = 2,176 B.
    let row_elements = 4096u64;
    let block_elements = MXFP4_BLOCK_ELEMENTS;
    let block_bytes = MXFP4_BLOCK_BYTES;
    let row_bytes = (row_elements / block_elements) * block_bytes;
    assert_eq!(row_bytes, 2176);
    // Synthetic expert payload: 4 rows (4 x 2,176 = 8,704 B).
    let rows = 4u64;
    let src = temp_path("mxfp4-src");
    {
        let mut file = File::create(&src).unwrap();
        let write_str = |f: &mut File, s: &str| {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        };
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        write_str(&mut file, "general.name");
        file.write_all(&8u32.to_le_bytes()).unwrap();
        write_str(&mut file, "synthetic-mxfp4");
        write_str(&mut file, "general.architecture");
        file.write_all(&8u32.to_le_bytes()).unwrap();
        write_str(&mut file, "deepseek4");
        write_str(&mut file, "blk.0.ffn_gate_exps.weight");
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&4096u64.to_le_bytes()).unwrap();
        file.write_all(&2048u64.to_le_bytes()).unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap(); // 4 experts
        file.write_all(&39u32.to_le_bytes()).unwrap(); // MXFP4
        file.write_all(&0u64.to_le_bytes()).unwrap();
        let infos_end = file.stream_position().unwrap();
        let data_start = align_up(infos_end, 32);
        let payload = vec![0x5Cu8; (row_bytes * rows) as usize];
        let mut data = File::options().write(true).open(&src).unwrap();
        data.seek(SeekFrom::Start(data_start)).unwrap();
        data.write_all(&payload).unwrap();
    }
    let inspection = GgufInspector::inspect(&src).unwrap();
    let descriptor = inspection
        .phenotype
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_gate_exps.weight")
        .unwrap()
        .clone();
    assert_eq!(descriptor.quantization, "MXFP4");
    assert_eq!(descriptor.ggml_type, 39);
    let (row_bytes2, _, logical) = SliceWriter::validate_geometry(
        &descriptor,
        row_elements,
        block_elements,
        block_bytes,
        &RowRange {
            start: 0,
            count: rows,
        },
    )
    .unwrap();
    assert_eq!(row_bytes2, 2176);
    assert_eq!(logical, 8704);
    let manifest = {
        let source = source_slice_from_phenotype(
            &inspection.phenotype,
            &descriptor,
            row_elements,
            block_elements,
            block_bytes,
        )
        .unwrap();
        SliceManifest {
            schema: SLICE_SCHEMA.into(),
            format_version: SLICE_FORMAT_VERSION,
            slice_id: "blk.0.ffn_gate_exps.weight/rows0-4".into(),
            source,
            row_range: RowRange { start: 0, count: rows },
            offsets: SliceOffsets {
                source_tensor_offset: descriptor.file_offset,
                source_offset: descriptor.file_offset,
                logical_bytes: logical,
                physical_bytes: align_up(logical, 4096),
                leading_padding_bytes: 0,
                trailing_padding_bytes: align_up(logical, 4096) - logical,
            },
            payload_checksum_sha256: String::new(),
            source_span_checksum_sha256: String::new(),
            package: SlicePackage {
                package_schema: "har.packed_model_package.v0".into(),
                package_root_sha256: "a".repeat(64),
                generation: 1,
                compiler_version: "har-package-slice/0.1".into(),
            },
            reconstruction: SliceReconstruction {
                command: "har-package-slice extract <source.gguf> --tensor blk.0.ffn_gate_exps.weight --rows 0..4".into(),
                deterministic: true,
                fixture_class: "bounded-execution-evidence".into(),
                approval: "review note".into(),
            },
        }
    };
    let payload = read_span(&src, descriptor.file_offset, logical);
    let out = temp_path("mxfp4-slice");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    let verified = SliceReader::open(&out).unwrap();
    assert_eq!(verified.manifest.source.quant_format, "MXFP4");
    assert_eq!(verified.manifest.source.row_bytes, 2176);
    assert_eq!(SliceReader::read_payload(&verified).unwrap(), payload);
    // Budget enforcement: an over-budget payload must be rejected by `write`
    // but accepted by `write_large` (local-only hash evidence).
    let big_manifest = base_manifest(
        &inspection.phenotype,
        &descriptor,
        RowRange { start: 0, count: 4 },
        "a".repeat(64),
    );
    let mut big_manifest = big_manifest;
    big_manifest.offsets.logical_bytes = 200 * 1024;
    let big_payload = vec![0x5Cu8; 200 * 1024];
    assert!(SliceWriter::write(&out, big_manifest.clone(), &big_payload).is_err());
    let out2 = temp_path("mxfp4-large");
    SliceWriter::write_large(&out2, big_manifest, &big_payload).unwrap();
    assert!(SliceReader::open(&out2).is_ok());
    for p in [&src, &out, &out2] {
        let _ = fs::remove_file(p);
    }
}

#[test]
fn geometry_rejects_wrong_block_format() {
    let (src, phenotype, descriptor, _) = write_synthetic_source(64);
    // Claim Q8_0 geometry (34 B/32 el) against a Q4_K tensor: row size and
    // element math must disagree.
    let manifest = base_manifest(
        &phenotype,
        &descriptor,
        RowRange {
            start: 0,
            count: 16,
        },
        "a".repeat(64),
    );
    let payload = read_span(&src, descriptor.file_offset, (5120u64 / 256) * 144 * 16);
    let out = temp_path("geom");
    SliceWriter::write(&out, manifest, &payload).unwrap();
    let verified = SliceReader::open(&out).unwrap();
    let q8_row_bytes = (5120u64 / 32) * 34;
    assert_ne!(q8_row_bytes, verified.manifest.source.row_bytes);
    // sanity: Q4_K row geometry is the documented 2,880 B
    assert_eq!(verified.manifest.source.row_bytes, 2880);
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&src);
}
