use std::fmt;

/// Stable, deterministic digest used for logical identity.
///
/// The current no-dependency implementation is an explicit four-lane FNV-1a
/// digest.  It is an identity namespace, not a cryptographic authorization
/// primitive.  A deployment requiring cryptographic collision resistance must
/// replace this implementation behind the same type and record the algorithm
/// in the runtime manifest.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StableDigest(String);

impl StableDigest {
    pub fn from_parts(parts: &[&[u8]]) -> Self {
        let mut bytes = Vec::new();
        for part in parts {
            bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
            bytes.extend_from_slice(part);
        }
        let mut words = [
            0xcbf29ce484222325u64,
            0x84222325cbf29ce4u64,
            0x9e3779b185ebca87u64,
            0x517cc1b727220a95u64,
        ];
        let primes = [
            0x100000001b3u64,
            0x100000001b3u64 ^ 0x9e3779b97f4a7c15,
            0x100000001b3u64 ^ 0xd6e8feb86659fd93,
            0x100000001b3u64 ^ 0xa0761d6478bd642f,
        ];
        for (index, byte) in bytes.iter().enumerate() {
            for lane in 0..4 {
                let mixed = (*byte as u64)
                    .wrapping_add((index as u64).rotate_left((lane * 7) as u32))
                    .wrapping_add((lane as u64) * 0x9e37);
                words[lane] ^= mixed;
                words[lane] = words[lane].wrapping_mul(primes[lane]);
                words[lane] ^= words[lane] >> 29;
            }
        }
        let mut text = String::with_capacity(64);
        for word in words {
            text.push_str(&format!("{word:016x}"));
        }
        Self(text)
    }

    pub fn from_text(text: &str) -> Self {
        Self::from_parts(&[text.as_bytes()])
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StableDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// The transitive causal closure for exact reuse and prefix/page identity.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct KVDependencyClosure {
    pub model_root: StableDigest,
    pub tokenizer_root: StableDigest,
    pub token_sequence_root: StableDigest,
    pub rope_config_root: StableDigest,
    pub layer_head_root: StableDigest,
    pub kv_type: String,
    pub codec_version: String,
    pub runtime_generation: u64,
    pub graph_generation: u64,
    pub decode_epoch: u64,
    pub authority_state_root: StableDigest,
}

impl KVDependencyClosure {
    pub fn root(&self) -> StableDigest {
        let generation = self.runtime_generation.to_le_bytes();
        let graph = self.graph_generation.to_le_bytes();
        let epoch = self.decode_epoch.to_le_bytes();
        StableDigest::from_parts(&[
            self.model_root.as_str().as_bytes(),
            self.tokenizer_root.as_str().as_bytes(),
            self.token_sequence_root.as_str().as_bytes(),
            self.rope_config_root.as_str().as_bytes(),
            self.layer_head_root.as_str().as_bytes(),
            self.kv_type.as_bytes(),
            self.codec_version.as_bytes(),
            &generation,
            &graph,
            &epoch,
            self.authority_state_root.as_str().as_bytes(),
        ])
    }

    pub fn compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    pub fn with_codec(&self, codec_version: impl Into<String>) -> Self {
        let mut result = self.clone();
        result.codec_version = codec_version.into();
        result
    }

    pub fn with_generation(
        &self,
        runtime_generation: u64,
        graph_generation: u64,
        decode_epoch: u64,
    ) -> Self {
        let mut result = self.clone();
        result.runtime_generation = runtime_generation;
        result.graph_generation = graph_generation;
        result.decode_epoch = decode_epoch;
        result
    }
}

/// Identity of a logical token prefix.  Physical page locations are absent by
/// design: moving/evicting a page cannot change this key.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PrefixIdentity {
    pub model_root: StableDigest,
    pub tokenizer_root: StableDigest,
    pub token_sequence: Vec<u32>,
    pub rope_config_root: StableDigest,
    pub layer_start: u32,
    pub layer_end: u32,
    pub head_start: u32,
    pub head_end: u32,
    pub kv_type: String,
    pub codec_version: String,
    pub runtime_generation: u64,
    pub authority_state_root: StableDigest,
}

impl PrefixIdentity {
    pub fn token_sequence_root(&self) -> StableDigest {
        let mut encoded = Vec::with_capacity(self.token_sequence.len() * 4);
        for token in &self.token_sequence {
            encoded.extend_from_slice(&token.to_le_bytes());
        }
        StableDigest::from_parts(&[&encoded])
    }

    pub fn layer_head_root(&self) -> StableDigest {
        StableDigest::from_parts(&[
            &self.layer_start.to_le_bytes(),
            &self.layer_end.to_le_bytes(),
            &self.head_start.to_le_bytes(),
            &self.head_end.to_le_bytes(),
        ])
    }

    pub fn closure(&self, graph_generation: u64, decode_epoch: u64) -> KVDependencyClosure {
        KVDependencyClosure {
            model_root: self.model_root.clone(),
            tokenizer_root: self.tokenizer_root.clone(),
            token_sequence_root: self.token_sequence_root(),
            rope_config_root: self.rope_config_root.clone(),
            layer_head_root: self.layer_head_root(),
            kv_type: self.kv_type.clone(),
            codec_version: self.codec_version.clone(),
            runtime_generation: self.runtime_generation,
            graph_generation,
            decode_epoch,
            authority_state_root: self.authority_state_root.clone(),
        }
    }
}
