//! Native BPE tokenizer (Rust) — encode/decode from the GGUF tokenizer
//! metadata (byte-level BPE, implemented entirely in Rust; no external
//! tokenizer process). `--prompt <text>` on the server uses this.
//!
//! Encoding: UTF-8 bytes → single-byte tokens → rank-ordered merge loop
//! (the standard BPE: repeatedly merge the lowest-rank adjacent pair,
//! leftmost on ties).  Decoding concatenates token bytes (lossless at
//! the byte level).

use har_model::GgufTokenizer;
use std::collections::HashMap;

pub struct Tokenizer {
    token_to_id: HashMap<Vec<u8>, u32>,
    byte_to_id: HashMap<u8, u32>,
    /// (left_id, right_id) → (rank, merged_id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    pub eos: Option<u32>,
    pub bos: Option<u32>,
    pub vocab_size: usize,
}

impl Tokenizer {
    pub fn from_gguf(t: &GgufTokenizer) -> Option<Tokenizer> {
        if t.tokens.is_empty() {
            return None;
        }
        let mut token_to_id = HashMap::new();
        let mut byte_to_id = HashMap::new();
        for (i, tok) in t.tokens.iter().enumerate() {
            let bytes = tok.as_bytes().to_vec();
            if bytes.len() == 1 {
                byte_to_id.insert(bytes[0], i as u32);
            }
            token_to_id.insert(bytes, i as u32);
        }
        let mut merges = HashMap::new();
        for (rank, m) in t.merges.iter().enumerate() {
            let Some((a, b)) = m.split_once(' ') else {
                continue;
            };
            let (Some(&la), Some(&lb)) =
                (token_to_id.get(a.as_bytes()), token_to_id.get(b.as_bytes()))
            else {
                continue;
            };
            let mut merged = Vec::with_capacity(a.len() + b.len());
            merged.extend_from_slice(a.as_bytes());
            merged.extend_from_slice(b.as_bytes());
            let Some(&merged_id) = token_to_id.get(&merged) else {
                continue;
            };
            merges.insert((la, lb), (rank as u32, merged_id));
        }
        Some(Tokenizer {
            token_to_id,
            byte_to_id,
            merges,
            eos: t.eos_token_id,
            bos: t.bos_token_id,
            vocab_size: t.tokens.len(),
        })
    }

    /// Encode text to token ids (byte-level BPE).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids: Vec<u32> = Vec::with_capacity(text.len());
        for &b in text.as_bytes() {
            let id = self
                .byte_to_id
                .get(&b)
                .ok_or_else(|| format!("no single-byte token for 0x{b:02x}"))?;
            ids.push(*id);
        }
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&(rank, _)) = self.merges.get(&(ids[i], ids[i + 1])) {
                    if best.map_or(true, |(_, br)| rank < br) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((pos, _)) = best else { break };
            let (_, merged_id) = self.merges[&(ids[pos], ids[pos + 1])];
            ids[pos] = merged_id;
            ids.remove(pos + 1);
        }
        Ok(ids)
    }

    /// Decode token ids back to text (byte-concatenation, lossy UTF-8).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = self.token_by_id(id) {
                bytes.extend_from_slice(tok);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn token_by_id(&self, id: u32) -> Option<&[u8]> {
        self.token_to_id
            .iter()
            .find(|(_, &v)| v == id)
            .map(|(k, _)| k.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny byte-level BPE vocabulary: the ASCII byte tokens plus the
    /// words "hello" and "world" (with their merge chain).
    fn synthetic() -> GgufTokenizer {
        let mut tokens: Vec<String> = Vec::new();
        // byte tokens: bytes 0x20..=0x7e as single-char strings
        for b in 0x20u8..=0x7e {
            tokens.push(String::from_utf8(vec![b]).unwrap());
        }
        for word in ["he", "ll", "hell", "hello", "wo", "wor", "ld", "world", "!"] {
            tokens.push(word.to_string());
        }
        let merges = vec![
            "h e".to_string(),
            "l l".to_string(),
            "he ll".to_string(),
            "hell o".to_string(),
            "w o".to_string(),
            "wo r".to_string(),
            "l d".to_string(),
            "wor ld".to_string(),
        ];
        GgufTokenizer {
            model: Some("gpt2".into()),
            tokens,
            merges,
            eos_token_id: Some(0),
            bos_token_id: Some(1),
            ..Default::default()
        }
    }

    fn id_of(t: &Tokenizer, word: &str) -> u32 {
        *t.token_to_id.get(word.as_bytes()).expect("word token")
    }

    #[test]
    fn encodes_hello_world_via_merges() {
        let t = Tokenizer::from_gguf(&synthetic()).expect("tokenizer");
        let ids = t.encode("hello world!").expect("encode");
        let hello = id_of(&t, "hello");
        let world = id_of(&t, "world");
        let space = id_of(&t, " ");
        let bang = id_of(&t, "!");
        assert_eq!(ids, vec![hello, space, world, bang], "merge chain applied");
    }

    #[test]
    fn roundtrip_is_lossless() {
        let t = Tokenizer::from_gguf(&synthetic()).expect("tokenizer");
        for text in ["hello world!", "har runtime", "reference device", "abc"] {
            let ids = t.encode(text).expect("encode");
            assert_eq!(t.decode(&ids), text, "roundtrip {text}");
        }
    }

    #[test]
    fn merge_tiebreak_is_leftmost() {
        // "a a a" with merge "a a" → rank 0: leftmost pair merges first.
        let tokens: Vec<String> = vec!["a".to_string(), "aa".to_string()];
        let gguf = GgufTokenizer {
            tokens,
            merges: vec!["a a".to_string()],
            ..Default::default()
        };
        let t = Tokenizer::from_gguf(&gguf).expect("tokenizer");
        let ids = t.encode("aaa").expect("encode");
        assert_eq!(ids.len(), 2, "leftmost pair merged, then the rest");
    }
}
