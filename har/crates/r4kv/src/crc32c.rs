//! CRC32C (Castagnoli), reflected, init/final xor 0xffffffff.
//! Used for page payload + header integrity. Small, dependency-free.

fn table() -> &'static [u32; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0x82f6_3b78 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            t[i as usize] = c;
        }
        t
    })
}

pub fn crc32c(data: &[u8]) -> u32 {
    let t = table();
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c = t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xffff_ffff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(crc32c(b""), 0x0000_0000);
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(crc32c(b"hello world"), 0xc994_65aa);
    }

    #[test]
    fn detects_single_bit_flip() {
        let d = [7u8; 512];
        let base = crc32c(&d);
        let mut flipped = d;
        flipped[123] ^= 0x10;
        assert_ne!(crc32c(&flipped), base);
    }
}
