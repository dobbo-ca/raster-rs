#![allow(dead_code)]
use raster::{Plane, PlaneMut};
use std::sync::atomic::AtomicBool;

pub static NO_CANCEL: AtomicBool = AtomicBool::new(false);

/// n pseudo-random bytes from seed (xorshift64*), as raster-go's `fixture.Bytes`.
pub fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect()
}

/// A w*h*ch raster with ramps, hard edges and noise, as raster-go's `fixture.Page`.
pub fn page(seed: u64, w: usize, h: usize, ch: usize) -> Vec<u8> {
    let mut b = bytes(seed, w * h * ch);
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let i = (y * w + x) * ch + c;
                b[i] = match (x / 16 + y / 16) % 5 {
                    0 => 255,
                    1 => (x * 3 + y * 5 + c * 40) as u8,
                    2 => 0,
                    _ => b[i] & 0x1f | (x + y) as u8,
                };
            }
        }
    }
    b
}

/// An owned plane with `pad` spare bytes at the end of each row.
pub struct Buf {
    pub pix: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub ch: usize,
}

impl Buf {
    pub fn plane(&self) -> Plane<'_> {
        Plane {
            pix: &self.pix,
            w: self.w,
            h: self.h,
            stride: self.stride,
            channels: self.ch,
        }
    }
    pub fn plane_mut(&mut self) -> PlaneMut<'_> {
        PlaneMut {
            pix: &mut self.pix,
            w: self.w,
            h: self.h,
            stride: self.stride,
            channels: self.ch,
        }
    }
    pub fn at(&self, x: usize, y: usize, c: usize) -> i32 {
        i32::from(self.pix[y * self.stride + x * self.ch + c])
    }
}

/// A page plane; the padding bytes are 0xEE so a kernel that reads them shows.
pub fn plane(seed: u64, w: usize, h: usize, ch: usize, pad: usize) -> Buf {
    let stride = w * ch + pad;
    let fill = page(seed, w, h, ch);
    let mut pix = vec![0xEE; (h - 1) * stride + w * ch];
    for y in 0..h {
        pix[y * stride..][..w * ch].copy_from_slice(&fill[y * w * ch..][..w * ch]);
    }
    Buf {
        pix,
        w,
        h,
        stride,
        ch,
    }
}

pub fn blank(w: usize, h: usize, ch: usize, pad: usize) -> Buf {
    let stride = w * ch + pad;
    Buf {
        pix: vec![0; (h - 1) * stride + w * ch],
        w,
        h,
        stride,
        ch,
    }
}

/// Every sample of `got` equals `want(x, y, c)`.
pub fn same(name: &str, got: &Buf, want: impl Fn(usize, usize, usize) -> i32) {
    for y in 0..got.h {
        for x in 0..got.w {
            for c in 0..got.ch {
                let (g, w) = (got.at(x, y, c), want(x, y, c));
                assert_eq!(g, w, "{name}: ({x},{y},{c})");
            }
        }
    }
}

/// The visible samples of a plane, row by row, without padding.
pub fn samples(b: &Buf) -> Vec<u8> {
    (0..b.h)
        .flat_map(|y| b.pix[y * b.stride..][..b.w * b.ch].iter().copied())
        .collect()
}

pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, c) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[test]
fn sha256_known() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
