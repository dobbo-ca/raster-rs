// One parity call: the input format of tests/parity/go/oracle (see docs/parity.md). Included by
// examples/oracle.rs and the `ops` fuzz target, so there is one parser.
//
//     [op][lim][flags][sch][dch] then sw sh sst dw dh dst dlen as u32 LE, then the source bytes.

use raster::{Error, Limits, Plane, PlaneMut};
use std::sync::atomic::AtomicBool;

pub const HDR_LEN: usize = 33;
/// A larger `dlen` is bad input: the oracles bound their own allocations.
pub const MAX_DST: u32 = 1 << 21;

pub struct Case<'a> {
    op: u8,
    lim: u8,
    flags: u8,
    sch: u8,
    dch: u8,
    /// sw sh sst dw dh dst dlen
    f: [u32; 7],
    pix: &'a [u8],
}

impl<'a> Case<'a> {
    pub fn parse(b: &'a [u8]) -> Option<Self> {
        let (h, pix) = b.split_first_chunk::<HDR_LEN>()?;
        let mut f = [0u32; 7];
        for (v, c) in f.iter_mut().zip(h[5..].as_chunks::<4>().0) {
            *v = u32::from_le_bytes(*c);
        }
        (h[0] <= 4 && f[6] <= MAX_DST).then_some(Self {
            op: h[0],
            lim: h[1],
            flags: h[2],
            sch: h[3],
            dch: h[4],
            f,
            pix,
        })
    }

    /// The limits for the `lim` byte; `max_alloc` is 64 MiB so the oracles stay small. Threads
    /// and pool are the defaults.
    pub fn limits(&self) -> Limits {
        let mut l = Limits {
            max_pixels: 1 << 26,
            max_alloc: 64 << 20,
            max_output: 1 << 28,
            ..Limits::default()
        };
        match self.lim {
            1 => l.max_pixels = 4095,
            2 => l.max_alloc = 1 << 20,
            3 => l.max_output = 1023,
            4 => (l.max_pixels, l.max_output) = (4096, 1024),
            _ => {}
        }
        l
    }

    /// Runs the call; the destination bytes come back whether it failed or not.
    pub fn run(&self, limits: &Limits) -> (Result<(), Error>, Vec<u8>) {
        self.run_with(limits, None)
    }

    /// `run`, with a resample going through `rs` if one is given.
    pub fn run_with(
        &self,
        limits: &Limits,
        rs: Option<&mut raster::Resampler>,
    ) -> (Result<(), Error>, Vec<u8>) {
        let [sw, sh, sst, dw, dh, dst, dlen] = self.f.map(|v| v as usize);
        let mut out = vec![0u8; dlen];
        let src = Plane {
            pix: self.pix,
            w: sw,
            h: sh,
            stride: sst,
            channels: usize::from(self.sch),
        };
        let mut d = PlaneMut {
            pix: &mut out,
            w: dw,
            h: dh,
            stride: dst,
            channels: usize::from(self.dch),
        };
        let cancel = AtomicBool::new(self.flags & 1 != 0);
        // Go checks its context once per row; this crate's resample also checks twice before
        // the tap tables, which write nothing, so those two are skipped over.
        #[cfg(feature = "cancel-hook")]
        {
            let n = usize::from(self.flags >> 1);
            raster::cancel_at_check(if n > 0 && self.op == 4 { n + 2 } else { n });
        }
        let r = match self.op {
            0 => raster::scale2to1(&mut d, &src, limits, &cancel),
            1 => raster::luma(&mut d, &src, limits, &cancel),
            2 | 3 => raster::rotate90(&mut d, &src, self.op == 2, limits, &cancel),
            _ => match rs {
                Some(rs) => rs.resample(&mut d, &src, limits, &cancel),
                None => raster::resample(&mut d, &src, limits, &cancel),
            },
        };
        #[cfg(feature = "cancel-hook")]
        raster::cancel_at_check(0);
        (r, out)
    }
}
