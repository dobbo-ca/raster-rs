//! `quality [--threads N] <ppm-dir> <out-dir>`: the Rust side of the real-file quality check. For every
//! binary P6 `<stem>.ppm` in `<ppm-dir>` it writes the seven outputs that `tests/parity/go/oracle
//! -quality` writes (`s2rgb.ppm`, `luma.pgm`, `s2gray.pgm`, `rsrgb.ppm`, `rsup.pgm`, `rot.ppm`,
//! `rotg.pgm` after the stem), byte for byte. Every resample of a run goes through one kept
//! `raster::Resampler`; the Go side calls the free function, so the outputs must be equal. Any error stops the run. See `docs/quality.md`.

use raster::{Limits, Plane, PlaneMut, Resampler};
use std::cell::RefCell;
use std::error::Error;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::{env, fs};

type Res<T> = Result<T, Box<dyn Error>>;

/// The most pixels of one tile: the same bound as the Go side.
const MAX_TILE_PIXELS: usize = 1 << 26;

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0c)
}

fn token<'a>(b: &'a [u8], pos: &mut usize) -> &'a [u8] {
    while b.get(*pos).is_some_and(|&c| is_space(c)) {
        *pos += 1;
    }
    let start = *pos;
    while b.get(*pos).is_some_and(|&c| !is_space(c)) {
        *pos += 1;
    }
    &b[start..*pos]
}

fn number(t: &[u8]) -> Option<usize> {
    std::str::from_utf8(t).ok()?.parse().ok()
}

/// `P6 <w> <h> 255`, one whitespace byte, then exactly w*h*3 bytes.
fn parse_p6(b: &[u8]) -> Result<(usize, usize, &[u8]), &'static str> {
    let mut pos = 0;
    let magic = token(b, &mut pos);
    let w = token(b, &mut pos);
    let h = token(b, &mut pos);
    let maxval = token(b, &mut pos);
    if magic != b"P6" || maxval != b"255" || pos >= b.len() {
        return Err("not a P6 PPM with maxval 255");
    }
    pos += 1;
    let size = number(w)
        .zip(number(h))
        .filter(|&(w, h)| w > 0 && h > 0 && w <= MAX_TILE_PIXELS && h <= MAX_TILE_PIXELS)
        .filter(|&(w, h)| w * h <= MAX_TILE_PIXELS);
    let Some((w, h)) = size else {
        return Err("bad PPM size");
    };
    if b.len() - pos != 3 * w * h {
        return Err("PPM data length does not match its size");
    }
    Ok((w, h, &b[pos..]))
}

fn buf(n: usize) -> Res<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)?;
    v.resize(n, 0);
    Ok(v)
}

fn plane(pix: &[u8], w: usize, h: usize, channels: usize) -> Plane<'_> {
    Plane {
        pix,
        w,
        h,
        stride: w * channels,
        channels,
    }
}

fn plane_mut(pix: &mut [u8], w: usize, h: usize, channels: usize) -> PlaneMut<'_> {
    PlaneMut {
        pix,
        w,
        h,
        stride: w * channels,
        channels,
    }
}

/// `<stem>.<ext>`: `P6` (3 channels) or `P5` header, then the samples.
fn put(
    out: &Path,
    stem: &str,
    ext: &str,
    w: usize,
    h: usize,
    channels: usize,
    pix: &[u8],
) -> Res<()> {
    let magic = if channels == 1 { "P5" } else { "P6" };
    let mut b = format!("{magic}\n{w} {h}\n255\n").into_bytes();
    b.try_reserve_exact(pix.len())?;
    b.extend_from_slice(pix);
    fs::write(out.join(format!("{stem}.{ext}")), b)?;
    Ok(())
}

struct Run<'a> {
    out: &'a Path,
    stem: &'a str,
    lim: &'a Limits,
    cancel: AtomicBool,
    rs: &'a RefCell<Resampler>,
}

impl Run<'_> {
    /// The 2:1 box of `src`; nothing for a plane under 2 samples in either direction.
    fn half(&self, ext: &str, src: &Plane) -> Res<()> {
        if src.w < 2 || src.h < 2 {
            return Ok(());
        }
        let (w, h, c) = (src.w / 2, src.h / 2, src.channels);
        let mut d = buf(w * h * c)?;
        raster::scale2to1(&mut plane_mut(&mut d, w, h, c), src, self.lim, &self.cancel)?;
        put(self.out, self.stem, ext, w, h, c, &d)
    }

    fn resize(&self, ext: &str, src: &Plane, w: usize, h: usize) -> Res<()> {
        let c = src.channels;
        let mut d = buf(w * h * c)?;
        self.rs.borrow_mut().resample(
            &mut plane_mut(&mut d, w, h, c),
            src,
            self.lim,
            &self.cancel,
        )?;
        put(self.out, self.stem, ext, w, h, c, &d)
    }
}

fn run_tile(path: &Path, out: &Path, stem: &str, lim: &Limits, rs: &RefCell<Resampler>) -> Res<()> {
    let bytes = fs::read(path)?;
    let (w, h, pix) = parse_p6(&bytes)?;
    let r = Run {
        out,
        stem,
        lim,
        cancel: AtomicBool::new(false),
        rs,
    };
    let rgb = plane(pix, w, h, 3);
    let mut grey = buf(w * h)?;
    raster::luma(&mut plane_mut(&mut grey, w, h, 1), &rgb, lim, &r.cancel)?;
    let gp = plane(&grey, w, h, 1);
    r.half("s2rgb.ppm", &rgb)?;
    put(out, stem, "luma.pgm", w, h, 1, &grey)?;
    r.half("s2gray.pgm", &gp)?;
    r.resize("rsrgb.ppm", &rgb, (w * 3 / 7).max(1), (h * 3 / 7).max(1))?;
    let (uw, uh) = (w * 3 / 2, h * 3 / 2);
    if uw * uh <= 1 << 22 {
        r.resize("rsup.pgm", &gp, uw, uh)?;
    }
    let mut rot = buf(3 * w * h)?;
    raster::rotate90(
        &mut plane_mut(&mut rot, h, w, 3),
        &rgb,
        true,
        lim,
        &r.cancel,
    )?;
    put(out, stem, "rot.ppm", h, w, 3, &rot)?;
    // The grey plane takes the 16x16 tile kernel of the AVX-512 tier.
    let mut rotg = buf(w * h)?;
    raster::rotate90(
        &mut plane_mut(&mut rotg, h, w, 1),
        &gp,
        true,
        lim,
        &r.cancel,
    )?;
    put(out, stem, "rotg.pgm", h, w, 1, &rotg)
}

fn usage() -> ! {
    eprintln!("usage: quality [--threads N] <ppm-dir> <out-dir>");
    std::process::exit(2);
}

fn main() -> Res<()> {
    let mut args: Vec<String> = env::args().collect();
    let mut limits = Limits {
        max_alloc: 1 << 30,
        ..Limits::default()
    };
    if args.get(1).is_some_and(|a| a == "--threads") {
        let Some(n) = args.get(2).and_then(|n| n.parse::<usize>().ok()) else {
            usage()
        };
        limits.max_threads = n;
        args.drain(1..3);
    }
    let [_, input, output] = args.as_slice() else {
        usage()
    };
    // First stdout line: the tier that produced the outputs being compared.
    // RASTER_EXPECT_AVX512=1 without RASTER_SIMD must run on AVX-512, so a host without it fails.
    let tier = raster::tier();
    println!("tier: {tier}");
    if env::var_os("RASTER_EXPECT_AVX512").is_some_and(|v| v == "1")
        && env::var_os("RASTER_SIMD").is_none()
        && tier != "avx512"
    {
        eprintln!("quality: RASTER_EXPECT_AVX512=1 but the tier is {tier}");
        std::process::exit(1);
    }
    fs::create_dir_all(output)?;
    let rs = RefCell::new(Resampler::default());
    let mut names = Vec::new();
    for e in fs::read_dir(input)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        if name.ends_with(".ppm") && e.file_type()?.is_file() {
            names.push(name);
        }
    }
    names.sort();
    for name in &names {
        let stem = name.strip_suffix(".ppm").unwrap_or(name);
        run_tile(
            &Path::new(input).join(name),
            Path::new(output),
            stem,
            &limits,
            &rs,
        )
        .map_err(|e| format!("{name}: {e}"))?;
    }
    println!("{} tiles", names.len());
    Ok(())
}
