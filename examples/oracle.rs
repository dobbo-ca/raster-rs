//! The Rust side of the parity check: `oracle <inputs> <outputs>` runs every input like
//! `tests/parity/go/oracle` and writes `<name>.out` (the destination bytes) or `<name>.err` (the error
//! class) and `<name>.partial.out` (the destination bytes left by the failed call). The input
//! format and the class table are in `docs/parity.md`.

use std::path::Path;
use std::{env, fs, io};

include!("../tests/parity/case.rs");

fn class(e: &Error) -> String {
    match e {
        Error::Invalid(_) => "invalid".into(),
        Error::PixelLimit { .. } => "limit-pixels".into(),
        Error::AllocLimit { .. } => "limit-scratch".into(),
        Error::OutputLimit { .. } => "limit-output".into(),
        Error::Cancelled => "cancelled".into(),
        e => format!("other:{e}"),
    }
}

fn main() -> io::Result<()> {
    let mut args: Vec<String> = env::args().collect();
    // `--threads N` sets `Limits::max_threads` for every case; the default is the crate's.
    let mut threads = None;
    if args.get(1).is_some_and(|a| a == "--threads") {
        threads = args.get(2).and_then(|n| n.parse::<usize>().ok());
        if threads.is_none() {
            eprintln!("usage: oracle [--threads N] <inputs> <outputs>");
            std::process::exit(2);
        }
        args.drain(1..3);
    }
    let [_, inputs, outputs] = args.as_slice() else {
        eprintln!("usage: oracle [--threads N] <inputs> <outputs>");
        std::process::exit(2);
    };
    // stderr, so the CI log shows which tier produced the outputs being diffed.
    eprintln!("tier: {}", raster::tier());
    fs::create_dir_all(outputs)?;
    for entry in fs::read_dir(inputs)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        let bytes = fs::read(Path::new(inputs).join(&name))?;
        let base = Path::new(outputs).join(&name);
        let with = |ext: &str| format!("{}.{ext}", base.display());
        let Some(case) = Case::parse(&bytes) else {
            fs::write(with("err"), "bad-input\n")?;
            continue;
        };
        let mut limits = case.limits();
        if let Some(n) = threads {
            limits.max_threads = n;
        }
        match case.run(&limits) {
            (Ok(()), out) => fs::write(with("out"), out)?,
            (Err(e), out) => {
                fs::write(with("err"), format!("{}\n", class(&e)))?;
                fs::write(with("partial.out"), out)?;
            }
        }
    }
    Ok(())
}
