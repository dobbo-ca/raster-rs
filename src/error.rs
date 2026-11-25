use std::fmt;

/// One variant per Go error. [`Error::Invalid`] is `ErrInvalid`, the three `*Limit` variants are
/// `*LimitError` with `Name` "pixels", "scratch bytes" ([`Error::AllocLimit`]) and "output bytes"
/// (all `ErrLimit`), and [`Error::Cancelled`] is `context.Canceled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A bad plane (size, stride, channels, short buffer), mismatched planes, or a size that
    /// overflows. Nothing was read or written.
    Invalid(&'static str),
    /// The larger of source and destination pixels passed `Limits::max_pixels`.
    PixelLimit { got: u64, max: u64 },
    /// The working memory of the call passed `Limits::max_alloc`: the up-front need of `resample`
    /// (Go's "scratch bytes"), or a charge on the ledger that would take the bytes held to `got`.
    AllocLimit { got: u64, max: u64 },
    /// The destination bytes passed `Limits::max_output`.
    OutputLimit { got: u64, max: u64 },
    /// The allocator refused a size that is within the limits.
    OutOfMemory,
    /// The cancel flag was set.
    Cancelled,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "raster: invalid argument: {why}"),
            Self::PixelLimit { got, max } => write!(f, "raster: pixels {got} exceeds limit {max}"),
            Self::AllocLimit { got, max } => {
                write!(f, "raster: scratch bytes {got} exceeds limit {max}")
            }
            Self::OutputLimit { got, max } => {
                write!(f, "raster: output bytes {got} exceeds limit {max}")
            }
            Self::OutOfMemory => f.write_str("raster: out of memory"),
            Self::Cancelled => f.write_str("raster: cancelled"),
        }
    }
}

impl std::error::Error for Error {}
