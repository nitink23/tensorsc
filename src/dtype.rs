//! Element types.
//!
//! A tensor's dtype is chosen at runtime (Python decides it), so storage holds
//! raw bytes tagged with a [`DType`]. Kernels are written once, generically
//! over [`Element`], and the [`with_dtype!`] family of macros picks the
//! concrete Rust type for a runtime dtype.

use std::fmt;

/// Runs `$body` with `$T` bound to the Rust type behind `$dtype`.
///
/// ```ignore
/// with_dtype!(tensor.dtype(), T => tensor.to_vec::<T>().len())
/// ```
macro_rules! with_dtype {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::dtype::DType::Bool => {
                type $T = bool;
                $body
            }
            $crate::dtype::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::dtype::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::dtype::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::dtype::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
        }
    };
}
pub(crate) use with_dtype;

/// Element type of a tensor.
///
/// The variants are declared in promotion order: combining two dtypes gives
/// the later one (`int32` + `float32` is `float32`), as in PyTorch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    Bool,
    U8,
    I32,
    I64,
    F32,
    F64,
}

/// The broad kind of a dtype, used by promotion rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Bool,
    Int,
    Float,
}

impl DType {
    pub const ALL: [DType; 6] = [
        DType::Bool,
        DType::U8,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
    ];

    /// The float dtype used when one isn't specified, e.g. for `1 / int_tensor`.
    pub const DEFAULT_FLOAT: DType = DType::F32;

    /// Size of one element in bytes.
    pub fn itemsize(self) -> usize {
        with_dtype!(self, T => std::mem::size_of::<T>())
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "uint8",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::F32 => "float32",
            DType::F64 => "float64",
        }
    }

    pub fn from_name(name: &str) -> Option<DType> {
        DType::ALL.into_iter().find(|d| d.name() == name)
    }

    pub fn kind(self) -> Kind {
        match self {
            DType::Bool => Kind::Bool,
            DType::U8 | DType::I32 | DType::I64 => Kind::Int,
            DType::F32 | DType::F64 => Kind::Float,
        }
    }

    pub fn is_float(self) -> bool {
        self.kind() == Kind::Float
    }

    /// The dtype two operands are combined in: the later one in declaration order.
    pub fn promote(self, other: DType) -> DType {
        self.max(other)
    }

    /// This dtype if it's a float, otherwise the default float. Used by
    /// operations whose result is always fractional (`exp`, `/`, `mean`).
    pub fn to_float(self) -> DType {
        if self.is_float() {
            self
        } else {
            DType::DEFAULT_FLOAT
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One value of any dtype, for crossing API boundaries (`item`, `fill`,
/// Python conversion) without knowing the dtype at compile time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl Scalar {
    /// The dtype a value of this kind gets when nothing else decides.
    pub fn default_dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::I64,
            Scalar::Float(_) => DType::DEFAULT_FLOAT,
        }
    }

    pub fn to_f64(self) -> f64 {
        match self {
            Scalar::Bool(b) => b as u8 as f64,
            Scalar::Int(i) => i as f64,
            Scalar::Float(x) => x,
        }
    }

    pub fn to_i64(self) -> i64 {
        match self {
            Scalar::Bool(b) => b as i64,
            Scalar::Int(i) => i,
            Scalar::Float(x) => x as i64,
        }
    }

    pub fn to_bool(self) -> bool {
        match self {
            Scalar::Bool(b) => b,
            Scalar::Int(i) => i != 0,
            Scalar::Float(x) => x != 0.0,
        }
    }

    /// Converts to element type `T`, the same way a tensor cast would.
    pub fn to<T: Element>(self) -> T {
        match self {
            Scalar::Bool(b) => T::from_bool(b),
            Scalar::Int(i) => T::from_i64(i),
            Scalar::Float(x) => T::from_f64(x),
        }
    }
}

macro_rules! scalar_from {
    ($($t:ty => $variant:ident as $as:ty),*) => {
        $(impl From<$t> for Scalar {
            fn from(value: $t) -> Scalar {
                Scalar::$variant(value as $as)
            }
        })*
    };
}

scalar_from!(bool => Bool as bool, u8 => Int as i64, i32 => Int as i64, i64 => Int as i64,
             f32 => Float as f64, f64 => Float as f64);

mod sealed {
    pub trait Sealed {}
}

/// A Rust type that can be a tensor element. Implemented for exactly the
/// types behind each [`DType`].
pub trait Element:
    Copy + PartialEq + PartialOrd + Send + Sync + fmt::Debug + 'static + sealed::Sealed
{
    const DTYPE: DType;

    fn to_f64(self) -> f64;
    fn to_i64(self) -> i64;
    fn to_bool(self) -> bool;
    /// Float to int truncates toward zero and saturates (NaN becomes 0).
    fn from_f64(value: f64) -> Self;
    /// Wraps on overflow, like NumPy and PyTorch.
    fn from_i64(value: i64) -> Self;
    fn from_bool(value: bool) -> Self;
    fn to_scalar(self) -> Scalar;
    /// Writes the value as Python would show it (`True`, `3`, `2.5`).
    fn write(self, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// Converts between element types without going through a lossy middle
    /// type: ints stay exact, floats round, anything non-zero is `true`.
    fn cast<U: Element>(self) -> U {
        match Self::DTYPE.kind() {
            Kind::Float => U::from_f64(self.to_f64()),
            Kind::Int => U::from_i64(self.to_i64()),
            Kind::Bool => U::from_bool(self.to_bool()),
        }
    }
}

macro_rules! numeric_element {
    ($($t:ty => $dtype:ident, $fmt:literal;)*) => {
        $(
            impl sealed::Sealed for $t {}

            impl Element for $t {
                const DTYPE: DType = DType::$dtype;

                fn to_f64(self) -> f64 { self as f64 }
                fn to_i64(self) -> i64 { self as i64 }
                fn to_bool(self) -> bool { self != (0 as $t) }
                fn from_f64(value: f64) -> Self { value as $t }
                fn from_i64(value: i64) -> Self { value as $t }
                fn from_bool(value: bool) -> Self { value as u8 as $t }
                fn to_scalar(self) -> Scalar { Scalar::from(self) }
                fn write(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, $fmt, self)
                }
            }
        )*
    };
}

numeric_element! {
    u8 => U8, "{}";
    i32 => I32, "{}";
    i64 => I64, "{}";
    f32 => F32, "{:?}";
    f64 => F64, "{:?}";
}

impl sealed::Sealed for bool {}

impl Element for bool {
    const DTYPE: DType = DType::Bool;

    fn to_f64(self) -> f64 {
        self as u8 as f64
    }
    fn to_i64(self) -> i64 {
        self as i64
    }
    fn to_bool(self) -> bool {
        self
    }
    fn from_f64(value: f64) -> Self {
        // NaN is truthy, as in Python.
        value != 0.0
    }
    fn from_i64(value: i64) -> Self {
        value != 0
    }
    fn from_bool(value: bool) -> Self {
        value
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Bool(self)
    }
    fn write(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self { "True" } else { "False" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_follows_declaration_order() {
        assert_eq!(DType::Bool.promote(DType::I64), DType::I64);
        assert_eq!(DType::U8.promote(DType::I32), DType::I32);
        assert_eq!(DType::I64.promote(DType::F32), DType::F32);
        assert_eq!(DType::F32.promote(DType::F64), DType::F64);
        assert_eq!(DType::I32.to_float(), DType::F32);
        assert_eq!(DType::F64.to_float(), DType::F64);
    }

    #[test]
    fn casts() {
        assert_eq!((-2.7f32).cast::<i64>(), -2);
        assert_eq!(f64::NAN.cast::<i32>(), 0);
        assert!(f32::NAN.cast::<bool>());
        assert_eq!(300i64.cast::<u8>(), 44);
        assert_eq!(
            i64::MAX.cast::<i64>(),
            i64::MAX,
            "ints don't pass through f64"
        );
        assert_eq!(true.cast::<f32>(), 1.0);
        assert!(!0u8.cast::<bool>());
    }

    #[test]
    fn names_round_trip() {
        for dtype in DType::ALL {
            assert_eq!(DType::from_name(dtype.name()), Some(dtype));
        }
        assert_eq!(DType::from_name("float16"), None);
    }

    #[test]
    fn itemsizes() {
        let sizes: Vec<usize> = DType::ALL.iter().map(|d| d.itemsize()).collect();
        assert_eq!(sizes, vec![1, 1, 4, 8, 4, 8]);
    }
}
