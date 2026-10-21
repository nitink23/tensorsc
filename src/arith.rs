//! Per-element arithmetic for each element type, and macros that pick the
//! Rust type for a runtime dtype within a family (numeric, float, bitwise).
//!
//! Integer arithmetic wraps on overflow and `//`/`%` follow Python (floor
//! division; the remainder takes the divisor's sign), matching NumPy and
//! PyTorch.

use crate::dtype::Element;

pub(crate) trait Numeric: Element {
    const ZERO: Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    /// True division for floats. Never used for integers: `/` converts them
    /// to float first.
    fn div(self, other: Self) -> Self;
    fn floor_div(self, other: Self) -> Self;
    fn rem(self, other: Self) -> Self;
    fn pow(self, exponent: Self) -> Self;
    /// NaN wins, as in NumPy's `maximum` (`f32::max` would drop it).
    fn maximum(self, other: Self) -> Self;
    fn minimum(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
}

pub(crate) trait Float: Numeric {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    /// Rounds half to even, like NumPy and PyTorch.
    fn round(self) -> Self;
    fn trunc(self) -> Self;
}

pub(crate) trait Bitwise: Element {
    fn and(self, other: Self) -> Self;
    fn or(self, other: Self) -> Self;
    fn xor(self, other: Self) -> Self;
    fn not(self) -> Self;
}

/// `base ** exponent` by squaring, wrapping on overflow. The caller rejects
/// negative exponents.
fn int_pow(base: i64, exponent: i64) -> i64 {
    let (mut result, mut base, mut exponent) = (1i64, base, exponent as u64);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}

macro_rules! int_numeric {
    ($($t:ty),*) => {
        // `x < 0` is always false for u8; that's fine.
        $(#[allow(unused_comparisons)]
        impl Numeric for $t {
            const ZERO: Self = 0;
            fn add(self, other: Self) -> Self { self.wrapping_add(other) }
            fn sub(self, other: Self) -> Self { self.wrapping_sub(other) }
            fn mul(self, other: Self) -> Self { self.wrapping_mul(other) }
            fn div(self, other: Self) -> Self { self.wrapping_div(other) }
            fn floor_div(self, other: Self) -> Self {
                let q = self.wrapping_div(other);
                // Truncation rounded toward zero; step down when the signs differ.
                if self.wrapping_rem(other) != 0 && ((self < 0) != (other < 0)) { q - 1 } else { q }
            }
            fn rem(self, other: Self) -> Self {
                let r = self.wrapping_rem(other);
                if r != 0 && ((r < 0) != (other < 0)) { r.wrapping_add(other) } else { r }
            }
            fn pow(self, exponent: Self) -> Self {
                // Computing in i64 and narrowing gives the right wrapped
                // result, since 2^bits divides 2^64.
                int_pow(self as i64, exponent as i64) as $t
            }
            fn maximum(self, other: Self) -> Self { self.max(other) }
            fn minimum(self, other: Self) -> Self { self.min(other) }
            fn neg(self) -> Self { (0 as $t).wrapping_sub(self) }
            fn abs(self) -> Self { if self < 0 { self.neg() } else { self } }
        })*
    };
}

int_numeric!(u8, i32, i64);

macro_rules! float_numeric {
    ($($t:ty),*) => {
        $(impl Numeric for $t {
            const ZERO: Self = 0.0;
            fn add(self, other: Self) -> Self { self + other }
            fn sub(self, other: Self) -> Self { self - other }
            fn mul(self, other: Self) -> Self { self * other }
            fn div(self, other: Self) -> Self { self / other }
            fn floor_div(self, other: Self) -> Self { float_divmod(self, other).0 }
            fn rem(self, other: Self) -> Self { float_divmod(self, other).1 }
            fn pow(self, exponent: Self) -> Self { self.powf(exponent) }
            fn maximum(self, other: Self) -> Self {
                if self.is_nan() || other.is_nan() { <$t>::NAN } else { self.max(other) }
            }
            fn minimum(self, other: Self) -> Self {
                if self.is_nan() || other.is_nan() { <$t>::NAN } else { self.min(other) }
            }
            fn neg(self) -> Self { -self }
            fn abs(self) -> Self { <$t>::abs(self) }
        }

        impl Float for $t {
            fn exp(self) -> Self { <$t>::exp(self) }
            fn ln(self) -> Self { <$t>::ln(self) }
            fn sqrt(self) -> Self { <$t>::sqrt(self) }
            fn sin(self) -> Self { <$t>::sin(self) }
            fn cos(self) -> Self { <$t>::cos(self) }
            fn tanh(self) -> Self { <$t>::tanh(self) }
            fn floor(self) -> Self { <$t>::floor(self) }
            fn ceil(self) -> Self { <$t>::ceil(self) }
            fn round(self) -> Self { <$t>::round_ties_even(self) }
            fn trunc(self) -> Self { <$t>::trunc(self) }
        })*
    };
}

float_numeric!(f32, f64);

/// Python's float `divmod`, ported from CPython's `float_divmod` so results
/// match `a // b` and `a % b` on Python floats exactly, signs of zero included.
/// Computing in f64 is exact for f32 inputs.
fn float_divmod<T: Element>(a: T, b: T) -> (T, T) {
    let (a64, b64) = (a.to_f64(), b.to_f64());
    if b64 == 0.0 {
        // Python raises here; NumPy and PyTorch follow IEEE instead.
        return (T::from_f64(a64 / b64), T::from_f64(f64::NAN));
    }
    let mut m = a64 % b64;
    let mut div = (a64 - m) / b64;
    if m != 0.0 {
        if (b64 < 0.0) != (m < 0.0) {
            m += b64;
            div -= 1.0;
        }
    } else {
        m = 0f64.copysign(b64);
    }
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        0f64.copysign(a64 / b64)
    };
    (T::from_f64(floordiv), T::from_f64(m))
}

macro_rules! int_bitwise {
    ($($t:ty),*) => {
        $(impl Bitwise for $t {
            fn and(self, other: Self) -> Self { self & other }
            fn or(self, other: Self) -> Self { self | other }
            fn xor(self, other: Self) -> Self { self ^ other }
            fn not(self) -> Self { !self }
        })*
    };
}

int_bitwise!(bool, u8, i32, i64);

/// Runs `$body` with `$T` bound to the numeric type behind `$dtype`;
/// evaluates `$bool` for `Bool`.
macro_rules! with_numeric {
    ($dtype:expr, $T:ident => $body:expr, bool => $bool:expr) => {
        match $dtype {
            $crate::dtype::DType::Bool => $bool,
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
pub(crate) use with_numeric;

/// Runs `$body` with `$T` bound to the float type behind `$dtype`, which the
/// caller has already made sure is a float.
macro_rules! with_float {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::dtype::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
            other => unreachable!("{other} is not a float dtype"),
        }
    };
}
pub(crate) use with_float;

/// Runs `$body` with `$T` bound to the integer or bool type behind `$dtype`;
/// evaluates `$float` for float dtypes.
macro_rules! with_bitwise {
    ($dtype:expr, $T:ident => $body:expr, float => $float:expr) => {
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
            $crate::dtype::DType::F32 | $crate::dtype::DType::F64 => $float,
        }
    };
}
pub(crate) use with_bitwise;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_division_follows_python() {
        // (a, b, a // b, a % b) as computed by Python.
        let cases: [(i64, i64, i64, i64); 4] = [
            (7, 2, 3, 1),
            (-7, 2, -4, 1),
            (7, -2, -4, -1),
            (-7, -2, 3, -1),
        ];
        for (a, b, q, r) in cases {
            assert_eq!(Numeric::floor_div(a, b), q, "{a} // {b}");
            assert_eq!(Numeric::rem(a, b), r, "{a} % {b}");
            assert_eq!(Numeric::floor_div(a as i32, b as i32), q as i32);
        }
        assert_eq!(
            Numeric::floor_div(i64::MIN, -1),
            i64::MIN,
            "wraps instead of panicking"
        );
        assert_eq!(Numeric::floor_div(7u8, 2), 3);
    }

    #[test]
    fn float_division_follows_python() {
        for (a, b, q, r) in [
            (7.5, 2.0, 3.0, 1.5),
            (-7.5, 2.0, -4.0, 0.5),
            (7.5, -2.0, -4.0, -0.5),
        ] {
            assert_eq!(Numeric::floor_div(a, b), q, "{a} // {b}");
            assert_eq!(Numeric::rem(a, b), r, "{a} % {b}");
        }
        assert!(Numeric::rem(-0.0f64, 3.0).is_sign_positive());
        assert!(Numeric::rem(1.0f64, -3.0) == -2.0);
        assert_eq!(Numeric::floor_div(1.0f64, 0.0), f64::INFINITY);
    }

    #[test]
    fn integer_pow_wraps() {
        assert_eq!(Numeric::pow(3i64, 4), 81);
        assert_eq!(Numeric::pow(2i32, 31), i32::MIN);
        assert_eq!(Numeric::pow(2u8, 9), 0);
        assert_eq!(Numeric::pow(-2i64, 3), -8);
        assert_eq!(Numeric::pow(5i64, 0), 1);
    }

    #[test]
    fn nan_wins_max_and_min() {
        assert!(Numeric::maximum(f32::NAN, 1.0).is_nan());
        assert!(Numeric::minimum(1.0f64, f64::NAN).is_nan());
        assert_eq!(Numeric::maximum(3i64, 5), 5);
    }

    #[test]
    fn rounding_is_half_to_even() {
        let rounded: Vec<f32> = [0.5f32, 1.5, 2.5, -0.5]
            .iter()
            .map(|&x| Float::round(x))
            .collect();
        assert_eq!(rounded, vec![0.0, 2.0, 2.0, -0.0]);
    }
}
