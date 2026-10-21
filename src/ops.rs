//! Elementwise math, comparisons and broadcasting, for every dtype.
//!
//! Broadcasting follows NumPy: shapes are lined up from the right, and an axis
//! of length 1 (or a missing one) stretches to match the other side. The
//! stretch is a view with stride 0, so the one element is read repeatedly
//! instead of being copied.
//!
//! Dtypes follow PyTorch: operands are promoted to a common dtype (see
//! [`DType::promote`]), true division and functions like `exp` produce
//! floats, and comparisons produce bools. Each family of operations is a
//! trait implemented by one zero-sized type per operation, so the dtype
//! handling is written once per family and every (operation, element type)
//! pair still compiles to its own tight loop.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::arith::{Bitwise, Float, Numeric, with_bitwise, with_float, with_numeric};
use crate::dtype::{DType, Element, Scalar, with_dtype};
use crate::error::{Result, TensorError};
use crate::tensor::Tensor;

/// The shape two tensors broadcast to, e.g. `[3, 1]` and `[4]` give `[3, 4]`.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let ndim = a.len().max(b.len());
    // Missing leading axes count as length 1.
    let dim = |shape: &[usize], axis: usize| {
        let pad = ndim - shape.len();
        if axis < pad { 1 } else { shape[axis - pad] }
    };
    (0..ndim)
        .map(|axis| match (dim(a, axis), dim(b, axis)) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(TensorError::BroadcastMismatch {
                left: a.to_vec(),
                right: b.to_vec(),
            }),
        })
        .collect()
}

// ---- operation families ----

/// What an arithmetic op does when both operands are bool.
enum OnBool {
    /// Stays bool, using this logical operation (`True + True` is `True`).
    Logical(fn(bool, bool) -> bool),
    /// Computes in int64 instead.
    AsInt,
    /// Not defined (e.g. subtraction, as in NumPy and PyTorch).
    Unsupported,
}

trait BinaryOp {
    const NAME: &'static str;
    const ON_BOOL: OnBool;

    /// The dtype the op computes in, which is also its result dtype.
    fn compute_dtype(a: DType, b: DType) -> DType {
        a.promote(b)
    }

    fn apply<T: Numeric>(a: T, b: T) -> T;

    /// Rejects inputs the op isn't defined for, before computing anything.
    /// `rhs` has dtype `T`.
    fn check<T: Numeric>(_rhs: &Tensor) -> Result<()> {
        Ok(())
    }
}

/// Integer `//` and `%` by zero is an error (float division gives inf/NaN).
fn check_int_divisor<T: Numeric>(rhs: &Tensor) -> Result<()> {
    if !T::DTYPE.is_float() && rhs.any_element::<T>(|x| x == T::ZERO) {
        return Err(TensorError::DivisionByZero);
    }
    Ok(())
}

struct AddOp;
struct SubOp;
struct MulOp;
struct DivOp;
struct FloorDivOp;
struct RemOp;
struct PowOp;
struct MaximumOp;
struct MinimumOp;

impl BinaryOp for AddOp {
    const NAME: &'static str = "addition";
    const ON_BOOL: OnBool = OnBool::Logical(|a, b| a | b);
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.add(b)
    }
}

impl BinaryOp for SubOp {
    const NAME: &'static str = "subtraction (use ^ or logical_xor for bools)";
    const ON_BOOL: OnBool = OnBool::Unsupported;
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.sub(b)
    }
}

impl BinaryOp for MulOp {
    const NAME: &'static str = "multiplication";
    const ON_BOOL: OnBool = OnBool::Logical(|a, b| a & b);
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.mul(b)
    }
}

impl BinaryOp for DivOp {
    const NAME: &'static str = "division";
    const ON_BOOL: OnBool = OnBool::Unsupported;
    fn compute_dtype(a: DType, b: DType) -> DType {
        a.promote(b).to_float()
    }
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.div(b)
    }
}

impl BinaryOp for FloorDivOp {
    const NAME: &'static str = "floor division";
    const ON_BOOL: OnBool = OnBool::AsInt;
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.floor_div(b)
    }
    fn check<T: Numeric>(rhs: &Tensor) -> Result<()> {
        check_int_divisor::<T>(rhs)
    }
}

impl BinaryOp for RemOp {
    const NAME: &'static str = "remainder";
    const ON_BOOL: OnBool = OnBool::AsInt;
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.rem(b)
    }
    fn check<T: Numeric>(rhs: &Tensor) -> Result<()> {
        check_int_divisor::<T>(rhs)
    }
}

impl BinaryOp for PowOp {
    const NAME: &'static str = "power";
    const ON_BOOL: OnBool = OnBool::AsInt;
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.pow(b)
    }
    fn check<T: Numeric>(rhs: &Tensor) -> Result<()> {
        if !T::DTYPE.is_float() && rhs.any_element::<T>(|x| x < T::ZERO) {
            return Err(TensorError::InvalidArgument(
                "integers to negative integer powers are not allowed".into(),
            ));
        }
        Ok(())
    }
}

impl BinaryOp for MaximumOp {
    const NAME: &'static str = "maximum";
    const ON_BOOL: OnBool = OnBool::Logical(|a, b| a | b);
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.maximum(b)
    }
}

impl BinaryOp for MinimumOp {
    const NAME: &'static str = "minimum";
    const ON_BOOL: OnBool = OnBool::Logical(|a, b| a & b);
    fn apply<T: Numeric>(a: T, b: T) -> T {
        a.minimum(b)
    }
}

fn binary_dtype<O: BinaryOp>(a: DType, b: DType) -> Result<DType> {
    match (O::compute_dtype(a, b), O::ON_BOOL) {
        (DType::Bool, OnBool::AsInt) => Ok(DType::I64),
        (DType::Bool, OnBool::Unsupported) => Err(TensorError::UnsupportedDType {
            op: O::NAME,
            dtype: DType::Bool,
        }),
        (dtype, _) => Ok(dtype),
    }
}

trait CompareOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool;
}

struct EqOp;
struct NeOp;
struct LtOp;
struct LeOp;
struct GtOp;
struct GeOp;

impl CompareOp for EqOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a == b
    }
}
impl CompareOp for NeOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a != b
    }
}
impl CompareOp for LtOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a < b
    }
}
impl CompareOp for LeOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a <= b
    }
}
impl CompareOp for GtOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a > b
    }
}
impl CompareOp for GeOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a >= b
    }
}

trait BitwiseOp {
    const NAME: &'static str;
    fn apply<T: Bitwise>(a: T, b: T) -> T;
}

struct AndOp;
struct OrOp;
struct XorOp;

impl BitwiseOp for AndOp {
    const NAME: &'static str = "bitwise and";
    fn apply<T: Bitwise>(a: T, b: T) -> T {
        a.and(b)
    }
}
impl BitwiseOp for OrOp {
    const NAME: &'static str = "bitwise or";
    fn apply<T: Bitwise>(a: T, b: T) -> T {
        a.or(b)
    }
}
impl BitwiseOp for XorOp {
    const NAME: &'static str = "bitwise xor";
    fn apply<T: Bitwise>(a: T, b: T) -> T {
        a.xor(b)
    }
}

/// Functions that only make sense on floats; integer inputs become floats.
trait FloatOp {
    fn apply<T: Float>(x: T) -> T;
}

macro_rules! float_ops {
    ($($Op:ident => |$x:ident| $body:expr;)*) => {
        $(
            struct $Op;
            impl FloatOp for $Op {
                fn apply<T: Float>($x: T) -> T {
                    $body
                }
            }
        )*
    };
}

float_ops! {
    ExpOp => |x| x.exp();
    LogOp => |x| x.ln();
    SqrtOp => |x| x.sqrt();
    SinOp => |x| x.sin();
    CosOp => |x| x.cos();
    TanhOp => |x| x.tanh();
    SigmoidOp => |x| {
        let one = T::from_f64(1.0);
        one.div(one.add(x.neg().exp()))
    };
}

/// Rounding functions: floats are rounded, integers are already whole.
trait RoundOp {
    fn apply<T: Float>(x: T) -> T;
}

macro_rules! round_ops {
    ($($Op:ident => $method:ident;)*) => {
        $(
            struct $Op;
            impl RoundOp for $Op {
                fn apply<T: Float>(x: T) -> T {
                    x.$method()
                }
            }
        )*
    };
}

round_ops! {
    FloorOp => floor;
    CeilOp => ceil;
    RoundHalfEvenOp => round;
    TruncOp => trunc;
}

impl Tensor {
    // ---- broadcasting ----

    /// A view stretched to `shape` by giving broadcast axes stride 0. New
    /// leading axes may be added; `-1` keeps an existing axis's size.
    pub fn expand(&self, shape: &[isize]) -> Result<Tensor> {
        let bad = || {
            TensorError::InvalidArgument(format!(
                "cannot expand shape {:?} to {shape:?}",
                self.shape
            ))
        };
        if shape.len() < self.ndim() {
            return Err(bad());
        }
        let pad = shape.len() - self.ndim();
        let mut new_shape = Vec::with_capacity(shape.len());
        let mut new_strides = Vec::with_capacity(shape.len());
        for (axis, &want) in shape.iter().enumerate() {
            if axis < pad {
                // A brand-new axis: every step along it revisits the same data.
                new_shape.push(usize::try_from(want).map_err(|_| bad())?);
                new_strides.push(0);
                continue;
            }
            let have = self.shape[axis - pad];
            let stride = self.strides[axis - pad];
            let want = if want == -1 {
                have
            } else {
                usize::try_from(want).map_err(|_| bad())?
            };
            if want == have {
                new_shape.push(have);
                new_strides.push(stride);
            } else if have == 1 {
                new_shape.push(want);
                new_strides.push(0);
            } else {
                return Err(bad());
            }
        }
        Ok(self.with_layout(new_shape, new_strides, self.offset))
    }

    /// [`Tensor::expand`] with a concrete shape (NumPy's spelling).
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        let shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        self.expand(&shape)
    }

    /// Sets every element of this view to `value`, converted to this dtype.
    pub fn fill(&self, value: impl Into<Scalar>) -> Result<()> {
        let value = value.into();
        with_dtype!(self.dtype(), T => self.fill_with::<T>(value.to::<T>()))
    }

    // ---- arithmetic ----

    fn binary<O: BinaryOp>(&self, other: &Tensor) -> Result<Tensor> {
        let dtype = binary_dtype::<O>(self.dtype(), other.dtype())?;
        let (a, b) = (self.to_dtype(dtype), other.to_dtype(dtype));
        with_numeric!(dtype, T => {
            O::check::<T>(&b)?;
            a.zip_map::<T, T>(&b, O::apply::<T>)
        }, bool => match O::ON_BOOL {
            OnBool::Logical(f) => a.zip_map::<bool, bool>(&b, f),
            _ => unreachable!("binary_dtype only keeps Bool for logical ops"),
        })
    }

    /// In place: the result must fit this tensor's dtype (`int_tensor /= 2`
    /// is an error, as in NumPy and PyTorch).
    fn binary_assign<O: BinaryOp>(&self, other: &Tensor) -> Result<()> {
        let dtype = binary_dtype::<O>(self.dtype(), other.dtype())?;
        if dtype != self.dtype() {
            return Err(TensorError::InvalidArgument(format!(
                "result type {dtype} can't be cast to the desired output type {}",
                self.dtype()
            )));
        }
        let b = other.to_dtype(dtype);
        with_numeric!(dtype, T => {
            O::check::<T>(&b)?;
            self.zip_map_assign::<T>(&b, O::apply::<T>)
        }, bool => match O::ON_BOOL {
            OnBool::Logical(f) => self.zip_map_assign::<bool>(&b, f),
            _ => unreachable!("binary_dtype only keeps Bool for logical ops"),
        })
    }

    // ---- comparisons (always produce bool) ----

    fn compare<O: CompareOp>(&self, other: &Tensor) -> Result<Tensor> {
        let dtype = self.dtype().promote(other.dtype());
        let (a, b) = (self.to_dtype(dtype), other.to_dtype(dtype));
        with_dtype!(dtype, T => a.zip_map::<T, bool>(&b, O::apply::<T>))
    }

    // ---- bitwise and logical ----

    fn bitwise<O: BitwiseOp>(&self, other: &Tensor) -> Result<Tensor> {
        let dtype = self.dtype().promote(other.dtype());
        let (a, b) = (self.to_dtype(dtype), other.to_dtype(dtype));
        with_bitwise!(dtype, T => a.zip_map::<T, T>(&b, O::apply::<T>), float => {
            Err(TensorError::UnsupportedDType { op: O::NAME, dtype })
        })
    }

    fn bitwise_assign<O: BitwiseOp>(&self, other: &Tensor) -> Result<()> {
        let dtype = self.dtype();
        if other.dtype().promote(dtype) != dtype {
            return Err(TensorError::InvalidArgument(format!(
                "result type {} can't be cast to the desired output type {dtype}",
                other.dtype().promote(dtype)
            )));
        }
        let b = other.to_dtype(dtype);
        with_bitwise!(dtype, T => self.zip_map_assign::<T>(&b, O::apply::<T>), float => {
            Err(TensorError::UnsupportedDType { op: O::NAME, dtype })
        })
    }

    /// `~x`: flips bits of integers, negates bools.
    pub fn bitwise_not(&self) -> Result<Tensor> {
        let dtype = self.dtype();
        with_bitwise!(dtype, T => Ok(self.map_elements::<T, T>(<T as Bitwise>::not)), float => {
            Err(TensorError::UnsupportedDType { op: "bitwise not", dtype })
        })
    }

    /// Truthiness of every element, as a bool tensor.
    fn truthy(&self) -> Tensor {
        self.to_dtype(DType::Bool)
    }

    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor> {
        self.truthy().bitwise::<AndOp>(&other.truthy())
    }

    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor> {
        self.truthy().bitwise::<OrOp>(&other.truthy())
    }

    pub fn logical_xor(&self, other: &Tensor) -> Result<Tensor> {
        self.truthy().bitwise::<XorOp>(&other.truthy())
    }

    pub fn logical_not(&self) -> Tensor {
        self.truthy().map_elements::<bool, bool>(|x| !x)
    }

    // ---- unary math ----

    fn float_unary<O: FloatOp>(&self) -> Tensor {
        let dtype = self.dtype().to_float();
        let x = self.to_dtype(dtype);
        with_float!(dtype, T => x.map_elements::<T, T>(O::apply::<T>))
    }

    fn rounding<O: RoundOp>(&self) -> Tensor {
        if !self.dtype().is_float() {
            return self.deep_copy();
        }
        with_float!(self.dtype(), T => self.map_elements::<T, T>(O::apply::<T>))
    }

    pub fn neg(&self) -> Result<Tensor> {
        let dtype = self.dtype();
        with_numeric!(dtype, T => Ok(self.map_elements::<T, T>(<T as Numeric>::neg)), bool => {
            Err(TensorError::UnsupportedDType { op: "negation (use ~ or logical_not)", dtype })
        })
    }

    pub fn abs(&self) -> Tensor {
        with_numeric!(self.dtype(), T => self.map_elements::<T, T>(<T as Numeric>::abs), bool => self.deep_copy())
    }

    /// `max(x, 0)`; NaN passes through, as in PyTorch.
    pub fn relu(&self) -> Result<Tensor> {
        let dtype = self.dtype();
        with_numeric!(dtype, T => Ok(self.map_elements::<T, T>(|x| <T as Numeric>::maximum(x, T::ZERO))), bool => {
            Err(TensorError::UnsupportedDType { op: "relu", dtype })
        })
    }

    /// Limits elements to `[min, max]`; either bound may be omitted. NaN stays
    /// NaN. Bounds promote like any other operand (`int.clamp(0.5)` is float).
    pub fn clamp(&self, min: Option<&Tensor>, max: Option<&Tensor>) -> Result<Tensor> {
        let mut out = self.shallow_clone();
        if let Some(min) = min {
            out = out.maximum(min)?;
        }
        if let Some(max) = max {
            out = out.minimum(max)?;
        }
        Ok(out)
    }

    pub fn isnan(&self) -> Tensor {
        self.float_test(|x| x.is_nan(), false)
    }

    pub fn isinf(&self) -> Tensor {
        self.float_test(|x| x.is_infinite(), false)
    }

    pub fn isfinite(&self) -> Tensor {
        self.float_test(|x| !x.is_nan() && !x.is_infinite(), true)
    }

    /// A float property per element; integers and bools get `for_ints`.
    fn float_test(&self, test: impl Fn(f64) -> bool, for_ints: bool) -> Tensor {
        if !self.dtype().is_float() {
            return Tensor::full(&self.shape, for_ints, DType::Bool);
        }
        with_float!(self.dtype(), T => self.map_elements::<T, bool>(|x| test(x.to_f64())))
    }

    /// True when every pair of (broadcast) elements satisfies
    /// `|a - b| <= atol + rtol * |b|`. NaNs never compare close.
    pub fn allclose(&self, other: &Tensor, rtol: f64, atol: f64) -> Result<bool> {
        let (a, b) = (self.to_dtype(DType::F64), other.to_dtype(DType::F64));
        let close = a.zip_map::<f64, bool>(&b, |x, y| (x - y).abs() <= atol + rtol * y.abs())?;
        Ok(!close.any_element::<bool>(|ok| !ok))
    }
}

/// `cond ? a : b` elementwise, broadcasting all three. `cond` is read for
/// truthiness; `a` and `b` are promoted to a common dtype.
pub fn where_(cond: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dtype = a.dtype().promote(b.dtype());
    let (a, b) = (a.to_dtype(dtype), b.to_dtype(dtype));
    with_dtype!(dtype, T => {
        Tensor::zip3_map::<T>(&cond.truthy(), &a, &b, |c, x, y| if c { x } else { y })
    })
}

/// `name(&self, other)` and its in-place `name_` for each arithmetic op.
macro_rules! binary_methods {
    ($($(#[$doc:meta])* $name:ident, $name_inplace:ident => $Op:ty;)*) => {
        impl Tensor {
            $(
                $(#[$doc])*
                pub fn $name(&self, other: &Tensor) -> Result<Tensor> {
                    self.binary::<$Op>(other)
                }

                pub fn $name_inplace(&self, other: &Tensor) -> Result<()> {
                    self.binary_assign::<$Op>(other)
                }
            )*
        }
    };
}

binary_methods! {
    add, add_ => AddOp;
    sub, sub_ => SubOp;
    mul, mul_ => MulOp;
    /// True division: integers produce floats.
    div, div_ => DivOp;
    /// Python's `//`: rounds toward negative infinity.
    floor_divide, floor_divide_ => FloorDivOp;
    /// Python's `%`: the result takes the sign of the divisor.
    remainder, remainder_ => RemOp;
    pow, pow_ => PowOp;
    /// Elementwise max; NaN wins.
    maximum, maximum_ => MaximumOp;
    /// Elementwise min; NaN wins.
    minimum, minimum_ => MinimumOp;
}

macro_rules! simple_methods {
    ($kind:ident: $($name:ident => $Op:ty),* $(,)?) => {
        impl Tensor {
            $(pub fn $name(&self) -> Tensor {
                self.$kind::<$Op>()
            })*
        }
    };
}

simple_methods!(float_unary: exp => ExpOp, log => LogOp, sqrt => SqrtOp, sin => SinOp,
                cos => CosOp, tanh => TanhOp, sigmoid => SigmoidOp);
simple_methods!(rounding: floor => FloorOp, ceil => CeilOp, round => RoundHalfEvenOp,
                trunc => TruncOp);

macro_rules! compare_methods {
    ($($name:ident => $Op:ty),* $(,)?) => {
        impl Tensor {
            $(pub fn $name(&self, other: &Tensor) -> Result<Tensor> {
                self.compare::<$Op>(other)
            })*
        }
    };
}

compare_methods!(eq => EqOp, ne => NeOp, lt => LtOp, le => LeOp, gt => GtOp, ge => GeOp);

macro_rules! bitwise_methods {
    ($($name:ident, $name_inplace:ident => $Op:ty;)*) => {
        impl Tensor {
            $(
                pub fn $name(&self, other: &Tensor) -> Result<Tensor> {
                    self.bitwise::<$Op>(other)
                }

                pub fn $name_inplace(&self, other: &Tensor) -> Result<()> {
                    self.bitwise_assign::<$Op>(other)
                }
            )*
        }
    };
}

bitwise_methods! {
    bitwise_and, bitwise_and_ => AndOp;
    bitwise_or, bitwise_or_ => OrOp;
    bitwise_xor, bitwise_xor_ => XorOp;
}

// ---- operator overloading ----
//
// `&a + &b`, `a + b`, `&a * 2.0`, `1 - &a`, ... all work. Operators can't
// return a `Result`, so invalid operations (mismatched shapes, unsupported
// dtypes) panic; use the named methods (`a.add(&b)?`) to handle those as
// errors. (If `std::ops::Add` is imported, `a.add(&b)` on an owned tensor
// picks the trait method; write `Tensor::add(&a, &b)` there.) Rust scalars
// are 0-d tensors of their own type: `int_tensor * 2.5f32` is float32.

macro_rules! binary_operator {
    ($Trait:ident, $method:ident, $op:ident, $($scalar:ty),*) => {
        impl $Trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$op(self, rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }

        impl $Trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                <&Tensor as $Trait<&Tensor>>::$method(&self, &rhs)
            }
        }

        impl $Trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                <&Tensor as $Trait<&Tensor>>::$method(&self, rhs)
            }
        }

        impl $Trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                <&Tensor as $Trait<&Tensor>>::$method(self, &rhs)
            }
        }

        $(
            impl $Trait<$scalar> for &Tensor {
                type Output = Tensor;
                fn $method(self, rhs: $scalar) -> Tensor {
                    <&Tensor as $Trait<&Tensor>>::$method(self, &Tensor::scalar(rhs))
                }
            }

            impl $Trait<$scalar> for Tensor {
                type Output = Tensor;
                fn $method(self, rhs: $scalar) -> Tensor {
                    <&Tensor as $Trait<&Tensor>>::$method(&self, &Tensor::scalar(rhs))
                }
            }

            impl $Trait<&Tensor> for $scalar {
                type Output = Tensor;
                fn $method(self, rhs: &Tensor) -> Tensor {
                    <&Tensor as $Trait<&Tensor>>::$method(&Tensor::scalar(self), rhs)
                }
            }

            impl $Trait<Tensor> for $scalar {
                type Output = Tensor;
                fn $method(self, rhs: Tensor) -> Tensor {
                    <&Tensor as $Trait<&Tensor>>::$method(&Tensor::scalar(self), &rhs)
                }
            }
        )*
    };
}

binary_operator!(Add, add, add, f32, f64, i64);
binary_operator!(Sub, sub, sub, f32, f64, i64);
binary_operator!(Mul, mul, mul, f32, f64, i64);
binary_operator!(Div, div, div, f32, f64, i64);

macro_rules! assign_operator {
    ($Trait:ident, $method:ident, $op:ident, $($scalar:ty),*) => {
        impl $Trait<&Tensor> for Tensor {
            fn $method(&mut self, rhs: &Tensor) {
                self.$op(rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }

        $(impl $Trait<$scalar> for Tensor {
            fn $method(&mut self, rhs: $scalar) {
                self.$op(&Tensor::scalar(rhs))
                    .unwrap_or_else(|e| panic!("{e}"))
            }
        })*
    };
}

assign_operator!(AddAssign, add_assign, add_, f32, f64, i64);
assign_operator!(SubAssign, sub_assign, sub_, f32, f64, i64);
assign_operator!(MulAssign, mul_assign, mul_, f32, f64, i64);
assign_operator!(DivAssign, div_assign, div_, f32, f64, i64);

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        -&self
    }
}

#[cfg(test)]
mod tests {
    // Not `super::*`: importing the operator traits would make `a.add(&b)`
    // resolve to `Add::add` (by value) instead of the fallible method.
    use super::{broadcast_shapes, where_};
    use crate::dtype::{DType, Scalar};
    use crate::error::TensorError;
    use crate::tensor::Tensor;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    fn ints(data: &[i64]) -> Tensor {
        Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap()
    }

    fn bools(data: &[bool]) -> Tensor {
        Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap()
    }

    fn v(x: &Tensor) -> Vec<f32> {
        x.to_vec::<f32>()
    }

    /// 0, 1, 2, ... as a float32 tensor of the given shape.
    fn range(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        t(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), shape)
    }

    #[test]
    fn broadcast_shape_rules() {
        assert_eq!(broadcast_shapes(&[3, 1], &[4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(
            broadcast_shapes(&[5, 1, 4], &[3, 1]).unwrap(),
            vec![5, 3, 4]
        );
        assert_eq!(broadcast_shapes(&[0], &[1]).unwrap(), vec![0]);
        assert!(matches!(
            broadcast_shapes(&[2, 3], &[4]),
            Err(TensorError::BroadcastMismatch { .. })
        ));
    }

    #[test]
    fn expand_is_a_zero_stride_view() {
        let x = t(&[1.0, 2.0, 3.0], &[3, 1]);
        let e = x.expand(&[2, -1, 4]).unwrap();
        assert_eq!(e.shape(), &[2, 3, 4]);
        assert_eq!(e.strides(), &[0, 1, 0]);
        assert!(e.shares_storage(&x));
        assert_eq!(e.get(&[1, 2, 3]).unwrap().to_f64(), 3.0);
        assert!(x.expand(&[3, 2, 1]).is_err());
        assert!(x.expand(&[4]).is_err());
        assert!(x.expand(&[-1, 2, 4]).is_err());
    }

    #[test]
    fn arithmetic_broadcasts() {
        let col = t(&[1.0, 2.0], &[2, 1]);
        let row = t(&[10.0, 20.0, 30.0], &[3]);
        let sum = col.add(&row).unwrap();
        assert_eq!(sum.shape(), &[2, 3]);
        assert_eq!(v(&sum), vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
        assert_eq!(
            v(&row.div(&Tensor::scalar(10.0f32)).unwrap()),
            vec![1.0, 2.0, 3.0]
        );
        assert!(row.add(&t(&[1.0, 2.0], &[2])).is_err());
    }

    #[test]
    fn result_dtypes_follow_pytorch() {
        let i = ints(&[1, 2]);
        let f = t(&[1.0, 2.0], &[2]);
        let b = bools(&[true, false]);
        let i32s = Tensor::from_vec(vec![1i32, 2], &[2]).unwrap();
        let f64s = Tensor::from_vec(vec![1.0f64, 2.0], &[2]).unwrap();

        assert_eq!(i.add(&i).unwrap().dtype(), DType::I64);
        assert_eq!(i.add(&f).unwrap().dtype(), DType::F32);
        assert_eq!(i32s.add(&i).unwrap().dtype(), DType::I64);
        assert_eq!(f.add(&f64s).unwrap().dtype(), DType::F64);
        assert_eq!(b.add(&i).unwrap().dtype(), DType::I64);
        assert_eq!(
            i.div(&i).unwrap().dtype(),
            DType::F32,
            "true division gives floats"
        );
        assert_eq!(f64s.div(&i).unwrap().dtype(), DType::F64);
        assert_eq!(i.exp().dtype(), DType::F32);
        assert_eq!(i.lt(&f).unwrap().dtype(), DType::Bool);
        let all_true = bools(&[true, true]);
        assert_eq!(
            all_true.floor_divide(&all_true).unwrap().dtype(),
            DType::I64
        );
    }

    #[test]
    fn integer_arithmetic() {
        let a = ints(&[7, -7, 3]);
        let b = ints(&[2, 2, -2]);
        assert_eq!(a.floor_divide(&b).unwrap().to_vec::<i64>(), vec![3, -4, -2]);
        assert_eq!(a.remainder(&b).unwrap().to_vec::<i64>(), vec![1, 1, -1]);
        assert_eq!(
            a.pow(&ints(&[2, 3, 0])).unwrap().to_vec::<i64>(),
            vec![49, -343, 1]
        );
        assert_eq!(v(&a.div(&b).unwrap()), vec![3.5, -3.5, -1.5]);
        assert!(matches!(
            a.floor_divide(&ints(&[1, 0, 1])),
            Err(TensorError::DivisionByZero)
        ));
        assert!(matches!(
            a.remainder(&Tensor::scalar(0i64)),
            Err(TensorError::DivisionByZero)
        ));
        assert!(a.pow(&ints(&[1, -1, 1])).is_err());
        // Floats divide by zero the IEEE way instead.
        assert_eq!(
            v(&t(&[1.0], &[1]).floor_divide(&t(&[0.0], &[1])).unwrap()),
            vec![f32::INFINITY]
        );
        // Overflow wraps.
        let max = Tensor::from_vec(vec![i32::MAX], &[1]).unwrap();
        assert_eq!(
            max.add(&Tensor::from_vec(vec![1i32], &[1]).unwrap())
                .unwrap()
                .to_vec::<i32>(),
            vec![i32::MIN]
        );
    }

    #[test]
    fn bool_arithmetic() {
        let a = bools(&[true, true, false]);
        let b = bools(&[true, false, false]);
        assert_eq!(a.add(&b).unwrap().to_vec::<bool>(), vec![true, true, false]);
        assert_eq!(
            a.mul(&b).unwrap().to_vec::<bool>(),
            vec![true, false, false]
        );
        assert!(matches!(
            a.sub(&b),
            Err(TensorError::UnsupportedDType { .. })
        ));
        assert!(a.neg().is_err());
        assert_eq!(
            a.add(&ints(&[1, 1, 1])).unwrap().to_vec::<i64>(),
            vec![2, 2, 1]
        );
    }

    #[test]
    fn comparisons() {
        let a = t(&[1.0, 2.0, f32::NAN], &[3]);
        let b = Tensor::scalar(2i64);
        assert_eq!(a.lt(&b).unwrap().to_vec::<bool>(), vec![true, false, false]);
        assert_eq!(a.ge(&b).unwrap().to_vec::<bool>(), vec![false, true, false]);
        assert_eq!(
            a.eq(&a).unwrap().to_vec::<bool>(),
            vec![true, true, false],
            "NaN != NaN"
        );
        assert_eq!(a.ne(&b).unwrap().to_vec::<bool>(), vec![true, false, true]);
        assert_eq!(
            ints(&[1, 2])
                .eq(&t(&[1.0, 2.5], &[2]))
                .unwrap()
                .to_vec::<bool>(),
            vec![true, false]
        );
    }

    #[test]
    fn bitwise_and_logical() {
        let a = ints(&[0b1100, 5]);
        let b = ints(&[0b1010, 3]);
        assert_eq!(a.bitwise_and(&b).unwrap().to_vec::<i64>(), vec![0b1000, 1]);
        assert_eq!(a.bitwise_or(&b).unwrap().to_vec::<i64>(), vec![0b1110, 7]);
        assert_eq!(a.bitwise_xor(&b).unwrap().to_vec::<i64>(), vec![0b0110, 6]);
        assert_eq!(a.bitwise_not().unwrap().to_vec::<i64>(), vec![!0b1100, !5]);
        assert!(t(&[1.0], &[1]).bitwise_and(&t(&[1.0], &[1])).is_err());

        let m = bools(&[true, false]);
        assert_eq!(m.bitwise_not().unwrap().to_vec::<bool>(), vec![false, true]);
        let x = t(&[0.0, 2.5], &[2]);
        assert_eq!(
            x.logical_and(&m).unwrap().to_vec::<bool>(),
            vec![false, false]
        );
        assert_eq!(x.logical_or(&m).unwrap().to_vec::<bool>(), vec![true, true]);
        assert_eq!(x.logical_not().to_vec::<bool>(), vec![true, false]);
    }

    #[test]
    fn operators() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        assert_eq!(v(&(&a + &b)), vec![4.0, 6.0]);
        assert_eq!(v(&(&a * 2.0f32)), vec![2.0, 4.0]);
        assert_eq!(v(&(1.0f32 - &a)), vec![0.0, -1.0]);
        assert_eq!(v(&(-&a)), vec![-1.0, -2.0]);
        assert_eq!(v(&(a.shallow_clone() / b)), vec![1.0 / 3.0, 0.5]);
        assert_eq!((&ints(&[1, 2]) * 2i64).to_vec::<i64>(), vec![2, 4]);
        assert_eq!((&ints(&[1, 2]) * 0.5f32).dtype(), DType::F32);

        let mut c = a.shallow_clone();
        c += 10.0f32;
        c *= &t(&[2.0], &[1]);
        assert_eq!(
            v(&a),
            vec![22.0, 24.0],
            "in-place ops write through shared storage"
        );
    }

    #[test]
    #[should_panic(expected = "cannot be broadcast")]
    fn operator_panics_on_bad_shapes() {
        let _ = &t(&[1.0, 2.0], &[2]) + &t(&[1.0, 2.0, 3.0], &[3]);
    }

    #[test]
    fn in_place_ops_write_through_views() {
        let x = Tensor::zeros(&[2, 3], DType::F32);
        x.select(0, 1)
            .unwrap()
            .add_(&t(&[1.0, 2.0, 3.0], &[3]))
            .unwrap();
        x.slice(1, Some(2), None, 1)
            .unwrap()
            .sub_(&Tensor::scalar(5.0f32))
            .unwrap();
        assert_eq!(v(&x), vec![0.0, 0.0, -5.0, 1.0, 2.0, -2.0]);
        // The right side may broadcast up to self, but self can't grow.
        assert!(matches!(
            t(&[1.0], &[1]).add_(&x),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn in_place_ops_keep_the_dtype() {
        let i = ints(&[1, 2]);
        i.add_(&Tensor::scalar(true)).unwrap();
        assert_eq!(i.to_vec::<i64>(), vec![2, 3]);
        assert!(
            i.add_(&t(&[0.5, 0.5], &[2])).is_err(),
            "float result can't go in an int tensor"
        );
        assert!(
            i.div_(&ints(&[1, 1])).is_err(),
            "true division of ints is float"
        );
        let f = t(&[1.0, 2.0], &[2]);
        f.add_(&ints(&[1, 1])).unwrap();
        assert_eq!(v(&f), vec![2.0, 3.0]);
    }

    #[test]
    fn overlapping_in_place() {
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        x.add_(&x.slice(0, None, None, -1).unwrap()).unwrap();
        assert_eq!(v(&x), vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn unary_math() {
        let x = t(&[-1.0, 0.0, 4.0], &[3]);
        assert_eq!(v(&x.abs()), vec![1.0, 0.0, 4.0]);
        assert_eq!(v(&x.relu().unwrap()), vec![0.0, 0.0, 4.0]);
        let clamp = |lo: Option<f32>, hi: Option<f32>| {
            let (lo, hi) = (lo.map(Tensor::scalar), hi.map(Tensor::scalar));
            v(&x.clamp(lo.as_ref(), hi.as_ref()).unwrap())
        };
        assert_eq!(clamp(Some(-0.5), Some(2.0)), vec![-0.5, 0.0, 2.0]);
        assert_eq!(clamp(None, Some(0.0)), vec![-1.0, 0.0, 0.0]);
        assert_eq!(x.sigmoid().get(&[1]).unwrap().to_f64(), 0.5);
        assert!(x.sqrt().get(&[0]).unwrap().to_f64().is_nan());
        assert_eq!(v(&x.exp().log()), v(&x));
        assert!(
            t(&[f32::NAN], &[1])
                .relu()
                .unwrap()
                .item()
                .unwrap()
                .to_f64()
                .is_nan()
        );
        assert_eq!(ints(&[-3, 4]).abs().to_vec::<i64>(), vec![3, 4]);
        assert_eq!(ints(&[-3, 4]).relu().unwrap().to_vec::<i64>(), vec![0, 4]);
    }

    #[test]
    fn rounding() {
        let x = t(&[-1.5, -0.5, 0.5, 1.5, 2.7], &[5]);
        assert_eq!(v(&x.floor()), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(v(&x.ceil()), vec![-1.0, -0.0, 1.0, 2.0, 3.0]);
        assert_eq!(v(&x.round()), vec![-2.0, -0.0, 0.0, 2.0, 3.0]);
        assert_eq!(v(&x.trunc()), vec![-1.0, -0.0, 0.0, 1.0, 2.0]);
        assert_eq!(ints(&[3]).floor().dtype(), DType::I64);
    }

    #[test]
    fn nan_and_inf_tests() {
        let x = t(&[1.0, f32::NAN, f32::INFINITY], &[3]);
        assert_eq!(x.isnan().to_vec::<bool>(), vec![false, true, false]);
        assert_eq!(x.isinf().to_vec::<bool>(), vec![false, false, true]);
        assert_eq!(x.isfinite().to_vec::<bool>(), vec![true, false, false]);
        assert_eq!(ints(&[1]).isfinite().to_vec::<bool>(), vec![true]);
    }

    #[test]
    fn maximum_propagates_nan() {
        let a = t(&[1.0, f32::NAN, 3.0], &[3]);
        let m = v(&a.maximum(&Tensor::scalar(2.0f32)).unwrap());
        assert_eq!(m[0], 2.0);
        assert!(m[1].is_nan());
        assert_eq!(v(&a.minimum(&Tensor::scalar(2.0f32)).unwrap())[2], 2.0);
    }

    #[test]
    fn where_picks_elementwise() {
        let cond = bools(&[true, false, true]);
        let out = where_(&cond, &ints(&[1, 2, 3]), &Tensor::scalar(0.5f32)).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(v(&out), vec![1.0, 0.5, 3.0]);
        // Any dtype works as a condition (truthiness), and shared storage is fine.
        let x = t(&[0.0, 2.0], &[2]);
        assert_eq!(
            v(&where_(&x, &x, &Tensor::scalar(-1.0f32)).unwrap()),
            vec![-1.0, 2.0]
        );
        let m = bools(&[true, false]);
        assert_eq!(
            where_(&m, &m, &m.logical_not()).unwrap().to_vec::<bool>(),
            vec![true, true]
        );
    }

    #[test]
    fn allclose_tolerances() {
        let a = t(&[1.0, 2.0], &[2]);
        assert!(a.allclose(&t(&[1.0, 2.000001], &[2]), 1e-5, 1e-8).unwrap());
        assert!(!a.allclose(&t(&[1.0, 2.1], &[2]), 1e-5, 1e-8).unwrap());
        assert!(
            a.allclose(
                &Tensor::ones(&[3, 1], DType::F32).mul(&a).unwrap(),
                0.0,
                0.0
            )
            .unwrap()
        );
        assert!(
            !t(&[f32::NAN], &[1])
                .allclose(&t(&[f32::NAN], &[1]), 1.0, 1.0)
                .unwrap()
        );
        assert!(ints(&[1, 2]).allclose(&a, 0.0, 0.0).unwrap());
    }

    #[test]
    fn copy_from_broadcasts_and_converts() {
        let x = Tensor::zeros(&[2, 3], DType::F32);
        x.copy_from(&t(&[1.0, 2.0, 3.0], &[3])).unwrap();
        assert_eq!(v(&x), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert!(x.copy_from(&t(&[1.0, 2.0], &[2])).is_err());
        let i = Tensor::zeros(&[2], DType::I64);
        i.copy_from(&t(&[1.9, -1.9], &[2])).unwrap();
        assert_eq!(i.to_vec::<i64>(), vec![1, -1]);
    }

    /// Reads every element with `get`, one index at a time: slow but obviously right.
    fn reference_to_vec(x: &Tensor) -> Vec<f32> {
        let mut out = Vec::new();
        let mut index = vec![0usize; x.ndim()];
        for _ in 0..x.numel() {
            out.push(x.get(&index).unwrap().to_f64() as f32);
            for axis in (0..x.ndim()).rev() {
                index[axis] += 1;
                if index[axis] < x.shape()[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        out
    }

    /// Same-shape [2, 3, 4] views of one buffer, covering the layouts the
    /// strided walk has to handle.
    fn awkward_layouts() -> Vec<(&'static str, Tensor)> {
        let base = range(&[2, 3, 4]);
        let big = range(&[4, 6, 4]);
        vec![
            ("contiguous", base.shallow_clone()),
            ("permuted", range(&[3, 4, 2]).permute(&[2, 0, 1]).unwrap()),
            ("reversed axis", base.slice(1, None, None, -1).unwrap()),
            (
                "fully reversed",
                range(&[24])
                    .slice(0, None, None, -1)
                    .unwrap()
                    .view(&[2, 3, 4])
                    .unwrap(),
            ),
            (
                "strided slices",
                big.slice(0, None, None, 2)
                    .unwrap()
                    .slice(1, Some(1), None, 2)
                    .unwrap(),
            ),
            (
                "broadcast column",
                base.select(2, 1)
                    .unwrap()
                    .unsqueeze(-1)
                    .unwrap()
                    .expand(&[2, 3, 4])
                    .unwrap(),
            ),
            (
                "broadcast scalar",
                Tensor::scalar(7.0f32).expand(&[2, 3, 4]).unwrap(),
            ),
        ]
    }

    #[test]
    fn strided_reads_match_reference() {
        for (name, x) in awkward_layouts() {
            assert_eq!(v(&x), reference_to_vec(&x), "{name}");
        }
        assert_eq!(v(&Tensor::scalar(3.0f32)), vec![3.0]);
        assert!(v(&Tensor::zeros(&[3, 0, 2], DType::F32)).is_empty());
    }

    #[test]
    fn binary_ops_match_reference_for_every_layout_pair() {
        for (a_name, a) in awkward_layouts() {
            for (b_name, b) in awkward_layouts() {
                let expected: Vec<f32> = reference_to_vec(&a)
                    .into_iter()
                    .zip(reference_to_vec(&b))
                    .map(|(x, y)| x * 100.0 - y)
                    .collect();
                let got = a.zip_map::<f32, f32>(&b, |x, y| x * 100.0 - y).unwrap();
                assert_eq!(v(&got), expected, "{a_name} op {b_name}");
            }
        }
    }

    #[test]
    fn in_place_ops_match_reference_for_every_layout_pair() {
        let count = awkward_layouts().len();
        for i in 0..count {
            for (b_name, b) in awkward_layouts() {
                // Each call builds fresh buffers, so every case writes through
                // its own (possibly strided) destination view.
                let (a_name, a) = awkward_layouts().swap_remove(i);
                let expected: Vec<f32> = reference_to_vec(&a)
                    .into_iter()
                    .zip(reference_to_vec(&b))
                    .map(|(x, y)| x - 2.0 * y)
                    .collect();
                let result = a.zip_map_assign::<f32>(&b, |x, y| x - 2.0 * y);
                if a_name.starts_with("broadcast") {
                    assert!(result.is_err(), "{a_name} -= {b_name} should be rejected");
                } else {
                    result.unwrap();
                    assert_eq!(v(&a), expected, "{a_name} -= {b_name}");
                }
            }
        }
    }

    #[test]
    fn ops_on_shared_storage() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        // Both operands read the same lock.
        assert_eq!(v(&a.add(&a).unwrap()), vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(
            v(&a.mul(&a.slice(0, None, None, -1).unwrap()).unwrap()),
            vec![4.0, 6.0, 6.0, 4.0]
        );
        // Shifted overlap reads the original values, like NumPy.
        let dst = a.slice(0, Some(1), None, 1).unwrap();
        dst.add_(&a.slice(0, None, Some(-1), 1).unwrap()).unwrap();
        assert_eq!(v(&a), vec![1.0, 3.0, 5.0, 7.0]);
        a.add_(&a).unwrap();
        assert_eq!(v(&a), vec![2.0, 6.0, 10.0, 14.0]);
    }

    #[test]
    fn writing_into_an_expanded_tensor_is_an_error() {
        let x = t(&[1.0, 2.0], &[2, 1]);
        let e = x.expand(&[2, 3]).unwrap();
        assert!(matches!(
            e.add_(&Tensor::scalar(1.0f32)),
            Err(TensorError::InvalidArgument(_))
        ));
        assert!(e.copy_from(&Tensor::zeros(&[2, 3], DType::F32)).is_err());
        assert_eq!(v(&x), vec![1.0, 2.0], "nothing was written");
        // A length-1 axis with stride 0 (from None/unsqueeze) is fine.
        let row = Tensor::zeros(&[3], DType::F32)
            .index(&[crate::TensorIndex::NewAxis])
            .unwrap();
        row.add_(&Tensor::scalar(1.0f32)).unwrap();
        assert_eq!(v(&row), vec![1.0; 3]);
    }

    #[test]
    fn fill_through_strided_views() {
        let x = Tensor::zeros(&[3, 4], DType::F32);
        x.slice(1, None, None, -2).unwrap().fill(1).unwrap();
        x.select(0, 0).unwrap().fill(2.0).unwrap();
        assert_eq!(
            v(&x),
            vec![2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
        let flags = Tensor::zeros(&[2], DType::Bool);
        flags.fill(Scalar::Int(5)).unwrap();
        assert_eq!(flags.to_vec::<bool>(), vec![true, true]);
    }
}
