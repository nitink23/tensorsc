//! Elementwise math and broadcasting.
//!
//! Broadcasting follows NumPy: shapes are lined up from the right, and an axis
//! of length 1 (or a missing one) stretches to match the other side. The
//! stretch is a view with stride 0, so the one element is read repeatedly
//! instead of being copied.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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

/// NaN-propagating max/min, matching NumPy's `maximum`/`minimum`
/// (`f32::max` would silently drop the NaN).
fn max_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

fn min_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

impl Tensor {
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

    /// A new tensor with `f` applied to every element.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let data = self.to_vec().into_iter().map(f).collect();
        Tensor::from_vec(data, &self.shape).expect("length matches shape")
    }

    /// A new tensor combining `self` and `other` elementwise after broadcasting.
    pub fn zip_map(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
        let shape = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&shape)?.to_vec();
        let b = other.broadcast_to(&shape)?.to_vec();
        let data = a.into_iter().zip(b).map(|(x, y)| f(x, y)).collect();
        Tensor::from_vec(data, &shape)
    }

    /// In-place [`Tensor::zip_map`]: writes `f(self, other)` back through this
    /// view. `other` must broadcast to this tensor's shape (not the reverse).
    pub fn zip_map_assign(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<()> {
        let b = other
            .broadcast_to(&self.shape)
            .map_err(|_| TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            })?
            .to_vec();
        // Read both sides before taking the write lock: they may share storage.
        let a = self.to_vec();
        let positions = self.storage_positions();
        let mut data = self.storage.write();
        for ((p, x), y) in positions.into_iter().zip(a).zip(b) {
            data[p] = f(x, y);
        }
        Ok(())
    }

    /// Limits every element to `[min, max]`; either bound may be omitted.
    /// NaN stays NaN.
    pub fn clamp(&self, min: Option<f32>, max: Option<f32>) -> Tensor {
        self.map(|x| match (min, max) {
            (Some(lo), _) if x < lo => lo,
            (_, Some(hi)) if x > hi => hi,
            _ => x,
        })
    }

    /// True when every pair of (broadcast) elements satisfies
    /// `|a - b| <= atol + rtol * |b|`. NaNs never compare close.
    pub fn allclose(&self, other: &Tensor, rtol: f32, atol: f32) -> Result<bool> {
        let shape = broadcast_shapes(&self.shape, &other.shape)?;
        let a = self.broadcast_to(&shape)?.to_vec();
        let b = other.broadcast_to(&shape)?.to_vec();
        Ok(a.into_iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs()))
    }
}

/// `name(&self, other) -> Result<Tensor>` plus the in-place `name_`.
macro_rules! binary_ops {
    ($($name:ident, $name_inplace:ident => $f:expr;)*) => {
        impl Tensor {
            $(
                pub fn $name(&self, other: &Tensor) -> Result<Tensor> {
                    self.zip_map(other, $f)
                }

                pub fn $name_inplace(&self, other: &Tensor) -> Result<()> {
                    self.zip_map_assign(other, $f)
                }
            )*
        }
    };
}

binary_ops! {
    add, add_ => |a, b| a + b;
    sub, sub_ => |a, b| a - b;
    mul, mul_ => |a, b| a * b;
    div, div_ => |a, b| a / b;
    pow, pow_ => f32::powf;
    maximum, maximum_ => max_nan;
    minimum, minimum_ => min_nan;
}

macro_rules! unary_ops {
    ($($name:ident => $f:expr;)*) => {
        impl Tensor {
            $(
                pub fn $name(&self) -> Tensor {
                    self.map($f)
                }
            )*
        }
    };
}

unary_ops! {
    neg => |x| -x;
    abs => f32::abs;
    exp => f32::exp;
    log => f32::ln;
    sqrt => f32::sqrt;
    sin => f32::sin;
    cos => f32::cos;
    tanh => f32::tanh;
    sigmoid => |x| 1.0 / (1.0 + (-x).exp());
    // Written as a comparison so NaN passes through, as in PyTorch.
    relu => |x| if x < 0.0 { 0.0 } else { x };
}

// ---- operator overloading ----
//
// `&a + &b`, `a + b`, `&a * 2.0`, `1.0 - &a`, ... all work. Operators can't
// return a `Result`, so incompatible shapes panic; use the named methods
// (`a.add(&b)?`) to handle that as an error instead. (If `std::ops::Add` is
// imported, `a.add(&b)` on an owned tensor picks the trait method; write
// `Tensor::add(&a, &b)` there.)

macro_rules! binary_operator {
    ($Trait:ident, $method:ident, $op:ident) => {
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

        impl $Trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                <&Tensor as $Trait<&Tensor>>::$method(self, &Tensor::scalar(rhs))
            }
        }

        impl $Trait<f32> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                <&Tensor as $Trait<&Tensor>>::$method(&self, &Tensor::scalar(rhs))
            }
        }

        impl $Trait<&Tensor> for f32 {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                <&Tensor as $Trait<&Tensor>>::$method(&Tensor::scalar(self), rhs)
            }
        }

        impl $Trait<Tensor> for f32 {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                <&Tensor as $Trait<&Tensor>>::$method(&Tensor::scalar(self), &rhs)
            }
        }
    };
}

binary_operator!(Add, add, add);
binary_operator!(Sub, sub, sub);
binary_operator!(Mul, mul, mul);
binary_operator!(Div, div, div);

macro_rules! assign_operator {
    ($Trait:ident, $method:ident, $op:ident) => {
        impl $Trait<&Tensor> for Tensor {
            fn $method(&mut self, rhs: &Tensor) {
                self.$op(rhs).unwrap_or_else(|e| panic!("{e}"))
            }
        }

        impl $Trait<f32> for Tensor {
            fn $method(&mut self, rhs: f32) {
                self.$op(&Tensor::scalar(rhs))
                    .unwrap_or_else(|e| panic!("{e}"))
            }
        }
    };
}

assign_operator!(AddAssign, add_assign, add_);
assign_operator!(SubAssign, sub_assign, sub_);
assign_operator!(MulAssign, mul_assign, mul_);
assign_operator!(DivAssign, div_assign, div_);

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    // Not `super::*`: importing the operator traits would make `a.add(&b)`
    // resolve to `Add::add` (by value) instead of the fallible method.
    use super::broadcast_shapes;
    use crate::error::TensorError;
    use crate::tensor::Tensor;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
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
        assert_eq!(e.get(&[1, 2, 3]).unwrap(), 3.0);
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
        assert_eq!(sum.to_vec(), vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
        assert_eq!(
            row.div(&Tensor::scalar(10.0)).unwrap().to_vec(),
            vec![1.0, 2.0, 3.0]
        );
        assert!(row.add(&t(&[1.0, 2.0], &[2])).is_err());
    }

    #[test]
    fn operators() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        assert_eq!((&a + &b).to_vec(), vec![4.0, 6.0]);
        assert_eq!((&a * 2.0).to_vec(), vec![2.0, 4.0]);
        assert_eq!((1.0 - &a).to_vec(), vec![0.0, -1.0]);
        assert_eq!((-&a).to_vec(), vec![-1.0, -2.0]);
        assert_eq!((a.shallow_clone() / b).to_vec(), vec![1.0 / 3.0, 0.5]);

        let mut c = a.shallow_clone();
        c += 10.0;
        c *= &t(&[2.0], &[1]);
        assert_eq!(
            a.to_vec(),
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
        let x = Tensor::zeros(&[2, 3]);
        x.select(0, 1)
            .unwrap()
            .add_(&t(&[1.0, 2.0, 3.0], &[3]))
            .unwrap();
        x.slice(1, Some(2), None, 1)
            .unwrap()
            .sub_(&Tensor::scalar(5.0))
            .unwrap();
        assert_eq!(x.to_vec(), vec![0.0, 0.0, -5.0, 1.0, 2.0, -2.0]);
        // The right side may broadcast up to self, but self can't grow.
        assert!(matches!(
            t(&[1.0], &[1]).add_(&x),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn overlapping_in_place() {
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        x.add_(&x.slice(0, None, None, -1).unwrap()).unwrap();
        assert_eq!(x.to_vec(), vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn unary_math() {
        let x = t(&[-1.0, 0.0, 4.0], &[3]);
        assert_eq!(x.abs().to_vec(), vec![1.0, 0.0, 4.0]);
        assert_eq!(x.relu().to_vec(), vec![0.0, 0.0, 4.0]);
        assert_eq!(
            x.clamp(Some(-0.5), Some(2.0)).to_vec(),
            vec![-0.5, 0.0, 2.0]
        );
        assert_eq!(x.clamp(None, Some(0.0)).to_vec(), vec![-1.0, 0.0, 0.0]);
        assert_eq!(x.sigmoid().get(&[1]).unwrap(), 0.5);
        assert!(x.sqrt().get(&[0]).unwrap().is_nan());
        assert_eq!(x.exp().log().to_vec(), x.to_vec());
        assert!(t(&[f32::NAN], &[1]).relu().item().unwrap().is_nan());
    }

    #[test]
    fn maximum_propagates_nan() {
        let a = t(&[1.0, f32::NAN, 3.0], &[3]);
        let m = a.maximum(&Tensor::scalar(2.0)).unwrap().to_vec();
        assert_eq!(m[0], 2.0);
        assert!(m[1].is_nan());
        assert_eq!(a.minimum(&Tensor::scalar(2.0)).unwrap().to_vec()[2], 2.0);
    }

    #[test]
    fn allclose_tolerances() {
        let a = t(&[1.0, 2.0], &[2]);
        assert!(a.allclose(&t(&[1.0, 2.000001], &[2]), 1e-5, 1e-8).unwrap());
        assert!(!a.allclose(&t(&[1.0, 2.1], &[2]), 1e-5, 1e-8).unwrap());
        assert!(
            a.allclose(&Tensor::ones(&[3, 1]).mul(&a).unwrap(), 0.0, 0.0)
                .unwrap()
        );
        assert!(
            !t(&[f32::NAN], &[1])
                .allclose(&t(&[f32::NAN], &[1]), 1.0, 1.0)
                .unwrap()
        );
    }

    #[test]
    fn copy_from_broadcasts() {
        let x = Tensor::zeros(&[2, 3]);
        x.copy_from(&t(&[1.0, 2.0, 3.0], &[3])).unwrap();
        assert_eq!(x.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert!(x.copy_from(&t(&[1.0, 2.0], &[2])).is_err());
    }
}
