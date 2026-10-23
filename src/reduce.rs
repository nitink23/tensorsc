//! Reductions: `sum`, `prod`, `mean`, `max`, `min`, `argmax`, `argmin`,
//! `all` and `any`, over every element or over chosen axes.
//!
//! One kernel serves them all. The input is walked once in memory order,
//! next to an "output view" laid over the input's shape with stride 0 on the
//! reduced axes. `for_each_row` then yields two kinds of rows: ones where the
//! output doesn't move (a reduced axis is innermost: fold the whole row into
//! one slot) and ones where it does (a kept axis is innermost: accumulate the
//! row elementwise into a row of outputs). Both are tight loops, and the input
//! is always read in cache order.
//!
//! Each output's inputs are visited in row-major order of the reduced axes,
//! which is what gives `argmax` its "first occurrence" index.

use std::marker::PhantomData;

use crate::autograd::{Saved, constant, fit, is_grad_enabled, record};
use crate::dtype::{DType, Element, with_dtype};
use crate::error::{Result, TensorError};
use crate::ops::where_;
use crate::strided::{for_each_row, positions};
use crate::tensor::{Tensor, contiguous_strides};
use crate::views::normalize_axis;

/// One way of folding many elements of type `T` into one output.
trait Reducer<T: Element> {
    type Acc: Copy;
    type Out: Element;

    fn init() -> Self::Acc;
    fn fold(acc: Self::Acc, x: T) -> Self::Acc;
    fn fold_row(acc: Self::Acc, row: &[T]) -> Self::Acc {
        row.iter().fold(acc, |a, &x| Self::fold(a, x))
    }
    /// `count` is how many elements went into `acc`.
    fn finish(acc: Self::Acc, count: usize) -> Self::Out;
}

/// Sums a row in f64 across 8 independent lanes: the lanes let the compiler
/// vectorize, and both they and the f64 accumulator keep float32 sums from
/// drifting the way a single f32 running total does.
fn sum_row_f64<T: Element>(acc: f64, row: &[T]) -> f64 {
    let mut lanes = [0f64; 8];
    let chunks = row.chunks_exact(8);
    let tail: f64 = chunks.remainder().iter().map(|x| x.to_f64()).sum();
    for chunk in chunks {
        for (lane, x) in lanes.iter_mut().zip(chunk) {
            *lane += x.to_f64();
        }
    }
    acc + lanes.iter().sum::<f64>() + tail
}

/// Float sum, accumulated in f64 and returned as `O`.
struct SumFloat<O>(PhantomData<O>);
/// Integer and bool sum, in wrapping int64 (as in PyTorch).
struct SumInt;
/// Mean, accumulated in f64 and returned as float `O`.
struct Mean<O>(PhantomData<O>);
struct ProdFloat<O>(PhantomData<O>);
struct ProdInt;
struct Max;
struct Min;
struct ArgMax;
struct ArgMin;
struct All;
struct Any;

impl<T: Element, O: Element> Reducer<T> for SumFloat<O> {
    type Acc = f64;
    type Out = O;
    fn init() -> f64 {
        0.0
    }
    fn fold(acc: f64, x: T) -> f64 {
        acc + x.to_f64()
    }
    fn fold_row(acc: f64, row: &[T]) -> f64 {
        sum_row_f64(acc, row)
    }
    fn finish(acc: f64, _: usize) -> O {
        O::from_f64(acc)
    }
}

impl<T: Element> Reducer<T> for SumInt {
    type Acc = i64;
    type Out = i64;
    fn init() -> i64 {
        0
    }
    fn fold(acc: i64, x: T) -> i64 {
        acc.wrapping_add(x.to_i64())
    }
    fn finish(acc: i64, _: usize) -> i64 {
        acc
    }
}

impl<T: Element, O: Element> Reducer<T> for Mean<O> {
    type Acc = f64;
    type Out = O;
    fn init() -> f64 {
        0.0
    }
    fn fold(acc: f64, x: T) -> f64 {
        acc + x.to_f64()
    }
    fn fold_row(acc: f64, row: &[T]) -> f64 {
        sum_row_f64(acc, row)
    }
    /// The mean of nothing is NaN, as in NumPy and PyTorch.
    fn finish(acc: f64, count: usize) -> O {
        O::from_f64(acc / count as f64)
    }
}

impl<T: Element, O: Element> Reducer<T> for ProdFloat<O> {
    type Acc = f64;
    type Out = O;
    fn init() -> f64 {
        1.0
    }
    fn fold(acc: f64, x: T) -> f64 {
        acc * x.to_f64()
    }
    fn finish(acc: f64, _: usize) -> O {
        O::from_f64(acc)
    }
}

impl<T: Element> Reducer<T> for ProdInt {
    type Acc = i64;
    type Out = i64;
    fn init() -> i64 {
        1
    }
    fn fold(acc: i64, x: T) -> i64 {
        acc.wrapping_mul(x.to_i64())
    }
    fn finish(acc: i64, _: usize) -> i64 {
        acc
    }
}

#[allow(clippy::eq_op)]
fn is_nan<T: Element>(x: T) -> bool {
    // Only NaN is unequal to itself; never true for integers or bools.
    x != x
}

/// Should `x` replace `best` as the maximum (`greater`) or minimum? NaN wins
/// and, once found, stays (as in NumPy's `max` and `argmax`). Ties keep the
/// earlier element.
fn replaces<T: Element>(best: T, x: T, greater: bool) -> bool {
    if is_nan(best) {
        return false;
    }
    is_nan(x) || if greater { x > best } else { x < best }
}

macro_rules! extreme_reducers {
    ($($Op:ident, $Arg:ident => $greater:expr;)*) => {
        $(
            impl<T: Element> Reducer<T> for $Op {
                type Acc = Option<T>;
                type Out = T;
                fn init() -> Option<T> {
                    None
                }
                fn fold(acc: Option<T>, x: T) -> Option<T> {
                    match acc {
                        Some(best) if !replaces(best, x, $greater) => Some(best),
                        _ => Some(x),
                    }
                }
                /// The same value as folding one by one, computed so the
                /// compiler can vectorize it: 8 independent running
                /// maxima/minima (like `sum_row_f64`'s lanes), plus a separate
                /// check for NaN, which wins if present.
                fn fold_row(acc: Option<T>, row: &[T]) -> Option<T> {
                    let pick = |best: T, x: T| {
                        let better = if $greater { x > best } else { x < best };
                        if better { x } else { best }
                    };
                    let start = acc.or_else(|| row.first().copied())?;
                    if is_nan(start) {
                        return Some(start);
                    }
                    let (mut lanes, mut nans) = ([start; 8], [false; 8]);
                    let chunks = row.chunks_exact(8);
                    let mut best = start;
                    let mut nan = false;
                    for &x in chunks.remainder() {
                        nan |= is_nan(x);
                        best = pick(best, x);
                    }
                    for chunk in chunks {
                        for ((lane, lane_nan), &x) in lanes.iter_mut().zip(&mut nans).zip(chunk) {
                            *lane_nan |= is_nan(x);
                            *lane = pick(*lane, x);
                        }
                    }
                    let best = lanes.into_iter().fold(best, pick);
                    Some(if nan || nans.contains(&true) { T::from_f64(f64::NAN) } else { best })
                }
                fn finish(acc: Option<T>, _: usize) -> T {
                    acc.expect("empty reductions are rejected up front")
                }
            }

            impl<T: Element> Reducer<T> for $Arg {
                /// (best value, its index, elements seen so far)
                type Acc = (Option<T>, i64, i64);
                type Out = i64;
                fn init() -> Self::Acc {
                    (None, 0, 0)
                }
                fn fold((best, best_at, seen): Self::Acc, x: T) -> Self::Acc {
                    match best {
                        Some(b) if !replaces(b, x, $greater) => (best, best_at, seen + 1),
                        _ => (Some(x), seen, seen + 1),
                    }
                }
                fn finish((_, best_at, _): Self::Acc, _: usize) -> i64 {
                    best_at
                }
            }
        )*
    };
}

extreme_reducers! {
    Max, ArgMax => true;
    Min, ArgMin => false;
}

impl<T: Element> Reducer<T> for All {
    type Acc = bool;
    type Out = bool;
    fn init() -> bool {
        true
    }
    fn fold(acc: bool, x: T) -> bool {
        acc && x.to_bool()
    }
    fn finish(acc: bool, _: usize) -> bool {
        acc
    }
}

impl<T: Element> Reducer<T> for Any {
    type Acc = bool;
    type Out = bool;
    fn init() -> bool {
        false
    }
    fn fold(acc: bool, x: T) -> bool {
        acc || x.to_bool()
    }
    fn finish(acc: bool, _: usize) -> bool {
        acc
    }
}

/// Which axes a reduction runs over.
#[derive(Clone)]
struct Plan {
    /// `reduced[axis]` is true for axes being reduced.
    reduced: Vec<bool>,
    /// The output shape with reduced axes kept as length 1.
    kept_shape: Vec<usize>,
    /// How many input elements feed each output.
    count: usize,
}

impl Tensor {
    /// `axes: None` reduces everything. Negative axes count from the end;
    /// repeating an axis is an error.
    fn plan(&self, axes: Option<&[isize]>) -> Result<Plan> {
        let mut reduced = vec![axes.is_none(); self.ndim()];
        for &axis in axes.unwrap_or(&[]) {
            let axis = normalize_axis(axis, self.ndim())?;
            if std::mem::replace(&mut reduced[axis], true) {
                return Err(TensorError::InvalidArgument(format!(
                    "axis {axis} appears more than once"
                )));
            }
        }
        let kept_shape = self
            .shape
            .iter()
            .zip(&reduced)
            .map(|(&dim, &r)| if r { 1 } else { dim })
            .collect();
        let count = self
            .shape
            .iter()
            .zip(&reduced)
            .filter(|&(_, &r)| r)
            .map(|(&dim, _)| dim)
            .product();
        Ok(Plan {
            reduced,
            kept_shape,
            count,
        })
    }

    fn reduce<T: Element, R: Reducer<T>>(&self, plan: &Plan, keepdim: bool) -> Result<Tensor> {
        // A contiguous full reduction is one fold into one scalar. Avoid an
        // accumulator allocation and building an output-stride layout for it.
        if plan.reduced.iter().all(|&r| r) && self.is_contiguous() {
            let data = self.storage.read::<T>();
            let value = R::finish(
                R::fold_row(R::init(), &data[self.offset..self.offset + self.numel()]),
                plan.count,
            );
            let shape = if keepdim {
                plan.kept_shape.as_slice()
            } else {
                &[]
            };
            return Tensor::from_vec(vec![value], shape);
        }
        let out_numel: usize = plan.kept_shape.iter().product();
        let mut acc = vec![R::init(); out_numel];
        // The output laid over the input's shape: stride 0 along reduced axes.
        let mut out_strides = contiguous_strides(&plan.kept_shape);
        for (stride, &r) in out_strides.iter_mut().zip(&plan.reduced) {
            if r {
                *stride = 0;
            }
        }
        let data = self.storage.read::<T>();
        for_each_row(
            &self.shape,
            [&out_strides, &self.strides],
            [0, self.offset],
            |[po, pi], len, [so, si]| match (so, si) {
                (0, 1) => acc[po] = R::fold_row(acc[po], &data[pi..pi + len]),
                (0, _) => {
                    acc[po] = positions(pi, len, si).fold(acc[po], |a, i| R::fold(a, data[i]))
                }
                // A row of inputs into a row of outputs (e.g. column sums).
                (1, 1) => {
                    for (a, &x) in acc[po..po + len].iter_mut().zip(&data[pi..pi + len]) {
                        *a = R::fold(*a, x);
                    }
                }
                _ => {
                    for (o, i) in positions(po, len, so).zip(positions(pi, len, si)) {
                        acc[o] = R::fold(acc[o], data[i]);
                    }
                }
            },
        );
        let out: Vec<R::Out> = acc.into_iter().map(|a| R::finish(a, plan.count)).collect();
        let shape: Vec<usize> = if keepdim {
            plan.kept_shape.clone()
        } else {
            self.shape
                .iter()
                .zip(&plan.reduced)
                .filter(|&(_, &r)| !r)
                .map(|(&dim, _)| dim)
                .collect()
        };
        Tensor::from_vec(out, &shape)
    }

    /// Max/min have no value for zero elements, so reducing an empty axis is
    /// an error (as in NumPy), unless there are no outputs either.
    fn ensure_nonempty(&self, plan: &Plan, op: &str) -> Result<()> {
        if plan.count == 0 && plan.kept_shape.iter().product::<usize>() > 0 {
            return Err(TensorError::InvalidArgument(format!(
                "{op} of an empty tensor (or along an empty axis) has no value"
            )));
        }
        Ok(())
    }

    /// Records a reduction whose input gradient is `local(g spread back
    /// over the input, input, output spread back)`.
    fn record_reduction(
        &self,
        out: Tensor,
        plan: Plan,
        name: &'static str,
        local: fn(&Tensor, &Tensor, &Tensor, &Plan) -> Result<Tensor>,
    ) -> Tensor {
        record(out, [self], name, |out, _| {
            let (x, out) = (Saved::new(self), Saved::new(out));
            Box::new(move |g| {
                let shape = x.shape();
                let grad = local(
                    &spread(g, &plan, shape)?,
                    x.get()?,
                    &spread(out.get()?, &plan, shape)?,
                    &plan,
                )?;
                Ok(vec![Some(fit(&grad, &x)?)])
            })
        })
    }

    /// Sum over `axes` (all if `None`). Floats keep their dtype; integers
    /// and bools sum to int64.
    pub fn sum(&self, axes: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
        let plan = self.plan(axes)?;
        let out = with_dtype!(self.dtype(), T => match self.dtype() {
            DType::F32 => self.reduce::<T, SumFloat<f32>>(&plan, keepdim),
            DType::F64 => self.reduce::<T, SumFloat<f64>>(&plan, keepdim),
            _ => self.reduce::<T, SumInt>(&plan, keepdim),
        })?;
        Ok(self.record_reduction(out, plan, "SumBackward", |g, _, _, _| Ok(g.shallow_clone())))
    }

    /// Product over `axes`; dtypes as for [`Tensor::sum`]. Not differentiable
    /// yet (its gradient needs care around zeros).
    pub fn prod(&self, axes: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
        if is_grad_enabled() && self.requires_grad() {
            return Err(TensorError::InvalidArgument(
                "prod doesn't support gradients yet; compute it under no_grad() or on detach()"
                    .into(),
            ));
        }
        let plan = self.plan(axes)?;
        with_dtype!(self.dtype(), T => match self.dtype() {
            DType::F32 => self.reduce::<T, ProdFloat<f32>>(&plan, keepdim),
            DType::F64 => self.reduce::<T, ProdFloat<f64>>(&plan, keepdim),
            _ => self.reduce::<T, ProdInt>(&plan, keepdim),
        })
    }

    /// Mean over `axes`. Integers and bools give the default float dtype.
    pub fn mean(&self, axes: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
        let plan = self.plan(axes)?;
        let out = with_dtype!(self.dtype(), T => match self.dtype().to_float() {
            DType::F64 => self.reduce::<T, Mean<f64>>(&plan, keepdim),
            _ => self.reduce::<T, Mean<f32>>(&plan, keepdim),
        })?;
        Ok(
            self.record_reduction(out, plan, "MeanBackward", |g, _, _, plan| {
                g.div(&constant(plan.count as f64, g))
            }),
        )
    }

    /// Largest value over `axes`; NaN wins. Keeps the dtype. The gradient
    /// goes to the largest elements, split evenly between ties.
    pub fn max(&self, axes: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
        let plan = self.plan(axes)?;
        self.ensure_nonempty(&plan, "max")?;
        let out = with_dtype!(self.dtype(), T => self.reduce::<T, Max>(&plan, keepdim))?;
        Ok(self.record_reduction(out, plan, "MaxBackward", extreme_grad))
    }

    /// Smallest value over `axes`; NaN wins. Keeps the dtype. Gradient as for
    /// [`Tensor::max`].
    pub fn min(&self, axes: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
        let plan = self.plan(axes)?;
        self.ensure_nonempty(&plan, "min")?;
        let out = with_dtype!(self.dtype(), T => self.reduce::<T, Min>(&plan, keepdim))?;
        Ok(self.record_reduction(out, plan, "MinBackward", extreme_grad))
    }

    /// Index of the largest value along `axis`, or into the flattened tensor
    /// if `None`. Ties and NaNs give the first occurrence. int64.
    pub fn argmax(&self, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
        self.arg_extreme::<true>(axis, keepdim)
    }

    /// Index of the smallest value; see [`Tensor::argmax`].
    pub fn argmin(&self, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
        self.arg_extreme::<false>(axis, keepdim)
    }

    fn arg_extreme<const MAX: bool>(&self, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
        let axes = axis.map(|a| [a]);
        let plan = self.plan(axes.as_ref().map(|a| a.as_slice()))?;
        self.ensure_nonempty(&plan, if MAX { "argmax" } else { "argmin" })?;
        with_dtype!(self.dtype(), T => if MAX {
            self.reduce::<T, ArgMax>(&plan, keepdim)
        } else {
            self.reduce::<T, ArgMin>(&plan, keepdim)
        })
    }

    /// True if every element (over `axes`) is truthy. Bool result.
    pub fn all(&self, axes: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
        let plan = self.plan(axes)?;
        with_dtype!(self.dtype(), T => self.reduce::<T, All>(&plan, keepdim))
    }

    /// True if any element (over `axes`) is truthy. Bool result.
    pub fn any(&self, axes: Option<&[isize]>, keepdim: bool) -> Result<Tensor> {
        let plan = self.plan(axes)?;
        with_dtype!(self.dtype(), T => self.reduce::<T, Any>(&plan, keepdim))
    }
}

/// A reduction's output (or its gradient) stretched back over the input's
/// shape: reduced axes are put back as length 1, then broadcast (a view).
fn spread(t: &Tensor, plan: &Plan, input_shape: &[usize]) -> Result<Tensor> {
    let kept: Vec<isize> = plan.kept_shape.iter().map(|&d| d as isize).collect();
    t.reshape(&kept)?.broadcast_to(input_shape)
}

/// Gradient of max/min: to every element equal to the extreme, divided by
/// how many there are.
fn extreme_grad(g: &Tensor, x: &Tensor, peak: &Tensor, plan: &Plan) -> Result<Tensor> {
    let hits = x.eq(peak)?;
    let axes: Vec<isize> = (0..x.ndim() as isize)
        .filter(|&a| plan.reduced[a as usize])
        .collect();
    let ties = hits.to_dtype(g.dtype()).sum(Some(&axes), true)?;
    where_(&hits, &g.div(&ties)?, &constant(0.0, g))
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::error::TensorError;
    use crate::tensor::Tensor;

    /// 0, 1, 2, ... as int64 of the given shape.
    fn range(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec((0..n as i64).collect(), shape).unwrap()
    }

    fn floats(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    /// Sum of every element along `axis` computed with `select`, for checking.
    fn reference_sum(x: &Tensor, axis: usize) -> Vec<i64> {
        let mut total = x.select(axis as isize, 0).unwrap().to_dtype(DType::I64);
        for i in 1..x.shape()[axis] {
            total = total
                .add(&x.select(axis as isize, i as isize).unwrap())
                .unwrap();
        }
        total.to_vec::<i64>()
    }

    #[test]
    fn sum_over_everything_and_each_axis() {
        let x = range(&[2, 3, 4]);
        assert_eq!(x.sum(None, false).unwrap().to_vec::<i64>(), vec![276]);
        assert_eq!(x.sum(None, false).unwrap().shape(), &[] as &[usize]);
        for axis in 0..3 {
            let got = x.sum(Some(&[axis as isize]), false).unwrap();
            assert_eq!(got.to_vec::<i64>(), reference_sum(&x, axis), "axis {axis}");
        }
        let both = x.sum(Some(&[0, -1]), true).unwrap();
        assert_eq!(both.shape(), &[1, 3, 1]);
        assert_eq!(both.to_vec::<i64>(), vec![60, 92, 124]);
    }

    #[test]
    fn reductions_follow_layout_not_memory_order() {
        // Transposed and reversed views reduce the same as their contiguous copies.
        let x = range(&[3, 4, 5]).permute(&[2, 0, 1]).unwrap();
        let x = x.slice(1, None, None, -1).unwrap();
        let copy = x.deep_copy();
        for axes in [None, Some(&[0][..]), Some(&[1, 2][..]), Some(&[2, 0][..])] {
            assert_eq!(
                x.sum(axes, false).unwrap().to_vec::<i64>(),
                copy.sum(axes, false).unwrap().to_vec::<i64>()
            );
            assert_eq!(
                x.argmax(axes.map(|a| a[0]), false).unwrap().to_vec::<i64>(),
                copy.argmax(axes.map(|a| a[0]), false)
                    .unwrap()
                    .to_vec::<i64>()
            );
        }
    }

    #[test]
    fn result_dtypes() {
        assert_eq!(range(&[3]).sum(None, false).unwrap().dtype(), DType::I64);
        let b = Tensor::from_vec(vec![true, true, false], &[3]).unwrap();
        assert_eq!(b.sum(None, false).unwrap().to_vec::<i64>(), vec![2]);
        assert_eq!(range(&[3]).mean(None, false).unwrap().dtype(), DType::F32);
        let f64s = Tensor::from_vec(vec![1.0f64, 2.0], &[2]).unwrap();
        assert_eq!(f64s.mean(None, false).unwrap().dtype(), DType::F64);
        assert_eq!(b.max(None, false).unwrap().dtype(), DType::Bool);
        assert_eq!(range(&[3]).argmax(None, false).unwrap().dtype(), DType::I64);
        assert_eq!(range(&[3]).all(None, false).unwrap().dtype(), DType::Bool);
    }

    #[test]
    fn float_sums_are_accurate() {
        // 1M copies of 0.1: an f32 running total drifts to ~100958; we don't.
        let x = Tensor::full(&[1000, 1000], 0.1f64, DType::F32);
        let total = x.sum(None, false).unwrap().item().unwrap().to_f64();
        assert!((total - 100_000.0).abs() < 0.01, "{total}");
        // Column sums go through the elementwise-accumulate path.
        let cols = x.sum(Some(&[0]), false).unwrap().to_vec::<f32>();
        assert!(cols.iter().all(|&c| (c - 100.0).abs() < 1e-3));
    }

    #[test]
    fn mean_prod_max_min() {
        let x = floats(&[1.0, -2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(x.mean(None, false).unwrap().to_vec::<f32>(), vec![1.5]);
        assert_eq!(
            x.mean(Some(&[1]), false).unwrap().to_vec::<f32>(),
            vec![-0.5, 3.5]
        );
        assert_eq!(x.prod(None, false).unwrap().to_vec::<f32>(), vec![-24.0]);
        assert_eq!(
            x.max(Some(&[0]), false).unwrap().to_vec::<f32>(),
            vec![3.0, 4.0]
        );
        assert_eq!(x.min(None, false).unwrap().to_vec::<f32>(), vec![-2.0]);
        assert_eq!(
            range(&[4]).prod(None, false).unwrap().to_vec::<i64>(),
            vec![0]
        );
    }

    #[test]
    fn nan_handling() {
        let x = floats(&[1.0, f32::NAN, 3.0, 0.0], &[4]);
        assert!(x.max(None, false).unwrap().to_vec::<f32>()[0].is_nan());
        assert!(x.min(None, false).unwrap().to_vec::<f32>()[0].is_nan());
        assert!(x.sum(None, false).unwrap().to_vec::<f32>()[0].is_nan());
        assert_eq!(x.argmax(None, false).unwrap().to_vec::<i64>(), vec![1]);
        assert_eq!(x.argmin(None, false).unwrap().to_vec::<i64>(), vec![1]);
    }

    #[test]
    fn vectorized_max_min_match_one_by_one_folding() {
        use super::{Max, Min, Reducer};
        fn one_by_one<R: Reducer<f32, Acc = Option<f32>>>(row: &[f32]) -> Option<f32> {
            row.iter().fold(R::init(), |a, &x| R::fold(a, x))
        }
        fn same(a: Option<f32>, b: Option<f32>) -> bool {
            match (a, b) {
                (Some(x), Some(y)) => x == y || (x.is_nan() && y.is_nan()),
                (None, None) => true,
                _ => false,
            }
        }
        for len in 0..26 {
            let base: Vec<f32> = (0..len).map(|i| ((i * 7919) % 13) as f32 - 6.0).collect();
            let mut rows = vec![base.clone()];
            for at in 0..len {
                let mut with_nan = base.clone();
                with_nan[at] = f32::NAN;
                rows.push(with_nan);
                let mut with_inf = base.clone();
                with_inf[at] = if at % 2 == 0 {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                };
                rows.push(with_inf);
            }
            for row in &rows {
                assert!(
                    same(Max::fold_row(None, row), one_by_one::<Max>(row)),
                    "max {row:?}"
                );
                assert!(
                    same(Min::fold_row(None, row), one_by_one::<Min>(row)),
                    "min {row:?}"
                );
                // Continuing from an existing accumulator, including a NaN one.
                for acc in [Some(100.0), Some(-100.0), Some(f32::NAN)] {
                    let expected = row.iter().fold(acc, |a, &x| Max::fold(a, x));
                    assert!(same(Max::fold_row(acc, row), expected), "max from {acc:?}");
                }
            }
        }
    }

    #[test]
    fn arg_extremes_take_the_first_occurrence() {
        let x = Tensor::from_vec(vec![3i64, 7, 7, 1, 1, 7], &[2, 3]).unwrap();
        assert_eq!(x.argmax(None, false).unwrap().to_vec::<i64>(), vec![1]);
        assert_eq!(x.argmin(None, false).unwrap().to_vec::<i64>(), vec![3]);
        assert_eq!(
            x.argmax(Some(1), false).unwrap().to_vec::<i64>(),
            vec![1, 2]
        );
        assert_eq!(
            x.argmax(Some(0), false).unwrap().to_vec::<i64>(),
            vec![0, 0, 0]
        );
        assert_eq!(x.argmin(Some(-1), true).unwrap().shape(), &[2, 1]);
    }

    #[test]
    fn all_and_any() {
        let x = floats(&[1.0, 0.0, 2.0, 3.0], &[2, 2]);
        assert_eq!(x.all(None, false).unwrap().to_vec::<bool>(), vec![false]);
        assert_eq!(x.any(None, false).unwrap().to_vec::<bool>(), vec![true]);
        assert_eq!(
            x.all(Some(&[1]), false).unwrap().to_vec::<bool>(),
            vec![false, true]
        );
    }

    #[test]
    fn empty_reductions() {
        let empty = Tensor::zeros(&[0, 3], DType::F32);
        assert_eq!(empty.sum(None, false).unwrap().to_vec::<f32>(), vec![0.0]);
        assert_eq!(
            empty.sum(Some(&[0]), false).unwrap().to_vec::<f32>(),
            vec![0.0; 3]
        );
        assert_eq!(empty.prod(None, false).unwrap().to_vec::<f32>(), vec![1.0]);
        assert!(empty.mean(None, false).unwrap().to_vec::<f32>()[0].is_nan());
        assert_eq!(empty.all(None, false).unwrap().to_vec::<bool>(), vec![true]);
        assert!(matches!(
            empty.max(None, false),
            Err(TensorError::InvalidArgument(_))
        ));
        assert!(empty.argmax(Some(0), false).is_err());
        // Reducing the non-empty axis leaves no outputs, which is fine.
        assert_eq!(empty.max(Some(&[1]), false).unwrap().shape(), &[0]);
    }

    #[test]
    fn bad_axes() {
        let x = range(&[2, 3]);
        assert!(matches!(
            x.sum(Some(&[2]), false),
            Err(TensorError::AxisOutOfRange { .. })
        ));
        assert!(x.sum(Some(&[0, -2]), false).is_err(), "same axis twice");
        assert_eq!(
            x.sum(Some(&[]), false).unwrap().shape(),
            &[2, 3],
            "no axes: nothing reduced"
        );
    }
}
