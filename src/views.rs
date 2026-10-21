//! Operations that return a new view onto the same storage.
//!
//! None of these copy data: they only compute a new shape, strides and offset.
//! The one exception is [`Tensor::reshape`], which falls back to a copy when
//! the requested shape can't be expressed with strides (see [`Tensor::view`]).

use crate::autograd::record;
use crate::error::{Result, TensorError};
use crate::tensor::{Tensor, contiguous_strides};

/// One entry of an indexing expression like `t[0, 1:5:2, None, ...]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorIndex {
    /// Pick one position and drop the axis. Negative counts from the end.
    Index(isize),
    /// Python-style `start:stop:step`; `None` means "from the edge".
    Slice {
        start: Option<isize>,
        stop: Option<isize>,
        step: isize,
    },
    /// Insert a new axis of length 1 (`None` in Python).
    NewAxis,
    /// Stands for as many full slices as needed (`...` in Python).
    Ellipsis,
}

impl TensorIndex {
    /// The full slice `:`.
    pub const FULL: TensorIndex = TensorIndex::Slice {
        start: None,
        stop: None,
        step: 1,
    };
}

/// Resolves a possibly-negative axis argument against `ndim` axes.
pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let resolved = if axis < 0 { axis + ndim as isize } else { axis };
    if resolved < 0 || resolved as usize >= ndim {
        return Err(TensorError::AxisOutOfRange { axis, ndim });
    }
    Ok(resolved as usize)
}

/// Resolves a possibly-negative position along an axis of length `size`.
fn normalize_index(index: isize, size: usize, axis: usize) -> Result<usize> {
    let resolved = if index < 0 {
        index + size as isize
    } else {
        index
    };
    if resolved < 0 || resolved as usize >= size {
        return Err(TensorError::IndexOutOfBounds { axis, index, size });
    }
    Ok(resolved as usize)
}

/// Python's `slice.indices`: returns the first position and how many elements
/// the slice selects on an axis of length `size`.
fn slice_range(
    start: Option<isize>,
    stop: Option<isize>,
    step: isize,
    size: usize,
) -> (isize, usize) {
    let size = size as isize;
    // Negative bounds count from the end, then get clamped into range.
    let resolve = |bound: isize, low: isize, high: isize| {
        let b = if bound < 0 { bound + size } else { bound };
        b.clamp(low, high)
    };
    let (start, stop) = if step > 0 {
        (
            start.map_or(0, |s| resolve(s, 0, size)),
            stop.map_or(size, |s| resolve(s, 0, size)),
        )
    } else {
        // Walking backwards, -1 stands for "just before the first element".
        (
            start.map_or(size - 1, |s| resolve(s, -1, size - 1)),
            stop.map_or(-1, |s| resolve(s, -1, size - 1)),
        )
    };
    let span = if step > 0 { stop - start } else { start - stop };
    let step_abs = step.abs();
    let len = if span <= 0 {
        0
    } else {
        // Ceiling division written so it can't overflow for huge steps.
        (span - 1) / step_abs + 1
    };
    (start, len as usize)
}

/// Turns a shape that may contain one `-1` into a concrete shape for `numel` elements.
fn infer_shape(numel: usize, shape: &[isize]) -> Result<Vec<usize>> {
    let bad = || {
        TensorError::InvalidArgument(format!(
            "cannot reshape a tensor of {numel} elements into shape {shape:?}"
        ))
    };
    let mut inferred = None;
    let mut known = 1usize;
    for (axis, &dim) in shape.iter().enumerate() {
        match dim {
            -1 if inferred.is_some() => {
                return Err(TensorError::InvalidArgument(
                    "only one dimension can be inferred (-1)".into(),
                ));
            }
            -1 => inferred = Some(axis),
            d if d < 0 => return Err(bad()),
            d => known *= d as usize,
        }
    }
    let mut out: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    match inferred {
        Some(axis) => {
            if known == 0 || !numel.is_multiple_of(known) {
                return Err(bad());
            }
            out[axis] = numel / known;
        }
        None if known != numel => return Err(bad()),
        None => {}
    }
    Ok(out)
}

/// Gradient of a reshape-like view: the output gradient reshaped back.
fn reshape_back(input: &Tensor) -> impl Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static {
    let shape: Vec<isize> = input.shape().iter().map(|&d| d as isize).collect();
    move |g| g.reshape(&shape)
}

/// Strides that let `new_shape` walk the same elements as (`shape`, `strides`)
/// without copying, or `None` if that's impossible.
///
/// This is PyTorch's algorithm: split the old axes into "chunks" that are laid
/// out contiguously relative to each other, then check that the new axes can
/// be carved out of those chunks without straddling a boundary.
fn view_strides(shape: &[usize], strides: &[isize], new_shape: &[usize]) -> Option<Vec<isize>> {
    let mut new_strides = vec![0isize; new_shape.len()];
    let mut view_d = new_shape.len() as isize - 1;
    let mut chunk_base_stride = *strides.last()?;
    let mut tensor_numel = 1usize;
    let mut view_numel = 1usize;
    for tensor_d in (0..shape.len()).rev() {
        tensor_numel *= shape[tensor_d];
        let chunk_ends = tensor_d == 0
            || (shape[tensor_d - 1] != 1
                && strides[tensor_d - 1] != tensor_numel as isize * chunk_base_stride);
        if chunk_ends {
            while view_d >= 0 && (view_numel < tensor_numel || new_shape[view_d as usize] == 1) {
                new_strides[view_d as usize] = view_numel as isize * chunk_base_stride;
                view_numel *= new_shape[view_d as usize];
                view_d -= 1;
            }
            if view_numel != tensor_numel {
                return None;
            }
            if tensor_d > 0 {
                chunk_base_stride = strides[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    (view_d == -1).then_some(new_strides)
}

impl Tensor {
    /// A view with a new shape (one dimension may be `-1`). Fails if the
    /// current layout can't be reinterpreted without copying; use
    /// [`Tensor::reshape`] to copy in that case.
    pub fn view(&self, shape: &[isize]) -> Result<Tensor> {
        let new_shape = infer_shape(self.numel(), shape)?;
        let strides = if self.is_contiguous() || self.numel() == 0 {
            Some(contiguous_strides(&new_shape))
        } else {
            view_strides(&self.shape, &self.strides, &new_shape)
        };
        match strides {
            Some(strides) => {
                let view = self.with_layout(new_shape, strides, self.offset);
                Ok(self.record_view(view, "ViewBackward", reshape_back(self)))
            }
            None => Err(TensorError::InvalidArgument(format!(
                "view shape {shape:?} is incompatible with this tensor's memory layout; \
                 use reshape (which copies when needed) or call contiguous() first"
            ))),
        }
    }

    /// Like [`Tensor::view`], but copies into fresh memory when a view is impossible.
    pub fn reshape(&self, shape: &[isize]) -> Result<Tensor> {
        match self.view(shape) {
            Err(TensorError::InvalidArgument(_)) if infer_shape(self.numel(), shape).is_ok() => {
                self.contiguous().view(shape)
            }
            result => result,
        }
    }

    /// Merges axes `start..=end` into one. `flatten(0, -1)` gives a 1-D tensor.
    pub fn flatten(&self, start: isize, end: isize) -> Result<Tensor> {
        if self.ndim() == 0 {
            return self.reshape(&[1]);
        }
        let start = normalize_axis(start, self.ndim())?;
        let end = normalize_axis(end, self.ndim())?;
        if start > end {
            return Err(TensorError::InvalidArgument(
                "flatten: start axis must not come after end axis".into(),
            ));
        }
        let merged: usize = self.shape[start..=end].iter().product();
        let shape: Vec<isize> = self.shape[..start]
            .iter()
            .copied()
            .chain([merged])
            .chain(self.shape[end + 1..].iter().copied())
            .map(|d| d as isize)
            .collect();
        self.reshape(&shape)
    }

    /// Reorders axes: result axis `i` is this tensor's axis `dims[i]`.
    pub fn permute(&self, dims: &[isize]) -> Result<Tensor> {
        if dims.len() != self.ndim() {
            return Err(TensorError::InvalidArgument(format!(
                "permute needs {} axes, got {}",
                self.ndim(),
                dims.len()
            )));
        }
        let mut seen = vec![false; self.ndim()];
        let mut shape = Vec::with_capacity(dims.len());
        let mut strides = Vec::with_capacity(dims.len());
        for &dim in dims {
            let axis = normalize_axis(dim, self.ndim())?;
            if std::mem::replace(&mut seen[axis], true) {
                return Err(TensorError::InvalidArgument(format!(
                    "permute: axis {dim} appears more than once"
                )));
            }
            shape.push(self.shape[axis]);
            strides.push(self.strides[axis]);
        }
        let view = self.with_layout(shape, strides, self.offset);
        // Undo it with the inverse permutation.
        let mut inverse = vec![0isize; dims.len()];
        for (i, &dim) in dims.iter().enumerate() {
            inverse[normalize_axis(dim, self.ndim())?] = i as isize;
        }
        Ok(self.record_view(view, "PermuteBackward", move |g| g.permute(&inverse)))
    }

    /// Swaps two axes.
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Result<Tensor> {
        let a = normalize_axis(dim0, self.ndim())?;
        let b = normalize_axis(dim1, self.ndim())?;
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(a, b);
        strides.swap(a, b);
        let view = self.with_layout(shape, strides, self.offset);
        Ok(self.record_view(view, "TransposeBackward", move |g| {
            g.transpose(a as isize, b as isize)
        }))
    }

    /// Reverses the order of all axes (`.T` in Python).
    pub fn reversed_axes(&self) -> Tensor {
        let shape = self.shape.iter().rev().copied().collect();
        let strides = self.strides.iter().rev().copied().collect();
        let view = self.with_layout(shape, strides, self.offset);
        self.record_view(view, "TransposeBackward", |g| Ok(g.reversed_axes()))
    }

    /// Removes axes of length 1: just `axis` if given (a no-op when that axis
    /// isn't length 1), otherwise all of them.
    pub fn squeeze(&self, axis: Option<isize>) -> Result<Tensor> {
        let only = axis.map(|a| normalize_axis(a, self.ndim())).transpose()?;
        let (shape, strides) = self
            .shape
            .iter()
            .zip(&self.strides)
            .enumerate()
            .filter(|&(i, (&dim, _))| !(dim == 1 && only.is_none_or(|a| a == i)))
            .map(|(_, (&dim, &stride))| (dim, stride))
            .unzip();
        let view = self.with_layout(shape, strides, self.offset);
        Ok(self.record_view(view, "SqueezeBackward", reshape_back(self)))
    }

    /// Inserts an axis of length 1 at position `axis` (which may be `ndim`).
    pub fn unsqueeze(&self, axis: isize) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.ndim() + 1)?;
        let stride = match axis {
            a if a < self.ndim() => self.strides[a] * self.shape[a] as isize,
            _ => 1,
        };
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.insert(axis, 1);
        strides.insert(axis, stride);
        let view = self.with_layout(shape, strides, self.offset);
        Ok(self.record_view(view, "UnsqueezeBackward", reshape_back(self)))
    }

    /// This tensor if it's already contiguous (sharing storage), otherwise a
    /// contiguous copy.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            self.shallow_clone()
        } else {
            self.deep_copy()
        }
    }

    /// NumPy-style basic indexing. Axes not mentioned are kept whole, so
    /// `t.index(&[Index(0)])` on a matrix is its first row.
    pub fn index(&self, items: &[TensorIndex]) -> Result<Tensor> {
        let consumed = items
            .iter()
            .filter(|i| matches!(i, TensorIndex::Index(_) | TensorIndex::Slice { .. }))
            .count();
        if consumed > self.ndim() {
            return Err(TensorError::TooManyIndices {
                ndim: self.ndim(),
                got: consumed,
            });
        }
        if items
            .iter()
            .filter(|i| **i == TensorIndex::Ellipsis)
            .count()
            > 1
        {
            return Err(TensorError::InvalidArgument(
                "an index can only have a single ellipsis (...)".into(),
            ));
        }

        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let mut offset = self.offset as isize;
        let mut axis = 0;
        for item in items {
            match *item {
                TensorIndex::Index(i) => {
                    let i = normalize_index(i, self.shape[axis], axis)?;
                    offset += i as isize * self.strides[axis];
                    axis += 1;
                }
                TensorIndex::Slice { start, stop, step } => {
                    if step == 0 {
                        return Err(TensorError::InvalidArgument(
                            "slice step cannot be zero".into(),
                        ));
                    }
                    let (first, len) = slice_range(start, stop, step, self.shape[axis]);
                    if len > 0 {
                        offset += first * self.strides[axis];
                    }
                    shape.push(len);
                    // With fewer than two elements the stride is never used, and
                    // `stride * step` could overflow for huge steps (`t[::10**30]`).
                    strides.push(if len > 1 {
                        self.strides[axis] * step
                    } else {
                        self.strides[axis]
                    });
                    axis += 1;
                }
                TensorIndex::NewAxis => {
                    shape.push(1);
                    strides.push(0);
                }
                TensorIndex::Ellipsis => {
                    for _ in 0..self.ndim() - consumed {
                        shape.push(self.shape[axis]);
                        strides.push(self.strides[axis]);
                        axis += 1;
                    }
                }
            }
        }
        shape.extend_from_slice(&self.shape[axis..]);
        strides.extend_from_slice(&self.strides[axis..]);
        let view = self.with_layout(shape, strides, offset as usize);
        // The gradient lands in the indexed positions of an otherwise-zero tensor.
        let (input_shape, items) = (self.shape.clone(), items.to_vec());
        Ok(self.record_view(view, "IndexBackward", move |g| {
            let grad = Tensor::zeros(&input_shape, g.dtype());
            grad.index(&items)?.copy_from(g)?;
            Ok(grad)
        }))
    }

    /// Records a view whose input gradient is `back(output gradient)`. Views
    /// save no tensors: their gradients only depend on shapes.
    fn record_view(
        &self,
        view: Tensor,
        name: &'static str,
        back: impl Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
    ) -> Tensor {
        record(view, [self], name, |_, _| {
            Box::new(move |g| Ok(vec![Some(back(g)?)]))
        })
    }

    /// The sub-tensor at position `index` along `axis` (that axis is removed).
    pub fn select(&self, axis: isize, index: isize) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.ndim())?;
        let mut items = vec![TensorIndex::FULL; axis];
        items.push(TensorIndex::Index(index));
        self.index(&items)
    }

    /// `start:stop:step` along one axis.
    pub fn slice(
        &self,
        axis: isize,
        start: Option<isize>,
        stop: Option<isize>,
        step: isize,
    ) -> Result<Tensor> {
        let axis = normalize_axis(axis, self.ndim())?;
        let mut items = vec![TensorIndex::FULL; axis];
        items.push(TensorIndex::Slice { start, stop, step });
        self.index(&items)
    }
}

#[cfg(test)]
mod tests {
    use super::TensorIndex::{Ellipsis, Index, NewAxis, Slice};
    use super::*;

    /// 0..n as an f32 tensor of the given shape.
    fn range(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape).unwrap()
    }

    fn s(start: Option<isize>, stop: Option<isize>, step: isize) -> TensorIndex {
        Slice { start, stop, step }
    }

    #[test]
    fn slice_range_matches_python() {
        // Compared against `range(10)[slice]` in Python.
        assert_eq!(slice_range(None, None, 1, 10), (0, 10));
        assert_eq!(slice_range(Some(2), Some(8), 3, 10), (2, 2));
        assert_eq!(slice_range(Some(-3), None, 1, 10), (7, 3));
        assert_eq!(slice_range(None, None, -1, 10), (9, 10));
        assert_eq!(slice_range(Some(8), Some(2), -2, 10), (8, 3));
        assert_eq!(slice_range(Some(5), Some(2), 1, 10).1, 0);
        assert_eq!(slice_range(Some(-100), Some(100), 1, 10), (0, 10));
        assert_eq!(slice_range(None, None, 1, 0).1, 0);
        // Huge steps (as CPython passes for `[::10**30]`) must not overflow.
        assert_eq!(
            slice_range(Some(0), Some(isize::MAX), isize::MAX, 10),
            (0, 1)
        );
        assert_eq!(
            slice_range(Some(isize::MAX), Some(isize::MIN), -isize::MAX, 10),
            (9, 1)
        );
    }

    #[test]
    fn view_and_reshape() {
        let x = range(&[2, 3]);
        let v = x.view(&[3, -1]).unwrap();
        assert_eq!(v.shape(), &[3, 2]);
        assert!(v.shares_storage(&x));
        assert!(x.view(&[4, -1]).is_err());
        assert!(x.view(&[-1, -1]).is_err());

        // A transpose can't be flattened in place: view refuses, reshape copies.
        let xt = x.transpose(0, 1).unwrap();
        assert!(xt.view(&[6]).is_err());
        let flat = xt.reshape(&[6]).unwrap();
        assert!(!flat.shares_storage(&x));
        assert_eq!(flat.to_vec::<f32>(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

        // But a strided slice whose rows line up can still be viewed.
        let every_other = range(&[4, 6]).slice(1, None, None, 2).unwrap();
        let flat = every_other.view(&[12]).unwrap();
        assert_eq!(flat.strides(), &[2]);
        assert_eq!(flat.to_vec::<f32>(), every_other.to_vec::<f32>());
    }

    #[test]
    fn view_strides_handles_split_and_merge() {
        // Axes (1, 0, 2) of [2, 3, 4]: shape [3, 2, 4], strides [4, 12, 1].
        // No two neighbouring axes line up, so no merge works, but splitting does.
        let x = range(&[2, 3, 4]).permute(&[1, 0, 2]).unwrap();
        let split = x.view(&[3, 2, 2, 2]).unwrap();
        assert_eq!(split.to_vec::<f32>(), x.to_vec::<f32>());
        assert!(x.view(&[3, 8]).is_err());
        assert!(x.view(&[6, 4]).is_err());

        // Axes (2, 0, 1): shape [4, 2, 3], strides [1, 12, 4]. The last two
        // line up (12 == 3 * 4), so they merge without a copy.
        let y = range(&[2, 3, 4]).permute(&[2, 0, 1]).unwrap();
        let merged = y.view(&[4, 6]).unwrap();
        assert_eq!(merged.strides(), &[1, 4]);
        assert_eq!(merged.to_vec::<f32>(), y.to_vec::<f32>());
    }

    #[test]
    fn flatten_squeeze_unsqueeze() {
        let x = range(&[2, 3, 4]);
        assert_eq!(x.flatten(0, -1).unwrap().shape(), &[24]);
        assert_eq!(x.flatten(1, 2).unwrap().shape(), &[2, 12]);
        assert_eq!(Tensor::scalar(1.0f32).flatten(0, -1).unwrap().shape(), &[1]);

        let y = range(&[1, 3, 1]);
        assert_eq!(y.squeeze(None).unwrap().shape(), &[3]);
        assert_eq!(y.squeeze(Some(-1)).unwrap().shape(), &[1, 3]);
        assert_eq!(y.squeeze(Some(1)).unwrap().shape(), &[1, 3, 1]);

        let z = range(&[2, 3]);
        assert_eq!(z.unsqueeze(0).unwrap().shape(), &[1, 2, 3]);
        assert_eq!(z.unsqueeze(-1).unwrap().shape(), &[2, 3, 1]);
        assert!(z.unsqueeze(0).unwrap().is_contiguous());
        assert!(z.unsqueeze(3).is_err());
    }

    #[test]
    fn permute_and_transpose() {
        let x = range(&[2, 3, 4]);
        let p = x.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.get(&[3, 1, 2]).unwrap(), x.get(&[1, 2, 3]).unwrap());
        assert!(x.permute(&[0, 0, 1]).is_err());
        assert!(x.permute(&[0, 1]).is_err());
        assert_eq!(x.reversed_axes().shape(), &[4, 3, 2]);
        assert_eq!(
            x.transpose(0, -1).unwrap().shape(),
            x.reversed_axes().shape()
        );
    }

    #[test]
    fn basic_indexing() {
        let x = range(&[3, 4]);
        assert_eq!(
            x.index(&[Index(1)]).unwrap().to_vec::<f32>(),
            vec![4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(
            x.index(&[Index(-1), Index(-1)])
                .unwrap()
                .item()
                .unwrap()
                .to_f64(),
            11.0
        );
        assert_eq!(
            x.index(&[TensorIndex::FULL, Index(1)])
                .unwrap()
                .to_vec::<f32>(),
            vec![1.0, 5.0, 9.0]
        );
        let rev = x
            .index(&[s(None, None, -1), s(Some(1), Some(3), 1)])
            .unwrap();
        assert_eq!(rev.shape(), &[3, 2]);
        assert_eq!(rev.to_vec::<f32>(), vec![9.0, 10.0, 5.0, 6.0, 1.0, 2.0]);
        assert_eq!(
            x.index(&[Ellipsis, Index(0)]).unwrap().to_vec::<f32>(),
            vec![0.0, 4.0, 8.0]
        );
        assert_eq!(
            x.index(&[NewAxis, Ellipsis, NewAxis]).unwrap().shape(),
            &[1, 3, 4, 1]
        );
        assert_eq!(x.index(&[s(Some(5), None, 1)]).unwrap().shape(), &[0, 4]);
        // A huge step selects one element without overflowing the stride.
        let one = x.index(&[s(None, None, isize::MAX)]).unwrap();
        assert_eq!(one.to_vec::<f32>(), vec![0.0, 1.0, 2.0, 3.0]);

        assert!(matches!(
            x.index(&[Index(0), Index(0), Index(0)]),
            Err(TensorError::TooManyIndices { .. })
        ));
        assert!(matches!(
            x.index(&[Index(3)]),
            Err(TensorError::IndexOutOfBounds { index: 3, .. })
        ));
        assert!(x.index(&[Ellipsis, Ellipsis]).is_err());
        assert!(x.index(&[s(None, None, 0)]).is_err());
    }

    #[test]
    fn writes_go_through_views() {
        let x = range(&[3, 4]);
        x.select(0, 1).unwrap().fill(-1.0).unwrap();
        x.slice(1, None, None, 3).unwrap().fill(9.0).unwrap();
        assert_eq!(
            x.to_vec::<f32>(),
            vec![
                9.0, 1.0, 2.0, 9.0, 9.0, -1.0, -1.0, 9.0, 9.0, 9.0, 10.0, 9.0
            ]
        );

        // Overlapping copy: reverse a row in place.
        let row = x.select(0, 2).unwrap();
        row.copy_from(&row.slice(0, None, None, -1).unwrap())
            .unwrap();
        assert_eq!(row.to_vec::<f32>(), vec![9.0, 10.0, 9.0, 9.0]);

        assert!(matches!(
            row.copy_from(&range(&[3])),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn contiguous_shares_only_when_possible() {
        let x = range(&[2, 3]);
        assert!(x.contiguous().shares_storage(&x));
        let xt = x.reversed_axes();
        let c = xt.contiguous();
        assert!(!c.shares_storage(&x));
        assert!(c.is_contiguous());
        assert_eq!(c.to_vec::<f32>(), xt.to_vec::<f32>());
    }
}
