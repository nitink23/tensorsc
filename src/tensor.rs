use std::fmt;
use std::sync::Arc;

use crate::device::Device;
use crate::dtype::{DType, Element, Scalar, with_dtype};
use crate::error::{Result, TensorError};
use crate::storage::Storage;

/// An n-dimensional view onto a [`Storage`].
///
/// The element at index `[i0, i1, ...]` lives at storage position
/// `offset + i0 * strides[0] + i1 * strides[1] + ...`. Changing shape, strides
/// and offset (without touching the buffer) is how reshape, transpose and
/// slicing avoid copying data; see `views.rs`. Strides are signed so reversed
/// views (`t[::-1]`) can be expressed too. The dtype is the storage's.
#[derive(Debug)]
pub struct Tensor {
    pub(crate) storage: Arc<Storage>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<isize>,
    pub(crate) offset: usize,
}

/// Row-major strides: the last axis is contiguous in memory.
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1] as isize;
    }
    strides
}

impl Tensor {
    // ---- creation ----

    /// Wraps `data` (row-major) in a new tensor with the given shape. The
    /// dtype comes from the element type.
    pub fn from_vec<T: Element>(data: Vec<T>, shape: &[usize]) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(TensorError::DataShapeMismatch {
                len: data.len(),
                shape: shape.to_vec(),
            });
        }
        let storage = Arc::new(Storage::from_vec(data, Device::Cpu));
        Ok(Tensor::from_parts(
            storage,
            shape.to_vec(),
            contiguous_strides(shape),
            0,
        ))
    }

    /// Builds a view onto existing storage. Callers must make sure every
    /// reachable position is inside the buffer.
    pub(crate) fn from_parts(
        storage: Arc<Storage>,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Tensor {
        debug_assert_eq!(shape.len(), strides.len());
        Tensor {
            storage,
            shape,
            strides,
            offset,
        }
    }

    /// Another view onto this tensor's storage with a different layout.
    pub(crate) fn with_layout(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Tensor {
        Tensor::from_parts(Arc::clone(&self.storage), shape, strides, offset)
    }

    pub fn full(shape: &[usize], value: impl Into<Scalar>, dtype: DType) -> Tensor {
        let value = value.into();
        let numel = shape.iter().product();
        with_dtype!(dtype, T => Tensor::from_vec(vec![value.to::<T>(); numel], shape))
            .expect("length matches shape")
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> Tensor {
        Tensor::full(shape, 0i64, dtype)
    }

    pub fn ones(shape: &[usize], dtype: DType) -> Tensor {
        Tensor::full(shape, 1i64, dtype)
    }

    /// A 0-dimensional tensor holding one value.
    pub fn scalar<T: Element>(value: T) -> Tensor {
        Tensor::from_vec(vec![value], &[]).expect("one element fills shape []")
    }

    /// Evenly spaced values in `[start, end)`, like `numpy.arange`. With all
    /// integer arguments the arithmetic is exact; otherwise it's done in f64.
    pub fn arange(start: Scalar, end: Scalar, step: Scalar, dtype: DType) -> Result<Tensor> {
        let ints = |s: Scalar| !matches!(s, Scalar::Float(_));
        if ints(start) && ints(end) && ints(step) {
            let (start, end, step) = (start.to_i64(), end.to_i64(), step.to_i64());
            if step == 0 {
                return Err(TensorError::InvalidArgument(
                    "arange step must not be zero".into(),
                ));
            }
            // Ceiling division of the span by the step, in i128 so it can't overflow.
            let (span, step_wide) = (end as i128 - start as i128, step as i128);
            let len = if (span > 0) == (step > 0) && span != 0 {
                ((span.abs() - 1) / step_wide.abs() + 1) as usize
            } else {
                0
            };
            return Ok(with_dtype!(dtype, T => Tensor::from_vec(
                (0..len as i64).map(|i| T::from_i64(start + i * step)).collect(),
                &[len],
            ))?);
        }
        let (start, end, step) = (start.to_f64(), end.to_f64(), step.to_f64());
        if step == 0.0 || !step.is_finite() || !start.is_finite() || !end.is_finite() {
            return Err(TensorError::InvalidArgument(
                "arange needs finite bounds and a non-zero finite step".into(),
            ));
        }
        let len = ((end - start) / step).ceil().max(0.0) as usize;
        with_dtype!(dtype, T => Tensor::from_vec(
            (0..len).map(|i| T::from_f64(start + i as f64 * step)).collect(),
            &[len],
        ))
    }

    /// A new tensor with its own contiguous copy of this tensor's elements.
    pub fn deep_copy(&self) -> Tensor {
        with_dtype!(self.dtype(), T => Tensor::from_vec(self.to_vec::<T>(), &self.shape))
            .expect("length matches shape")
    }

    /// Another handle to the same view: same storage, same layout, no copy.
    pub fn shallow_clone(&self) -> Tensor {
        self.with_layout(self.shape.clone(), self.strides.clone(), self.offset)
    }

    /// This tensor converted to `dtype`: shares memory if it already has that
    /// dtype, otherwise a converted copy (see [`Element::cast`]).
    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if dtype == self.dtype() {
            return self.shallow_clone();
        }
        with_dtype!(dtype, T => Tensor::from_vec(self.to_vec::<T>(), &self.shape))
            .expect("length matches shape")
    }

    // ---- metadata ----

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Strides in elements (not bytes), like PyTorch.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    /// False for tensors over read-only memory (e.g. a read-only NumPy array).
    pub fn is_writable(&self) -> bool {
        self.storage.is_writable()
    }

    /// True when elements are laid out row-major with no gaps, i.e. the view
    /// could be handed to code that expects a plain flat array.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1isize;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            // Axes of length 1 never move, so their stride doesn't matter.
            if dim != 1 && stride != expected {
                return false;
            }
            expected *= dim as isize;
        }
        true
    }

    /// True when both tensors are views onto the same `Storage` object (and
    /// so the same lock).
    pub fn shares_storage(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// True when the two tensors' memory may overlap: the same storage, or
    /// different storages wrapping the same bytes. Writing into one while
    /// reading the other then needs a copy.
    pub fn overlaps_memory(&self, other: &Tensor) -> bool {
        self.shares_storage(other) || self.storage.overlaps(&other.storage)
    }

    // ---- element access ----

    fn storage_index(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.ndim() {
            return Err(TensorError::IndexRankMismatch {
                expected: self.ndim(),
                got: index.len(),
            });
        }
        let mut pos = self.offset as isize;
        for (axis, (&i, &size)) in index.iter().zip(&self.shape).enumerate() {
            if i >= size {
                return Err(TensorError::IndexOutOfBounds {
                    axis,
                    index: i as isize,
                    size,
                });
            }
            pos += i as isize * self.strides[axis];
        }
        Ok(pos as usize)
    }

    pub fn get(&self, index: &[usize]) -> Result<Scalar> {
        let pos = self.storage_index(index)?;
        Ok(with_dtype!(self.dtype(), T => self.storage.read::<T>()[pos].to_scalar()))
    }

    /// Writes one element, converted to this tensor's dtype. Takes `&self`
    /// because storage is shared: the write is visible through every view.
    pub fn set(&self, index: &[usize], value: impl Into<Scalar>) -> Result<()> {
        let pos = self.storage_index(index)?;
        let value = value.into();
        with_dtype!(self.dtype(), T => self.storage.write::<T>()?[pos] = value.to::<T>());
        Ok(())
    }

    /// The value of a one-element tensor, whatever its shape.
    pub fn item(&self) -> Result<Scalar> {
        if self.numel() != 1 {
            return Err(TensorError::InvalidArgument(format!(
                "only one-element tensors can be converted to a scalar, this one has {}",
                self.numel()
            )));
        }
        Ok(with_dtype!(self.dtype(), T => self.storage.read::<T>()[self.offset].to_scalar()))
    }

    /// Copies `src` element-by-element into this view, converting to this
    /// tensor's dtype. `src` is broadcast to this view's shape. Safe even when
    /// `src` overlaps this view, e.g. `t[0] = t[1]`.
    pub fn copy_from(&self, src: &Tensor) -> Result<()> {
        if src.numel() == 1 {
            return self.fill(src.item()?);
        }
        let src = src.to_dtype(self.dtype());
        with_dtype!(self.dtype(), T => self.zip_map_assign::<T>(&src, |_, y| y))
    }

    /// All elements in logical row-major order, converted to `T`.
    pub fn to_vec<T: Element>(&self) -> Vec<T> {
        with_dtype!(self.dtype(), S => self.map_to_vec::<S, T>(|x| x.cast::<T>()))
    }
}

fn fmt_nested<T: Element>(f: &mut fmt::Formatter<'_>, data: &[T], shape: &[usize]) -> fmt::Result {
    match shape.split_first() {
        None => data[0].write(f),
        Some((&len, rest)) => {
            let chunk: usize = rest.iter().product();
            f.write_str("[")?;
            for i in 0..len {
                if i > 0 {
                    f.write_str(", ")?;
                }
                fmt_nested(f, &data[i * chunk..(i + 1) * chunk], rest)?;
            }
            f.write_str("]")
        }
    }
}

impl fmt::Display for Tensor {
    /// `tensor([[1.0, 2.0]])`, with the dtype spelled out (as PyTorch does)
    /// unless it's what the values would get by default.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tensor(")?;
        with_dtype!(self.dtype(), T => fmt_nested(f, &self.to_vec::<T>(), &self.shape))?;
        if !matches!(self.dtype(), DType::F32 | DType::I64 | DType::Bool) {
            write!(f, ", dtype={}", self.dtype())?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn contiguous_strides_are_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
        assert_eq!(contiguous_strides(&[]), Vec::<isize>::new());
    }

    #[test]
    fn from_vec_checks_length() {
        assert_eq!(
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[2, 2]).unwrap_err(),
            TensorError::DataShapeMismatch {
                len: 3,
                shape: vec![2, 2]
            }
        );
    }

    #[test]
    fn dtype_comes_from_the_element_type() {
        assert_eq!(
            Tensor::from_vec(vec![1i32], &[1]).unwrap().dtype(),
            DType::I32
        );
        assert_eq!(Tensor::scalar(true).dtype(), DType::Bool);
        assert_eq!(Tensor::zeros(&[2], DType::U8).dtype(), DType::U8);
    }

    #[test]
    fn get_and_set() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(x.get(&[1, 2]).unwrap(), Scalar::Float(6.0));
        x.set(&[0, 1], 9).unwrap();
        assert_eq!(x.to_vec::<f32>(), vec![1.0, 9.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matches!(
            x.get(&[2, 0]),
            Err(TensorError::IndexOutOfBounds { axis: 0, .. })
        ));
        assert!(matches!(
            x.get(&[0]),
            Err(TensorError::IndexRankMismatch { .. })
        ));

        let ints = Tensor::from_vec(vec![1i64, 2], &[2]).unwrap();
        ints.set(&[0], 7.9).unwrap();
        assert_eq!(
            ints.get(&[0]).unwrap(),
            Scalar::Int(7),
            "converted to the dtype"
        );
    }

    #[test]
    fn writes_are_shared_between_views() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let flat = Tensor::from_parts(x.storage.clone(), vec![4], vec![1], 0);
        flat.set(&[3], 40.0).unwrap();
        assert_eq!(x.get(&[1, 1]).unwrap().to_f64(), 40.0);
        assert!(x.shares_storage(&flat));
        assert!(!x.shares_storage(&x.deep_copy()));
    }

    #[test]
    fn strided_views_read_in_logical_order() {
        // [[1, 2, 3], [4, 5, 6]] viewed transposed as [[1, 4], [2, 5], [3, 6]].
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let xt = Tensor::from_parts(x.storage.clone(), vec![3, 2], vec![1, 3], 0);
        assert_eq!(xt.to_vec::<f32>(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(!xt.is_contiguous());
        assert!(xt.deep_copy().is_contiguous());

        // Reversed view via a negative stride.
        let rev = Tensor::from_parts(x.storage.clone(), vec![6], vec![-1], 5);
        assert_eq!(rev.to_vec::<f32>(), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn creation_helpers() {
        assert_eq!(
            Tensor::zeros(&[2, 2], DType::F32).to_vec::<f32>(),
            vec![0.0; 4]
        );
        assert_eq!(Tensor::ones(&[3], DType::I64).to_vec::<i64>(), vec![1; 3]);
        assert_eq!(Tensor::scalar(7.0f32).item().unwrap(), Scalar::Float(7.0));
        assert!(
            Tensor::zeros(&[2, 0], DType::F32)
                .to_vec::<f32>()
                .is_empty()
        );
    }

    #[test]
    fn arange_float_and_int() {
        let f = |a: f64, b: f64, s: f64| {
            Tensor::arange(a.into(), b.into(), s.into(), DType::F32)
                .unwrap()
                .to_vec::<f32>()
        };
        assert_eq!(f(0.0, 1.0, 0.25), vec![0.0, 0.25, 0.5, 0.75]);
        assert_eq!(f(5.0, 0.0, -2.0), vec![5.0, 3.0, 1.0]);
        assert!(f(0.0, -1.0, 1.0).is_empty());
        assert!(Tensor::arange(0.0.into(), 1.0.into(), 0.0.into(), DType::F32).is_err());

        let i = |a: i64, b: i64, s: i64| {
            Tensor::arange(a.into(), b.into(), s.into(), DType::I64)
                .unwrap()
                .to_vec::<i64>()
        };
        assert_eq!(i(0, 5, 2), vec![0, 2, 4]);
        assert_eq!(i(5, 0, -2), vec![5, 3, 1]);
        assert!(i(0, 5, -1).is_empty());
        assert!(i(3, 3, 1).is_empty());
        // Exact near the edge of the range, where f64 would lose precision.
        assert_eq!(
            i(i64::MAX - 2, i64::MAX, 1),
            vec![i64::MAX - 2, i64::MAX - 1]
        );
    }

    #[test]
    fn to_dtype_converts_or_shares() {
        let x = t(&[1.5, -2.5, 0.0], &[3]);
        assert!(x.to_dtype(DType::F32).shares_storage(&x));
        let ints = x.to_dtype(DType::I32);
        assert_eq!(ints.to_vec::<i32>(), vec![1, -2, 0]);
        assert_eq!(
            x.to_dtype(DType::Bool).to_vec::<bool>(),
            vec![true, true, false]
        );
    }

    #[test]
    fn display_shows_values_and_non_default_dtypes() {
        assert_eq!(
            t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).to_string(),
            "tensor([[1.0, 2.0], [3.0, 4.0]])"
        );
        assert_eq!(Tensor::scalar(2.5f32).to_string(), "tensor(2.5)");
        assert_eq!(
            Tensor::from_vec(vec![1i64, 2], &[2]).unwrap().to_string(),
            "tensor([1, 2])"
        );
        assert_eq!(
            Tensor::from_vec(vec![true, false], &[2])
                .unwrap()
                .to_string(),
            "tensor([True, False])"
        );
        assert_eq!(
            Tensor::from_vec(vec![1i32], &[1]).unwrap().to_string(),
            "tensor([1], dtype=int32)"
        );
    }
}
